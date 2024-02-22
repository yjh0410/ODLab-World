import torch
import torch.distributed as dist

import time
import os
import numpy as np
import random

# ----------------- Extra Components -----------------
from utils import distributed_utils
from utils.misc import ModelEMA
from utils.misc import MetricLogger, SmoothedValue
from utils.vis_tools import vis_data

# ----------------- Optimizer & LrScheduler Components -----------------
from utils.solver.optimizer import build_yolo_optimizer, build_rtdetr_optimizer
from utils.solver.lr_scheduler import build_lambda_lr_scheduler
from utils.solver.lr_scheduler import build_wp_lr_scheduler, build_lr_scheduler

# ----------------- Dataset Components -----------------
from dataset.build import build_transform


class YoloTrainer(object):
    def __init__(self,
                 # Basic parameters
                 args,
                 cfg,
                 device,
                 # Model parameters
                 model,
                 criterion,
                 # Data parameters
                 train_transform,
                 val_transform,
                 dataset,
                 train_loader,
                 evaluator,
                 ):
        # ------------------- basic parameters -------------------
        self.args = args
        self.cfg  = cfg
        self.epoch = 0
        self.best_map = -1.
        self.device = device
        self.criterion = criterion
        self.heavy_eval = False
        # weak augmentatino stage
        self.second_stage = False
        self.third_stage  = False
        self.second_stage_epoch = cfg.no_aug_epoch
        self.third_stage_epoch  = cfg.no_aug_epoch // 2
        # path to save model
        self.path_to_save = os.path.join(args.save_folder, args.dataset, args.model)
        os.makedirs(self.path_to_save, exist_ok=True)

        # ---------------------------- Transform ----------------------------
        self.train_transform = train_transform
        self.val_transform   = val_transform

        # ---------------------------- Dataset & Dataloader ----------------------------
        self.dataset      = dataset
        self.train_loader = train_loader

        # ---------------------------- Evaluator ----------------------------
        self.evaluator = evaluator

        # ---------------------------- Build Grad. Scaler ----------------------------
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

        # ---------------------------- Build Optimizer ----------------------------
        cfg.base_lr = cfg.per_image_lr * args.batch_size
        self.optimizer, self.start_epoch = build_yolo_optimizer(cfg, model, args.resume)

        # ---------------------------- Build LR Scheduler ----------------------------
        self.lr_scheduler, self.lf = build_lambda_lr_scheduler(cfg, self.optimizer)
        self.lr_scheduler.last_epoch = self.start_epoch - 1  # do not move
        if self.args.resume and self.args.resume != 'None':
            self.lr_scheduler.step()

        # ---------------------------- Build Model-EMA ----------------------------
        if cfg.use_ema and distributed_utils.get_rank() in [-1, 0]:
            print('Build ModelEMA for {} ...'.format(args.model))
            update_init = self.start_epoch * len(self.train_loader)
            self.model_ema = ModelEMA(model, cfg.ema_decay, cfg.ema_tau, update_init)
        else:
            self.model_ema = None

    def train(self, model):
        for epoch in range(self.start_epoch, self.cfg.max_epoch):
            if self.args.distributed:
                self.train_loader.batch_sampler.sampler.set_epoch(epoch)

            # check second stage
            if epoch >= (self.cfg.max_epoch - self.second_stage_epoch - 1) and not self.second_stage:
                self.check_second_stage()
                # save model of the last mosaic epoch
                weight_name = '{}_last_mosaic_epoch.pth'.format(self.args.model)
                checkpoint_path = os.path.join(self.path_to_save, weight_name)
                print('Saving state of the last Mosaic epoch-{}.'.format(self.epoch))
                torch.save({'model': model.state_dict(),
                            'mAP': round(self.evaluator.map*100, 1),
                            'optimizer': self.optimizer.state_dict(),
                            'epoch': self.epoch,
                            'args': self.args}, 
                            checkpoint_path)

            # check third stage
            if epoch >= (self.cfg.max_epoch - self.third_stage_epoch - 1) and not self.third_stage:
                self.check_third_stage()
                # save model of the last mosaic epoch
                weight_name = '{}_last_weak_augment_epoch.pth'.format(self.args.model)
                checkpoint_path = os.path.join(self.path_to_save, weight_name)
                print('Saving state of the last weak augment epoch-{}.'.format(self.epoch))
                torch.save({'model': model.state_dict(),
                            'mAP': round(self.evaluator.map*100, 1),
                            'optimizer': self.optimizer.state_dict(),
                            'epoch': self.epoch,
                            'args': self.args}, 
                            checkpoint_path)

            # train one epoch
            self.epoch = epoch
            self.train_one_epoch(model)

            # eval one epoch
            if self.heavy_eval:
                model_eval = model.module if self.args.distributed else model
                self.eval(model_eval)
            else:
                model_eval = model.module if self.args.distributed else model
                if (epoch % self.cfg.eval_epoch) == 0 or (epoch == self.cfg.max_epoch - 1):
                    self.eval(model_eval)

            if self.args.debug:
                print("For debug mode, we only train 1 epoch")
                break

    def eval(self, model):
        # chech model
        model_eval = model if self.model_ema is None else self.model_ema.ema

        if distributed_utils.is_main_process():
            # check evaluator
            if self.evaluator is None:
                print('No evaluator ... save model and go on training.')
                print('Saving state, epoch: {}'.format(self.epoch))
                weight_name = '{}_no_eval.pth'.format(self.args.model)
                checkpoint_path = os.path.join(self.path_to_save, weight_name)
                torch.save({'model': model_eval.state_dict(),
                            'mAP': -1.,
                            'optimizer': self.optimizer.state_dict(),
                            'epoch': self.epoch,
                            'args': self.args}, 
                            checkpoint_path)               
            else:
                print('eval ...')
                # set eval mode
                model_eval.trainable = False
                model_eval.eval()

                # evaluate
                with torch.no_grad():
                    self.evaluator.evaluate(model_eval)

                # save model
                cur_map = self.evaluator.map
                if cur_map > self.best_map:
                    # update best-map
                    self.best_map = cur_map
                    # save model
                    print('Saving state, epoch:', self.epoch)
                    weight_name = '{}_best.pth'.format(self.args.model)
                    checkpoint_path = os.path.join(self.path_to_save, weight_name)
                    torch.save({'model': model_eval.state_dict(),
                                'mAP': round(self.best_map*100, 1),
                                'optimizer': self.optimizer.state_dict(),
                                'epoch': self.epoch,
                                'args': self.args}, 
                                checkpoint_path)                      

                # set train mode.
                model_eval.trainable = True
                model_eval.train()

        if self.args.distributed:
            # wait for all processes to synchronize
            dist.barrier()

    def train_one_epoch(self, model):
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('size', SmoothedValue(window_size=1, fmt='{value:d}'))
        metric_logger.add_meter('grad_norm', SmoothedValue(window_size=1, fmt='{value:.1f}'))
        header = 'Epoch: [{} / {}]'.format(self.epoch, self.cfg.max_epoch)
        epoch_size = len(self.train_loader)
        print_freq = 10

        # basic parameters
        epoch_size = len(self.train_loader)
        img_size   = self.cfg.train_img_size
        nw = epoch_size * self.cfg.warmup_epoch

        # Train one epoch
        for iter_i, (images, targets) in enumerate(metric_logger.log_every(self.train_loader, print_freq, header)):
            ni = iter_i + self.epoch * epoch_size
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                for j, x in enumerate(self.optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(
                        ni, xi, [self.cfg.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(self.epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [self.cfg.warmup_momentum, self.cfg.momentum])
                                
            # To device
            images = images.to(self.device, non_blocking=True).float()

            # Multi scale
            images, targets, img_size = self.rescale_image_targets(
                images, targets, self.cfg.max_stride, self.cfg.multi_scale)
                
            # Visualize train targets
            if self.args.vis_tgt:
                vis_data(images,
                         targets,
                         self.cfg.num_classes,
                         self.cfg.normalize_coords,
                         self.train_transform.color_format,
                         self.cfg.pixel_mean,
                         self.cfg.pixel_std,
                         self.cfg.box_format)

            # Inference
            with torch.cuda.amp.autocast(enabled=self.args.fp16):
                outputs = model(images)
                # Compute loss
                loss_dict = self.criterion(outputs=outputs, targets=targets)
                losses = loss_dict['losses']
                loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

            # Backward
            self.scaler.scale(losses).backward()

            # Optimize
            grad_norm = None
            if self.cfg.clip_max_norm > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.cfg.clip_max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # ModelEMA
            if self.model_ema is not None:
                self.model_ema.update(model)

            # Update log
            metric_logger.update(**loss_dict_reduced)
            metric_logger.update(lr=self.optimizer.param_groups[2]["lr"])
            metric_logger.update(grad_norm=grad_norm)
            metric_logger.update(size=img_size)

            if self.args.debug:
                print("For debug mode, we only train 1 iteration")
                break

        # LR Schedule
        if not self.second_stage:
            self.lr_scheduler.step()

        # Gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)

    def rescale_image_targets(self, images, targets, max_stride, multi_scale_range=[0.5, 1.5]):
        """
            Deployed for Multi scale trick.
        """
        # During training phase, the shape of input image is square.
        old_img_size = images.shape[-1]
        min_img_size = old_img_size * multi_scale_range[0]
        max_img_size = old_img_size * multi_scale_range[1]

        # Choose a new image size
        new_img_size = random.randrange(min_img_size, max_img_size + max_stride, max_stride)
        
        # Resize
        if new_img_size != old_img_size:
            # interpolate
            images = torch.nn.functional.interpolate(
                                input=images, 
                                size=new_img_size, 
                                mode='bilinear', 
                                align_corners=False)
        # rescale targets
        if not self.cfg.normalize_coords:
            for tgt in targets:
                boxes = tgt["boxes"].clone()
                labels = tgt["labels"].clone()
                boxes = torch.clamp(boxes, 0, old_img_size)
                # rescale box
                boxes[:, [0, 2]] = boxes[:, [0, 2]] / old_img_size * new_img_size
                boxes[:, [1, 3]] = boxes[:, [1, 3]] / old_img_size * new_img_size
                # refine tgt
                tgt_boxes_wh = boxes[..., 2:] - boxes[..., :2]
                min_tgt_size = torch.min(tgt_boxes_wh, dim=-1)[0]
                keep = (min_tgt_size >= 1)

                tgt["boxes"] = boxes[keep]
                tgt["labels"] = labels[keep]

        return images, targets, new_img_size

    def check_second_stage(self):
        # set second stage
        print('============== Second stage of Training ==============')
        self.second_stage = True

        # close mosaic augmentation
        if self.train_loader.dataset.mosaic_prob > 0.:
            print(' - Close < Mosaic Augmentation > ...')
            self.train_loader.dataset.mosaic_prob = 0.
            self.heavy_eval = True

        # close mixup augmentation
        if self.train_loader.dataset.mixup_prob > 0.:
            print(' - Close < Mixup Augmentation > ...')
            self.train_loader.dataset.mixup_prob = 0.
            self.heavy_eval = True

        # close rotation augmentation
        if 'degrees' in self.cfg.affine_params.keys() and self.cfg.affine_params['degrees'] > 0.0:
            print(' - Close < degress of rotation > ...')
            self.cfg.affine_params['degrees'] = 0.0
        if 'shear' in self.cfg.affine_params.keys() and self.cfg.affine_params['shear'] > 0.0:
            print(' - Close < shear of rotation >...')
            self.cfg.affine_params['shear'] = 0.0
        if 'perspective' in self.cfg.affine_params.keys() and self.cfg.affine_params['perspective'] > 0.0:
            print(' - Close < perspective of rotation > ...')
            self.cfg.affine_params['perspective'] = 0.0

        # build a new transform for second stage
        print(' - Rebuild transforms ...')
        self.train_transform = build_transform(self.cfg, is_train=True)
        self.train_loader.dataset.transform = self.train_transform
        
    def check_third_stage(self):
        # set third stage
        print('============== Third stage of Training ==============')
        self.third_stage = True

        # close random affine
        if 'translate' in self.cfg.affine_params.keys() and self.cfg.affine_params['translate'] > 0.0:
            print(' - Close < translate of affine > ...')
            self.cfg.affine_params['translate'] = 0.0
        if 'scale' in self.cfg.affine_params.keys():
            print(' - Close < scale of affine >...')
            self.cfg.affine_params['scale'] = [1.0, 1.0]

        # build a new transform for second stage
        print(' - Rebuild transforms ...')
        self.train_transform = build_transform(self.cfg, is_train=True)
        self.train_loader.dataset.transform = self.train_transform

class RTDetrTrainer(object):
    def __init__(self,
                 # Basic parameters
                 args,
                 cfg,
                 device,
                 # Model parameters
                 model,
                 criterion,
                 # Data parameters
                 train_transform,
                 val_transform,
                 dataset,
                 train_loader,
                 evaluator,
                 ):
        # ------------------- basic parameters -------------------
        self.args = args
        self.cfg  = cfg
        self.epoch = 0
        self.best_map = -1.
        self.device = device
        self.criterion = criterion
        self.heavy_eval = False
        # path to save model
        self.path_to_save = os.path.join(args.save_folder, args.dataset, args.model)
        os.makedirs(self.path_to_save, exist_ok=True)

        # ---------------------------- Transform ----------------------------
        self.train_transform = train_transform
        self.val_transform   = val_transform

        # ---------------------------- Dataset & Dataloader ----------------------------
        self.dataset      = dataset
        self.train_loader = train_loader

        # ---------------------------- Evaluator ----------------------------
        self.evaluator = evaluator

        # ---------------------------- Build Grad. Scaler ----------------------------
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

        # ---------------------------- Build Optimizer ----------------------------
        cfg.base_lr = cfg.per_image_lr * args.batch_size
        self.optimizer, self.start_epoch = build_rtdetr_optimizer(cfg, model, args.resume)

        # ---------------------------- Build LR Scheduler ----------------------------
        self.wp_lr_scheduler = build_wp_lr_scheduler(cfg)
        self.lr_scheduler    = build_lr_scheduler(cfg, self.optimizer, args.resume)

        # ---------------------------- Build Model-EMA ----------------------------
        if cfg.use_ema and distributed_utils.get_rank() in [-1, 0]:
            print('Build ModelEMA for {} ...'.format(args.model))
            update_init = self.start_epoch * len(self.train_loader)
            self.model_ema = ModelEMA(model, cfg.ema_decay, cfg.ema_tau, update_init)
        else:
            self.model_ema = None

    def train(self, model):
        for epoch in range(self.start_epoch, self.cfg.max_epoch):
            if self.args.distributed:
                self.train_loader.batch_sampler.sampler.set_epoch(epoch)

            # train one epoch
            self.epoch = epoch
            self.train_one_epoch(model)

            # eval one epoch
            if self.heavy_eval:
                model_eval = model.module if self.args.distributed else model
                self.eval(model_eval)
            else:
                model_eval = model.module if self.args.distributed else model
                if (epoch % self.cfg.eval_epoch) == 0 or (epoch == self.cfg.max_epoch - 1):
                    self.eval(model_eval)

            if self.args.debug:
                print("For debug mode, we only train 1 epoch")
                break

    def eval(self, model):
        # chech model
        model_eval = model if self.model_ema is None else self.model_ema.ema

        if distributed_utils.is_main_process():
            # check evaluator
            if self.evaluator is None:
                print('No evaluator ... save model and go on training.')
                print('Saving state, epoch: {}'.format(self.epoch))
                weight_name = '{}_no_eval.pth'.format(self.args.model)
                checkpoint_path = os.path.join(self.path_to_save, weight_name)
                torch.save({'model': model_eval.state_dict(),
                            'mAP': -1.,
                            'optimizer': self.optimizer.state_dict(),
                            'lr_scheduler': self.lr_scheduler.state_dict(),
                            'epoch': self.epoch,
                            'args': self.args}, 
                            checkpoint_path)               
            else:
                print('eval ...')
                # set eval mode
                model_eval.eval()

                # evaluate
                with torch.no_grad():
                    self.evaluator.evaluate(model_eval)

                # save model
                cur_map = self.evaluator.map
                if cur_map > self.best_map:
                    # update best-map
                    self.best_map = cur_map
                    # save model
                    print('Saving state, epoch:', self.epoch)
                    weight_name = '{}_best.pth'.format(self.args.model)
                    checkpoint_path = os.path.join(self.path_to_save, weight_name)
                    torch.save({'model': model_eval.state_dict(),
                                'mAP': round(self.best_map*100, 1),
                                'optimizer': self.optimizer.state_dict(),
                                'epoch': self.epoch,
                                'args': self.args}, 
                                checkpoint_path)                      

                # set train mode.
                model_eval.train()

        if self.args.distributed:
            # wait for all processes to synchronize
            dist.barrier()

    def train_one_epoch(self, model):
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('size', SmoothedValue(window_size=1, fmt='{value:d}'))
        metric_logger.add_meter('grad_norm', SmoothedValue(window_size=1, fmt='{value:.1f}'))
        header = 'Epoch: [{} / {}]'.format(self.epoch, self.cfg.max_epoch)
        epoch_size = len(self.train_loader)
        print_freq = 10

        # basic parameters
        epoch_size = len(self.train_loader)
        img_size   = self.cfg.train_img_size
        nw         = self.cfg.warmup_iters
        lr_warmup_stage = True

        # Train one epoch
        for iter_i, (images, targets) in enumerate(metric_logger.log_every(self.train_loader, print_freq, header)):
            ni = iter_i + self.epoch * epoch_size
            # WarmUp
            if ni < nw and lr_warmup_stage:
                self.wp_lr_scheduler(ni, self.optimizer)
            elif ni == nw and lr_warmup_stage:
                print('Warmup stage is over.')
                lr_warmup_stage = False
                self.wp_lr_scheduler.set_lr(self.optimizer, self.cfg.base_lr, self.cfg.base_lr)
                                
            # To device
            images = images.to(self.device, non_blocking=True).float()
            for tgt in targets:
                tgt['boxes'] = tgt['boxes'].to(self.device)
                tgt['labels'] = tgt['labels'].to(self.device)

            # Multi scale
            images, targets, img_size = self.rescale_image_targets(
                images, targets, self.cfg.max_stride, self.cfg.multi_scale)
                
            # Visualize train targets
            if self.args.vis_tgt:
                vis_data(images,
                         targets,
                         self.cfg.num_classes,
                         self.cfg.normalize_coords,
                         self.train_transform.color_format,
                         self.cfg.pixel_mean,
                         self.cfg.pixel_std,
                         self.cfg.box_format)

            # Inference
            with torch.cuda.amp.autocast(enabled=self.args.fp16):
                outputs = model(images, targets)    
                loss_dict = self.criterion(outputs, targets)
                losses = sum(loss_dict.values())
                loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

            # Backward
            self.scaler.scale(losses).backward()

            # Optimize
            grad_norm = None
            if self.cfg.clip_max_norm > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.cfg.clip_max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # ModelEMA
            if self.model_ema is not None:
                self.model_ema.update(model)

            # Update log
            metric_logger.update(loss=losses.item(), **loss_dict_reduced)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.update(grad_norm=grad_norm)
            metric_logger.update(size=img_size)

            if self.args.debug:
                print("For debug mode, we only train 1 iteration")
                break

        # LR Scheduler
        self.lr_scheduler.step()

    def rescale_image_targets(self, images, targets, max_stride, multi_scale_range=[0.5, 1.5]):
        """
            Deployed for Multi scale trick.
        """
        # During training phase, the shape of input image is square.
        old_img_size = images.shape[-1]
        min_img_size = old_img_size * multi_scale_range[0]
        max_img_size = old_img_size * multi_scale_range[1]

        # Choose a new image size
        new_img_size = random.randrange(min_img_size, max_img_size + max_stride, max_stride)
        
        # Resize
        if new_img_size != old_img_size:
            # interpolate
            images = torch.nn.functional.interpolate(
                                input=images, 
                                size=new_img_size, 
                                mode='bilinear', 
                                align_corners=False)

        return images, targets, new_img_size


# Build Trainer
def build_trainer(args, cfg, device, model, criterion, train_transform, val_transform, dataset, train_loader, evaluator):
    # ----------------------- Det trainers -----------------------
    if   cfg.trainer == 'yolo':
        return YoloTrainer(args, cfg, device, model, criterion, train_transform, val_transform, dataset, train_loader, evaluator)
    elif cfg.trainer == 'rtdetr':
        return RTDetrTrainer(args, cfg, device, model, criterion, train_transform, val_transform, dataset, train_loader, evaluator)
    else:
        raise NotImplementedError(cfg.trainer)
    