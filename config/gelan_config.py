# yolo Config


def build_gelan_config(args):
    if   args.model == 'gelan_c':
        return GElanCConfig()
    elif args.model == 'gelan_m':
        return GElanMConfig()
    else:
        raise NotImplementedError("No config for model: {}".format(args.model))
    
# GELAN-Base config
class GElanBaseConfig(object):
    def __init__(self) -> None:
        # ---------------- Model config ----------------
        self.reg_max  = 16
        self.out_stride = [8, 16, 32]
        self.max_stride = 32
        self.num_levels = 3
        ## Backbone
        self.bk_act = 'silu'
        self.bk_norm = 'BN'
        self.bk_depthwise = False
        self.bk_down_pooling = True
        self.backbone_feats = {
            "c1": [64],
            "c2": [128, [128, 64],  256],
            "c3": [256, [256, 128], 512],
            "c4": [512, [512, 256], 512],
            "c5": [512, [512, 256], 512],
        }
        self.backbone_depth = 1
        ## Neck
        self.neck           = 'spp_elan'
        self.neck_act       = 'silu'
        self.neck_norm      = 'BN'
        self.spp_pooling_size  = 5
        self.spp_inter_dim     = 256
        self.spp_out_dim       = 512
        ## FPN
        self.fpn      = 'gelan_pafpn'
        self.fpn_act  = 'silu'
        self.fpn_norm = 'BN'
        self.fpn_depthwise = False
        self.fpn_down_pooling = True
        self.fpn_depth    = 1
        self.fpn_feats_td = {
            "p4": [[512, 256], 512],
            "p3": [[256, 128], 256],
        }
        self.fpn_feats_bu = {
            "p4": [[512, 256], 512],
            "p5": [[512, 256], 512],
        }
        ## Head
        self.head      = 'gelan_head'
        self.head_act  = 'silu'
        self.head_norm = 'BN'
        self.head_depthwise = False
        self.num_cls_head   = 2
        self.num_reg_head   = 2

        # ---------------- Post-process config ----------------
        ## Post process
        self.val_topk = 1000
        self.val_conf_thresh = 0.001
        self.val_nms_thresh  = 0.7
        self.test_topk = 100
        self.test_conf_thresh = 0.2
        self.test_nms_thresh  = 0.5

        # ---------------- Assignment config ----------------
        ## Matcher
        self.tal_topk_candidates = 10
        self.tal_alpha = 0.5
        self.tal_beta  = 6.0
        ## Loss weight
        self.loss_cls = 0.5
        self.loss_box = 7.5
        self.loss_dfl = 1.5

        # ---------------- ModelEMA config ----------------
        self.use_ema = True
        self.ema_decay = 0.9998
        self.ema_tau   = 2000

        # ---------------- Optimizer config ----------------
        self.trainer      = 'yolo'
        self.optimizer    = 'adamw'
        self.per_image_lr = 0.001 / 64
        self.base_lr      = None      # base_lr = per_image_lr * batch_size
        self.min_lr_ratio = 0.01      # min_lr  = base_lr * min_lr_ratio
        self.momentum     = 0.9
        self.weight_decay = 0.05
        self.clip_max_norm   = -1.
        self.warmup_bias_lr  = 0.1
        self.warmup_momentum = 0.8

        # ---------------- Lr Scheduler config ----------------
        self.warmup_epoch = 3
        self.lr_scheduler = "linear"
        self.max_epoch    = 500
        self.eval_epoch   = 10
        self.no_aug_epoch = 20

        # ---------------- Data process config ----------------
        self.aug_type = 'yolo'
        self.box_format = 'xyxy'
        self.normalize_coords = False
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.15
        self.copy_paste  = 1.0
        self.multi_scale = [0.5, 1.5]   # multi scale: [img_size * 0.5, img_size * 1.5]
        ## Pixel mean & std
        self.pixel_mean = [0., 0., 0.]
        self.pixel_std  = [255., 255., 255.]
        ## Transforms
        self.train_img_size = 640
        self.test_img_size  = 640
        self.use_ablu = True
        self.affine_params = {
            'degrees': 0.0,
            'translate': 0.1,
            'scale': [0.1, 2.0],
            'shear': 0.0,
            'perspective': 0.0,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
        }

    def print_config(self):
        config_dict = {key: value for key, value in self.__dict__.items() if not key.startswith('__')}
        for k, v in config_dict.items():
            print("{} : {}".format(k, v))

# GELAN-M (not complete yet)
class GElanMConfig(GElanBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        ## Backbone
        self.bk_down_pooling = True
        self.backbone_feats = {
            "c1": [48],
            "c2": [48,  [96,  64],  96],
            "c3": [96,  [192, 128], 192],
            "c4": [192, [384, 256], 384],
            "c5": [384, [576, 256], 576],
        }
        self.backbone_depth = 1
        ## Neck
        self.spp_inter_dim  = 256
        self.spp_out_dim    = 576
        ## FPN
        self.fpn_down_pooling = True
        self.fpn_depth    = 1
        self.fpn_feats_td = {
            "p4": [[384, 256], 384],
            "p3": [[192, 128], 192],
        }
        self.fpn_feats_bu = {
            "p4": [[384, 256], 384],
            "p5": [[576, 256], 576],
        }
        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.15
        self.copy_paste  = 1.0

# GELAN-C
class GElanCConfig(GElanBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.15
        self.copy_paste  = 1.0

