# yolo Config


def build_yolo_config(args):
    if   args.model == 'yolo_p':
        return YoloPConfig()
    elif args.model == 'yolo_n':
        return YoloNConfig()
    elif args.model == 'yolo_s':
        return YoloSConfig()
    elif args.model == 'yolo_m':
        return YoloMConfig()
    elif args.model == 'yolo_l':
        return YoloLConfig()
    elif args.model == 'yolo_x':
        return YoloXConfig()
    else:
        raise NotImplementedError("No config for model: {}".format(args.model))
    
# YOLO-Base config
class YoloBaseConfig(object):
    def __init__(self) -> None:
        # ---------------- Model config ----------------
        self.width    = 1.0
        self.depth    = 1.0
        self.ratio    = 1.0
        self.reg_max  = 16
        self.out_stride = [8, 16, 32]
        self.max_stride = 32
        self.num_levels = 3
        self.scale      = "b"
        ## Backbone
        self.backbone = 'yolo_backbone'
        self.bk_act   = 'silu'
        self.bk_norm  = 'BN'
        self.bk_depthwise = False
        self.use_pretrained = False
        ## Neck
        self.neck           = 'sppf'
        self.neck_act       = 'silu'
        self.neck_norm      = 'BN'
        self.neck_depthwise = False
        self.neck_expand_ratio = 0.5
        self.spp_pooling_size  = 5
        ## FPN
        self.fpn      = 'yolo_pafpn'
        self.fpn_act  = 'silu'
        self.fpn_norm = 'BN'
        self.fpn_depthwise = False
        ## Head
        self.head      = 'yolo_head'
        self.head_act  = 'silu'
        self.head_norm = 'BN'
        self.head_depthwise = False
        self.num_cls_head   = 2
        self.num_reg_head   = 2

        # ---------------- Post-process config ----------------
        ## Post process
        self.postprocess_type = "level_wise"  # optional: [level_wise, gather] 
        self.val_topk = 1000                  # 1000 for 'level_wise'; 30000 for 'gather'
        self.val_conf_thresh = 0.001
        self.val_nms_thresh  = 0.7
        self.test_topk = 100
        self.test_conf_thresh = 0.2
        self.test_nms_thresh  = 0.5
        self.max_det = 300                    # only work for 'gather'

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
        self.copy_paste  = 0.5
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

# YOLO-P config
class YoloPConfig(YoloBaseConfig):
    def __init__(self) -> None:
        # ---------------- Model config ----------------
        self.width = 0.25
        self.depth = 0.34
        self.ratio = 2.0
        self.scale = "p"
        ## Backbone
        self.bk_depthwise   = True
        self.use_pretrained = True
        ## Neck
        self.neck_depthwise = True
        ## FPN
        self.fpn_depthwise  = True
        ## Head
        self.head_depthwise = True

        # ---------------- Data process config ----------------
        self.mosaic_prob = 0.5
        self.mixup_prob  = 0.0
        self.copy_paste  = 0.0
        self.affine_params = {
            'degrees': 0.0,
            'translate': 0.1,
            'scale': [0.5, 1.5],
            'shear': 0.0,
            'perspective': 0.0,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
        }

# YOLO-N
class YoloNConfig(YoloBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 0.25
        self.depth = 0.34
        self.ratio = 2.0
        self.scale = "n"

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.0
        self.copy_paste  = 0.0
        self.affine_params = {
            'degrees': 0.0,
            'translate': 0.1,
            'scale': [0.5, 1.5],
            'shear': 0.0,
            'perspective': 0.0,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
        }

# YOLO-S
class YoloSConfig(YoloBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 0.50
        self.depth = 0.34
        self.ratio = 2.0
        self.scale = "s"

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.0
        self.copy_paste  = 0.0

# YOLO-M
class YoloMConfig(YoloBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 0.75
        self.depth = 0.67
        self.ratio = 1.5
        self.scale = "m"

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.1

# YOLO-L
class YoloLConfig(YoloBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 1.0
        self.depth = 1.0
        self.ratio = 1.0
        self.scale = "l"

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.15

# YOLO-X
class YoloXConfig(YoloBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 1.25
        self.depth = 1.0
        self.ratio = 1.0
        self.scale = "x"

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.2
