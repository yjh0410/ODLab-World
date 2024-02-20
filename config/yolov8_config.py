# yolov8-v2 Config


def build_yolov8_config(args):
    if   args.model == 'yolov8_n':
        return YOLOv8nConfig()
    elif args.model == 'yolov8_s':
        return YOLOv8sConfig()
    elif args.model == 'yolov8_m':
        return YOLOv8mConfig()
    elif args.model == 'yolov8_l':
        return YOLOv8lConfig()
    elif args.model == 'yolov8_x':
        return YOLOv8xConfig()
    else:
        raise NotImplementedError("No config for model: {}".format(args.model))
    
# YOLOv8-Base config
class YOLOv8BaseConfig(object):
    def __init__(self) -> None:
        # ---------------- Model config ----------------
        self.width    = 1.0
        self.depth    = 1.0
        self.ratio    = 1.0
        self.reg_max  = 16
        self.out_stride = [8, 16, 32]
        self.max_stride = 32
        self.num_levels = 3
        ## Backbone
        self.backbone = 'yolov8_backbone'
        self.bk_act   = 'silu'
        self.bk_norm  = 'BN'
        self.bk_depthwise = False
        ## Neck
        self.neck           = 'sppf'
        self.neck_act       = 'silu'
        self.neck_norm      = 'BN'
        self.neck_depthwise = False
        self.neck_expand_ratio = 0.5
        self.spp_pooling_size  = 5
        ## FPN
        self.fpn      = 'yolov8_pafpn'
        self.fpn_act  = 'silu'
        self.fpn_norm = 'BN'
        self.fpn_depthwise = False
        ## Head
        self.head      = 'yolov8_head'
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
        self.trainer = 'yolo'
        self.optimizer = 'adamw'
        self.per_image_lr = 0.001 / 64
        self.base_lr   = None      # base_lr = per_image_lr * batch_size
        self.min_lr_ratio = 0.01   # min_lr  = base_lr * min_lr_ratio
        self.momentum  = None
        self.weight_decay = 0.05
        self.clip_max_norm = 35.
        self.warmup_bias_lr = 0.1
        self.warmup_momentum = 0.8

        # ---------------- Lr Scheduler config ----------------
        self.warmup_epoch = 3
        self.lr_scheduler = "linear"
        self.max_epoch = 500
        self.eval_epoch = 10
        self.no_aug_epoch = 10

        # ---------------- Data process config ----------------
        self.aug_type = 'yolo'
        self.box_format = 'xyxy'
        self.normalize_coords = False
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.15
        self.multi_scale = [0.5, 1.5]
        ## Pixel mean & std
        self.pixel_mean = [0., 0., 0.]
        self.pixel_std  = [255., 255., 255.]
        ## Transforms
        self.train_img_size = 640
        self.test_img_size  = 640
        self.use_ablu = True
        self.affine_params = {
            'degrees': 0.0,
            'translate': 0.2,
            'scale': [0.1, 2.0],
            'shear': 0.0,
            'perspective': 0.0,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
        }

# YOLOv8-N
class YOLOv8nConfig(YOLOv8BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 0.25
        self.depth = 0.34
        self.ratio = 2.0

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.0

# YOLOv8-S
class YOLOv8sConfig(YOLOv8BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 0.50
        self.depth = 0.34
        self.ratio = 2.0

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.0

# YOLOv8-M
class YOLOv8mConfig(YOLOv8BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 0.75
        self.depth = 0.67
        self.ratio = 1.5

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.1

# YOLOv8-L
class YOLOv8lConfig(YOLOv8BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 1.0
        self.depth = 1.0
        self.ratio = 1.0

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.15

# YOLOv8-X
class YOLOv8xConfig(YOLOv8BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 1.25
        self.depth = 1.0
        self.ratio = 1.0

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.2
