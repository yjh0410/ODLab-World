# Real-time Transformer-based Object Detector


def build_rtdetr_config(args):
    if args.model == "rtdetr_r18":
        return RTDetrR18Config()
    elif args.model == "rtdetr_r34":
        return RTDetrR34Config()
    elif args.model == "rtdetr_r50":
        return RTDetrR50Config()
    elif args.model == "rtdetr_r50x":
        return RTDetrR50XConfig()
    elif args.model == "rtdetr_r101":
        return RTDetrR101Config()
    elif args.model == "rtdetr_s":
        return RTDetrSConfig()
    elif args.model == "rtdetr_m":
        return RTDetrMConfig()
    elif args.model == "rtdetr_l":
        return RTDetrLConfig()
    elif args.model == "rtdetr_x":
        return RTDetrXConfig()
    raise NotImplementedError("No config for model: {}".format(args.model))   
 
# rtdetr-Base config
class RTDetrBaseConfig(object):
    def __init__(self) -> None:
        # ---------------- Model config ----------------
        self.out_stride = [8, 16, 32]
        self.max_stride = 32
        ## Backbone
        self.backbone        = 'resnet18'
        self.backbone_norm   = 'BN'
        self.pretrained_weight  = 'imagenet1k_v1'
        self.pretrained = True
        self.freeze_at = 0
        self.freeze_stem_only = False
        ## Image Encoder - FPN
        self.fpn      = 'hybrid_encoder'
        self.fpn_num_blocks = 3
        self.fpn_expand_ratio = 0.5
        self.fpn_act  = 'silu'
        self.fpn_norm = 'BN'
        self.fpn_depthwise = False
        self.hidden_dim = 256
        self.en_num_heads = 8
        self.en_num_layers = 1
        self.en_ffn_dim = 1024
        self.en_dropout = 0.0
        self.en_act = 'gelu'
        ## Transformer Decoder
        self.transformer   = 'rtdetr_transformer'
        self.de_num_heads  = 8
        self.de_num_layers = 3
        self.de_ffn_dim    = 1024
        self.de_dropout    = 0.0
        self.de_act        = 'relu'
        self.de_num_points = 4
        self.num_queries   = 300
        self.learnt_init_query = False
        ## DN
        self.dn_num_denoising     = 100
        self.dn_label_noise_ratio = 0.5
        self.dn_box_noise_scale   = 1

        # ---------------- Post-process config ----------------
        ## Post process
        self.val_topk = 300
        self.val_conf_thresh = 0.001
        self.val_nms_thresh  = 0.7
        self.test_topk = 300
        self.test_conf_thresh = 0.3
        self.test_nms_thresh  = 0.5

        # ---------------- Assignment config ----------------
        ## Matcher
        self.cost_class = 2.0
        self.cost_bbox  = 5.0
        self.cost_giou  = 2.0
        ## Loss weight
        self.loss_cls  = 1.0
        self.loss_box  = 5.0
        self.loss_giou = 2.0

        # ---------------- ModelEMA config ----------------
        self.use_ema = True
        self.ema_decay = 0.9999
        self.ema_tau   = 2000

        # ---------------- Optimizer config ----------------
        self.trainer = 'rtdetr'
        self.optimizer = 'adamw'
        self.per_image_lr = 0.0001 / 16
        self.base_lr      = None      # base_lr = per_image_lr * batch_size
        self.min_lr_ratio      = 0.0
        self.backbone_lr_ratio = 0.1
        self.momentum  = None
        self.weight_decay = 0.0001
        self.clip_max_norm = 0.1

        # ---------------- Lr Scheduler config ----------------
        self.warmup = 'linear'
        self.warmup_iters = 2000
        self.warmup_factor = 0.00066667
        self.lr_scheduler = "step"
        self.lr_epoch = [100]
        self.max_epoch = 72
        self.eval_epoch = 1

        # ---------------- Data process config ----------------
        self.aug_type = 'rtdetr'
        self.box_format = 'xywh'
        self.normalize_coords = True
        self.mosaic_prob = 0.0
        self.mixup_prob  = 0.0
        self.copy_paste  = 0.0
        self.multi_scale = [0.75, 1.25]
        ## Pixel mean & std
        self.pixel_mean = [123.675, 116.28, 103.53]   # RGB format
        self.pixel_std  = [58.395, 57.12, 57.375]     # RGB format
        ## Transforms
        self.train_img_size = 640
        self.test_img_size  = 640

    def print_config(self):
        config_dict = {key: value for key, value in self.__dict__.items() if not key.startswith('__')}
        for k, v in config_dict.items():
            print("{} : {}".format(k, v))
    
# RT-DETR-R18
class RTDetrR18Config(RTDetrBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        ## Backbone
        self.backbone        = 'resnet18'
        self.backbone_norm   = 'BN'
        self.pretrained_weight  = 'imagenet1k_v1'
        self.pretrained = True
        self.freeze_at = -1
        self.freeze_stem_only = False

# RT-DETR-R34
class RTDetrR34Config(RTDetrBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        ## Backbone
        self.backbone        = 'resnet34'
        self.backbone_norm   = 'BN'
        self.pretrained_weight  = 'imagenet1k_v1'
        self.freeze_at = -1
        self.freeze_stem_only = False
        ## Transformer Decoder
        self.de_num_layers = 6

# RT-DETR-R50
class RTDetrR50Config(RTDetrBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        ## Backbone
        self.backbone        = 'resnet50'
        self.backbone_norm   = 'BN'
        self.pretrained_weight  = 'imagenet1k_v2'
        self.freeze_at = -1
        self.freeze_stem_only = False
        ## Transformer Decoder
        self.de_num_layers = 6

# RT-DETR-R50-X
class RTDetrR50XConfig(RTDetrBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        ## Backbone
        self.backbone        = 'resnet50'
        self.backbone_norm   = 'BN'
        self.pretrained_weight  = 'imagenet1k_v2'
        self.freeze_at = -1
        self.freeze_stem_only = False
        ## Image Encoder - FPN
        self.fpn_expand_ratio = 1.0
        ## Transformer Decoder
        self.de_num_layers = 6

# RT-DETR-R101
class RTDetrR101Config(RTDetrBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        ## Backbone
        self.backbone          = 'resnet101'
        self.backbone_norm     = 'BN'
        self.pretrained_weight = 'imagenet1k_v2'
        self.freeze_at = -1
        self.freeze_stem_only = False
        ## Image Encoder - FPN
        self.fpn_expand_ratio = 1.0
        ## Transformer Decoder
        self.de_num_layers = 6

# RT-DETR-S (not complete yet)
class RTDetrSConfig(RTDetrBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width    = 0.50
        self.depth    = 0.34
        self.ratio    = 2.0
        self.out_stride = [8, 16, 32]
        self.max_stride = 32
        self.scale      = "s"
        ## Backbone
        self.backbone = 'rtcnet'
        self.bk_act   = 'silu'
        self.bk_norm  = 'BN'
        self.bk_depthwise = False
        self.use_pretrained = True
        ## Image Encoder - FPN
        self.fpn_num_blocks = 1
        self.fpn_expand_ratio = 0.5
        self.hidden_dim = 256
        self.en_num_heads = 8
        self.en_num_layers = 1
        self.en_ffn_dim = 1024
        ## Transformer Decoder
        self.de_num_layers = 3
        self.de_ffn_dim    = 1024

# RT-DETR-M (not complete yet)
class RTDetrMConfig(RTDetrBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width    = 0.75
        self.depth    = 0.67
        self.ratio    = 1.5
        self.out_stride = [8, 16, 32]
        self.max_stride = 32
        self.scale      = "m"
        ## Backbone
        self.backbone = 'rtcnet'
        self.bk_act   = 'silu'
        self.bk_norm  = 'BN'
        self.bk_depthwise = False
        self.use_pretrained = True
        ## Image Encoder - FPN
        self.fpn_num_blocks = 2
        self.fpn_expand_ratio = 1.0
        self.hidden_dim = 192
        self.en_num_heads = 8
        self.en_num_layers = 1
        self.en_ffn_dim = 1024
        ## Transformer Decoder
        self.de_num_layers = 4
        self.de_ffn_dim    = 1024

# RT-DETR-L (not complete yet)
class RTDetrLConfig(RTDetrBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width    = 1.0
        self.depth    = 1.0
        self.ratio    = 1.0
        self.out_stride = [8, 16, 32]
        self.max_stride = 32
        self.scale      = "l"
        ## Backbone
        self.backbone = 'rtcnet'
        self.bk_act   = 'silu'
        self.bk_norm  = 'BN'
        self.bk_depthwise = False
        self.use_pretrained = True
        ## Image Encoder - FPN
        self.fpn_num_blocks = 3
        self.fpn_expand_ratio = 1.0
        self.hidden_dim = 256
        self.en_num_heads = 8
        self.en_num_layers = 1
        self.en_ffn_dim = 1024
        ## Transformer Decoder
        self.de_num_layers = 6
        self.de_ffn_dim    = 1024

# RT-DETR-X (not complete yet)
class RTDetrXConfig(RTDetrBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width    = 1.25
        self.depth    = 1.0
        self.ratio    = 1.0
        self.out_stride = [8, 16, 32]
        self.max_stride = 32
        self.scale      = "l"
        ## Backbone
        self.backbone = 'rtcnet'
        self.bk_act   = 'silu'
        self.bk_norm  = 'BN'
        self.bk_depthwise = False
        self.use_pretrained = True
        ## Image Encoder - FPN
        self.fpn_num_blocks = 3
        self.fpn_expand_ratio = 1.0
        self.hidden_dim = 320
        self.en_num_heads = 8
        self.en_num_layers = 1
        self.en_ffn_dim = 1280
        ## Transformer Decoder
        self.de_num_layers = 6
        self.de_ffn_dim    = 1280
