# ------------------ Model Config ------------------
from .yolo_config   import build_yolo_config
from .rtdetr_config import build_rtdetr_config

def build_config(args):
    print('==============================')
    print('Model: {} ...'.format(args.model.upper()))
    # YOLOv1
    if 'yolo' in args.model:
        cfg = build_yolo_config(args)
    # RT-DETR
    elif 'rtdetr' in args.model:
        cfg = build_rtdetr_config(args)
    else:
        raise NotImplementedError("Unknown model config: {}".format(args.model))
    
    # Print model config
    cfg.print_config()

    return cfg

