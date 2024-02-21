# ------------------ Model Config ------------------
from .yolo_config   import build_yolo_config
from .rtdetr_config import build_rtdetr_config

def build_config(args):
    print('==============================')
    print('Model: {} ...'.format(args.model.upper()))
    # YOLOv1
    if args.model in ['yolo_n', 'yolo_s', 'yolo_m', 'yolo_l', 'yolo_x']:
        cfg = build_yolo_config(args)
    # RT-DETR
    elif args.model in ['rtdetr_r18', 'rtdetr_r34', 'rtdetr_r50', 'rtdetr_r101']:
        cfg = build_rtdetr_config(args)
    else:
        raise NotImplementedError("Unknown model config: {}".format(args.model))

    return cfg

