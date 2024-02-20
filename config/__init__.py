# ------------------ Model Config ------------------
from .yolov8_config import build_yolov8_config
from .rtdetr_config import build_rtdetr_config

def build_config(args):
    print('==============================')
    print('Model: {} ...'.format(args.model.upper()))
    # YOLOv1
    if args.model in ['yolov8_n', 'yolov8_s', 'yolov8_m', 'yolov8_l', 'yolov8_x']:
        cfg = build_yolov8_config(args)
    # RT-DETR
    elif args.model in ['rtdetr_r18', 'rtdetr_r34', 'rtdetr_r50', 'rtdetr_r101']:
        cfg = build_rtdetr_config(args)
    else:
        raise NotImplementedError("Unknown model config: {}".format(args.model))

    return cfg

