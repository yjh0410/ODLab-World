#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
from .yolov8.build import build_yolov8
from .rtdetr.build import build_rtdetr


# build object detector
def build_model(args, cfg, is_val=False):
    # ------------ build object detector ------------
    ## YOLOv8    
    if 'yolov8' in args.model:
        model, criterion = build_yolov8(cfg, is_val)
    # RT-DETR
    elif 'rtdetr' in args.model:
        model, criterion = build_rtdetr(cfg, is_val)

    if is_val:
        # ------------ Load pretrained weight ------------
        if args.pretrained is not None:
            print('Loading COCO pretrained weight ...')
            checkpoint = torch.load(args.pretrained, map_location='cpu')
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            # model state dict
            model_state_dict = model.state_dict()
            # check
            for k in list(checkpoint_state_dict.keys()):
                if k in model_state_dict:
                    shape_model = tuple(model_state_dict[k].shape)
                    shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                    if shape_model != shape_checkpoint:
                        checkpoint_state_dict.pop(k)
                        print(k)
                else:
                    checkpoint_state_dict.pop(k)
                    print(k)

            model.load_state_dict(checkpoint_state_dict, strict=False)

        # ------------ Keep training from the given weight ------------
        if args.resume and args.resume != "None":
            print('keep training: ', args.resume)
            checkpoint = torch.load(args.resume, map_location='cpu')
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            model.load_state_dict(checkpoint_state_dict)
            del checkpoint, checkpoint_state_dict

        return model, criterion

    else:      
        return model