# YOLO:

|   Model   |  Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-----------|--------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLO-N  | 8xb16  |  640  |          37.0          |        52.9       |        8.8        |         3.2        | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolo_n_coco.pth) |
| YOLO-S  | 8xb16  |  640  |          43.5          |        60.4       |       28.8        |         11.2       | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolo_s_coco.pth) |
| YOLO-M  | 8xb16  |  640  |                        |                   |                   |                    |  |
| YOLO-L  | 8xb16  |  640  |          50.7          |        68.3       |       165.7       |         43.7       | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolo_l_coco.pth) |

- For training, we train YOLO series with 500 epochs on COCO.
- For data augmentation, we use the random affine, hsv augmentation, mosaic augmentation and mixup augmentation, following the setting of [YOLO](https://github.com/ultralytics/yolov8).
- For optimizer, we use AdamW with weight decay 0.05 and base per image lr 0.001 / 64, which is different from the official YOLO. We have tried SGD, but it has weakened performance. For example, when using SGD, YOLO-N's AP was only 35.8%, lower than the current result (36.8 %), perhaps because some hyperparameters were not set properly.
- For learning rate scheduler, we use linear decay scheduler.


## Train YOLO
### Single GPU
Taking training YOLO-S on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m yolo_s -bs 16 -size 640 --wp_epoch 3 --max_epoch 500 --eval_epoch 10 --no_aug_epoch 20 --ema --fp16 --multi_scale 
```

### Multi GPU
Taking training YOLO on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda -dist -d coco --root /data/datasets/ -m yolo_s -bs 128 -size 640 --wp_epoch 3 --max_epoch 500  --eval_epoch 10 --no_aug_epoch 20 --ema --fp16 --sybn --multi_scale --save_folder weights/ 
```

## Test YOLO
Taking testing YOLO on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m yolo_s --weight path/to/yolo.pth -size 640 -vt 0.4 --show 
```

## Evaluate YOLO
Taking evaluating YOLO on COCO-val as the example,
```Shell
python eval.py --cuda -d coco-val --root path/to/coco -m yolo_s --weight path/to/yolo.pth 
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m yolo_s --weight path/to/weight -size 640 -vt 0.4 --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m yolo_s --weight path/to/weight -size 640 -vt 0.4 --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m yolo_s --weight path/to/weight -size 640 -vt 0.4 --show --gif
```
