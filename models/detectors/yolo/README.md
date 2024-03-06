# YOLO:
Our YOLO detector is equal to YOLOv8, including model structure, label assignment and loss function.

|  Model  |  Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | ckpt | logs |
|---------|--------|-------|------------------------|-------------------|-------------------|--------------------|--------|------|
| YOLO-P  | 8xb16  |  640  |                        |                   |                   |                    |  |  |
| YOLO-N  | 8xb16  |  640  |                        |                   |                   |                    |  |  |
| YOLO-S  | 8xb16  |  640  |                        |                   |                   |                    |  |  |
| YOLO-M  | 8xb16  |  640  |                        |                   |                   |                    |  |  |
| YOLO-L  | 8xb16  |  640  |                        |                   |                   |                    |  |  |
| YOLO-X  | 8xb16  |  640  |                        |                   |                   |                    |  |  |

- For training, I replace the `SGD` used in YOLOv8 with the `AdamW` with weight decay of `0.05` and per image's base lr of `0.001 / 64` as the optimizer. I am not good at using the SGD optimizer.

- All the models are trained from the scratch with `500 epoch`, except that `YOLO-P` used the imagenet-1k pretrained weight. Based on our experience, for lightweight detectors that heavily use depthwise convolution, using imagenet-1k pre training weights could significantly improve model performance, while relying solely on the train from scratch strategy on COCO may not be sufficient.

## Train YOLO
### Single GPU
Taking training YOLO-S on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m yolo_s -bs 16  --fp16
```

### Multi GPU
Taking training YOLO on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda --distributed -d coco --root /data/datasets/ -m yolo_s -bs 128 --fp16 --sybn 
```

## Test YOLO
Taking testing YOLO on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m yolo_s --weight path/to/yolo.pth --show 
```

## Evaluate YOLO
Taking evaluating YOLO on COCO-val as the example,
```Shell
python eval.py --cuda -d coco --root path/to/coco -m yolo_s --weight path/to/yolo.pth
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m yolo_s --weight path/to/weight --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m yolo_s --weight path/to/weight --show
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m yolo_s --weight path/to/weight --show
```
