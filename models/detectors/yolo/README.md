# YOLO:

|  Model  |  Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight | Logs |
|---------|--------|-------|------------------------|-------------------|-------------------|--------------------|--------|------|
| YOLO-P  | 8xb16  |  640  |                        |                   |                   |                    |  |
| YOLO-N  | 8xb16  |  640  |                        |                   |                   |                    |  |
| YOLO-S  | 8xb16  |  640  |                        |                   |                   |                    |  |
| YOLO-M  | 8xb16  |  640  |                        |                   |                   |                    |  |
| YOLO-L  | 8xb16  |  640  |                        |                   |                   |                    |  |
| YOLO-X  | 8xb16  |  640  |                        |                   |                   |                    |  |


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
python train.py --cuda -d coco --root path/to/coco -m yolo_s -bs 16 --fp16 --resume path/to/yolo.pth --eval_first
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
