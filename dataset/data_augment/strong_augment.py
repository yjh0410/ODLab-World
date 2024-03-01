import random
import cv2
import numpy as np

from .yolo_augment import random_perspective


# ------------------------- Strong augmentations -------------------------
## Mosaic Augmentation
class MosaicAugment(object):
    def __init__(self,
                 img_size,
                 affine_params,
                 is_train=False,
                 ) -> None:
        self.img_size = img_size
        self.is_train = is_train
        self.affine_params = affine_params

    def __call__(self, image_list, target_list):
        assert len(image_list) == 4
        # mosaic center
        yc, xc = [int(random.uniform(-x, 2*self.img_size + x)) for x in [-self.img_size // 2, -self.img_size // 2]]

        mosaic_bboxes = []
        mosaic_labels = []
        mosaic_img = np.zeros([self.img_size*2, self.img_size*2, image_list[0].shape[2]], dtype=np.uint8)
        for i in range(4):
            img_i, target_i = image_list[i], target_list[i]
            bboxes_i = target_i["boxes"]
            labels_i = target_i["labels"]
            orig_h, orig_w, _ = img_i.shape

            # ------------------ Resize ------------------
            img_i = cv2.resize(img_i, (self.img_size, self.img_size))
            h, w, _ = img_i.shape

            # ------------------ Create mosaic image ------------------
            ## Place image in mosaic image
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            ## Mosaic target
            bboxes_i_ = bboxes_i.copy()
            if len(bboxes_i) > 0:
                # a valid target, and modify it.
                bboxes_i_[:, 0] = (w * bboxes_i[:, 0] / orig_w + padw)
                bboxes_i_[:, 1] = (h * bboxes_i[:, 1] / orig_h + padh)
                bboxes_i_[:, 2] = (w * bboxes_i[:, 2] / orig_w + padw)
                bboxes_i_[:, 3] = (h * bboxes_i[:, 3] / orig_h + padh)    

                mosaic_bboxes.append(bboxes_i_)
                mosaic_labels.append(labels_i)

        if len(mosaic_bboxes) == 0:
            mosaic_bboxes = np.array([]).reshape(-1, 4)
            mosaic_labels = np.array([]).reshape(-1)
        else:
            mosaic_bboxes = np.concatenate(mosaic_bboxes)
            mosaic_labels = np.concatenate(mosaic_labels)

        # clip
        mosaic_bboxes = mosaic_bboxes.clip(0, self.img_size * 2)

        # ----------------------- Random perspective -----------------------
        mosaic_targets = np.concatenate([mosaic_labels[..., None], mosaic_bboxes], axis=-1)
        mosaic_img, mosaic_targets = random_perspective(
            mosaic_img,
            mosaic_targets,
            self.affine_params['degrees'],
            translate   = self.affine_params['translate'],
            scale       = self.affine_params['scale'],
            shear       = self.affine_params['shear'],
            perspective = self.affine_params['perspective'],
            border      = [-self.img_size//2, -self.img_size//2]
            )

        # target
        mosaic_target = {
            "boxes": mosaic_targets[..., 1:],
            "labels": mosaic_targets[..., 0],
        }

        return mosaic_img, mosaic_target

## Mixup Augmentation
class MixupAugment(object):
    def __init__(self, img_size) -> None:
        self.img_size = img_size

    def __call__(self, origin_image, origin_target, new_image, new_target):
        if origin_image.shape[:2] != new_image.shape[:2]:
            img_size = max(new_image.shape[:2])
            # origin_image is not a mosaic image
            orig_h, orig_w = origin_image.shape[:2]
            scale_ratio = img_size / max(orig_h, orig_w)
            if scale_ratio != 1: 
                interp = cv2.INTER_LINEAR if scale_ratio > 1 else cv2.INTER_AREA
                resize_size = (int(orig_w * scale_ratio), int(orig_h * scale_ratio))
                origin_image = cv2.resize(origin_image, resize_size, interpolation=interp)

            # pad new image
            pad_origin_image = np.zeros([img_size, img_size, origin_image.shape[2]], dtype=np.uint8)
            pad_origin_image[:resize_size[1], :resize_size[0]] = origin_image
            origin_image = pad_origin_image.copy()
            del pad_origin_image

        r = np.random.beta(32.0, 32.0)
        mixup_image = r * origin_image.astype(np.float32) + \
                    (1.0 - r)* new_image.astype(np.float32)
        mixup_image = mixup_image.astype(np.uint8)
        
        cls_labels = new_target["labels"].copy()
        box_labels = new_target["boxes"].copy()

        mixup_bboxes = np.concatenate([origin_target["boxes"], box_labels], axis=0)
        mixup_labels = np.concatenate([origin_target["labels"], cls_labels], axis=0)

        mixup_target = {
            "boxes": mixup_bboxes,
            "labels": mixup_labels,
        }
        
        return mixup_image, mixup_target
