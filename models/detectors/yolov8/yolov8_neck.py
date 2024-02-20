import torch
import torch.nn as nn

from .yolov8_basic import Conv


# Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
class SPPF(nn.Module):
    """
        This code referenced to https://github.com/ultralytics/yolov5
    """
    def __init__(self, cfg, in_dim, out_dim):
        super().__init__()
        inter_dim = round(in_dim * cfg.neck_expand_ratio)
        self.out_dim = out_dim
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=cfg.neck_act, norm_type=cfg.neck_norm)
        self.cv2 = Conv(inter_dim * 4, out_dim, k=1, act_type=cfg.neck_act, norm_type=cfg.neck_norm)
        self.m = nn.MaxPool2d(kernel_size=cfg.spp_pooling_size, stride=1, padding=cfg.spp_pooling_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)

        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
