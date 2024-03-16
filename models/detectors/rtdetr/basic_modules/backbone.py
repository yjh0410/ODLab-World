from typing import List
import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import (ResNet18_Weights,
                                       ResNet34_Weights,
                                       ResNet50_Weights,
                                       ResNet101_Weights)
from .norm import FrozenBatchNorm2d
from .conv import BasicConv


# IN1K pretrained weights
pretrained_urls = {
    # ResNet series
    'resnet18':  ResNet18_Weights,
    'resnet34':  ResNet34_Weights,
    'resnet50':  ResNet50_Weights,
    'resnet101': ResNet101_Weights,
    # RTCNet series
    'rtcnet_n': "https://github.com/yjh0410/ICLab/releases/download/in1k_pretrained/rtcnet_n_in1k_62.1.pth",
    'rtcnet_s': "https://github.com/yjh0410/ICLab/releases/download/in1k_pretrained/rtcnet_s_in1k_71.3.pth",
    'rtcnet_m': None,
    'rtcnet_l': None,
    'rtcnet_x': None,

}


# ----------------- Model functions -----------------
## Build backbone network
def build_backbone(cfg, pretrained):
    print('==============================')
    print('Backbone: {}'.format(cfg.backbone))
    # ResNet
    if 'resnet' in cfg.backbone:
        pretrained_weight = cfg.pretrained_weight if pretrained else None
        model = build_resnet(cfg, pretrained_weight)
    elif 'rtcnet' in cfg.backbone:
        model = build_rtcnet(cfg)
    else:
        raise NotImplementedError("Unknown backbone: <>.".format(cfg.backbone))
    
    return model


# ----------------- ResNet Backbone -----------------
class ResNet(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self,
                 name: str,
                 norm_type: str,
                 pretrained_weights: str = "imagenet1k_v1",
                 freeze_at: int = -1,
                 freeze_stem_only: bool = False):
        super().__init__()
        # Pretrained
        assert pretrained_weights in [None, "imagenet1k_v1", "imagenet1k_v2"]
        if pretrained_weights is not None:
            if name in ('resnet18', 'resnet34'):
                pretrained_weights = pretrained_urls[name].IMAGENET1K_V1
            else:
                if pretrained_weights == "imagenet1k_v1":
                    pretrained_weights = pretrained_urls[name].IMAGENET1K_V1
                else:
                    pretrained_weights = pretrained_urls[name].IMAGENET1K_V2
        else:
            pretrained_weights = None
        print('- Backbone pretrained weight: ', pretrained_weights)

        # Norm layer
        print("- Norm layer of backbone: {}".format(norm_type))
        if norm_type == 'BN':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'FrozeBN':
            norm_layer = FrozenBatchNorm2d

        # Backbone
        backbone = getattr(torchvision.models, name)(norm_layer=norm_layer, weights=pretrained_weights)
        return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.feat_dims = [128, 256, 512] if name in ('resnet18', 'resnet34') else [512, 1024, 2048]

        # Freeze
        print("- Freeze at: {}".format(freeze_at))
        if freeze_at >= 0:
            for name, parameter in backbone.named_parameters():
                if freeze_stem_only:
                    print("- Freeze stem layer only")
                    if 'layer1' not in name and 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                        parameter.requires_grad_(False)
                else:
                    print("- Freeze stem layer only + layer1")
                    if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                        parameter.requires_grad_(False)

    def forward(self, x):
        xs = self.body(x)
        fmp_list = []
        for name, fmp in xs.items():
            fmp_list.append(fmp)

        return fmp_list

def build_resnet(cfg, pretrained_weight=None):
    # ResNet series
    backbone = ResNet(cfg.backbone,
                      cfg.backbone_norm,
                      pretrained_weight,
                      cfg.freeze_at,
                      cfg.freeze_stem_only)

    return backbone


# ----------------- Yolo Backbone -----------------
class RTCNet(nn.Module):
    def __init__(self, cfg):
        super(RTCNet, self).__init__()
        # ------------------ Basic setting ------------------
        self.model_scale = cfg.scale
        self.feat_dims = [round(64  * cfg.width),
                          round(128 * cfg.width),
                          round(256 * cfg.width),
                          round(512 * cfg.width),
                          round(512 * cfg.width * cfg.ratio)]
        
        # ------------------ Network setting ------------------
        ## P1/2
        self.layer_1 = BasicConv(3, self.feat_dims[0],
                                 kernel_size=3, padding=1, stride=2,
                                 act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise)
        # P2/4
        self.layer_2 = nn.Sequential(
            BasicConv(self.feat_dims[0], self.feat_dims[1],
                      kernel_size=3, padding=1, stride=2,
                      act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),
            ELANLayer(in_dim     = self.feat_dims[1],
                      out_dim    = self.feat_dims[1],
                      num_blocks = round(3*cfg.depth),
                      expansion  = 0.5,
                      shortcut   = True,
                      act_type   = cfg.bk_act,
                      norm_type  = cfg.bk_norm,
                      depthwise  = cfg.bk_depthwise)
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            BasicConv(self.feat_dims[1], self.feat_dims[2],
                      kernel_size=3, padding=1, stride=2,
                      act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),
            ELANLayer(in_dim     = self.feat_dims[2],
                      out_dim    = self.feat_dims[2],
                      num_blocks = round(6*cfg.depth),
                      expansion  = 0.5,
                      shortcut   = True,
                      act_type   = cfg.bk_act,
                      norm_type  = cfg.bk_norm,
                      depthwise  = cfg.bk_depthwise)
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            BasicConv(self.feat_dims[2], self.feat_dims[3],
                      kernel_size=3, padding=1, stride=2,
                      act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),
            ELANLayer(in_dim     = self.feat_dims[3],
                      out_dim    = self.feat_dims[3],
                      num_blocks = round(6*cfg.depth),
                      expansion  = 0.5,
                      shortcut   = True,
                      act_type   = cfg.bk_act,
                      norm_type  = cfg.bk_norm,
                      depthwise  = cfg.bk_depthwise)
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            BasicConv(self.feat_dims[3], self.feat_dims[4],
                      kernel_size=3, padding=1, stride=2,
                      act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),
            ELANLayer(in_dim     = self.feat_dims[4],
                      out_dim    = self.feat_dims[4],
                      num_blocks = round(3*cfg.depth),
                      expansion  = 0.5,
                      shortcut   = True,
                      act_type   = cfg.bk_act,
                      norm_type  = cfg.bk_norm,
                      depthwise  = cfg.bk_depthwise)
        )

        # Initialize all layers
        self.init_weights()
        
        # Load imagenet pretrained weight
        if cfg.use_pretrained:
            self.load_pretrained()

    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    def load_pretrained(self):
        url = pretrained_urls["rtcnet_{}".format(self.model_scale)]
        if url is not None:
            print('Loading backbone pretrained weight from : {}'.format(url))
            # checkpoint state dict
            checkpoint = torch.hub.load_state_dict_from_url(
                url=url, map_location="cpu", check_hash=True)
            checkpoint_state_dict = checkpoint.pop("model")
            # model state dict
            model_state_dict = self.state_dict()
            # check
            for k in list(checkpoint_state_dict.keys()):
                if k in model_state_dict:
                    shape_model = tuple(model_state_dict[k].shape)
                    shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                    if shape_model != shape_checkpoint:
                        checkpoint_state_dict.pop(k)
                else:
                    checkpoint_state_dict.pop(k)
                    print('Unused key: ', k)
            # load the weight
            self.load_state_dict(checkpoint_state_dict)
        else:
            print('No pretrained weight for model scale: {}.'.format(self.model_scale))

    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)
        outputs = [c3, c4, c5]

        return outputs

class YoloBottleneck(nn.Module):
    def __init__(self,
                 in_dim      :int,
                 out_dim     :int,
                 kernel_size :List  = [1, 3],
                 expansion   :float = 0.5,
                 shortcut    :bool  = False,
                 act_type    :str   = 'silu',
                 norm_type   :str   = 'BN',
                 depthwise   :bool  = False,
                 ) -> None:
        super(YoloBottleneck, self).__init__()
        inter_dim = int(out_dim * expansion)
        # ----------------- Network setting -----------------
        self.conv_layer1 = BasicConv(in_dim, inter_dim,
                                     kernel_size=kernel_size[0], padding=kernel_size[0]//2, stride=1,
                                     act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.conv_layer2 = BasicConv(inter_dim, out_dim,
                                     kernel_size=kernel_size[1], padding=kernel_size[1]//2, stride=1,
                                     act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.conv_layer2(self.conv_layer1(x))

        return x + h if self.shortcut else h

class ELANLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expansion  :float = 0.5,
                 num_blocks :int   = 1,
                 shortcut   :bool  = False,
                 act_type   :str   = 'silu',
                 norm_type  :str   = 'BN',
                 depthwise  :bool  = False,
                 ) -> None:
        super(ELANLayer, self).__init__()
        inter_dim = round(out_dim * expansion)
        self.input_proj  = BasicConv(in_dim, inter_dim * 2, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.output_proj = BasicConv((2 + num_blocks) * inter_dim, out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.module      = nn.ModuleList([YoloBottleneck(inter_dim,
                                                         inter_dim,
                                                         kernel_size = [3, 3],
                                                         expansion   = 1.0,
                                                         shortcut    = shortcut,
                                                         act_type    = act_type,
                                                         norm_type   = norm_type,
                                                         depthwise   = depthwise)
                                                         for _ in range(num_blocks)])

    def forward(self, x):
        # Input proj
        x1, x2 = torch.chunk(self.input_proj(x), 2, dim=1)
        out = list([x1, x2])

        # Bottlenecl
        out.extend(m(out[-1]) for m in self.module)

        # Output proj
        out = self.output_proj(torch.cat(out, dim=1))

        return out

def build_rtcnet(cfg): 
    # model
    backbone = RTCNet(cfg)
        
    return backbone