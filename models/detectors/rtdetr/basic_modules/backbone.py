import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import (ResNet18_Weights,
                                       ResNet34_Weights,
                                       ResNet50_Weights,
                                       ResNet101_Weights)
try:
    from .basic import FrozenBatchNorm2d
except:
    from basic  import FrozenBatchNorm2d
   

# IN1K pretrained weights
pretrained_urls = {
    # ResNet series
    'resnet18':  ResNet18_Weights,
    'resnet34':  ResNet34_Weights,
    'resnet50':  ResNet50_Weights,
    'resnet101': ResNet101_Weights,
    # ShuffleNet series
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
                    if 'layer1' not in name and 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                        parameter.requires_grad_(False)
                else:
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
