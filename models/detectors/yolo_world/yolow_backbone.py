import torch
import torch.nn as nn

try:
    from .yolow_basic import BasicConv, ELANLayer
except:
    from  yolow_basic import BasicConv, ELANLayer


# -------------------- Vision Backbone -----------------------
class YolowVisionBackbone(nn.Module):
    def __init__(self, cfg):
        super(YolowVisionBackbone, self).__init__()
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
        
    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)
        outputs = [c3, c4, c5]

        return outputs

def build_vision_backbone(cfg): 
    # model
    backbone = YolowVisionBackbone(cfg)
        
    return backbone


# -------------------- Prompt Backbone -----------------------
class YolowPromptBackbone(nn.Module):
    def __init__(self, cfg):
        super(YolowPromptBackbone, self).__init__()

    def forward(self, x):
        return

def build_prompt_backbone(cfg): 
    # model
    backbone = YolowPromptBackbone(cfg)
        
    return backbone