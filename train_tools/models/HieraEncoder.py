from torch import nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2



class HieraEncoderWrapper(nn.Module):
    """
    Wraps the SAM2 Hiera encoder so that it mimics the SegFormer encoder API for MAnet.
    """

    def __init__(self, hiera_cfg="sam2_hiera_l.yaml"):
        super().__init__()
        self.encoder = build_sam2(hiera_cfg).image_encoder  # Hiera backbone
        self.encoder.scalp = 0 #@todo abbreviation study
        #self.encoder.scalp = 0 # dont drop features (default drops last feature res (HW/32))##@todo check if pretrained weights are bound to scalp = 1
        self._out_channels = (256, 256, 256, 256)
        self._output_stride = 1

    def forward(self, x):
        feats = self.encoder(x)  
        
        return feats["backbone_fpn"]

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def output_stride(self):
        return self._output_stride