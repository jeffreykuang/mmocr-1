from mmcv.runner import auto_fp16
from torch import nn

from mmdet.models.builder import NECKS


@NECKS.register_module()
class IDENTITY(nn.Module):
    """FPN-like fusion module in Shape Robust Text Detection with Progressive
    Scale Expansion Network."""

    def __init__(self):
        super().__init__()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        pass

    @auto_fp16()
    def forward(self, inputs):
        return inputs
