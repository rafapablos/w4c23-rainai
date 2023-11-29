"""
Simple bilinear interpolation for upsampling center region
"""

import torch
from .upsample import Upsample


class NearestUpsample(Upsample):
    def __init__(self, center_region, output_size, forecast_length):
        super().__init__(center_region, forecast_length)
        self.upsample = torch.nn.Upsample(size=output_size, mode="nearest")
