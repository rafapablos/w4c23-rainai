"""
Simple bilinear interpolation for upsampling center region
"""

import torch
import torchvision
from einops import rearrange
from abc import ABC


class Upsample(torch.nn.Module, ABC):
    def __init__(self, center_region, forecast_length):
        super().__init__()

        self.center_crop = torchvision.transforms.CenterCrop(center_region)
        self.upsample = None

        self.stack_time_in_batch = lambda x: rearrange(x, "b c t h w -> (b t) c h w")

        self.unstack_time_from_batch = lambda x: rearrange(
            x,
            "(b t) c h w -> b c t h w",
            t=forecast_length,
        )

    def forward(self, x):
        """
        Crop center region and upsample it to the required size
        Input: B C T H W
        """
        if self.upsample is None:
            raise Exception("Upsample model must be iniialized")
        x = self.stack_time_in_batch(x)
        x = self.center_crop(x)
        x = self.upsample(x)
        x = self.unstack_time_from_batch(x)
        return x
