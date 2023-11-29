"""
Simple UNet where channels are used for the time dimension.
"""

import torch
import segmentation_models_pytorch as smp


class UNet(torch.nn.Module):
    def __init__(self, in_channels: int, classes: int):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name="tu-resnet18", in_channels=in_channels, classes=classes
        )

    def forward(self, inputs):
        x = self.unet(inputs)
        return x
