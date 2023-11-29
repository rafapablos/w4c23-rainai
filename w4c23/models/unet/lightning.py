"""
Simple UNet where channels are used for the time dimension as well.
"""

import torch
import torchvision

import torch.nn as nn

from einops import rearrange, repeat

from .model import UNet
from w4c23.models.baseModule import BaseModule


class UNetModule(BaseModule):
    def __init__(self, model_params: dict, params: dict):
        super().__init__(model_params, params)

        self.input_crop = model_params["input_crop"]
        self.padding = model_params["padding"]
        self.conditioning_lead_time = model_params["conditioning_lead_time"]

        self.example_input_array = torch.rand(
            self.bs,
            self.in_channels,
            self.history_length,
            self.input_crop,
            self.input_crop,
        )

        self.center_crop = torchvision.transforms.CenterCrop(self.input_crop)

        # UNet Model
        if self.conditioning_lead_time:
            extra_channels = 1
            classes = self.num_classes
        else:
            extra_channels = 0
            classes = self.forecast_length * self.num_classes
        self.model = UNet(
            in_channels=extra_channels + self.history_length * self.in_channels,
            classes=classes,
        )

        # Combine time and channels
        self.stack_time = lambda x: rearrange(x, "b c t h w -> b (c t) h w")

        # UNET processes 4D inputs / outputs
        if self.conditioning_lead_time:
            # Stack forecast timesteps as different samples in the batch
            self.stack_time_in_batch = lambda x: rearrange(
                x, "b t c h w -> (b t) c h w"
            )

            self.unstack_time_from_batch = lambda x: rearrange(
                x,
                "(b t) c h w -> b c t h w",
                b=self.bs,
                t=self.forecast_length,
            )
        else:
            self.unstack_time_from_channel = lambda x: rearrange(
                x,
                "b (c t) h w -> b c t h w",
                t=self.forecast_length,
                c=self.num_classes,
            )

        self.save_hyperparameters()

    def add_leadtime_input(self, input):
        """
        Add forecast lead time to the input as an additional channel and geenrate sample for every lead time
        Input: B,T_HxC,H,W
        Output: B,T_F,T_HxC+1,H,W
        T_H is history timesteps, T_F is forecast timesteps
        """
        b, _, h, w = input.shape

        # Repeat sample for every forecast timestep
        input = repeat(input, "b c h w -> b t c h w", t=self.forecast_length)

        # Create conditioning lead time channel
        clt = torch.arange(self.forecast_length, device=self.device)
        clt = repeat(clt, "t -> b t c h w", b=b, c=1, h=h, w=w)

        # Add conditioning lead time to input
        return torch.cat([input, clt], dim=2)

    def transform_input(self, input):
        # TODO - Change to data loader
        input = self.center_crop(input)
        return input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad input to be divisible by 32
        if self.padding:
            x = torchvision.transforms.Pad(
                self.padding, fill=0, padding_mode="constant"
            )(x)

        # Combine timestep as different channels
        # B,C,T,H,W -> B,TxC,H,W (t is history)
        x = self.stack_time(x)

        if self.conditioning_lead_time:
            # Add conditioning leadtime with an extra channel and sample for every lead time
            # B,T_HxC,H,W -> B,T_F,T_HxC+1,H,W
            x = self.add_leadtime_input(x)
            # Combine forecast timesteps as different samples in batch
            x = self.stack_time_in_batch(x)
            # Run through UNet
            x = self.model(x)
            # Unstack channels to obtain sequences of timesteps
            # BxT,C,H,W -> B,C,T,H,W (t is forecast)
            x = self.unstack_time_from_batch(x)
        else:
            # Obtain output from UNet with channel for each bucket and forecast timestep
            x = self.model(x)
            # Unstack channels to obtain bucket channels for each timestep in different dimensions
            # B,TxC,H,W -> B,C,T,H,W (t is forecast)
            x = self.unstack_time_from_channel(x)

        # Take original center region
        if self.padding:
            x = x[:, :, :, self.padding : -self.padding, self.padding : -self.padding]

        # Apply activation function (i.e. softmax if probabilistic loss)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x
