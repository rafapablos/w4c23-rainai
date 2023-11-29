"""
Base module for models with intensity output and probabilities output. Abstract class that requires validation and prediction implementation.
"""

import torch
import random
from torch import nn
from pytorch_lightning import LightningModule
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from torchvision.transforms import v2

from w4c23.utils.buckets import BUCKET_CONSTANTS
from w4c23.losses import *
from w4c23.models.upsample import *


@dataclass
class ValidationOutput:
    """Output from validation phase of a model

    Attributes:
        intensity: tensor of shape (b, 1, t, h, w)
        probabilities: tensor of shape (b, c, t, h, w)
    """

    intensity: torch.Tensor
    probabilities: Optional[torch.Tensor] = None


class BaseModule(LightningModule, ABC):
    def __init__(self, model_params: dict, params: dict) -> None:
        super().__init__()
        self.history_length = model_params["history_length"]
        self.in_channels = model_params["in_channels"]
        self.forecast_length = model_params["forecast_length"]

        self.lr = float(params["lr"])
        self.weight_decay = float(params["weight_decay"])
        self.bs = params["batch_size"]

        self.static_data = params["static_data"]
        self.transform = params["transform"]

        self.activation = model_params["activation"]
        self.activation_fn = {
            "none": None,
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "softmax": nn.Softmax(dim=1),
        }[self.activation]

        self.loss = params["loss"]
        self.weights = params["weights"]
        self.buckets = BUCKET_CONSTANTS[model_params["buckets"]]

        self.loss_fn = {
            "MSE": MaskedMSELoss(),
            "CrossEntropy": MaskedCrossEntropyLoss(self.buckets, self.weights),
        }[self.loss]
        self.probabilistic = {
            "MSE": False,
            "CrossEntropy": True,
        }[self.loss]
        self.num_classes = {
            "MSE": 1,
            "CrossEntropy": self.buckets.num_buckets,
        }[self.loss]

        if self.probabilistic:
            # Store bucket means (but not as model parameter) as the channel dimension of the data
            self.register_buffer(
                "bucket_means",
                torch.tensor(self.buckets.means).view(1, -1, 1, 1, 1),
            )
            self.bucket_means: torch.Tensor

        if model_params["upsample"] == "bilinear":
            self.upsample = BilinearUpsample(42, 252, self.forecast_length)
        elif model_params["upsample"] == "nearest":
            self.upsample = NearestUpsample(42, 252, self.forecast_length)
        elif model_params["upsample"] == "ninasr":
            self.upsample = NinaSRUpsample(
                42, 252, self.forecast_length, self.num_classes
            )
        elif model_params["upsample"] == "edsr":
            self.upsample = EDSRUpsample(
                42, 252, self.forecast_length, self.num_classes
            )
        else:
            self.upsample = None

    def integrate(self, predictions):
        """Obtain a value from probability distribution."""
        return (predictions * self.bucket_means).sum(dim=1, keepdim=True)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def transform_input(self, input):
        # TODO - Change to data loader
        return input

    def augment_batch(self, batch):
        """Apply augmentation on training batches (flips and 90-degrees rotation)"""
        # TODO - Change to data loader
        if not self.transform:
            return batch
        input, label, metadata = batch
        angle = random.choice([-90, 0, 90, 180])
        transformations = [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation([angle, angle]),
        ]

        t = random.choice(transformations)
        input = t(input).contiguous()
        label = t(label).contiguous()
        # Transform masks
        metadata["input"]["mask"] = t(metadata["input"]["mask"])
        metadata["target"]["mask"] = t(metadata["target"]["mask"])
        # Transform static data if any
        if self.static_data:
            metadata["input"]["topo"] = t(metadata["input"]["topo"])
            metadata["target"]["topo"] = t(metadata["target"]["topo"])
            metadata["input"]["lat-long"] = t(metadata["input"]["lat-long"])
            metadata["target"]["lat-long"] = t(metadata["target"]["lat-long"])
        return input, label, metadata

    def add_static(self, input, metadata):
        lat_long = (
            metadata["input"]["lat-long"]
            .unsqueeze(2)
            .repeat(1, 1, self.history_length, 1, 1)
        )
        topo = (
            metadata["input"]["topo"]
            .unsqueeze(2)
            .repeat(1, 1, self.history_length, 1, 1)
        )
        input = torch.cat([input, lat_long, topo], dim=1)
        return input

    def training_step(self, batch):
        batch = self.augment_batch(batch)
        input, label, metadata = batch
        # Add static data to input if required
        if self.static_data:
            input = self.add_static(input, metadata)
        input = self.transform_input(input)
        prediction = self.forward(input)
        if self.upsample:
            prediction = self.upsample(prediction)
        mask = metadata["target"]["mask"]
        loss = self.loss_fn(prediction, label, mask)
        self.log("train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx) -> ValidationOutput:
        input, label, metadata = batch
        # Add static data to input if required
        if self.static_data:
            input = self.add_static(input, metadata)
        input = self.transform_input(input)
        prediction = self.forward(input)
        if self.upsample:
            prediction = self.upsample(prediction)
        mask = metadata["target"]["mask"]
        loss = self.loss_fn(prediction, label, mask)
        self.log("val/loss", loss, sync_dist=True)
        if self.probabilistic:
            # If no softmax, apply as it is required for the metrics (i.e. CRPS)
            if self.activation == "none":
                prediction = nn.functional.softmax(prediction, dim=1)
            probabilities = prediction
            intensity = self.integrate(prediction)
        else:
            probabilities = None
            intensity = prediction
        return ValidationOutput(intensity=intensity, probabilities=probabilities)

    def predict_step(self, batch, batch_idx=None) -> torch.Tensor:
        input, _, metadata = batch
        # Add static data to input if required
        if self.static_data:
            input = self.add_static(input, metadata)
        input = self.transform_input(input)
        prediction = self.forward(input)
        if self.upsample:
            prediction = self.upsample(prediction)
        if self.probabilistic:
            # If no softmax, apply as it to sum 1
            if self.activation == "none":
                prediction = nn.functional.softmax(prediction, dim=1)
            probabilities = prediction
            intensity = self.integrate(prediction)
        else:
            probabilities = None
            intensity = prediction
        intensity = intensity[:, :, : self.forecast_length, :, :]
        return intensity

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
