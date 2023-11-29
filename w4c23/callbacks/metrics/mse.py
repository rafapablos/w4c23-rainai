"""
Report MSE without considering masked values. Allows for MSE at each lead time.
"""

import torch
from torchmetrics import Metric
from einops import rearrange
from torch.nn.functional import mse_loss


class MeanSquaredError(Metric):
    full_state_update = False
    higher_is_better = False

    def __init__(self, num_leadtimes=None):
        super().__init__()
        if num_leadtimes is None or num_leadtimes == 1:
            default = torch.tensor([0])
            self.reduce_time = True
        elif num_leadtimes > 1:
            default = torch.zeros(num_leadtimes)
            self.reduce_time = False
        else:
            raise ValueError("`num_leadtimes` must be > 0")

        # Error as the sum of the squared errors without the average
        self.add_state(
            "error",
            default=default.clone(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count",
            default=default.clone(),
            dist_reduce_fx="sum",
        )

    def update(self, prediction, label, mask):
        prediction = prediction.intensity
        # Array with each predicition/target
        if self.reduce_time:
            prediction = prediction[~mask]
            label = label[~mask]
            # Compute error and count
            self.error += mse_loss(prediction, label, reduction="none").sum().long()
            self.count += prediction.numel()

        # Combine other dimensions, to have prediciton/target for each time
        else:
            prediction = rearrange(prediction, "b c t h w -> (b c h w) t")
            label = rearrange(label, "b c t h w -> (b c h w) t")
            mask = rearrange(mask, "b c t h w -> (b c h w) t")
            loss = mse_loss(prediction, label, reduction="none")
            loss = loss.masked_fill(mask, 0)
            self.error += loss.sum(dim=0).long()
            self.count += prediction.shape[0] - mask.sum(dim=0)

    def compute(self):
        return self.error / self.count
