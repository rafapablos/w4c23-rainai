"""
Report Critical Success Index (CSI) considering masked values.
Note that CSI requires binary values.
CSI = TP / (TP + FN + FP)
The CSI measures the forecast accuracy at a given rain rate.
"""

import torch

from torchmetrics import Metric
from einops import rearrange


class CriticalSuccessIndex(Metric):
    full_state_update = False
    higher_is_better = True

    def __init__(self, threshold, num_leadtimes=None):
        super().__init__()
        if num_leadtimes is None or num_leadtimes == 1:
            default = torch.tensor([0])
            self.reduce_time = True
        elif num_leadtimes > 1:
            default = torch.zeros(num_leadtimes)
            self.reduce_time = False
        else:
            raise ValueError("`num_leadtimes` must be > 0")

        self.threshold = threshold
        self.add_state(
            "true_positives",
            default=default.clone(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "false_guesses",
            default=default.clone(),
            dist_reduce_fx="sum",
        )

    def update(self, prediction, label, mask):
        prediction = prediction.intensity
        # Convert intensity to binary
        prediction = prediction >= self.threshold
        label = label >= self.threshold

        # Array with each predicition/label after masking
        if self.reduce_time:
            prediction = prediction[~mask]
            label = label[~mask]

        # Combine other dimensions, to have prediciton/label for each time considering masks
        # Masks as TN since they do not impact CSI
        else:
            prediction = rearrange(prediction, "b c t h w -> (b c h w) t")
            label = rearrange(label, "b c t h w -> (b c h w) t")
            mask = rearrange(mask, "b c t h w -> (b c h w) t")
            prediction = torch.logical_and(prediction, ~mask)
            label = torch.logical_and(label, ~mask)

        # Count totals
        self.true_positives += torch.logical_and(prediction, label).sum(dim=0)
        self.false_guesses += (prediction != label).sum(dim=0)

    def compute(self):
        return self.true_positives / (self.true_positives + self.false_guesses)
