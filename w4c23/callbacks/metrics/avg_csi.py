"""
Report Average Critical Success Index (CSI) considering masked values for the different thresholds.
Note that CSI for each threshold requires binary values.
CSI = TP / (TP + FN + FP)
The CSI measures the forecast accuracy at a given rain rate.
"""

import torch

from torchmetrics import Metric
from einops import rearrange


class AverageCriticalSuccessIndex(Metric):
    full_state_update = False
    higher_is_better = True

    def __init__(self, thresholds, num_leadtimes=None):
        super().__init__()
        if num_leadtimes is None or num_leadtimes == 1:
            default = torch.zeros(len(thresholds))
            self.reduce_time = True
        elif num_leadtimes > 1:
            default = torch.zeros(len(thresholds), num_leadtimes)
            self.reduce_time = False
        else:
            raise ValueError("`num_leadtimes` must be > 0")

        self.thresholds = thresholds
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
        for i, threshold in enumerate(self.thresholds):
            pred = prediction.intensity

            # Convert intensity to binary
            pred = pred >= threshold
            lab = label >= threshold

            # Array with each predicition/label after masking
            if self.reduce_time:
                pred = pred[~mask]
                lab = lab[~mask]

            # Combine other dimensions, to have prediciton/label for each time considering masks
            # Masks as TN since they do not impact CSI
            else:
                pred = rearrange(pred, "b c t h w -> (b c h w) t")
                lab = rearrange(lab, "b c t h w -> (b c h w) t")
                m = rearrange(mask, "b c t h w -> (b c h w) t")
                pred = torch.logical_and(pred, ~m)
                lab = torch.logical_and(lab, ~m)

            # Count totals
            self.true_positives[i] += torch.logical_and(pred, lab).sum(dim=0)
            self.false_guesses[i] += (pred != lab).sum(dim=0)

    def compute(self):
        return torch.mean(
            (self.true_positives / (self.true_positives + self.false_guesses)), 0
        )
