"""
Report MAE without considering masked values.
"""

import torch
from torchmetrics import Metric
from torch.nn.functional import l1_loss


class MeanAverageError(Metric):
    full_state_update = False
    higher_is_better = False

    def __init__(self):
        super().__init__()
        default = torch.tensor([0])

        # Error as the sum of the absolute errors without the average
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
        prediction = prediction[~mask]
        label = label[~mask]
        # Compute error and count
        self.error += l1_loss(prediction, label, reduction="none").sum().long()
        self.count += prediction.numel()

    def compute(self):
        return self.error / self.count
