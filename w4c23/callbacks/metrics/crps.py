"""
Report CRPS without considering masked values.
Continuous Ranked Probability Score (CRPS)
https://arxiv.org/pdf/2111.07470.pdf

CRPS: Brier Score integrated over all rates
CRPS = Sum of each Brier Score times intensity range

BS: error between ground truth rate and the probability for that rate
BS_r = Average over all samples the squared error between the predicted probability and ground truth.
"""

import torch
from torchmetrics import Metric


class ContinuousRankedProbabilityScore(Metric):
    full_state_update = False
    higher_is_better = False

    def __init__(self, buckets):
        super().__init__()

        self.bucket_ranges = torch.tensor(buckets.ranges).cuda()
        self.bucket_boundaries = torch.tensor(buckets.boundaries).cuda()

        default = torch.zeros(len(self.bucket_ranges))

        # Class error as the sum of the squared differences without the average
        self.add_state(
            "class_errors",
            default=default.clone(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "class_count",
            default=default.clone(),
            dist_reduce_fx="sum",
        )

    def _brier_score(self, prediction, label, mask, bucket):
        p_r = prediction[:, bucket].unsqueeze(1)
        if bucket > 0:
            filter = label >= self.bucket_boundaries[bucket - 1]
        else:
            filter = label
        error = torch.square(p_r - 1 * filter)
        error = error[~mask]
        self.class_errors[bucket] += error.sum()
        self.class_count[bucket] += error.numel()

    def update(self, prediction, label, mask):
        prediction = prediction.probabilities

        # Compute BS_r for each bucket
        for i in range(len(self.class_errors)):
            self._brier_score(prediction, label, mask, i)

    def compute(self):
        return (self.class_errors / self.class_count * self.bucket_ranges).sum()
