import torch
from torch.nn.functional import cross_entropy


class MaskedCrossEntropyLoss(torch.nn.Module):
    """
    Obtain cross entropy using the buckets as classes ignoring masks.
    Input:
    - prediction: [B,C,T,H,W]
    - label: [B,1,T,H,W] representing the amount to be converted to class
    - mask: [B,1,T,H,W]
    """

    def __init__(self, buckets, weights) -> None:
        super().__init__()
        # Store bucket boundaries but not as model parameter
        self.register_buffer(
            "bucket_boundaries",
            torch.tensor(buckets.boundaries),
        )
        self.bucket_boundaries: torch.Tensor
        if weights:
            self.register_buffer(
                "bucket_weights",
                torch.tensor(buckets.weights),
            )
            self.bucket_weights: torch.Tensor
        else:
            self.bucket_weights = None

    def forward(self, prediction, label, mask):
        mask = torch.squeeze(mask, 1)  # Transform [B,C,T,H,W] to [B,T,H,W]
        # Transform [B,C,T,H,W] to [B,T,H,W] representing the class
        label_class = torch.bucketize(label, self.bucket_boundaries)
        label_class = torch.squeeze(label_class, 1).long()

        loss = cross_entropy(
            prediction, label_class, weight=self.bucket_weights, reduction="none"
        )
        # Add masks to unreduced loss [B,T,H,W]
        avg_masked = torch.mean(loss[~mask])

        return avg_masked
