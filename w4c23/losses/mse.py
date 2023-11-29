import torch
from torch.nn.functional import mse_loss


class MaskedMSELoss(torch.nn.Module):
    """Obtain MSE ignoring masks"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, prediction, label, mask):
        loss = mse_loss(prediction[~mask], label[~mask])
        return loss
