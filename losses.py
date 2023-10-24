import torch
from torch.nn.modules.loss import L1Loss, BCELoss


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        true = true.view(-1)
        intersection = (pred * true).sum()
        return 1 - (
            (2 * intersection + self.smooth) / (pred.sum() + true.sum() + self.smooth)
        )
