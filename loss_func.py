import torch
import torch.nn as nn

class NegDiceLoss(nn.Module):
    def __init__(self):
        super(NegDiceLoss, self).__init__()
    def forward(self, logits, targets):
        smooth = 1.
        logits = torch.flatten(logits)
        targets = torch.flatten(targets)
        intersection = (logits * targets).sum()

        negative_dice = 1 -(((2. * intersection) + smooth) / (logits.sum() + targets.sum() + smooth))
        return negative_dice
