import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets):

        epsilon = 1.
        # axes = tuple(range(3, len(logits.shape) - 1))
        axes = tuple(range(2, len(logits.shape))) # 3D
        # axes = tuple(range(3, len(logits.shape))) # 2D
        numerator = 2 * torch.sum(logits * targets, axes)
        denominator = torch.sum(torch.square(logits) + torch.square(targets), axes)

        return 1 - torch.mean((numerator + epsilon) / (denominator + epsilon))
