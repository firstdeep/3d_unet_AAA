import torch
import torch.nn as nn
from torch import einsum


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, net_output, gt):
        smooth = 1e-5

        axes = tuple(range(2, len(net_output.shape))) # 3D
        numerator = torch.sum(net_output * gt, axes)
        denominator = torch.sum(net_output, axes) + torch.sum(gt, axes)
        dice = torch.mean((2 * numerator + smooth) / (denominator + smooth))
        dc = 1-dice

        return dc


class DiceLoss_bh(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss_bh, self).__init__()

        self.smooth = smooth

    def forward(self, net_output, gt):

        intersection: torch.Tensor = einsum("bcxyz, bcxyz->bc", net_output, gt)
        union: torch.Tensor = (einsum("bcxyz->bc", net_output) + einsum("bcxyz->bc", gt))
        divided: torch.Tensor = 1 - (2 * einsum("bc->b", intersection) + self.smooth) / (
                    einsum("bc->b", union) + self.smooth)
        dc = divided.mean()

        return dc


class GDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape  # (batch size,class_num,x,y,z)
        shp_y = gt.shape  # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        # copy from https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        w: torch.Tensor = (einsum("bcxyz->bc", y_onehot).type(torch.float32) + 1e-10) ** 2
        intersection: torch.Tensor = w * einsum("bcxyz, bcxyz->bc", net_output, y_onehot)
        union: torch.Tensor = w * (einsum("bcxyz->bc", net_output) + einsum("bcxyz->bc", y_onehot))
        divided: torch.Tensor = 1 - (2 * einsum("bc->b", intersection) + self.smooth) / (
                    einsum("bc->b", union) + self.smooth)
        gdc = divided.mean()

        return gdc

class DiceLoss_normal(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(DiceLoss_normal, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape  # (batch size,class_num,x,y,z)
        shp_y = gt.shape  # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        intersection: torch.Tensor = einsum("bcxyz, bcxyz->bc", net_output, y_onehot)
        union: torch.Tensor = (einsum("bcxyz->bc", net_output) + einsum("bcxyz->bc", y_onehot))
        divided: torch.Tensor = 1 - (2 * einsum("bc->b", intersection) + self.smooth) / (
                    einsum("bc->b", union) + self.smooth)
        gdc = divided.mean()

        return gdc
