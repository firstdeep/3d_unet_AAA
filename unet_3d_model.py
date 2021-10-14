""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from unet_3d_parts import *
import torch.nn as nn

# DeepAAA
# https://dl.acm.org/doi/10.1007/978-3-030-32245-8_80
# pooling (2 * 2 * 1(depth))
# Batch normalization
# dropout regularization in bottleneck module (rate 0.2)
#

class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels

        self.dropout = nn.Dropout(0.2)

        self.down1 = DoubleConv(n_channels, out_channels=64, mid_channels=32)

        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down_drop(256, 512)

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.final = FinalConv(64, n_classes)



    def forward(self, x):
        x1 = self.down1(x)
        # print(x1.shape)
        x2 = self.down2(x1)
        # print(x2.shape)
        x3 = self.down3(x2)
        # print(x3.shape)
        x4 = self.down4(x3)
        # print(x4.shape)

        x = self.up1(x4, x3)
        # print(x.shape)
        x  = self.up2(x, x2)
        # print(x.shape)
        x = self.up3(x, x1)
        # print(x.shape)

        logits = self.final(x)
        # print(logits.shape)

        return logits

