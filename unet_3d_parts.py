""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()


        self.up = nn.ConvTranspose3d(in_channels , out_channels, kernel_size=(1,2,2), stride=(1,2,2))

        self.conv = DoubleConv(out_channels*2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x2, x1), dim=1)
        output = self.conv(x)
        return output


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),
            SingleConv(in_channels, in_channels),
            SingleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class SingleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)



class Down_drop(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv_drop = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),
            SingleConvWithDrop(in_channels, in_channels),
            SingleConvWithDrop(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv_drop(x)


class SingleConvWithDrop(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv_drop = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.single_conv_drop(x)



class FinalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
