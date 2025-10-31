import torch
import pytorch_lightning as pl
import torch.nn.functional as F

import pandas as pd
from pathlib import Path

class StandardBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        kernel_size=3,
        dilation=1,
        **kwargs,
    ):
        super().__init__()
        padding = kernel_size // 2
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
                dilation=dilation,
            ),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
                dilation=dilation,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ResBlock(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, mid_channels=None, kernel_size=3, sf=1
    ):
        super().__init__()
        self._scaling_factor = sf

        padding = kernel_size // 2
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            torch.nn.BatchNorm2d(out_channels),
        )
        if in_channels != out_channels:
            self.projection_conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                torch.nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.double_conv(x)

        if hasattr(self, "projection_conv"):
            x = self.projection_conv(x)

        out = out * self._scaling_factor + x

        return F.relu(out)


class Down(torch.nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, block, **kwargs):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2), block(in_channels, out_channels, **kwargs)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(torch.nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, block, bilinear=True, **kwargs):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = torch.nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = block(in_channels, out_channels, in_channels // 2, **kwargs)
        else:
            self.up = torch.nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = block(in_channels, out_channels, **kwargs)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.out = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.out(x)

class UNetSolver(torch.nn.Module):
    def __init__(
        self, n_channels=1, n_hidden=64, n_classes=1, bilinear=True, block=ResBlock, add_input=False
    ):
        super(UNetSolver, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.add_input = add_input
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # block-wise weight scaling factors for stabilised gradients
        sfs = 1 / torch.arange(1, 10).sqrt()

        # define modules
        self.inc = StandardBlock(n_channels, n_hidden)
        self.down1 = Down(n_hidden, n_hidden * 2, block, sf=sfs[1])
        self.down2 = Down(n_hidden * 2, n_hidden * 4, block, sf=sfs[2])
        self.down3 = Down(n_hidden * 4, n_hidden * 8, block, sf=sfs[3])
        self.down4 = Down(n_hidden * 8, n_hidden * 16 // factor, block, sf=sfs[4])

        self.up1 = Up(n_hidden * 16, n_hidden * 8 // factor, block, bilinear, sf=sfs[5])
        self.up2 = Up(n_hidden * 8, n_hidden * 4 // factor, block, bilinear, sf=sfs[6])
        self.up3 = Up(n_hidden * 4, n_hidden * 2 // factor, block, bilinear, sf=sfs[7])
        self.up4 = Up(n_hidden * 2, n_hidden, block, bilinear, sf=sfs[8])
        self.outc = OutConv(n_hidden, n_classes)

    def forward(self, batch):
        x = batch.input.nan_to_num()
        if self.add_input:
            inp = x[:, -1].unsqueeze(1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        out = self.outc(x)
        if self.add_input:
            out += inp

        return out