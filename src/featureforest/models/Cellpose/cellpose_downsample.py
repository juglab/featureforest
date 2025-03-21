"""
Copyright © 2023 Howard Hughes Medical Institute,
Authored by Carsen Stringer and Marius Pachitariu.
"""

import torch.nn as nn


def batchconv(in_channels, out_channels, sz, conv_3D=False):
    conv_layer = nn.Conv3d if conv_3D else nn.Conv2d
    batch_norm = nn.BatchNorm3d if conv_3D else nn.BatchNorm2d
    return nn.Sequential(
        batch_norm(in_channels, eps=1e-5, momentum=0.05),
        nn.ReLU(inplace=True),
        conv_layer(in_channels, out_channels, sz, padding=sz // 2),
    )


def batchconv0(in_channels, out_channels, sz, conv_3D=False):
    conv_layer = nn.Conv3d if conv_3D else nn.Conv2d
    batch_norm = nn.BatchNorm3d if conv_3D else nn.BatchNorm2d
    return nn.Sequential(
        batch_norm(in_channels, eps=1e-5, momentum=0.05),
        conv_layer(in_channels, out_channels, sz, padding=sz // 2),
    )


class resdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz, conv_3D=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.proj = batchconv0(in_channels, out_channels, 1, conv_3D)
        for t in range(4):
            if t == 0:
                self.conv.add_module(
                    f"conv_{t}", batchconv(in_channels, out_channels, sz, conv_3D)
                )
            else:
                self.conv.add_module(
                    f"conv_{t}", batchconv(out_channels, out_channels, sz, conv_3D)
                )

    def forward(self, x):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x


class downsample(nn.Module):
    def __init__(self, nbase, sz, conv_3D=False, max_pool=True):
        super().__init__()
        self.down = nn.Sequential()
        if max_pool:
            self.maxpool = (
                nn.MaxPool3d(2, stride=2) if conv_3D else nn.MaxPool2d(2, stride=2)
            )
        else:
            self.maxpool = (
                nn.AvgPool3d(2, stride=2) if conv_3D else nn.AvgPool2d(2, stride=2)
            )
        for n in range(len(nbase) - 1):
            self.down.add_module(
                f"res_down_{n}", resdown(nbase[n], nbase[n + 1], sz, conv_3D)
            )

    def forward(self, x):
        xd = []
        for n in range(len(self.down)):
            if n > 0:  # noqa: SIM108
                y = self.maxpool(xd[n - 1])
            else:
                y = x
            xd.append(self.down[n](y))
        return xd
