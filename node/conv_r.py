import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        # 深度卷积步骤，使用3x3卷积核，不改变卷积后的输出尺寸
        # self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=in_channels)
        # 逐点卷积步骤，增加通道数

        # self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)
        # self.conv1 = self._conv_st(in_channels,out_channels,1)
        self.dsw = self._conv_dw(in_channels, out_channels, 1)
        self._conv_st = self._conv_x3(in_channels, out_channels, 1)

    def forward(self, x):
        # 先进行深度卷积
        out = x
        # x = self.depthwise(x)
        # 然后进行逐点卷积，增加通道数
        # x = self.pointwise(x)
        # 普通卷积，增强特征表示
        # x = self.conv1(x)
        # x = self.dsw(x)

        x = self._conv_st(x)

        return x

    def _conv_dw(self, in_channels, out_channels, stride):
        return nn.Sequential(
            # 深度卷积
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels,
                      bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            # 点卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    # 定义深度可分离卷积数目块（blocks为块数）
    def _conv_x3(self, in_channel, out_channel, blocks):
        layers = []
        for i in range(blocks):
            layers.append(self._conv_dw(in_channel, out_channel, 1))
        return nn.Sequential(*layers)

    # 定义普通卷积

    def _conv_st(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )