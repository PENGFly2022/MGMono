



##############################################################################################################
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_
import torch.nn.functional as F



class NWCHead(nn.Module):
    def __init__(self, in_channels, k=3):
        """
        NWC 预测头

        参数:
            in_channels (int): 输入特征图的通道数
            k (int): 卷积核大小，默认 3
        """
        super(NWCHead, self).__init__()
        self.k = k
        self.k_sq = k * k

        # 深度向量预测卷积层
        self.conv_depth = nn.Conv2d(in_channels, self.k_sq, kernel_size=k, padding=k // 2)

        # 置信度向量预测卷积层
        self.conv_confidence = nn.Conv2d(in_channels, self.k_sq, kernel_size=k, padding=k // 2)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv_depth.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv_depth.bias, 0)
        nn.init.kaiming_normal_(self.conv_confidence.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv_confidence.bias, 0)

    def forward(self, F_up):
        """
        前向传播

        参数:
            F_up (torch.Tensor): 上采样后的特征图，形状为 [batch_size, in_channels, height, width]

        返回:
            D (torch.Tensor): 最终深度估计，形状为 [batch_size, 1, height, width]
        """
        batch_size, _, height, width = F_up.size()

        # 预测深度向量 V
        V = self.conv_depth(F_up)  # [batch_size, k_sq, height, width]
        V = torch.sigmoid(V)  # 使用 Sigmoid 激活函数

        # 预测置信度向量 P
        P = self.conv_confidence(F_up)  # [batch_size, k_sq, height, width]
        P = F.softmax(P, dim=1)  # 使用 Softmax 激活函数，使得 P 在 k_sq 维度上和为 1

        # 计算最终深度 D
        # Reshape V 和 P 为 [batch_size, k_sq, height * width]
        V = V.view(batch_size, self.k_sq, -1)  # [batch_size, k_sq, height * width]
        P = P.view(batch_size, self.k_sq, -1)  # [batch_size, k_sq, height * width]

        # 逐元素相乘并在 k_sq 维度上求和
        D = torch.sum(V * P, dim=1, keepdim=True)  # [batch_size, 1, height * width]
        D = D.view(batch_size, 1, height, width)  # [batch_size, 1, height, width]

        return D


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
def upsample(x, scale_factor=2, mode='bilinear', align_corners=True):
    return torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3x3, self).__init__()
        # 确保 in_channels 和 out_channels 是整数
        assert isinstance(in_channels, int), f"in_channels 应为 int，但得到 {type(in_channels)}"
        assert isinstance(out_channels, int), f"out_channels 应为 int，但得到 {type(out_channels)}"

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

def upsample(x, scale_factor=2, mode='bilinear', align_corners=True):
    return torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

class DepthDecoder(nn.Module):
    def __init__(self, num_output_channels=1, use_skips=True,k=3):
        """
        参数:
            num_ch_enc (list): 编码器每个阶段的通道数，例如 [48, 48, 96, 192]
            num_output_channels (int): 输出深度图的通道数
            use_skips (bool): 使用跳跃连接
        """
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips

        # 硬编码 num_ch_dec
        self.num_ch_dec = [24, 24, 48, 96]  # 对应 [48, 48, 96, 192] / 2

        # 定义卷积块，用于处理跳跃连接后的特征
        self.upconv0 = ConvBlock(192, 96)  # [192 -> 96]
        self.upconv1 = ConvBlock(96 + 96, 48)  # [96 + 96 -> 48]
        self.upconv2 = ConvBlock(48 + 48, 24)  # [48 + 48 -> 24]
        self.upconv3 = ConvBlock(24 + 48, 24)  # [24 + 48 -> 24]

        # 定义用于生成深度图的卷积层
        self.dispconv1 = Conv3x3(24, self.num_output_channels)  # A1
        self.dispconv2 = Conv3x3(24, self.num_output_channels)  # A2
        self.dispconv3 = Conv3x3(48, self.num_output_channels)  # A3
        self.nwc_head1 = NWCHead(self.num_ch_dec[0], k=k)  # 对应 A1
        self.nwc_head2 = NWCHead(self.num_ch_dec[1], k=k)  # 对应 A2
        self.nwc_head3 = NWCHead(self.num_ch_dec[2], k=k)  # 对应 A3


        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        """
        参数:
            input_features (list): 编码器输出的特征列表，长度为4，顺序从高分辨率到低分辨率
                features[0]: [8, 48, 96, 320]
                features[1]: [8, 48, 48, 160]
                features[2]: [8, 96, 24, 80]
                features[3]: [8, 192, 12, 40]
        返回:
            outputs (dict): 包含多尺度深度图的字典
                ("disp", 0): A1 深度图 [8, 1, 192, 640]
                ("disp", 1): A2 深度图 [8, 1, 96, 320]
                ("disp", 2): A3 深度图 [8, 1, 48, 160]
        """
        # 检查输入特征数量
        if len(input_features) != 4:
            raise ValueError(f"DepthDecoder expects exactly 4 input features, but got {len(input_features)}")

        self.outputs = {}

        # 第一步：对最小特征图 (features[3]) 进行上采样并通过 upconv0
        x = input_features[3]        # [8, 192, 12, 40]
        x = self.upconv0(x)           # [8, 96, 12, 40]
        x = upsample(x)               # [8, 96, 24, 80]
        # print(f"After upconv0 and upsample: {x.shape}")  # 调试信息

        # 与 features[2] 进行跳跃连接
        if self.use_skips:
            x = torch.cat([x, input_features[2]], dim=1)  # [8, 96 + 96, 24, 80] = [8, 192, 24, 80]
        x = self.upconv1(x)           # [8, 48, 24, 80]
        x = upsample(x)               # [8, 48, 48, 160]
        # print(f"After upconv1 and upsample (A3): {x.shape}")  # 调试信息
        A3 = x.clone()               # [8, 48, 48, 160]

        # 与 features[1] 进行跳跃连接
        if self.use_skips:
            x = torch.cat([x, input_features[1]], dim=1)  # [8, 48 + 48, 48, 160] = [8, 96, 48, 160]
        x = self.upconv2(x)           # [8, 24, 48, 160]
        x = upsample(x)               # [8, 24, 96, 320]
        # print(f"After upconv2 and upsample (A2): {x.shape}")  # 调试信息
        A2 = x.clone()               # [8, 24, 96, 320]

        # 与 features[0] 进行跳跃连接
        if self.use_skips:
            x = torch.cat([x, input_features[0]], dim=1)  # [8, 24 + 48, 96, 320] = [8, 72, 96, 320]
        x = self.upconv3(x)           # [8, 24, 96, 320]
        x = upsample(x)               # [8, 24, 192, 640]
        # print(f"After upconv3 and upsample (A1): {x.shape}")  # 调试信息
        A1 = x.clone()               # [8, 24, 192, 640]

        # 生成深度图
        disp1 = self.nwc_head1(A1)
        disp2 = self.nwc_head2(A2)
        disp3 = self.nwc_head3(A3)
        # disp1 = self.sigmoid(self.dispconv1(A1))  # [8, 1, 192, 640]
        # disp2 = self.sigmoid(self.dispconv2(A2))  # [8, 1, 96, 320]
        # disp3 = self.sigmoid(self.dispconv3(A3))  # [8, 1, 48, 160]

        # 将多个尺度的深度图加入输出
        self.outputs[("disp", 0)] = disp1
        self.outputs[("disp", 1)] = disp2
        self.outputs[("disp", 2)] = disp3

        return self.outputs



