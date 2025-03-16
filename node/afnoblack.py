import torch
from torch import nn
import numpy as np
import torch

from timm.layers import to_2tuple
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math

from torch import nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1,
                 hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        # 下面四个量是用来声明可学习的权重，2代表实部和虚部，w * x + b的方式来线性组合傅里叶模态下的值
        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x, spatial_size=None):
        # B, N, C = x.shape

        # x = x.reshape(B, C, H, W)
        bias = x
        dtype = x.dtype
        x = x.float()
        # if spatial_size == None:
        #     H = W = int(math.sqrt(N))
        # else:
        #     H, W = spatial_size

        B, C, H, W = x.shape
        N = H * W
        # 从这开始，输入的x变成了[batchsize, height, width, channel]的数据维度
        x = x.reshape(B, H, W, C)

        # x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")  # 对height和width做2维的FFT变换
        # print(x.size(),"rfft2后形状")
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)
        # 0初始化线性组合后的结果，后面可以看出0初始化只需要部分赋值即可实现截断
        o1_real = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor],
                              device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor],
                              device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)
        # 由于是0初始化，所以第三个维度的0:kept_modes被赋了值，kept_modes:-1就被截断了
        o1_real[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :, :kept_modes] = (
                torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) - \
                torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) + \
                self.b2[0]
        )

        o2_imag[:, :, :kept_modes] = (
                torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) + \
                torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) + \
                self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)  # softshrink是一种激活函数
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")  # 傅里叶逆变换，恢复回原来的域
        # x = x.reshape(B, N, C)
        x = x.reshape(B, C, H, W)
        x = x.type(dtype)
        return x + bias  # shortcut
class Black(nn.Module):
    def __init__(self, hidden_size, num_blocks, num_afno_layers, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1,dim=48, mlp_ratio=4., drop=0., drop_path=0.,):
        super().__init__()
        self.num_afno_layers = num_afno_layers
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)


        # 创建一个nn.ModuleList来存储AFNO块
        self.afno_layers = nn.ModuleList([
            AFNO2D(hidden_size, num_blocks, sparsity_threshold, hard_thresholding_fraction, hidden_size_factor)
            for _ in range(num_afno_layers)
        ])

    def forward(self, x):
        # 依次通过每个AFNO块
        self.norm1 = nn.LayerNorm(x.shape[1])
        for layer in self.afno_layers:
            x = layer(x)+x
        B, C, H, W = x.shape
        x = x.reshape(B, H * W, C)
        x = self.norm2(x)
        x = self.mlp(x)
        x = x.reshape(B,C,H,W)
        return x
#
# if __name__ == '__main__':
#
#     x = torch.randn(2, 48, 16, 16)
#
#     model = Black(hidden_size=48, num_blocks=16, num_afno_layers=16)
#
#     y = model(x)
#
#     print(y.shape)




    #
    #
    #
    # model = AFNO2D(hidden_size=48, num_blocks=16, sparsity_threshold=0.01,
    #                                  hard_thresholding_fraction=1, hidden_size_factor=1)
    #
    # y = model(x)
    #
    # print(y.shape)
    #
    #
