import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math
import torch.cuda
import sys
sys.path.append('/home/lsh/桌面/mg-mono/segment-anything-2-main')  # 添加根目录路径
# from hrseg.hrseg_model import create_hrnet
from sam2.modeling.c_modell import create_segment_anything_model
from node.dwconv import *
from node.CONV import *
from node.MLP import *
from node.AFNO import *
from node.patch import *
from node.pool import *

import torch.fft


# gf


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
        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x, spatial_size=None):

        bias = x
        dtype = x.dtype
        x = x.float()


        B, C, H, W = x.shape
        N = H * W
        x = x.reshape(B, H, W, C)

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")  # 对height和width做2维的FFT变换
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)
        o1_real = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor],
                              device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor],
                              device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)
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

class feature_extractor(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, patch_size=4,
                 H=48, W=160, embed_dim=768, dw_num=3, afno_num=3):
        super().__init__()
        """
        in_channels: input channel dimensionality
        out_channels: output channel dimensionality
        stride:
        patch_size: patch size
        H: input'H
        W: input'W
        embed_dim:
        """
        # 定义1D卷积层
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1)



        # 定义多个深度可分离卷积
        dw_block = []
        for i in range(dw_num):
            dw_block.append(DepthwiseSeparableConv(in_channels, out_channels))
        self.dw = nn.ModuleList(dw_block)

        # 定义多个AFNO
        afno_block = []
        for i in range(afno_num):
            afno_block.append(AFNO2D(hidden_size=in_channels, num_blocks=16, sparsity_threshold=0.01,
                                     hard_thresholding_fraction=1, hidden_size_factor=1))
        self.afno = nn.ModuleList(afno_block)

    def forward(self, x):

        x_dw = x
        x_afno = x

        B, C, H, W = x.shape
        # x_afno = self.patch_embed(x_afno)  # 将x进行patch的划分和embedding，将每个patch映射成embed_dim维的向量
        # x_afno = x_afno + self.pos_embed  # 切块后加位置编码 [B,N=12(48/4)*40(160/4),C]

        features = []
        features.append(self.conv1d(x))  # 1d output
        for dw in self.dw:
            x_dw = dw(x_dw)
        features.append(x_dw)  # dw output
        # print(x_dw.size(),"x_dw")

        for afno in self.afno:
            x_afno = afno(x_afno)  # 多个afnoo的输出

        # global_feature = x_afno.transpose(1, 2).view(B, C, H, W) # [B,N,E]=>[B,E,N]=>[B,C,H,W]
        features.append(x_afno)  # features[1d,dw,afno]
        # features = features[0] + features[1] + features[2]

        return features


class Select_channels(nn.Module):
    def __init__(self, in_cha, M=3, r=4, L=4):
        super(Select_channels, self).__init__()
        d = max(int(in_cha / r), L)
        self.M = M  # 分支数
        self.in_cha = in_cha  # 输入通道数
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(nn.Conv2d(in_cha, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))

        # self.fc = nn.Conv2d(in_channels, d, kernel_size=1, stride=1, bias=False)

        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, in_cha, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        feats = x
        B, C, H, W = feats[0].shape
        # print(len(feats),"len")

        feats = torch.cat(feats, dim=1)
        feats = feats.view(B, self.M, C, feats.shape[2], feats.shape[3])

        feats_U = torch.sum(feats, dim=1)
        # print(feats_U.size(),"sum")
        feats_S = self.gap(feats_U)
        # print(feats_S.size(),"gap")
        feats_Z = self.fc(feats_S)
        # print(feats_Z.size(),"fc")

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(B, self.M, self.in_cha, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(feats * attention_vectors, dim=1)
        # print(feats_V.size(),"output")

        return feats_V


class AvgPool(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)
        # print(x.size(),"平均池化")

        return x


import torch
import torch.nn as nn
import torch.optim as optim


# 定义SE模块
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # return x * y.expand_as(x)

        return y.expand_as(x)


# 定义空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在空间维度上生成注意力权重
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([max_pool, avg_pool], dim=1)
        x = self.conv1(x)
        # print(x.shape)
        return self.sigmoid(x)

# 定义混合注意力模块
class ChannelSpatialAttention(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(ChannelSpatialAttention, self).__init__()
        # 通道注意力
        # self.channel_attention = SEModule(channel, reduction)
        # 空间注意力
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 先应用通道注意力
        # x = self.channel_attention(x)
        # 再应用空间注意力
        x = self.spatial_attention(x)
        return x



class MGMono(nn.Module):
    """
    NG-Mono
    """

    def __init__(self, in_chans=3, model='MG-Mono', height=192, width=640, patch_size=4, drop_rate=0.,
                 global_block=[1, 1, 1], uniform_drop=False,
                 drop_path_rate=0.2, options = None,
                 heads=[8, 8, 8], use_pos_embd_xca=[True, False, False], **kwargs):

        super().__init__()
        self.models = {}
        self.opt = options
        # self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        self.models["seg"] = create_segment_anything_model().cuda()
        self.Spatial = SpatialAttention(kernel_size=7)
        self.conv = nn.Conv2d(in_channels=96, out_channels=48, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels=192, out_channels=96, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=384, out_channels=192, kernel_size=1)

        # self.channelSpatialAttention = ChannelSpatialAttention(channel=,kernel_size=7,reduction=16)
        # seg_map, seg_feature = self.models["seg"](inputs["color_aug", 0, 0])




        self.downsample_layers = nn.ModuleList()
        stem1 = nn.Sequential(
            Conv(in_chans, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
        )


        self.stem2 = nn.Sequential(
            Conv(self.dims[0] + 3, self.dims[0], kSize=3, stride=2, padding=1, bn_act=False),
        )

        self.downsample_layers.append(stem1)

        self.input_downsample = nn.ModuleList()
        for i in range(1, 5):
            self.input_downsample.append(AvgPool(i))

        for i in range(2):
            downsample_layer = nn.Sequential(
                Conv(self.dims[i] * 2 + 3, self.dims[i + 1], kSize=3, stride=2, padding=1, bn_act=False),
            )
            self.downsample_layers.append(downsample_layer)


        cur = 0
        if model == 'MG-Mono':
            in_cha = [48, 96, 192]
            h_size = [48, 24, 12]
            w_size = [160, 80, 40]
            dw_nums = [3, 5, 9]
            afno_nums = [3, 5, 9]

        elif model == 'MG-Mono-tiny':
            in_cha = [48, 96, 192]
            h_size = [48, 24, 12]
            w_size = [160, 80, 40]
            dw_nums = [2, 4, 6]
            afno_nums = [2, 3, 6]



        channelSpatialAtt = []
        for i in range(3):
            channelSpatialAtt.append(ChannelSpatialAttention(channel=self.dims[i], kernel_size=7, reduction=16))
        self.channelSpatialAttention = nn.ModuleList(channelSpatialAtt)

        # --------------------------初始化三个三分支特征提取器----------------

        stage_blocks = []
        for i in range(3):
            # stage_blocks.append(Three_brasnch(in_channels=in_cha[i], out_channels=self.dims[i], kernel_size=3, stride=1, dilation=1, expan_ratio=6
            #                            , h=h_size[i], w=w_size[i] ))
            stage_blocks.append(
                feature_extractor(in_channels=in_cha[i], out_channels=self.dims[i], stride=1,
                                  H=h_size[i], W=w_size[i], embed_dim=in_cha[i] * 16,
                                  dw_num=dw_nums[i],afno_num=afno_nums[i])

            )
        self.three_extractor = nn.ModuleList(stage_blocks)
        # ----------------------------------------------------------------
        # print(self.stages)

        #-----------------------------初始化三个sk--------------------------

        sk_blocks = []
        for i in range(3):
            sk_blocks.append(
                Select_channels(in_cha=in_cha[i],M=3, r=4, L=4)
            )
        self.sk_fusion = nn.ModuleList(sk_blocks)
        # -----------------------------------------------------------------

        self.apply(self._init_weights)
        self.CONV = nn.Conv2d(48, 24, kernel_size=3,padding=1)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):

        features = []
        seg_a = [None] * 3
        seg_feature = self.models["seg"](x)
        # print(self.conv(seg_feature[0]).shape)
        seg_a[0] =self.conv(seg_feature[0])
        seg_a[1] =self.conv1(seg_feature[1])
        seg_a[2] =self.conv2(seg_feature[2])


        x = (x - 0.45) / 0.225

        # print("一阶段")
        x_down = []


        for i in range(4):
            x_down.append(self.input_downsample[i](x))

        tmp_x = []
        x = self.downsample_layers[0](x)
        features.append(x)    #将第一阶段的特征加入features

        # x =

        x = self.stem2(torch.cat((x, x_down[0]), dim=1))


        tmp_x.append(x)

        x = self.three_extractor[0](x) # 输入先送入三分支
        x = self.sk_fusion[0](x) # 三分支的输出送入 sk

        attention_M = []
        tmp_x.append(x)
        features.append(x)    #将第一阶段的特征加入features

        # features.append(x)
        for i in range(0,3):
            # attention_map = self.Spatial(seg_feature[i])
            attention_map = self.channelSpatialAttention[i](seg_a[i])
            attention_M.append(attention_map*seg_a[i])
            # print(attention_M[i].shape)



        for i in range(1, 3):

            tmp_x.append(x_down[i])
            x = torch.cat(tmp_x, dim=1)
            x = self.downsample_layers[i](x)

            tmp_x = [x]
            x = self.three_extractor[i](x) # 输入先送入三分支
            x = self.sk_fusion[i](x) # 三分支输出送入sk


            tmp_x.append(x)

            features.append(x)
        # print(features[0].shape, 'features')
        # print(features[1].shape, 'features')
        # print(features[2].shape, 'features')
        # print(features[3].shape, 'features')


        return features, attention_M


        # return features

    def forward(self, x):
        x = self.forward_features(x)
        # print(len(x))

        return x
