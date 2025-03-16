import torch
import torch.nn as nn

def _split_channels(num_chan, num_groups):		# 根据组数对输入通道分组
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split

class MixConv(nn.Module):
    def __init__(self, 						# 默认dilation均为1
                    in_channels,
                    kernel_size=[3,5,7],
                    padding=[1,2,3],
                    stride=1):
        super(MixConv, self).__init__()
        if padding is None:					# 默认为same padding模式
            padding = [(k-1)//2 for k in kernel_size]

        self.num_groups = len(kernel_size)
        self.in_splits = _split_channels(in_channels, self.num_groups)
        self.layer = nn.ModuleList([])			# 按照每个分组初始化卷积
        for c, k, p in zip(self.in_splits, kernel_size, padding):
            self.layer.append(
                nn.Conv2d(c, c, k, stride, p, groups=c, bias=False)
            )

    def forward(self, x):
        out = []
        x_split = torch.split(x, self.in_splits, dim=1)
        for m, _x in zip(self.layer, x_split):	               # 循环计算每个分组输出
            out.append(m(_x))

        return torch.cat(out, dim=1)			       # 按通道维度拼接在一起

#
# temp = torch.randn((16, 3, 32, 32))
# # group = GroupConv2D(3, 16, n_chunks=2)
# # print(group(temp).size())
#
#
# group = MixConv(3)
#
# print(group(temp).size())
