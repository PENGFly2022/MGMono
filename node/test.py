import torch
from torch import nn


class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8, H=96, W=320, patch_size=4):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h
        self.H = H
        self.W = W
        self.patch_size = patch_size

    def forward(self, x, a=None, b=None):
        B, N, C = x.shape
        print(x.shape, '输入的BNC')
        # if spatial_size is None:
        #     a = b = int(math.sqrt(N))
        # else:
        #     a, b = spatial_size
        a = int(self.H / self.patch_size)
        b = int(self.W / self.patch_size)

        x = x.view(B, a, b, C)
        print(x.shape)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        print('x', x.shape)
        weight = torch.view_as_complex(self.complex_weight)
        print(weight.shape, 'w')
        x = x * weight

        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)
        print(x.shape, '输出的BNC')

        return x

