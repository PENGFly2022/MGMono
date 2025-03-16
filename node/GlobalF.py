import torch
from torch import nn

import torch.cuda

import torch.fft

class GlobalFilter(nn.Module):
    def __init__(self, dim, h=4, w=4):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w//2+1, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B,H,W,C = x.shape
#         if spatial_size is None:
#             a = b = int(math.sqrt(N))
#         else:
#             a, b = spatial_size

#         x = x.view(B, a, b, C)
        # print(x.size(),"input")
        x = x.to(torch.float32)
        # print(x.size(),"x.size-gf")
        # print(self.complex_weight.shape)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        # print(x.size(),"x.size-gf")
        # weight = torch.view_as_complex(self.complex_weight).permute(2,0,1)
        weight = torch.view_as_complex(self.complex_weight)
        # print(weight.shape)
        # print(x.shape,"x.shape")
        x = x * weight
        x = torch.fft.irfft2(x, s=(H,W), dim=(1, 2), norm='ortho')

        return x

