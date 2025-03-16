from timm.layers import to_2tuple
from torch import nn


# self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
class PatchEmbed(nn.Module):
    def __init__(self, H=48, W=160, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.H = H
        self.W = W
        patch_size = to_2tuple(patch_size)
        num_patches = (H // patch_size[1]) * (W // patch_size[0])
        # 计算patch的数目

        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 通过2维卷积层将输入数据的维度映射到embed_dim，而kernel_size和stride_size都是patch_size
        # 这意味着patch在mixing时是借用了卷积来实现了这个操作，实现了无重叠的划分和mixing

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
