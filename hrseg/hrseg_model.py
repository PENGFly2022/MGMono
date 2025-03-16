import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from hrseg.hrseg_lib.models import seg_hrnet
from hrseg.hrseg_lib.config import config
from hrseg.hrseg_lib.config import update_config
import torch.nn.functional as F
import torchvision.transforms as transforms

def create_hrnet():
    args = {}
    args['cfg'] = '/home/nenu/a25/lsh/seg-mono/hr-mono/hrseg/hrseg_lib/pascal_ctx/seg_hrnet_w48_cls59_480x480_sgd_lr4e-3_wd1e-4_bs_16_epoch200.yaml'
    args['opt'] = []
    update_config(config, args)
    if torch.__version__.startswith('1'):
        module = eval('seg_hrnet')
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval(config.MODEL.NAME + '.get_seg_model')(config)
    pretrained_dict = torch.load('/home/nenu/a25/lsh/seg-mono/hr-mono/hrseg/hrnet_w48_pascal_context_cls59_480x480.pth')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    for param in model.parameters():
        param.requires_grad = False
    print('HRNet load')
    return model
# model = model.cuda()
def padtensor(input_):
    mul = 16
    h, w = input_.shape[2], input_.shape[3]
    H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
    padh = H - h if h % mul != 0 else 0
    padw = W - w if w % mul != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
    return input_
def re_size(input_, size):
    return F.interpolate(input_, size=size, mode='bilinear', align_corners=True)
if __name__ == '__main__':
    from PIL import Image
    net = create_hrnet().cuda()
    image_path = "/home/nenu/a25/lsh/seg-mono/hr-mono/segment-anything-main/0000000000.png"
    image = Image.open(image_path)
    # 定义图像处理的转换流程
    transform = transforms.Compose([
        # 缩放图像的短边为1024，保持长宽比
        transforms.Resize(1024),
        # 中心裁剪为 1024x1024
        transforms.CenterCrop((1024, 1024)),
        # 将图像转换为 PyTorch Tensor
        transforms.ToTensor()
    ])
    # 应用转换到图像
    image_tensor = transform(image)
    # 增加批次维度以适应模型输入
    image_tensor = image_tensor.unsqueeze(0).cuda()
    seg_map, seg_feature = net(image_tensor)
    print(seg_map.shape, "-------map--------")
    print(seg_map[0][0], "-------map--------")
    print(seg_feature[0].shape, "-------feature--------")
    print(seg_feature[1].shape, "-------feature--------")
    print(seg_feature[2].shape, "-------feature--------")
    print(seg_feature[3].shape, "-------feature--------")
    plt.figure(figsize=(16, 16))
    for i in range(1):
        plt.subplot(1, 3, i + 1)

        plt.imshow(seg_feature[3][0, i, :, :].cpu().detach().numpy())

        # plt.imshow(low_out[0, i, :, :].cpu().numpy())
        plt.axis('off')
        plt.title('input_low')
    plt.show()


