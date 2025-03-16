import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from sam2.modeling.backbones.hieradet import Hiera

from hrseg.hrseg_lib.config import config
from hrseg.hrseg_lib.config import update_config
import torch.nn.functional as F
import torchvision.transforms as transforms
def create_segment_anything_model():
    # 假设 Hieradet 模型已经被定义在 `segment anything 2` 中
    model = Hiera()  # 替换为 Hieradet 的实例
    pretrained_dict = torch.load('/home/nenu/下载/sam2_hiera_tiny.pt')  # 加载预训练模型
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    for param in model.parameters():
        param.requires_grad = False
    print('Hieradet Model Loaded')
    return model

if __name__ == '__main__':
    from PIL import Image

    net = create_segment_anything_model().cuda()
    image_path = "/home/nenu/a25/lsh/seg-mono/hr-mono/segment-anything-main/0000000000.png"
    image = Image.open(image_path)
    # 定义图像处理的转换流程
    transform = transforms.Compose([
        # 缩放图像的短边为1024，保持长宽比
        transforms.Resize(512),
        # 中心裁剪为 1024x1024
        transforms.CenterCrop((512, 512)),
        # 将图像转换为 PyTorch Tensor
        transforms.ToTensor()
    ])
    # 应用转换到图像
    image_tensor = transform(image)
    # 增加批次维度以适应模型输入
    image_tensor = image_tensor.unsqueeze(0).cuda()
    seg_feature = net(image_tensor)
    # print(seg_map.shape, "-------map--------")
    # print(seg_map[0][0], "-------map--------")
    print(seg_feature[0].shape, "-------feature--------")
    print(seg_feature[1].shape, "-------feature--------")
    print(seg_feature[2].shape, "-------feature--------")
    print(seg_feature[3].shape, "-------feature--------")
    plt.figure(figsize=(16, 16))
    for i in range(3):
        plt.subplot(1, 3, i + 1)

        plt.imshow(seg_feature[0][0, i, :, :].cpu().detach().numpy())

        # plt.imshow(low_out[0, i, :, :].cpu().numpy())
        plt.axis('off')
        plt.title('input_low')
    plt.show()


