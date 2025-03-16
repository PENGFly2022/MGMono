# import sys
# sys.path.append('/home/nenu/a25/lsh/seg-mono/hr-mono/segment-anything-2-main')  # 添加根目录路径

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 加载模型权重和配置
checkpoint = "/home/nenu/下载/sam2_hiera_tiny.pt"  # 模型权重路径
model_cfg = "./sam2_hiera_t.yaml"  # 配置文件路径
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# 加载图像
image_path = "/home/nenu/a25/lsh/z-litti/2011_09_28/2011_09_28_drive_0002_sync/image_03/data/0000000000.png"
image = Image.open(image_path)

# 定义图像处理的转换流程
transform = transforms.Compose([
    transforms.Resize(512),  # 将短边调整为512
    transforms.CenterCrop((512, 512)),  # 中心裁剪为512x512
    transforms.ToTensor()  # 转换为 PyTorch Tensor 格式
])

# 应用转换到图像
image_tensor = transform(image)
image_numpy = image_tensor.permute(1, 2, 0).cpu().numpy()  # 转换为 (H, W, C) 格式

# 设置图像进行推理
predictor.set_image(image_numpy)

# 设置提示输入
input_points = np.array([[100, 150]])  # 像素坐标
input_labels = np.array([1])  # 1表示前景，0表示背景

# 进行推理
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False  # 设置为 False 以输出单个掩码
    )

# 输出掩码的形状
print("Masks shape:", masks.shape)

# 可视化原始图像和分割掩码
plt.imshow(image_numpy)  # 显示原始图像
plt.axis('off')
plt.title("Original Image")
plt.show()

plt.imshow(masks[0], cmap='gray')  # 显示第一个分割掩码
plt.axis('off')
plt.title("Segmentation Mask")
plt.show()
