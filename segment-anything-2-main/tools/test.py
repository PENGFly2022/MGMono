import torch
from sam2.build_sam import build_sam2
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
from PIL import Image
import numpy as np
# 加载模型权重和配置
checkpoint = "/home/nenu/下载/sam2_hiera_tiny.pt"  #
model_cfg = "./sam2_hiera_t.yaml"  #
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
# 加载图像
image_path = "/home/nenu/a25/lsh/seg-mono/hr-mono/segment-anything-main/0000000001.png"  # 图片路径
image = Image.open(image_path)
image = np.array(image)  # 转换为 numpy 数组格式
print(image.shape)
# 设置图像进行推理
predictor.set_image(image)
# 设置提示输入
input_points = np.array([[100, 150]])  # 像素坐标
input_labels = np.array([1])  # 1表示前景，0表示背景
# 进行推理
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False  # 设置为 True 可以输出多个掩码
    )
# 输出掩码的形状
print("Masks shape:", masks.shape)
import matplotlib.pyplot as plt
# 显示原始图像
plt.imshow(image)
plt.axis('off')
plt.title("Original Image")
plt.show()
# 显示第一个掩码
plt.imshow(masks[0], cmap='gray')  # 显示第一个分割掩码
plt.axis('off')
plt.title("Segmentation Mask")
plt.show()
