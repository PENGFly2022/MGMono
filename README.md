
# MG-Mono: A Lightweight Multi-Granularity Method for Self-Supervised Monocular Depth Estimation
## Video result



https://github.com/user-attachments/assets/4f08bb9f-8356-4737-b00b-1a2dd1d34910




## Overview
![MG-Mono](https://github.com/user-attachments/assets/326462d8-922f-4071-9eba-7645b2f75c35)
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/user-attachments/assets/efd0939c-682c-4f91-ad25-c5af758ea5d7" width="32%" />
  <img src="https://github.com/user-attachments/assets/2ec3dc6a-323e-4778-bf6c-3db8719074bf" width="32%" />
  <img src="https://github.com/user-attachments/assets/bc644336-6b81-48cb-8da3-9d29fa388b47" width="32%" />
</div>

## KITTI Results
![Comparative Visualization](https://github.com/user-attachments/assets/f29021ec-fe0a-40f5-843f-b5f088d4a1ab)
![Comparison Results](https://github.com/user-attachments/assets/55517697-affb-4b78-90ec-7308a29dcd03)

## Quick Start

### Prerequisites

  conda create -n mg_mono python=3.8
  conda activate mg_mono

### Install Dependencies
  pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
  pip install -r requirements.txt
### Download MGMono Model Weights
320 × 1024
The files shared via Baidu Netdisk: encoder.pth and depth.pth
Download link: ([https://pan.baidu.com/s/1bZVLKNh6VKu5vUl38BWcnA?pwd=49nk]) key: 49nk 

192 × 640
The files shared via Baidu Netdisk: encoder.pth and depth.pth
Download link: ([https://pan.baidu.com/s/12vWnzY9j6D2VzDO0NlBhzA?pwd=sivq]) key: sivq 
## Test
  python test_simple.py --load_weights_folder path/to/your/weights/folder --image_path path/to/your/test/image
## Evaluation
  python evaluate_depth.py --load_weights_folder path/to/your/weights/folder --data_path path/to/kitti_data/ --model MG-Mono
## Training
### dependency installation
  pip install 'git+https://github.com/saadnaeem-dev/pytorch-linear-warmup-cosine-annealing-warm-restarts-weight-decay'
### preparing pre-trained weights
320 × 1024
Download link: Stay tuned for more updates!

192 × 640
Download link:([https://pan.baidu.com/s/1ZzfYVaQt5Kl_0kb4DcXxrg?pwd=kd2j]) key: kd2j 
### start training
    python train.py --data_path path/to/your/data --model_name mytrain --num_epochs 30 --batch_size 12 --mypretrain path/to/your/pretrained/weights  --lr 0.0001 5e-6 31 0.0001 1e-5 31
    
