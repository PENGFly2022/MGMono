
# MG-Mono: A Lightweight Multi-Granularity Method for Self-Supervised Monocular Depth Estimation

## Table of Contents
- [Overview](#overview)
- [Video Result](#video-result)
- [KITTI Results](#kitti-results)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Install Dependencies](#install-dependencies)
  - [Download MGMono Model Weights](#download-mgmono-model-weights)
- [Test](#test)
- [Evaluation](#evaluation)
- [Training](#training)
  - [Dependency Installation](#dependency-installation)
  - [Preparing Pre-trained Weights](#preparing-pre-trained-weights)
  - [Start Training](#start-training)



## Video result



https://github.com/user-attachments/assets/4f08bb9f-8356-4737-b00b-1a2dd1d34910




## Overview
![MG-Mono](https://github.com/user-attachments/assets/326462d8-922f-4071-9eba-7645b2f75c35)
![image](https://github.com/user-attachments/assets/f9cfeb15-c767-40e0-a217-b22ca5ae16db)
![image](https://github.com/user-attachments/assets/9e4f2ddb-3c68-45e5-8574-7e7d60e0852a)


## KITTI Results
![Comparative Visualization](https://github.com/user-attachments/assets/f29021ec-fe0a-40f5-843f-b5f088d4a1ab)
![Comparison Results](https://github.com/user-attachments/assets/55517697-affb-4b78-90ec-7308a29dcd03)
![Comparison](https://github.com/user-attachments/assets/40581484-a963-4e08-8b6a-ed2da09ee65a)

## Quick Start


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
    
