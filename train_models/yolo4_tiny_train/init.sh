#!/bin/bash

# YOLOv4-tiny-Darknet-Roboflow.ipynb
# https://github.com/tzutalin/labelImg
# https://www.makesense.ai/
# https://youtu.be/33XDY0cb86Q?t=311
# http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

/usr/local/cuda/bin/nvcc --version
nvidia-smi
# env compute_capability=61 # 1080 ti
env compute_capability=75 # 1660
rm -rf darknet
git clone https://github.com/roboflow-ai/darknet.git
cd darknet/
sed -i 's/OPENCV=0/OPENCV=1/g' Makefile
sed -i 's/GPU=0/GPU=1/g' Makefile
sed -i 's/CUDNN=0/CUDNN=1/g' Makefile
sed -i "s/ARCH= -gencode arch=compute_60,code=sm_60/ARCH= -gencode arch=compute_${compute_capability},code=sm_${compute_capability}/g" Makefile
sudo apt update
sudo apt-get install libopencv-dev
make
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29

