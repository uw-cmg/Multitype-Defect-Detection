#!/usr/bin/env bash

# Ron the short-list GPU queue
#SBATCH -p slurm_courtesy

## Request one CPU core from the scheduler
#SBATCH -c 1

## Request a GPU from the scheduler, we don't care what kind
#SBATCH --gres=gpu:1
#SBATCH -t 10-2:00 # time (D-HH:MM)

## Create a unique output file for the job
#SBATCH -o cuda_Training-%j.log

## Load CUDA into your environment
module load cuda/9.0

## Run the installer
#pip uninstall chainercv
pip uninstall cupy
#pip install cupy
pip install cupy-cuda90
#pip install -U numpy
#pip install chainercv

# conda for anaconda to use cv2
# conda install -c conda-forge opencv
#conda install -c conda-forge opencv=2.4
pip install opencv-python
## Run Training Code
#python sample_train.py
#python train_faster_rCNN.py
#python caffe2npz.py [resnet50|resnet101|resnet152] <source>.caffemodel <target>.npz
#python caffe2npz.py resnet50 ResNet-50-model.caffemodel  ResNet-50-model.npz
#python caffe2npz.py resnet101 ResNet-101-model.caffemodel  ResNet-101-model.npz
#python caffe2npz.py resnet152 ResNet-152-model.caffemodel  ResNet-152-model.npz
python train_resnet50.py --model resnet50 
