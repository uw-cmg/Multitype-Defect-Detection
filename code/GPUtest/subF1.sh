#!/usr/bin/env bash

# Ron the short-list GPU queue
#SBATCH -p sbel_cmg
#SBATCH --account=cmg --qos=cmg_owner

## Request one CPU core from the scheduler
#SBATCH -c 1

## Request a GPU from the scheduler, we don't care what kind
#SBATCH --gres=gpu:1
#SBATCH -t 2-20:10 # time (D-HH:MM)

## Request a GPU from the scheduler, we don't care what kind
#SBATCH -t 14-3:00 # time (D-HH:MM)

## Create a unique output file for the job
#SBATCH -o cuda_Training-%j.log

## Load CUDA into your environment
## load custimized CUDA and cudaToolkit

module load usermods
module load user/cuda
#module load cuda/9.0

# activate retina virtual environment
source activate chainercv
#conda install -c anaconda --name retina cudatoolkit==9.0 --yes
#conda install -c anaconda --name retina cudnn --yes
#conda install -c anaconda --name retina keras --yes

# install tensorflow and other libraries for machine learning

#/srv/home/shenmr/anaconda3/envs/retina/bin/pip install tensorflow-gpu==1.6
#/srv/home/shenmr/anaconda3/envs/retina/bin/pip install numpy scipy scikit-learn pandas matplotlib seaborn
#
#/srv/home/shenmr/anaconda3/envs/retina/bin/pip msgpack 
#conda install -c anaconda --name retina keras-gpu
##/srv/home/shenmr/anaconda3/envs/retina/bin/pip install keras 
#

#/srv/home/shenmr/anaconda3/envs/chainercv/bin/pip install scikit-image 
#/srv/home/shenmr/anaconda3/envs/chainercv/bin/pip install cupy-cuda90 
#/srv/home/shenmr/anaconda3/envs/chainercv/bin/pip install opencv-python
#/srv/home/shenmr/anaconda3/envs/chainercv/bin/pip install Pillow

# run the training scripts
python ErrorIoU_Batch_GPU.py 
