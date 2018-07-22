#!/usr/bin/env bash

# Ron the short-list GPU queue
#SBATCH -p slurm_courtesy

## Request one CPU core from the scheduler
#SBATCH -c 1

## Request a GPU from the scheduler, we don't care what kind
#SBATCH --gres=gpu:1
#SBATCH -t 1-2:00 # time (D-HH:MM)

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
conda install -c conda-forge opencv

## Run Training Code

python3 sample_train.py 
