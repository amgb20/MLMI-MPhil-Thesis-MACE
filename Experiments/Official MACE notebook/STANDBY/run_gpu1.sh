#!/bin/bash

# Set GPU 1 as the only visible device
export CUDA_VISIBLE_DEVICES=1

echo "🔧 Using GPU 1 (CUDA_VISIBLE_DEVICES=1)"
echo "🔧 Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myvenv

# Run the training script
python "Experiments/Official MACE notebook/test_train.py" 