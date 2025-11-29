#!/bin/bash

# 修改这里选择GPU
GPU_ID=7

echo "Using GPU: $GPU_ID"

CUDA_VISIBLE_DEVICES=$GPU_ID python -u compute_pca_rotation.py
