#!/bin/bash

# 修改这里选择GPU
GPU_ID=1

echo "Using GPU: $GPU_ID"

CUDA_VISIBLE_DEVICES=$GPU_ID python compute_pca_rotation.py
