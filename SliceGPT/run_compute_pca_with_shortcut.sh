#!/bin/bash

# 计算 PCA 旋转矩阵并保存 shortcut Q 矩阵
# 实现与 SliceGPT 完全相同的残差连接处理逻辑

# 配置参数
GPU_ID=1

# 运行计算
CUDA_VISIBLE_DEVICES=$GPU_ID python -u compute_pca_with_shortcut.py
