#!/bin/bash

# 模型困惑度评估脚本
# 使用与 SliceGPT 完全相同的方法评估模型

# 配置参数
GPU_ID=7
MODEL_PATH="rotated_llama_model"  # 旋转后的模型路径
SEQLEN=2048
BATCH_SIZE=1

# 运行评估
CUDA_VISIBLE_DEVICES=$GPU_ID python eval_model_ppl.py \
    --model-path $MODEL_PATH \
    --seqlen $SEQLEN \
    --batch-size $BATCH_SIZE \
    --device cuda

# 如果要使用简单方法（不使用批处理），添加 --simple 参数：
# CUDA_VISIBLE_DEVICES=$GPU_ID python eval_model_ppl.py \
#     --model-path $MODEL_PATH \
#     --seqlen $SEQLEN \
#     --simple \
#     --device cuda
