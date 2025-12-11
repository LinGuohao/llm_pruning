#!/bin/bash

# 评估旋转模型（带 shortcut Q）的困惑度
# 使用与 eval_model_ppl.py 完全相同的数据集和评估方法

# 配置参数
GPU_ID=7
MODEL_PATH="rotated_llama_model_with_shortcut"  # 旋转后的模型路径（带 shortcut Q）
SEQLEN=4096
BATCH_SIZE=1

# 运行评估
CUDA_VISIBLE_DEVICES=$GPU_ID python -u eval_rotated_model_ppl.py \
    --model-path $MODEL_PATH \
    --seqlen $SEQLEN \
    --batch-size $BATCH_SIZE
