#!/bin/bash

# --- Configuration ---
# 模型路径
MODEL_PATH="/gpfs/volcano/models/meta-llama/Llama-2-13b-hf"

# 最佳个体JSON文件路径 (修改这个!)
BEST_INDIVIDUAL_PATH="genetic/checkpoints/20250312-143022/best_individual.json"

# 使用的GPU ID
GPU_ID="0"

# 评估参数
EVAL_SAMPLES=128        # 评估样本数
SEQLEN=2048             # 序列长度

# 结果保存路径 (可选，留空则自动保存在best_individual.json同目录下)
OUTPUT_PATH=""

# --- Script Logic ---

echo "=========================================="
echo "Evaluating Best Individual"
echo "=========================================="
echo "Best individual file: ${BEST_INDIVIDUAL_PATH}"
echo "Model path: ${MODEL_PATH}"
echo "GPU ID: ${GPU_ID}"
echo "Eval samples: ${EVAL_SAMPLES}"
echo "Sequence length: ${SEQLEN}"
echo "=========================================="
echo ""

# 检查文件是否存在
if [ ! -f "$BEST_INDIVIDUAL_PATH" ]; then
    echo "Error: Best individual file not found: $BEST_INDIVIDUAL_PATH"
    echo "Please edit run_eval_best.sh and set BEST_INDIVIDUAL_PATH to the correct path"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    echo "Please edit run_eval_best.sh and set MODEL_PATH to the correct path"
    exit 1
fi

# 构建命令
CMD="python genetic/eval_best_individual.py \
    --model_path \"${MODEL_PATH}\" \
    --best_individual_path \"${BEST_INDIVIDUAL_PATH}\" \
    --gpu_id \"${GPU_ID}\" \
    --seqlen ${SEQLEN} \
    --eval_samples ${EVAL_SAMPLES}"

if [ -n "$OUTPUT_PATH" ]; then
    CMD="$CMD --output_path \"${OUTPUT_PATH}\""
fi

# 执行命令
echo "Running command:"
echo "$CMD"
echo ""
eval $CMD

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
