#!/bin/bash


# checkpoint_step1000.pt 包含optimizer state，适合继续训练
# best_rmsnorm.pt 或 final_rmsnorm.pt 没有optimizer state，不太适合resume（但也可以用，就是会重新初始化optimizer）
# MAX_STEPS是总步数，所以如果从1000继续，想再训练2000步，就设置成3000
# --- Configuration ---
# 模型路径
MODEL_PATH="/gpfs/volcano/models/meta-llama/Llama-2-13b-hf"

# 最佳个体JSON文件路径 (修改这个!)
CHROMOSOME_PATH="/data/algorithm/linguohao/nas/llm_pruning/genetic/checkpoints/20251211-230928/best_individual.json"

# 使用的GPU ID
GPU_ID="4"

# 训练参数
MAX_STEPS=3000          # 最大训练步数
BATCH_SIZE=1            # 批大小
LEARNING_RATE=0.00001    # 学习率 (1e-4)
WARMUP_STEPS=100        # 预热步数
GRADIENT_ACCUM=1        # 梯度累积步数

# 数据参数
SEQLEN=2048             # 序列长度
NUM_TRAIN_SAMPLES=5000  # 训练样本数 (0 = 使用全部)
EVAL_SAMPLES=128        # 评估样本数

# 评估和保存间隔
EVAL_INTERVAL=50        # 每多少步评估一次
SAVE_INTERVAL=200       # 每多少步保存checkpoint

# 输出路径 (留空则自动生成时间戳目录)
OUTPUT_DIR=""

# Resume配置 (留空则从头训练，填入checkpoint路径则继续训练)
# 例如: RESUME_FROM="genetic/outputs/rmsnorm_20251212-134756/checkpoint_step1000.pt"
RESUME_FROM="/data/algorithm/linguohao/nas/llm_pruning/genetic/outputs/rmsnorm_20251212-152432/checkpoint_step1000.pt"

# --- Script Logic ---

echo "=========================================="
echo "RMSNorm Fine-tuning"
echo "=========================================="
echo "Chromosome file: ${CHROMOSOME_PATH}"
echo "Model path: ${MODEL_PATH}"
echo "GPU ID: ${GPU_ID}"
echo "Max steps: ${MAX_STEPS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "=========================================="
echo ""

# 检查文件是否存在
if [ ! -f "$CHROMOSOME_PATH" ]; then
    echo "Error: Chromosome file not found: $CHROMOSOME_PATH"
    echo "Please edit run_finetune_rmsnorm.sh and set CHROMOSOME_PATH to the correct path"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    echo "Please edit run_finetune_rmsnorm.sh and set MODEL_PATH to the correct path"
    exit 1
fi

# 如果没有指定输出目录，自动生成
if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
    OUTPUT_DIR="genetic/outputs/rmsnorm_${TIMESTAMP}"
    echo "Auto-generated output directory: ${OUTPUT_DIR}"
    echo ""
fi

# 导出环境变量供Python脚本使用
export FT_MODEL_PATH="${MODEL_PATH}"
export FT_CHROMOSOME_PATH="${CHROMOSOME_PATH}"
export FT_OUTPUT_DIR="${OUTPUT_DIR}"
export FT_GPU_ID="${GPU_ID}"
export FT_MAX_STEPS="${MAX_STEPS}"
export FT_BATCH_SIZE="${BATCH_SIZE}"
export FT_LEARNING_RATE="${LEARNING_RATE}"
export FT_WARMUP_STEPS="${WARMUP_STEPS}"
export FT_GRADIENT_ACCUM="${GRADIENT_ACCUM}"
export FT_SEQLEN="${SEQLEN}"
export FT_NUM_TRAIN_SAMPLES="${NUM_TRAIN_SAMPLES}"
export FT_EVAL_SAMPLES="${EVAL_SAMPLES}"
export FT_EVAL_INTERVAL="${EVAL_INTERVAL}"
export FT_SAVE_INTERVAL="${SAVE_INTERVAL}"
export FT_RESUME_FROM="${RESUME_FROM}"

# 运行Python脚本
python genetic/finetune_rmsnorm.py

echo ""
echo "=========================================="
echo "✓ Fine-tuning complete!"
echo "=========================================="
