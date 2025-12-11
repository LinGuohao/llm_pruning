#!/bin/bash

# --- Configuration ---
# 默认模型路径
MODEL_PATH="/gpfs/volcano/models/meta-llama/Llama-2-13b-hf"

# 指定要使用的物理GPU ID列表，例如 "0,1,2,3" 或 "1,5"。
GPU_IDS="0,1"

# 核心遗传算法参数
MAX_GENERATIONS=50      # 进化代数
POPULATION_SIZE=20      # 种群大小
MUTATION_RATE=0.05      # 变异率
CROSSOVER_RATE=0.8      # 交叉率
CROSSOVER_TYPE="uniform" # uniform, onepoint, twopoint
SELECTION_METHOD="topNw" # tournament, top20, topNw

# 约束条件
MAX_PARAM_RATIO=0.5     # 最大参数比例 (0.5 = 50%)
MAX_LOOP_COUNT=5        # 最大循环次数

# 评估参数
EVAL_SAMPLES=128        # 每次评估使用的样本数
SEQLEN=2048             # 序列长度

# 其他
CHECKPOINT_INTERVAL=5   # 每几代保存一次
SEED=42                 # 随机种子

# Resume (可选，留空则从头开始)
# RESUME_FROM="/path/to/checkpoint_gen10.json"
RESUME_FROM=""

# --- Script Logic ---

echo "Starting Genetic Algorithm Evolution..."
echo "Model: ${MODEL_PATH}"
echo "GPUs: ${GPU_IDS}"
echo "Config: Gen=${MAX_GENERATIONS}, Pop=${POPULATION_SIZE}, Mut=${MUTATION_RATE}, Cross=${CROSSOVER_RATE}"

CMD="python genetic/main_evolution.py \
    --model_path \"${MODEL_PATH}\" \
    --gpu_ids \"${GPU_IDS}\" \
    --max_generations ${MAX_GENERATIONS} \
    --population_size ${POPULATION_SIZE} \
    --mutation_rate ${MUTATION_RATE} \
    --crossover_rate ${CROSSOVER_RATE} \
    --crossover_type \"${CROSSOVER_TYPE}\" \
    --selection_method \"${SELECTION_METHOD}\" \
    --max_param_ratio ${MAX_PARAM_RATIO} \
    --max_loop_count ${MAX_LOOP_COUNT} \
    --eval_samples ${EVAL_SAMPLES} \
    --seqlen ${SEQLEN} \
    --checkpoint_interval ${CHECKPOINT_INTERVAL} \
    --seed ${SEED}"

if [ -n "$RESUME_FROM" ]; then
    CMD="$CMD --resume_from \"${RESUME_FROM}\""
fi

# 执行命令
echo "Running command:"
echo "$CMD"
eval $CMD
