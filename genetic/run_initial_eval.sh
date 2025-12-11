#!/bin/bash

# --- Configuration ---
# 默认模型路径，请根据您的实际情况修改。
# 例如: MODEL_PATH="/path/to/your/llama-2-13b-hf"
# 如果 /gpfs/volcano/models/meta-llama/Llama-2-13b-hf 不存在，请务必修改此项。
MODEL_PATH="/gpfs/volcano/models/meta-llama/Llama-2-13b-hf"

# 指定要使用的物理GPU ID列表，例如 "1,5" 或 "0,2,3"。
# Python脚本将直接尝试加载模型到这些物理GPU上。
# 请确保这些GPU是您系统上实际可用的，并且有足够的显存加载模型。
# 如果不指定或留空，将尝试使用CPU。
GPU_IDS="0,1,4" # 示例：指定物理GPU 1和物理GPU 5

# 其他评估和遗传算法参数
SEQLEN=2048         # 评估序列长度
EVAL_SAMPLES=128    # 用于PPL评估的WikiText2样本数 (可调，例如10-256)
POPULATION_SIZE=20  # 初始种群大小
MAX_PARAM_RATIO=0.5 # 最大参数比例，例如0.5表示50%
MAX_LOOP_COUNT=2    # 染色体中模块的最大循环次数
SEED=42             # 随机种子，用于复现性

# --- Script Logic ---

echo "Starting Genetic Algorithm Initial Population Evaluation Test..."
echo "Model Path: ${MODEL_PATH}"
echo "Physical GPUs to use: ${GPU_IDS}"
echo "Sequence Length: ${SEQLEN}"
echo "Eval Samples: ${EVAL_SAMPLES}"
echo "Population Size: ${POPULATION_SIZE}"
echo "Max Param Ratio: ${MAX_PARAM_RATIO}"
echo "Max Loop Count: ${MAX_LOOP_COUNT}"
echo "Random Seed: ${SEED}"
echo "----------------------------------------------------------------------"

# ！！！重要：这里不再设置 CUDA_VISIBLE_DEVICES。
# Python脚本将直接使用 --gpu_ids 参数中指定的物理GPU编号。

# 运行Python脚本
python genetic/main_eval_initial_population.py \
    --model_path "${MODEL_PATH}" \
    --gpu_ids "${GPU_IDS}" \
    --seqlen "${SEQLEN}" \
    --eval_samples "${EVAL_SAMPLES}" \
    --population_size "${POPULATION_SIZE}" \
    --max_param_ratio "${MAX_PARAM_RATIO}" \
    --max_loop_count "${MAX_LOOP_COUNT}" \
    --seed "${SEED}"

# 检查Python脚本的退出状态
if [ $? -eq 0 ]; then
    echo "----------------------------------------------------------------------"
    echo "Python script finished successfully."
else
    echo "----------------------------------------------------------------------"
    echo "Python script failed. Please check the error messages above."
fi
echo "----------------------------------------------------------------------"
echo "Overall Test Finished."
echo "----------------------------------------------------------------------"
