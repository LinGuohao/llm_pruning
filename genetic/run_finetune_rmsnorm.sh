#!/bin/bash

# --- Configuration ---
# 模型路径
MODEL_PATH="/gpfs/volcano/models/meta-llama/Llama-2-13b-hf"

# 最佳个体JSON文件路径 (修改这个!)
CHROMOSOME_PATH="genetic/checkpoints/20250312-143022/best_individual.json"

# 使用的GPU ID
GPU_ID="0"

# 训练参数
MAX_STEPS=1000          # 最大训练步数
BATCH_SIZE=4            # 批大小
LEARNING_RATE=1e-4      # 学习率
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

# 构建Python脚本内容（内联配置）
# 这样可以通过shell变量直接控制所有参数
python -c "
import torch
import torch.nn as nn
import os
import sys
import json
import numpy as np
from transformers import LlamaForCausalLM, AutoTokenizer
from datasets import load_dataset

# Add paths
current_dir = os.path.dirname(os.path.abspath('genetic/finetune_rmsnorm.py'))
project_root = os.path.dirname(current_dir)
sys.path.extend([project_root, 'genetic'])

from modeling import DecoupledLlamaModel

# Configuration from shell
MODEL_PATH = '${MODEL_PATH}'
CHROMOSOME_PATH = '${CHROMOSOME_PATH}'
OUTPUT_DIR = '${OUTPUT_DIR}'
GPU_ID = '${GPU_ID}'
MAX_STEPS = ${MAX_STEPS}
BATCH_SIZE = ${BATCH_SIZE}
LEARNING_RATE = ${LEARNING_RATE}
WARMUP_STEPS = ${WARMUP_STEPS}
GRADIENT_ACCUMULATION_STEPS = ${GRADIENT_ACCUM}
SEQLEN = ${SEQLEN}
NUM_TRAIN_SAMPLES = ${NUM_TRAIN_SAMPLES}
EVAL_SAMPLES = ${EVAL_SAMPLES}
EVAL_INTERVAL = ${EVAL_INTERVAL}
SAVE_INTERVAL = ${SAVE_INTERVAL}

def freeze_all_except_rmsnorm(model):
    \"\"\"Freeze all parameters except RMSNorm weights.\"\"\"
    for param in model.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    rmsnorm_params = 0
    rmsnorm_count = 0

    print('\\nUnfreezing RMSNorm parameters:')
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'LlamaRMSNorm':
            rmsnorm_count += 1
            for param_name, param in module.named_parameters(recurse=False):
                param.requires_grad = True
                rmsnorm_params += param.numel()
                if rmsnorm_count <= 5:
                    print(f'  {name}.{param_name}: {param.shape}')

    if rmsnorm_count > 5:
        print(f'  ... and {rmsnorm_count - 5} more RMSNorm modules')

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'\\n{'='*60}')
    print(f'Parameter Summary:')
    print(f'{'='*60}')
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print(f'Trainable percentage: {trainable_params/total_params*100:.4f}%')
    print(f'RMSNorm modules: {rmsnorm_count}')
    print(f'RMSNorm parameters: {rmsnorm_params:,}')
    print(f'{'='*60}\\n')

    return trainable_params

def get_wikitext2_data(tokenizer, seqlen, split='train'):
    \"\"\"Load WikiText2 data.\"\"\"
    print(f'Loading WikiText2 {split} data...')
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)

    text = '\\n\\n'.join(data['text'])
    encoded = tokenizer(text, return_tensors='pt')

    total_samples = encoded.input_ids.numel() // seqlen
    input_ids = encoded.input_ids[0, :total_samples * seqlen]
    input_ids = input_ids.reshape(total_samples, seqlen)

    print(f'Loaded {input_ids.shape[0]} samples of length {seqlen}')
    return input_ids

def evaluate_ppl(model, eval_data, device, max_samples=128):
    \"\"\"Evaluate perplexity.\"\"\"
    model.eval()
    total_loss = 0.0
    num_samples = min(max_samples, eval_data.shape[0])

    with torch.no_grad():
        for i in range(num_samples):
            input_ids = eval_data[i:i+1].to(device)
            outputs = model(input_ids, labels=input_ids)
            total_loss += outputs.loss.item()

    avg_loss = total_loss / num_samples
    ppl = np.exp(avg_loss)

    model.train()
    return ppl, avg_loss

# Main execution
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = f'cuda:{GPU_ID}'

print(f'{'='*60}')
print(f'RMSNorm Fine-tuning')
print(f'{'='*60}')
print(f'Model: {MODEL_PATH}')
print(f'Chromosome: {CHROMOSOME_PATH}')
print(f'Output: {OUTPUT_DIR}')
print(f'Device: {device}')
print(f'Max steps: {MAX_STEPS}')
print(f'Batch size: {BATCH_SIZE}')
print(f'Learning rate: {LEARNING_RATE}')
print(f'{'='*60}\\n')

# Load chromosome
print(f'Loading chromosome from {CHROMOSOME_PATH}...')
with open(CHROMOSOME_PATH, 'r') as f:
    best_data = json.load(f)

chromosome = best_data['chromosome']
original_ppl = best_data.get('fitness', None)

print(f'Chromosome loaded: {len(chromosome)} genes')
print(f'Original PPL: {original_ppl:.4f}\\n' if original_ppl else 'Original PPL: N/A\\n')

# Load tokenizer
print(f'Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
print(f'Loading model (this may take a few minutes)...')
base_model = LlamaForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map=device,
    low_cpu_mem_usage=True
)

# Create decoupled model
print(f'Creating DecoupledLlamaModel...')
model = DecoupledLlamaModel(base_model, chromosome)

# Freeze all except RMSNorm
trainable_params = freeze_all_except_rmsnorm(model)

if trainable_params == 0:
    print('ERROR: No trainable parameters found!')
    sys.exit(1)

# Load data
train_data = get_wikitext2_data(tokenizer, SEQLEN, split='train')
if NUM_TRAIN_SAMPLES > 0:
    train_data = train_data[:NUM_TRAIN_SAMPLES]
    print(f'Using {NUM_TRAIN_SAMPLES} training samples')

eval_data = get_wikitext2_data(tokenizer, SEQLEN, split='test')

# Setup optimizer
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
    betas=(0.9, 0.95),
    weight_decay=0.01
)

# Evaluate before training
print(f'\\n{'='*60}')
print(f'Evaluation BEFORE training')
print(f'{'='*60}')
initial_ppl, initial_loss = evaluate_ppl(model, eval_data, device, EVAL_SAMPLES)
print(f'Initial PPL: {initial_ppl:.4f}')
print(f'Initial Loss: {initial_loss:.6f}')
print(f'{'='*60}\\n')

# Training
print(f'{'='*60}')
print(f'Starting Training')
print(f'{'='*60}\\n')

model.train()
global_step = 0
best_ppl = initial_ppl

num_samples = train_data.shape[0]
num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE

epoch = 0
while global_step < MAX_STEPS:
    epoch += 1
    print(f'\\n--- Epoch {epoch} ---')

    # Shuffle data
    indices = torch.randperm(num_samples)
    train_data_shuffled = train_data[indices]

    for batch_idx in range(num_batches):
        if global_step >= MAX_STEPS:
            break

        # Get batch
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, num_samples)
        batch = train_data_shuffled[start_idx:end_idx].to(device)

        # Forward
        outputs = model(batch, labels=batch)
        loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS

        # Backward
        loss.backward()

        # Update
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            # Print progress
            if global_step % 10 == 0:
                print(f'Step {global_step}/{MAX_STEPS} | Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}', flush=True)

            # Evaluate
            if global_step % EVAL_INTERVAL == 0:
                eval_ppl, eval_loss = evaluate_ppl(model, eval_data, device, EVAL_SAMPLES)

                print(f'\\n{'='*60}')
                print(f'Evaluation at Step {global_step}')
                print(f'{'='*60}')
                print(f'PPL: {eval_ppl:.4f} (Initial: {initial_ppl:.4f})')
                print(f'Loss: {eval_loss:.6f}')
                print(f'Improvement: {initial_ppl - eval_ppl:.4f} ({(initial_ppl - eval_ppl) / initial_ppl * 100:.2f}%)')
                print(f'{'='*60}\\n', flush=True)

                # Save if best
                if eval_ppl < best_ppl:
                    best_ppl = eval_ppl
                    save_path = os.path.join(OUTPUT_DIR, 'best_rmsnorm.pt')

                    # Save only RMSNorm parameters
                    rmsnorm_state = {}
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            rmsnorm_state[name] = param.cpu().clone()

                    torch.save({
                        'rmsnorm_weights': rmsnorm_state,
                        'chromosome': chromosome,
                        'step': global_step,
                        'ppl': eval_ppl,
                        'loss': eval_loss,
                        'initial_ppl': initial_ppl
                    }, save_path)

                    print(f'  ✓ Saved NEW BEST model (PPL: {eval_ppl:.4f})\\n', flush=True)

            # Save checkpoint
            if global_step % SAVE_INTERVAL == 0:
                ckpt_path = os.path.join(OUTPUT_DIR, f'checkpoint_step{global_step}.pt')

                rmsnorm_state = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        rmsnorm_state[name] = param.cpu().clone()

                torch.save({
                    'rmsnorm_weights': rmsnorm_state,
                    'chromosome': chromosome,
                    'step': global_step,
                    'optimizer': optimizer.state_dict()
                }, ckpt_path)

                print(f'  ✓ Saved checkpoint at step {global_step}\\n', flush=True)

# Final evaluation
print(f'\\n{'='*60}')
print(f'FINAL Evaluation')
print(f'{'='*60}')
final_ppl, final_loss = evaluate_ppl(model, eval_data, device, EVAL_SAMPLES)

print(f'Initial PPL:  {initial_ppl:.4f}')
print(f'Final PPL:    {final_ppl:.4f}')
print(f'Best PPL:     {best_ppl:.4f}')
print(f'')
print(f'Total Improvement: {initial_ppl - best_ppl:.4f} ({(initial_ppl - best_ppl) / initial_ppl * 100:.2f}%)')
print(f'{'='*60}\\n')

# Save final model
final_path = os.path.join(OUTPUT_DIR, 'final_rmsnorm.pt')
rmsnorm_state = {}
for name, param in model.named_parameters():
    if param.requires_grad:
        rmsnorm_state[name] = param.cpu().clone()

torch.save({
    'rmsnorm_weights': rmsnorm_state,
    'chromosome': chromosome,
    'step': global_step,
    'ppl': final_ppl,
    'loss': final_loss,
    'initial_ppl': initial_ppl,
    'best_ppl': best_ppl
}, final_path)

# Save summary
summary = {
    'initial_ppl': float(initial_ppl),
    'final_ppl': float(final_ppl),
    'best_ppl': float(best_ppl),
    'improvement': float(initial_ppl - best_ppl),
    'improvement_percent': float((initial_ppl - best_ppl) / initial_ppl * 100),
    'total_steps': global_step,
    'trainable_params': trainable_params,
    'config': {
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'max_steps': MAX_STEPS,
        'warmup_steps': WARMUP_STEPS
    }
}

with open(os.path.join(OUTPUT_DIR, 'training_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f'✓ Training complete!')
print(f'Results saved to: {OUTPUT_DIR}')
print(f'  - Best model: best_rmsnorm.pt (PPL: {best_ppl:.4f})')
print(f'  - Final model: final_rmsnorm.pt (PPL: {final_ppl:.4f})')
print(f'  - Summary: training_summary.json')
"

echo ""
echo "=========================================="
echo "✓ Fine-tuning complete!"
echo "=========================================="
