#!/usr/bin/env python3
"""
Fine-tune only RMSNorm parameters for pruned Llama model.

Usage:
    python genetic/finetune_rmsnorm.py
"""

import torch
import torch.nn as nn
import os
import sys
import json
import time
import numpy as np
from transformers import LlamaForCausalLM, AutoTokenizer
from datasets import load_dataset

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.extend([project_root, current_dir])

from modeling import DecoupledLlamaModel


def freeze_all_except_rmsnorm(model):
    """Freeze all parameters except RMSNorm weights."""
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())

    # Unfreeze RMSNorm parameters
    rmsnorm_params = 0
    rmsnorm_count = 0

    print("\nUnfreezing RMSNorm parameters:")
    for name, module in model.named_modules():
        # LlamaRMSNorm is the class name in transformers
        if module.__class__.__name__ == 'LlamaRMSNorm':
            rmsnorm_count += 1
            for param_name, param in module.named_parameters(recurse=False):
                param.requires_grad = True
                rmsnorm_params += param.numel()
                if rmsnorm_count <= 5:  # Only print first 5
                    print(f"  {name}.{param_name}: {param.shape}")

    if rmsnorm_count > 5:
        print(f"  ... and {rmsnorm_count - 5} more RMSNorm modules")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"Parameter Summary:")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {trainable_params/total_params*100:.4f}%")
    print(f"RMSNorm modules: {rmsnorm_count}")
    print(f"RMSNorm parameters: {rmsnorm_params:,}")
    print(f"{'='*60}\n")

    return trainable_params


def get_wikitext2_data(tokenizer, seqlen, split='train'):
    """Load WikiText2 data."""
    print(f"Loading WikiText2 {split} data...")
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)

    text = "\n\n".join(data['text'])
    encoded = tokenizer(text, return_tensors='pt')

    total_samples = encoded.input_ids.numel() // seqlen
    input_ids = encoded.input_ids[0, :total_samples * seqlen]
    input_ids = input_ids.reshape(total_samples, seqlen)

    print(f"Loaded {input_ids.shape[0]} samples of length {seqlen}")
    return input_ids


def evaluate_ppl(model, eval_data, device, max_samples=128):
    """Evaluate perplexity."""
    model.eval()
    total_loss = 0.0
    num_samples = min(max_samples, eval_data.shape[0])

    loss_fct = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i in range(num_samples):
            input_ids = eval_data[i:i+1].to(device)

            # DecoupledLlamaModel only returns logits
            logits = model(input_ids)

            # Shift logits and labels for language modeling
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            # Calculate loss
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            total_loss += loss.item()

    avg_loss = total_loss / num_samples
    ppl = np.exp(avg_loss)

    model.train()
    return ppl, avg_loss


def main():
    # ==================== Configuration ====================
    # Read from environment variables (set by run_finetune_rmsnorm.sh)
    MODEL_PATH = os.getenv("FT_MODEL_PATH", "/gpfs/volcano/models/meta-llama/Llama-2-13b-hf")
    CHROMOSOME_PATH = os.getenv("FT_CHROMOSOME_PATH", "genetic/checkpoints/20250312-143022/best_individual.json")
    OUTPUT_DIR = os.getenv("FT_OUTPUT_DIR", "genetic/outputs/rmsnorm_tuned")
    GPU_ID = os.getenv("FT_GPU_ID", "0")

    MAX_STEPS = int(os.getenv("FT_MAX_STEPS", "1000"))
    BATCH_SIZE = int(os.getenv("FT_BATCH_SIZE", "4"))
    LEARNING_RATE = float(os.getenv("FT_LEARNING_RATE", "0.0001"))
    WARMUP_STEPS = int(os.getenv("FT_WARMUP_STEPS", "100"))
    GRADIENT_ACCUMULATION_STEPS = int(os.getenv("FT_GRADIENT_ACCUM", "1"))

    SEQLEN = int(os.getenv("FT_SEQLEN", "2048"))
    NUM_TRAIN_SAMPLES = int(os.getenv("FT_NUM_TRAIN_SAMPLES", "5000"))
    EVAL_SAMPLES = int(os.getenv("FT_EVAL_SAMPLES", "128"))
    EVAL_INTERVAL = int(os.getenv("FT_EVAL_INTERVAL", "50"))
    SAVE_INTERVAL = int(os.getenv("FT_SAVE_INTERVAL", "200"))

    # ==================== Setup ====================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = f"cuda:{GPU_ID}"

    print(f"{'='*60}")
    print(f"RMSNorm Fine-tuning")
    print(f"{'='*60}")
    print(f"Model: {MODEL_PATH}")
    print(f"Chromosome: {CHROMOSOME_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: {device}")
    print(f"Max steps: {MAX_STEPS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"{'='*60}\n")

    # Load chromosome
    print(f"Loading chromosome from {CHROMOSOME_PATH}...")
    with open(CHROMOSOME_PATH, 'r') as f:
        best_data = json.load(f)

    chromosome = best_data['chromosome']
    original_ppl = best_data.get('fitness', None)

    print(f"Chromosome loaded: {len(chromosome)} genes")
    print(f"Original PPL: {original_ppl:.4f}\n" if original_ppl else "Original PPL: N/A\n")

    # Load tokenizer
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model (this may take a few minutes)...")
    base_model = LlamaForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True
    )

    # Create decoupled model
    print(f"Creating DecoupledLlamaModel...")
    model = DecoupledLlamaModel(base_model, chromosome)

    # Freeze all except RMSNorm
    trainable_params = freeze_all_except_rmsnorm(model)

    if trainable_params == 0:
        print("ERROR: No trainable parameters found!")
        return

    # Load data
    train_data = get_wikitext2_data(tokenizer, SEQLEN, split='train')
    if NUM_TRAIN_SAMPLES > 0:
        train_data = train_data[:NUM_TRAIN_SAMPLES]
        print(f"Using {NUM_TRAIN_SAMPLES} training samples")

    eval_data = get_wikitext2_data(tokenizer, SEQLEN, split='test')

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )

    # Evaluate before training
    print(f"\n{'='*60}")
    print(f"Evaluation BEFORE training")
    print(f"{'='*60}")
    initial_ppl, initial_loss = evaluate_ppl(model, eval_data, device, EVAL_SAMPLES)
    print(f"Initial PPL: {initial_ppl:.4f}")
    print(f"Initial Loss: {initial_loss:.6f}")
    print(f"{'='*60}\n")

    # Training
    print(f"{'='*60}")
    print(f"Starting Training")
    print(f"{'='*60}\n")

    model.train()
    global_step = 0
    best_ppl = initial_ppl

    num_samples = train_data.shape[0]
    num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE

    epoch = 0
    while global_step < MAX_STEPS:
        epoch += 1
        print(f"\n--- Epoch {epoch} ---")

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
            logits = model(batch)

            # Shift logits and labels for language modeling
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()

            # Calculate loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss = loss / GRADIENT_ACCUMULATION_STEPS

            # Backward
            loss.backward()

            # Update
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                # Print progress
                if global_step % 10 == 0:
                    print(f"Step {global_step}/{MAX_STEPS} | Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}", flush=True)

                # Evaluate
                if global_step % EVAL_INTERVAL == 0:
                    eval_ppl, eval_loss = evaluate_ppl(model, eval_data, device, EVAL_SAMPLES)

                    print(f"\n{'='*60}")
                    print(f"Evaluation at Step {global_step}")
                    print(f"{'='*60}")
                    print(f"PPL: {eval_ppl:.4f} (Initial: {initial_ppl:.4f})")
                    print(f"Loss: {eval_loss:.6f}")
                    print(f"Improvement: {initial_ppl - eval_ppl:.4f} ({(initial_ppl - eval_ppl) / initial_ppl * 100:.2f}%)")
                    print(f"{'='*60}\n", flush=True)

                    # Save if best
                    if eval_ppl < best_ppl:
                        best_ppl = eval_ppl
                        save_path = os.path.join(OUTPUT_DIR, "best_rmsnorm.pt")

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

                        print(f"  ✓ Saved NEW BEST model (PPL: {eval_ppl:.4f})\n", flush=True)

                # Save checkpoint
                if global_step % SAVE_INTERVAL == 0:
                    ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint_step{global_step}.pt")

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

                    print(f"  ✓ Saved checkpoint at step {global_step}\n", flush=True)

    # Final evaluation
    print(f"\n{'='*60}")
    print(f"FINAL Evaluation")
    print(f"{'='*60}")
    final_ppl, final_loss = evaluate_ppl(model, eval_data, device, EVAL_SAMPLES)

    print(f"Initial PPL:  {initial_ppl:.4f}")
    print(f"Final PPL:    {final_ppl:.4f}")
    print(f"Best PPL:     {best_ppl:.4f}")
    print(f"")
    print(f"Total Improvement: {initial_ppl - best_ppl:.4f} ({(initial_ppl - best_ppl) / initial_ppl * 100:.2f}%)")
    print(f"{'='*60}\n")

    # Save final model
    final_path = os.path.join(OUTPUT_DIR, "final_rmsnorm.pt")
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

    with open(os.path.join(OUTPUT_DIR, "training_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Training complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"  - Best model: best_rmsnorm.pt (PPL: {best_ppl:.4f})")
    print(f"  - Final model: final_rmsnorm.pt (PPL: {final_ppl:.4f})")
    print(f"  - Summary: training_summary.json")


if __name__ == "__main__":
    main()
