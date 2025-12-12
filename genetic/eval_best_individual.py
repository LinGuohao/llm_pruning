#!/usr/bin/env python3
"""
Evaluate the best individual found by genetic algorithm.
Loads the best individual from JSON and re-evaluates its PPL on WikiText2.

Usage:
    python genetic/eval_best_individual.py \
        --model_path /path/to/Llama-2-13b \
        --best_individual_path genetic/checkpoints/xxx/best_individual.json \
        --gpu_id 0 \
        --seqlen 2048 \
        --eval_samples 128
"""

import torch
import torch.nn as nn
import os
import sys
import argparse
import json
import numpy as np
from transformers import LlamaForCausalLM, AutoTokenizer
from datasets import load_dataset

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(current_dir)

from modeling import DecoupledLlamaModel, decode_chromosome


def get_wikitext2_data(tokenizer, seqlen):
    """Load and prepare WikiText2 dataset."""
    print("Loading WikiText2 test data...")
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Join all text
    text = "\n\n".join(testdata['text'])

    # Tokenize
    testenc = tokenizer(text, return_tensors='pt')

    # Reshape into chunks of seqlen
    nsamples = testenc.input_ids.numel() // seqlen
    input_ids = testenc.input_ids[0, :nsamples * seqlen]
    input_ids = input_ids.reshape(nsamples, seqlen)

    print(f"Prepared {nsamples} samples of length {seqlen}")
    return input_ids


def evaluate_ppl(model, chromosome, tokenizer, seqlen, eval_samples, device):
    """Evaluate perplexity for a given chromosome."""
    print(f"\nEvaluating chromosome on {device}...")
    print(f"Chromosome: {chromosome}")

    # Decode chromosome to execution path
    execution_path = decode_chromosome(chromosome)
    print(f"Execution path length (effective depth): {len(execution_path)}")
    print(f"Unique modules selected: {sum(1 for v in chromosome if v > 0)}/80")

    # Create decoupled model
    print("Creating DecoupledLlamaModel...")
    decoupled_model = DecoupledLlamaModel(model, chromosome)
    decoupled_model.eval()

    # Load data
    eval_data = get_wikitext2_data(tokenizer, seqlen)
    nsamples = min(eval_samples, eval_data.shape[0])

    print(f"\nEvaluating on {nsamples} samples...")

    loss_fct = nn.CrossEntropyLoss().to(device)
    acc_loss = 0.0

    with torch.no_grad():
        for i in range(nsamples):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{nsamples} samples", flush=True)

            # Get single sample
            input_ids = eval_data[i:i+1].to(device)

            # Forward pass
            logits = decoupled_model(input_ids)

            # Shift logits and labels
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            # Compute loss
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            acc_loss += loss.item()

    # Calculate PPL
    avg_loss = acc_loss / nsamples
    ppl = np.exp(avg_loss)

    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"{'='*60}")
    print(f"  Samples evaluated: {nsamples}")
    print(f"  Average loss: {avg_loss:.6f}")
    print(f"  Perplexity (PPL): {ppl:.6f}")
    print(f"{'='*60}")

    # Cleanup
    del decoupled_model
    torch.cuda.empty_cache()

    return ppl, avg_loss


def main():
    parser = argparse.ArgumentParser(description="Evaluate best individual from genetic algorithm")

    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the Llama-2-13b model")
    parser.add_argument("--best_individual_path", type=str, required=True,
                        help="Path to best_individual.json file")

    # Optional arguments
    parser.add_argument("--gpu_id", type=str, default="0",
                        help="GPU ID to use (default: 0)")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Sequence length (default: 2048)")
    parser.add_argument("--eval_samples", type=int, default=128,
                        help="Number of samples to evaluate (default: 128)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save evaluation results (default: same dir as best_individual.json)")

    args = parser.parse_args()

    # Setup device
    device = f"cuda:{args.gpu_id}"
    print(f"Using device: {device}")

    # Load best individual
    print(f"\nLoading best individual from: {args.best_individual_path}")
    with open(args.best_individual_path, 'r') as f:
        best_data = json.load(f)

    chromosome = best_data['chromosome']
    original_fitness = best_data.get('fitness', None)
    params_ratio = best_data.get('params_ratio', None)
    num_modules = best_data.get('num_modules', None)
    effective_depth = best_data.get('effective_depth', None)

    print(f"\nBest Individual Info:")
    print(f"  Original fitness (PPL): {original_fitness}")
    print(f"  Parameters ratio: {params_ratio:.2%}" if params_ratio else "  Parameters ratio: N/A")
    print(f"  Unique modules: {num_modules}/80" if num_modules else "  Unique modules: N/A")
    print(f"  Effective depth: {effective_depth}" if effective_depth else "  Effective depth: N/A")

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model from {args.model_path}...")
    print("(This may take a few minutes...)")
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True
    )
    model.eval()
    print("Model loaded successfully!")

    # Evaluate
    ppl, avg_loss = evaluate_ppl(
        model=model,
        chromosome=chromosome,
        tokenizer=tokenizer,
        seqlen=args.seqlen,
        eval_samples=args.eval_samples,
        device=device
    )

    # Save results
    if args.output_path is None:
        # Save in same directory as best_individual.json
        output_dir = os.path.dirname(args.best_individual_path)
        args.output_path = os.path.join(output_dir, "best_individual_eval_results.json")

    results = {
        "chromosome": chromosome,
        "original_fitness": original_fitness,
        "re_evaluated_ppl": ppl,
        "re_evaluated_loss": avg_loss,
        "params_ratio": params_ratio,
        "num_modules": num_modules,
        "effective_depth": effective_depth,
        "eval_config": {
            "model_path": args.model_path,
            "seqlen": args.seqlen,
            "eval_samples": args.eval_samples,
            "device": device
        }
    }

    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output_path}")

    # Compare with original
    if original_fitness is not None:
        diff = ppl - original_fitness
        diff_percent = (diff / original_fitness) * 100
        print(f"\nComparison with original fitness:")
        print(f"  Original PPL: {original_fitness:.6f}")
        print(f"  Re-evaluated PPL: {ppl:.6f}")
        print(f"  Difference: {diff:+.6f} ({diff_percent:+.2f}%)")

        if abs(diff_percent) < 1.0:
            print(f"  ✓ Results are consistent (difference < 1%)")
        elif abs(diff_percent) < 5.0:
            print(f"  ⚠ Small difference detected (1-5%)")
        else:
            print(f"  ⚠ Large difference detected (>5%), this may indicate:")
            print(f"     - Different random seed")
            print(f"     - Different eval_samples count")
            print(f"     - Numerical precision issues")


if __name__ == "__main__":
    main()
