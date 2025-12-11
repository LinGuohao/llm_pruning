import torch
import os
import sys
import argparse
from transformers import LlamaForCausalLM, AutoTokenizer
from typing import List, Dict

# Ensure genetic_pruning and modeling can be imported
# This makes sure that Python can find genetic_pruning.py and modeling.py
# when running this script directly.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from genetic_pruning import GeneticPruning, Individual
# from modeling import DecoupledLlamaModel # Not directly used here, but good to import if needed

def main():
    parser = argparse.ArgumentParser(description="Evaluate initial population of Genetic Algorithm.")
    parser.add_argument("--model_path", type=str, 
                        default="/gpfs/volcano/models/meta-llama/Llama-2-13b-hf",
                        help="Path to the Llama-2-13b model. Update this to your local path.")
    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3').")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Sequence length for PPL evaluation.")
    parser.add_argument("--eval_samples", type=int, default=128,
                        help="Number of samples from WikiText2 to use for evaluation (default: 128).")
    parser.add_argument("--population_size", type=int, default=20,
                        help="Size of the initial population.")
    parser.add_argument("--max_param_ratio", type=float, default=0.5,
                        help="Maximum parameter ratio allowed for pruned models (default: 0.5 for 50%).")
    parser.add_argument("--max_loop_count", type=int, default=2,
                        help="Maximum loop count for modules (default: 2).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    
    args = parser.parse_args()

    # Parse GPU IDs
    raw_gpu_ids_str = [g.strip() for g in args.gpu_ids.split(',') if g.strip()]
    if not raw_gpu_ids_str:
        print("No GPU IDs specified. Falling back to CPU evaluation.")
        device_map_keys = ["cpu"]
    else:
        # User wants to specify physical GPU IDs directly (e.g., "cuda:1", "cuda:5")
        device_map_keys = [f"cuda:{g_id}" for g_id in raw_gpu_ids_str]
    
    print(f"Using devices for evaluation: {device_map_keys}")

    # Load Tokenizer
    print(f"Loading tokenizer from {args.model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Please ensure the model path is correct and accessible.")
        return

    # Load Model Map (Each GPU gets a copy)
    model_map: Dict[str, LlamaForCausalLM] = {}
    for dev_id in device_map_keys:
        if dev_id == "cpu": # Handle CPU case separately
            print(f"Loading Llama-2-13b on CPU...")
        else:
            print(f"Loading Llama-2-13b on {dev_id}...")
        try:
            model = LlamaForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.float16,
                local_files_only=True,
                device_map=dev_id, # Load directly to the specified device
                low_cpu_mem_usage=True
            )
            model.eval()
            model_map[dev_id] = model
            print(f"✓ Model loaded on {dev_id}.")
        except Exception as e:
            print(f"Error loading model on {dev_id}: {e}")
            print(f"Skipping {dev_id}. Ensure model path is correct and device is available/has enough memory.")
            continue
    
    if not model_map:
        print("No models were successfully loaded. Exiting.")
        return

    # Initialize GeneticPruning
    print("\nInitializing GeneticPruning engine...")
    ga = GeneticPruning(
        model_map=model_map,
        tokenizer=tokenizer,
        seqlen=args.seqlen,
        population_size=args.population_size,
        max_param_ratio=args.max_param_ratio,
        max_loop_count=args.max_loop_count,
        eval_samples=args.eval_samples,
        seed=args.seed
    )
    print("✓ GeneticPruning engine initialized.")

    # Generate Initial Population
    print("\nGenerating initial population...")
    initial_population = ga.initialize_population()
    print("✓ Initial population generated.")

    # Evaluate Initial Population
    print("\nEvaluating initial population (in parallel)...")
    evaluated_population = ga.evaluate_fitness_batch_parallel(initial_population)
    print("✓ Initial population evaluation complete.")

    # Print Results
    print(f"\n{'='*80}")
    print(f"Initial Population Evaluation Results ({len(evaluated_population)} individuals)")
    print(f"{'='*80}")

    # Sort by fitness (PPL) for better readability
    evaluated_population.sort(key=lambda ind: ind.fitness)

    for i, ind in enumerate(evaluated_population):
        gene_counts = {}
        for gene in ind.chromosome:
            gene_counts[gene] = gene_counts.get(gene, 0) + 1

        print(f"Individual {i+1:02d} (Fitness: {ind.fitness:.4f}):")
        print(f"  Valid: {ind.is_valid}")
        print(f"  Params Ratio: {ind.params_ratio:.2%}")
        print(f"  Effective Depth: {ind.effective_depth}")
        print(f"  Unique Modules: {ind.num_modules}")
        print(f"  Gene Distribution: {gene_counts}")
        print(f"  Chromosome Start: {ind.chromosome[:15]}...")
        print("-" * 40)

    print(f"\n{'='*80}")
    print("Test Finished.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
