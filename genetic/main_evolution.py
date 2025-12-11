import torch
import os
import sys
import argparse
from transformers import LlamaForCausalLM, AutoTokenizer
from typing import List, Dict
import time

# Ensure genetic_pruning and modeling can be imported
current_dir = os.path.dirname(os.path.abspath(__file__)) # This is genetic/
project_root = os.path.dirname(current_dir) # This is llm_pruning/

sys.path.append(project_root) # Add project root to sys.path
sys.path.append(current_dir)  # Add genetic/ directory to sys.path (for direct imports like modeling.py if needed, though project_root handles genetic.modeling)

from genetic.genetic_pruning import GeneticPruning, Individual

def main():
    parser = argparse.ArgumentParser(description="Run Genetic Algorithm for Llama-13B Pruning.")
    
    # Model and Data
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the Llama-2-13b model.")
    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="Comma-separated list of physical GPU IDs to use (e.g., '0,1,2,3').")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Sequence length for PPL evaluation.")
    parser.add_argument("--eval_samples", type=int, default=128,
                        help="Number of samples from WikiText2 to use for evaluation.")
    
    # Genetic Algorithm Parameters
    parser.add_argument("--population_size", type=int, default=20,
                        help="Size of the population.")
    parser.add_argument("--max_generations", type=int, default=50,
                        help="Maximum number of generations to run.")
    parser.add_argument("--mutation_rate", type=float, default=0.05,
                        help="Probability of mutation for each gene.")
    parser.add_argument("--crossover_rate", type=float, default=0.8,
                        help="Probability of crossover between parents.")
    parser.add_argument("--crossover_type", type=str, default="uniform",
                        choices=["uniform", "onepoint", "twopoint"],
                        help="Type of crossover operation.")
    parser.add_argument("--selection_method", type=str, default="tournament",
                        choices=["tournament", "top20", "topNw"],
                        help="Method for selecting parents.")
    parser.add_argument("--tournament_size", type=int, default=3,
                        help="Tournament size for tournament selection.")
    parser.add_argument("--top_percent", type=float, default=0.5,
                        help="Percent of top individuals to consider for topNw selection.")
    
    # Constraints
    parser.add_argument("--max_param_ratio", type=float, default=0.5,
                        help="Maximum parameter ratio allowed (0.5 = 50% of original params).")
    parser.add_argument("--max_loop_count", type=int, default=2,
                        help="Maximum allowable loop count for a module.")
    
    # Checkpointing and Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory to save checkpoints.")
    parser.add_argument("--checkpoint_interval", type=int, default=5,
                        help="Save checkpoint every N generations.")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a checkpoint file to resume from.")
    
    args = parser.parse_args()

    # --- Print Configuration ---
    print("=" * 80)
    print("Genetic Algorithm Evolution Configuration")
    print("=" * 80)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 80)
    time.sleep(1) # Small delay for better readability in console

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Parse GPU IDs
    raw_gpu_ids_str = [g.strip() for g in args.gpu_ids.split(',') if g.strip()]
    if not raw_gpu_ids_str:
        print("No GPU IDs specified. Falling back to CPU.")
        device_map_keys = ["cpu"]
    else:
        # Use physical IDs
        device_map_keys = [f"cuda:{g_id}" for g_id in raw_gpu_ids_str]
    
    print(f"Using devices: {device_map_keys}")

    # Load Tokenizer
    print(f"Loading tokenizer from {args.model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Load Model Map (Copy per GPU)
    print("Loading model copies to GPUs...")
    model_map: Dict[str, LlamaForCausalLM] = {}
    
    # Optimization: Load once to CPU then move to GPUs? 
    # Or load directly to each GPU (might be slower disk IO but safer memory wise if sequential).
    # Loading sequentially to avoid OOM on CPU if loading multiple copies at once.
    for dev_id in device_map_keys:
        print(f"  Loading on {dev_id}...")
        try:
            model = LlamaForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.float16,
                local_files_only=True,
                device_map=dev_id, 
                low_cpu_mem_usage=True
            )
            model.eval()
            model_map[dev_id] = model
        except Exception as e:
            print(f"  Failed to load on {dev_id}: {e}")
            continue
    
    if not model_map:
        print("CRITICAL: No models loaded. Exiting.")
        return

    # Setup Checkpoint Directory
    if args.checkpoint_dir === None:
        # Default: genetic/checkpoints/<timestamp>
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.checkpoint_dir = os.path.join(current_dir, "checkpoints", timestamp)
    
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        print(f"Created checkpoint directory: {args.checkpoint_dir}")

    # Initialize Genetic Algorithm
    print("\nInitializing Genetic Algorithm Engine...")
    ga = GeneticPruning(
        model_map=model_map,
        tokenizer=tokenizer,
        seqlen=args.seqlen,
        population_size=args.population_size,
        max_generations=args.max_generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        crossover_type=args.crossover_type,
        selection_method=args.selection_method,
        tournament_size=args.tournament_size,
        top_percent=args.top_percent,
        max_param_ratio=args.max_param_ratio,
        max_loop_count=args.max_loop_count,
        eval_samples=args.eval_samples,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed
    )

    # Save initial config
    with open(os.path.join(args.checkpoint_dir, "run_config.json"), 'w') as f:
        import json
        json.dump(vars(args), f, indent=2)

    # Run Evolution
    start_time = time.time()
    best_individual = ga.evolve(resume_from=args.resume_from)
    end_time = time.time()

    print(f"\nTotal evolution time: {(end_time - start_time) / 60:.2f} minutes")
    print(f"Best Individual Found: {best_individual}")
    
    # Save best chromosome explicitly
    if best_individual:
        best_path = os.path.join(args.checkpoint_dir, "best_individual.json")
        with open(best_path, 'w') as f:
            json.dump({
                "chromosome": best_individual.chromosome,
                "fitness": best_individual.fitness,
                "params_ratio": best_individual.params_ratio,
                "num_modules": best_individual.num_modules,
                "effective_depth": best_individual.effective_depth
            }, f, indent=2)
        print(f"Best individual saved to {best_path}")

if __name__ == "__main__":
    main()

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Parse GPU IDs
    raw_gpu_ids_str = [g.strip() for g in args.gpu_ids.split(',') if g.strip()]
    if not raw_gpu_ids_str:
        print("No GPU IDs specified. Falling back to CPU.")
        device_map_keys = ["cpu"]
    else:
        # Use physical IDs
        device_map_keys = [f"cuda:{g_id}" for g_id in raw_gpu_ids_str]
    
    print(f"Using devices: {device_map_keys}")

    # Load Tokenizer
    print(f"Loading tokenizer from {args.model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Load Model Map (Copy per GPU)
    print("Loading model copies to GPUs...")
    model_map: Dict[str, LlamaForCausalLM] = {}
    
    # Optimization: Load once to CPU then move to GPUs? 
    # Or load directly to each GPU (might be slower disk IO but safer memory wise if sequential).
    # Loading sequentially to avoid OOM on CPU if loading multiple copies at once.
    for dev_id in device_map_keys:
        print(f"  Loading on {dev_id}...")
        try:
            model = LlamaForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.float16,
                local_files_only=True,
                device_map=dev_id, 
                low_cpu_mem_usage=True
            )
            model.eval()
            model_map[dev_id] = model
        except Exception as e:
            print(f"  Failed to load on {dev_id}: {e}")
            continue
    
    if not model_map:
        print("CRITICAL: No models loaded. Exiting.")
        return

    # Setup Checkpoint Directory
    if args.checkpoint_dir is None:
        # Default: genetic/checkpoints/<timestamp>
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.checkpoint_dir = os.path.join(current_dir, "checkpoints", timestamp)
    
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        print(f"Created checkpoint directory: {args.checkpoint_dir}")

    # Initialize Genetic Algorithm
    print("\nInitializing Genetic Algorithm Engine...")
    ga = GeneticPruning(
        model_map=model_map,
        tokenizer=tokenizer,
        seqlen=args.seqlen,
        population_size=args.population_size,
        max_generations=args.max_generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        crossover_type=args.crossover_type,
        selection_method=args.selection_method,
        tournament_size=args.tournament_size,
        top_percent=args.top_percent,
        max_param_ratio=args.max_param_ratio,
        max_loop_count=args.max_loop_count,
        eval_samples=args.eval_samples,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed
    )

    # Save initial config
    with open(os.path.join(args.checkpoint_dir, "run_config.json"), 'w') as f:
        import json
        json.dump(vars(args), f, indent=2)

    # Run Evolution
    start_time = time.time()
    best_individual = ga.evolve(resume_from=args.resume_from)
    end_time = time.time()

    print(f"\nTotal evolution time: {(end_time - start_time) / 60:.2f} minutes")
    print(f"Best Individual Found: {best_individual}")
    
    # Save best chromosome explicitly
    if best_individual:
        best_path = os.path.join(args.checkpoint_dir, "best_individual.json")
        with open(best_path, 'w') as f:
            json.dump({
                "chromosome": best_individual.chromosome,
                "fitness": best_individual.fitness,
                "params_ratio": best_individual.params_ratio,
                "num_modules": best_individual.num_modules,
                "effective_depth": best_individual.effective_depth
            }, f, indent=2)
        print(f"Best individual saved to {best_path}")

if __name__ == "__main__":
    main()
