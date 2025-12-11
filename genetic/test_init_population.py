import torch
import sys
import os
import argparse
from transformers import LlamaForCausalLM, AutoTokenizer

# Ensure we can import modules from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from genetic_pruning import GeneticPruning, Individual

def test_initialization():
    parser = argparse.ArgumentParser(description="Test Genetic Algorithm Initialization")
    # Default path from SliceGPT reference
    parser.add_argument("--model", type=str, default="/gpfs/volcano/models/meta-llama/Llama-2-13b-hf", 
                        help="Path to Llama-2-13b model")
    args = parser.parse_args()
    
    model_path = args.model

    print(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist.")
        print("Please provide a valid path using --model <path>")
        return

    try:
        # Reference loading logic from SliceGPT
        # using local_files_only=True and float16
        # Added device_map='cpu' to avoid OOM on initialization test if GPU is small
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=True,
            device_map="cpu", 
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("✓ Model loaded successfully.")

    # Initialize Genetic Algorithm
    print("\nInitializing GeneticPruning Engine...")
    try:
        ga = GeneticPruning(
            model=model,
            population_size=20,
            max_param_ratio=0.5,
            max_loop_count=2
        )
    except Exception as e:
        print(f"Error initializing GeneticPruning: {e}")
        import traceback
        traceback.print_exc()
        return

    # Generate Initial Population
    print("\nGenerating Initial Population...")
    population = ga.initialize_population()

    # Print Results
    print(f"\n{'='*60}")
    print(f"Initial Population Summary ({len(population)} individuals)")
    print(f"{'='*60}")

    for i, ind in enumerate(population):
        # Count gene types
        counts = {}
        for gene in ind.chromosome:
            counts[gene] = counts.get(gene, 0) + 1
            
        print(f"Individual {i:02d}:")
        print(f"  Valid: {ind.is_valid}")
        print(f"  Params Ratio: {ind.params_ratio:.2%}")
        print(f"  Gene Distribution: {counts}")
        print(f"  Chromosome Start: {ind.chromosome[:15]}...")
        print("-" * 40)

if __name__ == "__main__":
    test_initialization()
