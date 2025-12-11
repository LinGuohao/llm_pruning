import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Union
from dataclasses import dataclass
import copy
import random
import json
import threading

def decode_chromosome(chromosome: List[int]) -> List[int]:
    """
    Decode a loop-encoded chromosome into an execution path.

    Encoding rules (updated for multi-value support):
    - 0 = skip
    - 1 = execute once (no loop)
    - 2+ = execute N times (participate in loop or standalone)
    - Consecutive identical values form a "loop block" that repeats together
    - Different loop count values create separate blocks

    Args:
        chromosome: List[int], values in {0,1,2,...,max_loop_count}

    Returns:
        path: List[int], execution path (module indices in order)

    Example:
        [1,1,2,2,3,3,2,2] → [0,1, 2,3,2,3, 4,5,4,5,4,5, 6,7,6,7]
        - Block 1: modules 0,1 (execute 1x)
        - Block 2: modules 2,3 (execute 2x)
        - Block 3: modules 4,5 (execute 3x)
        - Block 4: modules 6,7 (execute 2x)
    """
    path = []
    i = 0

    while i < len(chromosome):
        # Skip 0
        if chromosome[i] == 0:
            i += 1
            continue

        # Execute once (value == 1)
        if chromosome[i] == 1:
            path.append(i)
            i += 1
            continue

        # Loop block - find consecutive identical values
        if chromosome[i] >= 2:
            current_loop_value = chromosome[i]
            block_modules = []

            # Find all consecutive positions with the exact same loop value
            while i < len(chromosome) and chromosome[i] == current_loop_value:
                block_modules.append(i)
                i += 1

            # Execute this specific loop block (current_loop_value) times
            for _ in range(current_loop_value):
                path.extend(block_modules)

            # Note: If there are more positions with different loop values,
            # they will be handled in separate blocks in the next iteration

    return path


@dataclass
class Individual:
    """Individual in genetic algorithm population."""
    chromosome: List[int]  # Ternary mask for 80 modules [0/1/2, 0/1/2, ...]
    fitness: float = float('inf')  # PPL (lower is better)
    params_ratio: float = 0.0  # Parameter ratio compared to original (based on unique modules)
    is_valid: bool = False  # Whether satisfies parameter constraint
    num_modules: int = 0  # Number of unique selected modules (>0)
    effective_depth: int = 0  # Effective depth after loop expansion

    def __repr__(self):
        return f"Individual(fitness={self.fitness:.2f}, params={self.params_ratio:.2%}, modules={self.num_modules}/80, depth={self.effective_depth}, valid={self.is_valid})"


class GeneticPruning:
    """
    Genetic algorithm for module selection pruning.
    Optimized for Llama-13B (40 layers, 80 modules).
    """

    def __init__(
        self,
        model: nn.Module,
        population_size: int = 20,
        max_param_ratio: float = 0.5,
        max_loop_count: int = 2,
        seed: int = 42
    ):
        self.model = model
        self.population_size = population_size
        self.max_param_ratio = max_param_ratio
        self.max_loop_count = max_loop_count
        
        # Hardcoded for Llama-13B structure
        self.num_layers = 40 
        self.num_modules = 80  # 40 Attention + 40 FFN
        
        # Calculate parameters (needed for ratio constraint)
        self.fixed_params, self.module_params, self.original_params = self._analyze_model_params(model)

        print(f"Genetic Pruning Initialized (Llama-13B Mode):")
        print(f"  Total modules: {self.num_modules}")
        print(f"  Target param ratio: ≤{max_param_ratio:.0%}")
        print(f"  Max loop count: {max_loop_count}")
        print(f"  Population size: {population_size}")

    def _analyze_model_params(self, model: nn.Module) -> Tuple[int, List[int], int]:
        """
        Analyze model parameters to calculate ratios.
        Assumes HF LlamaForCausalLM structure.
        """
        # Fixed params: Embeddings + Norm + Head
        # Note: Depending on implementation, embed_tokens might be tied to lm_head.
        # We count them as fixed costs.
        fixed_params = sum(p.numel() for p in model.model.embed_tokens.parameters())
        fixed_params += sum(p.numel() for p in model.model.norm.parameters())
        fixed_params += sum(p.numel() for p in model.lm_head.parameters())
        
        module_params = []
        for layer in model.model.layers:
            # Attention params (Attention + Input LayerNorm)
            attn_params = sum(p.numel() for p in layer.self_attn.parameters())
            attn_params += sum(p.numel() for p in layer.input_layernorm.parameters())
            module_params.append(attn_params)

            # FFN params (MLP + Post Attention LayerNorm)
            ffn_params = sum(p.numel() for p in layer.mlp.parameters())
            ffn_params += sum(p.numel() for p in layer.post_attention_layernorm.parameters())
            module_params.append(ffn_params)
            
        original_params = sum(p.numel() for p in model.parameters())
        
        # Verification for Llama-13B structure
        if len(module_params) != self.num_modules:
             print(f"Warning: Model has {len(module_params)} modules, expected {self.num_modules} for Llama-13B.")

        return fixed_params, module_params, original_params

    def _calculate_params_ratio(self, chromosome: List[int]) -> float:
        """Calculate parameter ratio based on unique modules selected."""
        # Only count parameters for modules that have value > 0
        # Loops do NOT add extra parameter cost (weight sharing)
        selected_params = sum(
            self.module_params[i] for i, value in enumerate(chromosome) if value > 0
        )
        total_params = self.fixed_params + selected_params
        return total_params / self.original_params

    def _repair_chromosome(self, chromosome: List[int]) -> List[int]:
        """
        Repair chromosome to satisfy max_param_ratio constraint.
        Randomly removes modules until constraint is met.
        """
        chromosome = chromosome[:] # Copy to avoid side effects
        
        # Safety: Ensure at least one module is selected so model isn't empty
        if sum(chromosome) == 0:
            chromosome[0] = 1
            
        # Randomly drop modules if ratio exceeded
        while self._calculate_params_ratio(chromosome) > self.max_param_ratio:
            selected_indices = [i for i, v in enumerate(chromosome) if v > 0]
            if len(selected_indices) <= 1:
                break 
            
            drop_idx = random.choice(selected_indices)
            chromosome[drop_idx] = 0
            
        return chromosome

    def initialize_population(self) -> List[Individual]:
        """
        Initialize population with diverse strategies.
        Strategies tailored for 80-module Llama-13B structure.
        """
        population = []

        # Strategy 1: Full Model (Baseline)
        # Usually full model is > max_ratio, so this will effectively produce 
        # a "maximally filled" individual satisfying the constraint.
        chromosome = [1] * self.num_modules
        chromosome = self._repair_chromosome(chromosome)
        population.append(Individual(chromosome=chromosome))

        # Strategy 2: Strided Pruning (Uniform Sparsity)
        # Keep every 2nd, 3rd, 4th, 5th module
        for k in [2, 3, 4, 5]:
            chromosome = [1 if i % k == 0 else 0 for i in range(self.num_modules)]
            chromosome = self._repair_chromosome(chromosome)
            population.append(Individual(chromosome=chromosome))

        # Strategy 3: Structure-Aware Pruning (Attention vs FFN)
        # Since we have 80 modules, Evens=Attention, Odds=FFN
        
        # Sub-strategy 3A: Keep Attention, Sample FFN
        for keep_ffn_ratio in [0.2, 0.4, 0.6]:
            chromosome = []
            for i in range(self.num_modules):
                if i % 2 == 0: # Attention
                    # Mostly keep Attn, occasional loop
                    val = random.randint(1, self.max_loop_count) if random.random() < 0.2 else 1
                    chromosome.append(val)
                else: # FFN
                    if random.random() < keep_ffn_ratio:
                        val = random.randint(1, self.max_loop_count) if random.random() < 0.2 else 1
                        chromosome.append(val)
                    else:
                        chromosome.append(0)
            chromosome = self._repair_chromosome(chromosome)
            population.append(Individual(chromosome=chromosome))
            
        # Sub-strategy 3B: Keep FFN, Sample Attention
        for keep_attn_ratio in [0.2, 0.4, 0.6]:
            chromosome = []
            for i in range(self.num_modules):
                if i % 2 == 0: # Attention
                    if random.random() < keep_attn_ratio:
                        val = random.randint(1, self.max_loop_count) if random.random() < 0.2 else 1
                        chromosome.append(val)
                    else:
                        chromosome.append(0)
                else: # FFN
                     # Mostly keep FFN
                     val = random.randint(1, self.max_loop_count) if random.random() < 0.2 else 1
                     chromosome.append(val)
            chromosome = self._repair_chromosome(chromosome)
            population.append(Individual(chromosome=chromosome))

        # Strategy 4: Random Density Fill
        # Fill remaining population slots with randomized chromosomes
        while len(population) < self.population_size:
            chromosome = []
            for _ in range(self.num_modules):
                r = random.random()
                # Bias towards sparsity (0) to hit constraints faster
                if r < 0.5: # 50% Skip
                    chromosome.append(0)
                elif r < 0.8: # 30% Keep Once
                    chromosome.append(1)
                else: # 20% Loop (if max_loop >= 2)
                    chromosome.append(random.randint(2, max(2, self.max_loop_count)))
            
            chromosome = self._repair_chromosome(chromosome)
            population.append(Individual(chromosome=chromosome))

        print(f"✓ Initialized population of {len(population)} individuals")
        return population