import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Union
from dataclasses import dataclass
import copy
import random
import json
import threading
import numpy as np
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import from our new modeling module
from modeling import decode_chromosome, DecoupledLlamaModel

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
        model_map: Dict[str, nn.Module], # Changed: Accept a map of device -> model instance
        tokenizer,
        seqlen: int = 2048,
        population_size: int = 20,
        max_generations: int = 50,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.8,
        crossover_type: str = "uniform",
        selection_method: str = "tournament",
        tournament_size: int = 3,
        top_percent: float = 0.5,
        max_param_ratio: float = 0.5,
        max_loop_count: int = 2,
        eval_samples: int = 128, 
        checkpoint_dir: str = None,
        checkpoint_interval: int = 10,
        seed: int = 42
    ):
        if not model_map:
            raise ValueError("model_map cannot be empty")
        
        self.model_map = model_map
        self.devices = list(model_map.keys())
        self.device = self.devices[0] # Primary device (e.g. for main thread ops)
        self.primary_model = self.model_map[self.device]
        
        self.use_multi_gpu = len(self.devices) > 1
        
        self.tokenizer = tokenizer
        self.seqlen = seqlen
        
        # Load and prepare WikiText2 data exactly like SliceGPT
        print("Loading WikiText2 validation/test data for evaluation...")
        self.eval_data = self._get_wikitext2_data(tokenizer, seqlen)
        print(f"Prepared {self.eval_data.shape[0]} samples of length {seqlen}")

        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.crossover_type = crossover_type
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.top_percent = top_percent
        self.max_param_ratio = max_param_ratio
        self.max_loop_count = max_loop_count
        self.eval_samples = min(eval_samples, self.eval_data.shape[0])

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval

        # Hardcoded for Llama-13B structure
        self.num_layers = 40 
        self.num_modules = 80  # 40 Attention + 40 FFN
        
        # Calculate parameters (needed for ratio constraint) using primary model
        self.fixed_params, self.module_params, self.original_params = self._analyze_model_params(self.primary_model)

        # Cache for evaluated chromosomes (thread-safe)
        self.evaluated_cache = {}
        self.cache_lock = threading.Lock()

        print(f"Genetic Pruning Initialized (Llama-13B Mode):")
        print(f"  Total modules: {self.num_modules}")
        print(f"  Target param ratio: ≤{max_param_ratio:.0%}")
        print(f"  Max loop count: {max_loop_count}")
        print(f"  Population size: {population_size}")
        print(f"  Max generations: {max_generations}")
        print(f"  Mutation rate: {mutation_rate}")
        print(f"  Crossover rate: {crossover_rate}")
        print(f"  Devices: {self.devices}")
        if self.use_multi_gpu:
            print(f"  Multi-GPU: ENABLED ({len(self.devices)} GPUs, parallel evaluation)")

    def _get_wikitext2_data(self, tokenizer, seqlen):
        """
        Load and prepare WikiText2 dataset.
        Identical logic to SliceGPT/eval_model_ppl.py: get_wikitext2 + reshaping
        """
        # Load dataset (using 'test' split as per SliceGPT example, 
        # but for evolution usually 'validation' is safer to avoid overfitting to test.
        # However, to be "strictly consistent" with the reference file provided:
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        
        # Join all text
        text = "\n\n".join(testdata['text'])
        
        # Tokenize
        testenc = tokenizer(text, return_tensors='pt')
        
        # Reshape into chunks of seqlen
        nsamples = testenc.input_ids.numel() // seqlen
        input_ids = testenc.input_ids[0, :nsamples * seqlen]
        input_ids = input_ids.reshape(nsamples, seqlen)
        
        return input_ids

    def _analyze_model_params(self, model: nn.Module) -> Tuple[int, List[int], int]:
        """
        Analyze model parameters to calculate ratios.
        Assumes HF LlamaForCausalLM structure.
        """
        # Fixed params: Embeddings + Norm + Head
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
        chromosome = [1] * self.num_modules
        chromosome = self._repair_chromosome(chromosome)
        population.append(Individual(chromosome=chromosome))

        # Strategy 2: Strided Pruning (Uniform Sparsity)
        for k in [2, 3, 4, 5]:
            chromosome = [1 if i % k == 0 else 0 for i in range(self.num_modules)]
            chromosome = self._repair_chromosome(chromosome)
            population.append(Individual(chromosome=chromosome))

        # Strategy 3: Structure-Aware Pruning (Attention vs FFN)
        # Evens=Attention, Odds=FFN
        
        # Sub-strategy 3A: Keep Attention, Sample FFN
        for keep_ffn_ratio in [0.2, 0.4, 0.6]:
            chromosome = []
            for i in range(self.num_modules):
                if i % 2 == 0: # Attention
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
                     val = random.randint(1, self.max_loop_count) if random.random() < 0.2 else 1
                     chromosome.append(val)
            chromosome = self._repair_chromosome(chromosome)
            population.append(Individual(chromosome=chromosome))

        # Strategy 4: Random Density Fill
        while len(population) < self.population_size:
            chromosome = []
            for _ in range(self.num_modules):
                r = random.random()
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

    def _evaluate_ppl_on_device(self, chromosome: List[int], device: str) -> float:
        """
        Evaluate perplexity for a chromosome on a specific device using DecoupledLlamaModel.
        Strictly follows logic from SliceGPT/eval_model_ppl.py (evaluate_ppl_simple).
        """
        try:
            # Get the model for this specific device
            original_model = self.model_map[device]
            decoupled_model = DecoupledLlamaModel(
                original_model,
                chromosome
            )
            decoupled_model.eval()
        except Exception as e:
            print(f"    Error building DecoupledLlamaModel on {device}: {e}")
            torch.cuda.empty_cache()
            raise

        loss_fct = nn.CrossEntropyLoss().to(device)
        acc_loss = 0.0
        
        # Use a subset of samples if specified, otherwise full data
        nsamples = self.eval_samples
        
        # Get data subset
        input_ids_all = self.eval_data[:nsamples]

        with torch.no_grad():
            for i in range(nsamples):
                try:
                    # Get single sample (1, seqlen)
                    input_ids = input_ids_all[i:i+1].to(device)

                    # Forward pass
                    # Note: DecoupledLlamaModel forward returns logits directly
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

                except Exception as e:
                    print(f"    Error in evaluation sample {i} on {device}: {e}")
                    raise

        # Compute average loss and PPL
        # Logic: avg_loss = acc_loss / nsamples; PPL = exp(avg_loss)
        avg_loss = acc_loss / nsamples
        ppl = np.exp(avg_loss)

        # Cleanup
        del decoupled_model
        torch.cuda.empty_cache()

        return float(ppl)

    def evaluate_fitness(self, individual: Individual) -> Individual:
        """
        Evaluate fitness (PPL) for an individual.
        Updates individual's fitness, params_ratio, is_valid, num_modules, effective_depth.
        """
        chromosome_tuple = tuple(individual.chromosome)
        with self.cache_lock:
            if chromosome_tuple in self.evaluated_cache:
                cached = self.evaluated_cache[chromosome_tuple]
                individual.fitness = cached['fitness']
                individual.params_ratio = cached['params_ratio']
                individual.is_valid = cached['is_valid']
                individual.num_modules = cached['num_modules']
                individual.effective_depth = cached['effective_depth']
                return individual

        # Calculate stats (num_modules, effective_depth)
        unique_modules_count = sum(1 for v in individual.chromosome if v > 0)
        execution_path = decode_chromosome(individual.chromosome)
        effective_depth = len(execution_path)

        individual.num_modules = unique_modules_count
        individual.effective_depth = effective_depth
        
        # Calculate params ratio
        params_ratio = self._calculate_params_ratio(individual.chromosome)
        individual.params_ratio = params_ratio

        # Check constraint and evaluate PPL if valid
        if params_ratio > self.max_param_ratio or unique_modules_count == 0:
            individual.is_valid = False
            individual.fitness = float('inf') # Invalid individuals get infinite PPL
        else:
            individual.is_valid = True
            try:
                ppl = self._evaluate_ppl_on_device(individual.chromosome, self.device)
                individual.fitness = ppl
            except Exception as e:
                print(f"    ⚠️  Evaluation failed for chromosome {individual.chromosome[:10]}...: {e}")
                individual.fitness = float('inf')
                individual.is_valid = False

        # Cache result
        with self.cache_lock:
            self.evaluated_cache[chromosome_tuple] = {
                'fitness': individual.fitness,
                'params_ratio': individual.params_ratio,
                'is_valid': individual.is_valid,
                'num_modules': individual.num_modules,
                'effective_depth': individual.effective_depth
            }

        return individual

    def evaluate_fitness_batch_parallel(self, individuals: List[Individual]) -> List[Individual]:
        """
        Evaluate fitness for multiple individuals in parallel across GPUs.
        """
        if not self.use_multi_gpu or len(individuals) == 0:
            # Fallback to sequential if single GPU or empty list
            return [self.evaluate_fitness(ind) for ind in individuals]

        # Use ThreadPoolExecutor to run evaluations in parallel
        # Each thread gets a dedicated GPU (round-robin assignment)
        num_gpus = len(self.devices)
        
        # We need to explicitly deepcopy the original_model for each worker
        # if the original_model is not already on a device that can be shared.
        # However, DecoupledLlamaModel is designed to reference.
        # For true parallel PPL evaluation on different GPUs, each worker usually needs
        # its own model instance on its dedicated device.
        # But DecoupledLlamaModel references *sub-modules* of original_model.
        # This implies original_model's sub-modules are already on different devices OR
        # DecoupledLlamaModel is created on each device and fetches the sub-modules to that device.

        # The current DecoupledLlamaModel implementation assumes original_model's sub-modules
        # are managed externally (e.g., already `to(device)`).
        # For multi-GPU, the typical pattern is to put the *original_model* onto `device_map="auto"`
        # or load it per device.
        # Here, we will make `_evaluate_ppl_on_device` handle the `to(device)` for inputs,
        # and `DecoupledLlamaModel` will implicitly use the original model's components as they are.

        # Separating cached and uncached individuals to avoid redundant computation
        uncached_individuals = []
        # Store (individual, original_index) tuples
        uncached_with_indices = []

        for idx, individual in enumerate(individuals):
            chromosome_tuple = tuple(individual.chromosome)
            with self.cache_lock:
                if chromosome_tuple in self.evaluated_cache:
                    # Load from cache
                    cached = self.evaluated_cache[chromosome_tuple]
                    individual.fitness = cached['fitness']
                    individual.params_ratio = cached['params_ratio']
                    individual.is_valid = cached['is_valid']
                    individual.num_modules = cached['num_modules']
                    individual.effective_depth = cached['effective_depth']
                else:
                    uncached_individuals.append(individual)
                    uncached_with_indices.append((individual, idx))

        # If all individuals are cached, return the list as is
        if not uncached_individuals:
            return individuals

        # Now evaluate uncached ones
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = {}
            for i, (individual, original_idx) in enumerate(uncached_with_indices):
                device = self.devices[i % num_gpus] # Round-robin assignment
                future = executor.submit(self._evaluate_single_individual_for_parallel, individual, device)
                futures[future] = original_idx # Map future to original index

            for future in as_completed(futures):
                evaluated_individual = future.result()
                original_idx = futures[future]
                # Update the original list at its correct position
                individuals[original_idx] = evaluated_individual

        return individuals

    def _evaluate_single_individual_for_parallel(self, individual: Individual, device: str) -> Individual:
        """Helper to evaluate a single individual, to be run in a thread."""
        chromosome_tuple = tuple(individual.chromosome)
        
        # Recalculate stats as they might not be set for uncached individuals
        unique_modules_count = sum(1 for v in individual.chromosome if v > 0)
        execution_path = decode_chromosome(individual.chromosome)
        effective_depth = len(execution_path)

        individual.num_modules = unique_modules_count
        individual.effective_depth = effective_depth

        params_ratio = self._calculate_params_ratio(individual.chromosome)
        individual.params_ratio = params_ratio

        if params_ratio > self.max_param_ratio or unique_modules_count == 0:
            individual.is_valid = False
            individual.fitness = float('inf')
        else:
            individual.is_valid = True
            try:
                ppl = self._evaluate_ppl_on_device(individual.chromosome, device)
                individual.fitness = ppl
            except Exception as e:
                print(f"    ⚠️  Evaluation failed for chromosome {individual.chromosome[:10]}... on {device}: {e}")
                individual.fitness = float('inf')
                individual.is_valid = False
        
        # Cache result (thread-safe)
        with self.cache_lock:
            self.evaluated_cache[chromosome_tuple] = {
                'fitness': individual.fitness,
                'params_ratio': individual.params_ratio,
                'is_valid': individual.is_valid,
                'num_modules': individual.num_modules,
                'effective_depth': individual.effective_depth
            }
        return individual

    def tournament_selection(self, population: List[Individual]) -> Individual:
        """
        Selects an individual using tournament selection.
        A specified number of individuals are randomly chosen from the population,
        and the one with the best fitness (lowest PPL) is selected.
        """
        if len(population) < self.tournament_size:
            raise ValueError(f"Population size ({len(population)}) is smaller than tournament size ({self.tournament_size}).")
        
        tournament = random.sample(population, self.tournament_size)
        # Select the winner: lowest fitness among valid individuals, or inf if invalid
        winner = min(tournament, key=lambda ind: ind.fitness if ind.is_valid else float('inf'))
        return winner

    def select_two_different_parents(self, population: List[Individual], percent: float = 0.2) -> Tuple[Individual, Individual]:
        """
        Selects two different parents from the top 'percent' of the population
        based on fitness.
        """
        # Get valid individuals and sort by fitness
        valid_pop = [ind for ind in population if ind.is_valid]
        if len(valid_pop) < 2:
            # If not enough valid individuals, fall back to general tournament selection for two
            print("Warning: Not enough valid individuals for top-percent selection. Falling back to general tournament.")
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            while parent1 == parent2: # Ensure they are different
                parent2 = self.tournament_selection(population)
            return parent1, parent2

        sorted_pop = sorted(valid_pop, key=lambda ind: ind.fitness)

        # Select from top percent, ensuring at least 2 individuals if possible
        top_n = max(2, int(len(sorted_pop) * percent))
        top_individuals = sorted_pop[:top_n]

        # Randomly choose two different individuals from the top group
        if len(top_individuals) < 2:
            # Should not happen if valid_pop >= 2 and top_n >= 2, but as a safeguard
            print(f"Warning: Only {len(top_individuals)} individuals in top {percent*100}% group. Cannot select two distinct parents.")
            # Fallback for extreme cases
            parent1 = top_individuals[0]
            parent2 = copy.deepcopy(parent1) # Return two copies of the same if no alternative
            return parent1, parent2
            
        parent1, parent2 = random.sample(top_individuals, 2)
        return parent1, parent2

    def select_weighted_parents(self, population: List[Individual], percent: float = 0.6) -> Tuple[Individual, Individual]:
        """
        Selects two different parents from the top 'percent' of the population
        using a weighted probability distribution based on fitness.
        Higher fitness (lower PPL) individuals have a higher chance of being selected.
        """
        valid_pop = [ind for ind in population if ind.is_valid]
        if len(valid_pop) < 2:
            print("Warning: Not enough valid individuals for weighted selection. Falling back to general selection.")
            return self.select_two_different_parents(population, percent=1.0) # Select from all valid

        sorted_pop = sorted(valid_pop, key=lambda ind: ind.fitness)

        top_n = max(2, int(len(sorted_pop) * percent))
        top_individuals = sorted_pop[:top_n]

        if len(top_individuals) < 2:
            print(f"Warning: Only {len(top_individuals)} individuals in top {percent*100}% group for weighted selection.")
            parent1 = top_individuals[0]
            parent2 = copy.deepcopy(parent1)
            return parent1, parent2

        # Calculate weights: simple rank-based weighting (e.g., higher rank = higher weight)
        # Using a linear rank-based weight: best gets N, next gets N-1, ..., worst gets 1
        weights = [top_n - i for i in range(top_n)]

        # Select first parent
        parent1 = random.choices(top_individuals, weights=weights, k=1)[0]

        # Select second parent, ensuring it's different and recalculating weights
        remaining_individuals = [ind for ind in top_individuals if ind != parent1]
        if not remaining_individuals: # Should not happen if len(top_individuals) >= 2
             parent2 = copy.deepcopy(parent1)
             return parent1, parent2

        remaining_weights = [weights[top_individuals.index(ind)] for ind in remaining_individuals]

        parent2 = random.choices(remaining_individuals, weights=remaining_weights, k=1)[0]

        return parent1, parent2

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Crossover with repair.

        Supports multiple crossover types:
        - 'uniform': Uniform crossover (each gene independently 50% from each parent)
        - 'onepoint': Single-point crossover (one random cut point)
        - 'twopoint': Two-point crossover (two random cut points)
        """
        # Check crossover probability
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        chrom_len = len(parent1.chromosome)

        if self.crossover_type == 'uniform':
            # Uniform crossover: each gene independently 50% from each parent
            child1_chromosome = []
            child2_chromosome = []

            for gene1, gene2 in zip(parent1.chromosome, parent2.chromosome):
                if random.random() < 0.5:
                    child1_chromosome.append(gene1)
                    child2_chromosome.append(gene2)
                else:
                    child1_chromosome.append(gene2)
                    child2_chromosome.append(gene1)

        elif self.crossover_type == 'onepoint':
            # Single-point crossover
            if chrom_len < 2: # Cannot cut if length < 2
                child1_chromosome = parent1.chromosome[:]
                child2_chromosome = parent2.chromosome[:]
            else:
                cut_point = random.randint(1, chrom_len - 1)
                child1_chromosome = parent1.chromosome[:cut_point] + parent2.chromosome[cut_point:]
                child2_chromosome = parent2.chromosome[:cut_point] + parent1.chromosome[cut_point:]

        elif self.crossover_type == 'twopoint':
            # Two-point crossover
            if chrom_len < 3: # Cannot make 2 cuts if length < 3
                child1_chromosome = parent1.chromosome[:]
                child2_chromosome = parent2.chromosome[:]
            else:
                # Ensure point1 < point2
                point1 = random.randint(1, chrom_len - 2)
                point2 = random.randint(point1 + 1, chrom_len - 1)

                # Swap the middle segment
                child1_chromosome = (parent1.chromosome[:point1] +
                                    parent2.chromosome[point1:point2] +
                                    parent1.chromosome[point2:])
                child2_chromosome = (parent2.chromosome[:point1] +
                                    parent1.chromosome[point1:point2] +
                                    parent2.chromosome[point2:])
        else:
            raise ValueError(f"Unknown crossover type: {self.crossover_type}. Must be 'uniform', 'onepoint', or 'twopoint'.")

        # Repair chromosomes to ensure they meet constraints (or at least try to)
        child1_chromosome = self._repair_chromosome(child1_chromosome)
        child2_chromosome = self._repair_chromosome(child2_chromosome)

        child1 = Individual(chromosome=child1_chromosome)
        child2 = Individual(chromosome=child2_chromosome)

        return child1, child2

    def mutate(self, individual: Individual) -> Individual:
        """
        Multi-value mutation with gradual transitions (loop-aware).

        Mutation rules (gradual transitions only, no jumps > 1):
        - 0 -> 1 (can only increase to 1)
        - 1 -> 0 or 2 (can go either direction)
        - 2~(max_loop_count-1) -> value-1 or value+1
        - max_loop_count -> max_loop_count-1 (can only decrease by 1)
        """
        mutated = copy.deepcopy(individual)

        for i in range(len(mutated.chromosome)):
            if random.random() < self.mutation_rate:
                current_value = mutated.chromosome[i]

                if current_value == 0:
                    # 0 can only mutate to 1
                    mutated.chromosome[i] = 1
                elif current_value == 1:
                    # 1 can mutate to 0 or 2 (50/50)
                    mutated.chromosome[i] = random.choice([0, 2])
                elif current_value == self.max_loop_count:
                    # max_loop_count can only decrease to max_loop_count-1
                    mutated.chromosome[i] = self.max_loop_count - 1
                else:
                    # Middle values can increase or decrease by 1
                    mutated.chromosome[i] = random.choice([current_value - 1, current_value + 1])

        # Repair to satisfy constraints
        mutated.chromosome = self._repair_chromosome(mutated.chromosome)

        return mutated

    def save_checkpoint(self, generation: int, population: List[Individual], best_ever: Individual):
        """Save checkpoint to resume training."""
        if not self.checkpoint_dir:
            return

        checkpoint = {
            'generation': generation,
            'population': [
                {
                    'chromosome': ind.chromosome,
                    'fitness': ind.fitness,
                    'params_ratio': ind.params_ratio,
                    'is_valid': ind.is_valid,
                    'num_modules': ind.num_modules,
                    'effective_depth': ind.effective_depth
                }
                for ind in population
            ],
            'best_ever': {
                'chromosome': best_ever.chromosome,
                'fitness': best_ever.fitness,
                'params_ratio': best_ever.params_ratio,
                'is_valid': best_ever.is_valid,
                'num_modules': best_ever.num_modules,
                'effective_depth': best_ever.effective_depth
            },
            'evaluated_cache': {
                str(k): v for k, v in self.evaluated_cache.items()
            },
            'config': {
                'population_size': self.population_size,
                'max_generations': self.max_generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'crossover_type': self.crossover_type,
                'selection_method': self.selection_method,
                'tournament_size': self.tournament_size,
                'top_percent': self.top_percent,
                'max_param_ratio': self.max_param_ratio,
                'max_loop_count': self.max_loop_count,
                'num_modules': self.num_modules,
            }
        }

        # Create directory if it doesn't exist
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        checkpoint_path = f"{self.checkpoint_dir}/checkpoint_gen{generation}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        print(f"  💾 Checkpoint saved: {checkpoint_path}")

        # Also save as latest
        latest_path = f"{self.checkpoint_dir}/checkpoint_latest.json"
        with open(latest_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        print(f"\\n📂 Loading checkpoint from: {checkpoint_path}")

        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)

        # Restore population
        population = []
        for ind_data in checkpoint['population']:
            ind = Individual(
                chromosome=ind_data['chromosome'],
                fitness=ind_data['fitness'],
                params_ratio=ind_data['params_ratio'],
                is_valid=ind_data['is_valid'],
                num_modules=ind_data['num_modules'],
                effective_depth=ind_data.get('effective_depth', 0)
            )
            population.append(ind)

        # Restore best ever
        best_data = checkpoint['best_ever']
        best_ever = Individual(
            chromosome=best_data['chromosome'],
            fitness=best_data['fitness'],
            params_ratio=best_data['params_ratio'],
            is_valid=best_data['is_valid'],
            num_modules=best_data['num_modules'],
            effective_depth=best_data.get('effective_depth', 0)
        )

        # Restore cache
        # Keys in JSON are strings, convert back to tuple of ints
        self.evaluated_cache = {
            eval(k): v for k, v in checkpoint['evaluated_cache'].items()
        }

        generation = checkpoint['generation']

        print(f"✓ Checkpoint loaded:")
        print(f"  Generation: {generation}")
        print(f"  Population size: {len(population)}")
        print(f"  Best fitness: {best_ever.fitness:.2f}")
        print(f"  Cache size: {len(self.evaluated_cache)}")

        # Note: We do NOT overwrite config from checkpoint by default, 
        # allowing user to change parameters (e.g. mutation rate) on resume if they wish.
        # But we warn if critical structural params match.
        saved_config = checkpoint.get('config', {})
        if saved_config.get('num_modules') != self.num_modules:
            print("⚠️ Warning: Loaded checkpoint has different num_modules than current model!")

        return generation, population, best_ever

    def evolve(self, resume_from: str = None) -> Individual:
        """Main genetic algorithm loop with checkpoint support."""
        print("\\n" + "="*80)
        print("Starting Genetic Algorithm Evolution")
        print("="*80)

        start_generation = 0
        best_ever = None
        population = []

        # Try to resume from checkpoint
        if resume_from:
            start_generation, population, best_ever = self.load_checkpoint(resume_from)
            start_generation += 1  # Start from next generation
            print(f"\\n▶️  Resuming from generation {start_generation}")
        else:
            # Initialize
            import sys
            print("\\nGenerating initial population...", flush=True)
            population = self.initialize_population()

            # Evaluate initial
            print("Evaluating initial population...", flush=True)
            population = self.evaluate_fitness_batch_parallel(population)
            print(f"✓ Initial population evaluation complete", flush=True)

            # Find best
            valid_pop = [ind for ind in population if ind.is_valid]
            if not valid_pop:
                # Should be rare with _repair_chromosome
                print("⚠️ No valid individuals in initial population! Picking best invalid one.", flush=True)
                best_ever = min(population, key=lambda x: x.fitness)
            else:
                best_ever = min(valid_pop, key=lambda x: x.fitness)

            best_ever = copy.deepcopy(best_ever) # Keep a separate copy
            print(f"Initial best: {best_ever}", flush=True)
            print(f"Initial best PPL: {best_ever.fitness:.4f}", flush=True)
            print(f"Initial best chromosome: {best_ever.chromosome}", flush=True)
            sys.stdout.flush()

            # Save initial checkpoint
            if self.checkpoint_dir:
                self.save_checkpoint(0, population, best_ever)

        # Evolution Loop
        for generation in range(start_generation, self.max_generations):
            import sys
            print(f"\\n{'='*80}", flush=True)
            print(f"🔄 Generation {generation + 1}/{self.max_generations} - Current Best PPL: {best_ever.fitness:.4f}", flush=True)
            print(f"{'='*80}", flush=True)
            sys.stdout.flush()

            new_population = []

            # 1. Elitism: Keep current generation best
            # Find current best valid individual
            valid_pop = [ind for ind in population if ind.is_valid]
            current_best = None
            if valid_pop:
                current_best = min(valid_pop, key=lambda x: x.fitness)
            
            if current_best:
                new_population.append(copy.deepcopy(current_best))
                print(f"  👑 Elitism: Kept best (Fitness: {current_best.fitness:.4f})", flush=True)
            else:
                # If no valid individual, keep *something* to maintain population size
                # Usually keep the one with lowest fitness (even if invalid) or just random
                # Here we force keep the best invalid one hoping it mutates to valid
                fallback_best = min(population, key=lambda x: x.fitness)
                new_population.append(copy.deepcopy(fallback_best))
                print(f"  ⚠️ Elitism: No valid individuals. Kept best invalid (Fitness: {fallback_best.fitness:.4f})", flush=True)

            # 2. Generate Offspring
            # We need to fill population_size - len(new_population) spots
            offspring_to_evaluate = []

            while len(new_population) + len(offspring_to_evaluate) < self.population_size:
                # Selection
                if self.selection_method == "top20":
                    parent1, parent2 = self.select_two_different_parents(population, percent=0.2)
                elif self.selection_method == "topNw":
                    parent1, parent2 = self.select_weighted_parents(population, percent=self.top_percent)
                else:  # tournament (default)
                    parent1 = self.tournament_selection(population)
                    parent2 = self.tournament_selection(population)

                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                offspring_to_evaluate.append(child1)
                if len(new_population) + len(offspring_to_evaluate) < self.population_size:
                    offspring_to_evaluate.append(child2)

            # 3. Evaluate Offspring
            print(f"  🧬 Evaluating {len(offspring_to_evaluate)} offspring...", flush=True)
            offspring_to_evaluate = self.evaluate_fitness_batch_parallel(offspring_to_evaluate)
            print(f"  ✓ Evaluation complete for {len(offspring_to_evaluate)} offspring", flush=True)
            
            # Add to new population
            new_population.extend(offspring_to_evaluate)
            
            # Replace old population
            population = new_population

            # 4. Update Global Best
            gen_best = min(population, key=lambda x: x.fitness if x.is_valid else float('inf'))

            if gen_best.is_valid and gen_best.fitness < best_ever.fitness:
                print(f"  🎉 New Global Best Found! Fitness: {gen_best.fitness:.4f} (was {best_ever.fitness:.4f})", flush=True)
                print(f"     Previous Best Chromosome: {best_ever.chromosome}", flush=True)
                print(f"     New Best Chromosome: {gen_best.chromosome}", flush=True)
                best_ever = copy.deepcopy(gen_best)
            else:
                print(f"  Generation Best: {gen_best.fitness:.4f} (Global Best: {best_ever.fitness:.4f})", flush=True)

            # Stats
            valid_count = sum(1 for ind in population if ind.is_valid)
            avg_fitness = np.mean([ind.fitness for ind in population if ind.is_valid]) if valid_count > 0 else float('inf')
            print(f"  📊 Stats: Valid={valid_count}/{self.population_size}, Avg Valid Fitness={avg_fitness:.4f}", flush=True)
            print(f"  ⭐ Current Global Best PPL: {best_ever.fitness:.4f}", flush=True)
            print(f"  🧬 Current Best Chromosome: {best_ever.chromosome}", flush=True)
            import sys
            sys.stdout.flush()  # Force flush to ensure output appears in nohup logs

            # 5. Checkpoint
            if self.checkpoint_dir and (generation + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(generation + 1, population, best_ever)

        import sys
        print("\\n" + "="*80, flush=True)
        print("Evolution Complete!", flush=True)
        print("="*80, flush=True)
        print(f"Best Individual Found: {best_ever}", flush=True)
        print(f"Final Best PPL: {best_ever.fitness:.4f}", flush=True)
        print(f"Final Best Chromosome: {best_ever.chromosome}", flush=True)
        sys.stdout.flush()

        # Final checkpoint
        if self.checkpoint_dir:
            self.save_checkpoint(self.max_generations, population, best_ever)

        return best_ever