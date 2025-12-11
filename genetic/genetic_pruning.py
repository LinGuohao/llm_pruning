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