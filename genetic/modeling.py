import torch
import torch.nn as nn
from typing import List, Union, Optional, Tuple
import copy

def decode_chromosome(chromosome: List[int]) -> List[int]:
    """
    Decode a loop-encoded chromosome into an execution path.
    
    Args:
        chromosome: List[int], values in {0,1,2,...,max_loop_count}
        
    Returns:
        path: List[int], execution path (module indices in order)
    """
    path = []
    i = 0
    n = len(chromosome)

    while i < n:
        val = chromosome[i]

        if val == 0:
            i += 1
            continue

        if val == 1:
            path.append(i)
            i += 1
            continue

        if val >= 2:
            current_loop_count = val
            block_indices = []
            while i < n and chromosome[i] == current_loop_count:
                block_indices.append(i)
                i += 1
            
            for _ in range(current_loop_count):
                path.extend(block_indices)
            continue

    return path

class DecoupledLlamaLayer(nn.Module):
    """
    Wrapper for Llama sub-modules (Attention or FFN) to provide a unified interface.
    This allows us to treat Attn and FFN blocks interchangeably in the execution loop.
    """
    def __init__(self, module_type: str, module: nn.Module, layernorm: nn.Module):
        """
        Args:
            module_type: 'attention' or 'ffn'
            module: The actual layer (LlamaAttention or LlamaMLP)
            layernorm: The associated LayerNorm (input_layernorm or post_attention_layernorm)
        """
        super().__init__()
        self.module_type = module_type
        self.module = module
        self.layernorm = layernorm

    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None, 
                position_ids: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                **kwargs) -> torch.Tensor:
        
        residual = hidden_states

        # 1. Apply LayerNorm
        hidden_states = self.layernorm(hidden_states)

        # 2. Apply Module
        if self.module_type == 'attention':
            # LlamaAttention requires specific arguments
            # We assume causal masking is handled by attention_mask or the model structure
            if position_ids is None:
                # Generate simple position_ids if not provided (though typically passed from model)
                seq_length = hidden_states.shape[1]
                position_ids = torch.arange(0, seq_length, dtype=torch.long, device=hidden_states.device)
                position_ids = position_ids.unsqueeze(0).expand(hidden_states.shape[0], -1)

            attn_outputs = self.module(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                **kwargs
            )
            # LlamaAttention returns (hidden_states, attentions, past_key_value)
            hidden_states = attn_outputs[0]
            
        elif self.module_type == 'ffn':
            # LlamaMLP is simpler
            hidden_states = self.module(hidden_states)

        # 3. Residual Connection
        hidden_states = residual + hidden_states
        
        return hidden_states

class DecoupledLlamaModel(nn.Module):
    """
    A lightweight 'Virtual' Llama Model.
    
    It does NOT copy the weights. It holds references to the original model's sub-modules.
    It dynamically assembles the execution path based on the chromosome.
    """
    def __init__(self, original_model: nn.Module, chromosome: List[int]):
        super().__init__()
        # We store references to the shared components
        # Note: We do NOT use nn.ModuleList/Dict to register them, 
        # because we don't want to own them or double-count parameters if this class is inspected.
        # But for .to(device) to work automatically, we might typically want them registered.
        # HOWEVER, since our goal is memory efficiency, we assume 'original_model' manages the device 
        # and life-cycle of weights. We just borrow them.
        
        self.config = original_model.config
        self.embed_tokens = original_model.model.embed_tokens
        self.norm = original_model.model.norm
        self.lm_head = original_model.lm_head
        
        # Original layers list (source of truth)
        self.original_layers = original_model.model.layers
        
        # Decode the chromosome to get the sequence of module indices to execute
        self.execution_path = decode_chromosome(chromosome)
        
        # Pre-resolve the execution path into a list of wrapper objects for speed
        # This avoids looking up logic inside the forward loop
        self.execution_chain = []
        for module_idx in self.execution_path:
            layer_idx = module_idx // 2
            is_attention = (module_idx % 2 == 0)
            
            source_layer = self.original_layers[layer_idx]
            
            if is_attention:
                wrapper = DecoupledLlamaLayer(
                    'attention', 
                    source_layer.self_attn, 
                    source_layer.input_layernorm
                )
            else: # FFN
                wrapper = DecoupledLlamaLayer(
                    'ffn', 
                    source_layer.mlp, 
                    source_layer.post_attention_layernorm
                )
            self.execution_chain.append(wrapper)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass following the genetic execution path.
        """
        # 1. Embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Create position_ids for the whole sequence length
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        
        # Llama expects specific attention mask format (causal mask)
        # For simple PPL eval on WikiText, we usually just pass None or simple mask
        # But LlamaModel._prepare_decoder_attention_mask is usually needed.
        # Here we assume 'attention_mask' passed in is already prepared (4D) or None is fine for inference
        
        # 2. Dynamic Execution Loop
        for layer_wrapper in self.execution_chain:
            hidden_states = layer_wrapper(
                hidden_states, 
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            
        # 3. Final Norm
        hidden_states = self.norm(hidden_states)
        
        # 4. LM Head
        logits = self.lm_head(hidden_states)
        
        return logits
