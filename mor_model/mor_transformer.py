"""
Main MoR Transformer model implementation.

Combines recursive layers, parameter sharing, and dynamic routing into a unified
small language model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union
import math
import warnings

from .config import MoRConfig
from .recursive_layers import (
    RecursiveTransformerBlock, 
    ParameterSharingManager,
    RMSNorm
)
from .routing import ExpertChoiceRouter, TokenChoiceRouter, RoutingLoss
from .kv_cache import EfficientKVCache


class MoREmbeddings(nn.Module):
    """Token and position embeddings for MoR model."""
    
    def __init__(self, config: MoRConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Forward pass for embeddings."""
        return self.word_embeddings(input_ids)


class MoRModel(nn.Module):
    """
    Core MoR model implementing recursive parameter sharing and dynamic routing.
    """
    
    def __init__(self, config: MoRConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        
        # Validate configuration
        config.validate()
        
        # Embeddings
        self.embeddings = MoREmbeddings(config)
        
        # Parameter sharing manager
        self.param_sharing = ParameterSharingManager(
            num_layers=config.num_recursion_blocks,  # We create Nr shared blocks
            num_recursion_blocks=config.num_recursion_blocks,
            strategy=config.parameter_sharing_strategy
        )
        
        # Create shared recursive blocks
        self.shared_blocks = nn.ModuleList([
            RecursiveTransformerBlock(config, layer_idx=i)
            for i in range(config.num_recursion_blocks)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Routing loss function
        self.routing_loss_fn = RoutingLoss(config.auxiliary_loss_weight)
        
        # Initialize KV cache for inference
        self.kv_cache = None
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def setup_kv_cache(self, max_batch_size: int, max_seq_length: int, device: torch.device):
        """Setup KV cache for efficient inference."""
        if self.config.enable_kv_cache:
            self.kv_cache = EfficientKVCache(
                max_batch_size=max_batch_size,
                max_seq_length=max_seq_length,
                num_heads=self.config.num_attention_heads,
                head_dim=self.config.hidden_size // self.config.num_attention_heads,
                num_layers=len(self.shared_blocks),
                num_recursion_blocks=self.config.num_recursion_blocks,
                device=device
            )
    
    def clear_kv_cache(self):
        """Clear KV cache."""
        if self.kv_cache is not None:
            self.kv_cache.clear_cache()
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        recursion_depths: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Dict]:
        """
        Forward pass with recursive computation and dynamic routing.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            recursion_depths: Per-token recursion depths [batch_size, seq_len]
                If None, uses uniform max recursion depth
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions if hasattr(self.config, 'output_attentions') else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states if hasattr(self.config, 'output_hidden_states') else False
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict if hasattr(self.config, 'use_return_dict') else True
        
        # Get input embeddings
        if inputs_embeds is None:
            # Ensure input_ids are Long tensors for embedding lookup
            if input_ids.dtype != torch.long:
                input_ids = input_ids.long()
            inputs_embeds = self.embeddings(input_ids)
        
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        device = inputs_embeds.device
        
        # Setup default recursion depths if not provided
        if recursion_depths is None:
            recursion_depths = torch.full(
                (batch_size, seq_len), 
                self.config.max_recursion_depth, 
                dtype=torch.long, 
                device=device
            )
        
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        
        # Convert attention mask to causal mask
        causal_mask = self._prepare_causal_attention_mask(
            attention_mask, (batch_size, seq_len), inputs_embeds.dtype, device
        )
        
        # Initialize hidden states
        hidden_states = inputs_embeds
        
        # Storage for outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        auxiliary_losses = []
        
        # Recursive computation with dynamic routing
        hidden_states, auxiliary_losses = self._recursive_forward(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            recursion_depths=recursion_depths,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            all_hidden_states=all_hidden_states,
            all_self_attentions=all_self_attentions,
            next_decoder_cache=next_decoder_cache,
        )
        
        # Final layer normalization
        hidden_states = self.norm(hidden_states)
        
        # Add final hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [
                hidden_states, 
                next_decoder_cache, 
                all_hidden_states, 
                all_self_attentions,
                auxiliary_losses
            ] if v is not None)
        
        return {
            'last_hidden_state': hidden_states,
            'past_key_values': next_decoder_cache,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attentions,
            'auxiliary_losses': auxiliary_losses,
        }
    
    def _recursive_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        recursion_depths: torch.LongTensor,
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        all_hidden_states: Optional[Tuple],
        all_self_attentions: Optional[Tuple],
        next_decoder_cache: Optional[Tuple],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Perform recursive forward pass with dynamic routing.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        auxiliary_losses = []
        max_depth = recursion_depths.max().item()
        
        # Initialize routing masks for each token
        # routing_masks[depth] contains mask for tokens that should be processed at this depth
        routing_masks = {}
        for depth in range(1, max_depth + 1):
            routing_masks[depth] = (recursion_depths >= depth).float()
        
        # Process each recursion depth
        for recursion_step in range(max_depth):
            current_depth = recursion_step + 1
            
            # Get tokens that should be processed at this depth
            current_mask = routing_masks.get(current_depth)
            if current_mask is None or current_mask.sum() == 0:
                continue  # No tokens to process at this depth
            
            # Determine which shared block to use
            block_idx = self.param_sharing.get_shared_block_idx(recursion_step)
            current_block = self.shared_blocks[block_idx]
            
            # Apply dynamic routing if configured
            if hasattr(current_block, 'router') and current_block.router is not None:
                if isinstance(current_block.router, ExpertChoiceRouter):
                    # Expert-choice routing
                    previous_mask = routing_masks.get(current_depth - 1) if current_depth > 1 else None
                    routing_scores, routing_mask, aux_loss = current_block.router(
                        hidden_states, recursion_step, previous_mask
                    )
                    if aux_loss is not None:
                        auxiliary_losses.append(aux_loss)
                    
                    # Update routing mask for current depth
                    current_mask = routing_mask
                    
                elif isinstance(current_block.router, TokenChoiceRouter):
                    # Token-choice routing
                    depth_probs, continue_mask = current_block.router(hidden_states, recursion_step)
                    current_mask = continue_mask
            
            # Forward pass through the recursive block
            layer_outputs = current_block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,  # Handle past_key_values if needed
                output_attentions=output_attentions,
                use_cache=use_cache,
                kv_cache=self.kv_cache,
                recursion_step=recursion_step,
                routing_mask=current_mask,
            )
            
            # Extract outputs
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
            
            if use_cache:
                next_decoder_cache = next_decoder_cache + (layer_outputs[-2],)
            
            # Collect auxiliary losses
            if len(layer_outputs) > 3:  # Has auxiliary loss
                auxiliary_losses.append(layer_outputs[-1])
            
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        
        return hidden_states, auxiliary_losses
    
    def _prepare_causal_attention_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], 
        dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Prepare causal attention mask."""
        batch_size, seq_length = input_shape
        
        # Create causal mask
        causal_mask = torch.full(
            (seq_length, seq_length), 
            torch.finfo(dtype).min, 
            device=device
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask[None, None, :, :].expand(
            batch_size, 1, seq_length, seq_length
        )
        
        # Apply attention mask
        if attention_mask is not None:
            expanded_mask = attention_mask[:, None, None, :].expand(
                batch_size, 1, seq_length, seq_length
            ).to(dtype)
            inverted_mask = 1.0 - expanded_mask
            causal_mask = causal_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(dtype).min
            )
        
        return causal_mask
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Calculate parameter reduction factor
        standard_layers = self.config.max_recursion_depth  # Equivalent standard model layers
        shared_blocks = self.config.num_recursion_blocks
        reduction_factor = standard_layers / shared_blocks if shared_blocks > 0 else 1.0
        
        stats = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_reduction_factor': reduction_factor,
            'shared_blocks': shared_blocks,
            'max_recursion_depth': self.config.max_recursion_depth,
        }
        
        if self.kv_cache is not None:
            cache_stats = self.kv_cache.memory_usage()
            stats.update(cache_stats)
        
        return stats


class MoRForCausalLM(nn.Module):
    """MoR model with causal language modeling head."""
    
    def __init__(self, config: MoRConfig):
        super().__init__()
        self.config = config
        self.model = MoRModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        recursion_depths: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Dict]:
        """Forward pass with optional language modeling loss."""
        
        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            recursion_depths=recursion_depths,
        )
        
        # Handle model outputs - always convert to consistent format
        if isinstance(outputs, dict):
            hidden_states = outputs['last_hidden_state']
            auxiliary_losses = outputs.get('auxiliary_losses', [])
            # Convert dict to tuple format for consistent handling below
            past_key_values = outputs.get('past_key_values')
            all_hidden_states = outputs.get('hidden_states')
            all_attentions = outputs.get('attentions')
        else:
            # Tuple/list output
            hidden_states = outputs[0]
            past_key_values = outputs[1] if len(outputs) > 1 else None
            all_hidden_states = outputs[2] if len(outputs) > 2 else None
            all_attentions = outputs[3] if len(outputs) > 3 else None
            auxiliary_losses = outputs[4] if len(outputs) > 4 else []
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Compute language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Add auxiliary losses
            if auxiliary_losses:
                total_aux_loss = sum(auxiliary_losses)
                loss = lm_loss + total_aux_loss
            else:
                loss = lm_loss
        
        if not return_dict:
            # Build tuple output using the extracted components
            output = (logits,)
            if past_key_values is not None:
                output += (past_key_values,)
            if all_hidden_states is not None:
                output += (all_hidden_states,)
            if all_attentions is not None:
                output += (all_attentions,)
            if auxiliary_losses:
                output += (auxiliary_losses,)
            return (loss,) + output if loss is not None else output
        
        result = {
            'loss': loss,
            'logits': logits,
            'past_key_values': outputs.get('past_key_values'),
            'hidden_states': outputs.get('hidden_states'),
            'attentions': outputs.get('attentions'),
            'auxiliary_losses': auxiliary_losses,
        }
        
        return result
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """Prepare inputs for generation."""
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        
        return model_inputs
    
    def setup_kv_cache(self, max_batch_size: int, max_seq_length: int, device: torch.device):
        """Setup KV cache for efficient inference."""
        self.model.setup_kv_cache(max_batch_size, max_seq_length, device)
    
    def clear_kv_cache(self):
        """Clear KV cache."""
        self.model.clear_kv_cache()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        return self.model.get_memory_usage()


# Alias for compatibility
MoRTransformer = MoRForCausalLM
