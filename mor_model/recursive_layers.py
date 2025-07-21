"""
Recursive transformer layers implementing parameter sharing strategies.

Implements the four parameter sharing strategies from the MoR paper:
- Cycle: Cyclical parameter sharing across all layers
- Sequence: Sequential parameter sharing  
- Middle-Cycle: Preserves first/last layers, cycles middle layers (optimal)
- Middle-Sequence: Preserves first/last layers, sequences middle layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math
from .routing import ExpertChoiceRouter, TokenChoiceRouter
from .kv_cache import EfficientKVCache


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embedding."""
        # Generate position indices
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        
        # Compute frequencies
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos_emb = emb.cos()[None, None, :, :]
        sin_emb = emb.sin()[None, None, :, :]
        
        return cos_emb, sin_emb


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MoRAttention(nn.Module):
    """Multi-head attention with MoR optimizations."""
    
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads")
        
        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Rotary position embedding
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        kv_cache: Optional[EfficientKVCache] = None,
        recursion_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary position embedding
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Update efficient KV cache if provided
        if kv_cache is not None:
            kv_cache.update_cache(self.layer_idx, key_states, value_states, recursion_step)
        
        # Repeat k/v heads if n_kv_heads < n_heads
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value


class MoRMLP(nn.Module):
    """Feed-forward network for MoR transformer."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()  # Swish activation
    
    def forward(self, x):
        # SwiGLU activation: SiLU(gate) * up
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class RecursiveTransformerBlock(nn.Module):
    """
    Recursive transformer block implementing MoR parameter sharing.
    
    This block can be used recursively with shared parameters according to
    the specified parameter sharing strategy.
    """
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Core transformer components
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MoRAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MoRMLP(config)
        
        # Routing mechanism
        if config.routing_strategy == "expert_choice":
            self.router = ExpertChoiceRouter(
                config.hidden_size,
                config.router_type,
                config.beta_percentile,
                config.auxiliary_loss_weight
            )
        elif config.routing_strategy == "token_choice":
            self.router = TokenChoiceRouter(
                config.hidden_size,
                config.max_recursion_depth,
                config.router_type
            )
        else:
            self.router = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        kv_cache: Optional[EfficientKVCache] = None,
        recursion_step: Optional[int] = None,
        routing_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]], Optional[torch.Tensor]]:
        
        residual = hidden_states
        
        # Pre-attention normalization
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            kv_cache=kv_cache,
            recursion_step=recursion_step,
        )
        
        # Apply routing mask if provided
        if routing_mask is not None:
            hidden_states = hidden_states * routing_mask.unsqueeze(-1)
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        # Pre-MLP normalization
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # MLP
        hidden_states = self.mlp(hidden_states)
        
        # Apply routing mask again
        if routing_mask is not None:
            hidden_states = hidden_states * routing_mask.unsqueeze(-1)
        
        # Final residual connection
        hidden_states = residual + hidden_states
        
        # Compute routing for next recursion step
        auxiliary_loss = None
        if self.router is not None and recursion_step is not None:
            if isinstance(self.router, ExpertChoiceRouter):
                _, next_routing_mask, auxiliary_loss = self.router(
                    hidden_states, recursion_step + 1, routing_mask
                )
            elif isinstance(self.router, TokenChoiceRouter):
                _, next_routing_mask = self.router(hidden_states, recursion_step + 1)
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if auxiliary_loss is not None:
            outputs += (auxiliary_loss,)
        
        return outputs


class ParameterSharingManager:
    """
    Manages parameter sharing strategies for recursive blocks.
    
    Implements the four strategies from the MoR paper:
    - Cycle: h^(ℓ+1) = f(h^ℓ; Φ_{ℓ mod Nr})
    - Sequence: h^(ℓ+1) = f(h^ℓ; Φ_{min(ℓ, Nr-1)})  
    - Middle-Cycle: Preserves first/last, cycles middle
    - Middle-Sequence: Preserves first/last, sequences middle
    """
    
    def __init__(self, num_layers: int, num_recursion_blocks: int, strategy: str):
        self.num_layers = num_layers
        self.num_recursion_blocks = num_recursion_blocks
        self.strategy = strategy
        self.layer_mapping = self._create_layer_mapping()
    
    def _create_layer_mapping(self) -> Dict[int, int]:
        """Create mapping from layer index to parameter block index."""
        mapping = {}
        
        if self.strategy == "cycle":
            # Cyclical sharing: layer ℓ uses parameters Φ_{ℓ mod Nr}
            for layer_idx in range(self.num_layers):
                mapping[layer_idx] = layer_idx % self.num_recursion_blocks
                
        elif self.strategy == "sequence":
            # Sequential sharing: layer ℓ uses parameters Φ_{min(ℓ, Nr-1)}
            for layer_idx in range(self.num_layers):
                mapping[layer_idx] = min(layer_idx, self.num_recursion_blocks - 1)
                
        elif self.strategy == "middle_cycle":
            # Middle-Cycle: preserve first and last layers, cycle middle layers
            if self.num_layers <= 2:
                # If only 1-2 layers, use direct mapping
                for layer_idx in range(self.num_layers):
                    mapping[layer_idx] = layer_idx
            else:
                # First layer uses block 0
                mapping[0] = 0
                
                # Middle layers cycle through blocks 1 to Nr-2
                middle_blocks = max(1, self.num_recursion_blocks - 2)
                for layer_idx in range(1, self.num_layers - 1):
                    mapping[layer_idx] = 1 + ((layer_idx - 1) % middle_blocks)
                
                # Last layer uses block Nr-1
                mapping[self.num_layers - 1] = self.num_recursion_blocks - 1
                
        elif self.strategy == "middle_sequence":
            # Middle-Sequence: preserve first and last layers, sequence middle layers
            if self.num_layers <= 2:
                for layer_idx in range(self.num_layers):
                    mapping[layer_idx] = layer_idx
            else:
                # First layer uses block 0
                mapping[0] = 0
                
                # Middle layers use sequential blocks 1 to Nr-2
                middle_blocks = max(1, self.num_recursion_blocks - 2)
                for layer_idx in range(1, self.num_layers - 1):
                    mapping[layer_idx] = 1 + min(layer_idx - 1, middle_blocks - 1)
                
                # Last layer uses block Nr-1
                mapping[self.num_layers - 1] = self.num_recursion_blocks - 1
        
        else:
            raise ValueError(f"Unknown parameter sharing strategy: {self.strategy}")
        
        return mapping
    
    def get_shared_block_idx(self, layer_idx: int) -> int:
        """Get the shared parameter block index for a given layer."""
        return self.layer_mapping.get(layer_idx, layer_idx)
