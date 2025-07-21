"""
Efficient Key-Value caching for MoR-SLM inference optimization.

Implements memory-efficient caching strategies for variable recursion depths.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import math


class EfficientKVCache:
    """
    Efficient Key-Value cache for MoR inference.
    
    Handles variable recursion depths and memory optimization for different
    parameter sharing strategies.
    """
    
    def __init__(self, max_batch_size: int, max_seq_length: int, 
                 num_heads: int, head_dim: int, num_layers: int,
                 num_recursion_blocks: int, device: torch.device):
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.num_recursion_blocks = num_recursion_blocks
        self.device = device
        
        # Initialize cache storage
        self.key_cache = {}
        self.value_cache = {}
        self.cache_lengths = {}
        
        # Track recursion-specific caches
        self.recursion_caches = {}
        
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialize cache tensors."""
        cache_shape = (
            self.max_batch_size,
            self.num_heads,
            self.max_seq_length,
            self.head_dim
        )
        
        # Standard layer caches
        for layer_idx in range(self.num_layers):
            self.key_cache[layer_idx] = torch.zeros(
                cache_shape, dtype=torch.float16, device=self.device
            )
            self.value_cache[layer_idx] = torch.zeros(
                cache_shape, dtype=torch.float16, device=self.device
            )
            self.cache_lengths[layer_idx] = torch.zeros(
                self.max_batch_size, dtype=torch.long, device=self.device
            )
        
        # Recursion block caches (shared parameters)
        for recursion_idx in range(self.num_recursion_blocks):
            self.recursion_caches[recursion_idx] = {
                'key': torch.zeros(cache_shape, dtype=torch.float16, device=self.device),
                'value': torch.zeros(cache_shape, dtype=torch.float16, device=self.device),
                'length': torch.zeros(self.max_batch_size, dtype=torch.long, device=self.device)
            }
    
    def get_cache(self, layer_idx: int, recursion_step: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get key-value cache for a specific layer or recursion step.
        
        Args:
            layer_idx: Layer index for standard layers
            recursion_step: Recursion step index for recursive layers
            
        Returns:
            key_cache: Cached keys
            value_cache: Cached values  
            cache_length: Current cache length
        """
        if recursion_step is not None:
            # Use recursion-specific cache
            cache = self.recursion_caches[recursion_step]
            return cache['key'], cache['value'], cache['length']
        else:
            # Use standard layer cache
            return (
                self.key_cache[layer_idx],
                self.value_cache[layer_idx], 
                self.cache_lengths[layer_idx]
            )
    
    def update_cache(self, layer_idx: int, key_states: torch.Tensor, 
                    value_states: torch.Tensor, recursion_step: Optional[int] = None):
        """
        Update cache with new key-value states.
        
        Args:
            layer_idx: Layer index
            key_states: New key states to cache
            value_states: New value states to cache
            recursion_step: Recursion step if applicable
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        if recursion_step is not None:
            # Update recursion cache
            cache = self.recursion_caches[recursion_step]
            current_length = cache['length'][0].item()  # Assume same length for batch
            
            # Ensure tensor shapes match for cache update
            if key_states.shape[0] != batch_size or key_states.dim() != 4:
                # Reshape if needed
                if key_states.dim() == 3:  # Missing batch dimension
                    key_states = key_states.unsqueeze(0)[:batch_size]
                    value_states = value_states.unsqueeze(0)[:batch_size]
                elif key_states.shape[0] != batch_size:
                    key_states = key_states[:batch_size]
                    value_states = value_states[:batch_size]
            
            # Update cache tensors with proper bounds checking
            end_length = min(current_length + seq_len, cache['key'].size(2))
            actual_seq_len = end_length - current_length
            
            cache['key'][:batch_size, :, current_length:end_length] = key_states[:, :, :actual_seq_len]
            cache['value'][:batch_size, :, current_length:end_length] = value_states[:, :, :actual_seq_len]
            cache['length'][:batch_size] += actual_seq_len
        else:
            # Update standard cache
            current_length = self.cache_lengths[layer_idx][0].item()
            
            # Ensure tensor shapes match for cache update
            if key_states.shape[0] != batch_size or key_states.dim() != 4:
                # Reshape if needed
                if key_states.dim() == 3:  # Missing batch dimension
                    key_states = key_states.unsqueeze(0)[:batch_size]
                    value_states = value_states.unsqueeze(0)[:batch_size]
                elif key_states.shape[0] != batch_size:
                    key_states = key_states[:batch_size]
                    value_states = value_states[:batch_size]
            
            # Update cache tensors with proper bounds checking
            end_length = min(current_length + seq_len, self.key_cache[layer_idx].size(2))
            actual_seq_len = end_length - current_length
            
            self.key_cache[layer_idx][:batch_size, :, current_length:end_length] = key_states[:, :, :actual_seq_len]
            self.value_cache[layer_idx][:batch_size, :, current_length:end_length] = value_states[:, :, :actual_seq_len]
            self.cache_lengths[layer_idx][:batch_size] += actual_seq_len
    
    def get_effective_cache(self, layer_idx: int, recursion_step: Optional[int] = None,
                           max_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get effective cache up to specified length.
        
        Returns:
            effective_keys: Keys up to max_length
            effective_values: Values up to max_length
        """
        key_cache, value_cache, cache_length = self.get_cache(layer_idx, recursion_step)
        
        if max_length is None:
            max_length = cache_length.max().item()
        
        return (
            key_cache[:, :, :max_length],
            value_cache[:, :, :max_length]
        )
    
    def clear_cache(self):
        """Clear all cached states."""
        for layer_idx in range(self.num_layers):
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()
            self.cache_lengths[layer_idx].zero_()
        
        for recursion_idx in range(self.num_recursion_blocks):
            self.recursion_caches[recursion_idx]['key'].zero_()
            self.recursion_caches[recursion_idx]['value'].zero_()
            self.recursion_caches[recursion_idx]['length'].zero_()
    
    def memory_usage(self) -> Dict[str, float]:
        """Calculate memory usage in MB."""
        def tensor_memory_mb(tensor):
            return tensor.numel() * tensor.element_size() / (1024 * 1024)
        
        total_standard = sum(
            tensor_memory_mb(self.key_cache[i]) + tensor_memory_mb(self.value_cache[i])
            for i in range(self.num_layers)
        )
        
        total_recursion = sum(
            tensor_memory_mb(cache['key']) + tensor_memory_mb(cache['value'])
            for cache in self.recursion_caches.values()
        )
        
        return {
            'standard_cache_mb': total_standard,
            'recursion_cache_mb': total_recursion,
            'total_mb': total_standard + total_recursion
        }


class DynamicKVCache(EfficientKVCache):
    """
    Dynamic KV cache that adapts to actual sequence lengths and recursion patterns.
    
    More memory efficient for variable-length sequences and sparse recursion patterns.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_caches = {}
        self.active_sequences = set()
    
    def allocate_dynamic_cache(self, batch_idx: int, actual_seq_length: int):
        """Allocate cache for specific batch item with actual sequence length."""
        if batch_idx not in self.dynamic_caches:
            cache_shape = (1, self.num_heads, actual_seq_length, self.head_dim)
            
            self.dynamic_caches[batch_idx] = {
                'keys': {},
                'values': {},
                'seq_length': actual_seq_length
            }
            
            # Allocate for each layer
            for layer_idx in range(self.num_layers):
                self.dynamic_caches[batch_idx]['keys'][layer_idx] = torch.zeros(
                    cache_shape, dtype=torch.float16, device=self.device
                )
                self.dynamic_caches[batch_idx]['values'][layer_idx] = torch.zeros(
                    cache_shape, dtype=torch.float16, device=self.device
                )
            
            self.active_sequences.add(batch_idx)
    
    def cleanup_inactive_caches(self):
        """Remove caches for inactive sequences to free memory."""
        inactive_sequences = []
        for batch_idx in self.active_sequences:
            # Add logic to determine if sequence is still active
            # For now, we'll keep all sequences active
            pass
        
        for batch_idx in inactive_sequences:
            if batch_idx in self.dynamic_caches:
                del self.dynamic_caches[batch_idx]
                self.active_sequences.discard(batch_idx)
