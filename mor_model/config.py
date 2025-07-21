"""
Configuration classes for MoR-SLM model.
"""

from dataclasses import dataclass
from typing import Optional, Literal
import yaml


@dataclass
class MoRConfig:
    """Configuration for Mixture-of-Recursions Transformer."""
    
    # Base transformer parameters
    vocab_size: int = 32000
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    
    # MoR-specific parameters
    num_recursion_blocks: int = 4  # Nr in paper
    parameter_sharing_strategy: Literal["cycle", "sequence", "middle_cycle", "middle_sequence"] = "middle_cycle"
    max_recursion_depth: int = 4
    min_recursion_depth: int = 1
    
    # Routing configuration
    routing_strategy: Literal["expert_choice", "token_choice"] = "expert_choice"
    router_type: Literal["linear", "mlp"] = "linear"
    beta_percentile: float = 0.8
    auxiliary_loss_weight: float = 0.01
    
    # KV Cache optimization
    enable_kv_cache: bool = True
    cache_strategy: str = "efficient"
    
    # Training parameters
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = 1
    eos_token_id: Optional[int] = 2
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'MoRConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract model config from nested structure
        model_config = config_dict.get('model', {})
        
        # Ensure proper type casting for numeric values
        if 'rms_norm_eps' in model_config:
            model_config['rms_norm_eps'] = float(model_config['rms_norm_eps'])
        if 'rope_theta' in model_config:
            model_config['rope_theta'] = float(model_config['rope_theta'])
        if 'auxiliary_loss_weight' in model_config:
            model_config['auxiliary_loss_weight'] = float(model_config['auxiliary_loss_weight'])
        if 'beta_percentile' in model_config:
            model_config['beta_percentile'] = float(model_config['beta_percentile'])
            
        return cls(**model_config)
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    def validate(self):
        """Validate configuration parameters."""
        assert self.num_recursion_blocks >= 1, "num_recursion_blocks must be >= 1"
        assert self.max_recursion_depth >= self.min_recursion_depth, "max_recursion_depth must be >= min_recursion_depth"
        assert self.min_recursion_depth >= 1, "min_recursion_depth must be >= 1"
        assert self.max_recursion_depth <= self.num_recursion_blocks, "max_recursion_depth must be <= num_recursion_blocks"
        assert 0.0 <= self.beta_percentile <= 1.0, "beta_percentile must be between 0 and 1"
        assert self.auxiliary_loss_weight >= 0.0, "auxiliary_loss_weight must be >= 0"
