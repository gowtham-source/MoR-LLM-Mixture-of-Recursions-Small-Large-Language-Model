"""
MoR-SLM: Mixture-of-Recursions Small Language Model

Core module implementing the MoR framework with recursive parameter sharing
and dynamic routing mechanisms.
"""

from .config import MoRConfig
from .mor_transformer import MoRTransformer, MoRForCausalLM, MoRModel
from .recursive_layers import RecursiveTransformerBlock, ParameterSharingManager
from .routing import ExpertChoiceRouter, TokenChoiceRouter, RoutingLoss
from .kv_cache import EfficientKVCache, DynamicKVCache

__all__ = [
    "MoRConfig",
    "MoRTransformer",
    "MoRForCausalLM",
    "MoRModel",
    "RecursiveTransformerBlock",
    "ParameterSharingManager",
    "ExpertChoiceRouter",
    "TokenChoiceRouter",
    "RoutingLoss",
    "EfficientKVCache",
    "DynamicKVCache"
]
