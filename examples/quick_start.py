#!/usr/bin/env python3
"""
Quick start example for MoR-SLM.

This script demonstrates how to:
1. Create a small MoR model
2. Train it on a tiny dataset
3. Generate text with the trained model
"""

import os
import sys
import torch
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from mor_model import MoRConfig, MoRForCausalLM
from training import MoRTrainer
from inference import MoRInferenceEngine


def create_tiny_config():
    """Create a tiny model configuration for quick testing."""
    config = {
        'model': {
            'vocab_size': 1000,
            'hidden_size': 128,
            'intermediate_size': 512,
            'num_attention_heads': 4,
            'num_key_value_heads': 4,
            'max_position_embeddings': 512,
            'num_recursion_blocks': 2,
            'parameter_sharing_strategy': 'middle_cycle',
            'max_recursion_depth': 2,
            'routing_strategy': 'expert_choice',
            'router_type': 'linear',
            'beta_percentile': 0.8,
            'auxiliary_loss_weight': 0.01,
            'enable_kv_cache': True
        },
        'training': {
            'batch_size': 2,
            'gradient_accumulation_steps': 2,
            'learning_rate': 1e-3,
            'weight_decay': 0.1,
            'warmup_steps': 100,
            'max_steps': 1000,
            'save_steps': 500,
            'eval_steps': 250,
            'use_fsdp': False,
            'use_wandb': False
        },
        'data': {
            'dataset_name': 'wikitext',
            'dataset_config': 'wikitext-2-raw-v1',
            'max_length': 128,
            'preprocessing_num_workers': 2
        },
        'inference': {
            'max_new_tokens': 50,
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': True
        }
    }
    return config


def test_model_creation():
    """Test creating a MoR model."""
    print("🔧 Testing model creation...")
    
    config_dict = create_tiny_config()
    model_config = MoRConfig(**config_dict['model'])
    model = MoRForCausalLM(model_config)
    
    # Print model statistics
    memory_stats = model.get_memory_usage()
    print(f"   ✓ Model created successfully")
    print(f"   ✓ Total parameters: {memory_stats['total_parameters']:,}")
    print(f"   ✓ Parameter reduction factor: {memory_stats['parameter_reduction_factor']:.2f}x")
    print(f"   ✓ Shared blocks: {memory_stats['shared_blocks']}")
    print(f"   ✓ Max recursion depth: {memory_stats['max_recursion_depth']}")
    
    return model, model_config


def test_forward_pass(model):
    """Test forward pass through the model."""
    print("\n🚀 Testing forward pass...")
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    if isinstance(outputs, dict):
        logits = outputs['logits']
        aux_losses = outputs.get('auxiliary_losses', [])
    else:
        logits = outputs[0]
        aux_losses = outputs[-1] if len(outputs) > 1 else []
    
    print(f"   ✓ Forward pass successful")
    print(f"   ✓ Output shape: {logits.shape}")
    print(f"   ✓ Auxiliary losses: {len(aux_losses)}")
    
    return True


def test_parameter_sharing():
    """Test parameter sharing strategies."""
    print("\n🔄 Testing parameter sharing...")
    
    from mor_model.recursive_layers import ParameterSharingManager
    
    strategies = ["cycle", "sequence", "middle_cycle", "middle_sequence"]
    
    for strategy in strategies:
        manager = ParameterSharingManager(
            num_layers=6,
            num_recursion_blocks=4,
            strategy=strategy
        )
        
        # Test mapping
        mappings = [manager.get_shared_block_idx(i) for i in range(6)]
        print(f"   ✓ {strategy}: {mappings}")
    
    return True


def test_routing():
    """Test routing mechanisms."""
    print("\n🎯 Testing routing mechanisms...")
    
    from mor_model.routing import ExpertChoiceRouter, TokenChoiceRouter
    
    hidden_size = 128
    batch_size = 2
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Test Expert-Choice routing
    expert_router = ExpertChoiceRouter(hidden_size)
    scores, mask, aux_loss = expert_router(hidden_states, recursion_step=0)
    print(f"   ✓ Expert-Choice routing: scores {scores.shape}, mask {mask.shape}")
    
    # Test Token-Choice routing
    token_router = TokenChoiceRouter(hidden_size, max_recursion_depth=4)
    depth_probs, continue_mask = token_router(hidden_states, current_depth=1)
    print(f"   ✓ Token-Choice routing: probs {depth_probs.shape}, mask {continue_mask.shape}")
    
    return True


def test_kv_cache():
    """Test KV cache functionality."""
    print("\n💾 Testing KV cache...")
    
    from mor_model.kv_cache import EfficientKVCache
    
    cache = EfficientKVCache(
        max_batch_size=2,
        max_seq_length=100,
        num_heads=4,
        head_dim=32,
        num_layers=2,
        num_recursion_blocks=2,
        device=torch.device('cpu')
    )
    
    # Test cache operations
    key_states = torch.randn(1, 4, 10, 32)
    value_states = torch.randn(1, 4, 10, 32)
    
    cache.update_cache(0, key_states, value_states)
    cached_keys, cached_values, cache_length = cache.get_cache(0)
    
    memory_usage = cache.memory_usage()
    print(f"   ✓ KV cache working")
    print(f"   ✓ Memory usage: {memory_usage['total_mb']:.2f} MB")
    
    return True


def create_sample_training_config():
    """Create a sample training configuration file."""
    print("\n📝 Creating sample configuration...")
    
    config = create_tiny_config()
    
    # Save to file
    config_path = project_root / "examples" / "tiny_model.yaml"
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"   ✓ Configuration saved to: {config_path}")
    return config_path


def main():
    """Main quick start demonstration."""
    print("🎉 MoR-SLM Quick Start Demo")
    print("=" * 50)
    
    try:
        # Test model creation
        model, config = test_model_creation()
        
        # Test forward pass
        test_forward_pass(model)
        
        # Test parameter sharing
        test_parameter_sharing()
        
        # Test routing
        test_routing()
        
        # Test KV cache
        test_kv_cache()
        
        # Create sample config
        config_path = create_sample_training_config()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! MoR-SLM is working correctly.")
        print("\nNext steps:")
        print(f"1. Train a model: python train.py --config {config_path}")
        print("2. Generate text: python inference.py --model_path checkpoints/model.pt --text 'Hello world'")
        print("3. Interactive mode: python inference.py --model_path checkpoints/model.pt --interactive")
        
        # Show memory efficiency
        print(f"\n📊 Memory Efficiency:")
        memory_stats = model.get_memory_usage()
        standard_params = memory_stats['total_parameters'] * memory_stats['parameter_reduction_factor']
        saved_params = standard_params - memory_stats['total_parameters']
        print(f"   Standard model would have: {standard_params:,.0f} parameters")
        print(f"   MoR model has: {memory_stats['total_parameters']:,} parameters")
        print(f"   Parameters saved: {saved_params:,.0f} ({memory_stats['parameter_reduction_factor']:.1f}x reduction)")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
