"""
Unit tests for MoR-SLM model components.
"""

import torch
import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mor_model import MoRConfig, MoRForCausalLM, RecursiveTransformerBlock
from mor_model.routing import ExpertChoiceRouter, TokenChoiceRouter
from mor_model.kv_cache import EfficientKVCache
from mor_model.recursive_layers import ParameterSharingManager


class TestMoRConfig:
    """Test MoR configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = MoRConfig()
        assert config.vocab_size == 32000
        assert config.hidden_size == 512
        assert config.num_recursion_blocks == 4
        assert config.parameter_sharing_strategy == "middle_cycle"
        assert config.routing_strategy == "expert_choice"
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = MoRConfig()
        config.validate()  # Should not raise
        
        # Test invalid configuration
        config.max_recursion_depth = 0
        with pytest.raises(AssertionError):
            config.validate()


class TestParameterSharingManager:
    """Test parameter sharing strategies."""
    
    def test_middle_cycle_strategy(self):
        """Test middle-cycle parameter sharing."""
        manager = ParameterSharingManager(
            num_layers=6,
            num_recursion_blocks=4,
            strategy="middle_cycle"
        )
        
        # First layer should use block 0
        assert manager.get_shared_block_idx(0) == 0
        # Last layer should use block 3
        assert manager.get_shared_block_idx(5) == 3
        # Middle layers should cycle through blocks 1-2
        assert manager.get_shared_block_idx(1) in [1, 2]
        assert manager.get_shared_block_idx(2) in [1, 2]
    
    def test_cycle_strategy(self):
        """Test cycle parameter sharing."""
        manager = ParameterSharingManager(
            num_layers=8,
            num_recursion_blocks=4,
            strategy="cycle"
        )
        
        for layer_idx in range(8):
            expected_block = layer_idx % 4
            assert manager.get_shared_block_idx(layer_idx) == expected_block


class TestMoRRouting:
    """Test routing mechanisms."""
    
    def test_expert_choice_router(self):
        """Test expert-choice routing."""
        hidden_size = 512
        batch_size = 2
        seq_len = 10
        
        router = ExpertChoiceRouter(hidden_size)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        scores, mask, aux_loss = router(hidden_states, recursion_step=0)
        
        assert scores.shape == (batch_size, seq_len)
        assert mask.shape == (batch_size, seq_len)
        assert isinstance(aux_loss, torch.Tensor)
        assert torch.all((mask >= 0) & (mask <= 1))  # Binary mask
    
    def test_token_choice_router(self):
        """Test token-choice routing."""
        hidden_size = 512
        max_depth = 4
        batch_size = 2
        seq_len = 10
        
        router = TokenChoiceRouter(hidden_size, max_depth)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        depth_probs, continue_mask = router(hidden_states, current_depth=1)
        
        assert depth_probs.shape == (batch_size, seq_len, max_depth)
        assert continue_mask.shape == (batch_size, seq_len)
        assert torch.allclose(depth_probs.sum(dim=-1), torch.ones(batch_size, seq_len))


class TestEfficientKVCache:
    """Test KV cache implementation."""
    
    def test_cache_initialization(self):
        """Test KV cache initialization."""
        cache = EfficientKVCache(
            max_batch_size=2,
            max_seq_length=100,
            num_heads=8,
            head_dim=64,
            num_layers=4,
            num_recursion_blocks=4,
            device=torch.device('cpu')
        )
        
        assert len(cache.key_cache) == 4
        assert len(cache.value_cache) == 4
        assert len(cache.recursion_caches) == 4
    
    def test_cache_update_and_retrieval(self):
        """Test cache update and retrieval."""
        cache = EfficientKVCache(
            max_batch_size=1,
            max_seq_length=100,
            num_heads=8,
            head_dim=64,
            num_layers=4,
            num_recursion_blocks=4,
            device=torch.device('cpu')
        )
        
        # Test cache update
        key_states = torch.randn(1, 8, 10, 64)
        value_states = torch.randn(1, 8, 10, 64)
        
        cache.update_cache(0, key_states, value_states)
        
        # Test cache retrieval
        cached_keys, cached_values, cache_length = cache.get_cache(0)
        assert cached_keys.shape[2] >= 10  # At least 10 positions cached
        assert cached_values.shape[2] >= 10


class TestMoRModel:
    """Test complete MoR model."""
    
    def test_model_creation(self):
        """Test model creation with small config."""
        config = MoRConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=1024,
            num_attention_heads=4,
            num_recursion_blocks=2,
            max_recursion_depth=2
        )
        
        model = MoRForCausalLM(config)
        
        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        
        # Check memory usage stats
        memory_stats = model.get_memory_usage()
        assert 'total_parameters' in memory_stats
        assert 'parameter_reduction_factor' in memory_stats
    
    def test_forward_pass(self):
        """Test forward pass through model."""
        config = MoRConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=512,
            num_attention_heads=4,
            num_recursion_blocks=2,
            max_recursion_depth=2
        )
        
        model = MoRForCausalLM(config)
        
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
        else:
            logits = outputs[0]
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
    
    def test_generation_preparation(self):
        """Test generation input preparation."""
        config = MoRConfig(
            vocab_size=1000,
            hidden_size=128,
            num_recursion_blocks=2
        )
        
        model = MoRForCausalLM(config)
        
        input_ids = torch.tensor([[1, 2, 3, 4]])
        attention_mask = torch.ones_like(input_ids)
        
        inputs = model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        assert 'input_ids' in inputs
        assert 'attention_mask' in inputs
        assert 'position_ids' in inputs


def run_tests():
    """Run all tests."""
    import subprocess
    import sys
    
    # Run pytest
    result = subprocess.run([
        sys.executable, '-m', 'pytest', 
        __file__, '-v'
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    # Simple test runner
    print("Running MoR-SLM tests...")
    
    try:
        # Test config
        test_config = TestMoRConfig()
        test_config.test_default_config()
        test_config.test_config_validation()
        print("✓ Config tests passed")
        
        # Test parameter sharing
        test_sharing = TestParameterSharingManager()
        test_sharing.test_middle_cycle_strategy()
        test_sharing.test_cycle_strategy()
        print("✓ Parameter sharing tests passed")
        
        # Test routing
        test_routing = TestMoRRouting()
        test_routing.test_expert_choice_router()
        test_routing.test_token_choice_router()
        print("✓ Routing tests passed")
        
        # Test KV cache
        test_cache = TestEfficientKVCache()
        test_cache.test_cache_initialization()
        test_cache.test_cache_update_and_retrieval()
        print("✓ KV cache tests passed")
        
        # Test model
        test_model = TestMoRModel()
        test_model.test_model_creation()
        test_model.test_forward_pass()
        test_model.test_generation_preparation()
        print("✓ Model tests passed")
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
