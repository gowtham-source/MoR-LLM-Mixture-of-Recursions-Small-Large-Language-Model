# Comprehensive Implementation Guide for Mixture-of-Recursions (MoR)

## Overview

You are tasked with implementing the **Mixture-of-Recursions (MoR)** framework, a unified architecture that combines parameter sharing, adaptive computation, and efficient KV caching for Transformer language models. This implementation will create a system that dynamically adjusts recursion depths for individual tokens while maintaining computational efficiency.

## Core Architecture Components

### 1. Base Recursive Transformer Implementation

**Create the foundational recursive transformer with parameter sharing:**

```python
# Key architectural requirements:
- Implement 4 parameter-sharing strategies: Cycle, Sequence, Middle-Cycle, Middle-Sequence
- Focus on Middle-Cycle as the optimal strategy (preserves first/last layers, shares middle layers)
- Support Nr recursion blocks sharing parameters Φ'
- Enable variable recursion depths per token (1 to Nr)

# Mathematical foundation:
# Standard: h_t^(ℓ+1) = f(h_t^ℓ; Φ_ℓ)
# Recursive: Uses shared parameters Φ' across recursion blocks
```

**Implementation specifications:**

- Use Llama-based Transformer architecture as the foundation
- Support model sizes from 135M to 1.7B+ parameters
- Implement parameter reduction by factor of Nr (recursion number)
- Enable Fully Sharded Data Parallel (FSDP) optimization for distributed training


### 2. Dynamic Routing System

**Implement two routing strategies with their specific mechanisms:**

#### Expert-Choice Routing

```python
# Core logic:
def expert_choice_routing(hidden_states, router_params, recursion_step):
    # Compute routing scores: g_t^r = G(θ_r^T * H_t^r)
    # Apply β-percentile threshold: P_β(G^r)
    # Select top-k tokens hierarchically (only previously selected can continue)
    # Update hidden states with residual connections
    
    # Critical implementation details:
    - Use sigmoid activation function (G)
    - Implement linear router architecture (not MLP)
    - Apply auxiliary loss for causality violation mitigation
    - Support hierarchical filtering: tokens selected at step r can proceed to r+1
    - Maintain static compute budget through predetermined top-k selection

# Mathematical formulation:
H_t^(r+1) = {
    g_t^r * f(H_t^r, Φ') + H_t^r,  if g_t^r > P_β(G^r)
    H_t^r,                           otherwise
}
```


#### Token-Choice Routing

```python
# Core logic:
def token_choice_routing(hidden_states, router_params):
    # Compute routing scores for all experts: g_t = G(θ_r^T * H_t^1)
    # Assign token to expert: i = argmax_j(g_t^j)
    # Process token through i sequential recursions
    
    # Critical implementation details:
    - Use softmax activation with MLP router
    - Implement balancing loss to prevent load imbalance
    - Support top-1 gating (each token assigned to one expert/depth)
    - Apply router z-loss for stability

# Balancing loss formula:
L_Balance = α * Σ(f_i * P_i) where:
- f_i = (Nr/T) * Σ I(Token t selects Expert i)
- P_i = (1/T) * Σ g_t^i
```


### 3. KV Caching Strategies

**Implement two KV caching mechanisms:**

#### Recursion-wise KV Caching

```python
# Implementation requirements:
- Cache KV pairs only for tokens routed to each recursion step
- Restrict attention to locally cached tokens per recursion depth
- Reduce KV memory to (Nr+1)/(2*Nr) of vanilla Transformer
- Reduce attention FLOPs to (k/N_ctx)^2 factor

# Key features:
- Block-local computation for memory efficiency
- Variable cache sizes per recursion depth based on routing decisions
- Compatible with both routing strategies
```


#### Recursive KV Sharing

```python
# Implementation requirements:
- Cache KV pairs exclusively at first recursion step
- Reuse first-step KV pairs across all subsequent recursions
- Maintain full sequence length for keys and values
- Reduce KV memory to 1/Nr of vanilla Transformer

# Key features:
- Skip KV projection and prefill operations at shared depths
- Ensure all tokens access past context without recomputation
- Handle potential distribution mismatch between recursion steps
```


### 4. Training Infrastructure

**Implement comprehensive training system:**

```python
# Training components required:
1. End-to-end router training from scratch
2. Auxiliary loss mechanisms for expert-choice routing
3. Balancing loss for token-choice routing
4. Router z-loss for stability
5. Trapezoid learning rate scheduler with checkpoint reuse
6. IsoFLOPs training capability for fair comparison

# Loss functions to implement:
- Primary language modeling loss
- Auxiliary loss: Binary cross-entropy for top-k prediction
- Balancing loss: Load balancing across experts
- Router z-loss: Regularization for router stability

# Training optimizations:
- Continuous depth-wise batching for inference
- FlashAttention 2 integration
- Static-sized cache compatibility with torch.compile
```


### 5. Inference Optimization

**Create efficient inference system:**

```python
# Key inference features:
1. Continuous depth-wise batching implementation
2. Early-exit token handling with FIFO queuing
3. Variable-length KV cache support within batches
4. Hierarchical token processing for expert-choice
5. Batch accumulation for exited tokens

# Performance targets:
- Up to 2.18× throughput improvement over vanilla Transformers
- Reduced memory footprint through selective KV caching
- Support for test-time scaling via deeper recursion
```


## Implementation Phases

### Phase 1: Core Architecture (Weeks 1-2)

1. Implement base recursive transformer with Middle-Cycle parameter sharing
2. Create flexible layer architecture supporting Nr recursion blocks
3. Implement basic forward pass with shared parameters
4. Add support for variable model sizes (135M-1.7B parameters)

### Phase 2: Routing Systems (Weeks 3-4)

1. Implement expert-choice routing with hierarchical filtering
2. Implement token-choice routing with load balancing
3. Create router architectures (linear for expert-choice, MLP for token-choice)
4. Add activation functions (sigmoid for expert-choice, softmax for token-choice)

### Phase 3: KV Caching (Weeks 5-6)

1. Implement recursion-wise KV caching with selective storage
2. Implement recursive KV sharing with first-step reuse
3. Add attention mechanisms for variable cache sizes
4. Optimize memory access patterns

### Phase 4: Training System (Weeks 7-8)

1. Implement auxiliary losses (auxiliary loss, balancing loss, z-loss)
2. Create end-to-end training pipeline
3. Add learning rate scheduling and checkpoint management
4. Implement evaluation metrics and benchmarks

### Phase 5: Inference Optimization (Weeks 9-10)

1. Implement continuous depth-wise batching
2. Add early-exit handling and token queuing
3. Optimize for FlashAttention 2 and torch.compile
4. Performance testing and throughput measurement

## Technical Specifications

### Model Configuration

```python
# Supported configurations:
model_sizes = [135M, 360M, 730M, 1.7B]  # Base sizes before parameter sharing
recursion_depths = [2, 3, 4]  # Nr values
parameter_sharing = "Middle-Cycle"  # Optimal strategy
sequence_lengths = [2048, 4096, 8192]  # Context lengths
```


### Training Hyperparameters

```python
# Key hyperparameters to implement:
learning_rate_schedule = "trapezoid"  # Warmup, stable, cooldown
auxiliary_loss_weight = 0.1  # For expert-choice routing
balancing_loss_weight = 0.01  # For token-choice routing
z_loss_coefficient = 1e-4  # Router stability
batch_size = 32  # Base batch size for throughput measurement
```


### Evaluation Metrics

```python
# Required evaluation capabilities:
1. Validation perplexity (negative log-likelihood)
2. Few-shot accuracy on 6 benchmarks (LD, HS, PQ, WG, ARC, MMLU)
3. Inference throughput (tokens/second)
4. Memory usage (KV cache size, peak memory)
5. FLOPs efficiency (training and inference)
6. Router metrics (dead token ratio, MaxVio for load balancing)
```


## Expected Outcomes

### Performance Targets

- **Efficiency**: 25-50% parameter reduction with competitive performance
- **Throughput**: Up to 2.18× inference speedup vs vanilla Transformers
- **Quality**: Match or exceed vanilla Transformer accuracy with fewer parameters
- **Scalability**: Consistent performance improvements across model scales


### Key Validation Points

1. **Parameter Efficiency**: MoR should achieve better perplexity than recursive baselines
2. **Adaptive Computation**: Routing should reflect token semantic importance
3. **Memory Efficiency**: KV caching should reduce memory usage without major performance loss
4. **Training Stability**: Auxiliary losses should enable stable router learning
5. **Inference Speed**: Continuous depth-wise batching should provide significant throughput gains

This implementation will create a complete MoR system that unifies parameter sharing, adaptive computation, and efficient KV caching, demonstrating that large-model quality can be achieved without large-model computational costs.
