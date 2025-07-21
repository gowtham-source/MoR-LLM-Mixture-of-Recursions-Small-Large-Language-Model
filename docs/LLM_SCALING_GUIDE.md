# MoR-SLM Large Language Model Scaling Guide

This guide explains how to scale your MoR-SLM from a small research model to a production-ready Large Language Model (LLM).

## 🚀 Model Size Scaling

### Current Small Model vs Large LLM

| Component | Small Model | Large LLM (7B) | Scaling Factor |
|-----------|-------------|-----------------|----------------|
| **Parameters** | 68M | ~7B | 100x |
| **Hidden Size** | 512 | 4096 | 8x |
| **Layers** | 4 | 32 | 8x |
| **Attention Heads** | 8 | 32 | 4x |
| **Context Length** | 2048 | 4096 | 2x |
| **Recursion Depth** | 4 | 8 | 2x |

### Parameter Scaling Options

#### **1. Medium Model (1.3B parameters)**
```yaml
hidden_size: 2048
intermediate_size: 5504
num_attention_heads: 16
num_recursion_blocks: 16
max_recursion_depth: 6
```

#### **2. Large Model (7B parameters)**
```yaml
hidden_size: 4096
intermediate_size: 11008
num_attention_heads: 32
num_recursion_blocks: 32
max_recursion_depth: 8
```

#### **3. Extra Large Model (13B parameters)**
```yaml
hidden_size: 5120
intermediate_size: 13824
num_attention_heads: 40
num_recursion_blocks: 40
max_recursion_depth: 10
```

## 📊 Dataset Scaling

### Small vs Large Datasets

| Dataset | Size | Tokens | Use Case |
|---------|------|--------|----------|
| **TinyStories** | 10MB | ~2M | Testing/Development |
| **WikiText-103** | 500MB | ~100M | Small-scale experiments |
| **OpenWebText** | 40GB | ~8B | Medium-scale training |
| **C4** | 800GB | ~180B | Large-scale training |
| **RedPajama-1T** | 5TB | ~1.2T | Production LLM training |

### Recommended Dataset Progression

1. **Development**: TinyStories → WikiText-103
2. **Validation**: OpenWebText (subset)
3. **Production**: C4 or RedPajama-1T

## 🖥️ Hardware Requirements

### GPU Memory Requirements (with FSDP)

| Model Size | Min GPUs | Recommended GPUs | Memory per GPU |
|------------|----------|------------------|----------------|
| **1.3B** | 2x A100 40GB | 4x A100 40GB | 40GB |
| **7B** | 4x A100 80GB | 8x A100 80GB | 80GB |
| **13B** | 8x A100 80GB | 8x H100 80GB | 80GB |

### Training Time Estimates

| Model Size | Dataset | GPUs | Training Time |
|------------|---------|------|---------------|
| **1.3B** | OpenWebText | 4x A100 | ~1 week |
| **7B** | C4 (subset) | 8x A100 | ~2-4 weeks |
| **13B** | RedPajama-1T | 8x H100 | ~2-3 months |

## ⚙️ Configuration Changes

### 1. Enable Distributed Training

```yaml
training:
  use_fsdp: true  # ESSENTIAL for large models
  fsdp_sharding_strategy: "full_shard"
  gradient_checkpointing: true
  use_mixed_precision: true
```

### 2. Optimize Memory Usage

```yaml
training:
  gradient_accumulation_steps: 32  # Increase effective batch size
  cpu_offload: true  # If running out of GPU memory
  dataloader_num_workers: 16  # Parallel data loading
```

### 3. Large Dataset Handling

```yaml
data:
  streaming: true  # Stream data instead of loading all
  buffer_size: 10000
  preprocessing_num_workers: 16
```

## 🚀 Training Commands

### Small to Medium Scale (1-4 GPUs)
```bash
# Single GPU
uv run train.py --config configs/large_llm.yaml

# Multi-GPU (same node)
torchrun --nproc_per_node=4 train.py --config configs/large_llm.yaml
```

### Large Scale (Multi-node)
```bash
# Node 0 (master)
torchrun --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" \
         --master_port=12355 --nproc_per_node=8 \
         train.py --config configs/large_llm.yaml

# Node 1 (worker)
torchrun --nnodes=2 --node_rank=1 --master_addr="192.168.1.1" \
         --master_port=12355 --nproc_per_node=8 \
         train.py --config configs/large_llm.yaml
```

## 🔧 Performance Optimizations

### 1. Mixed Precision Training
- **FP16**: 2x memory reduction, slight accuracy loss
- **BF16**: Better numerical stability (recommended for large models)

### 2. Gradient Checkpointing
- Trades compute for memory
- Essential for large models with limited GPU memory

### 3. FSDP Optimizations
```yaml
fsdp_backward_prefetch: "backward_pre"
fsdp_forward_prefetch: true
fsdp_use_orig_params: false
```

## 📈 Monitoring and Evaluation

### Key Metrics to Track
1. **Training Loss**: Should decrease steadily
2. **Perplexity**: Lower is better
3. **GPU Memory Usage**: Should be near capacity but not OOM
4. **Throughput**: Tokens/second/GPU
5. **Model Quality**: Periodic evaluation on benchmarks

### Evaluation Benchmarks
- **HellaSwag**: Commonsense reasoning
- **MMLU**: Multi-task language understanding  
- **HumanEval**: Code generation
- **GSM8K**: Mathematical reasoning

## 🎯 Production Deployment

### Model Serving
```python
# Large model inference
engine = MoRInferenceEngine("checkpoints/large_model_final.pt")
result = engine.generate(
    prompt="Explain quantum computing",
    max_new_tokens=512,
    temperature=0.7,
    recursion_strategy="adaptive"
)
```

### Optimization for Inference
- **Model Quantization**: INT8/INT4 for faster inference
- **KV Cache Optimization**: Efficient memory usage
- **Batch Inference**: Process multiple requests together

## 💡 Best Practices

1. **Start Small**: Validate on smaller models first
2. **Incremental Scaling**: Gradually increase model size
3. **Monitor Resources**: Watch GPU memory and utilization
4. **Checkpoint Frequently**: Save progress regularly
5. **Evaluate Often**: Track model quality throughout training
6. **Use Wandb**: Log metrics and visualize training progress

## 🚨 Common Issues and Solutions

### Out of Memory (OOM)
- Reduce batch size
- Enable gradient checkpointing
- Use CPU offloading
- Increase gradient accumulation steps

### Slow Training
- Increase number of data workers
- Use faster storage (NVMe SSD)
- Optimize data preprocessing
- Use mixed precision training

### Poor Model Quality
- Increase model size gradually
- Use better datasets
- Tune learning rate and warmup
- Train for more steps

## 📚 Next Steps

1. **Choose your target model size** based on available hardware
2. **Select appropriate dataset** for your use case
3. **Configure distributed training** for your hardware setup
4. **Start with medium-scale experiments** before full production training
5. **Monitor and iterate** based on results
