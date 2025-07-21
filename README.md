# MoR-SLM: Mixture-of-Recursions Small Language Model

A novel transformer architecture implementing the Mixture-of-Recursions (MoR) framework for efficient small language models with limited compute resources.

## Architecture Overview

This implementation features:
- **Recursive Parameter Sharing**: Middle-Cycle strategy for optimal parameter efficiency
- **Dynamic Routing**: Expert-Choice and Token-Choice routing mechanisms
- **Efficient KV Caching**: Optimized inference with memory management
- **FSDP Support**: Distributed training capabilities
- **Variable Recursion Depths**: Adaptive computation per token

## Key Components

- `mor_model/`: Core MoR transformer implementation
- `routing/`: Dynamic routing strategies
- `training/`: Training loop and optimization
- `inference/`: Efficient inference engine
- `data/`: Data processing utilities
- `configs/`: Model and training configurations

## Quick Start

```bash
# Install dependencies
uv pip install -r requirements.txt | pip install -r requirements.txt

# Train a small model
python train.py --config configs/small_model.yaml

# Run inference
python inference.py --model_path checkpoints/mor_small --text "Once upon a time there was a"

or

# Using uv package manager
uv init
uv add -r requirements.txt

uv run python train.py --config configs/small_model.yaml # For training small model (MoR-SLM with tiny dataset  for testing)
uv run python train.py --config configs/medium_llm.yaml # For training medium model (MoR-SLM)
uv run python train.py --config configs/large_llm.yaml # For training large language model (MoR-LLM)

uv run python inference.py --model_path checkpoints/mor_small --text "Once upon a time there was a" # For inference
```

## Paper Reference

Based on "Mixture-of-Recursions: A Unified Framework for Parameter Sharing and Adaptive Computation in Transformer Language Models" (arXiv:2507.10524v1)
