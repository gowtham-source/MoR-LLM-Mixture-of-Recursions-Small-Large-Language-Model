# MoR-SLM Dataset Guide

## Overview
This guide explains how to use datasets with MoR-SLM, including built-in datasets, custom datasets, and data formatting requirements.

## Built-in Datasets (Hugging Face Hub)

### 1. WikiText (Recommended for beginners)
```yaml
data:
  dataset_name: "wikitext"
  dataset_config: "wikitext-2-raw-v1"  # Options: wikitext-2-raw-v1, wikitext-103-raw-v1
  max_length: 512
```

### 2. OpenWebText
```yaml
data:
  dataset_name: "openwebtext"
  dataset_config: null
  max_length: 512
```

### 3. BookCorpus
```yaml
data:
  dataset_name: "bookcorpus"
  dataset_config: null
  max_length: 512
```

### 4. C4 (Colossal Clean Crawled Corpus)
```yaml
data:
  dataset_name: "c4"
  dataset_config: "en"
  max_length: 512
```

### 5. TinyStories (Good for testing)
```yaml
data:
  dataset_name: "roneneldan/TinyStories"
  dataset_config: null
  max_length: 256
```

## Custom Dataset Formats

### 1. Text Files (.txt)
Simple text files with one document per line or paragraph-separated documents.

**Format:**
```
This is the first document. It can be multiple sentences.

This is the second document. Each document should contain meaningful text.

This is the third document. Documents are separated by blank lines.
```

### 2. JSON Lines (.jsonl)
Each line is a JSON object with a "text" field.

**Format:**
```json
{"text": "This is the first document with some meaningful text."}
{"text": "This is the second document. It can contain multiple sentences."}
{"text": "This is the third document with more content."}
```

### 3. CSV Files (.csv)
CSV files with a "text" column.

**Format:**
```csv
text,label
"This is the first document.",positive
"This is the second document.",neutral
"This is the third document.",negative
```

### 4. Parquet Files (.parquet)
Efficient columnar storage format with a "text" column.

## Dataset Configuration Examples

### Small Dataset (Quick Testing)
```yaml
# configs/small_dataset.yaml
model:
  vocab_size: 5000
  hidden_size: 256
  num_recursion_blocks: 2
  max_recursion_depth: 2

data:
  dataset_name: "roneneldan/TinyStories"
  max_length: 256
  preprocessing_num_workers: 2

training:
  batch_size: 4
  max_steps: 1000
```

### Medium Dataset (Real Training)
```yaml
# configs/medium_dataset.yaml
model:
  vocab_size: 32000
  hidden_size: 512
  num_recursion_blocks: 4
  max_recursion_depth: 4

data:
  dataset_name: "wikitext"
  dataset_config: "wikitext-103-raw-v1"
  max_length: 512
  preprocessing_num_workers: 4

training:
  batch_size: 8
  max_steps: 10000
```

### Custom Local Dataset
```yaml
# configs/custom_dataset.yaml
model:
  vocab_size: 32000
  hidden_size: 512
  num_recursion_blocks: 4

data:
  dataset_name: "json"
  dataset_config: null
  data_files:
    train: "path/to/train.jsonl"
    validation: "path/to/val.jsonl"
  max_length: 512
  preprocessing_num_workers: 4

training:
  batch_size: 8
  max_steps: 5000
```

## Data Preprocessing

The MoR-SLM automatically handles:
- **Tokenization** using GPT-2 tokenizer (configurable)
- **Truncation** to max_length
- **Padding** for batch processing
- **Attention masks** for proper masking

### Custom Tokenizer
```yaml
data:
  tokenizer_name: "gpt2"  # Options: gpt2, microsoft/DialoGPT-medium, etc.
  dataset_name: "wikitext"
  dataset_config: "wikitext-2-raw-v1"
```

## Dataset Size Recommendations

### For Limited Compute Resources:
- **Tiny**: TinyStories (~10MB) - 1000 steps
- **Small**: WikiText-2 (~12MB) - 5000 steps  
- **Medium**: WikiText-103 (~500MB) - 10000 steps

### For Full Training:
- **Large**: OpenWebText (~50GB) - 100000+ steps
- **Very Large**: C4 (~300GB) - 500000+ steps

## Memory & Performance Tips

1. **Batch Size**: Start with 4-8 for limited GPU memory
2. **Max Length**: Use 256-512 tokens for most tasks
3. **Gradient Accumulation**: Use 2-8 steps if batch size is small
4. **Workers**: Set `preprocessing_num_workers` to your CPU cores

## Example Configurations

### 1. Quick Test Training (5 minutes)
```bash
# Use tiny model with TinyStories
uv run train.py --config examples/tiny_model.yaml
```

### 2. Real Training (few hours)
```yaml
# Create configs/real_training.yaml
data:
  dataset_name: "wikitext"
  dataset_config: "wikitext-103-raw-v1"
  max_length: 512

training:
  batch_size: 8
  gradient_accumulation_steps: 4
  max_steps: 20000
  learning_rate: 3e-4
```

### 3. Custom Dataset Training
```bash
# Prepare your data as JSONL
echo '{"text": "Your custom text here."}' > custom_data.jsonl
echo '{"text": "More custom text content."}' >> custom_data.jsonl

# Update config to use custom data
# Then run training
uv run train.py --config configs/custom_dataset.yaml
```

## Validation & Metrics

The trainer automatically:
- Splits data into train/validation
- Computes perplexity and loss
- Saves checkpoints periodically
- Logs metrics to console (and wandb if enabled)

## Next Steps

1. **Choose a dataset** from the options above
2. **Create/modify config** with your dataset settings  
3. **Start training**: `uv run train.py --config your_config.yaml`
4. **Monitor progress** through logged metrics
5. **Test inference** with trained model
