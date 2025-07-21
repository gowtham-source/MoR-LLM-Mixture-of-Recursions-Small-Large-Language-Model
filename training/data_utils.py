"""
Data utilities for MoR-SLM training.
"""

import torch
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Union
import logging


class MoRDataCollator:
    """Data collator for MoR-SLM training."""
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate examples into a batch."""
        batch = {}
        
        # Get input_ids from examples
        input_ids = [example['input_ids'] for example in examples]
        
        # Pad sequences - ensure input_ids are Long tensors for embedding
        batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids, dtype=torch.long) for ids in input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id or 0
        )
        
        # Create attention mask
        batch['attention_mask'] = (batch['input_ids'] != (self.tokenizer.pad_token_id or 0)).long()
        
        # Labels are the same as input_ids for causal LM
        batch['labels'] = batch['input_ids'].clone()
        
        # Truncate if necessary
        if batch['input_ids'].size(1) > self.max_length:
            batch['input_ids'] = batch['input_ids'][:, :self.max_length]
            batch['attention_mask'] = batch['attention_mask'][:, :self.max_length]
            batch['labels'] = batch['labels'][:, :self.max_length]
        
        return batch


def preprocess_function(examples, tokenizer, max_length: int = 512):
    """Preprocess text examples for training."""
    # Tokenize the texts
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding=False,
        max_length=max_length,
        return_overflowing_tokens=False,
    )
    
    return tokenized


def create_dataloader(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    split: str = 'train',
    batch_size: int = 8,
    max_length: int = 512,
    num_workers: int = 4,
    is_distributed: bool = False,
    tokenizer_name: str = "gpt2"
) -> DataLoader:
    """Create a DataLoader for training or evaluation."""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    
    # Preprocess dataset
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {split} dataset"
    )
    
    # Create data collator
    data_collator = MoRDataCollator(tokenizer, max_length)
    
    # Create sampler for distributed training
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(tokenized_dataset, shuffle=(split == 'train'))
    
    # Create DataLoader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=(split == 'train' and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def create_recursion_depth_schedule(
    batch_size: int, 
    seq_len: int, 
    min_depth: int = 1, 
    max_depth: int = 4,
    strategy: str = "uniform"
) -> torch.Tensor:
    """
    Create recursion depth schedule for training.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        min_depth: Minimum recursion depth
        max_depth: Maximum recursion depth
        strategy: Scheduling strategy ("uniform", "progressive", "random")
    
    Returns:
        Tensor of shape [batch_size, seq_len] with recursion depths
    """
    if strategy == "uniform":
        # All tokens use maximum depth
        depths = torch.full((batch_size, seq_len), max_depth, dtype=torch.long)
    
    elif strategy == "progressive":
        # Gradually increase depth during training
        # This would typically be called with different parameters during training
        depths = torch.randint(min_depth, max_depth + 1, (batch_size, seq_len))
    
    elif strategy == "random":
        # Random depths for each token
        depths = torch.randint(min_depth, max_depth + 1, (batch_size, seq_len))
    
    elif strategy == "position_based":
        # Depth based on position in sequence (later positions get more depth)
        position_weights = torch.linspace(0, 1, seq_len)
        depths = torch.zeros(batch_size, seq_len, dtype=torch.long)
        for i in range(seq_len):
            depth = min_depth + int(position_weights[i] * (max_depth - min_depth))
            depths[:, i] = depth
    
    else:
        raise ValueError(f"Unknown recursion depth strategy: {strategy}")
    
    return depths
