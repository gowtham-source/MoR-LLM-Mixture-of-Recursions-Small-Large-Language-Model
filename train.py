#!/usr/bin/env python3
"""
Main training script for MoR-SLM.
"""

import os
import sys
import argparse
import logging
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.trainer import MoRTrainer


def setup_distributed():
    """Setup distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize process group
        init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            rank=rank,
            world_size=world_size
        )
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        return True
    return False


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train MoR-SLM model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/small_model.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=0,
        help="Local rank for distributed training"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Setup distributed training
    is_distributed = setup_distributed()
    
    try:
        # Create trainer
        trainer = MoRTrainer(args.config, args.resume)
        
        # Start training
        logger.info("Starting MoR-SLM training...")
        trainer.train()
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Cleanup
        if is_distributed:
            cleanup_distributed()


if __name__ == "__main__":
    main()
