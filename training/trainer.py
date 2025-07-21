"""
Training module for MoR-SLM with support for FSDP and efficient training.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import wandb
from tqdm import tqdm
import os
from typing import Dict, Optional, Tuple, List
import logging

from mor_model import MoRForCausalLM, MoRConfig
from mor_model.recursive_layers import RecursiveTransformerBlock
from training.data_utils import create_dataloader
from training.utils import get_lr_scheduler, save_checkpoint, load_checkpoint


class MoRTrainer:
    """
    Trainer for MoR-SLM with support for distributed training and efficient optimization.
    """
    
    def __init__(self, config_path: str, resume_from: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize distributed training if available
        self.setup_distributed()
        
        # Initialize model
        self.model = self._create_model()
        
        # Setup FSDP if enabled
        if self.config['training'].get('use_fsdp', False) and self.is_distributed:
            self.model = self._setup_fsdp(self.model)
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup data loaders
        self.train_loader, self.eval_loader = self._create_dataloaders()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # Setup wandb logging
        if self.is_main_process() and self.config['training'].get('use_wandb', False):
            wandb.init(
                project="mor-slm",
                config=self.config,
                name=f"mor_{self.config['model']['name']}"
            )
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_distributed(self):
        """Setup distributed training."""
        self.is_distributed = dist.is_available() and dist.is_initialized()
        if self.is_distributed:
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            torch.cuda.set_device(self.local_rank)
        else:
            self.local_rank = 0
            self.world_size = 1
            self.rank = 0
        
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
    
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0
    
    def _create_model(self) -> MoRForCausalLM:
        """Create MoR model from configuration."""
        model_config = MoRConfig(**self.config['model'])
        model = MoRForCausalLM(model_config)
        model.to(self.device)
        
        if self.is_main_process():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.info(f"Model created with {total_params:,} total parameters")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
            
            # Log memory usage
            memory_stats = model.get_memory_usage()
            self.logger.info(f"Parameter reduction factor: {memory_stats['parameter_reduction_factor']:.2f}")
        
        return model
    
    def _setup_fsdp(self, model: nn.Module) -> FSDP:
        """Setup Fully Sharded Data Parallel."""
        auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls={RecursiveTransformerBlock}
        )
        
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=None,  # Can be configured for fp16/bf16
            device_id=self.local_rank,
            sync_module_states=True,
        )
        
        self.logger.info("FSDP setup completed")
        return fsdp_model
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        train_config = self.config['training']
        
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name or 'embeddings' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_groups = [
            {'params': decay_params, 'weight_decay': train_config['weight_decay']},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = optim.AdamW(
            optimizer_groups,
            lr=train_config['learning_rate'],
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        train_config = self.config['training']
        return get_lr_scheduler(
            self.optimizer,
            warmup_steps=train_config['warmup_steps'],
            max_steps=train_config['max_steps']
        )
    
    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create training and evaluation data loaders."""
        data_config = self.config['data']
        train_config = self.config['training']
        
        train_loader = create_dataloader(
            dataset_name=data_config['dataset_name'],
            dataset_config=data_config.get('dataset_config'),
            split='train',
            batch_size=train_config['batch_size'],
            max_length=data_config['max_length'],
            num_workers=data_config.get('preprocessing_num_workers', 4),
            is_distributed=self.is_distributed
        )
        
        eval_loader = create_dataloader(
            dataset_name=data_config['dataset_name'],
            dataset_config=data_config.get('dataset_config'),
            split='validation',
            batch_size=train_config['batch_size'],
            max_length=data_config['max_length'],
            num_workers=data_config.get('preprocessing_num_workers', 4),
            is_distributed=self.is_distributed
        )
        
        return train_loader, eval_loader
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        
        # Scale loss for gradient accumulation
        train_config = self.config['training']
        loss = loss / train_config['gradient_accumulation_steps']
        
        # Backward pass
        loss.backward()
        
        # Log auxiliary losses if available
        aux_losses = outputs.get('auxiliary_losses', []) if isinstance(outputs, dict) else []
        total_aux_loss = sum(aux_losses) if aux_losses else 0.0
        
        return {
            'loss': loss.item() * train_config['gradient_accumulation_steps'],
            'auxiliary_loss': total_aux_loss.item() if isinstance(total_aux_loss, torch.Tensor) else total_aux_loss,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single evaluation step."""
        self.model.eval()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
        
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        aux_losses = outputs.get('auxiliary_losses', []) if isinstance(outputs, dict) else []
        total_aux_loss = sum(aux_losses) if aux_losses else 0.0
        
        return {
            'eval_loss': loss.item(),
            'eval_auxiliary_loss': total_aux_loss.item() if isinstance(total_aux_loss, torch.Tensor) else total_aux_loss
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        train_config = self.config['training']
        
        total_loss = 0.0
        total_aux_loss = 0.0
        num_steps = 0
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.epoch}",
            disable=not self.is_main_process()
        )
        
        for step, batch in enumerate(progress_bar):
            # Training step
            step_metrics = self.train_step(batch)
            
            total_loss += step_metrics['loss']
            total_aux_loss += step_metrics['auxiliary_loss']
            num_steps += 1
            
            # Gradient accumulation
            if (step + 1) % train_config['gradient_accumulation_steps'] == 0:
                # Gradient clipping
                if train_config.get('max_grad_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        train_config['max_grad_norm']
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % train_config.get('log_steps', 10) == 0 and self.is_main_process():
                    avg_loss = total_loss / num_steps
                    avg_aux_loss = total_aux_loss / num_steps
                    
                    self.logger.info(
                        f"Step {self.global_step}: loss={avg_loss:.4f}, "
                        f"aux_loss={avg_aux_loss:.4f}, lr={step_metrics['learning_rate']:.2e}"
                    )
                    
                    if self.config['training'].get('use_wandb', False):
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/auxiliary_loss': avg_aux_loss,
                            'train/learning_rate': step_metrics['learning_rate'],
                            'train/global_step': self.global_step
                        })
                
                # Evaluation
                if self.global_step % train_config['eval_steps'] == 0:
                    eval_metrics = self.evaluate()
                    if self.is_main_process():
                        self.logger.info(f"Eval metrics: {eval_metrics}")
                        
                        if self.config['training'].get('use_wandb', False):
                            wandb.log(eval_metrics)
                
                # Save checkpoint
                if self.global_step % train_config['save_steps'] == 0:
                    if self.is_main_process():
                        self.save_checkpoint()
                
                # Check if max steps reached
                if self.global_step >= train_config['max_steps']:
                    break
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{step_metrics['loss']:.4f}",
                'lr': f"{step_metrics['learning_rate']:.2e}"
            })
        
        return {
            'train_loss': total_loss / num_steps if num_steps > 0 else 0.0,
            'train_auxiliary_loss': total_aux_loss / num_steps if num_steps > 0 else 0.0
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_aux_loss = 0.0
        num_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating", disable=not self.is_main_process()):
                step_metrics = self.eval_step(batch)
                total_loss += step_metrics['eval_loss']
                total_aux_loss += step_metrics['eval_auxiliary_loss']
                num_steps += 1
        
        avg_loss = total_loss / num_steps if num_steps > 0 else float('inf')
        avg_aux_loss = total_aux_loss / num_steps if num_steps > 0 else 0.0
        
        # Update best eval loss
        if avg_loss < self.best_eval_loss:
            self.best_eval_loss = avg_loss
            if self.is_main_process():
                self.save_checkpoint(is_best=True)
        
        return {
            'eval/loss': avg_loss,
            'eval/auxiliary_loss': avg_aux_loss,
            'eval/best_loss': self.best_eval_loss
        }
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        train_config = self.config['training']
        
        while self.global_step < train_config['max_steps']:
            epoch_metrics = self.train_epoch()
            
            if self.is_main_process():
                self.logger.info(f"Epoch {self.epoch} completed: {epoch_metrics}")
            
            self.epoch += 1
            
            if self.global_step >= train_config['max_steps']:
                break
        
        # Final evaluation
        final_eval_metrics = self.evaluate()
        if self.is_main_process():
            self.logger.info(f"Final evaluation: {final_eval_metrics}")
            self.save_checkpoint(is_final=True)
        
        if self.config['training'].get('use_wandb', False) and self.is_main_process():
            wandb.finish()
        
        self.logger.info("Training completed!")
    
    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Determine checkpoint filename
        model_name = "mor_small"  # Fixed model name since we removed it from config
        if is_final:
            filename = f"{model_name}_final.pt"
        elif is_best:
            filename = f"{model_name}_best.pt"
        else:
            filename = f"{model_name}_step_{self.global_step}.pt"
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_eval_loss': self.best_eval_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_eval_loss = checkpoint['best_eval_loss']
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"Resuming from step {self.global_step}, epoch {self.epoch}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MoR-SLM model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    trainer = MoRTrainer(args.config, args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
