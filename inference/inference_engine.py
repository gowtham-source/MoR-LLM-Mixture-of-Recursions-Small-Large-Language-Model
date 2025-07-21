"""
Inference engine for MoR-SLM with efficient generation and KV caching.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import List, Optional, Dict, Union
import time
import logging

from mor_model import MoRForCausalLM, MoRConfig


class MoRInferenceEngine:
    """
    Efficient inference engine for MoR-SLM with optimized generation.
    """
    
    def __init__(
        self, 
        model_path: str, 
        device: Optional[torch.device] = None,
        tokenizer_name: str = "gpt2",
        max_batch_size: int = 1,
        max_seq_length: int = 2048
    ):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        
        # Load model and tokenizer
        self.model = self._load_model(model_path)
        self.tokenizer = self._load_tokenizer(tokenizer_name)
        
        # Setup KV cache
        self.model.setup_kv_cache(max_batch_size, max_seq_length, self.device)
        
    
    def _load_model(self, model_path: str) -> MoRForCausalLM:
        """Load MoR model from checkpoint."""
        if model_path.endswith('.pt') or model_path.endswith('.pth'):
            # Load from PyTorch checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            config = MoRConfig(**checkpoint['config']['model'])
            model = MoRForCausalLM(config)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Load from Hugging Face format (if implemented)
            raise NotImplementedError("Hugging Face format loading not implemented yet")
        
        model.to(self.device)
        model.eval()
        
        self.logger.info(f"Model loaded from {model_path}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    def _load_tokenizer(self, tokenizer_name: str):
        """Load tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        recursion_strategy: str = "adaptive",
        min_recursion_depth: int = 1,
        max_recursion_depth: int = 4,
        stop_tokens: Optional[List[str]] = None,
    ) -> Union[str, List[str]]:
        """
        Generate text using MoR-SLM.
        
        Args:
            prompt: Input text prompt(s)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            num_return_sequences: Number of sequences to return per prompt
            recursion_strategy: Strategy for recursion depths ("uniform", "adaptive", "progressive")
            min_recursion_depth: Minimum recursion depth
            max_recursion_depth: Maximum recursion depth
            stop_tokens: List of stop tokens
        
        Returns:
            Generated text(s)
        """
        # Handle single prompt
        if isinstance(prompt, str):
            prompts = [prompt]
            return_single = True
        else:
            prompts = prompt
            return_single = False
        
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length - max_new_tokens
        ).to(self.device)
        
        batch_size = inputs.input_ids.shape[0]
        prompt_length = inputs.input_ids.shape[1]
        
        # Clear KV cache
        self.model.clear_kv_cache()
        
        # Generate recursion depths
        recursion_depths = self._create_recursion_schedule(
            batch_size, 
            prompt_length + max_new_tokens,
            recursion_strategy,
            min_recursion_depth,
            max_recursion_depth
        )
        
        # Generation loop
        generated_sequences = []
        
        for _ in range(num_return_sequences):
            generated_ids = self._generate_sequence(
                inputs.input_ids,
                inputs.attention_mask,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                do_sample,
                recursion_depths,
                stop_tokens
            )
            
            generated_sequences.append(generated_ids)
        
        # Decode generated sequences
        results = []
        for seq_batch in generated_sequences:
            batch_results = []
            for seq in seq_batch:
                # Decode only the new tokens
                new_tokens = seq[prompt_length:]
                decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                batch_results.append(decoded)
            results.extend(batch_results)
        
        if return_single and len(results) == 1:
            return results[0]
        
        return results
    
    def _generate_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        do_sample: bool,
        recursion_depths: torch.Tensor,
        stop_tokens: Optional[List[str]]
    ) -> torch.Tensor:
        """Generate a single sequence."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize generation
        generated_ids = input_ids.clone()
        current_attention_mask = attention_mask.clone()
        
        # Convert stop tokens to IDs
        stop_token_ids = set()
        if stop_tokens:
            for token in stop_tokens:
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                stop_token_ids.update(token_ids)
        
        # Generation loop
        for step in range(max_new_tokens):
            current_seq_len = generated_ids.shape[1]
            
            # Get current recursion depths
            current_depths = recursion_depths[:, :current_seq_len]
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=current_attention_mask,
                    recursion_depths=current_depths,
                    use_cache=True
                )
            
            # Get logits for the last token - handle both tuple and dict outputs
            if isinstance(outputs, tuple):
                logits = outputs[0][:, -1, :]  # [batch_size, vocab_size]
            else:
                logits = outputs['logits'][:, -1, :]  # [batch_size, vocab_size]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits_filtered = torch.full_like(logits, float('-inf'))
                logits_filtered.scatter_(1, top_k_indices, top_k_logits)
                logits = logits_filtered
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Sample or select next token
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1)
            else:
                next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
            
            # Update attention mask
            current_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones(batch_size, 1, device=device, dtype=current_attention_mask.dtype)
            ], dim=-1)
            
            # Check for stop tokens
            if stop_token_ids and next_token_ids.item() in stop_token_ids:
                break
            
            # Check for EOS token
            if next_token_ids.item() == self.tokenizer.eos_token_id:
                break
        
        return generated_ids
    
    def _create_recursion_schedule(
        self,
        batch_size: int,
        seq_len: int,
        strategy: str,
        min_depth: int,
        max_depth: int
    ) -> torch.Tensor:
        """Create recursion depth schedule for generation."""
        if strategy == "uniform":
            # Use maximum depth for all tokens
            depths = torch.full((batch_size, seq_len), max_depth, dtype=torch.long, device=self.device)
        
        elif strategy == "adaptive":
            # Adaptive depth based on position and complexity
            # Start with lower depths and increase for later positions
            position_factor = torch.linspace(0, 1, seq_len, device=self.device)
            depths = torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device)
            
            for i in range(seq_len):
                depth = min_depth + int(position_factor[i] * (max_depth - min_depth))
                depths[:, i] = depth
        
        elif strategy == "progressive":
            # Progressive increase in depth
            depths = torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device)
            step_size = max(1, seq_len // (max_depth - min_depth + 1))
            
            current_depth = min_depth
            for i in range(0, seq_len, step_size):
                end_idx = min(i + step_size, seq_len)
                depths[:, i:end_idx] = current_depth
                current_depth = min(current_depth + 1, max_depth)
        
        else:
            raise ValueError(f"Unknown recursion strategy: {strategy}")
        
        return depths
    
    def benchmark(
        self,
        prompt: str = "The future of artificial intelligence is",
        max_new_tokens: int = 100,
        num_runs: int = 5
    ) -> Dict[str, float]:
        """Benchmark inference performance."""
        self.logger.info("Starting inference benchmark...")
        
        times = []
        memory_usage = []
        
        for run in range(num_runs):
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            # Generate
            _ = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for consistent timing
                temperature=1.0
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            generation_time = end_time - start_time
            times.append(generation_time)
            
            # Memory usage
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                memory_usage.append(memory_mb)
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        tokens_per_second = max_new_tokens / avg_time
        avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0
        
        stats = {
            'average_generation_time_s': avg_time,
            'tokens_per_second': tokens_per_second,
            'average_memory_usage_mb': avg_memory,
            'max_new_tokens': max_new_tokens,
            'num_runs': num_runs
        }
        
        self.logger.info(f"Benchmark results: {stats}")
        return stats
    
    def get_model_info(self) -> Dict:
        """Get model information and statistics."""
        memory_stats = self.model.get_memory_usage()
        
        info = {
            'model_config': self.model.config.to_dict(),
            'device': str(self.device),
            'memory_stats': memory_stats,
            'tokenizer_vocab_size': len(self.tokenizer),
        }
        
        return info


def main():
    """CLI interface for inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MoR-SLM Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="The future of AI is", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling instead of greedy")
    parser.add_argument("--recursion_strategy", type=str, default="adaptive", 
                       choices=["uniform", "adaptive", "progressive"], help="Recursion depth strategy")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize inference engine
    engine = MoRInferenceEngine(args.model_path)
    
    if args.benchmark:
        # Run benchmark
        stats = engine.benchmark(args.prompt, args.max_new_tokens)
        print(f"Benchmark Results: {stats}")
    else:
        # Generate text
        result = engine.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            recursion_strategy=args.recursion_strategy
        )
        
        print(f"Prompt: {args.prompt}")
        print(f"Generated: {result}")


if __name__ == "__main__":
    main()
