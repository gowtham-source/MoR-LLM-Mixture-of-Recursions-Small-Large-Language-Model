#!/usr/bin/env python3
"""
Main inference script for MoR-SLM.
"""

import os
import sys
import argparse
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.inference_engine import MoRInferenceEngine


def main():
    parser = argparse.ArgumentParser(description="MoR-SLM Text Generation")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--text", 
        type=str, 
        default="The future of artificial intelligence is",
        help="Input text prompt"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=100,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Sampling temperature (0.0 = greedy, higher = more random)"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--do_sample", 
        action="store_true",
        help="Use sampling instead of greedy decoding"
    )
    parser.add_argument(
        "--recursion_strategy", 
        type=str, 
        default="adaptive",
        choices=["uniform", "adaptive", "progressive"],
        help="Recursion depth strategy"
    )
    parser.add_argument(
        "--min_recursion_depth", 
        type=int, 
        default=1,
        help="Minimum recursion depth"
    )
    parser.add_argument(
        "--max_recursion_depth", 
        type=int, 
        default=4,
        help="Maximum recursion depth"
    )
    parser.add_argument(
        "--num_return_sequences", 
        type=int, 
        default=1,
        help="Number of sequences to generate"
    )
    parser.add_argument(
        "--benchmark", 
        action="store_true",
        help="Run performance benchmark"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Interactive mode"
    )
    parser.add_argument(
        "--tokenizer", 
        type=str, 
        default="gpt2",
        help="Tokenizer to use"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize inference engine
        logger.info(f"Loading model from {args.model_path}")
        engine = MoRInferenceEngine(
            model_path=args.model_path,
            tokenizer_name=args.tokenizer
        )
        
        # Print model info
        model_info = engine.get_model_info()
        logger.info(f"Model loaded successfully")
        logger.info(f"Parameters: {model_info['memory_stats']['total_parameters']:,}")
        logger.info(f"Parameter reduction factor: {model_info['memory_stats']['parameter_reduction_factor']:.2f}x")
        
        if args.benchmark:
            # Run benchmark
            logger.info("Running performance benchmark...")
            stats = engine.benchmark(
                prompt=args.text,
                max_new_tokens=args.max_new_tokens
            )
            print("\n=== BENCHMARK RESULTS ===")
            print(f"Average generation time: {stats['average_generation_time_s']:.3f}s")
            print(f"Tokens per second: {stats['tokens_per_second']:.2f}")
            print(f"Memory usage: {stats['average_memory_usage_mb']:.1f} MB")
            
        elif args.interactive:
            # Interactive mode
            print("\n=== MoR-SLM Interactive Mode ===")
            print("Type 'quit' or 'exit' to stop")
            print(f"Settings: temp={args.temperature}, top_p={args.top_p}, max_tokens={args.max_new_tokens}")
            print(f"Recursion: {args.recursion_strategy} ({args.min_recursion_depth}-{args.max_recursion_depth})")
            print("-" * 50)
            
            while True:
                try:
                    prompt = input("\nPrompt: ").strip()
                    if prompt.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if not prompt:
                        continue
                    
                    print("Generating...")
                    result = engine.generate(
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        do_sample=args.do_sample,
                        recursion_strategy=args.recursion_strategy,
                        min_recursion_depth=args.min_recursion_depth,
                        max_recursion_depth=args.max_recursion_depth,
                        num_return_sequences=args.num_return_sequences
                    )
                    
                    if isinstance(result, list):
                        for i, text in enumerate(result):
                            print(f"\nGeneration {i+1}: {text}")
                    else:
                        print(f"\nGenerated: {result}")
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Generation error: {e}")
            
            print("\nGoodbye!")
            
        else:
            # Single generation
            logger.info(f"Generating text for prompt: '{args.text}'")
            result = engine.generate(
                prompt=args.text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=args.do_sample,
                recursion_strategy=args.recursion_strategy,
                min_recursion_depth=args.min_recursion_depth,
                max_recursion_depth=args.max_recursion_depth,
                num_return_sequences=args.num_return_sequences
            )
            
            print(f"\n=== INPUT ===")
            print(f"{args.text}")
            print(f"\n=== OUTPUT ===")
            
            if isinstance(result, list):
                for i, text in enumerate(result):
                    print(f"\nGeneration {i+1}:")
                    print(f"{args.text}{text}")
            else:
                print(f"{args.text}{result}")
    
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()
