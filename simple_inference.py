#!/usr/bin/env python3
"""
Simple inference script for OLMo 3.5 Hybrid model using vLLM.
Supports both single prompt and batch inference.
"""

import argparse
from vllm import LLM, SamplingParams


def run_inference(
    model_name: str,
    prompts: str | list[str],
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.95,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
):
    """
    Run inference on one or more prompts.
    
    Args:
        model_name: HuggingFace model name or local path
        prompts: Single prompt string or list of prompts for batch inference
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use
    
    Returns:
        Single response string or list of response strings
    """
    # Handle single prompt vs batch
    is_single = isinstance(prompts, str)
    if is_single:
        prompts = [prompts]
    
    # Initialize model
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,  # Required for hybrid models
    )
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    
    # Generate
    outputs = llm.generate(prompts, sampling_params)
    
    # Extract responses
    responses = [output.outputs[0].text for output in outputs]
    
    return responses[0] if is_single else responses


def main():
    parser = argparse.ArgumentParser(description="Simple OLMo 3.5 Hybrid Inference")
    parser.add_argument("--model", type=str, default="allenai/OLMo-3.5-Hybrid-1B", help="Model name or path")
    parser.add_argument("--prompt", type=str, help="Single prompt for inference")
    parser.add_argument("--prompts", type=str, nargs="+", help="Multiple prompts for batch inference")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-memory", type=float, default=0.8, help="GPU memory utilization")
    
    args = parser.parse_args()
    
    # Determine prompts
    if args.prompt:
        prompts = args.prompt
    elif args.prompts:
        prompts = args.prompts
    else:
        # Default demo prompts
        prompts = [
            "The capital of France is",
            "Write a haiku about coding:",
            "2 + 2 =",
        ]
        print("No prompts provided. Running demo with default prompts.\n")
    
    # Run inference
    responses = run_inference(
        model_name=args.model,
        prompts=prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory,
    )
    
    # Print results
    if isinstance(prompts, str):
        print(f"Prompt: {prompts}")
        print(f"Response: {responses}")
    else:
        for prompt, response in zip(prompts, responses):
            print(f"Prompt: {prompt}")
            print(f"Response: {response}\n")


if __name__ == "__main__":
    main()