#!/usr/bin/env python3
"""
Example script demonstrating LLM-DNA DNA extraction API.

Usage:
    python scripts/calc_dna.py
    python scripts/calc_dna.py --model Qwen/Qwen2.5-0.5B-Instruct
    python scripts/calc_dna.py --model distilgpt2 --gpu 0 --samples 50
    python scripts/calc_dna.py --llm-list ./configs/llm_list.txt --gpus 0,1
"""

import argparse
import sys
from pathlib import Path

# Add src to path for local development
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_dna import DNAExtractionConfig, calc_dna, calc_dna_parallel


def main():
    parser = argparse.ArgumentParser(
        description="Extract DNA vector from an LLM using LLM-DNA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="distilgpt2",
        help="Model name or Hugging Face model ID (ignored if --llm-list is provided)"
    )
    parser.add_argument(
        "--llm-list",
        type=Path,
        default=None,
        help="Path to file containing model names (one per line) for batch processing"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="squad,cqa,hs,wg,arc,mmlu",
        help="Dataset for probe generation"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU ID to use for single model (None for CPU)"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs for batch mode (e.g., '0,1,2')"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of probe samples to use for DNA extraction"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining models if one fails (batch mode only)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to disk"
    )
    parser.add_argument("--sentence-encoder", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--normalize-embeddings", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    # Set paths relative to project root
    metadata_file = ROOT / "configs" / "llm_metadata.json"
    data_root = str(ROOT / "data")
    output_dir = ROOT / "out"

    # Parse GPU IDs for batch mode
    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]

    # Create configuration using the public API
    config = DNAExtractionConfig(
        model_name=args.model,
        dataset=args.dataset,
        sentence_encoder=args.sentence_encoder,
        normalize_embeddings=args.normalize_embeddings,
        gpu_id=args.gpu,
        max_samples=args.samples,
        data_root=data_root,
        metadata_file=metadata_file if metadata_file.exists() else None,
        output_dir=output_dir,
        save=not args.no_save,
        trust_remote_code=True,
    )

    # Batch mode: process multiple models from file
    if args.llm_list:
        print(f"Batch processing models from: {args.llm_list}")
        print(f"Using {args.samples} probe samples per model")
        if gpu_ids:
            print(f"GPUs: {gpu_ids}")
        
        results = calc_dna_parallel(
            config=config,
            llm_list=args.llm_list,
            gpu_ids=gpu_ids,
            continue_on_error=args.continue_on_error,
        )
        
        # Display batch results
        print(f"\n{'='*60}")
        print(f"Processed {len(results)} model(s):")
        for result in results:
            print(f"  - {result.model_name}: shape={result.vector.shape}, time={result.elapsed_seconds:.2f}s")
            if result.output_path:
                print(f"    Saved to: {result.output_path}")
        return 0

    # Single model mode
    print(f"Extracting DNA from: {args.model}")
    print(f"Using {args.samples} probe samples")
    result = calc_dna(config)

    # Display results
    print(f"\nDNA vector shape: {result.vector.shape}")
    print(f"Extraction time: {result.elapsed_seconds:.2f}s")
    print(f"First 10 values: {result.vector[:10]}")

    if result.output_path:
        print(f"\nSaved to: {result.output_path}")
        print(f"Summary: {result.summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

