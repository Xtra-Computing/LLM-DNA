#!/usr/bin/env python3
"""
Unified script for computing LLM DNA signatures.

This script supports both synthetic probes and real datasets for DNA extraction.
Use --dataset to specify data source: 'syn' for synthetic probes, or dataset IDs.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import re
from transformers import AutoTokenizer

from ..dna.EmbeddingDNAExtractor import EmbeddingDNAExtractor
from ..dna.DNASignature import DNASignature
from ..data.DatasetLoader import DatasetLoader, DatasetConfig
from ..data.ProbeGenerator import ProbeSetGenerator
from ..models.ModelLoader import ModelLoader
from ..utils.DataUtils import setup_logging, get_cache_dir


def load_model_metadata(metadata_file: Path) -> Dict[str, Any]:
    """Loads and indexes the LLM metadata file by model_name."""
    if not metadata_file.exists():
        logging.warning(f"Metadata file not found at {metadata_file}. Proceeding without metadata.")
        return {}
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Index by model_name for fast lookups
    return {model['model_name']: model for model in data.get('models', [])}


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract DNA signatures from Language Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--model-name", "-m",
        type=str,
        required=True,
        help="Model name or path to model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to local model (overrides model-name if specified)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "huggingface", "openai", "openrouter", "gemini", "anthropic"],
        help="Type of model to load"
    )
    
    # Dataset/Probe arguments
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="rand",
        help="Dataset ID: 'syn' for synthetic probes, single dataset ID "
             "(e.g. 'squad', 'cqa', 'hs', 'wg', 'arc', 'mmlu', 'rand'), "
             "or comma-separated list of datasets (e.g. 'rand,squad,arc')"
    )
    parser.add_argument(
        "--embedding-merge",
        type=str,
        default="sum",
        choices=["sum", "max", "mean", "concat"],
        help="Aggregation across probe embeddings for embedding DNA (sum|max|mean|concat)"
    )
    parser.add_argument(
        "--probe-set",
        type=str,
        default="general",
        help="Synthetic probe set name (used when dataset='syn')"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of samples to use"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root directory for dataset storage"
    )
    
    # DNA extraction arguments
    parser.add_argument(
        "--extractor-type", "-e",
        choices=["embedding"],
        default="embedding",
        help="Type of DNA extractor to use: 'embedding' (token embeddings)"
    )
    parser.add_argument(
        "--dna-dim",
        type=int,
        default=256,
        help="DNA signature dimensionality"
    )
    parser.add_argument(
        "--reduction-method",
        type=str,
        default="pca",
        choices=["pca", "svd", "random_projection"],
        help="Dimensionality reduction method (for embedding DNA)"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length for generation"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default="./out",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save-format",
        choices=["json"],
        default="json",
        help="Format for saving DNA signatures (JSON only)"
    )
    
    # Quantization arguments
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Force 8-bit quantization (auto-enabled for large models)"
    )
    parser.add_argument(
        "--load-in-4bit", 
        action="store_true",
        help="Force 4-bit quantization (more aggressive memory saving)"
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true", 
        help="Disable automatic quantization even for large models"
    )
    
    # Metadata and Authentication arguments
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default="./configs/llm_metadata.json",
        help="Path to the pre-computed llm_metadata.json file"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face Hub token for accessing private or gated models"
    )
    parser.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        help="Allow executing custom code from model repositories when loading models/tokenizers"
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable execution of custom code from model repositories"
    )
    parser.set_defaults(trust_remote_code=True)
    
    # System arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for computation (auto, cpu, cuda, or specific GPU like cuda:0, cuda:1)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        default=False,
        help="Apply chat templates for chat models. By default, chat templates are not applied."
    )
    
    return parser.parse_args(argv)


def get_dataset_name(dataset_id: str) -> str:
    """Convert dataset ID to full dataset name."""
    dataset_mapping = {
        "syn": "synthetic",
        "squad": "squad", 
        "cqa": "commonsense_qa",
        "hs": "hellaswag",
        "wg": "winogrande", 
        "arc": "arc",
        "mmlu": "mmlu",
        "embed": "embedllm",
        "mix": "mixed",
        "rand": "rand",
        "rand_chinese": "rand_chinese"
    }
    return dataset_mapping.get(dataset_id, dataset_id)


def _get_cache_dir() -> Path:
    """Project-level cache directory (defaults to ./cache)."""
    return get_cache_dir()


def _safe_dataset_key(ds_id: str) -> str:
    # Make filename-safe key from dataset id (allow letters, digits, underscore, dash)
    import re
    return re.sub(r"[^A-Za-z0-9_-]+", "_", ds_id.strip())


def _dataset_cache_path(dataset_id: str, max_samples: int, random_seed: int) -> Path:
    safe = _safe_dataset_key(dataset_id)
    filename = f"dataset_{safe}_n{int(max_samples)}_seed{int(random_seed)}.json"
    return _get_cache_dir() / filename


def _load_cached_dataset(dataset_id: str, max_samples: int, random_seed: int) -> Optional[List[str]]:
    """Load cached probe texts from JSON file."""
    path = _dataset_cache_path(dataset_id, max_samples, random_seed)
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Basic integrity check
        if (
            isinstance(data, dict)
            and data.get('dataset_id') == dataset_id
            and int(data.get('max_samples', -1)) == int(max_samples)
            and int(data.get('random_seed', -1)) == int(random_seed)
            and isinstance(data.get('probe_texts'), list)
        ):
            logging.info(f"Loaded cached dataset samples from {path}")
            return data['probe_texts']
    except Exception as e:
        logging.warning(f"Failed to load dataset cache {path}: {e}")
    return None


def _save_cached_dataset(dataset_id: str, max_samples: int, random_seed: int, probe_texts: List[str]) -> None:
    """Save probe texts to cache as JSON for visibility."""
    path = _dataset_cache_path(dataset_id, max_samples, random_seed)
    if path.exists():
        return  # Already cached
    
    payload = {
        'dataset_id': dataset_id,
        'max_samples': int(max_samples),
        'random_seed': int(random_seed),
        'count': int(len(probe_texts)),
        'probe_texts': probe_texts,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved dataset cache to {path}")
    except Exception as e:
        logging.warning(f"Failed to save dataset cache {path}: {e}")


def get_probe_texts(
    dataset_id: str,
    probe_set: str,
    max_samples: int,
    data_root: str,
    random_seed: int,
) -> List[str]:
    """Get probe texts based on dataset selection."""
    # Try dataset cache first for non-synthetic datasets
    cached = _load_cached_dataset(dataset_id, max_samples, random_seed)
    if cached is not None:
        return cached
    
    if dataset_id == "syn":
        # Use synthetic probe generation
        logging.info(f"Using synthetic probe set: {probe_set}")
        probe_generator = ProbeSetGenerator()
        probe_set_obj = probe_generator.load_standard_probes(probe_set)
        probe_texts = probe_set_obj.probes[:max_samples]
        
    else:
        # Check if dataset_id contains comma-separated datasets
        if ',' in dataset_id:
            # Multiple datasets - split and process
            dataset_ids = [name.strip() for name in dataset_id.split(',')]
            dataset_names = [get_dataset_name(name) for name in dataset_ids]
            
            # Use max_samples as samples per dataset (not total)
            samples_per_dataset = max_samples
            
            logging.info(f"Multi-dataset processing: {len(dataset_names)} datasets with {samples_per_dataset} samples each")
            logging.info(f"Dataset details: {dict(zip(dataset_names, [samples_per_dataset] * len(dataset_names)))}")
            
            data_loader = DatasetLoader(data_root=data_root, cache_embeddings=True)
            
            # Create mixed dataset probes
            probe_texts = data_loader.create_probe_dataset(
                dataset_names=dataset_names,
                samples_per_dataset=samples_per_dataset,
                mix_datasets=True,
                seed=random_seed
            )
        else:
            # Single dataset
            dataset_name = get_dataset_name(dataset_id)
            logging.info(f"Loading single dataset: {dataset_name}")
            
            data_loader = DatasetLoader(data_root=data_root, cache_embeddings=True)
            
            # Load single dataset using predefined configurations
            if dataset_name in data_loader.dataset_configs:
                base_config = data_loader.dataset_configs[dataset_name]
                config = DatasetConfig(
                    name=base_config.name,
                    subset=base_config.subset,
                    split=base_config.split,
                    text_column=base_config.text_column,
                    max_samples=max_samples,
                    cache_dir=base_config.cache_dir,
                    download=base_config.download
                )
                probe_texts = data_loader.load_dataset(dataset_name, config)
            else:
                # Fallback to direct loading
                config = DatasetConfig(
                    name=dataset_name,
                    max_samples=max_samples
                )
                probe_texts = data_loader.load_dataset(dataset_name, config)
    
    logging.info(f"Loaded {len(probe_texts)} probe texts")
    # Save to cache (only for non-synthetic)
    if dataset_id != "syn":
        _save_cached_dataset(dataset_id, max_samples, random_seed, probe_texts)
    return probe_texts


def extract_dna_signature(
    model_name: str,
    model_path: Optional[str],
    model_type: str,
    probe_texts: List[str],
    extractor_type: str,
    model_metadata: Dict[str, Any], # Add model_metadata as an argument
    args: argparse.Namespace
) -> DNASignature:
    """Extract DNA signature from model."""

    # Apply chat template when available for chat-oriented tokenizers.
    if extractor_type == "embedding" and args.use_chat_template:
        is_chat_model = model_metadata.get("chat_model", {}).get("is_chat_model", False)
        should_try_template = is_chat_model or "chat_model" not in model_metadata
        try:
            # Prefer new 'token' kw; fallback to 'use_auth_token' for older Transformers
            tokenizer_kwargs = {"trust_remote_code": args.trust_remote_code}
            if args.token is not None:
                tokenizer_kwargs["token"] = args.token
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path or model_name,
                    **tokenizer_kwargs
                )
            except TypeError:
                fallback_kwargs = tokenizer_kwargs.copy()
                token_value = fallback_kwargs.pop("token", None)
                if token_value is not None:
                    fallback_kwargs["use_auth_token"] = token_value
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path or model_name,
                    **fallback_kwargs
                )

            tokenizer_has_chat_template = bool(getattr(tokenizer, "chat_template", None))
            if should_try_template and tokenizer_has_chat_template:
                logging.info("Applying tokenizer chat template to probe texts.")
                formatted_probes = []
                for text in probe_texts:
                    chat_message = [{"role": "user", "content": text}]
                    formatted_prompt = tokenizer.apply_chat_template(
                        chat_message, tokenize=False, add_generation_prompt=True
                    )
                    formatted_probes.append(formatted_prompt)
                probe_texts = formatted_probes  # Replace raw text with formatted prompts
        except Exception as e:
            logging.warning(f"Failed to apply chat template: {e}. Proceeding with raw probes.")
    
    # Load model with quantization settings
    logging.info(f"Loading model: {model_name}")
    model_loader = ModelLoader()
    
    # Determine quantization settings using pre-computed metadata
    load_in_4bit = args.load_in_4bit
    load_in_8bit = args.load_in_8bit
    
    if not args.no_quantization and not load_in_4bit and not load_in_8bit:
        size_in_billions = model_metadata.get('size', {}).get('parameter_count_billions')
        if size_in_billions and size_in_billions >= 7.0:
            logging.info(f"Auto-enabling 8-bit quantization for large model ({size_in_billions:.1f}B params).")
            load_in_8bit = True
    
    # Log quantization choice
    if load_in_4bit:
        logging.info(f"Using 4-bit quantization for model: {model_name}")
    elif load_in_8bit:
        logging.info(f"Using 8-bit quantization for model: {model_name}")
    elif args.no_quantization:
        logging.info(f"Quantization disabled for model: {model_name}")
    else:
        logging.info(f"No quantization for model: {model_name}")
    
    model = model_loader.load_model(
        model_path_or_name=model_path or model_name,
        model_type=model_type,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        token=args.token, # Pass the token for loading gated models
        is_chat_model=model_metadata.get('chat_model', {}).get('is_chat_model', False)
    )
    
    # Initialize DNA extractor
    logging.info(f"Initializing {extractor_type} DNA extractor")
    if extractor_type != "embedding":
        raise ValueError(f"Unsupported extractor type '{extractor_type}'. Only 'embedding' is supported.")

    extractor = EmbeddingDNAExtractor(
        dna_dim=args.dna_dim,
        reduction_method=args.reduction_method,
        aggregation_method=args.embedding_merge,
        device=args.device,
        random_seed=args.random_seed
    )
    signature = extractor.extract_dna(
        model=model,
        probe_inputs=probe_texts,
        probe_set_id=f"{args.dataset}_{len(probe_texts)}",
        max_length=args.max_length,
    )
    
    return signature


def validate_device_argument(device: str) -> str:
    """Validate and normalize device argument."""
    device = device.lower().strip()
    
    # Valid device patterns
    if device in ["auto", "cpu", "cuda"]:
        return device
    
    # Check for specific CUDA device (cuda:N)
    if device.startswith("cuda:"):
        try:
            gpu_id = int(device.split(":")[1])
            if gpu_id >= 0:
                return device
        except (ValueError, IndexError):
            pass
    
    # Invalid device
    raise ValueError(f"Invalid device '{device}'. Must be 'auto', 'cpu', 'cuda', or 'cuda:N' where N is a GPU ID.")



def main():
    """Main function."""
    args = parse_arguments()
    
    # --- FIX: Validate quantization flags ---
    quant_flags = [args.load_in_4bit, args.load_in_8bit, args.no_quantization]
    if sum(quant_flags) > 1:
        logging.error("Contradictory quantization flags. Use only one of --load-in-4bit, --load-in-8bit, or --no-quantization.")
        return 1
    
    # --- FIX: Add dependency check for bitsandbytes ---
    if args.load_in_4bit or args.load_in_8bit:
        try:
            import bitsandbytes
        except ImportError:
            logging.error("Quantization requires the 'bitsandbytes' library. Please install it with 'pip install bitsandbytes'.")
            return 1
    
    # Validate device argument
    try:
        args.device = validate_device_argument(args.device)
    except ValueError as e:
        logging.error(str(e))
        return 1
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Load metadata and get info for the current model.
    all_metadata = load_model_metadata(args.metadata_file)
    model_meta = all_metadata.get(
        args.model_name,
        {
            "model_name": args.model_name,
            "architecture": {"is_generative": True},
            "repository": {},
        },
    )
    if args.model_name not in all_metadata:
        logging.info(
            "Metadata for '%s' not found in %s. Proceeding with runtime defaults.",
            args.model_name,
            args.metadata_file,
        )
    
    # If model_path is not provided but metadata has a local_path, use it
    # This allows using short names in list files while still loading from full paths
    if not args.model_path and model_meta.get('repository', {}).get('local_path'):
        args.model_path = model_meta['repository']['local_path']
        logging.info(f"Using full path from metadata: {args.model_path}")
    elif not args.model_path and model_meta.get('repository', {}).get('model_id'):
        # Fallback to model_id if it looks like a local path
        model_id = model_meta['repository']['model_id']
        if Path(model_id).exists():
            args.model_path = model_id
            logging.info(f"Using model_id as local path: {args.model_path}")

    # Early exit for non-generative models based on metadata
    if model_meta.get('architecture', {}).get('is_generative') is False:
        arch_type = model_meta.get('architecture', {}).get('type')
        logging.warning(
            f"Skipping model '{args.model_name}' (architecture: {arch_type}). "
            "Encoder-only/non-generative models are skipped."
        )
        return 0
    
    # Create base output root
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log HuggingFace cache directory being used
    if 'HF_HOME' in os.environ:
        logging.info(f"Using HF_HOME: {os.environ['HF_HOME']}")
    else:
        default_cache = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
        logging.info(f"HF_HOME not set, using default: {default_cache}")
    
    try:
        start_time = time.time()
        
        logging.info(f"Preparing probes from dataset: {args.dataset}")
        probe_texts = get_probe_texts(
            dataset_id=args.dataset,
            probe_set=args.probe_set,
            max_samples=args.max_samples,
            data_root=args.data_root,
            random_seed=args.random_seed,
        )
        
        # Extract DNA signature
        logging.info(f"Starting DNA extraction for {args.model_name}")
        signature = extract_dna_signature(
            model_name=args.model_name,
            model_path=args.model_path,
            model_type=args.model_type,
            probe_texts=probe_texts,
            extractor_type=args.extractor_type,
            model_metadata=model_meta,  # Pass the metadata
            args=args
        )
        
        total_time = time.time() - start_time
        
        # Create structured output directory: out/<dataset_identifier>/<model_name>
        # Use underscores for multi-dataset identifiers instead of commas
        safe_model_name = args.model_name.replace("/", "_").replace(":", "_")
        dataset_identifier = args.dataset.replace(',', '_')
        structured_dir = args.output_dir / dataset_identifier / safe_model_name
        
        structured_dir.mkdir(parents=True, exist_ok=True)
        # Filename contains only model name and dna suffix; dataset is encoded by directory
        # Force JSON extension (JSON-only support)
        output_filename = f"{safe_model_name}_dna.json"
        output_path = structured_dir / output_filename
        
        # Validate signature: refuse all-zero signatures and mark as failed
        try:
            import numpy as _np
            sig_arr = _np.array(signature.signature, dtype=_np.float32)
            if sig_arr.size == 0 or _np.allclose(sig_arr, 0.0):
                raise ValueError("All-zero DNA signature detected; refusing to save")
        except Exception as _e:
            logging.error(str(_e))
            return 1

        # Save signature
        signature.save(output_path, format="json")
        logging.info(f"DNA signature saved to: {output_path}")
        
        # Save summary
        # Create safe args dict without sensitive information
        safe_args = vars(args).copy()
        # Remove sensitive fields that should not be saved to output files
        sensitive_fields = ['token', 'OPENROUTER_API_KEY', 'OPENAI_API_KEY']
        for field in sensitive_fields:
            safe_args.pop(field, None)
        
        summary = {
            "model_name": args.model_name,
            "dataset": args.dataset,
            "dataset_full": get_dataset_name(args.dataset),
            "extractor_type": args.extractor_type,
            "dna_dimension": args.dna_dim,
            "reduction_method": args.reduction_method,
            "num_probes": len(probe_texts),
            "total_time_seconds": total_time,
            "signature_stats": signature.get_statistics(),
            "metadata": signature.metadata.__dict__,
            "output_file": str(output_path),
            "args": safe_args
        }
        
        # Keep summary filename model-only as well
        summary_path = structured_dir / f"{safe_model_name}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str, ensure_ascii=False, allow_nan=False)
        
        logging.info(f"Extraction completed in {total_time:.2f}s")
        logging.info(f"Summary saved to: {summary_path}")
        
        # Print signature statistics
        stats = signature.get_statistics()
        print(f"\nDNA Signature Statistics:")
        print(f"  Model: {signature.model_name}")
        print(f"  Dataset: {args.dataset} ({get_dataset_name(args.dataset)})")
        print(f"  Extractor: {args.extractor_type}")
        print(f"  Dimension: {signature.dimension}")
        print(f"  Probes: {len(probe_texts)}")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std: {stats['std']:.4f}")
        print(f"  L2 Norm: {stats['l2_norm']:.4f}")
        print(f"  Time: {total_time:.2f}s")
        
        return 0
        
    except KeyboardInterrupt:
        logging.info("DNA extraction interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"DNA extraction failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
