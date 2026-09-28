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

from ..dna.DNASignature import DNASignature
from ..data.DatasetLoader import DatasetLoader, DatasetConfig
from ..data.ProbeGenerator import ProbeSetGenerator
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
    """Compatibility entrypoint using the same defaults as the public CLI."""
    from ..cli import parse_arguments as parse_cli
    return parse_cli(argv)


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
    model_metadata: Dict[str, Any],
    args: argparse.Namespace,
) -> DNASignature:
    """Compatibility wrapper over the public text-response extraction pipeline."""
    from dataclasses import fields
    from ..api import DNAExtractionConfig, _generate_responses_for_model, _extract_signature_from_text_responses, _resolve_device, _make_text_extractor

    options = {field.name: getattr(args, field.name) for field in fields(DNAExtractionConfig)
               if hasattr(args, field.name)}
    options.update(model_name=model_name, model_path=model_path, model_type=model_type,
                   extractor_type=extractor_type)
    config = DNAExtractionConfig(**options)
    _make_text_extractor(config)  # Reject obsolete selectors before model loading.
    device = _resolve_device(config)
    responses = _generate_responses_for_model(
        model_name, config, model_metadata, probe_texts, device, config.token,
    )
    signature, _, _ = _extract_signature_from_text_responses(
        model_name, responses, config, model_metadata, device, probe_texts=probe_texts,
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
    """Use the canonical CLI for historical module invocations."""
    from ..cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
