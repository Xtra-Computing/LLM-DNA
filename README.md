# LLM-DNA

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/llm-dna.svg)](https://badge.fury.io/py/llm-dna)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-17%20passed-brightgreen.svg)](#tests)

**Extract LLM DNA vectors** — low-dimensional, training-free representations that capture functional behavior and evolutionary relationships between language models.

> 📄 **Paper**: [LLM DNA: Tracing Model Evolution via Functional Representations](https://openreview.net/pdf?id=UIxHaAqFqQ) (ICLR 2026 Oral)

## Overview

The explosive growth of large language models has created a vast but opaque landscape: millions of models exist, yet their evolutionary relationships through fine-tuning, distillation, or adaptation are often undocumented. **LLM-DNA** provides a general, scalable, training-free pipeline for extracting LLM DNA — mathematically-grounded representations that satisfy inheritance and genetic determinism properties.

**Key Features:**
- 🧬 Extract DNA vectors from any HuggingFace or local model
- 🚀 Training-free, works across architectures and tokenizers  
- 📊 Tested on 305+ LLMs with superior or competitive performance
- 🔍 Uncover undocumented relationships between models
- 🌳 Build evolutionary trees using phylogenetic algorithms

## Installation

```bash
pip install llm-dna
```

Use `llm-dna` for install/package naming, and `llm_dna` for Python imports.

## Quick Start

```python
from llm_dna import DNAExtractionConfig, calc_dna

config = DNAExtractionConfig(
    model_name="distilgpt2",
    dataset="rand",
    gpu_id=0,
    max_samples=100,
)

result = calc_dna(config)
print(f"DNA shape: {result.vector.shape}")  # (128,)
```

## Python API

```python
from llm_dna import DNAExtractionConfig, calc_dna

config = DNAExtractionConfig(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    dataset="rand",
    gpu_id=0,
    max_samples=100,
    dna_dim=128,
    reduction_method="random_projection",  # or "pca", "svd"
    trust_remote_code=True,
)

result = calc_dna(config)

# DNA vector (numpy.ndarray)
vector = result.vector

# Saved paths (when save=True)
print(result.output_path)
print(result.summary_path)
```

## CLI

```bash
# Single model
calc-dna --model-name distilgpt2 --dataset rand --gpus 0

# Multiple models with round-robin GPU assignment
calc-dna --llm-list ./configs/llm_list.txt --gpus 0,1

# With hyperparameters
calc-dna \
  --model-name mistralai/Mistral-7B-v0.1 \
  --dna-dim 256 \
  --max-samples 200 \
  --reduction-method pca \
  --load-in-8bit
```

## Notes

- **Metadata auto-fetched**: Model metadata is automatically retrieved from HuggingFace Hub and cached.
- **Auth token**: Pass via `token=...` or set `HF_TOKEN` environment variable.
- **Chat templates**: Disabled by default. Enable with `--use-chat-template` (CLI) or `use_chat_template=True` (API).

## Tests

```bash
# All tests (including integration tests with real model loading)
pytest tests/ -v

# Fast tests only (skip real model loading)
pytest tests/ -m "not slow"
```

## Citation

If you use LLM-DNA in your research, please cite:

```bibtex
@inproceedings{wu2026llmdna,
  title={LLM DNA: Tracing Model Evolution via Functional Representations},
  author={Wu, Zhaomin and Zhao, Haodong and Wang, Ziyang and Guo, Jizhou and Wang, Qian and He, Bingsheng},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/pdf?id=UIxHaAqFqQ}
}
```

## License

Apache 2.0
