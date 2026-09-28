# LLM-DNA

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/llm-dna.svg)](https://badge.fury.io/py/llm-dna)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

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

Optional extras are available for model families that need additional runtime dependencies:

```bash
# Apple Silicon / MLX-backed models
pip install "llm-dna[apple]"

# Quantized HuggingFace models (bitsandbytes, GPTQ, compressed-tensors, optimum)
pip install "llm-dna[quantization]"

# Architecture-specific model families such as Mamba or TIMM-backed models
pip install "llm-dna[model_families]"

# Everything above
pip install "llm-dna[full]"
```

Extra guidance:
- `apple`: required for MLX and `mlx-community/*` style model families on Apple Silicon.
- `quantization`: required for many GPTQ, bitsandbytes, and compressed-tensors model families.
- `model_families`: required for specific architectures whose modeling code depends on packages like `mamba-ssm` or `timm`.

## Extraction method and defaults

The public single-model API, batch API, and CLI now use `TextDNAExtractor`,
restoring the original experiment's text-response pipeline:

1. Use the same ordered probes for every model. The default is 100 probes each
   from `squad,cqa,hs,wg,arc,mmlu`, for 600 probes total.
2. Encode generated answers with `Qwen/Qwen3-Embedding-8B` and keep the first 64
   coordinates of each answer. Empty answers keep their original slots as zeros.
3. Optionally L2 normalize each 64-dimensional answer embedding.
4. Concatenate all 600 slots into 38,400 dimensions and project the entire vector
   into 128 dimensions using a shared float32 Gaussian matrix, seed 42, with
   unit-length columns, exactly as in the private implementation.

There is no per-model fitted scaler, per-probe final projection, or truncation
of the concatenated vector. The resulting DNA is not normalized. PCA/SVD fitting
per model is rejected because it does not give models a shared coordinate system.
`TextDNAExtractor` is the sole supported extractor. Both API and CLI default to
`extractor_type="text"` / `--extractor-type text`. The old hidden-state
`EmbeddingDNAExtractor` and its unused base classes have been removed, including
their package exports. Replace explicit `extractor_type="embedding"` settings
with `"text"`, or omit the option to use the default.

`normalize_embeddings` controls the extra L2 normalization **after the 64-coordinate
slice**, not any normalization inside the encoder or the projection matrix's
column normalization. The private experiment enabled this step, while the paper's
algorithm does not explicitly describe it. Both settings are available; see the
[normalization experiment](experiments/normalization_20260929/README.md) for the
measured comparison and default selection, and the
[source audit](docs/reproduction_audit.md) for implementation and paper provenance.

The provisional default is **`normalize_embeddings=False`**. Across six projection
seeds, its mean F1 is 0.92434 versus 0.92313 with L2 on the original paper cohort;
this small difference does not establish superiority. A separate deduplicated,
pair-group-disjoint validation failed to demonstrate useful F1 for either arm.
For exact reconstruction of the original saved experiment, explicitly use
`normalize_embeddings=True` / `--normalize-embeddings`: all 313 saved signatures
match within 2.1e-6 and the published Table 1 Qwen row is reproduced.

Generation defaults are `max_length=1024`, `temperature=0.7`, `top_p=0.9`, with
available chat templates enabled. These settings follow the original experiment;
backend, model revision, quantization, and generation randomness can still change
answers. `random_seed` seeds probe sampling and the shared projection; it does not
control provider/Hugging Face sampling. Replaying the original caches is the reproducible comparison. The
8B sentence encoder needs substantially more memory than the old MPNet default.
An explicitly selected alternative encoder produces a different DNA space.

Vectors from the former public pipeline are incompatible with the corrected
vectors and must be recomputed. Compare only vectors with matching encoder,
preprocessing, ordered probes, aggregation, and projection. Saved metadata records
these settings, the probe-order hash, and the projection hash. Response caches
with a different model, probe count/order, or recorded generation settings are rejected.
Legacy caches without generation settings are explicitly marked as having unknown
generation provenance.

To regenerate answers even when a valid cache exists, set
`DNAExtractionConfig(use_response_cache=False, ...)` or pass
`--ignore-response-cache` to either CLI. This applies to single-model and batch
runs and keeps the existing save policy. The separate
`calc_dna_parallel(..., use_response_cache=False)` argument still disables both
response-cache reads and writes.

## Quick Start

```python
from llm_dna import DNAExtractionConfig, calc_dna

config = DNAExtractionConfig(
    model_name="distilgpt2",
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
    gpu_id=0,
    max_samples=100,
    dna_dim=128,
    reduction_method="random_projection",
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
calc-dna --model-name distilgpt2 --gpus 0

# Multiple models with round-robin GPU assignment
calc-dna --llm-list ./configs/llm_list.txt --gpus 0,1

# With hyperparameters
calc-dna \
  --model-name mistralai/Mistral-7B-v0.1 \
  --dna-dim 256 \
  --max-samples 200 \
  --reduction-method random_projection \
  --load-in-8bit
```

## Notes

- **Metadata auto-fetched**: Model metadata is automatically retrieved from HuggingFace Hub and cached.
- **Auth token**: Pass via `token=...` or set `HF_TOKEN` environment variable.
- **Chat templates**: Enabled when available. Disable with `--no-use-chat-template` or `use_chat_template=False`.

## Offline replay and normalization ablation

With an existing checkout of the private experiment repository and its caches:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/ablate_normalization.py \
  --private-root /home/zhaomin/project/dna \
  --output-dir experiments/normalization_20260929
```

The script reads original inputs without modifying them and records source hashes,
cohort selection, paired metrics, and vector reconstruction errors. It performs no
model generation or provider API calls. See its `--help` for controls and the
experiment report for the exact command used.

To extract from your own raw encoder embedding cache without loading a model:

```python
import numpy as np
from llm_dna import TextDNAExtractor

raw = np.load("ordered_response_embeddings.npy", allow_pickle=False)  # [probes, encoder_dim]
extractor = TextDNAExtractor(normalize_embeddings=True)  # original implementation
signature = extractor.extract_dna_from_embeddings(raw, model_name="my-model")
```

Use `normalize_embeddings=False` or `--no-normalize-embeddings` for the ablation
without extra row normalization. The same cached raw embeddings and projection
must be used for both settings.

## Tests

```bash
# Offline tests (slow model/API integration tests are skipped)
pytest tests/ -v

# Include real model/API integration tests explicitly
pytest tests/ -v --run-slow

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
