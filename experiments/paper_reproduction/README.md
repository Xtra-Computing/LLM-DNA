# Reproducing experiments from cached inputs

Run these commands from the root of the `repro/paper-relationship-routing`
branch. Python 3.10 or newer and the project dependencies are required. Reuse
an existing environment, or create one:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

The input cache directory is read-only. Use a separate output directory.
The runners use saved signatures and embeddings without model inference.
Provide the data files listed below in a regular directory; evaluator source
files and Git metadata are not required. Pass the directory with `--cache-root`.

## Relationship table

The relationship runners read `configs/valid_llm_relation.csv` in this checkout
by default. This file contains the original positive relation records. For each
dataset seed 0–4, the runner samples an equal number of negative directed pairs
and shuffles the combined records using the original algorithm. Sampling happens
before filtering to models with available signatures. Self-pairs, known positive
directed pairs, and repeated negatives are excluded; the reverse direction is
treated as a separate pair. No remote metadata is fetched.

```bash
bash scripts/reproduce_relationship_table.sh --cache-root /path/to/dna-cache
```

To use another positive-relation CSV, add `--relations /path/to/relations.csv`.
It must have `model_id`, `related_model`, and `relation` columns, with nonempty
relation values other than `nan`. The option is also supported by
`reproduce_relationship.py` and `reproduce_paper.py`. Enriched per-seed CSVs are
reconstructed in memory and do not need to be supplied or saved.

The launcher uses `REPRO_PYTHON`, the checkout's `.venv/bin/python`, or `python3`,
in that order. To choose a Python environment and output directory explicitly:

```bash
REPRO_PYTHON=/path/to/venv/bin/python \
bash scripts/reproduce_relationship_table.sh \
  --cache-root /path/to/dna-cache \
  --output-dir /path/to/new/table-output
```

For the relationship table, `REPRO_CACHE_ROOT` can supply the cache directory
instead of `--cache-root`; an explicit argument takes precedence. Without either,
the command reports a missing cache path. Without `--output-dir`,
each invocation creates a UTC-timestamped directory under
`experiments/paper_reproduction/relationship_table/`. Output directories must
be new or empty.

Required inputs, relative to the cache root:

| Input | Path |
| --- | --- |
| Qwen3 DNA | `out/squad_cqa_hs_wg_arc_mmlu/<model>/<model>_dna.json` |
| BGE DNA | `out/squad_cqa_hs_wg_arc_mmlu/<model>/bge/<model>_dna.json` |
| MPNet DNA | `out/squad_cqa_hs_wg_arc_mmlu/<model>/mpnet/<model>_dna.json` |
| PhyloLM signatures | `out_phyloLM/squad_cqa_hs_wg_arc_mmlu/<model>/<model>_dna.json` |

Use the original cache directory structure and model metadata. The cohort is
the intersection of available Qwen3 and PhyloLM models; BGE and MPNet signatures
are required for every model in that cohort.

The table contains ten rows:

| Representation | Features for the directed pair (A, B) | Encoders |
| --- | --- | --- |
| Original difference | `[abs(z_A - z_B), z_A - z_B]` | Qwen3, BGE, MPNet |
| Direct concatenation | `[z_A, z_B]` | Qwen3, BGE, MPNet |
| Euclidean landmarks | Concatenated Euclidean distances from A and B to reference models | Qwen3, BGE, MPNet |
| PhyloLM | Concatenated Nei distances from A and B to reference models | PhyloLM |

Each row uses the original directed, stratified 80/20 record split: five
dataset seeds × five split seeds, all 0–4, for 250 SVM fits in total. A/B order
is preserved. Only identical `(A, B, relation)` triples are deduplicated; records
with different relation sources remain separate. Pairs and models can occur
in both training and test records.

Reference models are the sorted unique models in the training records.
`StandardScaler` is fitted only on training rows. All rows use an RBF SVM
with `C=1`, `gamma="scale"`, `class_weight="balanced"`,
`probability=True`, and `random_state=split_seed`. DNA vectors receive no
additional normalization. PhyloLM uses the original Nei distance with a
`1e-9` similarity cutoff and a maximum distance of 10.

Default outputs:

- `table.md`: formatted table.
- `table.csv`: full-precision numeric values.
- `table.tex`: LaTeX table.
- `results.json`: metrics, cohorts, settings, and software versions.

Accuracy, Precision, Recall, F1, and AUC are reported as mean ± sample standard
deviation over 25 runs per row. Add `--save-runs` to also write `runs.json`
and `metrics.csv` with individual splits, predictions, and metrics.
Generated output directories are ignored by Git.

## Routing

Routing reads its original labeled splits and frozen features from `--cache-root`.
These large data files are external inputs and are not included in the repository:

- `data/embedllm/{train,val,test}.csv`, with `model_name`, `model_id`, `prompt_id`,
  `prompt`, and binary `label` columns. Each label records the model's correctness
  on that prompt. The three splits must have disjoint prompt IDs.
- DNA signatures under `out/squad_cqa_hs_wg_arc_mmlu/` and
  `out/squad_cqa_hs_wg_arc_mmlu_embed/`.
- Query embedding files under `cache/route_fixed_model/`:
  `enc-Qwen__Qwen3-Embedding-8B_dim4096_poolcls_len1024.npz` and
  `enc-sentence-transformers__all-mpnet-base-v2_bs32_prompts35673_squad_cqa_hs_wg_arc_mmlu_embed.npz`.

Train routers from frozen DNA and query caches:

```bash
.venv/bin/python scripts/reproduce_routing.py \
  --cache-root /path/to/dna-cache \
  --output-dir experiments/paper_reproduction/routing_train \
  --mode train --device cuda:0
```

Defaults are seeds 0–4, 50 epochs, and batch size 2048. Use `--device cpu`
for CPU execution. `--seeds`, `--epochs`, `--batch-size`, and `--threads`
control the run.

To recompute predictions from existing router checkpoints, use `--mode replay`
with a separate output directory. This mode also requires the
`mf_model_seed{0..4}.pt` and
`dna_router_squad_cqa_hs_wg_arc_mmlu_enc-Qwen__Qwen3-Embedding-8B_qd4096_mf232_dna_seed{0..4}.pt`
files under `cache/route_fixed_model/`.

Routing writes `summary.csv` and `results.json`, together with per-seed
metrics, predictions, and training outputs. To combine separately executed
seed runs:

```bash
.venv/bin/python scripts/merge_routing_runs.py \
  --inputs /path/to/seed_0 /path/to/seed_1 /path/to/seed_2 /path/to/seed_3 /path/to/seed_4 \
  --output-dir experiments/paper_reproduction/routing_merged
```

## Run both experiments

```bash
.venv/bin/python scripts/reproduce_paper.py \
  --cache-root /path/to/dna-cache \
  --output-dir experiments/paper_reproduction/run \
  --routing-mode train --routing-device cuda:0
```

Use `--experiments relationship` or `--experiments routing` to run one part.
Use `--routing-mode replay` to load router checkpoints. Run any entrypoint
with `--help` to list its options.
