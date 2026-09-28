# Original implementation and reproduction audit

Audit date: 2026-09-29. All original inputs were read locally; nothing was pushed.

## Sources and provenance

The private checkout is `/home/zhaomin/project/dna`, clean at commit
`6e38570ad79989e7b591431c61ec73bf7ce2d094` (2025-11-26). Its operative implementation is
`src/dna/TextDNAExtractor.py`, SHA-256
`f209f74561c3a0eaeb5bd19e61584ace8f149ec9c3387f49cb873f35eaf406e0`.
Relevant source locations in that checkout:

- Lines 344–359: shared random projection creation and cache.
- Lines 429–449: sentence-transformer encoding with `normalize_embeddings=False`.
- Lines 502–533: Qwen Matryoshka slicing and row L2 normalization.
- Lines 838–919: extraction from saved answers, fixed empty-answer slots, concatenation, projection.
- `scripts/calc_dna.sh`: original batch experiment parameters.
- `src/summary/print_dna_relation_prediction.py`: original relation classifier and evaluation.

Private history records concat-before-projection in `562aad0` (2025-09-11), fixed empty-answer
slots in `82cb7c9` (2025-09-14), and the Qwen slicing strategy in `1487288` (2025-09-14).
The public history confirms `7edb5953c3ba9dbd8c8dedda3fc2f648b7d60de3` (2026-02-08)
replaced the legacy extractor's `concat` branch, previously max pooling, with flattened
per-probe projections followed by truncation. Neither branch implements the original text pipeline.

The paper was checked against [arXiv v3, Algorithm 1 and Table 1](https://arxiv.org/html/2509.24496v3).
Algorithm 1 concatenates response embeddings before a shared Gaussian projection and does
not specify a separate normalization step. The private implementation adds Qwen row L2
normalization and uses unit-column Gaussian projections. These implementation details must
be stated explicitly; literal pseudocode and saved-artifact compatibility are distinct targets.

## Exact saved-artifact construction

For each model, keep the ordered 600 answers. Encode only nonempty answers into float32
4096-dimensional Qwen3-Embedding-8B embeddings. Let `u_i = embedding_i[:64]` and
`v_i = u_i / ||u_i||_2`, with a zero norm replaced by one. Empty answers retain their original
probe positions as zero vectors. The original method refuses an entirely empty answer set.

Flatten the resulting `(600, 64)` array in row-major order to a 38,400-dimensional vector `x`.
The exact projection is:

```python
rng = np.random.RandomState(42)
g = rng.randn(38400, 128).astype(np.float32)
r = g / np.linalg.norm(g, axis=0)
dna = x.reshape(1, -1) @ r
```

There is no per-model `StandardScaler`, max pooling, or normal-path truncation after
concatenation. The projection matrix is shared by every model with the same input/output
dimensions and seed. It is not sklearn's default GaussianRandomProjection scaling/layout.

The original file `cache/projection_matrix_38400x128_seed42.npy` is float32 with SHA-256
`8acbe4e4025f16cc587ddfe367f0c72c6cbf841b4ce3eb5a9f830be70b93692d`.
An independent regeneration using the code above was bit-for-bit equal, maximum absolute
error `0.0`.

“Normalization” can mean three different things here: Qwen per-answer L2 normalization
before concatenation; unit-column scaling of the random matrix; or downstream training-only
standardization of classifier pair features. The public extractor's per-model StandardScaler
is a fourth operation and was absent from the original text extractor. Ablations must name
the operation they change.

## Available cached experiment inputs

All paths below are relative to the private checkout.

| Input | Local inventory / use |
| --- | --- |
| `out/squad_cqa_hs_wg_arc_mmlu/*/*_dna.json` | 313 main signatures, all 600 probes, 64 embedding coordinates, 128 DNA coordinates |
| Same model directories, `responses.json` | Ordered `items`, each with `prompt` and `response`; model identity is top-level `model` |
| `out_no_chat_template/squad_cqa_hs_wg_arc_mmlu` | 176 alternative signatures; keep this distinct from the main cohort |
| `out/rand` | 47 signatures, each **100** probes according to saved metadata; do not describe these as 600-probe cached runs |
| `cache/precomputed_dim4096_<hash>.pkl` | Raw answer-embedding cache; verify exact ordered `texts` against nonempty saved answers |
| `cache/dataset_squad_cqa_hs_wg_arc_mmlu_n100_seed42.pkl` | Original mixed-probe cache where available |
| `data/relation/valid_llm_relation_enriched_seed{0..4}.csv` | Five relation datasets, each 166 pairs; union covers 47 model IDs |
| `out_phyloLM/squad_cqa_hs_wg_arc_mmlu` | Availability determines the paper's common comparison cohort |

The original encoding filename uses MD5 of `str(sorted(texts))`, but the payload preserves
ordered `texts`. Filename matching alone is insufficient: verify ordered text equality,
embedding shape and finiteness. The historical filename does not identify the encoder,
so use encoder/dimension provenance and saved-DNA numerical comparisons as additional checks.

All 313 main summaries record six datasets (`squad,cqa,hs,wg,arc,mmlu`), `max_samples=100`
**per dataset**, Qwen3-Embedding-8B, pre-aggregation dimension 64, concat, DNA dimension 128,
seed 42, generation length 1024, temperature 0.7, top-p 0.9, and `use_vllm=True`.
For example, see `out/squad_cqa_hs_wg_arc_mmlu/Qwen_Qwen2.5-7B/Qwen_Qwen2.5-7B_summary.json`.
The loader mixes all six datasets and shuffles with NumPy seed 42; exact cached probe order
is preferable to re-downloading changing datasets.

Generation caveat: `compute_dna.py` passes `do_sample=False`, but the private vLLM wrapper
(`src/models/ModelWrapper.py`, lines 1040–1062) ignores it and directly supplies temperature
0.7/top-p 0.9 to SamplingParams. Thus greedy regeneration is not the recorded vLLM setting.
Most main summaries predate the chat-template flag (309 absent; four explicitly true).
Do not infer every historical template decision solely from the current shell default.
Cached replay avoids changing generation, tokenizer, encoder, or probe ordering.

## Published Table 1 reproduced from saved signatures

Run from the private checkout using the public environment's Python:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/zhaomin/project/LLM-DNA/.venv/bin/python \
  src/summary/print_dna_relation_prediction.py \
  --no-false-negative --local-files-only --exclude-ablation-bottleneck
```

This completed successfully on 2026-09-29 and reproduced the Qwen row of
[published Table 1](https://arxiv.org/html/2509.24496v3#S5.T1) at its reported precision:

| Metric | Local original evaluator | Published Qwen row |
| --- | --- | --- |
| Accuracy | 0.919 ± 0.048 | 0.919 ± 0.048 |
| Precision | 0.898 ± 0.049 | 0.898 ± 0.049 |
| Recall | 0.957 ± 0.072 | 0.957 ± 0.072 |
| F1 | 0.925 ± 0.047 | 0.925 ± 0.047 |
| AUC | 0.979 ± 0.027 | 0.979 ± 0.027 |

The original evaluator builds `[abs(dna_a - dna_b), dna_a - dna_b]`, then fits
training-only StandardScaler plus balanced RBF SVC with `probability=True`. It uses a
stratified 80/20 **pair** split with seeds 0–4 separately for each of five enriched dataset
seed files: 25 evaluations, pooled mean and sample standard deviation. This is not a
model-disjoint evaluation. Despite the paper caption referring to five seeds, the checked-in
script's numerical reproduction uses this nested 5×5 procedure.

The essential cohort detail is the intersection with available PhyloLM signatures:
dataset seeds 0–4 contain respectively **144, 149, 146, 141, 147** eligible pairs, covering
43 models. This is the cohort for the paper comparison, even when only the Qwen arm is
being recomputed. Omitting the intersection changes the scientific comparison.

For clarity, running the same source with `--methods dna_svm` instead uses all 166 pairs
per dataset and gives accuracy 0.867 ± 0.063, precision 0.875 ± 0.086, recall 0.868 ± 0.096,
F1 0.867 ± 0.064, and AUC 0.921 ± 0.070. That difference is explained by cohort selection;
it is not evidence of an extraction mismatch. Both cohorts are useful for a normalization
ablation if labeled and kept identical across treatments.

This verification establishes reproduction of the published **Qwen Table 1 row from saved
signatures**. Reconstructing those signatures from raw embedding caches is a separate check
performed by the cache-ablation experiment. It does not establish regeneration of all model
responses, every paper table, model routing, or the phylogenetic analysis.


## Completed raw-cache replay

The [normalization experiment](../experiments/normalization_20260929/README.md)
subsequently reconstructed all 313 historical signatures from matched raw Qwen
embeddings, with maximum absolute error 2.0862e-6 when row L2 is enabled. It also
reproduced the same Table 1 metrics. It checked 316 models against the public
extractor in both modes at seed 42 and three representative models per additional
seed. The provisional public default disables the extra row L2; the report
explains why the observed small performance difference is not reliable evidence
of superiority. Explicitly enable normalization for historical-vector reproduction.

The final offline test suite passed 122 tests, with 15 slow model/API tests skipped;
commands and implementation hashes are in
[validation.json](../experiments/normalization_20260929/validation.json).
