# Changelog

## 1.0.0 — 2026-09-29

Restore the original text-response DNA extraction pipeline across the public API,
batch extraction, and CLI. Each model now contributes all ordered response
embeddings to one shared projection, fixing the earlier per-probe projection and
post-concatenation truncation.

- Default to Qwen3-Embedding-8B, 64 coordinates per response, six datasets with
  100 probes each, concatenation, and a shared 128-dimensional projection.
- Expose optional L2 normalization after the 64-coordinate slice. The provisional
  default is disabled; enable `normalize_embeddings=True` or
  `--normalize-embeddings` to reconstruct the original saved experiment.
- Preserve empty-answer positions and record preprocessing, ordered-probe, and
  projection provenance. Validate response-cache model identity, probe order, and
  recorded generation settings; identify legacy generation provenance as unknown.
- Reject per-model PCA/SVD fitting in the public text pipeline. The legacy
  hidden-state extractor remains available through explicit imports.
- Include the offline normalization ablation, input hashes, per-run results, and
  reproduction audit. All 313 historical DNA vectors were reconstructed within
  2.1e-6 maximum absolute error, and the published Qwen relation-prediction row was
  reproduced from the original caches. The normalization comparison did not show
  a reliable advantage for either setting; the report documents evaluation limits.

### Migration

DNA vectors from the former public implementation are incompatible with the
corrected coordinate system and must be recomputed. Compare vectors only when
encoder, preprocessing, ordered probes, and projection settings match.

Generation defaults now use temperature 0.7, top-p 0.9, and available chat
templates. `random_seed` controls probes and projection, not generation sampling.
The default 8B encoder requires more memory than the former MPNet default.
