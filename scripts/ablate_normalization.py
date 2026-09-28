#!/usr/bin/env python3
"""Offline, paired Qwen pre-projection normalization ablation and artifact replay.

Only read trusted local pickle caches: Python pickle is not a safe interchange
format for untrusted inputs. No models, APIs, or network access are used.

Run from the checkout with its Python environment, for example:
  OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/ablate_normalization.py \
    --private-root /home/zhaomin/project/dna \
    --output experiments/normalization_20260929
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
import pickle
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
METRICS = ("accuracy", "precision", "recall", "f1", "auc")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def provenance(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    return {"path": str(root), "head": git("rev-parse", "HEAD"),
            "status": git("status", "--short")}


def project_matrix(seed, dimensions=38400, output_dimensions=128):
    """Independent historical reference: MT19937 float32 column-unit Gaussian."""
    matrix = np.random.RandomState(seed).randn(dimensions, output_dimensions).astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=0)


def features(signatures, records, model_index):
    a = np.asarray([model_index[r[0]] for r in records])
    b = np.asarray([model_index[r[1]] for r in records])
    diff = signatures[a] - signatures[b]
    return np.concatenate([np.abs(diff), diff], axis=1)


def labels(records):
    return np.asarray([int(r[2].lower() != "nan") for r in records], dtype=np.int32)


def evaluate(X, y, train, test, seed):
    # This scaler belongs to the downstream supervised classifier, not DNA extraction.
    classifier = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True,
                                class_weight="balanced", random_state=seed))
    classifier.fit(X[train], y[train])
    predictions = classifier.predict(X[test])
    probabilities = classifier.predict_proba(X[test])[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y[test], predictions, average="binary", zero_division=0)
    return dict(zip(METRICS, map(float, [accuracy_score(y[test], predictions), precision,
                recall, f1, roc_auc_score(y[test], probabilities)])))


def summarize(rows, group_fields):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in group_fields)].append(row)
    result = []
    for key, values in sorted(grouped.items()):
        out = dict(zip(group_fields, key))
        out["runs"] = len(values)
        for metric in METRICS:
            a = np.asarray([r[metric] for r in values])
            out[metric + "_mean"] = float(a.mean())
            out[metric + "_std"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        result.append(out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--output", "--output-dir", type=Path, required=True)
    parser.add_argument("--projection-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 42])
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--holdout-seed", type=int, default=1729)
    args = parser.parse_args()
    private = args.private_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = (private / "cache").resolve()
    dna_root = private / "out/squad_cqa_hs_wg_arc_mmlu"
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": ["none", "l2"],
        "encoder": "Qwen/Qwen3-Embedding-8B",
        "raw_embedding_dimensions": 4096, "first_dimensions": 64,
        "probe_count": 600, "dna_dimensions": 128,
        "projection_seeds": args.projection_seeds, "split_seeds": args.split_seeds,
        "heldout_seed": args.holdout_seed,
        "paper_replay": "Each enriched seed file separately; sorted unique triples; stratified 80/20 pair-row split for every split seed; historical SVM pipeline. Report DNA-only full47 and the paper Table1 common43-model cohort restricted by saved PhyloLM availability.",
        "selection": "Deduplicate ordered model pairs across enriched files. Split by unordered pair group, stratifying groups by any positive label. Hold out 20% groups once; split remaining groups 75/25 for each split seed. Choose normalization by mean validation F1 across projection/split seeds, then AUC, then none. Test labels never used in the choice.",
        "scope": "Frozen cached responses and embeddings; no answer regeneration or routing rerun. Models may occur in multiple partitions. Repeated seeds are sensitivity analyses, not independent datasets.",
    }
    write_json(output / "protocol.json", protocol)
    manifest = {"protocol": protocol, "command": sys.argv, "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "scikit-learn"]},
        "private_repository": provenance(private), "public_repository": provenance(ROOT),
        "script_sha256": sha256(Path(__file__)), "sources": [], "models": [], "excluded": []}
    for relative in ["src/dna/TextDNAExtractor.py", "src/summary/print_dna_relation_prediction.py"]:
        path = private / relative
        manifest["sources"].append({"path": str(path), "sha256": sha256(path)})

    dataset_path = cache / "dataset_squad_cqa_hs_wg_arc_mmlu_n100_seed42.pkl"
    with dataset_path.open("rb") as handle:
        probes = pickle.load(handle)["probe_texts"]
    assert len(probes) == 600
    manifest["probe_source"] = {"path": str(dataset_path), "sha256": sha256(dataset_path),
                                "ordered_prompts_sha256": json_hash(probes)}
    cache_paths = collections.defaultdict(list)
    for path in sorted(cache.glob("*_dim4096_*.pkl")):
        cache_paths[path.stem.rsplit("_", 1)[1]].append(path)

    model_ids, sliced_embeddings, saved_vectors = [], [], {}
    norm_statistics, duplicate_cache_differences = [], []
    for response_path in sorted(dna_root.glob("*/responses.json")):
        response_data = json.loads(response_path.read_text())
        model = response_data["model"]
        items = response_data["items"]
        if [i["prompt"] for i in items] != probes:
            manifest["excluded"].append({"model": model, "reason": "ordered probe mismatch"})
            continue
        positions = [i for i, row in enumerate(items) if row["response"] and row["response"].strip()]
        texts = [items[i]["response"] for i in positions]
        key = hashlib.md5(str(sorted(texts)).encode()).hexdigest()
        candidates = cache_paths[key]
        candidates = sorted(candidates, key=lambda p: (not p.name.startswith(model.replace("/", "--") + "_"), p.name))
        accepted = None
        for candidate in candidates:
            with candidate.open("rb") as handle:
                data = pickle.load(handle)
            if data.get("encoder_name") != protocol["encoder"] or data.get("texts") != texts:
                continue
            raw = np.asarray(data["encodings"], dtype=np.float32)
            if raw.shape != (len(positions), 4096) or not np.isfinite(raw).all():
                continue
            if accepted is not None:
                difference = float(np.max(np.abs(raw - accepted[1])))
                duplicate_cache_differences.append({"model": model, "cache": str(candidate), "max_abs": difference})
                continue
            accepted = (candidate, raw)
        if accepted is None:
            manifest["excluded"].append({"model": model, "reason": "no exact ordered Qwen4096 cache"})
            continue
        path, raw = accepted
        matrix = np.zeros((600, 64), dtype=np.float32)
        matrix[positions] = raw[:, :64]
        model_ids.append(model)
        sliced_embeddings.append(matrix)
        full_norm = np.linalg.norm(raw, axis=1)
        slice_norm = np.linalg.norm(raw[:, :64], axis=1)
        norm_statistics.append({"model": model, "nonempty_answers": len(positions),
            "raw_norm_min": float(full_norm.min()), "raw_norm_max": float(full_norm.max()),
            "slice_norm_min": float(slice_norm.min()), "slice_norm_mean": float(slice_norm.mean()),
            "slice_norm_max": float(slice_norm.max())})
        item = {"model": model, "response_path": str(response_path), "response_sha256": sha256(response_path),
            "cache_path": str(path), "cache_sha256": sha256(path), "ordered_texts_sha256": json_hash(texts),
            "nonempty_indices": positions, "cache_dimensions": list(raw.shape)}
        saved_path = response_path.parent / (response_path.parent.name + "_dna.json")
        if saved_path.exists():
            payload = json.loads(saved_path.read_text())
            saved = np.asarray(payload["signature"], dtype=np.float32)
            if saved.shape == (128,):
                saved_vectors[model] = saved
                item["saved_dna_path"] = str(saved_path)
                item["saved_dna_sha256"] = sha256(saved_path)
        manifest["models"].append(item)
    if not model_ids:
        raise RuntimeError("No matching caches; inputs were not modified")
    model_index = {m: i for i, m in enumerate(model_ids)}
    raw64 = np.stack(sliced_embeddings)
    norms = np.linalg.norm(raw64, axis=2, keepdims=True)
    normalized64 = raw64 / np.where(norms == 0, 1, norms)
    flat = {"none": raw64.reshape(len(model_ids), -1), "l2": normalized64.reshape(len(model_ids), -1)}
    print(f"Matched {len(model_ids)} models, {len(saved_vectors)} saved signatures", flush=True)
    write_csv(output / "embedding_norms.csv", norm_statistics)
    manifest["duplicate_cache_comparisons"] = duplicate_cache_differences

    signatures = {}
    projection_checks, public_checks = [], []
    from llm_dna.dna.TextDNAExtractor import TextDNAExtractor
    manifest["public_extractor_sha256"] = sha256(ROOT / "src/llm_dna/dna/TextDNAExtractor.py")
    for seed in args.projection_seeds:
        reference_matrix = project_matrix(seed)
        extractor = TextDNAExtractor(dna_dim=128, pre_agg_embed_dim=64, normalize_embeddings=False, random_seed=seed)
        matrix = extractor._get_or_create_projection_matrix(38400, 128)
        np.testing.assert_array_equal(matrix, reference_matrix)
        cached_matrix = cache / f"projection_matrix_38400x128_seed{seed}.npy"
        check = {"seed": seed, "public_equals_historical_generator": True,
                 "array_sha256": hashlib.sha256(matrix.tobytes()).hexdigest()}
        if cached_matrix.exists():
            old = np.load(cached_matrix)
            check.update(cached_path=str(cached_matrix), cached_sha256=sha256(cached_matrix),
                         cached_max_abs_difference=float(np.max(np.abs(matrix-old))))
            np.testing.assert_array_equal(matrix, old)
        projection_checks.append(check)
        for variant in ["none", "l2"]:
            signatures[(variant, seed)] = flat[variant] @ matrix
            extractor.normalize_embeddings = variant == "l2"
            check_indices = range(len(model_ids)) if seed == 42 else [0, len(model_ids) // 2, len(model_ids)-1]
            for index in check_indices:
                # Feeding the first64 slice is mathematically equivalent; trailing
                # encoder coordinates are deliberately discarded by this pathway.
                direct = extractor.reduce_embeddings(raw64[index])
                batched = signatures[(variant, seed)][index]
                maximum_error = float(np.max(np.abs(direct - batched)))
                np.testing.assert_allclose(direct, batched, atol=5e-6, rtol=1e-4)
                public_checks.append({"seed": seed, "variant": variant, "model": model_ids[index],
                                      "batch_vs_public_max_abs": maximum_error})
    manifest["projection_checks"] = projection_checks
    manifest["public_implementation_checks"] = public_checks
    np.savez_compressed(output / "dna_vectors.npz", models=np.asarray(model_ids),
        **{f"{variant}_seed{seed}": vectors for (variant, seed), vectors in signatures.items()})

    replay = []
    if 42 in args.projection_seeds:
        for model, saved in sorted(saved_vectors.items()):
            for variant in ["l2", "none"]:
                actual = signatures[(variant, 42)][model_index[model]]
                error = actual - saved
                replay.append({"model": model, "variant": variant,
                    "max_abs_error": float(np.max(np.abs(error))), "l2_error": float(np.linalg.norm(error)),
                    "cosine_similarity": float(np.dot(actual, saved) / (np.linalg.norm(actual)*np.linalg.norm(saved))),
                    "within_1e_5": bool(np.max(np.abs(error)) <= 1e-5)})
    write_csv(output / "saved_signature_replay.csv", replay)

    relation_data, dataset_coverage = {}, []
    for path in sorted((private / "data/relation").glob("valid_llm_relation_enriched_seed*.csv")):
        with path.open() as handle:
            original = [(r["model_id"].strip(), r["related_model"].strip(), r["relation"].strip()) for r in csv.DictReader(handle)]
        # The original evaluator's method intersection uses a sorted set of triples.
        records = sorted({r for r in original if r[0] in model_index and r[1] in model_index})
        relation_data[path.stem] = records
        dataset_coverage.append({"file": str(path), "sha256": sha256(path), "original_rows": len(original),
                                 "usable_unique_rows": len(records), "positive_rows": int(labels(records).sum()),
                                 "models": len({m for r in records for m in r[:2]})})
    manifest["relation_files"] = dataset_coverage
    phylo_root = private / "out_phyloLM/squad_cqa_hs_wg_arc_mmlu"
    phylo_models = set()
    phylo_sources = []
    for model in model_ids:
        safe_name = model.replace("/", "_")
        path = phylo_root / safe_name / (safe_name + "_dna.json")
        if path.exists():
            payload = json.loads(path.read_text())
            if isinstance(payload.get("signature"), dict) and payload["signature"]:
                phylo_models.add(model)
                phylo_sources.append({"path": str(path), "sha256": sha256(path), "model": model})
    manifest["phylolm_cohort_sources"] = phylo_sources
    cohort_data = {
        "full_dna": relation_data,
        "paper_common": {name: [r for r in records if r[0] in phylo_models and r[1] in phylo_models
                                and "bottleneck" not in r[0].lower() and "bottleneck" not in r[1].lower()]
                         for name, records in relation_data.items()},
    }
    manifest["cohort_counts"] = {cohort: {name: {"rows": len(records), "models": len({m for r in records for m in r[:2]})}
                               for name, records in datasets.items()} for cohort, datasets in cohort_data.items()}
    paper_runs, historical_runs, paper_splits = [], [], []
    saved_matrix = np.stack([saved_vectors.get(m, np.zeros(128)) for m in model_ids])
    for cohort, datasets in cohort_data.items():
        for dataset, records in datasets.items():
            y = labels(records)
            for split_seed in args.split_seeds:
                train, test = train_test_split(np.arange(len(y)), test_size=.2, random_state=split_seed, stratify=y)
                train_pairs = {tuple(sorted(records[i][:2])) for i in train}
                test_pairs = {tuple(sorted(records[i][:2])) for i in test}
                paper_splits.append({"cohort": cohort, "dataset": dataset, "split_seed": split_seed,
                    "records_sha256": json_hash(records), "train_indices": train.tolist(), "test_indices": test.tolist(),
                    "unordered_pair_overlap_count": len(train_pairs & test_pairs)})
                for (variant, projection_seed), vectors in signatures.items():
                    scores = evaluate(features(vectors, records, model_index), y, train, test, split_seed)
                    paper_runs.append({"cohort": cohort, "variant": variant, "projection_seed": projection_seed,
                        "dataset": dataset, "split_seed": split_seed, **scores})
                if all(m in saved_vectors for r in records for m in r[:2]):
                    scores = evaluate(features(saved_matrix, records, model_index), y, train, test, split_seed)
                    historical_runs.append({"cohort": cohort, "variant": "saved_private", "projection_seed": 42,
                        "dataset": dataset, "split_seed": split_seed, **scores})
    write_csv(output / "paper_protocol_runs.csv", paper_runs + historical_runs)
    write_json(output / "paper_protocol_splits.json", paper_splits)

    # Selection is kept separate from historical reporting. Collapse metadata
    # relation aliases for each ordered pair. Reverse directions stay together.
    pair_labels = collections.defaultdict(set)
    for records in relation_data.values():
        for a, b, relation in records:
            pair_labels[(a, b)].add(int(relation.lower() != "nan"))
    if any(len(v) != 1 for v in pair_labels.values()):
        raise ValueError("Contradictory labels for an ordered pair; selection requires review")
    unique_records = [(a, b, "related" if next(iter(v)) else "nan") for (a, b), v in sorted(pair_labels.items())]
    groups = sorted({tuple(sorted(r[:2])) for r in unique_records})
    group_label = [max(next(iter(pair_labels[k])) for k in [g, g[::-1]] if k in pair_labels) for g in groups]
    development_groups, heldout_groups = train_test_split(np.arange(len(groups)), test_size=.2,
                    random_state=args.holdout_seed, stratify=group_label)
    y = labels(unique_records)
    row_groups = [groups.index(tuple(sorted(r[:2]))) for r in unique_records]
    def rows_for(group_indices):
        present = set(group_indices)
        return np.asarray([i for i, group in enumerate(row_groups) if group in present])
    heldout_rows = rows_for(heldout_groups)
    development_rows = rows_for(development_groups)
    selector = {"records": unique_records, "groups": groups, "row_group_indices": row_groups,
        "heldout_group_indices": heldout_groups.tolist(), "heldout_indices": heldout_rows.tolist(),
        "development_indices": development_rows.tolist(), "validation_splits": []}
    validation_runs = []
    for split_seed in args.split_seeds:
        train_groups, validation_groups = train_test_split(development_groups, test_size=.25,
            random_state=split_seed, stratify=np.asarray(group_label)[development_groups])
        train, validation = rows_for(train_groups), rows_for(validation_groups)
        selector["validation_splits"].append({"split_seed": split_seed, "train_indices": train.tolist(),
                                              "validation_indices": validation.tolist()})
        for (variant, projection_seed), vectors in signatures.items():
            scores = evaluate(features(vectors, unique_records, model_index), y, train, validation, split_seed)
            validation_runs.append({"variant": variant, "projection_seed": projection_seed,
                                    "split_seed": split_seed, **scores})
    validation_summary = summarize(validation_runs, ["variant"])
    ranked = sorted(validation_summary, key=lambda row: (-row["f1_mean"], -row["auc_mean"], row["variant"] != "none"))
    winner = ranked[0]["variant"]
    selector.update(selected_variant=winner, selection_metrics=validation_summary,
                    decision_rule=protocol["selection"])
    write_json(output / "selection.json", selector)  # Freeze choice before reading test metrics.
    write_csv(output / "validation_runs.csv", validation_runs)
    print(f"Validation selected {winner}: {validation_summary}", flush=True)
    heldout_runs = []
    for (variant, projection_seed), vectors in signatures.items():
        scores = evaluate(features(vectors, unique_records, model_index), y, development_rows, heldout_rows, args.holdout_seed)
        heldout_runs.append({"variant": variant, "projection_seed": projection_seed, **scores})
    write_csv(output / "heldout_runs.csv", heldout_runs)

    original_summary = summarize(paper_runs + historical_runs, ["cohort", "variant", "projection_seed"])
    paired_projection_runs = []
    for cohort in cohort_data:
        for projection_seed in args.projection_seeds:
            paired = {r["variant"]: r for r in original_summary
                      if r["cohort"] == cohort and r["projection_seed"] == projection_seed}
            row = {"cohort": cohort, "projection_seed": projection_seed}
            for metric in METRICS:
                row[metric + "_none"] = paired["none"][metric + "_mean"]
                row[metric + "_l2"] = paired["l2"][metric + "_mean"]
                row[metric + "_delta_none_minus_l2"] = row[metric + "_none"] - row[metric + "_l2"]
            paired_projection_runs.append(row)
    write_csv(output / "paired_projection_summary.csv", paired_projection_runs)
    overview = {"models": len(model_ids), "saved_signatures": len(saved_vectors), "selected_variant": winner,
        "normalization_default": winner == "l2", "paper_protocol": original_summary,
        "all_projection_seeds": summarize(paper_runs, ["cohort", "variant"]), "validation": validation_summary,
        "heldout": summarize(heldout_runs, ["variant"]),
        "replay": {variant: {"count": sum(r["variant"] == variant for r in replay),
            "within_1e_5": sum(r["variant"] == variant and r["within_1e_5"] for r in replay),
            "max_abs_error": max((r["max_abs_error"] for r in replay if r["variant"] == variant), default=0.0)} for variant in ["l2", "none"]},
        "original_pair_overlap": {"min": min(r["unordered_pair_overlap_count"] for r in paper_splits),
                                   "max": max(r["unordered_pair_overlap_count"] for r in paper_splits)},
        "selection_data": {"ordered_pairs": len(unique_records), "unordered_groups": len(groups),
                           "positive_ordered_pairs": int(y.sum()), "heldout_rows": len(heldout_rows),
                           "heldout_positives": int(y[heldout_rows].sum()),
                           "reverse_direction_label_disagreements": sum(len({next(iter(pair_labels[k])) for k in [g, g[::-1]] if k in pair_labels}) > 1 for g in groups)}}
    write_json(output / "results.json", overview)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["script_sha256_at_finish"] = sha256(Path(__file__))
    write_json(output / "manifest.json", manifest)
    print(json.dumps(overview, indent=2), flush=True)


if __name__ == "__main__":
    main()
