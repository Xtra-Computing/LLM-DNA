#!/usr/bin/env python3
"""Reproduce relationship evaluation from positive labels and cached signatures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Keep this CPU-only replay predictable without changing the caller's settings.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from llm_dna.dna.DNASignature import DNASignature

PAPER_URL = "https://arxiv.org/html/2509.24496v3#S5.T1"
DATASET = "squad_cqa_hs_wg_arc_mmlu"
SEEDS = tuple(range(5))
DEFAULT_RELATIONS = ROOT / "configs/valid_llm_relation.csv"
METRICS = ("accuracy", "precision", "recall", "f1", "auc")
METHODS = {
    "random": "Random", "greedy": "Greedy", "phylolm_svm": "PhyloLM + SVM",
    "dna_svm": "DNA (Qwen3) + SVM", "dna_bge": "DNA (BGE) + SVM",
    "dna_mpnet": "DNA (MPNet) + SVM",
}
PAPER = {
    "random": [(0.493, 0.036), (0.515, 0.034), (0.516, 0.042), (0.516, 0.036), (0.492, 0.036)],
    "greedy": [(0.593, 0.018), (0.759, 0.081), (0.329, 0.000), (0.458, 0.014), (0.606, 0.023)],
    "phylolm_svm": [(0.742, 0.058), (0.741, 0.073), (0.812, 0.123), (0.766, 0.057), (0.788, 0.060)],
    "dna_svm": [(0.919, 0.048), (0.898, 0.049), (0.957, 0.072), (0.925, 0.047), (0.979, 0.027)],
    "dna_bge": [(0.934, 0.053), (0.905, 0.068), (0.982, 0.059), (0.940, 0.050), (0.992, 0.023)],
    "dna_mpnet": [(0.932, 0.048), (0.914, 0.065), (0.966, 0.069), (0.937, 0.047), (0.990, 0.020)],
}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(handle.read()).hexdigest()


def read_records(path: Path) -> list[tuple[str, str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [(r["model_id"].strip(), r["related_model"].strip(), r["relation"].strip()) for r in csv.DictReader(handle)]
    if not rows or any(not a or not b or not relation for a, b, relation in rows):
        raise ValueError(f"Empty relation dataset or missing model/relation in {path}")
    return rows


def load_relation_datasets(relations: Path = DEFAULT_RELATIONS):
    """Reconstruct the original seeded datasets from a positive-relation CSV."""
    relations = relations.resolve()
    positives = read_records(relations)
    if any(relation.lower() == "nan" for _, _, relation in positives):
        raise ValueError("Relations CSV must contain positive records only; negative pairs are sampled automatically")
    positive_pairs = {(a, b) for a, b, _ in positives}
    models = sorted({model for pair in positive_pairs for model in pair})
    available = len(models) * (len(models) - 1) - sum(a != b for a, b in positive_pairs)
    if available < len(positives):
        raise ValueError("Not enough distinct directed pairs to sample one negative per positive record")
    datasets = []
    for seed in SEEDS:
        rng = random.Random(seed)
        used = set(positive_pairs)
        negatives = []
        for _ in range(len(positives) * 20):
            a, b = rng.choice(models), rng.choice(models)
            if a == b or (a, b) in used:
                continue
            negatives.append((a, b, "nan"))
            used.add((a, b))
            if len(negatives) == len(positives):
                break
        if len(negatives) != len(positives):
            raise ValueError(f"Unable to sample the required negative pairs for dataset seed {seed}")
        records = positives + negatives
        rng.shuffle(records)
        datasets.append(records)
    source = {
        "path": str(relations.relative_to(ROOT)) if relations.is_relative_to(ROOT) else relations.name,
        "sha256": sha256(relations), "kind": "positive_relation_csv",
        "positive_records": len(positives), "negative_records_per_seed": len(positives),
    }
    return datasets, source


def resolve_model(metadata: dict, path: Path, known: set[str]) -> str | None:
    """Use explicit metadata, then exact filename identity; never guess an ID."""
    for key in ("model_id", "repo_id", "model_name", "huggingface_id"):
        value = metadata.get(key)
        if isinstance(value, str) and "/" in value:
            if value in known and path.stem.removesuffix("_dna") != value.replace("/", "_"):
                raise ValueError(f"Conflicting metadata and filename model identity: {path}")
            return value
    safe = path.stem.removesuffix("_dna")
    matches = [model for model in known if model.replace("/", "_") == safe]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous model identity in {path}: {matches}")
    return matches[0] if matches else None


def load_signatures(cache_root: Path, known: set[str]):
    vectors, manifest, inventory = {}, [], {}
    roots = {
        "dna_svm": (cache_root / "out" / DATASET, "*/*_dna.json"),
        "dna_bge": (cache_root / "out" / DATASET, "*/bge/*_dna.json"),
        "dna_mpnet": (cache_root / "out" / DATASET, "*/mpnet/*_dna.json"),
        "phylolm_svm": (cache_root / "out_phyloLM" / DATASET, "*/*_dna.json"),
    }
    for method, (root, pattern) in roots.items():
        paths = sorted(root.glob(pattern))
        selected = {}
        for path in paths:
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
                raise ValueError(f"Malformed signature metadata: {path}")
            model = resolve_model(payload["metadata"], path, known)
            if model not in known:
                continue
            if model in selected:
                raise ValueError(f"Duplicate {method} signature for {model}: {path}")
            if method == "phylolm_svm":
                signature = payload.get("signature")
                if not isinstance(signature, dict) or not signature:
                    raise ValueError(f"Empty or invalid PhyloLM signature: {path}")
                if payload["metadata"].get("probe_count") != 600:
                    raise ValueError(f"Unexpected PhyloLM probe count: {path}")
                for probe, distribution in signature.items():
                    if not isinstance(distribution, dict) or any(
                        not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 or v > 1
                        for v in distribution.values()
                    ):
                        raise ValueError(f"Invalid PhyloLM probabilities at {probe}: {path}")
            else:
                dna = DNASignature.load(path)
                signature = dna.signature
                if signature.shape != (128,) or not np.isfinite(signature).all():
                    raise ValueError(f"Expected finite 128-dimensional {method} DNA: {path}")
                if dna.metadata.dna_dimension != 128 or dna.metadata.probe_count != 600:
                    raise ValueError(f"Unexpected DNA dimension/probe metadata: {path}")
                expected_encoder = {
                    "dna_svm": "Qwen/Qwen3-Embedding-8B", "dna_bge": "BAAI/bge-large-en-v1.5",
                    "dna_mpnet": "sentence-transformers/all-mpnet-base-v2",
                }[method]
                if dna.metadata.extractor_config.get("encoder") != expected_encoder:
                    raise ValueError(f"Unexpected encoder for {method}: {path}")
            selected[model] = signature
            manifest.append({"path": str(path.relative_to(cache_root)), "sha256": sha256(path), "method": method, "model": model})
        vectors[method] = selected
        inventory[method] = {"files_found": len(paths), "relation_models_loaded": len(selected), "missing_relation_models": sorted(known - selected.keys())}
    return vectors, manifest, inventory


def nei_distance(left: dict, right: dict) -> float:
    """Original PhyloLM formula, including its finite-distance clamp.

    Sorting the summation order makes floating point output hash-seed independent.
    This only changes accumulation rounding relative to the original set iteration.
    """
    product = left_sq = right_sq = 0.0
    for probe in sorted(left.keys() | right.keys()):
        a, b = left.get(probe, {}), right.get(probe, {})
        for allele in sorted(a.keys() | b.keys()):
            pa, pb = a.get(allele, 0.0), b.get(allele, 0.0)
            product += pa * pb
            left_sq += pa * pa
            right_sq += pb * pb
    denom = np.sqrt(left_sq * right_sq)
    similarity = (1.0 if product == 0 else 0.0) if denom == 0 else float(np.clip(product / denom, 0, 1))
    return 10.0 if similarity < 1e-9 else float(-np.log(similarity))


def dna_features(records, signatures):
    differences = np.vstack([signatures[a] - signatures[b] for a, b, _ in records])
    return np.concatenate([np.abs(differences), differences], axis=1)


def landmark_features(records, train_indices, distances):
    landmarks = sorted({model for i in train_indices for model in records[i][:2]})
    features = np.asarray([
        [distances[(a, landmark)] for landmark in landmarks] + [distances[(b, landmark)] for landmark in landmarks]
        for a, b, _ in records
    ])
    return features, landmarks


def scores(labels, predictions, probabilities):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    return dict(zip(METRICS, map(float, (accuracy_score(labels, predictions), precision, recall, f1, roc_auc_score(labels, probabilities)))))


def evaluate(records, vectors, dataset_seed, split_seeds=SEEDS):
    labels = np.asarray([0 if relation.lower() == "nan" else 1 for _, _, relation in records], dtype=np.int32)
    if len(np.unique(labels)) != 2:
        raise ValueError("Relationship cohort must contain both positive and negative labels")
    indices = np.arange(len(records))
    distances = {}
    phylo = vectors["phylolm_svm"]
    models = sorted({m for a, b, _ in records for m in (a, b)})
    for i, a in enumerate(models):
        for b in models[i:]:
            distances[a, b] = distances[b, a] = nei_distance(phylo[a], phylo[b])
    features = {method: dna_features(records, signatures) for method, signatures in vectors.items() if method.startswith("dna") and set(models) <= signatures.keys()}
    runs = []
    for method in METHODS:
        if method.startswith("dna") and method not in features:
            continue
        for seed in (None,) if method == "greedy" else split_seeds:
            landmarks = None
            train_indices, test_indices = np.asarray([], dtype=int), indices
            if method == "greedy":
                predictions = np.asarray([int(a.split("/")[0] == b.split("/")[0]) for a, b, _ in records])
                probabilities = predictions
            elif method == "random":
                predictions = np.random.default_rng(seed).integers(0, 2, size=len(labels))
                probabilities = predictions
            else:
                train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=seed, stratify=labels)
                if method == "phylolm_svm":
                    x, landmarks = landmark_features(records, train_indices, distances)
                else:
                    x = features[method]
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("classifier", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
                ])
                model.fit(x[train_indices], labels[train_indices])
                predictions = model.predict(x[test_indices])
                probabilities = model.predict_proba(x[test_indices])[:, 1]
            train_models = {m for i in train_indices for m in records[i][:2]}
            test_models = {m for i in test_indices for m in records[i][:2]}
            runs.append({
                "method": method, "dataset_seed": dataset_seed, "split_seed": seed,
                "evaluation_scope": "full_common_cohort" if method in ("random", "greedy") else "stratified_pair_test_20_percent",
                "train_indices": train_indices.tolist(), "test_indices": test_indices.tolist(),
                "train_test_shared_models": sorted(train_models & test_models),
                "landmark_models": landmarks,
                "labels": labels[test_indices].tolist(), "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist(),
                "metrics": scores(labels[test_indices], predictions, probabilities),
            })
    return runs


def aggregate(runs):
    result = {}
    for method in METHODS:
        selected = [run for run in runs if run["method"] == method]
        result[method] = {"label": METHODS[method], "runs": len(selected), "status": "complete" if selected else "unavailable", "metrics": {}}
        for metric, (paper_mean, paper_std) in zip(METRICS, PAPER[method]):
            values = [run["metrics"][metric] for run in selected]
            mean = statistics.mean(values) if values else None
            std = statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)
            result[method]["metrics"][metric] = {
                "mean": mean, "std": std, "paper_mean": paper_mean, "paper_std": paper_std,
                "mean_delta": mean - paper_mean if mean is not None else None,
                "matches_paper_3dp": mean is not None and f"{mean:.3f}" == f"{paper_mean:.3f}" and f"{std:.3f}" == f"{paper_std:.3f}",
            }
    return result


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run(cache_root: Path, output: Path, relations: Path = DEFAULT_RELATIONS):
    cache_root, output = cache_root.resolve(), output.resolve()
    if not cache_root.is_dir():
        raise ValueError(f"Cache root does not exist: {cache_root}")
    if output == cache_root or cache_root in output.parents:
        raise ValueError("Output must be outside the read-only cache root")
    datasets, relation_source = load_relation_datasets(relations)
    known = {model for rows in datasets for a, b, _ in rows for model in (a, b)}
    vectors, manifest, inventory = load_signatures(cache_root, known)
    if not vectors["dna_svm"] or not vectors["phylolm_svm"]:
        raise ValueError("Qwen and PhyloLM caches are required to reconstruct the paper comparison cohort")
    common = vectors["dna_svm"].keys() & vectors["phylolm_svm"].keys()
    cohorts, runs, warnings = [], [], []
    for seed, rows in zip(SEEDS, datasets):
        records = sorted(set(row for row in rows if row[0] in common and row[1] in common))
        if len(records) != (144, 149, 146, 141, 147)[seed]:
            warnings.append(f"Dataset seed {seed} has {len(records)} eligible pairs; saved paper cohort expected {(144, 149, 146, 141, 147)[seed]}.")
        models = sorted({m for a, b, _ in records for m in (a, b)})
        counts = Counter(0 if relation.lower() == "nan" else 1 for _, _, relation in records)
        cohort = {"dataset_seed": seed, "input_pairs": len(rows), "records": records, "models": models,
                  "positive": counts[1], "negative": counts[0], "dropped_records": [
                      {"record": row, "missing": {method: [m for m in row[:2] if m not in vectors[method]] for method in ("dna_svm", "phylolm_svm") if any(m not in vectors[method] for m in row[:2])}}
                      for row in rows if row[0] not in common or row[1] not in common]}
        cohorts.append(cohort)
        for method in ("dna_bge", "dna_mpnet"):
            missing = set(models) - vectors[method].keys()
            if missing:
                warnings.append(f"{method} unavailable for dataset seed {seed}; missing {sorted(missing)}")
        print(f"Dataset seed {seed}: {len(records)} pairs, {len(models)} models", flush=True)
        runs.extend(evaluate(records, vectors, seed))
    manifest.append(relation_source)
    summary = aggregate(runs)
    for method, row in summary.items():
        if row["runs"] != (5 if method == "greedy" else 25):
            row["status"] = "partial" if row["runs"] else "unavailable"
    result = {
        "experiment": "Table 1 relationship prediction", "paper_url": PAPER_URL,
        "protocol": {"dataset_seeds": SEEDS, "split_seeds": SEEDS, "test_size": 0.2,
                     "relations": relation_source,
                     "std_ddof": 1, "dna_features": "concat(abs(a-b), a-b)", "pair_split_not_model_disjoint": True,
                     "random_and_greedy_evaluate_full_cohort": True},
        "provenance": {"implementation_sha256": sha256(Path(__file__)),
                       "dna_loader_sha256": sha256(ROOT / "src/llm_dna/dna/DNASignature.py"),
                       "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
                       "thread_environment": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}},
        "inventory": inventory, "cohorts": cohorts, "summary": summary, "warnings": warnings,
    }
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "results.json", result)
    save_json(output / "runs.json", runs)
    save_json(output / "input_manifest.json", sorted(manifest, key=lambda r: r["path"]))
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "dataset_seed", "split_seed", "evaluation_scope", *METRICS])
        writer.writeheader()
        writer.writerows({**{key: row[key] for key in ("method", "dataset_seed", "split_seed", "evaluation_scope")}, **row["metrics"]} for row in runs)
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "metric", "runs", "status", "mean", "std", "paper_mean", "paper_std", "mean_delta", "matches_paper_3dp"])
        writer.writeheader()
        writer.writerows({"method": method, "metric": metric, "runs": row["runs"], "status": row["status"], **value} for method, row in summary.items() for metric, value in row["metrics"].items())
    matched = sum(v["matches_paper_3dp"] for row in summary.values() for v in row["metrics"].values())
    print(f"Table 1: {matched}/30 mean±std cells match the paper at 3 decimals; artifacts: {output}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True,
                        help="Directory containing cached DNA and PhyloLM signatures.")
    parser.add_argument("--relations", type=Path, default=DEFAULT_RELATIONS,
                        help="Positive-relation CSV; default: configs/valid_llm_relation.csv in this checkout.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/paper_reproduction/relationship")
    args = parser.parse_args(argv)
    run(args.cache_root, args.output_dir, args.relations)


if __name__ == "__main__":
    main()
