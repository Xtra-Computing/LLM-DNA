#!/usr/bin/env python3
"""Refit PhyloLM and three DNA methods from positive labels and cached signatures.

Use the original directed, stratified 80/20 record split with dataset and split
seeds 0-4. No historical result files, model inference, or downloads are needed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import sys

for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[variable] = "1"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reproduce_relationship as replay

np = replay.np
ROOT = replay.ROOT
ENCODERS = {"dna_svm": "Qwen3", "dna_bge": "BGE", "dna_mpnet": "MPNet"}
REPRESENTATIONS = {
    "difference": "Original difference",
    "concatenation": "Direct concatenation",
    "landmark_euclidean": "Euclidean landmarks",
}


def binary_labels(records):
    return np.asarray([int(relation.lower() != "nan") for _, _, relation in records], dtype=np.int32)


def split_records(records, seed):
    return replay.train_test_split(np.arange(len(records)), test_size=0.2, random_state=seed,
                                   stratify=binary_labels(records))


def dna_distances(signatures, models, metric="euclidean"):
    if metric != "euclidean":
        raise ValueError("DNA landmarks use Euclidean distance")
    matrix = np.asarray([signatures[m] for m in models], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("DNA vectors must be finite")
    distances = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=2)
    np.fill_diagonal(distances, 0.0)
    return {(a, b): float(distances[i, j]) for i, a in enumerate(models) for j, b in enumerate(models)}


def pair_features(records, signatures, train, representation, distances=None):
    if representation == "difference":
        return replay.dna_features(records, signatures), None
    if representation == "concatenation":
        return np.vstack([np.concatenate([signatures[a], signatures[b]]) for a, b, _ in records]), None
    if representation not in ("landmark_euclidean", "landmark_nei"):
        raise ValueError(f"Unknown feature representation: {representation}")
    if distances is None:
        raise ValueError("Landmark features require precomputed distances")
    return replay.landmark_features(records, train, distances)


def fit_one(records, signatures, partition, representation, distances=None):
    train = np.asarray(partition["train_indices"], dtype=int)
    test = np.asarray(partition["test_indices"], dtype=int)
    labels = binary_labels(records)
    matrix, landmarks = pair_features(records, signatures, train, representation, distances)
    model = replay.Pipeline([
        ("scaler", replay.StandardScaler()),
        ("classifier", replay.SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                                  class_weight="balanced", random_state=partition["split_seed"])),
    ])
    model.fit(matrix[train], labels[train])
    predictions = model.predict(matrix[test])
    probabilities = model.predict_proba(matrix[test])[:, 1]
    return {
        **partition, "protocol": "record", "representation": representation,
        "feature_dimension": matrix.shape[1], "feature_dtype": str(matrix.dtype),
        "landmark_models": landmarks, "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(), "metrics": replay.scores(labels[test], predictions, probabilities),
        "training_accuracy": float(np.mean(model.predict(matrix[train]) == labels[train])),
        "effective_gamma": float(model.named_steps["classifier"]._gamma),
    }


def load_inputs(cache_root, relations=replay.DEFAULT_RELATIONS):
    datasets, relation_source = replay.load_relation_datasets(relations)
    known = {m for records in datasets for a, b, _ in records for m in (a, b)}
    vectors, manifest, inventory = replay.load_signatures(cache_root, known)
    if not vectors["dna_svm"] or not vectors["phylolm_svm"]:
        raise ValueError("Qwen and PhyloLM caches are required to reconstruct the original common cohort")
    common = vectors["dna_svm"].keys() & vectors["phylolm_svm"].keys()
    cohorts = []
    for seed, source_rows in zip(replay.SEEDS, datasets):
        records = sorted(set(r for r in source_rows if r[0] in common and r[1] in common))
        models = sorted({m for a, b, _ in records for m in (a, b)})
        labels = binary_labels(records)
        if set(labels.tolist()) != {0, 1}:
            raise ValueError(f"Dataset {seed} must contain both classes after applying the original cohort filter")
        for method in ENCODERS:
            missing = set(models) - vectors[method].keys()
            if missing:
                raise ValueError(f"Missing {ENCODERS[method]} DNA in dataset {seed}: {', '.join(sorted(missing))}")
        cohorts.append({"dataset_seed": seed, "input_records": len(source_rows), "records": records,
                        "models": models, "positive": int(labels.sum()), "negative": int(len(labels) - labels.sum())})
    manifest.append(relation_source)
    return cohorts, vectors, sorted(manifest, key=lambda e: e["path"]), inventory


def evaluate(cohorts, vectors):
    models = sorted({m for cohort in cohorts for m in cohort["models"]})
    print(f"Computing PhyloLM Nei distances from cached signatures for {len(models)} models", flush=True)
    nei_distances = {}
    for i, a in enumerate(models):
        for b in models[i:]:
            nei_distances[a, b] = nei_distances[b, a] = replay.nei_distance(
                vectors["phylolm_svm"][a], vectors["phylolm_svm"][b])
    runs = []
    for cohort in cohorts:
        records, seed = cohort["records"], cohort["dataset_seed"]
        labels = binary_labels(records)
        for method in ("phylolm_svm", *ENCODERS):
            distances = (nei_distances if method == "phylolm_svm" else
                         dna_distances(vectors[method], cohort["models"]))
            representations = ("landmark_nei",) if method == "phylolm_svm" else REPRESENTATIONS
            for split_seed in replay.SEEDS:
                train, test = split_records(records, split_seed)
                train_models = {m for i in train for m in records[i][:2]}
                test_models = {m for i in test for m in records[i][:2]}
                partition = {"method": method, "dataset_seed": seed, "split_seed": split_seed,
                             "train_indices": train.tolist(), "test_indices": test.tolist(),
                             "labels": labels[test].tolist(), "train_test_shared_models": sorted(train_models & test_models)}
                for representation in representations:
                    fitted = fit_one(records, vectors[method], partition, representation,
                                     distances if representation.startswith("landmark_") else None)
                    if representation.startswith("landmark_") and fitted["landmark_models"] != sorted(train_models):
                        raise ValueError("Landmarks must be the sorted training-model identities")
                    runs.append(fitted)
        print(f"Dataset {seed}: refitted 50 SVMs (5 PhyloLM + 45 DNA)", flush=True)
    return runs


def summarize(runs):
    rows = []
    for method, representation in dict.fromkeys((r["method"], r["representation"]) for r in runs):
        selected = [r for r in runs if r["method"] == method and r["representation"] == representation]
        for metric in replay.METRICS:
            values = [r["metrics"][metric] for r in selected]
            rows.append({"protocol": "record", "method": method, "representation": representation,
                         "metric": metric, "runs": len(values), "mean": statistics.mean(values),
                         "std": statistics.stdev(values)})
    return rows


def table_rows(summary):
    lookup = {(r["method"], r["representation"], r["metric"]): r for r in summary}
    rows = []
    specifications = [("phylolm_svm", "PhyloLM", "landmark_nei", "Nei landmarks")]
    specifications += [(method, encoder, representation, label) for method, encoder in ENCODERS.items()
                       for representation, label in REPRESENTATIONS.items()]
    for method, encoder, representation, label in specifications:
        row = {"encoder": encoder, "method": label, "representation": representation}
        for metric in replay.METRICS:
            item = lookup[method, representation, metric]
            row[f"{metric}_mean"], row[f"{metric}_std"] = item["mean"], item["std"]
        row["runs"] = lookup[method, representation, "f1"]["runs"]
        rows.append(row)
    return rows


def markdown_table(rows):
    lines = ["| Encoder | Method | Accuracy | Precision | Recall | F1 | AUC |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        cells = [f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}" for metric in replay.METRICS]
        lines.append("| " + " | ".join([row["encoder"], row["method"], *cells]) + " |")
    return "\n".join(lines) + "\n"


def latex_table(rows):
    lines = ["% Original directed 80/20 record splits; 25 runs per row; sample standard deviation.",
             r"\begin{tabular}{llccccc}", r"\hline",
             r"Encoder & Method & Accuracy & Precision & Recall & F1 & AUC \\", r"\hline"]
    for row in rows:
        cells = [f"${row[f'{metric}_mean']:.3f} \\pm {row[f'{metric}_std']:.3f}$" for metric in replay.METRICS]
        lines.append(" & ".join([row["encoder"], row["method"], *cells]) + r" \\")
    lines += [r"\hline", r"\end{tabular}", ""]
    return "\n".join(lines)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(cache_root, output, save_runs=False, relations=replay.DEFAULT_RELATIONS):
    cache_root, output = cache_root.resolve(), output.resolve()
    if not cache_root.is_dir():
        raise ValueError(f"Cache root does not exist: {cache_root}")
    if output == cache_root or cache_root in output.parents:
        raise ValueError("Output must be outside the read-only cache root")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output directory must be new or empty: {output}")
    cohorts, vectors, manifest, _ = load_inputs(cache_root, relations)
    relation_source = next(row for row in manifest if row.get("kind") == "positive_relation_csv")
    runs = evaluate(cohorts, vectors)
    summary = summarize(runs)
    result = {
        "experiment": "DNA and PhyloLM relationship table", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "fits": len(runs), "methods": ["phylolm_svm", *ENCODERS], "representations": ["landmark_nei", *REPRESENTATIONS],
        "method_representations": {"phylolm_svm": ["landmark_nei"], **{method: list(REPRESENTATIONS) for method in ENCODERS}},
        "protocol": {"split": "directed_record", "test_size": 0.2, "dataset_seeds": list(replay.SEEDS),
                     "split_seeds": list(replay.SEEDS), "std_ddof": 1, "pair_disjoint": False, "model_disjoint": False,
                     "split_source": "positive-relation CSV with seeded directed negative sampling",
                     "relations": relation_source, "historical_results_required": False,
                     "landmarks": "sorted unique training-record models only", "additional_dna_normalization": False,
                     "phylolm": {"refitted": True, "distance": "Nei", "similarity_cutoff": 1e-9, "distance_clamp": 10.0},
                     "classifier": {"scaler": "StandardScaler fitted on training rows only", "kernel": "rbf", "C": 1.0,
                                    "gamma": "scale", "class_weight": "balanced", "probability": True, "random_state": "split_seed"}},
        "cohorts": cohorts, "summary": summary, "cache_root": str(cache_root),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "sklearn": replay.sklearn.__version__, "torch": sys.modules["torch"].__version__},
    }
    rows = table_rows(summary)
    output.mkdir(parents=True, exist_ok=True)
    replay.save_json(output / "results.json", result)
    write_csv(output / "table.csv", rows)
    if save_runs:
        replay.save_json(output / "runs.json", runs)
        write_csv(output / "metrics.csv", [{**{k: r[k] for k in (
            "method", "representation", "dataset_seed", "split_seed", "feature_dimension", "training_accuracy", "effective_gamma")},
            **r["metrics"]} for r in runs])
    (output / "table.md").write_text(markdown_table(rows))
    (output / "table.tex").write_text(latex_table(rows))
    print("\n" + markdown_table(rows), flush=True)
    print(f"Refitted {len(runs)} SVMs; tables: {output}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=os.environ.get("REPRO_CACHE_ROOT") or None,
                        help="Directory containing DNA/PhyloLM caches; required unless REPRO_CACHE_ROOT is set.")
    parser.add_argument("--relations", type=Path, default=replay.DEFAULT_RELATIONS,
                        help="Positive-relation CSV; default: configs/valid_llm_relation.csv in this checkout.")
    parser.add_argument("--output-dir", type=Path,
                        help="New or empty output directory; default: experiments/paper_reproduction/relationship_table/<UTC timestamp>.")
    parser.add_argument("--save-runs", action="store_true", help="Also save per-run predictions and metrics.")
    args = parser.parse_args(argv)
    if args.cache_root is None:
        parser.error("--cache-root or REPRO_CACHE_ROOT is required")
    output = args.output_dir or ROOT / "experiments/paper_reproduction/relationship_table" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        run(args.cache_root, output, args.save_runs, args.relations)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
