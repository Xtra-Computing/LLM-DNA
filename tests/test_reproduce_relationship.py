"""Scientific protocol checks for the offline Table 1 runner."""

import importlib.util
import csv
import json
from pathlib import Path
import random

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "reproduce_relationship", Path(__file__).resolve().parents[1] / "scripts/reproduce_relationship.py"
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def test_seeded_relation_sampling_matches_original_records_and_keeps_direction(tmp_path):
    path = tmp_path / "relations.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_id", "related_model", "relation"])
        writer.writerows([("a", "b", "cardData"), ("a", "b", "tags"), ("b", "c", "finetune")])
    state = random.getstate()
    datasets, source = replay.load_relation_datasets(path)
    assert random.getstate() == state
    assert datasets[0] == [("a", "b", "tags"), ("b", "a", "nan"), ("c", "b", "nan"),
                           ("b", "c", "finetune"), ("c", "a", "nan"), ("a", "b", "cardData")]
    assert datasets[1] == [("c", "b", "nan"), ("a", "b", "tags"), ("b", "c", "finetune"),
                           ("a", "b", "cardData"), ("b", "a", "nan"), ("a", "c", "nan")]
    assert source["sha256"] == replay.sha256(path)
    assert source["positive_records"] == source["negative_records_per_seed"] == 3
    for records in datasets:
        positives = [r for r in records if r[2] != "nan"]
        negatives = [(a, b) for a, b, relation in records if relation == "nan"]
        assert sorted(positives) == [("a", "b", "cardData"), ("a", "b", "tags"), ("b", "c", "finetune")]
        assert len(set(negatives)) == len(negatives) == 3
        assert all(a != b and (a, b) not in {("a", "b"), ("b", "c")} for a, b in negatives)


@pytest.mark.parametrize("rows,message", [
    ([("a", "b", "nan")], "positive records only"),
    ([("a", "b", "cardData"), ("b", "a", "cardData")], "Not enough distinct directed pairs"),
])
def test_invalid_relation_source_is_rejected(tmp_path, rows, message):
    path = tmp_path / "relations.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_id", "related_model", "relation"])
        writer.writerows(rows)
    with pytest.raises(ValueError, match=message):
        replay.load_relation_datasets(path)


def test_dna_features_preserve_signed_pair_order_and_float32():
    signatures = {"a": np.array([1, 5], dtype=np.float32), "b": np.array([3, 1], dtype=np.float32)}
    features = replay.dna_features([("a", "b", "related"), ("b", "a", "nan")], signatures)
    np.testing.assert_array_equal(features, [[2, 4, -2, 4], [2, 4, 2, -4]])
    assert features.dtype == np.float32


def test_phylo_landmarks_never_include_test_only_models():
    records = [("a", "b", "related"), ("b", "c", "nan"), ("c", "held_out", "related")]
    distances = {(a, b): float(i + j) for i, a in enumerate(["a", "b", "c", "held_out"]) for j, b in enumerate(["a", "b", "c", "held_out"])}
    features, landmarks = replay.landmark_features(records, [0, 1], distances)
    assert landmarks == ["a", "b", "c"]
    assert features.shape == (3, 6)
    np.testing.assert_array_equal(features[-1], [2, 3, 4, 3, 4, 5])


def test_nei_distance_formula_and_historical_zero_overlap_clamp():
    assert replay.nei_distance({"p": {"x": 1.0}}, {"p": {"x": 1.0}}) == 0
    assert replay.nei_distance({"p": {"x": 1.0}}, {"p": {"y": 1.0}}) == 10
    a, b = {"p": {"x": 0.5, "y": 0.5}}, {"p": {"x": 1.0}}
    assert replay.nei_distance(a, b) == pytest.approx(-np.log(1 / np.sqrt(2)))
    assert replay.nei_distance(a, b) == replay.nei_distance(b, a)


def test_original_baseline_scopes_run_counts_and_pair_splits():
    models = [f"org/model{i}" for i in range(12)]
    records = [(models[i], models[(i + j + 1) % 12], "related" if (i + j) % 2 else "nan") for j in range(3) for i in range(12)]
    rng = np.random.default_rng(91)
    signatures = {model: rng.normal(size=128).astype(np.float32) for model in models}
    vectors = {method: signatures for method in ("dna_svm", "dna_bge", "dna_mpnet")}
    vectors["phylolm_svm"] = {model: {"probe": {str(i): 0.5, "shared": 0.5}} for i, model in enumerate(models)}
    runs = replay.evaluate(records, vectors, dataset_seed=0, split_seeds=(0, 1))
    assert len(runs) == 11  # Greedy once, five other methods twice.
    greedy = next(r for r in runs if r["method"] == "greedy")
    assert greedy["split_seed"] is None and len(greedy["test_indices"]) == 36
    for run in runs:
        if run["method"] in ("random", "greedy"):
            assert run["train_indices"] == []
            assert len(run["test_indices"]) == 36
        else:
            assert set(run["train_indices"]).isdisjoint(run["test_indices"])
            assert len(run["train_indices"]) == 28
            assert len(run["test_indices"]) == 8
            assert set(run["labels"]) == {0, 1}
            assert run["train_test_shared_models"]
    dna_runs = [r for r in runs if r["method"].startswith("dna") and r["split_seed"] == 0]
    assert all(r["test_indices"] == dna_runs[0]["test_indices"] for r in dna_runs)


def test_aggregation_uses_sample_std_and_marks_absent_rows():
    runs = [{"method": "random", "metrics": dict.fromkeys(replay.METRICS, value)} for value in (0.0, 1.0)]
    summary = replay.aggregate(runs)
    assert summary["random"]["metrics"]["accuracy"]["std"] == pytest.approx(np.sqrt(0.5))
    assert summary["dna_bge"]["status"] == "unavailable"
    assert summary["dna_bge"]["metrics"]["auc"]["mean"] is None


def _write_signature(tmp_path, signature, **metadata_overrides):
    path = tmp_path / "out" / replay.DATASET / "org_model" / "org_model_dna.json"
    path.parent.mkdir(parents=True)
    metadata = dict(model_name="precomputed", extraction_method="text_embeddings", probe_set_id="test",
                    probe_count=600, dna_dimension=128, embedding_dimension=64, reduction_method="random_projection",
                    extraction_time="test", computation_time_seconds=0, model_metadata={},
                    extractor_config={"encoder": "Qwen/Qwen3-Embedding-8B"})
    metadata.update(metadata_overrides)
    path.write_text(json.dumps({"signature": signature, "metadata": metadata}))
    return path


@pytest.mark.parametrize("signature,metadata,match", [
    ([0.0] * 127, {}, "128-dimensional"),
    ([float("nan")] * 128, {}, "finite"),
    ([0.0] * 128, {"probe_count": 100}, "dimension/probe"),
    ([0.0] * 128, {"model_name": "org/wrong"}, "model identity"),
    ([0.0] * 128, {"extractor_config": {"encoder": "wrong"}}, "encoder"),
])
def test_malformed_cache_is_rejected(tmp_path, signature, metadata, match):
    _write_signature(tmp_path, signature, **metadata)
    with pytest.raises(ValueError, match=match):
        replay.load_signatures(tmp_path, {"org/model", "org/wrong"})


def test_missing_encoder_is_reported_without_replacing_dna(tmp_path):
    _write_signature(tmp_path, [0.0] * 128)
    vectors, manifest, inventory = replay.load_signatures(tmp_path, {"org/model"})
    assert vectors["dna_bge"] == {}
    assert inventory["dna_bge"]["missing_relation_models"] == ["org/model"]
    assert len(manifest) == 1 and len(manifest[0]["sha256"]) == 64
