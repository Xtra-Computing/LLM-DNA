"""Table generation from relation labels and cached model signatures."""

import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "reproduce_relationship_table", Path(__file__).resolve().parents[1] / "scripts/reproduce_relationship_table.py"
)
table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(table)


@pytest.fixture
def relations_csv(tmp_path):
    models = [f"org/model{i}" for i in range(8)]
    pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1:]]
    records = [(a, b, source) for a, b in pairs[:12] for source in ("cardData", "tags")]
    records += records[:3]  # Exact triples must be deduplicated, while sources remain.
    path = tmp_path / "relations.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_id", "related_model", "relation"])
        writer.writerows(records)
    return path


@pytest.fixture
def cache_root(tmp_path):
    root = tmp_path / "cache"
    models = [f"org/model{i}" for i in range(8)]
    rng = np.random.default_rng(29)
    encoders = {"": "Qwen/Qwen3-Embedding-8B", "bge": "BAAI/bge-large-en-v1.5",
                "mpnet": "sentence-transformers/all-mpnet-base-v2"}
    for model in models:
        name = model.replace("/", "_")
        for folder, encoder in encoders.items():
            path = root / "out" / table.replay.DATASET / name / folder / f"{name}_dna.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            metadata = dict(model_name=model, extraction_method="text_embeddings", probe_set_id="fixture",
                            probe_count=600, dna_dimension=128, embedding_dimension=64, reduction_method="random_projection",
                            extraction_time="fixture", computation_time_seconds=0, model_metadata={},
                            extractor_config={"encoder": encoder})
            path.write_text(json.dumps({"signature": rng.normal(size=128).astype(np.float32).tolist(), "metadata": metadata}))
        phylo = root / "out_phyloLM" / table.replay.DATASET / name / f"{name}_dna.json"
        phylo.parent.mkdir(parents=True)
        phylo.write_text(json.dumps({"signature": {"probe": {"shared": 0.5, model: 0.5}},
                                     "metadata": {"model_name": model, "probe_count": 600}}))
    return root


def test_relation_csv_and_signatures_refit_phylolm_and_three_dna_methods(cache_root, relations_csv, tmp_path, monkeypatch):
    output = tmp_path / "table"
    before = {p: table.replay.sha256(p) for p in cache_root.rglob("*") if p.is_file()}
    original_read = Path.read_text
    original_distances = table.dna_distances
    original_evaluate = table.evaluate
    fitted_runs = []
    assert not (cache_root / "data").exists()

    def reject_historical_outputs(path, *args, **kwargs):
        assert path.name not in {"results.json", "runs.json", "input_manifest.json"}
        return original_read(path, *args, **kwargs)

    def euclidean_only(signatures, models, metric="euclidean"):
        assert metric == "euclidean"
        return original_distances(signatures, models, metric)

    def capture_runs(cohorts, vectors):
        runs = original_evaluate(cohorts, vectors)
        fitted_runs.extend(runs)
        return runs

    with monkeypatch.context() as context:
        context.setattr(Path, "read_text", reject_historical_outputs)
        context.setattr(table, "dna_distances", euclidean_only)
        context.setattr(table, "evaluate", capture_runs)
        result = table.run(cache_root, output, relations=relations_csv)
    assert result["fits"] == 250
    assert result["representations"] == ["landmark_nei", "difference", "concatenation", "landmark_euclidean"]
    assert result["method_representations"]["phylolm_svm"] == ["landmark_nei"]
    assert result["protocol"]["phylolm"]["refitted"] is True
    assert result["protocol"]["historical_results_required"] is False
    assert result["protocol"]["std_ddof"] == 1
    assert result["protocol"]["relations"]["sha256"] == table.replay.sha256(relations_csv)
    assert result["protocol"]["relations"]["positive_records"] == 27
    assert before == {p: table.replay.sha256(p) for p in cache_root.rglob("*") if p.is_file()}
    with (output / "table.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10 and all(row["runs"] == "25" for row in rows)
    assert len([row for row in rows if row["encoder"] == "PhyloLM"]) == 1
    assert rows[0]["encoder"] == "PhyloLM" and rows[0]["representation"] == "landmark_nei"
    assert len([line for line in (output / "table.md").read_text().splitlines() if line.startswith("|")]) == 12
    assert len([line for line in (output / "table.tex").read_text().splitlines() if line.endswith(r"\\")]) == 11
    for row in rows:
        assert 0 <= float(row["accuracy_mean"]) <= 1
        assert float(row["accuracy_std"]) >= 0
    assert {path.name for path in output.iterdir()} == {"table.md", "table.csv", "table.tex", "results.json"}
    runs = fitted_runs
    assert len({(r["method"], r["representation"], r["dataset_seed"], r["split_seed"]) for r in runs}) == 250
    assert len([r for r in runs if r["method"] == "phylolm_svm"]) == 25
    cohorts, vectors, _, _ = table.load_inputs(cache_root, relations_csv)
    records = cohorts[0]["records"]
    assert len(records) == 51
    assert ("org/model0", "org/model1", "cardData") in records
    assert ("org/model0", "org/model1", "tags") in records
    original = table.replay.evaluate(records, vectors, dataset_seed=0, split_seeds=(0,))
    original = {r["method"]: r for r in original if r["method"] in (*table.ENCODERS, "phylolm_svm")}
    for run in runs:
        if run["dataset_seed"] == run["split_seed"] == 0:
            for field in ("train_indices", "test_indices", "labels"):
                assert run[field] == original[run["method"]][field]
            if run["representation"] in ("difference", "landmark_nei"):
                for field in ("predictions", "probabilities", "metrics"):
                    assert run[field] == original[run["method"]][field]
        if run["representation"].startswith("landmark_"):
            dataset_records = cohorts[run["dataset_seed"]]["records"]
            expected = sorted({m for i in run["train_indices"] for m in dataset_records[i][:2]})
            assert run["landmark_models"] == expected
            assert run["feature_dimension"] == 2 * len(expected)


def test_missing_encoder_stops_before_any_fitting(cache_root, relations_csv, tmp_path, monkeypatch):
    next((cache_root / "out").glob("*/*/bge/*_dna.json")).unlink()

    def unexpected_fit(*args):
        pytest.fail("All encoder inputs must be checked before fitting")

    monkeypatch.setattr(table, "evaluate", unexpected_fit)
    with pytest.raises(ValueError, match="Missing BGE DNA"):
        table.run(cache_root, tmp_path / "output", relations=relations_csv)
    assert not (tmp_path / "output").exists()


def test_output_guards_preserve_cache_and_existing_results(cache_root, tmp_path):
    with pytest.raises(ValueError, match="outside the read-only cache"):
        table.run(cache_root, cache_root / "output")
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "table.md"
    sentinel.write_text("Existing table\n")
    with pytest.raises(ValueError, match="new or empty"):
        table.run(cache_root, output)
    assert sentinel.read_text() == "Existing table\n"


def test_original_runner_uses_relations_without_enriched_csvs(cache_root, relations_csv, tmp_path, monkeypatch):
    output = tmp_path / "original"
    before = {p: table.replay.sha256(p) for p in cache_root.rglob("*") if p.is_file()}
    assert not any((cache_root / name).exists() for name in ("src", "scripts", ".git", "data"))
    monkeypatch.setenv("PATH", str(tmp_path / "no-executables"))

    table.replay.main(["--cache-root", str(cache_root), "--relations", str(relations_csv), "--output-dir", str(output)])

    result = json.loads((output / "results.json").read_text())
    runs = json.loads((output / "runs.json").read_text())
    assert len(runs) == 130
    assert all(row["status"] == "complete" for row in result["summary"].values())
    assert set(result["summary"]) == set(table.replay.METHODS)
    assert before == {p: table.replay.sha256(p) for p in cache_root.rglob("*") if p.is_file()}
    manifest = json.loads((output / "input_manifest.json").read_text())
    assert {Path(row["path"]).parts[0] for row in manifest} == {"relations.csv", "out", "out_phyloLM"}


def test_unified_entrypoint_runs_from_relations_without_git(cache_root, relations_csv, tmp_path):
    output = tmp_path / "unified"
    environment = dict(os.environ, PATH=str(tmp_path / "no-executables"), PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, str(table.ROOT / "scripts/reproduce_paper.py"),
                    "--cache-root", str(cache_root), "--relations", relations_csv.name,
                    "--output-dir", str(output), "--experiments", "relationship"],
                   cwd=tmp_path, env=environment, capture_output=True, text=True, check=True, timeout=60)
    result = json.loads((output / "relationship/results.json").read_text())
    assert result["fits"] == 250
    assert result["protocol"]["relations"]["sha256"] == table.replay.sha256(relations_csv)
    assert len(table.table_rows(result["summary"])) == 10
    assert json.loads((output / "run_manifest.json").read_text())[0]["returncode"] == 0


@pytest.mark.parametrize("cache_setting", [None, ""])
def test_table_requires_an_explicit_cache_location(cache_setting, tmp_path, monkeypatch, capsys):
    if cache_setting is None:
        monkeypatch.delenv("REPRO_CACHE_ROOT", raising=False)
    else:
        monkeypatch.setenv("REPRO_CACHE_ROOT", cache_setting)
    output = tmp_path / "table"
    with pytest.raises(SystemExit) as error:
        table.main(["--output-dir", str(output)])
    assert error.value.code == 2
    assert "--cache-root or REPRO_CACHE_ROOT is required" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("cache_option", ["argument", "environment"])
def test_launcher_uses_explicit_paths_from_an_unrelated_directory(cache_root, relations_csv, tmp_path, cache_option):
    shell = shutil.which("bash")
    dirname = shutil.which("dirname")
    if shell is None or dirname is None:
        pytest.skip("The shell launcher requires bash and dirname")
    commands = tmp_path / "commands"
    commands.mkdir()
    (commands / "dirname").symlink_to(dirname)
    working_directory = tmp_path / "unrelated"
    working_directory.mkdir()
    output = tmp_path / "table"
    environment = dict(os.environ, PATH=str(commands), REPRO_PYTHON=sys.executable,
                       REPRO_CACHE_ROOT=str(cache_root if cache_option == "environment" else tmp_path / "missing"))
    command = [shell, str(table.ROOT / "scripts/reproduce_relationship_table.sh"),
               "--relations", str(relations_csv), "--output-dir", str(output)]
    if cache_option == "argument":
        command += ["--cache-root", str(cache_root)]
    subprocess.run(command, cwd=working_directory, env=environment, capture_output=True,
                   text=True, check=True, timeout=60)
    result = json.loads((output / "results.json").read_text())
    assert result["cache_root"] == str(cache_root)
    assert result["fits"] == 250
    assert result["protocol"]["relations"]["sha256"] == table.replay.sha256(relations_csv)
    assert len(table.table_rows(result["summary"])) == 10


def test_concatenation_preserves_raw_vectors_and_endpoint_order():
    signatures = {"a": np.asarray([1, 4], dtype=np.float32), "b": np.asarray([3, 1], dtype=np.float32)}
    features, landmarks = table.pair_features([("a", "b", "related"), ("b", "a", "nan")], signatures, [0], "concatenation")
    np.testing.assert_array_equal(features, [[1, 4, 3, 1], [3, 1, 1, 4]])
    assert features.dtype == np.float32
    assert landmarks is None


def test_euclidean_landmarks_match_hand_computed_geometry():
    signatures = {"a": [1, 0], "b": [0, 1], "c": [1, 1], "held_out": [2, 1]}
    signatures = {k: np.asarray(v, dtype=np.float32) for k, v in signatures.items()}
    records = [("a", "b", "related"), ("b", "c", "nan"), ("c", "held_out", "related")]
    distances = table.dna_distances(signatures, sorted(signatures))
    features, landmarks = table.pair_features(records, signatures, [0, 1], "landmark_euclidean", distances)
    assert landmarks == ["a", "b", "c"]
    assert features.shape == (3, 6)
    assert features.dtype == np.float64
    np.testing.assert_allclose(features[2], [1, 1, 0, np.sqrt(2), 2, 1], atol=1e-15)


def test_swapping_endpoints_swaps_landmark_blocks():
    signatures = {"a": np.asarray([1., 2.]), "b": np.asarray([3., -1.])}
    records = [("a", "b", "related"), ("b", "a", "nan")]
    distances = table.dna_distances(signatures, sorted(signatures))
    features, _ = table.pair_features(records, signatures, [0], "landmark_euclidean", distances)
    np.testing.assert_array_equal(features[0, :2], features[1, 2:])
    np.testing.assert_array_equal(features[0, 2:], features[1, :2])


@pytest.mark.parametrize("representation", table.REPRESENTATIONS)
def test_training_features_and_scaler_exclude_test_only_vectors(monkeypatch, representation):
    models = [f"org/model{i}" for i in range(8)]
    pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1:]]
    records = [(a, b, source) for a, b in pairs[:12] for source in ("cardData", "tags", "merge")]
    records += [(a, b, "nan") for a, b in pairs[12:]]
    records = sorted(records)
    train, test = table.split_records(records, 3)
    _, b, label = records[test[0]]
    records[test[0]] = ("held_out", b, label)
    rng = np.random.default_rng(9)
    signatures = {m: rng.normal(size=128).astype(np.float32) for m in [*models, "held_out"]}
    partition = {"method": "dna_svm", "dataset_seed": 0, "split_seed": 3,
                 "train_indices": train.tolist(), "test_indices": test.tolist(),
                 "labels": table.binary_labels(records)[test].tolist()}
    fitted_inputs = []

    class RecordingScaler(table.replay.StandardScaler):
        def fit(self, X, y=None, sample_weight=None):
            fitted_inputs.append(X.copy())
            return super().fit(X, y, sample_weight=sample_weight)

    monkeypatch.setattr(table.replay, "StandardScaler", RecordingScaler)
    distances = table.dna_distances(signatures, sorted(signatures))
    expected, landmarks = table.pair_features(records, signatures, train, representation, distances)
    first = table.fit_one(records, signatures, partition, representation, distances)
    signatures["held_out"] = rng.normal(size=128).astype(np.float32) * 1000
    distances = table.dna_distances(signatures, sorted(signatures))
    second = table.fit_one(records, signatures, partition, representation, distances)
    assert first["train_indices"] == second["train_indices"] == train.tolist()
    assert first["test_indices"] == second["test_indices"] == test.tolist()
    assert first["landmark_models"] == second["landmark_models"] == landmarks
    if representation == "landmark_euclidean":
        assert landmarks == sorted({m for i in train for m in records[i][:2]})
        assert "held_out" not in landmarks
    assert len(fitted_inputs) == 2
    np.testing.assert_array_equal(fitted_inputs[0], expected[train])
    np.testing.assert_array_equal(fitted_inputs[0], fitted_inputs[1])
