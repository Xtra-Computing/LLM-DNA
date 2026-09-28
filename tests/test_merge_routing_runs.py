"""Integrity checks for combining independently executed routing seeds."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("merge_routing_runs", SCRIPTS / "merge_routing_runs.py")
merger = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(SCRIPTS))
try:
    spec.loader.exec_module(merger)
finally:
    sys.path.remove(str(SCRIPTS))


def make_source(tmp_path, name, seeds, *, methods=None):
    directory = tmp_path / name
    directory.mkdir()
    protocol = {
        "mode": "train",
        "seeds": list(seeds),
        "device": "cuda:0",
        "training": {"epochs": 50, "batch_size": 2048, "learning_rate": 1e-4},
        "std_ddof": 1,
        "cohorts": {"dna": {"model_names": ["org__a", "org__b"]}},
        "public_source_sha256": {"scripts/reproduce_routing.py": "same-code-hash"},
        "environment": {"python": "3.12", "torch": "2.8", "numpy": "2.2", "pandas": "2.3"},
    }
    methods = list(merger.PAPER) if methods is None else methods
    runs = [{"method": method, "seed": seed, "mode": "train",
             "test": {"router_accuracy": 0.1 + 0.2 * seed}}
            for method in methods for seed in seeds]
    (directory / "results.json").write_text(json.dumps({"mode": "train", "protocol": protocol, "runs": runs}))
    manifest = {"data/embedllm/test.csv": {"sha256": "same-labels-hash", "role": "routing labeled split"}}
    for seed in seeds:
        manifest[f"result/seed_{seed}/metrics.json"] = {"sha256": f"archive-{seed}", "role": "comparison only"}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def change_report(directory, change):
    path = directory / "results.json"
    report = json.loads(path.read_text())
    change(report)
    path.write_text(json.dumps(report))


def test_merge_three_seeds_uses_sample_std_and_retains_provenance(tmp_path):
    first = make_source(tmp_path, "gpu0", [2, 0])
    second = make_source(tmp_path, "gpu1", [1])
    change_report(second, lambda report: report["protocol"].update(device="cuda:1"))
    output = tmp_path / "merged"

    result = merger.merge([first, second], output, [0, 1, 2])

    assert len(result["runs"]) == 12
    for row in result["aggregate"].values():
        assert row["n"] == 3
        assert row["seeds"] == [0, 1, 2]
        assert row["mean"] == pytest.approx(0.3)
        assert row["std_sample"] == pytest.approx(0.2)
    assert result["protocol"]["seeds"] == [0, 1, 2]
    assert [p["device"] for p in result["protocol"]["execution_protocols"]] == ["cuda:0", "cuda:1"]
    assert all("source_run_dir" in row for row in result["runs"])
    assert len(json.loads((output / "manifest.json").read_text())) == 4
    saved = json.loads((output / "results.json").read_text())
    assert saved == result
    assert (output / "summary.csv").is_file()


def test_duplicate_method_seed_is_rejected_before_writing(tmp_path):
    first = make_source(tmp_path, "a", [0])
    second = make_source(tmp_path, "b", [0])
    output = tmp_path / "merged"
    with pytest.raises(ValueError, match="Duplicate run"):
        merger.merge([first, second], output, [0])
    assert not output.exists()


@pytest.mark.parametrize("actual,expected", [([0, 1], [0, 1, 2]), ([0, 1, 2], [0, 1])])
def test_missing_or_unexpected_seeds_are_rejected(tmp_path, actual, expected):
    source = make_source(tmp_path, "source", actual)
    with pytest.raises(ValueError, match="Incomplete or unexpected"):
        merger.merge([source], tmp_path / "merged", expected)


def test_missing_seed_for_only_one_method_is_rejected(tmp_path):
    source = make_source(tmp_path, "source", [0, 1, 2])
    change_report(source, lambda report: report.update(runs=[
        row for row in report["runs"] if (row["method"], row["seed"]) != ("dna", 1)
    ]))
    with pytest.raises(ValueError, match="Incomplete or unexpected"):
        merger.merge([source], tmp_path / "merged", [0, 1, 2])


def test_entire_missing_method_is_rejected_even_with_all_five_seeds(tmp_path):
    source = make_source(tmp_path, "source", [0, 1, 2, 3, 4], methods=["dna"])
    with pytest.raises(ValueError, match="Incomplete or unexpected"):
        merger.merge([source], tmp_path / "merged", [0, 1, 2, 3, 4])


def test_explicit_single_method_merge_is_allowed(tmp_path):
    source = make_source(tmp_path, "source", [0, 1, 2], methods=["dna"])
    result = merger.merge([source], tmp_path / "merged", [0, 1, 2], expected_methods=["dna"])
    assert list(result["aggregate"]) == ["dna"]
    assert result["aggregate"]["dna"]["n"] == 3


@pytest.mark.parametrize("methods", [[], ["dna", "dna"], ["unknown"]])
def test_expected_methods_must_be_nonempty_unique_and_known(tmp_path, methods):
    source = make_source(tmp_path, "source", [0])
    with pytest.raises(ValueError, match="Expected methods"):
        merger.merge([source], tmp_path / "merged", [0], expected_methods=methods)


def test_merge_cannot_overwrite_input_results(tmp_path):
    source = make_source(tmp_path, "source", [0])
    before = (source / "results.json").read_bytes()
    with pytest.raises(ValueError, match="must not overwrite an input"):
        merger.merge([source], source, [0])
    assert (source / "results.json").read_bytes() == before


@pytest.mark.parametrize("expected", [[], [0, 0]])
def test_expected_seed_list_must_be_nonempty_and_unique(tmp_path, expected):
    source = make_source(tmp_path, "source", [0])
    with pytest.raises(ValueError, match="unique expected seeds"):
        merger.merge([source], tmp_path / "merged", expected)


@pytest.mark.parametrize("key,value", [
    ("mode", "replay"),
    ("training", {"epochs": 1, "batch_size": 2048, "learning_rate": 1e-4}),
    ("std_ddof", 0),
    ("cohorts", {"dna": {"model_names": ["org__b", "org__a"]}}),
    ("public_source_sha256", {"scripts/reproduce_routing.py": "different-code"}),
    ("environment", {"python": "3.12", "torch": "2.7", "numpy": "2.2", "pandas": "2.3"}),
])
def test_incompatible_protocols_are_rejected(tmp_path, key, value):
    first = make_source(tmp_path, "a", [0])
    second = make_source(tmp_path, "b", [1])
    change_report(second, lambda report: report["protocol"].update({key: value}))
    output = tmp_path / "merged"
    with pytest.raises(ValueError, match="Incompatible protocol or source code"):
        merger.merge([first, second], output, [0, 1])
    assert not output.exists()


def test_changed_label_input_hash_is_rejected(tmp_path):
    first = make_source(tmp_path, "a", [0])
    second = make_source(tmp_path, "b", [1])
    path = second / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["data/embedllm/test.csv"]["sha256"] = "different-labels"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Input changed between runs"):
        merger.merge([first, second], tmp_path / "merged", [0, 1])
