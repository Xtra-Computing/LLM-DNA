"""Offline regression coverage for historical routing protocol and cache guards."""
from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "reproduce_routing", Path(__file__).parents[1] / "scripts/reproduce_routing.py")
routing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(routing)


def frame():
    return pd.DataFrame({"model_name": ["a", "a", "b", "b", "c", "c"],
                         "model_id": [0, 0, 1, 1, 2, 2],
                         "prompt_id": [0, 1, 0, 1, 0, 1], "label": [0, 1, 1, 0, 0, 0]})


def test_csv_deduplicates_max_label_and_rejects_conflicting_prompts(tmp_path):
    data = frame()
    data["prompt"] = ["first", "second"] * 3
    duplicate = data.iloc[[0]].copy()
    duplicate["label"] = 1
    path = tmp_path / "split.csv"
    pd.concat([data, duplicate]).to_csv(path, index=False)
    read = routing.read_split(path)
    assert len(read) == 6
    assert read.iloc[0].label == 1
    duplicate["prompt"] = "conflicting"
    pd.concat([data, duplicate]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Conflicting text"):
        routing.read_split(path)


def test_random_matches_python_sampler_order():
    seed, samples = 11, 1024
    rng = random.Random(seed)
    groups = {0: [0, 1, 0], 1: [1, 0, 0]}
    expected = sum(rng.choice(groups[rng.choice(list(groups))]) for _ in range(samples)) / samples
    actual = routing.random_baseline(frame(), seed, samples)
    assert actual["router_accuracy"] == expected
    assert actual["exact_expectation"] == pytest.approx(1 / 3)


def test_single_best_uses_training_labels():
    train = frame()
    train.loc[train.model_name == "b", "label"] = 1
    test = frame()
    test.loc[test.model_name == "b", "label"] = 0
    result = routing.single_best({"train": train, "val": test, "test": test})
    assert result["model"] == "b"
    assert result["test"]["router_accuracy"] == 0


def test_cached_inference_matches_direct_multiclass_forward(tmp_path):
    torch.manual_seed(0)
    queries, dna = torch.randn(2, 7), torch.randn(3, 128)
    model = routing.Router(queries, dna, "dna")
    assert model.classifier.out_features == 3
    assert not model.Q.weight.requires_grad and not model.dna_table.weight.requires_grad
    models, prompts, labels = routing.frame_tensors(frame(), ["a", "b", "c"])
    with torch.no_grad():
        logits = model(models, prompts, test_mode=True)
        probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()
    expected = []
    for pid in (0, 1):
        candidates = np.flatnonzero(prompts.numpy() == pid)
        best = candidates[np.argmax(probabilities[candidates])]
        expected.append(int(labels[best]))
    path = tmp_path / "predictions.csv"
    result = routing.evaluate(model.state_dict(), frame(), ["a", "b", "c"], path, batch_size=2)
    assert result["router_accuracy"] == np.mean(expected)
    selected = pd.read_csv(path)
    assert selected.prompt_id.tolist() == [0, 1]
    assert np.allclose(selected.probability, [max(probabilities[[0, 2, 4]]), max(probabilities[[1, 3, 5]])])


def test_query_cache_rejects_duplicate_or_missing_ids(tmp_path):
    path = tmp_path / "features.npz"
    np.savez(path, ids=[0, 0], embeddings=np.ones((2, 7)))
    with pytest.raises(ValueError, match="Prompt IDs"):
        routing.load_queries(path, np.array([0, 1]), 7)
    np.savez(path, ids=[0, 1], embeddings=np.ones((2, 7)))
    assert routing.load_queries(path, np.array([0, 1]), 7).shape == (2, 7)


def test_checkpoint_rejects_changed_query_cache():
    queries = torch.randn(2, 7)
    dna = torch.randn(3, 128)
    model = routing.Router(queries, dna, "dna")
    checkpoint = {"model_names": ["a", "b", "c"], "seed": 0, "state_dict": model.state_dict()}
    routing.validate_checkpoint(checkpoint, ["a", "b", "c"], queries, dna.numpy(), "dna", 0)
    with pytest.raises(ValueError, match="query table"):
        routing.validate_checkpoint(checkpoint, ["a", "b", "c"], queries + 1, dna.numpy(), "dna", 0)


def test_legacy_precomputed_metadata_uses_model_directory(tmp_path, monkeypatch):
    path = tmp_path / "out/dataset/org_model/model_dna.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    signature = SimpleNamespace(model_name="precomputed", signature=np.ones(128, dtype=np.float32), dimension=128)
    monkeypatch.setattr(routing.DNASignature, "load", lambda _: signature)
    loaded = routing.load_dna(tmp_path, "dataset", ["org__model"], {})
    assert np.array_equal(loaded["org__model"], signature.signature)
    signature.model_name = "wrong/model"
    with pytest.raises(ValueError, match="identity"):
        routing.load_dna(tmp_path, "dataset", ["org__model"], {})


def test_fresh_fit_changes_head_but_preserves_cached_dna(tmp_path):
    torch.set_num_threads(1)
    queries = torch.randn(2, 7)
    dna = np.random.default_rng(4).normal(size=(3, 128)).astype(np.float32)
    args = Namespace(device="cpu", epochs=2, batch_size=3)
    state, history = routing.train_router({"train": frame()}, ["a", "b", "c"], queries,
                                          dna, "dna", args, 1, tmp_path)
    assert len(history) == 2
    assert all(np.isfinite(item["train_loss"]) for item in history)
    assert np.array_equal(state["dna_table.weight"].numpy(), dna)
    assert torch.equal(state["Q.weight"], queries)
    checkpoint = torch.load(tmp_path / "router.pt", weights_only=True)
    assert "Q.weight" not in checkpoint["state_dict"]
    assert checkpoint["epochs"] == 2


def test_vectorized_loader_preserves_batches_and_rng_state():
    tensors = (torch.arange(17), torch.arange(17) * 3, torch.arange(17) % 2)
    outputs = []
    rng_states = []
    for vectorized in (False, True):
        torch.manual_seed(43)
        dataset = routing.VectorizedPairs(*tensors) if vectorized else torch.utils.data.TensorDataset(*tensors)
        loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True,
                                             collate_fn=routing.identity_collate if vectorized else None)
        batches = list(loader)
        outputs.append(batches)
        rng_states.append(torch.get_rng_state())
    assert torch.equal(rng_states[0], rng_states[1])
    for original, optimized in zip(*outputs):
        assert all(torch.equal(first, second) for first, second in zip(original, optimized))


def test_summary_uses_sample_standard_deviation():
    runs = [{"method": "dna", "seed": seed, "test": {"router_accuracy": value}}
            for seed, value in enumerate([.65, .67, .69])]
    args = Namespace(methods=["dna"], mode="replay")
    summary = routing.summarize(runs, args, {})["aggregate"]["dna"]
    assert summary["std_sample"] == pytest.approx(.02)


@pytest.mark.parametrize("mode", ["train", "replay"])
def test_routing_entrypoint_needs_only_cached_inputs(tmp_path, monkeypatch, mode):
    cache_root, output = tmp_path / "cache", tmp_path / "output"
    labels = cache_root / "data/embedllm"
    labels.mkdir(parents=True)
    names = [f"org__model{i}" for i in range(3)]
    for index, split in enumerate(("train", "val", "test")):
        rows = [{"model_name": name, "model_id": model, "prompt_id": prompt,
                 "prompt": f"Question {prompt}", "label": int((model + prompt) % 2 == 0)}
                for model, name in enumerate(names) for prompt in (2 * index, 2 * index + 1)]
        pd.DataFrame(rows).to_csv(labels / f"{split}.csv", index=False)
    rng = np.random.default_rng(27)
    for method, dimension in (("dna", 4096), ("embedllm", 768)):
        dataset = routing.DATASET + ("_embed" if method == "embedllm" else "")
        vectors = rng.normal(size=(len(names), 128)).astype(np.float32)
        for name, vector in zip(names, vectors):
            safe_name = name.replace("__", "_")
            path = cache_root / "out" / dataset / safe_name / f"{safe_name}_dna.json"
            path.parent.mkdir(parents=True)
            metadata = dict(model_name=name.replace("__", "/"), extraction_method="text_embeddings",
                            probe_set_id="fixture", probe_count=600, dna_dimension=128,
                            embedding_dimension=64, reduction_method="random_projection", extraction_time="fixture",
                            computation_time_seconds=0, model_metadata={}, extractor_config={})
            path.write_text(json.dumps({"signature": vector.tolist(), "metadata": metadata}))
        query_path = cache_root / "cache/route_fixed_model" / routing.QUERY_FILES[method]
        query_path.parent.mkdir(parents=True, exist_ok=True)
        queries = rng.normal(size=(6, dimension)).astype(np.float32)
        np.savez(query_path, ids=np.arange(6), embeddings=queries)
        if mode == "replay":
            torch.manual_seed(0)
            model = routing.Router(torch.from_numpy(queries), torch.from_numpy(vectors), method)
            torch.save({"state_dict": model.state_dict(), "model_names": names, "seed": 0},
                       routing.checkpoint_path(cache_root, method, 0))
    before = {p: routing.sha256(p) for p in cache_root.rglob("*") if p.is_file()}
    assert not any((cache_root / name).exists() for name in ("src", "scripts", ".git", "result"))
    if mode == "train":
        assert not list(cache_root.rglob("*.pt"))
    monkeypatch.setenv("PATH", str(tmp_path / "no-executables"))
    report = routing.main(["--cache-root", str(cache_root), "--output-dir", str(output), "--mode", mode,
                           "--seeds", "0", "--epochs", "2", "--batch-size", "3", "--threads", "1"])

    assert set(report["aggregate"]) == set(routing.PAPER)
    assert len(report["runs"]) == 4
    for run in report["runs"]:
        assert 0 <= run["test"]["router_accuracy"] <= 1
        if run["method"] in ("dna", "embedllm"):
            assert len(run["history"]) == (2 if mode == "train" else 0)
            for split in ("val", "test"):
                predictions = pd.read_csv(output / run["method"] / "seed_0" / f"{split}_predictions.csv")
                assert len(predictions) == 2
    assert before == {p: routing.sha256(p) for p in cache_root.rglob("*") if p.is_file()}
    manifest = json.loads((output / "manifest.json").read_text())
    assert {Path(path).parts[0] for path in manifest} == {"data", "out", "cache"}
    assert (output / "summary.csv").is_file()
