#!/usr/bin/env python3
"""Offline Table 2 reproduction: cached DNA/features -> router training or replay.

Inputs are cached signatures, labels, query embeddings, and replay checkpoints.
The historical DNA router has 102 output logits for binary labels; preserving
that quirk is necessary for a faithful reproduction of the archived experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from torch import nn

from llm_dna.dna.DNASignature import DNASignature

PAPER_URL = "https://arxiv.org/html/2509.24496v3#S5.SS3"
DATASET = "squad_cqa_hs_wg_arc_mmlu"
QUERY_FILES = {
    "dna": "enc-Qwen__Qwen3-Embedding-8B_dim4096_poolcls_len1024.npz",
    "embedllm": "enc-sentence-transformers__all-mpnet-base-v2_bs32_prompts35673_squad_cqa_hs_wg_arc_mmlu_embed.npz",
}
PAPER = {"random": (0.402, 0.016), "single-best": (0.607, 0.000),
         "embedllm": (0.665, 0.003), "dna": (0.672, 0.008)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def add_input(manifest: dict, path: Path, cache_root: Path, role: str) -> None:
    relative = str(path.relative_to(cache_root))
    if relative not in manifest:
        manifest[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size, "role": role}


def read_split(path: Path, prompt_text: dict[int, str] | None = None) -> pd.DataFrame:
    """Preserve max-label deduplication and stable model-first order.

    Prompt consistency is checked before dropping repeated text to bound memory.
    This makes grouping on (model_name, model_id, prompt_id) equivalent to the
    original grouping that also contains prompt text.
    """
    prompt_text = {} if prompt_text is None else prompt_text
    parts = []
    for chunk in pd.read_csv(path, usecols=["model_name", "model_id", "prompt_id", "prompt", "label"],
                             chunksize=200_000):
        for pid, text in chunk[["prompt_id", "prompt"]].drop_duplicates().itertuples(index=False, name=None):
            text = str(text)
            pid = int(pid)
            if pid in prompt_text and prompt_text[pid] != text:
                raise ValueError(f"Conflicting text for prompt_id={pid} in {path}")
            prompt_text[pid] = text
        compact = chunk.drop(columns="prompt")
        for key in ("model_id", "prompt_id", "label"):
            compact[key] = compact[key].astype(np.int64)
        if not compact.label.isin([0, 1]).all():
            raise ValueError(f"Labels must be binary in {path}")
        parts.append(compact)
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.groupby(["model_name", "model_id", "prompt_id"], as_index=False, sort=True)["label"].max()
    if frame.duplicated(["model_name", "prompt_id"]).any():
        raise ValueError(f"Multiple model_ids for the same model/prompt in {path}")
    return frame


def prepare_splits(cache_root: Path, output: Path, manifest: dict) -> dict[str, pd.DataFrame]:
    # The compact cache is local to this output directory, with raw-source hashes
    # checked on every invocation. It never trusts a cache copied from elsewhere.
    paths = {split: cache_root / "data/embedllm" / f"{split}.csv" for split in ("train", "val", "test")}
    for path in paths.values():
        add_input(manifest, path, cache_root, "routing labeled split")
    fingerprint = {split: manifest[str(path.relative_to(cache_root))]["sha256"] for split, path in paths.items()}
    cachedir = output / "_inputs"
    cache_meta = cachedir / "sources.json"
    valid = cache_meta.exists() and json.loads(cache_meta.read_text()) == fingerprint
    frames = {}
    text_map: dict[int, str] = {}
    for split, path in paths.items():
        cache = cachedir / f"{split}.npz"
        if valid and cache.exists():
            with np.load(cache, allow_pickle=False) as values:
                frames[split] = pd.DataFrame({key: values[key] for key in values.files})
        else:
            print(f"Reading and validating {split}.csv", flush=True)
            frames[split] = read_split(path, text_map)
            cachedir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **{key: frames[split][key].to_numpy(dtype=str if key == "model_name" else np.int64)
                                         for key in frames[split].columns})
    write_json(cache_meta, fingerprint)
    prompt_sets = {key: set(frame.prompt_id) for key, frame in frames.items()}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        if prompt_sets[first] & prompt_sets[second]:
            raise ValueError(f"Prompt leakage between {first} and {second}")
    return frames


def load_dna(cache_root: Path, dataset: str, names: list[str], manifest: dict) -> dict[str, np.ndarray]:
    vectors = {}
    for name in names:
        directory = cache_root / "out" / dataset / name.replace("__", "_").replace(":", "_")
        files = sorted(directory.glob("*_dna.json"))
        if len(files) > 1:
            raise ValueError(f"Ambiguous DNA cache, choose explicitly: {directory}")
        if not files:
            continue
        path = files[0]
        signature = DNASignature.load(path)
        if signature.model_name != "precomputed" and signature.model_name.replace("/", "__").replace(":", "__") != name:
            raise ValueError(f"DNA metadata model identity disagrees with its directory: {path}")
        vec = signature.signature
        if signature.dimension != 128 or not np.isfinite(vec).all():
            raise ValueError(f"Invalid 128-dimensional DNA: {path}")
        add_input(manifest, path, cache_root, f"{dataset} DNA cache via main DNASignature.load")
        if np.linalg.norm(vec) > 0:
            vectors[name] = vec
    if not vectors:
        raise ValueError(f"No valid cached DNA found for {dataset}")
    return vectors


def restrict_frames(frames: dict, names: list[str]) -> dict:
    result = {key: frame[frame.model_name.isin(names)].copy().reset_index(drop=True)
              for key, frame in frames.items()}
    if any(frame.empty for frame in result.values()):
        raise ValueError("A routing split is empty after DNA filtering")
    return result


def load_queries(path: Path, prompt_ids: np.ndarray, dimension: int) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as data:
        ids = data["ids"].astype(np.int64)
        values = data["embeddings"].astype(np.float32)
    if len(np.unique(ids)) != len(ids) or set(ids) != set(prompt_ids):
        raise ValueError(f"Prompt IDs do not match cache: {path}")
    if values.shape != (len(ids), dimension) or not np.isfinite(values).all():
        raise ValueError(f"Invalid query embeddings: {path}")
    queries = torch.zeros((int(ids.max()) + 1, dimension), dtype=torch.float32)
    queries[torch.from_numpy(ids)] = torch.from_numpy(values)
    return queries


class Router(nn.Module):
    """The original MF architecture, including its DNA multiclass classifier."""
    def __init__(self, questions: torch.Tensor, vectors: torch.Tensor, method: str, alpha: float = .05):
        super().__init__()
        self.method = method
        self.alpha = alpha
        # Initialization order preserves the original torch RNG consumption.
        if method == "embedllm":
            self.P = nn.Embedding(vectors.shape[0], 232)
        self.Q = nn.Embedding(questions.shape[0], questions.shape[1])
        self.Q.weight.data.copy_(questions)
        self.Q.weight.requires_grad_(False)
        latent = 232 if method == "embedllm" else vectors.shape[1]
        self.text_proj = nn.Linear(questions.shape[1], latent)
        if method != "embedllm":
            self.dna_table = nn.Embedding.from_pretrained(vectors, freeze=True)
        self.classifier = nn.Linear(latent, 2 if method == "embedllm" else vectors.shape[0])

    def forward(self, models, prompts, test_mode=False):
        vectors = self.P(models) if self.method == "embedllm" else self.dna_table(models)
        query = self.Q(prompts)
        if not test_mode:
            query = query + torch.randn_like(query) * self.alpha
        return self.classifier(vectors * self.text_proj(query))


def frame_tensors(frame: pd.DataFrame, names: list[str]):
    indices = {name: index for index, name in enumerate(names)}
    return (torch.tensor(frame.model_name.map(indices).to_numpy(), dtype=torch.long),
            torch.tensor(frame.prompt_id.to_numpy(), dtype=torch.long),
            torch.tensor(frame.label.to_numpy(), dtype=torch.long))


class VectorizedPairs(torch.utils.data.Dataset):
    """Batch indexing without changing DataLoader sampler or torch RNG draws."""
    def __init__(self, *tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __getitems__(self, indices):
        return tuple(tensor[indices] for tensor in self.tensors)


def identity_collate(batch):
    return batch


def train_router(frames, names, queries, vectors, method, args, seed, destination):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    matrix = (np.random.default_rng(seed).standard_normal(vectors.shape, dtype=np.float32)
              if method == "random-dna" else vectors)
    model = Router(queries, torch.from_numpy(matrix), method).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    dataset = VectorizedPairs(*frame_tensors(frames["train"], names))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=identity_collate)
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        total = 0.0
        for batch in loader:
            models, prompts, labels = (item.to(args.device) for item in batch)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(models, prompts), labels)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(labels)
        entry = {"epoch": epoch, "train_loss": total / len(dataset), "seconds": time.perf_counter() - start}
        history.append(entry)
        write_json(destination / "history.json", history)
        print(f"{method} seed={seed} epoch={epoch}/{args.epochs} loss={entry['train_loss']:.6f} seconds={entry['seconds']:.1f}", flush=True)
    # Store small learned weights; frozen input tables are reproducibly reloaded
    # from hashed caches. Saved state is never read automatically by train mode.
    state = {key: value.detach().cpu() for key, value in model.state_dict().items() if key != "Q.weight"}
    torch.save({"state_dict": state, "model_names": names, "seed": seed, "epochs": args.epochs,
                "method": method, "query_table_external": True}, destination / "router.pt")
    state["Q.weight"] = queries
    return state, history


def evaluate(state: dict, frame: pd.DataFrame, names: list[str], destination: Path, batch_size: int = 2048) -> dict:
    """Evaluate saved scores, projecting each distinct query just once.

    This deterministic inference optimization preserves the model equation and
    tie-breaking order; it does not bypass fitting when running --mode train.
    """
    indices = {name: i for i, name in enumerate(names)}
    prompt_ids = np.sort(frame.prompt_id.unique())
    queries = state["Q.weight"][torch.tensor(prompt_ids, dtype=torch.long)]
    with torch.no_grad():
        projected = nn.functional.linear(queries, state["text_proj.weight"], state["text_proj.bias"])
    projected_lookup = {int(pid): i for i, pid in enumerate(prompt_ids)}
    model_vectors = state.get("dna_table.weight", state.get("P.weight"))
    if model_vectors is None:
        raise ValueError("Checkpoint is missing model embeddings")
    model_ids = frame.model_name.map(indices).to_numpy(dtype=np.int64, copy=True)
    pids = frame.prompt_id.to_numpy(dtype=np.int64, copy=True)
    projected_ids = np.array([projected_lookup[int(pid)] for pid in pids])
    labels = frame.label.to_numpy(dtype=np.int64, copy=True)
    probabilities = np.empty(len(frame), dtype=np.float32)
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            stop = start + batch_size
            features = model_vectors[torch.from_numpy(model_ids[start:stop])] * projected[torch.from_numpy(projected_ids[start:stop])]
            logits = nn.functional.linear(features, state["classifier.weight"], state["classifier.bias"])
            target = torch.from_numpy(labels[start:stop])
            probabilities[start:stop] = torch.softmax(logits, dim=1)[:, 1].numpy()
            correct += int((logits.argmax(dim=1) == target).sum())
            loss_sum += float(nn.functional.cross_entropy(logits, target, reduction="sum"))
    # pandas idxmax retains the first model in the original source order on ties.
    scored = pd.DataFrame({"prompt_id": pids, "model_index": model_ids, "label": labels,
                           "probability": probabilities})
    chosen = scored.loc[scored.groupby("prompt_id", sort=False).probability.idxmax()].copy()
    chosen["model_name"] = [names[index] for index in chosen.model_index]
    chosen = chosen.sort_values("prompt_id")
    chosen.to_csv(destination, index=False)
    return {"router_accuracy": float(chosen.label.mean()), "correct_routes": int(chosen.label.sum()),
            "prompts": len(chosen), "pair_accuracy": correct / len(frame),
            "pair_loss": loss_sum / len(frame), "pairs": len(frame)}


def random_baseline(frame: pd.DataFrame, seed: int, samples: int = 1024) -> dict:
    groups = {}
    for pid, label in zip(frame.prompt_id, frame.label):
        groups.setdefault(int(pid), []).append(int(label))
    rng = random.Random(seed)
    prompt_ids = list(groups)
    correct = sum(rng.choice(groups[rng.choice(prompt_ids)]) for _ in range(samples))
    return {"router_accuracy": correct / samples, "samples": samples,
            "exact_expectation": float(np.mean([np.mean(labels) for labels in groups.values()]))}


def single_best(frames: dict) -> dict:
    means = frames["train"].groupby("model_name").label.mean()
    model = str(means.idxmax())
    return {"model": model, **{split: {"router_accuracy": float(frame[frame.model_name == model].label.mean())}
                               for split, frame in frames.items()}}


def checkpoint_path(cache_root: Path, method: str, seed: int) -> Path:
    cache = cache_root / "cache/route_fixed_model"
    if method == "embedllm":
        return cache / f"mf_model_seed{seed}.pt"
    source = "random" if method == "random-dna" else "dna"
    return cache / f"dna_router_{DATASET}_enc-Qwen__Qwen3-Embedding-8B_qd4096_mf232_{source}_seed{seed}.pt"


def validate_checkpoint(checkpoint, names, queries, vectors, method, seed):
    if checkpoint["model_names"] != names or checkpoint.get("seed") != seed:
        raise ValueError("Checkpoint model order or seed differs from the cached DNA cohort")
    state = checkpoint["state_dict"]
    if not torch.equal(state["Q.weight"], queries):
        raise ValueError("Checkpoint frozen query table differs from the independent query cache")
    if method == "dna" and not np.array_equal(state["dna_table.weight"].numpy(), vectors):
        raise ValueError("Checkpoint frozen DNA table differs from main-loaded DNA signatures")
    if method == "random-dna":
        expected = np.random.default_rng(seed).standard_normal(vectors.shape, dtype=np.float32)
        if not np.array_equal(state["dna_table.weight"].numpy(), expected):
            raise ValueError("Checkpoint frozen random DNA table differs from seeded generation")
    expected_classes = 2 if method == "embedllm" else len(names)
    if state["classifier.weight"].shape[0] != expected_classes:
        raise ValueError("Checkpoint classifier differs from the historical source architecture")


def saved_reference(cache_root: Path, method: str, seed: int, manifest: dict):
    if method in ("dna", "random-dna"):
        parent = cache_root / "result/dna_routing/route_fixed_model"
        if method == "random-dna":
            parent /= "random"
    else:
        parent = cache_root / "result/embedllm_routing/route_fixed_model"
    path = parent / f"seed_{seed}/metrics.json"
    if not path.exists():
        return None
    add_input(manifest, path, cache_root, "archived scores, comparison only (not used to compute predictions)")
    block = json.loads(path.read_text())["results"]
    if method == "random":
        return block["random"]["test_accuracy"]
    if method == "single-best":
        return block["single_best"]["test_accuracy"]
    return block["mf"]["test"]["router_accuracy"]


def summarize(runs: list[dict], args, protocol: dict) -> dict:
    summary = {}
    for method in args.methods:
        selected = [run for run in runs if run["method"] == method]
        if not selected:
            continue
        values = [run["test"]["router_accuracy"] for run in selected]
        mean = float(np.mean(values))
        # The paper's printed ± values match sample SD (ddof=1), not population SD.
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        target = PAPER.get(method)
        summary[method] = {"mean": mean, "std_sample": std, "n": len(values), "seeds": [run["seed"] for run in selected],
                           "paper_mean": target[0] if target else None, "paper_std": target[1] if target else None,
                           "delta_mean": mean - target[0] if target else None,
                           "matches_paper_3dp": (f"{mean:.3f}" == f"{target[0]:.3f}" and f"{std:.3f}" == f"{target[1]:.3f}") if target else None}
    return {"mode": args.mode, "protocol": protocol, "aggregate": summary, "runs": runs}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True,
                        help="Directory containing routing labels, DNA, query embeddings, and replay checkpoints.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "replay"), default="train")
    parser.add_argument("--methods", nargs="+", choices=(*PAPER, "random-dna"), default=list(PAPER))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cache_root, output = args.cache_root.resolve(), args.output_dir.resolve()
    if not cache_root.is_dir():
        raise ValueError(f"Cache root does not exist: {cache_root}")
    if output == cache_root or cache_root in output.parents:
        raise ValueError("Output must be outside the read-only cache root")
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1:
        raise ValueError("epochs, batch-size, and threads must be positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        raise ValueError("Duplicate seeds or methods are not allowed")
    torch.set_num_threads(args.threads)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    frames = prepare_splits(cache_root, output, manifest)
    all_names = sorted(set().union(*(set(frame.model_name) for frame in frames.values())))
    cohorts = {}
    need = {"dna" if method in ("dna", "random-dna") else "embedllm" for method in args.methods}
    for key in sorted(need):
        vectors = load_dna(cache_root, DATASET + ("_embed" if key == "embedllm" else ""), all_names, manifest)
        names = sorted(vectors)
        cohorts[key] = (names, np.stack([vectors[name] for name in names]), restrict_frames(frames, names))
    protocol = {
        "paper": PAPER_URL, "mode": args.mode, "seeds": args.seeds, "epochs": args.epochs if args.mode == "train" else None,
        "training": {"epochs": args.epochs, "batch_size": args.batch_size, "optimizer": "Adam", "learning_rate": 1e-4,
                     "weight_decay": 1e-5, "query_noise_std": .05, "checkpoint_selection": "last epoch", "loss": "cross entropy"},
        "std_ddof": 1, "device": args.device, "threads": args.threads,
        "cohorts": {key: {"model_names": value[0], "num_models": len(value[0]),
                           "excluded_models": sorted(set(all_names) - set(value[0])),
                           "splits": {split: {"pairs": len(frame), "prompts": int(frame.prompt_id.nunique())}
                                      for split, frame in value[2].items()}}
                    for key, value in cohorts.items()},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                        "pandas": pd.__version__, "platform": platform.platform()},
        "public_source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                                 (ROOT / "src/llm_dna/dna/DNASignature.py", Path(__file__))},
    }
    write_json(output / "protocol.json", protocol)
    runs = []
    for method in args.methods:
        key = "dna" if method in ("dna", "random-dna") else "embedllm"
        names, vectors, selected = cohorts[key]
        queries = None
        if method not in ("random", "single-best"):
            query_path = cache_root / "cache/route_fixed_model" / QUERY_FILES[key]
            add_input(manifest, query_path, cache_root, "frozen cached query features")
            pids = np.unique(np.concatenate([frame.prompt_id.to_numpy() for frame in selected.values()]))
            queries = load_queries(query_path, pids, 4096 if key == "dna" else 768)
        for seed in args.seeds:
            destination = output / method / f"seed_{seed}"
            destination.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            run = {"method": method, "seed": seed, "mode": args.mode, "num_models": len(names)}
            if method == "random":
                run.update({split: random_baseline(selected[split], seed + (split == "test")) for split in ("val", "test")})
            elif method == "single-best":
                run.update(single_best(selected))
            else:
                if args.mode == "replay":
                    path = checkpoint_path(cache_root, method, seed)
                    add_input(manifest, path, cache_root, "fitted historical router checkpoint, replay only")
                    checkpoint = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
                    validate_checkpoint(checkpoint, names, queries, vectors, method, seed)
                    state = checkpoint["state_dict"]
                    run["checkpoint"] = str(path.relative_to(cache_root))
                    run["history"] = []
                else:
                    state, run["history"] = train_router(selected, names, queries, vectors, method, args, seed, destination)
                # Training labels determine parameters and the single-best arm;
                # only held-out val/test predictions are needed for Table 2.
                for split in ("val", "test"):
                    run[split] = evaluate(state, selected[split], names, destination / f"{split}_predictions.csv", args.batch_size)
                del state
            run["seconds"] = time.perf_counter() - start
            reference = saved_reference(cache_root, method, seed, manifest)
            run["archived_test_accuracy"] = reference
            run["delta_from_archive"] = run["test"]["router_accuracy"] - reference if reference is not None else None
            write_json(destination / "metrics.json", run)
            runs.append(run)
            write_json(output / "results.json", summarize(runs, args, protocol))
            write_json(output / "manifest.json", manifest)
            print(f"{method} seed={seed} test={run['test']['router_accuracy']:.6f} elapsed={run['seconds']:.1f}s", flush=True)
    report = summarize(runs, args, protocol)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "mean", "std_sample", "n", "paper_mean", "paper_std", "delta_mean", "matches_paper_3dp"])
        writer.writeheader()
        for method, values in report["aggregate"].items():
            writer.writerow({"method": method, **{key: value for key, value in values.items() if key != "seeds"}})
    return report


if __name__ == "__main__":
    main()
