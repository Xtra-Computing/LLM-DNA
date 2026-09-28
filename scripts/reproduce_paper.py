#!/usr/bin/env python3
"""Run the paper's relationship and routing experiments from local caches.

No generation, encoder inference, downloads, or remote writes are performed.
Routing defaults to fresh downstream training. Use --routing-mode replay to
re-evaluate historical checkpoints instead. See experiments/paper_reproduction.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True,
                        help="Directory containing the cached experiment inputs.")
    parser.add_argument("--relations", type=Path, default=ROOT / "configs/valid_llm_relation.csv",
                        help="Positive-relation CSV for classification; default: configs/valid_llm_relation.csv in this checkout.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", choices=("relationship", "routing"),
                        default=["relationship", "routing"])
    parser.add_argument("--routing-mode", choices=("train", "replay"), default="train")
    parser.add_argument("--routing-device", default="cpu")
    parser.add_argument("--routing-threads", type=int, default=8)
    parser.add_argument("--routing-epochs", type=int, default=50)
    parser.add_argument("--routing-seeds", type=int, nargs="+", default=list(range(5)))
    args = parser.parse_args()
    cache_root = args.cache_root.resolve()
    output = args.output_dir.resolve()
    if not cache_root.is_dir():
        parser.error(f"Cache root does not exist: {cache_root}")
    if output == cache_root or cache_root in output.parents:
        parser.error("Output must be outside the read-only cache root")
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       HF_DATASETS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    runs = []
    for experiment in dict.fromkeys(args.experiments):
        destination = output / ("relationship" if experiment == "relationship"
                                else f"routing_{args.routing_mode}")
        runner = "reproduce_relationship_table.py" if experiment == "relationship" else "reproduce_routing.py"
        command = [sys.executable, "-u", str(ROOT / "scripts" / runner),
                   "--cache-root", str(cache_root), "--output-dir", str(destination)]
        if experiment == "relationship":
            command += ["--relations", str(args.relations.resolve())]
        if experiment == "routing":
            command += ["--mode", args.routing_mode, "--device", args.routing_device,
                        "--threads", str(args.routing_threads), "--epochs", str(args.routing_epochs),
                        "--seeds", *map(str, args.routing_seeds)]
        started = datetime.now(timezone.utc).isoformat()
        print(f"Running {experiment}; output: {destination}", flush=True)
        with (output / f"{experiment}.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
        runs.append({"experiment": experiment, "command": command, "started_utc": started,
                     "finished_utc": datetime.now(timezone.utc).isoformat(),
                     "output_dir": str(destination), "returncode": returncode})
        (output / "run_manifest.json").write_text(json.dumps(runs, indent=2) + "\n")
        if returncode:
            return returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
