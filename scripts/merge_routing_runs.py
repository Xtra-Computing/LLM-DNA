#!/usr/bin/env python3
"""Validate and combine routing seed runs executed on separate devices."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from reproduce_routing import PAPER, summarize, write_json


def merge(inputs: list[Path], output: Path, expected_seeds: list[int], expected_methods: list[str] | None = None):
    methods = list(PAPER) if expected_methods is None else list(expected_methods)
    if not methods or len(set(methods)) != len(methods) or any(method not in PAPER for method in methods):
        raise ValueError("Expected methods must be nonempty, unique paper methods")
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("Methods and unique expected seeds must be nonempty")
    output = output.resolve()
    if output in {directory.resolve() for directory in inputs}:
        raise ValueError("Output must not overwrite an input run directory")
    runs, protocols, sources, manifest = [], [], [], {}
    seen = set()
    reference = None
    for directory in inputs:
        directory = directory.resolve()
        report = json.loads((directory / "results.json").read_text())
        protocol = report["protocol"]
        comparable = {key: protocol[key] for key in
                      ("mode", "training", "std_ddof", "cohorts", "public_source_sha256", "environment")}
        if reference is None:
            reference = comparable
        elif comparable != reference:
            raise ValueError(f"Incompatible protocol or source code: {directory}")
        protocols.append(protocol)
        sources.append(str(directory))
        for row in report["runs"]:
            key = (row["method"], row["seed"])
            if key in seen:
                raise ValueError(f"Duplicate run {key}")
            seen.add(key)
            runs.append({**row, "source_run_dir": str(directory / row["method"] / f"seed_{row['seed']}")})
        for path, value in json.loads((directory / "manifest.json").read_text()).items():
            if path in manifest and manifest[path]["sha256"] != value["sha256"]:
                raise ValueError(f"Input changed between runs: {path}")
            manifest[path] = value
    expected = {(method, seed) for method in methods for seed in expected_seeds}
    if seen != expected:
        raise ValueError(f"Incomplete or unexpected runs; missing={expected - seen}, extra={seen - expected}")
    runs.sort(key=lambda row: (methods.index(row["method"]), row["seed"]))
    protocol = {**protocols[0], "seeds": expected_seeds, "device": "multiple; see execution_protocols",
                "execution_protocols": protocols, "source_directories": sources}
    merged = summarize(runs, SimpleNamespace(methods=methods, mode=protocol["mode"]), protocol)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "results.json", merged)
    write_json(output / "protocol.json", protocol)
    write_json(output / "manifest.json", manifest)
    fields = ["method", "mean", "std_sample", "n", "paper_mean", "paper_std", "delta_mean", "matches_paper_3dp"]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method, values in merged["aggregate"].items():
            writer.writerow({"method": method, **{k: v for k, v in values.items() if k != "seeds"}})
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--expected-methods", nargs="+", choices=list(PAPER), default=list(PAPER),
                        help="Required result rows; defaults to all four Table 2 methods")
    args = parser.parse_args()
    merge(args.inputs, args.output_dir, args.expected_seeds, args.expected_methods)


if __name__ == "__main__":
    main()
