"""CLI entrypoint for LLM-DNA DNA extraction."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional


def _load_models_from_file(path: Path) -> List[str]:
    """Load model names from a file, one per line."""
    if not path.exists():
        raise FileNotFoundError(f"Model list file not found: {path}")

    models: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        models.append(line)

    if not models:
        raise ValueError(f"No valid model names found in {path}")
    return models


def _parse_gpu_ids(raw_gpus: str) -> List[int]:
    """Parse comma-separated GPU IDs."""
    text = raw_gpus.strip()
    if not text:
        return []

    gpu_ids: List[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        gpu_ids.append(int(item))
    return gpu_ids


def parse_arguments(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract DNA vectors from LLMs using LLM-DNA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model selection
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        help="Model name/path. Repeat to pass multiple models.",
    )
    parser.add_argument(
        "--llm-list",
        type=Path,
        default=Path("./configs/llm_list.txt"),
        help="Fallback model list file used when --model-name is omitted.",
    )

    # GPU and execution
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU IDs (used for round-robin assignment).",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=None,
        help="Number of parallel API submission workers for batch llm-list runs.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining models if one model fails.",
    )
    parser.add_argument(
        "--print-vector",
        action="store_true",
        help="Print DNA vector JSON to stdout for each model.",
    )

    # Model configuration
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "huggingface", "openai", "openrouter", "gemini", "anthropic"],
    )

    # Dataset and probes
    parser.add_argument("--dataset", type=str, default="rand")
    parser.add_argument("--probe-set", type=str, default="rand")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--data-root", type=str, default="./data")

    # DNA extraction
    parser.add_argument(
        "--extractor-type",
        type=str,
        default="embedding",
        choices=["embedding"],
    )
    parser.add_argument("--dna-dim", type=int, default=128)
    parser.add_argument(
        "--reduction-method",
        type=str,
        default="random_projection",
        choices=["pca", "svd", "random_projection"],
    )
    parser.add_argument(
        "--embedding-merge",
        type=str,
        default="concat",
        choices=["sum", "max", "mean", "concat"],
    )
    parser.add_argument("--max-length", type=int, default=1024)

    # Output
    parser.add_argument("--output-dir", type=Path, default=Path("./out"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional explicit output file for single-model runs.",
    )
    parser.add_argument("--no-save", action="store_true")

    # Quantization
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-quantization", action="store_true")

    # Hugging Face
    parser.add_argument(
        "--metadata-file", type=Path, default=Path("./configs/llm_metadata.json")
    )
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        help="Allow execution of custom code from model repositories.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
    )
    parser.set_defaults(trust_remote_code=True)

    # Device and logging
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device (auto, cpu, cuda, cuda:N).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        default=False,
        help="Apply chat template for HuggingFace models (default: disabled).",
    )

    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Main CLI entrypoint for DNA extraction."""
    from .api import DNAExtractionConfig, calc_dna, calc_dna_parallel

    args = parse_arguments(argv)

    # Resolve model names
    if args.model_name:
        model_names = args.model_name
    else:
        model_names = _load_models_from_file(args.llm_list)

    gpu_ids = _parse_gpu_ids(args.gpus)
    if args.device == "cpu":
        gpu_ids = []

    failures = 0
    single_model_run = len(model_names) == 1

    # Batch mode for llm_list: schedule model generation across worker devices.
    if not args.model_name and len(model_names) > 1:
        batch_config = DNAExtractionConfig(
            model_name=model_names[0],
            model_path=args.model_path,
            model_type=args.model_type,
            dataset=args.dataset,
            probe_set=args.probe_set,
            max_samples=args.max_samples,
            data_root=args.data_root,
            extractor_type=args.extractor_type,
            dna_dim=args.dna_dim,
            reduction_method=args.reduction_method,
            embedding_merge=args.embedding_merge,
            max_length=args.max_length,
            output_dir=args.output_dir,
            output_path=None,
            save=not args.no_save,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            no_quantization=args.no_quantization,
            metadata_file=args.metadata_file,
            token=args.token,
            trust_remote_code=args.trust_remote_code,
            device=args.device,
            gpu_id=None,
            log_level=args.log_level,
            random_seed=args.random_seed,
            use_chat_template=args.use_chat_template,
        )
        try:
            results = calc_dna_parallel(
                config=batch_config,
                llm_list=args.llm_list,
                gpu_ids=gpu_ids if gpu_ids else None,
                n_processes=args.n_processes,
                continue_on_error=args.continue_on_error,
            )
        except Exception as exc:
            logging.error("Batch DNA extraction failed: %s", exc)
            return 1

        for result in results:
            output_message = f"model={result.model_name} dim={result.vector.shape[0]}"
            if result.output_path is not None:
                output_message += f" saved={result.output_path}"
            print(output_message)
            if args.print_vector:
                print(json.dumps(result.vector.tolist()))
        return 0 if len(results) == len(model_names) else 1

    for index, model_name in enumerate(model_names):
        gpu_id = None
        if args.device in {"auto", "cuda"} and gpu_ids:
            gpu_id = gpu_ids[index % len(gpu_ids)]

        try:
            config = DNAExtractionConfig(
                model_name=model_name,
                model_path=args.model_path,
                model_type=args.model_type,
                dataset=args.dataset,
                probe_set=args.probe_set,
                max_samples=args.max_samples,
                data_root=args.data_root,
                extractor_type=args.extractor_type,
                dna_dim=args.dna_dim,
                reduction_method=args.reduction_method,
                embedding_merge=args.embedding_merge,
                max_length=args.max_length,
                output_dir=args.output_dir,
                output_path=args.output_path if single_model_run else None,
                save=not args.no_save,
                load_in_8bit=args.load_in_8bit,
                load_in_4bit=args.load_in_4bit,
                no_quantization=args.no_quantization,
                metadata_file=args.metadata_file,
                token=args.token,
                trust_remote_code=args.trust_remote_code,
                device=args.device,
                gpu_id=gpu_id,
                log_level=args.log_level,
                random_seed=args.random_seed,
                use_chat_template=args.use_chat_template,
            )

            result = calc_dna(config)

            output_message = f"model={model_name} dim={result.vector.shape[0]}"
            if result.output_path is not None:
                output_message += f" saved={result.output_path}"
            print(output_message)

            if args.print_vector:
                print(json.dumps(result.vector.tolist()))

        except Exception as exc:
            failures += 1
            logging.error("DNA extraction failed for %s: %s", model_name, exc)
            if not args.continue_on_error:
                return 1

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
