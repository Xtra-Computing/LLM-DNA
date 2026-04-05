"""Public API for programmatic DNA extraction."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    from .dna.DNASignature import DNASignature


@dataclass(slots=True)
class DNAExtractionConfig:
    """Configuration for extracting one model DNA vector."""

    model_name: str
    model_path: Optional[str] = None
    model_type: str = "auto"
    dataset: str = "rand"
    probe_set: str = "rand"
    max_samples: int = 100
    data_root: str = "./data"
    extractor_type: str = "embedding"
    dna_dim: int = 128
    reduction_method: str = "random_projection"
    embedding_merge: str = "concat"
    max_length: int = 1024
    output_dir: Path = Path("./out")
    output_path: Optional[Path] = None
    save: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    no_quantization: bool = False
    metadata_file: Optional[Path] = Path("./configs/llm_metadata.json")
    token: Optional[str] = None
    trust_remote_code: bool = True
    device: str = "auto"
    gpu_id: Optional[int] = None
    log_level: str = "INFO"
    random_seed: int = 42
    use_chat_template: bool = False


@dataclass(slots=True)
class DNAExtractionResult:
    """Result payload for a DNA extraction run."""

    model_name: str
    dataset: str
    vector: np.ndarray
    signature: "DNASignature"
    output_path: Optional[Path]
    summary_path: Optional[Path]
    elapsed_seconds: float


def _resolve_hf_token(explicit_token: Optional[str]) -> Optional[str]:
    if explicit_token:
        return explicit_token

    env_vars = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")
    for env_var in env_vars:
        value = os.getenv(env_var, "").strip()
        if value:
            return value
    return None


def _default_model_metadata(model_name: str) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "architecture": {"is_generative": True},
        "repository": {},
    }


def _load_model_metadata_for_model(
    model_name: str,
    metadata_file: Optional[Path],
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Load model metadata from file or fetch from HuggingFace Hub."""
    # Try to load from metadata file first
    if metadata_file is not None:
        try:
            from .core import extraction as core
            all_metadata = core.load_model_metadata(metadata_file)
            model_meta = all_metadata.get(model_name)
            if model_meta:
                return model_meta
        except Exception as exc:
            logging.debug("Failed to load metadata file %s: %s", metadata_file, exc)

    # Fetch metadata from HuggingFace Hub (with caching)
    logging.info("Fetching metadata for '%s' from HuggingFace Hub...", model_name)
    try:
        from .utils.metadata import get_model_metadata
        return get_model_metadata(model_name, token=token)
    except Exception as exc:
        logging.warning("Failed to fetch metadata for '%s': %s", model_name, exc)
        return {
            "model_name": model_name,
            "architecture": {"is_generative": True},
            "repository": {},
        }


def _resolve_model_path(model_path: Optional[str], model_meta: Dict[str, Any]) -> Optional[str]:
    if model_path:
        return model_path

    repository = model_meta.get("repository", {})
    local_path = repository.get("local_path")
    if local_path and Path(local_path).exists():
        return local_path

    model_id = repository.get("model_id")
    if model_id and Path(model_id).exists():
        return model_id

    return None


def _resolve_device(config: DNAExtractionConfig) -> str:
    from .core import extraction as core

    if config.gpu_id is not None:
        return core.validate_device_argument(f"cuda:{int(config.gpu_id)}")
    return core.validate_device_argument(config.device)


def _validate_quantization(config: DNAExtractionConfig) -> None:
    quant_flags = [config.load_in_4bit, config.load_in_8bit, config.no_quantization]
    if sum(quant_flags) > 1:
        raise ValueError(
            "Use only one of load_in_4bit, load_in_8bit, or no_quantization."
        )

    if config.load_in_4bit or config.load_in_8bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                "Quantization requires bitsandbytes. Install with `pip install bitsandbytes`."
            ) from exc


def _signature_output_paths(config: DNAExtractionConfig) -> tuple[Path, Path]:
    if config.output_path is not None:
        output_path = Path(config.output_path)
        summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
        return output_path, summary_path

    safe_model_name = config.model_name.replace("/", "_").replace(":", "_")
    dataset_identifier = config.dataset.replace(",", "_")
    structured_dir = Path(config.output_dir) / dataset_identifier / safe_model_name
    output_path = structured_dir / f"{safe_model_name}_dna.json"
    summary_path = structured_dir / f"{safe_model_name}_summary.json"
    return output_path, summary_path


def _save_signature_outputs(
    signature: "DNASignature",
    config: DNAExtractionConfig,
    output_path: Path,
    summary_path: Path,
    elapsed_seconds: float,
) -> None:
    from .core import extraction as core

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    signature.save(output_path, format="json")

    config_dump = asdict(config)
    config_dump["output_dir"] = str(config.output_dir)
    if config.output_path is not None:
        config_dump["output_path"] = str(config.output_path)
    if config.metadata_file is not None:
        config_dump["metadata_file"] = str(config.metadata_file)

    summary = {
        "model_name": config.model_name,
        "dataset": config.dataset,
        "dataset_full": core.get_dataset_name(config.dataset),
        "extractor_type": config.extractor_type,
        "dna_dimension": config.dna_dim,
        "reduction_method": config.reduction_method,
        "num_probes": signature.metadata.probe_count,
        "total_time_seconds": elapsed_seconds,
        "signature_stats": signature.get_statistics(),
        "metadata": signature.metadata.__dict__,
        "output_file": str(output_path),
        "config": config_dump,
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str, ensure_ascii=False, allow_nan=False)


def _validate_signature(signature: "DNASignature") -> np.ndarray:
    from .dna.DNASignature import DNASignature

    if not isinstance(signature, DNASignature):
        raise TypeError(f"Expected DNASignature, got {type(signature)}")

    vector = np.asarray(signature.signature, dtype=np.float32)
    if vector.size == 0:
        raise ValueError("Empty DNA signature detected.")
    if np.allclose(vector, 0.0):
        raise ValueError("All-zero DNA signature detected.")
    return vector


def _load_model_names_from_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Model list file not found: {path}")

    models: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        models.append(line)

    if not models:
        raise ValueError(f"No valid model names found in {path}")
    return models


def _resolve_generation_devices(
    config: DNAExtractionConfig,
    gpu_ids: Optional[list[int]],
) -> list[str]:
    from .core import extraction as core
    import torch

    if gpu_ids:
        return [core.validate_device_argument(f"cuda:{int(gpu_id)}") for gpu_id in gpu_ids]

    if config.gpu_id is not None:
        return [core.validate_device_argument(f"cuda:{int(config.gpu_id)}")]

    device = config.device.strip().lower()
    if device == "auto":
        if torch.cuda.is_available():
            return ["cuda:0"]
        return ["cpu"]
    if device == "cuda":
        return ["cuda:0"]
    return [core.validate_device_argument(config.device)]


def _is_api_model_type(model_type: str) -> bool:
    return model_type in {"openai", "openrouter", "gemini", "anthropic"}


def _is_api_parallel_mode(config: DNAExtractionConfig, model_names: list[str]) -> bool:
    if _is_api_model_type(config.model_type):
        return True
    if config.model_type != "auto":
        return False

    try:
        from .models.ModelLoader import ModelLoader

        loader = ModelLoader()
        detected_types = [loader._detect_model_type(name) for name in model_names]
        return bool(detected_types) and all(_is_api_model_type(model_type) for model_type in detected_types)
    except Exception as exc:
        logging.debug("Failed to infer API parallel mode from model names: %s", exc)
        return False


def _response_cache_path(config: DNAExtractionConfig, model_name: str) -> Path:
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    dataset_identifier = config.dataset.replace(",", "_")
    return Path(config.output_dir) / dataset_identifier / safe_model_name / "responses.json"


def _normalize_responses(responses: list[str], expected_count: int) -> list[str]:
    normalized = [item if isinstance(item, str) else "" if item is None else str(item) for item in responses]
    if len(normalized) < expected_count:
        normalized.extend([""] * (expected_count - len(normalized)))
    if len(normalized) > expected_count:
        normalized = normalized[:expected_count]
    return normalized


def _load_cached_responses(path: Path, expected_count: int) -> Optional[list[str]]:
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logging.warning("Failed to parse cached responses from %s: %s", path, exc)
        return None

    if isinstance(payload, dict):
        complete = payload.get("complete")
        if complete is False:
            logging.warning("Ignoring incomplete cached responses at %s", path)
            return None

    responses: list[str]
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        responses = [str(item.get("response", "")) for item in payload["items"] if isinstance(item, dict)]
    elif isinstance(payload, list):
        responses = [str(item) for item in payload]
    else:
        logging.warning("Unexpected cached response format at %s", path)
        return None

    if not responses:
        return None

    non_empty_count = sum(1 for response in responses if response.strip())
    if non_empty_count == 0:
        logging.warning("Ignoring cached responses at %s because all responses are empty.", path)
        return None

    if len(responses) != expected_count:
        logging.warning(
            "Cached responses at %s have probe count mismatch (%s != %s); normalizing by truncating/padding.",
            path,
            len(responses),
            expected_count,
        )
    else:
        logging.info("Loaded cached responses from %s (%d items).", path, len(responses))

    return _normalize_responses(responses, expected_count=expected_count)


def _save_response_cache(
    path: Path,
    model_name: str,
    dataset: str,
    prompts: list[str],
    responses: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = [{"prompt": str(prompt), "response": str(response)} for prompt, response in zip(prompts, responses)]
    payload = {
        "model": model_name,
        "dataset": dataset,
        "count": len(items),
        "items": items,
        "generated_at": datetime.now().isoformat(),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _generate_responses_for_model(
    model_name: str,
    config: DNAExtractionConfig,
    model_meta: Dict[str, Any],
    probe_texts: list[str],
    device: str,
    resolved_token: Optional[str],
    incremental_save_path: Optional[Path] = None,
) -> list[str]:
    from .models.ModelLoader import ModelLoader

    model_loader = ModelLoader()
    load_in_4bit = config.load_in_4bit
    load_in_8bit = config.load_in_8bit

    if not config.no_quantization and not load_in_4bit and not load_in_8bit:
        size_in_billions = model_meta.get("size", {}).get("parameter_count_billions")
        if size_in_billions and size_in_billions >= 7.0:
            load_in_8bit = True

    resolved_model_path = _resolve_model_path(config.model_path, model_meta)
    model = model_loader.load_model(
        model_path_or_name=resolved_model_path or model_name,
        model_type=config.model_type,
        device=device,
        trust_remote_code=config.trust_remote_code,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        token=resolved_token,
        is_chat_model=model_meta.get("chat_model", {}).get("is_chat_model", False),
    )

    # Set up incremental saving callback
    incremental_items: list[dict] = []
    def _save_response_incrementally(idx: int, prompt: str, response: str) -> None:
        """Callback to save each response as it's generated."""
        incremental_items.append({"prompt": prompt, "response": response})
        if incremental_save_path is not None:
            incremental_save_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model": model_name,
                "dataset": config.dataset,
                "count": len(incremental_items),
                "complete": len(incremental_items) >= len(probe_texts),
                "items": incremental_items,
                "generated_at": datetime.now().isoformat(),
            }
            with open(incremental_save_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)

    try:
        if hasattr(model, "generate_batch") and callable(getattr(model, "generate_batch")):
            responses = model.generate_batch(
                probe_texts,
                max_length=config.max_length,
                temperature=0.0,
                do_sample=False,
                top_p=1.0,
                use_chat_template=config.use_chat_template,
                on_response_callback=_save_response_incrementally if incremental_save_path else None,
            )
        else:
            responses = []
            for idx, prompt in enumerate(probe_texts):
                response = model.generate(
                    prompt,
                    max_length=config.max_length,
                    temperature=0.0,
                    do_sample=False,
                    top_p=1.0,
                    use_chat_template=config.use_chat_template,
                )
                responses.append(response)
                if incremental_save_path:
                    _save_response_incrementally(idx, prompt, response)
    finally:
        try:
            if hasattr(model, "release") and callable(getattr(model, "release")):
                model.release()
        except Exception:
            logging.debug("Failed to release model %s cleanly.", model_name)

    return _normalize_responses(list(responses), expected_count=len(probe_texts))


def _extract_signature_from_text_responses(
    model_name: str,
    responses: list[str],
    config: DNAExtractionConfig,
    model_meta: Dict[str, Any],
    generation_device: str,
    sentence_encoder: str = "all-mpnet-base-v2",
    encoder_device: Optional[str] = None,
) -> tuple["DNASignature", np.ndarray, float]:
    from .dna.DNASignature import DNAMetadata, DNASignature
    from .dna.EmbeddingDNAExtractor import EmbeddingDNAExtractor
    from sentence_transformers import SentenceTransformer

    if not responses:
        raise ValueError("No responses available for text-response embedding extraction.")

    resolved_encoder_device = encoder_device or generation_device or "cpu"
    logging.info(
        "Encoding %d cached response(s) with sentence encoder '%s' on %s",
        len(responses),
        sentence_encoder,
        resolved_encoder_device,
    )

    encode_started = time.time()
    encoder = SentenceTransformer(sentence_encoder, device=resolved_encoder_device)
    embeddings = encoder.encode(
        responses,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    encode_seconds = time.time() - encode_started

    reducer = EmbeddingDNAExtractor(
        dna_dim=config.dna_dim,
        reduction_method=config.reduction_method,
        aggregation_method=config.embedding_merge,
        device="cpu",
        random_seed=config.random_seed,
    )

    reduce_started = time.time()
    reduced_vector = reducer._reduce_features(embeddings)
    reduce_seconds = time.time() - reduce_started

    metadata = DNAMetadata(
        model_name=model_name,
        extraction_method=f"text_response_embeddings_{config.reduction_method}_{config.embedding_merge}",
        probe_set_id=f"{config.dataset}_{len(responses)}",
        probe_count=len(responses),
        dna_dimension=config.dna_dim,
        embedding_dimension=int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        reduction_method=config.reduction_method,
        extraction_time=datetime.now().isoformat(),
        computation_time_seconds=encode_seconds + reduce_seconds,
        model_metadata=model_meta,
        extractor_config={
            "mode": "single_cached_response_embeddings",
            "sentence_encoder": sentence_encoder,
            "generation_device": generation_device,
            "encoder_device": resolved_encoder_device,
            "max_length": config.max_length,
            "dataset": config.dataset,
        },
        aggregation_method=config.embedding_merge,
    )
    signature = DNASignature(signature=np.asarray(reduced_vector, dtype=np.float32), metadata=metadata)
    vector = _validate_signature(signature)
    return signature, vector, (encode_seconds + reduce_seconds)


def calc_dna(config: DNAExtractionConfig) -> DNAExtractionResult:
    """Compute DNA vector for one model and optionally persist outputs."""

    from .core import extraction as core
    from .utils.DataUtils import setup_logging

    setup_logging(level=config.log_level)
    _validate_quantization(config)

    start_time = time.time()
    metadata_file = Path(config.metadata_file) if config.metadata_file is not None else None
    resolved_device = _resolve_device(config)
    resolved_token = _resolve_hf_token(config.token)
    probe_texts = core.get_probe_texts(
        dataset_id=config.dataset,
        probe_set=config.probe_set,
        max_samples=config.max_samples,
        data_root=config.data_root,
        random_seed=config.random_seed,
    )

    signature: "DNASignature"
    vector: np.ndarray

    if config.extractor_type != "embedding":
        raise ValueError(f"Unsupported extractor_type for calc_dna: {config.extractor_type}")

    response_path = _response_cache_path(config, config.model_name)
    cached_responses = _load_cached_responses(response_path, expected_count=len(probe_texts))

    if cached_responses is not None:
        logging.info(
            "Using cached responses for '%s' from %s; skipping provider model/key checks.",
            config.model_name,
            response_path,
        )
        model_meta = _default_model_metadata(config.model_name)
        responses = cached_responses
    else:
        if _is_api_model_type(config.model_type):
            logging.info(
                "Generating responses for API model '%s' via provider API...",
                config.model_name,
            )

        model_meta = _load_model_metadata_for_model(config.model_name, metadata_file, token=resolved_token)
        is_generative = model_meta.get("architecture", {}).get("is_generative")
        if is_generative is False:
            arch_type = model_meta.get("architecture", {}).get("type")
            raise ValueError(
                f"Model '{config.model_name}' is non-generative (architecture={arch_type})."
            )

        responses = _generate_responses_for_model(
            model_name=config.model_name,
            config=config,
            model_meta=model_meta,
            probe_texts=probe_texts,
            device=resolved_device,
            resolved_token=resolved_token,
            incremental_save_path=response_path if config.save else None,
        )
        if config.save:
            _save_response_cache(
                path=response_path,
                model_name=config.model_name,
                dataset=config.dataset,
                prompts=probe_texts,
                responses=responses,
            )

    signature, vector, _ = _extract_signature_from_text_responses(
        model_name=config.model_name,
        responses=responses,
        config=config,
        model_meta=model_meta,
        generation_device=resolved_device,
        encoder_device=resolved_device,
    )

    elapsed_seconds = time.time() - start_time

    output_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    if config.save:
        output_path, summary_path = _signature_output_paths(config)
        _save_signature_outputs(
            signature=signature,
            config=config,
            output_path=output_path,
            summary_path=summary_path,
            elapsed_seconds=elapsed_seconds,
        )

    return DNAExtractionResult(
        model_name=config.model_name,
        dataset=config.dataset,
        vector=vector,
        signature=signature,
        output_path=output_path,
        summary_path=summary_path,
        elapsed_seconds=elapsed_seconds,
    )


def calc_dna_parallel(
    config: DNAExtractionConfig,
    llm_list: Optional[Path | str] = None,
    gpu_ids: Optional[list[int]] = None,
    n_processes: Optional[int] = None,
    continue_on_error: bool = False,
    sentence_encoder: str = "all-mpnet-base-v2",
    encoder_device: str = "auto",
    use_response_cache: bool = True,
) -> list[DNAExtractionResult]:
    """Batch DNA extraction with GPU worker scheduling and shared text encoding.

    This additive API keeps ``calc_dna`` unchanged. When ``llm_list`` is provided,
    ``config.model_name`` is ignored and a warning is emitted.
    """

    from .core import extraction as core
    from .dna.DNASignature import DNAMetadata, DNASignature
    from .dna.EmbeddingDNAExtractor import EmbeddingDNAExtractor
    from .utils.DataUtils import setup_logging

    setup_logging(level=config.log_level)
    _validate_quantization(config)

    if config.max_samples < 2:
        raise ValueError("Batch text-embedding mode requires max_samples >= 2.")

    metadata_file = Path(config.metadata_file) if config.metadata_file is not None else None
    resolved_token = _resolve_hf_token(config.token)

    if llm_list is not None:
        if config.model_name:
            logging.warning(
                "llm_list is provided; ignoring model_name=%s and using list entries.",
                config.model_name,
            )
        model_names = _load_model_names_from_file(Path(llm_list))
    else:
        model_names = [config.model_name]

    if not model_names:
        raise ValueError("No models provided for batch extraction.")

    probe_texts = core.get_probe_texts(
        dataset_id=config.dataset,
        probe_set=config.probe_set,
        max_samples=config.max_samples,
        data_root=config.data_root,
        random_seed=config.random_seed,
    )
    if len(probe_texts) < 2:
        raise ValueError("Probe set must contain at least two prompts for dimensionality reduction.")

    generation_devices = _resolve_generation_devices(config, gpu_ids)
    worker_devices = list(generation_devices)
    if n_processes is not None:
        worker_count = int(n_processes)
        if worker_count <= 0:
            raise ValueError("n_processes must be >= 1 when provided.")
        if _is_api_parallel_mode(config, model_names):
            worker_devices = [
                generation_devices[idx % len(generation_devices)]
                for idx in range(worker_count)
            ]
            logging.info(
                "Using API submission worker pool with n_processes=%d over devices=%s",
                worker_count,
                generation_devices,
            )
        elif worker_count != len(generation_devices):
            logging.warning(
                "Ignoring n_processes=%d for non-API workload; using %d generation worker(s).",
                worker_count,
                len(generation_devices),
            )

    logging.info(
        "Starting batch generation for %d model(s) across %d worker device(s): %s",
        len(model_names),
        len(worker_devices),
        worker_devices,
    )

    task_queue: queue.Queue[Optional[str]] = queue.Queue()
    result_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    for model_name in model_names:
        task_queue.put(model_name)
    for _ in worker_devices:
        task_queue.put(None)

    def _worker(device: str) -> None:
        while True:
            model_name = task_queue.get()
            if model_name is None:
                return

            started = time.time()
            try:
                response_path = _response_cache_path(config, model_name)
                responses: Optional[list[str]] = None
                if use_response_cache:
                    responses = _load_cached_responses(response_path, expected_count=len(probe_texts))

                if responses is not None:
                    # Cached responses allow downstream encoding without provider API keys.
                    model_meta = _default_model_metadata(model_name)
                else:
                    model_meta = _load_model_metadata_for_model(
                        model_name=model_name,
                        metadata_file=metadata_file,
                        token=resolved_token,
                    )
                    if model_meta.get("architecture", {}).get("is_generative") is False:
                        arch_type = model_meta.get("architecture", {}).get("type")
                        raise ValueError(
                            f"Model '{model_name}' is non-generative (architecture={arch_type})."
                        )

                    responses = _generate_responses_for_model(
                        model_name=model_name,
                        config=config,
                        model_meta=model_meta,
                        probe_texts=probe_texts,
                        device=device,
                        resolved_token=resolved_token,
                        incremental_save_path=response_path if use_response_cache else None,
                    )
                    if use_response_cache:
                        _save_response_cache(
                            path=response_path,
                            model_name=model_name,
                            dataset=config.dataset,
                            prompts=probe_texts,
                            responses=responses,
                        )

                result_queue.put(
                    {
                        "success": True,
                        "model_name": model_name,
                        "device": device,
                        "metadata": model_meta,
                        "responses": responses,
                        "generation_seconds": time.time() - started,
                    }
                )
            except Exception as exc:
                result_queue.put(
                    {
                        "success": False,
                        "model_name": model_name,
                        "device": device,
                        "error": str(exc),
                        "generation_seconds": time.time() - started,
                    }
                )

    threads = [threading.Thread(target=_worker, args=(device,), daemon=True) for device in worker_devices]
    for thread in threads:
        thread.start()

    successful_payloads: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    completed = 0
    total = len(model_names)

    while completed < total:
        event = result_queue.get()
        completed += 1
        model_name = event["model_name"]
        if event["success"]:
            successful_payloads.append(event)
            logging.info(
                "[%d/%d] Generated responses for %s on %s in %.2fs",
                completed,
                total,
                model_name,
                event["device"],
                event["generation_seconds"],
            )
        else:
            failures.append(event)
            logging.error(
                "[%d/%d] Failed generation for %s on %s: %s",
                completed,
                total,
                model_name,
                event["device"],
                event["error"],
            )

    for thread in threads:
        thread.join()

    if not successful_payloads:
        raise RuntimeError("Batch generation failed for all models.")

    if failures and not continue_on_error:
        failed_models = ", ".join(item["model_name"] for item in failures)
        raise RuntimeError(f"Batch generation failed for model(s): {failed_models}")

    ordered_payloads = [item for name in model_names for item in successful_payloads if item["model_name"] == name]

    all_responses: list[str] = []
    model_slices: dict[str, tuple[int, int]] = {}
    for payload in ordered_payloads:
        start = len(all_responses)
        all_responses.extend(payload["responses"])
        model_slices[payload["model_name"]] = (start, len(all_responses))

    if encoder_device == "auto":
        encoder_device = worker_devices[0] if worker_devices else "cpu"

    logging.info(
        "Encoding %d response(s) with sentence encoder '%s' on %s",
        len(all_responses),
        sentence_encoder,
        encoder_device,
    )

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(sentence_encoder, device=encoder_device)
    encode_started = time.time()
    all_embeddings = encoder.encode(
        all_responses,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    )
    all_embeddings = np.asarray(all_embeddings, dtype=np.float32)
    encode_seconds = time.time() - encode_started

    reducer = EmbeddingDNAExtractor(
        dna_dim=config.dna_dim,
        reduction_method=config.reduction_method,
        aggregation_method=config.embedding_merge,
        device="cpu",
        random_seed=config.random_seed,
    )

    results: list[DNAExtractionResult] = []
    for payload in ordered_payloads:
        model_name = payload["model_name"]
        start, end = model_slices[model_name]
        model_embeddings = all_embeddings[start:end]

        reduce_started = time.time()
        reduced_vector = reducer._reduce_features(model_embeddings)
        reduce_seconds = time.time() - reduce_started

        metadata = DNAMetadata(
            model_name=model_name,
            extraction_method=f"text_response_embeddings_{config.reduction_method}_{config.embedding_merge}",
            probe_set_id=f"{config.dataset}_{len(probe_texts)}",
            probe_count=len(probe_texts),
            dna_dimension=config.dna_dim,
            embedding_dimension=int(model_embeddings.shape[1]),
            reduction_method=config.reduction_method,
            extraction_time=datetime.now().isoformat(),
            computation_time_seconds=payload["generation_seconds"] + encode_seconds + reduce_seconds,
            model_metadata=payload["metadata"],
            extractor_config={
                "mode": "llm_list_parallel_batch",
                "sentence_encoder": sentence_encoder,
                "generation_device": payload["device"],
                "encoder_device": encoder_device,
                "max_length": config.max_length,
                "dataset": config.dataset,
            },
            aggregation_method=config.embedding_merge,
        )
        signature = DNASignature(signature=np.asarray(reduced_vector, dtype=np.float32), metadata=metadata)
        vector = _validate_signature(signature)

        model_config = replace(config, model_name=model_name, output_path=None)
        output_path: Optional[Path] = None
        summary_path: Optional[Path] = None
        elapsed_seconds = payload["generation_seconds"] + encode_seconds + reduce_seconds
        if model_config.save:
            output_path, summary_path = _signature_output_paths(model_config)
            _save_signature_outputs(
                signature=signature,
                config=model_config,
                output_path=output_path,
                summary_path=summary_path,
                elapsed_seconds=elapsed_seconds,
            )

        results.append(
            DNAExtractionResult(
                model_name=model_name,
                dataset=model_config.dataset,
                vector=vector,
                signature=signature,
                output_path=output_path,
                summary_path=summary_path,
                elapsed_seconds=elapsed_seconds,
            )
        )

    return results


def calc_dna_batch(
    configs: list[DNAExtractionConfig],
    gpu_ids: Optional[list[int]] = None,
    continue_on_error: bool = False,
) -> list[DNAExtractionResult]:
    """Compute DNA vectors for multiple models with optional GPU round-robin."""

    results: list[DNAExtractionResult] = []
    failures = 0
    for index, config in enumerate(configs):
        run_config = config
        if config.gpu_id is None and gpu_ids:
            run_config = replace(config, gpu_id=gpu_ids[index % len(gpu_ids)])

        try:
            results.append(calc_dna(run_config))
        except Exception:
            failures += 1
            if not continue_on_error:
                raise

    if failures > 0 and not continue_on_error:
        raise RuntimeError("One or more DNA extraction runs failed.")
    return results


__all__ = [
    "DNAExtractionConfig",
    "DNAExtractionResult",
    "calc_dna",
    "calc_dna_parallel",
    "calc_dna_batch",
]
