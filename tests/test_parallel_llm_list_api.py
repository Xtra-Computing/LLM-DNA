from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

import llm_dna.api as api
from llm_dna.api import DNAExtractionConfig


def _write_llm_list(path: Path, models: list[str]) -> Path:
    path.write_text("\n".join(models) + "\n", encoding="utf-8")
    return path


class _FakeSentenceEncoder:
    def __init__(self, *_args, **kwargs):
        self.device = kwargs.get("device")

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32, normalize_embeddings=False):
        embeddings = []
        for index, text in enumerate(texts):
            base = float(len(text) % 11)
            embeddings.append(
                [
                    base + (index * 0.01),
                    base + 1.0,
                    base + 2.0,
                    base + 3.0,
                    base + 4.0,
                    base + 5.0,
                ]
            )
        output = np.asarray(embeddings, dtype=np.float32)
        if convert_to_numpy:
            return output
        return output.tolist()


def test_calc_dna_parallel_uses_llm_list_and_multi_gpu(monkeypatch, tmp_path):
    model_names = ["distilgpt2", "gpt2", "Qwen/Qwen2.5-0.5B-Instruct"]
    llm_list = _write_llm_list(tmp_path / "llm_list.txt", model_names)
    devices_seen: list[str] = []

    monkeypatch.setattr(
        "llm_dna.core.extraction.get_probe_texts",
        lambda **_kwargs: ["prompt A", "prompt B", "prompt C"],
    )
    monkeypatch.setattr(
        api,
        "_load_model_metadata_for_model",
        lambda *args, **kwargs: {
            "architecture": {"is_generative": True},
            "repository": {},
            "size": {},
            "chat_model": {"is_chat_model": False},
        },
    )

    def fake_generate(
        model_name: str,
        config: DNAExtractionConfig,
        model_meta,
        probe_texts,
        device: str,
        resolved_token,
        incremental_save_path=None,
    ):
        del incremental_save_path
        devices_seen.append(device)
        return [f"{model_name}::{idx}" for idx in range(len(probe_texts))]

    monkeypatch.setattr(api, "_generate_responses_for_model", fake_generate)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeSentenceEncoder)

    config = DNAExtractionConfig(
        model_name="ignored-model-name",
        dataset="rand",
        max_samples=3,
        dna_dim=4,
        save=False,
        output_dir=tmp_path / "out",
        device="cuda",
    )

    results = api.calc_dna_parallel(
        config=config,
        llm_list=llm_list,
        gpu_ids=[0, 1],
        continue_on_error=True,
    )

    assert [item.model_name for item in results] == model_names
    assert all(result.vector.shape == (4,) for result in results)
    assert "cuda:0" in devices_seen
    assert "cuda:1" in devices_seen


def test_calc_dna_parallel_uses_cached_responses(monkeypatch, tmp_path):
    model_names = ["distilgpt2", "gpt2"]
    llm_list = _write_llm_list(tmp_path / "llm_list.txt", model_names)

    monkeypatch.setattr(
        "llm_dna.core.extraction.get_probe_texts",
        lambda **_kwargs: ["prompt A", "prompt B", "prompt C"],
    )
    monkeypatch.setattr(
        api,
        "_load_model_metadata_for_model",
        lambda *args, **kwargs: {
            "architecture": {"is_generative": True},
            "repository": {},
            "size": {},
            "chat_model": {"is_chat_model": False},
        },
    )

    safe_model_name = "distilgpt2"
    cache_path = tmp_path / "out" / "rand" / safe_model_name / "responses.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        """{
  "model": "distilgpt2",
  "dataset": "rand",
  "count": 3,
  "items": [
    {"prompt": "prompt A", "response": "cached response 1"},
    {"prompt": "prompt B", "response": "cached response 2"},
    {"prompt": "prompt C", "response": "cached response 3"}
  ]
}""",
        encoding="utf-8",
    )

    generated_for: list[str] = []

    def fake_generate(
        model_name: str,
        config: DNAExtractionConfig,
        model_meta,
        probe_texts,
        device: str,
        resolved_token,
        incremental_save_path=None,
    ):
        del incremental_save_path
        generated_for.append(model_name)
        return [f"{model_name}::{idx}" for idx in range(len(probe_texts))]

    monkeypatch.setattr(api, "_generate_responses_for_model", fake_generate)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeSentenceEncoder)

    config = DNAExtractionConfig(
        model_name="ignored-model-name",
        dataset="rand",
        max_samples=3,
        dna_dim=4,
        save=False,
        output_dir=tmp_path / "out",
        device="cpu",
    )

    results = api.calc_dna_parallel(
        config=config,
        llm_list=llm_list,
        continue_on_error=True,
    )

    assert len(results) == 2
    assert generated_for == ["gpt2"]


def test_calc_dna_parallel_uses_cached_responses_without_metadata_or_generation(monkeypatch, tmp_path):
    model_names = ["openrouter/pony-alpha"]
    llm_list = _write_llm_list(tmp_path / "llm_list.txt", model_names)

    monkeypatch.setattr(
        "llm_dna.core.extraction.get_probe_texts",
        lambda **_kwargs: ["prompt A", "prompt B", "prompt C"],
    )
    monkeypatch.setattr(
        api,
        "_load_model_metadata_for_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("metadata should not be loaded when cache exists")
        ),
    )
    monkeypatch.setattr(
        api,
        "_generate_responses_for_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("generation should not run when cache exists")
        ),
    )
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeSentenceEncoder)

    safe_model_name = "openrouter_pony-alpha"
    cache_path = tmp_path / "out" / "rand" / safe_model_name / "responses.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Empty replies retain their original probe slots.
    cache_path.write_text(
        """{
  "model": "openrouter/pony-alpha",
  "dataset": "rand",
  "count": 3,
  "items": [
    {"prompt": "prompt A", "response": "cached response 1"},
    {"prompt": "prompt B", "response": ""},
    {"prompt": "prompt C", "response": ""}
  ]
}""",
        encoding="utf-8",
    )

    config = DNAExtractionConfig(
        model_name="ignored-model-name",
        model_type="openrouter",
        dataset="rand",
        max_samples=3,
        dna_dim=4,
        save=False,
        output_dir=tmp_path / "out",
        device="cpu",
    )

    results = api.calc_dna_parallel(
        config=config,
        llm_list=llm_list,
        continue_on_error=False,
    )

    assert len(results) == 1
    assert results[0].model_name == "openrouter/pony-alpha"


def test_calc_dna_single_model_uses_cached_responses_without_metadata_or_generation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "llm_dna.core.extraction.get_probe_texts",
        lambda **_kwargs: ["prompt A", "prompt B", "prompt C"],
    )
    monkeypatch.setattr(
        api,
        "_load_model_metadata_for_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("metadata should not be loaded when single-model cache exists")
        ),
    )
    monkeypatch.setattr(
        "llm_dna.core.extraction.extract_dna_signature",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("core extraction should not run when single-model cache exists")
        ),
    )

    safe_model_name = "deepseek_deepseek-v3.2"
    cache_path = tmp_path / "out" / "rand" / safe_model_name / "responses.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        """{
  "model": "deepseek/deepseek-v3.2",
  "dataset": "rand",
  "count": 3,
  "items": [
    {"prompt": "prompt A", "response": "cached response 1"},
    {"prompt": "prompt B", "response": "cached response 2"},
    {"prompt": "prompt C", "response": "cached response 3"}
  ]
}""",
        encoding="utf-8",
    )

    seen: dict[str, object] = {}
    signature_obj = SimpleNamespace(metadata=SimpleNamespace(extractor_config={}))
    vector_obj = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    def fake_extract_signature(
        model_name,
        responses,
        config,
        model_meta,
        generation_device,
        sentence_encoder="all-mpnet-base-v2",
        encoder_device=None,
        probe_texts=None,
    ):
        del sentence_encoder
        seen["model_name"] = model_name
        seen["responses"] = list(responses)
        seen["config_model_name"] = config.model_name
        seen["model_meta"] = dict(model_meta)
        seen["generation_device"] = generation_device
        seen["encoder_device"] = encoder_device
        return signature_obj, vector_obj, 0.01

    monkeypatch.setattr(api, "_extract_signature_from_text_responses", fake_extract_signature)

    config = DNAExtractionConfig(
        model_name="deepseek/deepseek-v3.2",
        model_type="auto",
        dataset="rand",
        max_samples=3,
        dna_dim=4,
        save=False,
        output_dir=tmp_path / "out",
        device="cpu",
    )

    result = api.calc_dna(config)

    assert seen["model_name"] == "deepseek/deepseek-v3.2"
    assert seen["responses"] == ["cached response 1", "cached response 2", "cached response 3"]
    assert seen["config_model_name"] == "deepseek/deepseek-v3.2"
    assert seen["generation_device"] == "cpu"
    assert seen["encoder_device"] == "cpu"
    assert seen["model_meta"]["model_name"] == "deepseek/deepseek-v3.2"
    assert result.model_name == "deepseek/deepseek-v3.2"
    assert result.signature is signature_obj
    assert np.array_equal(result.vector, vector_obj)


def test_calc_dna_parallel_failure_behavior(monkeypatch, tmp_path):
    model_names = ["distilgpt2", "broken/model"]
    llm_list = _write_llm_list(tmp_path / "llm_list.txt", model_names)

    monkeypatch.setattr(
        "llm_dna.core.extraction.get_probe_texts",
        lambda **_kwargs: ["prompt A", "prompt B", "prompt C"],
    )
    monkeypatch.setattr(
        api,
        "_load_model_metadata_for_model",
        lambda *args, **kwargs: {
            "architecture": {"is_generative": True},
            "repository": {},
            "size": {},
            "chat_model": {"is_chat_model": False},
        },
    )

    def fake_generate(
        model_name: str,
        config: DNAExtractionConfig,
        model_meta,
        probe_texts,
        device: str,
        resolved_token,
        incremental_save_path=None,
    ):
        del incremental_save_path
        if model_name == "broken/model":
            raise RuntimeError("generation failed")
        return [f"{model_name}::{idx}" for idx in range(len(probe_texts))]

    monkeypatch.setattr(api, "_generate_responses_for_model", fake_generate)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeSentenceEncoder)

    config = DNAExtractionConfig(
        model_name="ignored-model-name",
        dataset="rand",
        max_samples=3,
        dna_dim=4,
        save=False,
        output_dir=tmp_path / "out",
        device="cpu",
    )

    with pytest.raises(RuntimeError):
        api.calc_dna_parallel(
            config=config,
            llm_list=llm_list,
            continue_on_error=False,
        )

    results = api.calc_dna_parallel(
        config=config,
        llm_list=llm_list,
        continue_on_error=True,
    )
    assert [result.model_name for result in results] == ["distilgpt2"]


def test_calc_dna_parallel_gpu_round_robin_distribution(monkeypatch, tmp_path):
    """Verify models are distributed across GPUs in round-robin order."""
    model_names = ["model-a", "model-b", "model-c", "model-d", "model-e"]
    llm_list = _write_llm_list(tmp_path / "llm_list.txt", model_names)
    
    model_device_map: dict[str, str] = {}

    monkeypatch.setattr(
        "llm_dna.core.extraction.get_probe_texts",
        lambda **_kwargs: ["prompt 1", "prompt 2"],
    )
    monkeypatch.setattr(
        api,
        "_load_model_metadata_for_model",
        lambda *args, **kwargs: {
            "architecture": {"is_generative": True},
            "repository": {},
            "size": {},
            "chat_model": {"is_chat_model": False},
        },
    )

    def fake_generate(model_name, config, model_meta, probe_texts, device, resolved_token, incremental_save_path=None):
        del incremental_save_path
        model_device_map[model_name] = device
        return [f"{model_name}::resp::{i}" for i in range(len(probe_texts))]

    monkeypatch.setattr(api, "_generate_responses_for_model", fake_generate)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeSentenceEncoder)

    config = DNAExtractionConfig(
        model_name="ignored",
        dataset="rand",
        max_samples=2,
        dna_dim=4,
        save=False,
        output_dir=tmp_path / "out",
        device="cuda",
    )

    results = api.calc_dna_parallel(
        config=config,
        llm_list=llm_list,
        gpu_ids=[0, 1, 2],
        continue_on_error=False,
    )

    assert len(results) == 5
    # All 3 GPUs should be used
    devices_used = set(model_device_map.values())
    assert devices_used == {"cuda:0", "cuda:1", "cuda:2"}


def test_calc_dna_parallel_single_gpu_multiple_models(monkeypatch, tmp_path):
    """Test that multiple models work with a single GPU."""
    model_names = ["model-1", "model-2", "model-3"]
    llm_list = _write_llm_list(tmp_path / "llm_list.txt", model_names)
    devices_seen: list[str] = []

    monkeypatch.setattr(
        "llm_dna.core.extraction.get_probe_texts",
        lambda **_kwargs: ["prompt A", "prompt B"],
    )
    monkeypatch.setattr(
        api,
        "_load_model_metadata_for_model",
        lambda *args, **kwargs: {
            "architecture": {"is_generative": True},
            "repository": {},
            "size": {},
            "chat_model": {"is_chat_model": False},
        },
    )

    def fake_generate(model_name, config, model_meta, probe_texts, device, resolved_token, incremental_save_path=None):
        del incremental_save_path
        devices_seen.append(device)
        return [f"{model_name}::resp" for _ in probe_texts]

    monkeypatch.setattr(api, "_generate_responses_for_model", fake_generate)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeSentenceEncoder)

    config = DNAExtractionConfig(
        model_name="ignored",
        dataset="rand",
        max_samples=2,
        dna_dim=4,
        save=False,
        output_dir=tmp_path / "out",
        device="cuda",
    )

    results = api.calc_dna_parallel(
        config=config,
        llm_list=llm_list,
        gpu_ids=[0],  # Single GPU
    )

    assert len(results) == 3
    assert all(d == "cuda:0" for d in devices_seen)


def test_calc_dna_parallel_empty_llm_list_raises(monkeypatch, tmp_path):
    """Empty LLM list file should raise an error."""
    llm_list = tmp_path / "empty_list.txt"
    llm_list.write_text("# only comments\n\n", encoding="utf-8")

    config = DNAExtractionConfig(
        model_name="ignored",
        dataset="rand",
        max_samples=5,
        dna_dim=4,
        save=False,
        output_dir=tmp_path / "out",
    )

    with pytest.raises(ValueError, match="No valid model names"):
        api.calc_dna_parallel(config=config, llm_list=llm_list)


def test_calc_dna_parallel_model_with_slashes_in_path(monkeypatch, tmp_path):
    """Models with slashes in name should be handled for cache paths."""
    model_names = ["org/model-name", "another-org/sub/deep-model"]
    llm_list = _write_llm_list(tmp_path / "llm_list.txt", model_names)

    monkeypatch.setattr(
        "llm_dna.core.extraction.get_probe_texts",
        lambda **_kwargs: ["prompt A", "prompt B"],
    )
    monkeypatch.setattr(
        api,
        "_load_model_metadata_for_model",
        lambda *args, **kwargs: {
            "architecture": {"is_generative": True},
            "repository": {},
            "size": {},
            "chat_model": {"is_chat_model": False},
        },
    )

    def fake_generate(model_name, config, model_meta, probe_texts, device, resolved_token, incremental_save_path=None):
        del incremental_save_path
        return [f"{model_name}::resp::{i}" for i in range(len(probe_texts))]

    monkeypatch.setattr(api, "_generate_responses_for_model", fake_generate)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeSentenceEncoder)

    config = DNAExtractionConfig(
        model_name="ignored",
        dataset="rand",
        max_samples=2,
        dna_dim=4,
        save=False,
        output_dir=tmp_path / "out",
        device="cpu",
    )

    results = api.calc_dna_parallel(config=config, llm_list=llm_list)
    
    assert len(results) == 2
    assert results[0].model_name == "org/model-name"
    assert results[1].model_name == "another-org/sub/deep-model"


def test_calc_dna_parallel_min_samples(monkeypatch, tmp_path):
    """Test extraction with minimum samples (max_samples=2)."""
    model_names = ["tiny-model"]
    llm_list = _write_llm_list(tmp_path / "llm_list.txt", model_names)

    monkeypatch.setattr(
        "llm_dna.core.extraction.get_probe_texts",
        lambda **_kwargs: ["p1", "p2"],
    )
    monkeypatch.setattr(
        api,
        "_load_model_metadata_for_model",
        lambda *args, **kwargs: {
            "architecture": {"is_generative": True},
            "repository": {},
            "size": {},
            "chat_model": {"is_chat_model": False},
        },
    )

    def fake_generate(model_name, config, model_meta, probe_texts, device, resolved_token, incremental_save_path=None):
        del incremental_save_path
        return ["resp1", "resp2"]

    monkeypatch.setattr(api, "_generate_responses_for_model", fake_generate)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeSentenceEncoder)

    config = DNAExtractionConfig(
        model_name="ignored",
        dataset="rand",
        max_samples=2,  # Minimum
        dna_dim=4,
        save=False,
        output_dir=tmp_path / "out",
        device="cpu",
    )

    results = api.calc_dna_parallel(config=config, llm_list=llm_list)
    assert len(results) == 1
    assert results[0].vector.shape == (4,)


def test_calc_dna_parallel_encoder_device_auto(monkeypatch, tmp_path):
    """Verify encoder device is set correctly when encoder_device='auto'."""
    model_names = ["test-model"]
    llm_list = _write_llm_list(tmp_path / "llm_list.txt", model_names)
    encoder_device_captured: list[str] = []

    class _CapturingEncoder:
        def __init__(self, *_args, **kwargs):
            encoder_device_captured.append(kwargs.get("device", "not_set"))

        def encode(self, texts, **kwargs):
            # Return varied embeddings to avoid all-zero reduction
            embeddings = []
            for i, text in enumerate(texts):
                embeddings.append([float(i + 1) * j for j in range(1, 7)])
            return np.asarray(embeddings, dtype=np.float32)

    monkeypatch.setattr(
        "llm_dna.core.extraction.get_probe_texts",
        lambda **_kwargs: ["prompt1", "prompt2"],
    )
    monkeypatch.setattr(
        api,
        "_load_model_metadata_for_model",
        lambda *args, **kwargs: {
            "architecture": {"is_generative": True},
            "repository": {},
            "size": {},
            "chat_model": {"is_chat_model": False},
        },
    )
    monkeypatch.setattr(
        api,
        "_generate_responses_for_model",
        lambda *args, **kwargs: ["response1", "response2"],
    )
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _CapturingEncoder)

    config = DNAExtractionConfig(
        model_name="ignored",
        dataset="rand",
        max_samples=2,
        dna_dim=4,
        save=False,
        output_dir=tmp_path / "out",
        device="cuda",
    )

    api.calc_dna_parallel(
        config=config,
        llm_list=llm_list,
        gpu_ids=[3],
        encoder_device="auto",
    )

    # encoder_device='auto' should use first generation device
    assert encoder_device_captured == ["cuda:3"]


def test_calc_dna_parallel_max_samples_too_small_raises(tmp_path):
    """max_samples < 2 should raise an error."""
    llm_list = _write_llm_list(tmp_path / "llm_list.txt", ["model"])

    config = DNAExtractionConfig(
        model_name="ignored",
        dataset="rand",
        max_samples=1,
        dna_dim=4,
        save=False,
        output_dir=tmp_path / "out",
    )

    with pytest.raises(ValueError, match="max_samples >= 2"):
        api.calc_dna_parallel(config=config, llm_list=llm_list)


def test_calc_dna_parallel_api_uses_n_processes_workers(monkeypatch, tmp_path):
    model_names = ["openrouter/a", "openrouter/b", "openrouter/c", "openrouter/d", "openrouter/e"]
    llm_list = _write_llm_list(tmp_path / "llm_list.txt", model_names)

    monkeypatch.setattr(
        "llm_dna.core.extraction.get_probe_texts",
        lambda **_kwargs: ["prompt A", "prompt B"],
    )
    monkeypatch.setattr(
        api,
        "_load_model_metadata_for_model",
        lambda *args, **kwargs: {
            "architecture": {"is_generative": True},
            "repository": {},
            "size": {},
            "chat_model": {"is_chat_model": False},
        },
    )

    lock = threading.Lock()
    active_workers = 0
    peak_active_workers = 0
    thread_ids: set[int] = set()

    def fake_generate(model_name, config, model_meta, probe_texts, device, resolved_token, incremental_save_path=None):
        del incremental_save_path
        del config, model_meta, probe_texts, device, resolved_token
        nonlocal active_workers, peak_active_workers
        with lock:
            thread_ids.add(threading.get_ident())
            active_workers += 1
            peak_active_workers = max(peak_active_workers, active_workers)
        time.sleep(0.05)
        with lock:
            active_workers -= 1
        return [f"{model_name}::resp::0", f"{model_name}::resp::1"]

    monkeypatch.setattr(api, "_generate_responses_for_model", fake_generate)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeSentenceEncoder)

    config = DNAExtractionConfig(
        model_name="ignored",
        model_type="openrouter",
        dataset="rand",
        max_samples=2,
        dna_dim=4,
        save=False,
        output_dir=tmp_path / "out",
        device="cpu",
    )

    results = api.calc_dna_parallel(
        config=config,
        llm_list=llm_list,
        n_processes=3,
        continue_on_error=False,
    )

    assert len(results) == len(model_names)
    assert len(thread_ids) == 3
    assert peak_active_workers >= 2
