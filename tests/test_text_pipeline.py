"""Offline integration checks for shared single-model and batch text DNA."""

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

import llm_dna.api as api


class DeterministicTextEncoder:
    """Embedding depends only on text, never its position in an encoding batch."""

    def __init__(self, *_args, **_kwargs):
        pass

    def encode(self, texts, *, normalize_embeddings, **_kwargs):
        assert normalize_embeddings is False
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")
            rows.append(np.random.RandomState(seed).normal(size=96) * (1 + seed % 11))
        return np.asarray(rows, dtype=np.float32)


@pytest.fixture
def offline_pipeline(monkeypatch, tmp_path):
    prompts = ["prompt A", "prompt B", "prompt C"]
    generated = []
    responses = {}
    monkeypatch.setattr("llm_dna.core.extraction.get_probe_texts", lambda **_: list(prompts))
    monkeypatch.setattr(api, "_resolve_hf_token", lambda _: None)
    monkeypatch.setattr(api, "_load_model_metadata_for_model", lambda *_, **__: {
        "architecture": {"is_generative": True}, "repository": {},
    })
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", DeterministicTextEncoder)

    def generate(model_name, probe_texts, **_kwargs):
        generated.append(model_name)
        return responses.get(model_name, [f"{model_name}::{prompt}" for prompt in probe_texts])

    monkeypatch.setattr(api, "_generate_responses_for_model", generate)
    config = api.DNAExtractionConfig(
        model_name="example/model-a", model_type="huggingface", dataset="rand",
        max_samples=3, save=False, output_dir=tmp_path, device="cpu",
    )
    return config, prompts, generated, responses


@pytest.mark.parametrize("normalize", [True, False])
def test_single_batch_and_parallel_have_same_coordinates(offline_pipeline, tmp_path, normalize):
    config, prompts, _, responses = offline_pipeline
    config = replace(config, normalize_embeddings=normalize)
    names = [config.model_name, "example/model-b"]
    # A missing answer in the first model must not shift either model's probe slots.
    responses[names[0]] = ["answer one", "", "answer three"]
    configs = [replace(config, model_name=name) for name in names]
    singles = [api.calc_dna(item) for item in configs]
    sequential = api.calc_dna_batch(configs)
    model_list = tmp_path / "models.txt"
    model_list.write_text("\n".join(names) + "\n", encoding="utf-8")
    parallel = api.calc_dna_parallel(config, llm_list=model_list, use_response_cache=False)

    for single, batch, shared in zip(singles, sequential, parallel):
        np.testing.assert_array_equal(single.vector, batch.vector)
        np.testing.assert_array_equal(single.vector, shared.vector)
        left = single.signature.metadata.extractor_config
        right = shared.signature.metadata.extractor_config
        assert left["projection_sha256"] == right["projection_sha256"]
        assert left["probe_order_sha256"] == right["probe_order_sha256"]
        assert left["projection_shape"] == [len(prompts) * 64, 128]
        assert left["normalize_embeddings"] == normalize
    assert singles[0].signature.metadata.extractor_config["zero_placeholder_indices"] == [1]


def test_last_answer_and_normalization_flag_change_public_api_output(offline_pipeline):
    config, prompts, _, responses = offline_pipeline
    config = replace(config, normalize_embeddings=True)
    baseline = api.calc_dna(config).vector
    responses[config.model_name] = [f"{config.model_name}::{prompt}" for prompt in prompts]
    responses[config.model_name][-1] = "changed final answer"

    changed = api.calc_dna(config).vector
    unnormalized = api.calc_dna(replace(config, normalize_embeddings=False)).vector

    assert not np.allclose(baseline, changed)
    assert not np.allclose(changed, unnormalized)
    raw = DeterministicTextEncoder().encode(responses[config.model_name], normalize_embeddings=False)
    prefix = raw[:, :64]
    matrix = np.random.RandomState(42).randn(prefix.size, 128).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=0)
    np.testing.assert_array_equal(unnormalized, (prefix.reshape(1, -1) @ matrix).reshape(-1))


@pytest.mark.parametrize("run", [api.calc_dna, api.calc_dna_parallel])
def test_invalid_per_model_reducer_is_rejected_before_generation(offline_pipeline, run):
    config, _, generated, _ = offline_pipeline
    with pytest.raises(ValueError, match="shared coordinates"):
        run(replace(config, reduction_method="pca"))
    assert generated == []


@pytest.mark.parametrize("run", [api.calc_dna, api.calc_dna_parallel])
def test_legacy_extractor_is_rejected_before_probe_loading(offline_pipeline, monkeypatch, run):
    config, _, generated, _ = offline_pipeline

    def unexpected_probes(**_kwargs):
        pytest.fail("An unsupported extractor must not load probes or generate responses")

    monkeypatch.setattr("llm_dna.core.extraction.get_probe_texts", unexpected_probes)
    with pytest.raises(ValueError, match="Only 'text' is supported"):
        run(replace(config, extractor_type="embedding"))
    assert generated == []


def test_saved_summary_identifies_text_extraction(offline_pipeline):
    config, _, _, _ = offline_pipeline
    result = api.calc_dna(replace(config, save=True))
    summary = json.loads(result.summary_path.read_text())
    assert summary["extractor_type"] == "text"
    assert summary["config"]["extractor_type"] == "text"


@pytest.mark.parametrize("run", [api.calc_dna, api.calc_dna_parallel])
def test_saved_summary_omits_explicit_token(offline_pipeline, run):
    config, _, _, _ = offline_pipeline
    config = replace(config, save=True, token="hf_example_secret_for_summary_test")

    result = run(config)
    if isinstance(result, list):
        result = result[0]

    summary_text = result.summary_path.read_text(encoding="utf-8")
    summary = json.loads(summary_text)
    assert "token" not in summary["config"]
    assert config.token not in summary_text
    assert config.token == "hf_example_secret_for_summary_test"


def test_only_text_extractor_is_exported():
    import importlib.util
    import llm_dna
    import llm_dna.dna as dna

    assert dna.TextDNAExtractor is llm_dna.TextDNAExtractor
    for name in ("DNAExtractor", "InferenceExtractor", "ParamExtractor", "EmbeddingDNAExtractor"):
        assert name not in dna.__all__
        assert not hasattr(dna, name)
    assert importlib.util.find_spec("llm_dna.dna.EmbeddingDNAExtractor") is None
    assert importlib.util.find_spec("llm_dna.dna.DNAExtractor") is None


@pytest.mark.parametrize("cache_variant", ["wrong_count", "wrong_order", "missing_order"])
def test_stale_response_cache_is_regenerated_for_current_probe_order(offline_pipeline, cache_variant):
    config, prompts, generated, _ = offline_pipeline
    items = [{"prompt": prompt, "response": "stale answer"} for prompt in prompts]
    if cache_variant == "wrong_count":
        items = items[:-1]
    elif cache_variant == "wrong_order":
        items = items[::-1]
    payload = ["stale answer"] * 3 if cache_variant == "missing_order" else {
        "model": config.model_name, "items": items,
    }
    path = api._response_cache_path(config, config.model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = api.calc_dna(config)

    assert generated == [config.model_name]
    assert result.signature.metadata.probe_count == 3
    assert result.signature.metadata.extractor_config["projection_shape"] == [192, 128]


def _expected_generation_config(config):
    return {
        "max_length": config.max_length,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "use_chat_template": config.use_chat_template,
        "do_sample": config.temperature > 0,
    }


def _write_cached_answers(config, prompts, *, generation_config=None, model_name=None):
    path = api._response_cache_path(config, config.model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_name or config.model_name,
        "dataset": config.dataset,
        "items": [{"prompt": prompt, "response": f"cached {prompt}"} for prompt in prompts],
    }
    if generation_config is not None:
        payload["generation_config"] = generation_config
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("run", [api.calc_dna, api.calc_dna_parallel])
@pytest.mark.parametrize("cache_variant", ["incomplete", "all_empty"])
def test_unusable_response_cache_is_regenerated(offline_pipeline, run, cache_variant):
    config, prompts, generated, _ = offline_pipeline
    config = replace(config, save=True)
    path = _write_cached_answers(
        config, prompts, generation_config=_expected_generation_config(config),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["complete"] = cache_variant != "incomplete"
    if cache_variant == "all_empty":
        for item, response in zip(payload["items"], ["", " ", "\n\t"]):
            item["response"] = response
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = run(config)
    if isinstance(result, list):
        result = result[0]

    assert generated == [config.model_name]
    provenance = result.signature.metadata.extractor_config["provenance"]
    assert provenance["response_source"] == "generated"
    saved_cache = json.loads(path.read_text(encoding="utf-8"))
    assert saved_cache["complete"] is True
    assert [item["response"] for item in saved_cache["items"]] == [
        f"{config.model_name}::{prompt}" for prompt in prompts
    ]


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("recorded", [False, True])
def test_cached_generation_provenance_distinguishes_known_and_legacy_settings(
    offline_pipeline, parallel, recorded,
):
    config, prompts, generated, _ = offline_pipeline
    settings = _expected_generation_config(config)
    _write_cached_answers(config, prompts, generation_config=settings if recorded else None)

    result = api.calc_dna_parallel(config)[0] if parallel else api.calc_dna(config)

    assert generated == []
    provenance = result.signature.metadata.extractor_config["provenance"]
    assert provenance["response_source"] == "cache"
    assert provenance["requested_generation_config"] == settings
    assert provenance["generation_config"] == (settings if recorded else None)
    assert provenance["generation_config_status"] == ("recorded" if recorded else "unknown_legacy_cache")


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("mismatch", ["model", "generation_settings"])
def test_known_model_or_generation_setting_mismatch_forces_regeneration(
    offline_pipeline, parallel, mismatch,
):
    config, prompts, generated, _ = offline_pipeline
    settings = _expected_generation_config(config)
    mismatched = dict(settings, use_chat_template=not config.use_chat_template)
    _write_cached_answers(
        config, prompts,
        generation_config=mismatched if mismatch == "generation_settings" else settings,
        model_name="example/wrong-model" if mismatch == "model" else config.model_name,
    )

    result = api.calc_dna_parallel(config)[0] if parallel else api.calc_dna(config)

    assert generated == [config.model_name]
    provenance = result.signature.metadata.extractor_config["provenance"]
    assert provenance["response_source"] == "generated"
    assert provenance["generation_config"] == settings
    assert provenance["generation_config_status"] == "recorded"


def test_generated_response_cache_persists_actual_generation_settings(offline_pipeline):
    config, _, generated, _ = offline_pipeline
    config = replace(config, save=True, use_chat_template=False, temperature=0.0)

    result = api.calc_dna(config)

    cache = json.loads(api._response_cache_path(config, config.model_name).read_text(encoding="utf-8"))
    assert generated == [config.model_name]
    assert cache["complete"] is True
    assert cache["generation_config"] == _expected_generation_config(config)
    assert cache["generation_config"]["do_sample"] is False
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    provenance = saved["metadata"]["extractor_config"]["provenance"]
    assert provenance["generation_config"] == cache["generation_config"]
    assert provenance["response_source"] == "generated"


@pytest.mark.parametrize("continue_on_error", [False, True])
def test_parallel_empty_model_responses_follow_continue_on_error(
    offline_pipeline, tmp_path, continue_on_error,
):
    config, _, _, responses = offline_pipeline
    responses[config.model_name] = ["", " ", ""]
    valid_model = "example/valid-model"
    model_list = tmp_path / "models.txt"
    model_list.write_text(f"{config.model_name}\n{valid_model}\n", encoding="utf-8")

    if continue_on_error:
        result = api.calc_dna_parallel(
            config, llm_list=model_list, continue_on_error=True, use_response_cache=False,
        )
        assert [item.model_name for item in result] == [valid_model]
    else:
        with pytest.raises(RuntimeError, match="failed for model"):
            api.calc_dna_parallel(
                config, llm_list=model_list, continue_on_error=False, use_response_cache=False,
            )
