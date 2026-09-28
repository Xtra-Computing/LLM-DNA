"""Cache bypass must work through every current public execution path."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import llm_dna.api as api
import llm_dna.cli as cli


@pytest.fixture
def cached_run(monkeypatch, tmp_path):
    prompts = ["first prompt", "second prompt"]
    generated = []
    monkeypatch.setattr("llm_dna.core.extraction.get_probe_texts", lambda **_: prompts)
    monkeypatch.setattr(api, "_resolve_hf_token", lambda _: None)
    monkeypatch.setattr(api, "_load_model_metadata_for_model", lambda *_, **__: {
        "architecture": {"is_generative": True}, "repository": {},
    })

    class Encoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            return np.array([[len(text), text.count("e"), 1] for text in texts], dtype=np.float32)

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", Encoder)

    def generate(model_name, **kwargs):
        generated.append(model_name)
        return ["fresh first answer", "fresh second answer"]

    monkeypatch.setattr(api, "_generate_responses_for_model", generate)
    config = api.DNAExtractionConfig(
        model_name="example/model", model_type="huggingface", dataset="rand",
        max_samples=2, output_dir=tmp_path, device="cpu", pre_agg_embed_dim=3,
    )
    path = api._response_cache_path(config, config.model_name)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "model": config.model_name, "complete": True,
        "items": [{"prompt": prompt, "response": "cached answer"} for prompt in prompts],
        "generation_config": api._generation_config(config),
    }), encoding="utf-8")
    return config, path, generated


@pytest.mark.parametrize("mode", ["single", "sequential", "parallel"])
@pytest.mark.parametrize("reuse", [True, False])
@pytest.mark.parametrize("save", [True, False])
def test_cache_reuse_config_preserves_existing_save_policy(cached_run, mode, reuse, save):
    config, path, generated = cached_run
    config.use_response_cache = reuse
    config.save = save
    before = path.read_bytes()

    if mode == "single":
        result = api.calc_dna(config)
    elif mode == "sequential":
        result = api.calc_dna_batch([config])[0]
    else:
        result = api.calc_dna_parallel(config)[0]

    assert generated == ([] if reuse else [config.model_name])
    provenance = result.signature.metadata.extractor_config["provenance"]
    assert provenance["response_source"] == ("cache" if reuse else "generated")
    assert (result.output_path is not None) == save
    # The parallel API historically controls response-cache writes separately
    # from signature saving; the new reuse setting must not change that policy.
    if not reuse and (save or mode == "parallel"):
        payload = json.loads(path.read_text())
        assert [item["response"] for item in payload["items"]] == [
            "fresh first answer", "fresh second answer",
        ]
        assert payload["complete"] is True
        assert payload["generation_config"] == api._generation_config(config)
    else:
        assert path.read_bytes() == before


@pytest.mark.parametrize("reuse", [True, False])
def test_parallel_cache_io_switch_takes_precedence(cached_run, reuse):
    config, path, generated = cached_run
    config.use_response_cache = reuse
    before = path.read_bytes()

    result = api.calc_dna_parallel(config, use_response_cache=False)[0]

    assert generated == [config.model_name]
    assert path.read_bytes() == before
    assert result.signature.metadata.extractor_config["provenance"]["response_source"] == "generated"


@pytest.mark.parametrize("entrypoint", ["cli", "script"])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("ignore", [False, True])
def test_cache_option_reaches_single_and_batch_entrypoints(monkeypatch, tmp_path, entrypoint, batch, ignore):
    captured = []

    def extract(config):
        captured.append(config)
        return SimpleNamespace(
            model_name=config.model_name, vector=np.ones(2), output_path=None,
            summary_path=None, elapsed_seconds=0.0,
        )

    def parallel(config, **kwargs):
        return [extract(config), extract(config)]

    models = tmp_path / "models.txt"
    models.write_text("example/one\nexample/two\n")
    args = ["--llm-list", str(models)] if batch else []
    if ignore:
        args.append("--ignore-response-cache")
    if entrypoint == "cli":
        monkeypatch.setattr(api, "calc_dna", extract)
        monkeypatch.setattr(api, "calc_dna_parallel", parallel)
        monkeypatch.setattr(cli, "load_dotenv", lambda **kwargs: None)
        if not batch:
            args.extend(["--model-name", "example/one"])
        assert cli.main(args) == 0
    else:
        script = Path(__file__).resolve().parents[1] / "scripts" / "calc_dna.py"
        spec = importlib.util.spec_from_file_location("calc_dna_example", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setattr(module, "calc_dna", extract)
        monkeypatch.setattr(module, "calc_dna_parallel", parallel)
        monkeypatch.setattr(sys, "argv", [str(script), *args])
        assert module.main() == 0

    assert len(captured) == (2 if batch else 1)
    assert all(config.use_response_cache is not ignore for config in captured)
    assert all(config.extractor_type == "text" for config in captured)
