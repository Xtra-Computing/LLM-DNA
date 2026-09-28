from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import llm_dna.api as api
from llm_dna.api import DNAExtractionConfig
from llm_dna.cli import main as cli_main


def _fake_result(model_name: str, dim: int) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=model_name,
        vector=np.ones(dim, dtype=np.float32),
        output_path=None,
    )


def test_cli_multi_model_gpu_round_robin(monkeypatch, capsys):
    captured_configs: list[DNAExtractionConfig] = []

    def fake_calc_dna(config: DNAExtractionConfig):
        captured_configs.append(config)
        return _fake_result(model_name=config.model_name, dim=config.dna_dim)

    monkeypatch.setattr(api, "calc_dna", fake_calc_dna)

    exit_code = cli_main(
        [
            "--model-name",
            "distilgpt2",
            "--model-name",
            "gpt2",
            "--model-name",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--gpus",
            "0,1",
            "--device",
            "cuda",
            "--no-save",
            "--print-vector",
        ]
    )

    assert exit_code == 0
    assert [cfg.model_name for cfg in captured_configs] == [
        "distilgpt2",
        "gpt2",
        "Qwen/Qwen2.5-0.5B-Instruct",
    ]
    assert [cfg.gpu_id for cfg in captured_configs] == [0, 1, 0]

    for cfg in captured_configs:
        assert cfg.save is False
        assert cfg.output_path is None

    stdout = capsys.readouterr().out
    assert stdout.count("model=") == 3


def test_cli_hyperparameter_variation_is_forwarded(monkeypatch, tmp_path):
    captured_configs: list[DNAExtractionConfig] = []
    output_path = tmp_path / "single_model_output.json"

    def fake_calc_dna(config: DNAExtractionConfig):
        captured_configs.append(config)
        return _fake_result(model_name=config.model_name, dim=config.dna_dim)

    monkeypatch.setattr(api, "calc_dna", fake_calc_dna)

    exit_code = cli_main(
        [
            "--model-name",
            "distilgpt2",
            "--dataset",
            "rand",
            "--max-samples",
            "16",
            "--dna-dim",
            "64",
            "--reduction-method",
            "random_projection",
            "--embedding-merge",
            "mean",
            "--max-length",
            "256",
            "--load-in-4bit",
            "--output-path",
            str(output_path),
            "--no-save",
        ]
    )

    assert exit_code == 0
    assert len(captured_configs) == 1
    cfg = captured_configs[0]
    assert cfg.model_name == "distilgpt2"
    assert cfg.dataset == "rand"
    assert cfg.max_samples == 16
    assert cfg.dna_dim == 64
    assert cfg.reduction_method == "random_projection"
    assert cfg.embedding_merge == "mean"
    assert cfg.max_length == 256
    assert cfg.load_in_4bit is True
    assert cfg.save is False
    # Explicit output path should be preserved for single-model runs.
    assert cfg.output_path == output_path


def test_cli_model_parameters_are_forwarded(monkeypatch):
    captured_configs: list[DNAExtractionConfig] = []

    def fake_calc_dna(config: DNAExtractionConfig):
        captured_configs.append(config)
        return _fake_result(model_name=config.model_name, dim=config.dna_dim)

    monkeypatch.setattr(api, "calc_dna", fake_calc_dna)

    exit_code = cli_main(
        [
            "--model-name",
            "gpt2",
            "--model-path",
            "/tmp/local-model-path",
            "--model-type",
            "huggingface",
            "--no-save",
        ]
    )

    assert exit_code == 0
    assert len(captured_configs) == 1
    cfg = captured_configs[0]
    assert cfg.model_name == "gpt2"
    assert cfg.model_path == "/tmp/local-model-path"
    assert cfg.model_type == "huggingface"


def test_cli_llm_list_batch_mode_uses_parallel_api(monkeypatch, tmp_path):
    llm_list_path = tmp_path / "llm_list.txt"
    llm_list_path.write_text("distilgpt2\ngpt2\n", encoding="utf-8")
    captured_kwargs = {}

    def fake_calc_dna_parallel(**kwargs):
        captured_kwargs.update(kwargs)
        return [
            _fake_result(model_name="distilgpt2", dim=16),
            _fake_result(model_name="gpt2", dim=16),
        ]

    monkeypatch.setattr(api, "calc_dna_parallel", fake_calc_dna_parallel)

    exit_code = cli_main(
        [
            "--llm-list",
            str(llm_list_path),
            "--gpus",
            "0,1",
            "--no-save",
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["llm_list"] == llm_list_path
    assert captured_kwargs["gpu_ids"] == [0, 1]
    assert captured_kwargs["continue_on_error"] is False
    assert isinstance(captured_kwargs["config"], DNAExtractionConfig)


def test_cli_continue_on_error_runs_remaining_models(monkeypatch):
    attempted_models: list[str] = []

    def fake_calc_dna(config: DNAExtractionConfig):
        attempted_models.append(config.model_name)
        if config.model_name == "broken/model":
            raise RuntimeError("simulated failure")
        return _fake_result(model_name=config.model_name, dim=8)

    monkeypatch.setattr(api, "calc_dna", fake_calc_dna)

    exit_code = cli_main(
        [
            "--model-name",
            "distilgpt2",
            "--model-name",
            "broken/model",
            "--model-name",
            "gpt2",
            "--continue-on-error",
        ]
    )

    assert exit_code == 1
    assert attempted_models == ["distilgpt2", "broken/model", "gpt2"]


def test_calc_dna_batch_assigns_gpu_round_robin_and_preserves_hyperparams(monkeypatch):
    called_configs: list[DNAExtractionConfig] = []

    def fake_calc_dna(config: DNAExtractionConfig):
        called_configs.append(config)
        return _fake_result(model_name=config.model_name, dim=config.dna_dim)

    monkeypatch.setattr(api, "calc_dna", fake_calc_dna)

    configs = [
        DNAExtractionConfig(
            model_name="distilgpt2",
            dataset="rand",
            dna_dim=32,
            reduction_method="random_projection",
            embedding_merge="sum",
            max_samples=10,
        ),
        DNAExtractionConfig(
            model_name="gpt2",
            dataset="rand",
            dna_dim=64,
            reduction_method="pca",
            embedding_merge="concat",
            max_samples=20,
            gpu_id=7,
        ),
        DNAExtractionConfig(
            model_name="tiiuae/falcon-rw-1b",
            dataset="rand",
            dna_dim=96,
            reduction_method="svd",
            embedding_merge="max",
            max_samples=30,
        ),
        DNAExtractionConfig(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            dataset="rand",
            dna_dim=128,
            reduction_method="random_projection",
            embedding_merge="mean",
            max_samples=40,
        ),
    ]

    results = api.calc_dna_batch(configs=configs, gpu_ids=[2, 3])

    assert len(results) == 4
    assert [cfg.gpu_id for cfg in called_configs] == [2, 7, 2, 3]
    assert [cfg.model_name for cfg in called_configs] == [
        "distilgpt2",
        "gpt2",
        "tiiuae/falcon-rw-1b",
        "Qwen/Qwen2.5-0.5B-Instruct",
    ]
    assert [cfg.dna_dim for cfg in called_configs] == [32, 64, 96, 128]
    assert [cfg.reduction_method for cfg in called_configs] == [
        "random_projection",
        "pca",
        "svd",
        "random_projection",
    ]
    assert [cfg.embedding_merge for cfg in called_configs] == [
        "sum",
        "concat",
        "max",
        "mean",
    ]
    assert [cfg.max_samples for cfg in called_configs] == [10, 20, 30, 40]


def test_calc_dna_batch_continue_on_error_returns_successful_results(monkeypatch):
    attempted_models: list[str] = []

    def fake_calc_dna(config: DNAExtractionConfig):
        attempted_models.append(config.model_name)
        if config.model_name == "broken/model":
            raise RuntimeError("simulated failure")
        return _fake_result(model_name=config.model_name, dim=config.dna_dim)

    monkeypatch.setattr(api, "calc_dna", fake_calc_dna)

    configs = [
        DNAExtractionConfig(model_name="distilgpt2", dna_dim=16),
        DNAExtractionConfig(model_name="broken/model", dna_dim=16),
        DNAExtractionConfig(model_name="gpt2", dna_dim=16),
    ]

    results = api.calc_dna_batch(configs=configs, gpu_ids=[0, 1], continue_on_error=True)

    assert attempted_models == ["distilgpt2", "broken/model", "gpt2"]
    assert [result.model_name for result in results] == ["distilgpt2", "gpt2"]
