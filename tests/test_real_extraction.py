"""Integration tests with real model loading and DNA extraction.

These tests actually load models and perform inference, so they are slower
and require GPU/CPU resources. Mark with @pytest.mark.slow to allow skipping.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download

from llm_dna import DNAExtractionConfig, calc_dna

MODEL_NAME = "distilgpt2"
MODEL_DEVICE = "cuda:0"


def _real_config(**overrides) -> DNAExtractionConfig:
    base = {
        "model_name": MODEL_NAME,
        "dataset": "rand",
        "max_samples": 5,
        "dna_dim": 32,
        "device": MODEL_DEVICE,
        "save": False,
        "temperature": 0.0,  # These integration tests check deterministic generation.
    }
    base.update(overrides)
    return DNAExtractionConfig(**base)


def _assert_valid_vector(vector: np.ndarray, dim: int) -> None:
    assert vector is not None
    assert vector.shape == (dim,)
    assert np.isfinite(vector).all()
    assert not np.allclose(vector, 0.0)


@pytest.mark.slow
class TestRealModelExecution:
    """Tests that actually load and execute models."""

    @pytest.fixture(scope="class", autouse=True)
    def require_cuda_and_local_model_cache(self):
        if not torch.cuda.is_available():
            pytest.skip("Real extraction CUDA tests require an available GPU.")

        try:
            snapshot_path = Path(snapshot_download(repo_id=MODEL_NAME, local_files_only=True))
        except Exception as exc:
            pytest.skip(
                f"Real extraction tests require local Hugging Face cache for {MODEL_NAME}: {exc}"
            )

        has_weights = any(snapshot_path.glob("*.safetensors")) or any(
            snapshot_path.glob("pytorch_model*.bin")
        )
        if not has_weights:
            pytest.skip(
                f"Cached snapshot for {MODEL_NAME} is incomplete (missing local model weights)."
            )

    def test_distilgpt2_real_extraction(self):
        """Test real DNA extraction with distilgpt2 (small, fast model)."""
        config = _real_config(dna_dim=64)

        result = calc_dna(config)

        # Verify result structure
        _assert_valid_vector(result.vector, dim=64)
        assert result.elapsed_seconds > 0
        assert result.model_name == MODEL_NAME
        assert result.dataset == "rand"

    @pytest.mark.parametrize("dim", [32, 64, 128])
    def test_different_dna_dimensions(self, dim):
        """Test extraction with different DNA dimension sizes."""
        result = calc_dna(_real_config(dna_dim=dim))
        _assert_valid_vector(result.vector, dim=dim)

    @pytest.mark.parametrize("normalize", [True, False])
    def test_normalization_settings(self, normalize):
        """Test the two supported normalization variants."""
        result = calc_dna(_real_config(normalize_embeddings=normalize))
        _assert_valid_vector(result.vector, dim=32)

    @pytest.mark.parametrize("samples", [3, 10, 25])
    def test_different_sample_counts(self, samples):
        """Test extraction with different sample counts."""
        result = calc_dna(_real_config(max_samples=samples))
        _assert_valid_vector(result.vector, dim=32)

    def test_extraction_is_deterministic(self):
        """Test that extraction with same seed produces same result."""
        config1 = _real_config(random_seed=42)
        config2 = _real_config(random_seed=42)

        result1 = calc_dna(config1)
        result2 = calc_dna(config2)

        np.testing.assert_allclose(result1.vector, result2.vector, rtol=1e-4, atol=1e-5)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        seeds = [42, 123, 999]
        vectors = [calc_dna(_real_config(random_seed=seed)).vector for seed in seeds]

        deltas = []
        for idx, vec_i in enumerate(vectors):
            for vec_j in vectors[idx + 1 :]:
                deltas.append(float(np.linalg.norm(vec_i - vec_j)))

        assert any(delta > 1e-6 for delta in deltas)

    def test_save_outputs_to_custom_directory(self, tmp_path):
        """Test that real extraction writes both signature and summary when save=True."""
        result = calc_dna(
            _real_config(
                dna_dim=16,
                max_samples=3,
                save=True,
                output_dir=tmp_path,
            )
        )

        assert result.output_path is not None
        assert result.summary_path is not None
        assert result.output_path.exists()
        assert result.summary_path.exists()
        assert result.output_path.parent == tmp_path / "rand" / MODEL_NAME
