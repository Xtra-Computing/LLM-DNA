"""Regression coverage for the original text/concat/shared-projection method."""

import hashlib
import json

import numpy as np
import pytest

from llm_dna.dna.TextDNAExtractor import TextDNAExtractor


def private_concat_formula(raw, *, normalize=True, seed=42, slot_dim=64, dna_dim=128):
    """Independent transcription of the private repository's normal Qwen path."""
    rows = raw[:, :slot_dim].astype(np.float32)
    if normalize:
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        rows = rows / norms
    matrix = np.random.RandomState(seed).randn(rows.size, dna_dim).astype(np.float32)
    matrix = matrix / np.linalg.norm(matrix, axis=0)
    return (rows.reshape(1, -1) @ matrix).reshape(-1)


def test_default_preserves_prefix_magnitudes():
    extractor = TextDNAExtractor()
    raw = np.arange(192, dtype=np.float32).reshape(2, 96)
    assert extractor.normalize_embeddings is False
    np.testing.assert_array_equal(extractor._pre_reduce_embeddings(raw), raw[:, :64])


@pytest.mark.parametrize("normalize", [True, False])
def test_concat_matches_private_formula_without_scaler_or_final_normalization(normalize):
    raw = np.random.RandomState(7).normal(size=(12, 96)).astype(np.float32)
    raw[4] = 0  # A missing answer must retain its position in the concatenation.
    original = raw.copy()
    extractor = TextDNAExtractor(normalize_embeddings=normalize)

    actual = extractor.reduce_embeddings(raw)

    np.testing.assert_array_equal(actual, private_concat_formula(raw, normalize=normalize))
    np.testing.assert_array_equal(raw, original)
    assert actual.shape == (128,)
    assert actual.dtype == np.float32
    assert not np.isclose(np.linalg.norm(actual), 1.0)


def test_last_probe_changes_dna_and_probe_order_matters():
    raw = np.random.RandomState(8).normal(size=(600, 64)).astype(np.float32)
    extractor = TextDNAExtractor()
    baseline = extractor.reduce_embeddings(raw)
    changed = raw.copy()
    changed[-1] *= -1

    assert not np.allclose(baseline, extractor.reduce_embeddings(changed))
    assert not np.allclose(baseline, extractor.reduce_embeddings(raw[::-1]))
    # Legacy concat projected each row first and truncated away almost all probes.
    changed[0] = raw[0]
    assert not np.array_equal(baseline, extractor.reduce_embeddings(changed))


def test_projection_is_shared_cached_and_does_not_change_global_rng():
    before = np.random.get_state()
    extractor = TextDNAExtractor(random_seed=123)
    matrix = extractor._get_or_create_projection_matrix(256, 128)
    after = np.random.get_state()

    assert matrix is extractor._get_or_create_projection_matrix(256, 128)
    np.testing.assert_array_equal(matrix, TextDNAExtractor(random_seed=123)._get_or_create_projection_matrix(256, 128))
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=0), 1.0, rtol=1e-6)
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert not matrix.flags.writeable
    assert not np.array_equal(matrix, TextDNAExtractor(random_seed=124)._get_or_create_projection_matrix(256, 128))


def test_normalization_is_after_prefix_and_removes_per_probe_scale_only_when_enabled():
    raw = np.random.RandomState(9).normal(size=(5, 96)).astype(np.float32)
    scaled = raw * np.array([0.5, 2, 4, 8, 16], dtype=np.float32)[:, None]
    normalized = TextDNAExtractor(normalize_embeddings=True)
    unnormalized = TextDNAExtractor(normalize_embeddings=False)

    np.testing.assert_allclose(normalized.reduce_embeddings(raw), normalized.reduce_embeddings(scaled), atol=1e-7)
    assert not np.allclose(unnormalized.reduce_embeddings(raw), unnormalized.reduce_embeddings(scaled))
    prefix = normalized._pre_reduce_embeddings(raw)
    np.testing.assert_allclose(np.linalg.norm(prefix, axis=1), 1.0, atol=1e-7)
    np.testing.assert_array_equal(unnormalized._pre_reduce_embeddings(raw), raw[:, :64])


@pytest.mark.parametrize("aggregation", ["concat", "mean", "sum", "max"])
def test_zero_placeholders_and_aggregation_follow_private_formulas(aggregation):
    raw = np.random.RandomState(10).normal(size=(4, 96)).astype(np.float32)
    raw[1] = 0
    extractor = TextDNAExtractor(aggregation_method=aggregation, normalize_embeddings=True)
    rows = extractor._pre_reduce_embeddings(raw)
    np.testing.assert_array_equal(rows[1], np.zeros(64))
    non_empty = rows[[0, 2, 3]]
    projection = extractor._get_or_create_projection_matrix(64, 128)
    if aggregation == "concat":
        expected = private_concat_formula(raw)
    elif aggregation == "mean":
        expected = (non_empty.mean(axis=0, keepdims=True) @ projection).reshape(-1)
    elif aggregation == "sum":
        expected = (non_empty @ projection).sum(axis=0)
    else:
        expected = (non_empty @ projection).max(axis=0)
    np.testing.assert_array_equal(extractor.reduce_embeddings(raw), expected)
    np.testing.assert_array_equal(extractor.reduce_embeddings(np.zeros_like(raw)), np.zeros(128))


class RecordingEncoder:
    def encode(self, answers, **kwargs):
        self.answers = answers
        self.kwargs = kwargs
        return np.arange(1, len(answers) * 96 + 1, dtype=np.float32).reshape(len(answers), 96)


def test_answer_encoding_preserves_slots_and_explicitly_disables_extra_encoder_normalization():
    encoder = RecordingEncoder()
    extractor = TextDNAExtractor(normalize_embeddings=True)
    answers = ["first", "", "   ", "last"]

    raw = extractor.encode_answers(answers, encoder=encoder)

    assert encoder.answers == ["first", "last"]
    assert encoder.kwargs["normalize_embeddings"] is False
    assert raw.shape == (4, 96)
    np.testing.assert_array_equal(raw[1:3], np.zeros((2, 96)))
    assert raw[-1, -1] == 192
    signature = extractor.extract_dna_from_answers(answers, model_name="example", encoder=encoder)
    np.testing.assert_array_equal(signature.signature, private_concat_formula(raw))
    assert signature.metadata.probe_count == 4
    with pytest.raises(ValueError, match="All provided answers are empty"):
        extractor.encode_answers(["", " "], encoder=encoder)


def test_metadata_records_projection_preprocessing_and_order():
    raw = np.random.RandomState(11).normal(size=(3, 96)).astype(np.float32)
    raw[1] = 0
    probes = ["first", "第二", "last"]
    extractor = TextDNAExtractor(normalize_embeddings=False)
    signature = extractor.extract_dna_from_embeddings(
        raw, model_name="example", probe_set_id="cached-600",
        probe_inputs=probes, model_metadata={"family": "example"},
        provenance={"cache_file": "example.npy"},
    )
    metadata = signature.metadata
    config = metadata.extractor_config
    assert metadata.model_name == "example"
    assert metadata.embedding_dimension == 64
    assert metadata.model_metadata == {"family": "example"}
    assert config["normalize_embeddings"] is False
    assert config["pre_projection_normalization"] == "none"
    assert config["pre_agg_method"] == "matryoshka_prefix"
    assert config["projection_shape"] == [192, 128]
    assert config["projection_column_normalization"] == "l2"
    assert config["zero_placeholder_indices"] == [1]
    assert config["provenance"] == {"cache_file": "example.npy"}
    assert config["probe_order_sha256"] == hashlib.sha256(
        json.dumps(probes, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    swapped = extractor.extract_dna_from_embeddings(raw, probe_inputs=probes[::-1])
    assert swapped.metadata.extractor_config["probe_order_sha256"] != config["probe_order_sha256"]
    with pytest.raises(ValueError, match="one string per embedding row"):
        extractor.extract_dna_from_embeddings(raw, probe_inputs=["incomplete"])


@pytest.mark.parametrize("raw", [np.empty((0, 96)), np.empty((3, 0)), np.ones(96), [[np.nan]], [[np.inf]], [[1e100]], [[1 + 2j]]])
def test_invalid_raw_embeddings_fail_instead_of_silent_cleanup(raw):
    with pytest.raises(ValueError):
        TextDNAExtractor().reduce_embeddings(raw)


@pytest.mark.parametrize("method", ["pca", "svd", "umap", "invalid"])
def test_per_model_fitting_is_rejected(method):
    with pytest.raises(ValueError, match="shared coordinates"):
        TextDNAExtractor(reduction_method=method)


@pytest.mark.parametrize("kwargs", [{"dna_dim": 0}, {"dna_dim": 2.5}, {"pre_agg_embed_dim": -1}, {"random_seed": -1}, {"normalize_embeddings": "false"}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TextDNAExtractor(**kwargs)
