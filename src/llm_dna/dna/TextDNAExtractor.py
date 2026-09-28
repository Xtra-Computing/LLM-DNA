"""Text DNA in a shared coordinate system, matching the original concat pipeline.

The Qwen path keeps the first ``pre_agg_embed_dim`` coordinates of each answer,
optionally L2 normalizes each row, concatenates the fixed probe slots, and applies
one shared random projection. There is no fitted scaler or final DNA normalization.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .DNASignature import DNAMetadata, DNASignature


class TextDNAExtractor:
    """Encode answers and project their ordered embeddings into comparable DNA.

    ``normalize_embeddings`` controls row normalization *after* per-answer
    dimension reduction, before the final projection. It does not change the
    encoder's output normalization or normalize the resulting DNA vector.
    """

    def __init__(
        self,
        dna_dim: int = 128,
        pre_agg_embed_dim: Optional[int] = 64,
        normalize_embeddings: bool = False,
        random_seed: int = 42,
        aggregation_method: str = "concat",
        reduction_method: str = "random_projection",
        encoder_name: str = "Qwen/Qwen3-Embedding-8B",
        device: str = "cpu",
    ):
        self.dna_dim = self._positive_dimension(dna_dim, "dna_dim")
        self.pre_agg_embed_dim = (
            None if pre_agg_embed_dim is None
            else self._positive_dimension(pre_agg_embed_dim, "pre_agg_embed_dim")
        )
        if not isinstance(normalize_embeddings, (bool, np.bool_)):
            raise ValueError("normalize_embeddings must be a boolean")
        self.normalize_embeddings = bool(normalize_embeddings)
        if isinstance(random_seed, (bool, np.bool_)) or not isinstance(random_seed, (int, np.integer)):
            raise ValueError("random_seed must be an integer in [0, 2**32 - 1]")
        self.random_seed = int(random_seed)
        if not 0 <= self.random_seed <= 2**32 - 1:
            raise ValueError("random_seed must be an integer in [0, 2**32 - 1]")
        self.aggregation_method = aggregation_method.lower()
        if self.aggregation_method not in {"concat", "mean", "sum", "max"}:
            raise ValueError("aggregation_method must be concat, mean, sum, or max")
        self.reduction_method = reduction_method.lower()
        if self.reduction_method != "random_projection":
            raise ValueError(
                "Text DNA requires random_projection to preserve shared coordinates; "
                "per-model PCA/SVD/UMAP fitting is not supported"
            )
        if not isinstance(encoder_name, str) or not encoder_name.strip():
            raise ValueError("encoder_name must be a non-empty string")
        self.encoder_name = encoder_name
        self.device = device
        self._encoder = None
        self._projection_matrices: dict[tuple[int, int, int], np.ndarray] = {}

    @staticmethod
    def _positive_dimension(value: int, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    @staticmethod
    def _validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
        raw = np.asarray(embeddings)
        if raw.ndim != 2 or 0 in raw.shape:
            raise ValueError("embeddings must be a non-empty [probe_count, embedding_dimension] matrix")
        if raw.dtype.kind not in "fiu" or not np.all(np.isfinite(raw)):
            raise ValueError("embeddings must contain finite real numeric values")
        with np.errstate(over="ignore", invalid="ignore"):
            raw = np.asarray(raw, dtype=np.float32)
        if not np.all(np.isfinite(raw)):
            raise ValueError("embeddings must be representable as finite float32 values")
        return raw

    def _is_qwen_embedding_encoder(self) -> bool:
        name = self.encoder_name.lower()
        return "qwen" in name and "embedding" in name

    def _get_or_create_projection_matrix(self, input_dim: int, output_dim: int) -> np.ndarray:
        """Return the original float32 Gaussian matrix with unit-length columns.

        A local RandomState reproduces ``np.random.seed(seed); randn(...)`` in
        the private implementation without modifying application-global RNG state.
        """
        input_dim = self._positive_dimension(input_dim, "input_dim")
        output_dim = self._positive_dimension(output_dim, "output_dim")
        key = (input_dim, output_dim, self.random_seed)
        if key not in self._projection_matrices:
            matrix = np.random.RandomState(self.random_seed).randn(input_dim, output_dim).astype(np.float32)
            matrix = matrix / np.linalg.norm(matrix, axis=0)
            matrix.setflags(write=False)
            self._projection_matrices[key] = matrix
        return self._projection_matrices[key]

    def _pre_reduce_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Keep Qwen's Matryoshka prefix, then optionally normalize each row.

        For other encoder families, use a shared random projection instead of
        assuming that their leading coordinates form a trained smaller embedding.
        Zero rows remain zero and keep their probe slots.
        """
        raw = self._validate_embeddings(embeddings)
        target = self.pre_agg_embed_dim
        if target is None or target == raw.shape[1]:
            reduced = raw.copy()
        elif self._is_qwen_embedding_encoder():
            reduced = raw[:, :target].copy()
            if reduced.shape[1] < target:
                reduced = np.pad(reduced, ((0, 0), (0, target - reduced.shape[1])))
        else:
            reduced = raw @ self._get_or_create_projection_matrix(raw.shape[1], target)
        if self.normalize_embeddings:
            with np.errstate(over="ignore"):
                norms = np.linalg.norm(reduced, axis=1, keepdims=True)
            if not np.all(np.isfinite(norms)):
                raise ValueError("embedding row norms overflow float32")
            norms[norms == 0] = 1.0
            reduced = reduced / norms
        return reduced.astype(np.float32, copy=False)

    def reduce_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce raw encoder embeddings; row order defines the probe slots.

        All-zero rows represent missing answers. Concat keeps their positions;
        the other aggregations omit those rows, as in the private implementation.
        An all-zero matrix maps to a zero vector for every aggregation.
        """
        raw = self._validate_embeddings(embeddings)
        rows = self._pre_reduce_embeddings(raw)
        if self.aggregation_method == "concat":
            vector = rows.reshape(1, -1)
            result = vector @ self._get_or_create_projection_matrix(vector.shape[1], self.dna_dim)
        else:
            rows = rows[np.any(raw != 0, axis=1)]
            if not len(rows):
                return np.zeros(self.dna_dim, dtype=np.float32)
            projection = self._get_or_create_projection_matrix(rows.shape[1], self.dna_dim)
            if self.aggregation_method == "mean":
                result = rows.mean(axis=0, keepdims=True) @ projection
            else:
                projected = rows @ projection
                result = projected.sum(axis=0) if self.aggregation_method == "sum" else projected.max(axis=0)
        result = np.asarray(result, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(result)):
            raise ValueError("projection produced non-finite DNA values")
        return result

    def encode_answers(self, answers: Sequence[str], encoder: Any = None) -> np.ndarray:
        """Encode non-empty answers and reinsert zeros into their original slots.

        Passing an encoder supports an already-loaded SentenceTransformer and
        offline experiments. Otherwise the encoder is loaded on first use only.
        Raw encoder output is retained; preprocessing happens in reduce_embeddings.
        """
        if isinstance(answers, (str, bytes)):
            raise ValueError("answers must be an ordered sequence of strings")
        answers = list(answers)
        if any(not isinstance(answer, str) for answer in answers):
            raise ValueError("every answer must be a string; use an empty string for missing answers")
        indices = [index for index, answer in enumerate(answers) if answer.strip()]
        if not indices:
            raise ValueError("All provided answers are empty; refusing DNA extraction")
        if encoder is None:
            if self._encoder is None:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(self.encoder_name, device=self.device)
            encoder = self._encoder
        encoded = encoder.encode(
            [answers[index] for index in indices],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        encoded = self._validate_embeddings(encoded)
        if encoded.shape[0] != len(indices):
            raise ValueError("encoder returned a different number of rows than non-empty answers")
        result = np.zeros((len(answers), encoded.shape[1]), dtype=np.float32)
        result[indices] = encoded
        return result

    def extract_dna_from_embeddings(
        self,
        embeddings: np.ndarray,
        model_name: str = "precomputed",
        probe_set_id: str = "default",
        *,
        probe_inputs: Optional[Sequence[str]] = None,
        model_metadata: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> DNASignature:
        """Create DNA with explicit preprocessing, projection and order provenance.

        ``embeddings`` must be raw encoder outputs, before prefix slicing or the
        optional row normalization. A provided probe list is hashed in order.
        """
        started = time.perf_counter()
        raw = self._validate_embeddings(embeddings)
        probe_order_sha256 = None
        if probe_inputs is not None:
            if isinstance(probe_inputs, (str, bytes)):
                raise ValueError("probe_inputs must be an ordered sequence of strings")
            probes = list(probe_inputs)
            if len(probes) != raw.shape[0] or any(not isinstance(probe, str) for probe in probes):
                raise ValueError("probe_inputs must contain one string per embedding row")
            probe_order_sha256 = hashlib.sha256(
                json.dumps(probes, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        dna = self.reduce_embeddings(raw)
        slot_dim = self.pre_agg_embed_dim or raw.shape[1]
        projection_input_dim = slot_dim * raw.shape[0] if self.aggregation_method == "concat" else slot_dim
        projection = self._get_or_create_projection_matrix(projection_input_dim, self.dna_dim)
        pre_agg_method = (
            "none" if self.pre_agg_embed_dim is None or self.pre_agg_embed_dim == raw.shape[1]
            else "matryoshka_prefix" if self._is_qwen_embedding_encoder()
            else "shared_random_projection"
        )
        config = {
            "pipeline_version": "text-dna-concat-v1",
            "encoder": self.encoder_name,
            "encoder_name": self.encoder_name,
            "encoder_encode_normalize_embeddings": False,
            "raw_embedding_dimension": int(raw.shape[1]),
            "pre_agg_embed_dim": self.pre_agg_embed_dim,
            "pre_agg_method": pre_agg_method,
            "normalize_embeddings": self.normalize_embeddings,
            "pre_projection_normalization": "per_probe_l2" if self.normalize_embeddings else "none",
            "aggregation_method": self.aggregation_method,
            "reduction_method": self.reduction_method,
            "dna_dim": self.dna_dim,
            "random_seed": self.random_seed,
            "projection_rng": "numpy.random.RandomState(MT19937).randn",
            "projection_dtype": "float32",
            "projection_column_normalization": "l2",
            "projection_shape": [projection_input_dim, self.dna_dim],
            "projection_sha256": hashlib.sha256(projection.tobytes(order="C")).hexdigest(),
            "probe_order": "provided_probe_inputs" if probe_inputs is not None else "provided_embedding_row_order",
            "probe_order_sha256": probe_order_sha256,
            "zero_placeholder_indices": np.flatnonzero(~np.any(raw != 0, axis=1)).tolist(),
            "final_dna_normalization": "none",
            "provenance": dict(provenance or {}),
        }
        metadata = DNAMetadata(
            model_name=model_name,
            extraction_method=f"text_embeddings_random_projection_{self.aggregation_method}",
            probe_set_id=probe_set_id,
            probe_count=int(raw.shape[0]),
            dna_dimension=self.dna_dim,
            embedding_dimension=int(slot_dim),
            reduction_method=self.reduction_method,
            extraction_time=datetime.now(timezone.utc).isoformat(),
            computation_time_seconds=time.perf_counter() - started,
            model_metadata=dict(model_metadata or {}),
            extractor_config=config,
            aggregation_method=self.aggregation_method,
        )
        return DNASignature(dna, metadata)

    def extract_dna_from_answers(
        self,
        answers: Sequence[str],
        model_name: str = "precomputed",
        probe_set_id: str = "default",
        *,
        encoder: Any = None,
        probe_inputs: Optional[Sequence[str]] = None,
        model_metadata: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> DNASignature:
        """Skip generation, encode the given answers, and construct their DNA."""
        started = time.perf_counter()
        raw = self.encode_answers(answers, encoder=encoder)
        signature = self.extract_dna_from_embeddings(
            raw, model_name=model_name, probe_set_id=probe_set_id,
            probe_inputs=probe_inputs, model_metadata=model_metadata, provenance=provenance,
        )
        signature.metadata.computation_time_seconds = time.perf_counter() - started
        return signature
