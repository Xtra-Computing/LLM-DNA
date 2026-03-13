"""LLM-DNA: LLM DNA extraction toolkit."""

__all__ = [
    "__version__",
    "DNAExtractionConfig",
    "DNAExtractionResult",
    "calc_dna",
    "calc_dna_parallel",
    "calc_dna_batch",
]

from importlib.metadata import version as _version

__version__ = _version("llm-dna")


def __getattr__(name: str):
    if name in {
        "DNAExtractionConfig",
        "DNAExtractionResult",
        "calc_dna",
        "calc_dna_parallel",
        "calc_dna_batch",
    }:
        from .api import (
            DNAExtractionConfig,
            DNAExtractionResult,
            calc_dna,
            calc_dna_parallel,
            calc_dna_batch,
        )

        exports = {
            "DNAExtractionConfig": DNAExtractionConfig,
            "DNAExtractionResult": DNAExtractionResult,
            "calc_dna": calc_dna,
            "calc_dna_parallel": calc_dna_parallel,
            "calc_dna_batch": calc_dna_batch,
        }
        return exports[name]
    raise AttributeError(f"module 'llm_dna' has no attribute {name!r}")
