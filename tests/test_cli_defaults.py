import pytest

from llm_dna import DNAExtractionConfig
from llm_dna.cli import parse_arguments


def test_public_and_cli_defaults_match_original_experiment():
    args = parse_arguments(["--model-name", "distilgpt2"])
    config = DNAExtractionConfig(model_name="distilgpt2")
    for name, expected in {
        "extractor_type": "text",
        "dataset": "squad,cqa,hs,wg,arc,mmlu", "max_samples": 100,
        "sentence_encoder": "Qwen/Qwen3-Embedding-8B", "pre_agg_embed_dim": 64,
        "dna_dim": 128, "embedding_merge": "concat", "reduction_method": "random_projection",
        "normalize_embeddings": False, "temperature": 0.7, "top_p": 0.9, "use_chat_template": True,
    }.items():
        assert getattr(args, name) == getattr(config, name) == expected
    assert args.normalize_embeddings == config.normalize_embeddings


def test_normalization_and_chat_template_can_be_disabled():
    args = parse_arguments(["--model-name", "distilgpt2", "--no-normalize-embeddings", "--no-use-chat-template"])
    assert args.normalize_embeddings is False
    assert args.use_chat_template is False


def test_cli_accepts_text_and_rejects_legacy_extractor():
    args = parse_arguments(["--model-name", "distilgpt2", "--extractor-type", "text"])
    assert args.extractor_type == "text"
    with pytest.raises(SystemExit) as exc:
        parse_arguments(["--model-name", "distilgpt2", "--extractor-type", "embedding"])
    assert exc.value.code == 2
