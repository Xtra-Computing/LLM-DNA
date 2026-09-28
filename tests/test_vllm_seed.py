"""Exercise vLLM seed forwarding without loading vLLM or model weights."""

import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_dna.models.ModelWrapper import VLLMWrapper


def _outputs(*texts):
    return [SimpleNamespace(outputs=[SimpleNamespace(text=text)]) for text in texts]


@pytest.fixture
def wrapper(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=SimpleNamespace))
    instance = VLLMWrapper.__new__(VLLMWrapper)
    instance.logger = logging.getLogger(__name__)
    instance._engine = Mock()
    return instance


@pytest.mark.parametrize("seed_kwargs, expected_seed", [({}, None), ({"seed": 0}, 0), ({"seed": 73}, 73)])
def test_generate_forwards_optional_seed(wrapper, seed_kwargs, expected_seed):
    wrapper._engine.generate.return_value = _outputs(" answer ")

    assert wrapper.generate("prompt", **seed_kwargs) == "answer"

    prompts, params = wrapper._engine.generate.call_args.args
    assert prompts == ["prompt"]
    assert vars(params) == {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "n": 1,
        "seed": expected_seed,
    }


@pytest.mark.parametrize("seed_kwargs, expected_seed", [({}, None), ({"seed": 0}, 0), ({"seed": 73}, 73)])
def test_generate_batch_forwards_optional_seed(wrapper, seed_kwargs, expected_seed):
    wrapper._engine.generate.return_value = _outputs(" first ", " second ")

    assert wrapper.generate_batch(["one", "two"], **seed_kwargs) == ["first", "second"]

    prompts, params = wrapper._engine.generate.call_args.args
    assert prompts == ["one", "two"]
    assert vars(params) == {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "n": 1,
        "seed": expected_seed,
    }


@pytest.mark.parametrize("seed_kwargs, expected_seed", [({}, None), ({"seed": 0}, 0), ({"seed": 73}, 73)])
def test_batch_failure_preserves_seed_for_sequential_fallback(wrapper, seed_kwargs, expected_seed):
    wrapper._engine.generate.side_effect = [
        RuntimeError("batch failed"),
        _outputs("first"),
        _outputs("second"),
    ]

    assert wrapper.generate_batch(["one", "two"], **seed_kwargs) == ["first", "second"]

    calls = wrapper._engine.generate.call_args_list
    assert [call.args[0] for call in calls] == [["one", "two"], ["one"], ["two"]]
    assert [call.args[1].seed for call in calls] == [expected_seed] * 3
