import pytest

from llm_dna.models.ModelLoader import ModelLoader


@pytest.fixture
def loader():
    return ModelLoader()


# --- OpenAI direct models (bare names, no org prefix) ---

@pytest.mark.parametrize("model_name", [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4o",
    "gpt-4-turbo",
    "o1-preview",
    "o3-mini",
    "text-davinci-003",
])
def test_openai_models(loader, model_name):
    assert loader._detect_model_type(model_name) == "openai"


# --- OpenRouter models (org/model format) ---

@pytest.mark.parametrize("model_name", [
    "openrouter/auto",
    "openrouter:some-model",
    "anthropic/claude-3-opus",
    "deepseek/deepseek-chat",
    "openai/gpt-3.5-turbo",
    "openai/gpt-4o",
    "openai/gpt-4-turbo",
    "x-ai/grok-2",
    "cohere/command-r-plus",
    "perplexity/llama-3.1-sonar",
    "z-ai/some-model",
    "google/gemini-1.5-pro",
])
def test_openrouter_models(loader, model_name):
    assert loader._detect_model_type(model_name) == "openrouter"


# --- HuggingFace models that must NOT be misclassified ---

@pytest.mark.parametrize("model_name", [
    "openai/gpt-oss",
    "openai/whisper-large-v3",
    "openai/clip-vit-large-patch14",
    "google/bert-base-uncased",
    "google/flan-t5-xl",
    "meta-llama/Llama-3-8B",
    "mistralai/Mistral-7B-v0.1",
    "distilgpt2",
    "bert-base-uncased",
])
def test_huggingface_models(loader, model_name):
    assert loader._detect_model_type(model_name) == "huggingface"


# --- Gemini models ---

@pytest.mark.parametrize("model_name", [
    "gemini-1.5-pro",
    "gemini-pro",
    "models/gemini-1.5-flash",
])
def test_gemini_models(loader, model_name):
    assert loader._detect_model_type(model_name) == "gemini"
