"""
Model loading utilities for different LLM types.
"""

import os
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging

import torch
from .ModelWrapper import (LLMWrapper, HuggingFaceWrapper, OpenAIWrapper, OpenRouterWrapper, GeminiWrapper,
                           DecoderOnlyWrapper, EncoderOnlyWrapper, EncoderDecoderWrapper)


class ModelLoader:
    """Factory class for loading different types of LLMs."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config_dict = config_dict or {}
        
    def load_model(
        self,
        model_path_or_name: str,
        model_type: str = "auto",
        device: str = "auto",
        **kwargs
    ) -> LLMWrapper:
        """
        Load a model with the appropriate wrapper.
        
        Args:
            model_path_or_name: Path to local model or HuggingFace model name
            model_type: Type of model ("auto", "huggingface", "openai", "openrouter", "gemini", "anthropic")
            device: Device for computation
            **kwargs: Additional arguments for model loading
            
        Returns:
            Wrapped model instance
        """
        # Auto-detect model type if not specified
        if model_type == "auto":
            model_type = self._detect_model_type(model_path_or_name)
            
        self.logger.info(f"Loading {model_type} model: {model_path_or_name}")
        
        if model_type == "huggingface":
            return self._load_huggingface_model(model_path_or_name, device, **kwargs)
        elif model_type == "openai":
            return self._load_openai_model(model_path_or_name, **kwargs)
        elif model_type == "openrouter":
            return self._load_openrouter_model(model_path_or_name, **kwargs)
        elif model_type == "gemini":
            return self._load_gemini_model(model_path_or_name, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Supported: huggingface, openai, openrouter, gemini")
            
    def _detect_model_type(self, model_path_or_name: str) -> str:
        """Auto-detect model type based on path/name patterns."""
        # Check for OpenAI model names (including newer models)
        openai_prefixes = [
            "gpt-3.5", "gpt-4", "gpt-4o", "gpt-4-turbo",
            "o1-", "o3-",  # Reasoning models
            "text-davinci", "text-curie", "text-babbage", "text-ada",
        ]
        
        model_lower = model_path_or_name.lower()
        if any(model_lower.startswith(prefix) for prefix in openai_prefixes):
            return "openai"

        # Check for OpenRouter model names.
        # Keep this conservative to avoid misclassifying common HuggingFace IDs.
        openrouter_prefixes = [
            "openrouter/",
            "openrouter:",
            "anthropic/claude-",
            "deepseek/",
            "openai/gpt-",
            "google/gemini-",
            "z-ai/",
            "x-ai/grok-",
            "cohere/command",
            "perplexity/",
        ]
        huggingface_prefixes_openai = "openai/gpt-oss"

        if any(model_lower.startswith(prefix) for prefix in openrouter_prefixes) and not model_lower.startswith(huggingface_prefixes_openai):
            return "openrouter"

        # Check for Google Gemini model names
        gemini_prefixes = [
            "gemini-",
            "models/gemini-",
            "gemini-pro",  # Older naming
        ]
        if any(model_lower.startswith(prefix) for prefix in gemini_prefixes):
            return "gemini"
            
        # Check if it's a local path
        if os.path.exists(model_path_or_name):
            return "huggingface"
            
        # Default to HuggingFace for unknown patterns
        return "huggingface"
        
    def _load_huggingface_model(
        self,
        model_path_or_name: str,
        device: str = "auto",
        **kwargs
    ) -> HuggingFaceWrapper:
        """Load HuggingFace model with appropriate wrapper based on model type."""
        # vLLM preference flag (only relevant for decoder-only text gen)
        try_vllm = bool(kwargs.pop('try_vllm', False))
        # Check for known problematic models first
        if self._is_unsupported_model(model_path_or_name):
            raise ValueError(f"Model {model_path_or_name} is not supported due to known issues")
        
        # Get model architecture type from config if available
        try:
            model_arch_type = self._get_model_architecture_type(model_path_or_name)
        except Exception as e:
            self.logger.warning(f"Could not determine architecture for {model_path_or_name}: {e}")
            model_arch_type = "decoder_only"  # Default fallback
        
        # Choose the appropriate wrapper class
        if model_arch_type == "decoder_only":
            wrapper_class = DecoderOnlyWrapper
        elif model_arch_type == "encoder_only":
            wrapper_class = EncoderOnlyWrapper
        elif model_arch_type == "encoder_decoder":
            wrapper_class = EncoderDecoderWrapper
        else:
            # Default to decoder-only for unknown types
            wrapper_class = DecoderOnlyWrapper
            
        self.logger.info(f"Using {wrapper_class.__name__} for model {model_path_or_name}")
        
        # Try vLLM fast path if requested and applicable
        if try_vllm and wrapper_class is DecoderOnlyWrapper:
            try:
                from .ModelWrapper import VLLMWrapper  # Optional dependency
                self.logger.info("Attempting to initialize vLLM engine for generation")
                return VLLMWrapper(
                    model_name=model_path_or_name,
                    device=device,
                    trust_remote_code=kwargs.get('trust_remote_code'),
                    torch_dtype=kwargs.get('torch_dtype'),
                    token=kwargs.get('token'),
                    is_chat_model=kwargs.get('is_chat_model')
                )
            except Exception as e:
                self.logger.warning(f"vLLM not available or failed ('{e}'); falling back to standard HF path")
        
        # Add memory management for large models (HF path)
        if self._is_large_model(model_path_or_name):
            self.logger.info(f"Large model detected: {model_path_or_name}. Enabling optimizations.")
            kwargs.setdefault('load_in_8bit', True)
        
        try:
            return wrapper_class(
                model_name=model_path_or_name,
                device=device,
                **kwargs
            )
        except Exception as e:
            # Enhanced error handling with specific error types
            error_msg = str(e).lower()
            if 'gated repo' in error_msg or 'access' in error_msg:
                raise ValueError(f"Access denied to gated repository {model_path_or_name}. Please check your HuggingFace authentication.")
            elif 'custom code' in error_msg:
                self.logger.warning(f"Model {model_path_or_name} requires custom code. Retrying with trust_remote_code=True")
                kwargs['trust_remote_code'] = True
                return wrapper_class(
                    model_name=model_path_or_name,
                    device=device,
                    **kwargs
                )
            elif 'multi_modality' in error_msg or 'transformers' in error_msg:
                raise ValueError(f"Model {model_path_or_name} uses unsupported architecture or requires newer transformers version")
            else:
                raise
        
    def _get_model_architecture_type(self, model_name: str) -> str:
        """Get model architecture type from config or auto-detect."""
        # Check if model is in experiment_models config
        if self.config_dict:
            experiment_models = self.config_dict.get("experiment_models", {})
            for model_key, model_config in experiment_models.items():
                if model_config.get("model_name") == model_name:
                    return model_config.get("model_type", "unknown")
        
        # Auto-detect based on model name patterns
        decoder_only_patterns = [
            "gpt", "llama", "mistral", "falcon", "opt", "bloom", "mpt", "pythia",
            "qwen", "yi", "gemma", "phi", "deepseek", "codellama", "vicuna", 
            "dolly", "stablelm", "open_llama", "tinyllama", "glm", "typhoon",
            "seed", "deci", "ministral", "mixtral"
        ]
        encoder_only_patterns = [
            "bert", "roberta", "distilbert", "electra", "deberta", "xlm-roberta"
        ]
        encoder_decoder_patterns = [
            "t5", "flan-t5", "ul2", "flan-ul2", "bart", "pegasus", "mbart", "mt5"
        ]
        
        model_name_lower = model_name.lower()
        
        for pattern in encoder_decoder_patterns:
            if pattern in model_name_lower:
                return "encoder_decoder"
                
        for pattern in encoder_only_patterns:
            if pattern in model_name_lower:
                return "encoder_only"
                
        for pattern in decoder_only_patterns:
            if pattern in model_name_lower:
                return "decoder_only"
        
        # Default to decoder_only for unknown models
        return "decoder_only"
        
    def _load_openai_model(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> OpenAIWrapper:
        """Load OpenAI model."""
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("APIKEY_OPENAI")
            
        if api_key is None:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        allowed_kwargs = {
            "batch_poll_interval_seconds",
            "batch_timeout_seconds",
            "batch_max_requests",
            "prefer_batch_api",
        }
        wrapper_kwargs = {key: value for key, value in kwargs.items() if key in allowed_kwargs}
            
        return OpenAIWrapper(
            model_name=model_name,
            api_key=api_key,
            **wrapper_kwargs
        )

    def _load_openrouter_model(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> OpenRouterWrapper:
        """Load OpenRouter model."""
        # Get API key from environment if not provided
        if api_key is None:
            api_key = (
                os.getenv("OPENROUTER_API_KEY")
                or os.getenv("APIKEY_OPENROUTER")
                or os.getenv("OPENROUTER_KEY")
            )
            
        if api_key is None:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY environment variable.")

        allowed_kwargs = {
            "batch_poll_interval_seconds",
            "batch_timeout_seconds",
            "batch_max_requests",
            "prefer_batch_api",
            "http_referer",
            "x_title",
        }
        wrapper_kwargs = {key: value for key, value in kwargs.items() if key in allowed_kwargs}
            
        return OpenRouterWrapper(
            model_name=model_name,
            api_key=api_key,
            **wrapper_kwargs
        )

    def _load_gemini_model(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> GeminiWrapper:
        """Load Gemini model."""
        if api_key is None:
            api_key = (
                os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
                or os.getenv("APIKEY_GOOGLE")
            )

        if api_key is None:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY.")

        allowed_kwargs = {
            "api_base",
            "batch_poll_interval_seconds",
            "batch_timeout_seconds",
            "batch_max_requests",
            "batch_max_payload_bytes",
            "prefer_batch_api",
        }
        wrapper_kwargs = {key: value for key, value in kwargs.items() if key in allowed_kwargs}

        return GeminiWrapper(
            model_name=model_name,
            api_key=api_key,
            **wrapper_kwargs
        )
    
    def _is_unsupported_model(self, model_name: str) -> bool:
        """Check if a model is known to be unsupported."""
        unsupported_patterns = [
            "janus",  # Multi-modal models
            "glm-4.1v",  # Vision models with unknown attributes
        ]
        
        model_name_lower = model_name.lower()
        return any(pattern in model_name_lower for pattern in unsupported_patterns)
    
    def _is_large_model(self, model_name: str) -> bool:
        """Determine if a model is large and needs memory optimizations."""
        large_model_patterns = [
            "20b", "24b", "30b", "32b", "34b", "70b", "72b", 
            "13b", "14b", "mixtral", "gpt-neox-20b", "flan-ul2",
            "mistral-small", "qwq", "glm-4-32b"
        ]
        
        model_name_lower = model_name.lower()
        return any(pattern in model_name_lower for pattern in large_model_patterns)
        
    def list_available_models(self, model_type: str = "huggingface") -> Dict[str, Any]:
        """
        List available models for a given type.
        
        Args:
            model_type: Type of models to list
            
        Returns:
            Dictionary with model information
        """
        if model_type == "huggingface":
            return self._list_huggingface_models()
        elif model_type == "openai":
            return self._list_openai_models()
        elif model_type == "openrouter":
            return self._list_openrouter_models()
        elif model_type == "gemini":
            return self._list_gemini_models()
        else:
            return {}
            
    def _list_huggingface_models(self) -> Dict[str, Any]:
        """List popular HuggingFace models for DNA extraction."""
        return {
            "language_models": [
                "microsoft/DialoGPT-medium",
                "microsoft/DialoGPT-large",
                "gpt2",
                "gpt2-medium",
                "gpt2-large",
                "distilgpt2",
                "facebook/opt-350m",
                "facebook/opt-1.3b",
                "EleutherAI/gpt-neo-125M",
                "EleutherAI/gpt-neo-1.3B",
                "EleutherAI/gpt-j-6B",
                "microsoft/DialoGPT-small"
            ],
            "instruction_tuned": [
                "microsoft/DialoGPT-medium",
                "facebook/blenderbot-400M-distill",
                "facebook/blenderbot_small-90M"
            ],
            "multilingual": [
                "microsoft/DialoGPT-medium",
                "facebook/mbart-large-cc25"
            ],
            "domain_specific": {
                "code": [
                    "microsoft/CodeGPT-small-py",
                    "microsoft/CodeGPT-small-java"
                ],
                "science": [
                    "allenai/scibert_scivocab_uncased"
                ],
                "biomedical": [
                    "dmis-lab/biobert-base-cased-v1.1"
                ]
            }
        }
        
    def _list_openai_models(self) -> Dict[str, Any]:
        """List available OpenAI models."""
        return {
            "chat_models": [
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4o"
            ],
            "completion_models": [
                "text-davinci-003",
                "text-curie-001",
                "text-babbage-001",
                "text-ada-001"
            ]
        }

    def _list_openrouter_models(self) -> Dict[str, Any]:
        """List available OpenRouter models (popular examples)."""
        return {
            "openai_models": [
                "openai/gpt-4o",
                "openai/gpt-4o-mini",
                "openai/gpt-4-turbo",
                "openai/gpt-3.5-turbo",
            ],
            "anthropic_models": [
                "anthropic/claude-3.5-sonnet",
                "anthropic/claude-3-opus",
                "anthropic/claude-3-haiku",
            ],
            "meta_models": [
                "meta-llama/llama-3.1-405b-instruct",
                "meta-llama/llama-3.1-70b-instruct",
                "meta-llama/llama-3.1-8b-instruct",
            ],
            "google_models": [
                "google/gemini-pro-1.5",
                "google/gemini-flash-1.5",
            ],
            "note": "OpenRouter provides access to 200+ models. See https://openrouter.ai/models for full list."
        }

    def _list_gemini_models(self) -> Dict[str, Any]:
        """List available Gemini models."""
        return {
            "chat_models": [
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
            ]
        }
        
    def get_model_info(self, model_path_or_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_path_or_name: Model identifier
            
        Returns:
            Dictionary with model information
        """
        model_type = self._detect_model_type(model_path_or_name)
        
        info = {
            "model_name": model_path_or_name,
            "detected_type": model_type,
            "is_local": os.path.exists(model_path_or_name)
        }
        
        if model_type == "huggingface":
            info.update(self._get_huggingface_info(model_path_or_name))
        elif model_type in ["openai", "openrouter", "gemini", "anthropic"]:
            info.update({"requires_api_key": True})
            
        return info
        
    def _get_huggingface_info(self, model_name: str) -> Dict[str, Any]:
        """Get HuggingFace model information."""
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            
            return {
                "model_type": getattr(config, 'model_type', 'unknown'),
                "vocab_size": getattr(config, 'vocab_size', None),
                "hidden_size": getattr(config, 'hidden_size', None),
                "num_layers": getattr(config, 'num_hidden_layers', 
                                    getattr(config, 'num_layers', None)),
                "max_position_embeddings": getattr(config, 'max_position_embeddings', None)
            }
        except Exception as e:
            self.logger.warning(f"Could not get model info for {model_name}: {e}")
            return {"error": str(e)}


# Convenience function for quick model loading
def load_model(
    model_path_or_name: str,
    model_type: str = "auto",
    device: str = "auto",
    config_dict: Optional[Dict[str, Any]] = None,
    **kwargs
) -> LLMWrapper:
    """
    Convenience function to load a model.
    
    Args:
        model_path_or_name: Model identifier
        model_type: Model type ("auto", "huggingface", "openai", "openrouter", "gemini", "anthropic")
        device: Computation device
        config_dict: Model configuration dictionary
        **kwargs: Additional model loading arguments
        
    Returns:
        Wrapped model instance
    """
    loader = ModelLoader(config_dict=config_dict)
    return loader.load_model(model_path_or_name, model_type, device, **kwargs)
