from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from market_forecasting_engine.openai_models import DEFAULT_BEDROCK_OPENAI_MODEL, DEFAULT_OPENAI_MODEL


class LLMProviderName(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    BEDROCK = "bedrock"
    LLM_STUDIO = "llm_studio"


class HuggingFaceModel:
    KIMI_K2_INSTRUCT = "moonshotai/Kimi-K2-Instruct"
    KIMI_K2_THINKING = "moonshotai/Kimi-K2-Thinking"
    GPT_OSS_20B_CEREBRAS = "openai/gpt-oss-20b:cerebras"
    QWEN3_32B = "Qwen/Qwen3-32B"
    LLAMA_3_1_8B_INSTRUCT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    MISTRAL_NEMO_INSTRUCT = "mistralai/Mistral-Nemo-Instruct-2407"


class LocalModel:
    LLAMA_3_1_8B_INSTRUCT = "llama-3.1-8b-instruct"
    QWEN2_5_7B_INSTRUCT = "qwen2.5-7b-instruct"
    QWEN3_5_9B = "qwen3.5-9b"
    QWEN3_VL_8B_INSTRUCT_MLX = "qwen3-vl-8b-instruct-mlx"
    NEMOTRON_MINI_4B_INSTRUCT = "nemotron-mini-4b-instruct"
    GEMMA_4_26B_A4B_IT = "gemma-4-26b-a4b-it"
    GEMMA_4_E4B_IT = "gemma-4-e4b-it"
    MISTRAL_7B_INSTRUCT = "mistral-7b-instruct"


DEFAULT_HUGGINGFACE_TRADER_MODEL = HuggingFaceModel.KIMI_K2_INSTRUCT
DEFAULT_LOCAL_TRADER_MODEL = LocalModel.QWEN3_5_9B
DEFAULT_FULL_LLM_OPTIONS_PROVIDER = LLMProviderName.LLM_STUDIO.value
DEFAULT_FULL_LLM_OPTIONS_MODEL = LocalModel.QWEN3_VL_8B_INSTRUCT_MLX
DEFAULT_LLM_STUDIO_WEB_SEARCH_MODEL = LocalModel.QWEN3_VL_8B_INSTRUCT_MLX
DEFAULT_LLM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_LLM_STUDIO_REMOTE_DEVICE = "DESKTOP-89TCN67"
DEFAULT_LLM_STUDIO_REMOTE_DEVICE_ID = "1db783ee9370e4b499419e97c1cdf0be"
DEFAULT_LLM_STUDIO_REMOTE_MODEL_PATH = f"{DEFAULT_LLM_STUDIO_REMOTE_DEVICE_ID}:lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
LLM_STUDIO_REMOTE_MODELS = {
    "qwen3_5_9b": {
        "device": DEFAULT_LLM_STUDIO_REMOTE_DEVICE,
        "device_id": DEFAULT_LLM_STUDIO_REMOTE_DEVICE_ID,
        "model": LocalModel.QWEN3_5_9B,
        "params": "9B",
        "quant": "Q4_K_M",
        "size_gb": 6.5,
        "loaded_in_screenshot": True,
    },
    "gemma_4_e4b_it": {
        "device": DEFAULT_LLM_STUDIO_REMOTE_DEVICE,
        "device_id": DEFAULT_LLM_STUDIO_REMOTE_DEVICE_ID,
        "model": LocalModel.GEMMA_4_E4B_IT,
        "params": "7.5B",
        "quant": "Q4_K_M",
        "size_gb": 6.3,
        "loaded_in_screenshot": False,
    },
    "gemma_4_26b_a4b_it": {
        "device": DEFAULT_LLM_STUDIO_REMOTE_DEVICE,
        "device_id": DEFAULT_LLM_STUDIO_REMOTE_DEVICE_ID,
        "model": LocalModel.GEMMA_4_26B_A4B_IT,
        "params": "26B-A4B",
        "quant": "Q4_K_M",
        "size_gb": 18.0,
        "loaded_in_screenshot": False,
    },
}


@dataclass(frozen=True)
class LLMModelProfile:
    provider: str
    model: str
    base_url: str | None = None


DEFAULT_TRADER_MODEL_BY_PROVIDER = {
    LLMProviderName.OPENAI.value: DEFAULT_OPENAI_MODEL,
    LLMProviderName.HUGGINGFACE.value: DEFAULT_HUGGINGFACE_TRADER_MODEL,
    LLMProviderName.BEDROCK.value: DEFAULT_BEDROCK_OPENAI_MODEL,
    LLMProviderName.LLM_STUDIO.value: DEFAULT_LOCAL_TRADER_MODEL,
}


PROVIDER_ALIASES = {
    "open_ai": LLMProviderName.OPENAI.value,
    "aws_bedrock": LLMProviderName.BEDROCK.value,
    "amazon_bedrock": LLMProviderName.BEDROCK.value,
    "bedrock_openai": LLMProviderName.BEDROCK.value,
    "aws_bedrock_openai": LLMProviderName.BEDROCK.value,
    "hf": LLMProviderName.HUGGINGFACE.value,
    "hugging_face": LLMProviderName.HUGGINGFACE.value,
    "hugging_face_hub": LLMProviderName.HUGGINGFACE.value,
    "lm_studio": LLMProviderName.LLM_STUDIO.value,
    "llmstudio": LLMProviderName.LLM_STUDIO.value,
    "local_llm_studio": LLMProviderName.LLM_STUDIO.value,
}


def normalize_llm_provider(value: str | None) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    return PROVIDER_ALIASES.get(normalized, normalized)


def default_model_for_provider(provider: str | None) -> str:
    normalized = normalize_llm_provider(provider or LLMProviderName.OPENAI.value)
    if normalized == LLMProviderName.HUGGINGFACE.value:
        return os.environ.get("HUGGINGFACE_MODEL") or os.environ.get("HF_MODEL") or DEFAULT_HUGGINGFACE_TRADER_MODEL
    if normalized == LLMProviderName.BEDROCK.value:
        return os.environ.get("BEDROCK_OPENAI_MODEL") or os.environ.get("BEDROCK_MODEL") or DEFAULT_BEDROCK_OPENAI_MODEL
    if normalized == LLMProviderName.LLM_STUDIO.value:
        return os.environ.get("LLM_STUDIO_MODEL") or os.environ.get("LOCAL_LLM_MODEL") or DEFAULT_LOCAL_TRADER_MODEL
    return os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL


def resolve_llm_model_profile(provider: str | None = None, model: str | None = None) -> LLMModelProfile:
    resolved_provider = normalize_llm_provider(provider or os.environ.get("LLM_PROVIDER") or LLMProviderName.OPENAI.value)
    resolved_model = model or default_model_for_provider(resolved_provider)
    base_url = None
    if resolved_provider == LLMProviderName.LLM_STUDIO.value:
        base_url = os.environ.get("LLM_STUDIO_BASE_URL") or os.environ.get("LOCAL_LLM_BASE_URL") or DEFAULT_LLM_STUDIO_BASE_URL
    return LLMModelProfile(provider=resolved_provider, model=resolved_model, base_url=base_url)


def resolve_llm_step_profile(
    step: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    fallback_provider: str | None = None,
    fallback_model: str | None = None,
) -> LLMModelProfile:
    env_prefix = step.upper().replace("-", "_")
    step_provider = provider or os.environ.get(f"{env_prefix}_LLM_PROVIDER")
    step_model = model or os.environ.get(f"{env_prefix}_LLM_MODEL")
    return resolve_llm_model_profile(
        provider=step_provider or fallback_provider,
        model=step_model or fallback_model,
    )


ALTERNATIVE_TRADER_MODEL_PROFILES = {
    "hf_router_fast": {
        "provider": "huggingface",
        "model": DEFAULT_HUGGINGFACE_TRADER_MODEL,
        "notes": "Hosted Hugging Face router profile using Kimi for non-OpenAI trader experiments.",
    },
    "hf_kimi_thinking": {
        "provider": "huggingface",
        "model": HuggingFaceModel.KIMI_K2_THINKING,
        "notes": "Hosted Hugging Face Kimi thinking profile for deeper trader reasoning experiments.",
    },
    "local_llm_studio": {
        "provider": "llm_studio",
        "model": DEFAULT_LOCAL_TRADER_MODEL,
        "base_url": DEFAULT_LLM_STUDIO_BASE_URL,
        "notes": "Local OpenAI-compatible chat-completions profile, intended for LM Studio or similar local runtimes.",
    },
}
