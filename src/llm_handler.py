"""Shared LLM provider routing for the ETF helper app."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


BEDROCK_OPENAI_BASE_URL = "https://bedrock-mantle.us-east-2.api.aws/openai/v1"
DEFAULT_OPENAI_MODEL = "gpt-4.1"
DEFAULT_BEDROCK_OPENAI_MODEL = "openai.gpt-5.4"
DEFAULT_HUGGINGFACE_MODEL = "openai/gpt-oss-20b:cerebras"
BEDROCK_OPENAI_MODELS = {"openai.gpt-5.4", "openai.gpt-5.5"}


class LLMProviderNotConfigured(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMRequest:
    provider: str
    model: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class LLMResult:
    provider: str
    model: str
    output_text: str
    response: Any


def normalize_provider_name(value: str | None) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "open_ai": "openai",
        "aws_bedrock": "bedrock",
        "amazon_bedrock": "bedrock",
        "bedrock_openai": "bedrock",
        "aws_bedrock_openai": "bedrock",
        "hf": "huggingface",
        "hugging_face": "huggingface",
        "hugging_face_hub": "huggingface",
        "lm_studio": "llm_studio",
        "llmstudio": "llm_studio",
    }
    return aliases.get(normalized, normalized)


def resolve_llm_model(model: str | None, provider: str) -> str:
    if model:
        return model
    provider = normalize_provider_name(provider)
    if provider == "bedrock":
        return os.getenv("BEDROCK_OPENAI_MODEL") or os.getenv("BEDROCK_MODEL") or DEFAULT_BEDROCK_OPENAI_MODEL
    if provider == "huggingface":
        return os.getenv("HUGGINGFACE_MODEL") or os.getenv("HF_MODEL") or DEFAULT_HUGGINGFACE_MODEL
    if provider == "llm_studio":
        return os.getenv("LLM_STUDIO_MODEL") or "local-model"
    return os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL


def call_llm(request: LLMRequest) -> LLMResult:
    provider = normalize_provider_name(request.provider)
    payload = {**request.payload, "model": request.model}
    if provider == "openai":
        client = _openai_client()
        response = client.responses.create(**payload)
        return LLMResult(provider=provider, model=request.model, output_text=_output_text(response), response=response)
    if provider == "bedrock":
        client = _bedrock_openai_client()
        response = client.responses.create(**payload)
        return LLMResult(provider=provider, model=request.model, output_text=_output_text(response), response=response)
    if provider == "huggingface":
        return _call_huggingface_router(request, payload)
    raise LLMProviderNotConfigured(f"LLM provider `{provider}` is not configured for this app yet.")


def _openai_client() -> Any:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMProviderNotConfigured("Set OPENAI_API_KEY to enable OpenAI interpretation.")
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise LLMProviderNotConfigured("The openai package is not installed. Run `pip install -r requirements.txt`.") from exc
    return OpenAI(api_key=api_key)


def _bedrock_openai_client() -> Any:
    try:
        from aws_bedrock_token_generator import provide_token
    except ImportError as exc:  # pragma: no cover
        raise LLMProviderNotConfigured("Install aws-bedrock-token-generator to use provider `bedrock`.") from exc
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise LLMProviderNotConfigured("The openai package is not installed. Run `pip install -r requirements.txt`.") from exc
    return OpenAI(api_key=provide_token(), base_url=BEDROCK_OPENAI_BASE_URL)


def _call_huggingface_router(request: LLMRequest, payload: dict[str, Any]) -> LLMResult:
    if payload.get("tools"):
        raise LLMProviderNotConfigured("Hugging Face interpretation does not support web search tools in this app.")
    api_key = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise LLMProviderNotConfigured("Set HF_TOKEN or HUGGINGFACE_API_KEY to use provider `huggingface`.")
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise LLMProviderNotConfigured("The openai package is not installed. Run `pip install -r requirements.txt`.") from exc
    client = OpenAI(api_key=api_key, base_url="https://router.huggingface.co/v1")
    messages = [
        {"role": "system", "content": str(payload.get("instructions") or "")},
        {"role": "user", "content": str(payload.get("input") or "")},
    ]
    response = client.chat.completions.create(model=request.model, messages=messages)
    output_text = str(response.choices[0].message.content or "")
    return LLMResult(provider="huggingface", model=request.model, output_text=output_text, response=response)


def _output_text(response: Any) -> str:
    return str(getattr(response, "output_text", "") or "").strip()
