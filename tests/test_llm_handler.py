from src.llm_handler import (
    BEDROCK_OPENAI_BASE_URL,
    DEFAULT_BEDROCK_OPENAI_MODEL,
    BEDROCK_OPENAI_MODELS,
    normalize_provider_name,
    resolve_llm_model,
)


def test_bedrock_openai_provider_resolves_utils_model_ids(monkeypatch):
    monkeypatch.delenv("BEDROCK_MODEL", raising=False)
    monkeypatch.delenv("BEDROCK_OPENAI_MODEL", raising=False)

    assert normalize_provider_name("bedrock-openai") == "bedrock"
    assert resolve_llm_model(None, "bedrock") == DEFAULT_BEDROCK_OPENAI_MODEL
    assert DEFAULT_BEDROCK_OPENAI_MODEL in BEDROCK_OPENAI_MODELS
    assert BEDROCK_OPENAI_BASE_URL == "https://bedrock-mantle.us-east-2.api.aws/openai/v1"
