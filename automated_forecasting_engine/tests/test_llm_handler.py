from __future__ import annotations

import json

from market_forecasting_engine.llm_handler import (
    BedrockOpenAIResponsesProvider,
    HuggingFaceChatProvider,
    LLMHandler,
    LLMProviderNotConfigured,
    LLMRequest,
    OpenAIResponsesProvider,
    OpenAICompatibleChatProvider,
    default_provider_registry,
    normalize_provider_name,
)
from market_forecasting_engine.openai_models import BEDROCK_OPENAI_BASE_URL, DEFAULT_BEDROCK_OPENAI_MODEL
from market_forecasting_engine.openai_responses import response_payload


class _FakeOpenAIResponse:
    output_text = '{"decision":"Hold"}'

    def model_dump(self, mode="json"):
        return {"id": "resp_1", "usage": {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14}}


class _FakeResponses:
    def create(self, **payload):
        self.payload = payload
        return _FakeOpenAIResponse()


class _FakeOpenAIClient:
    api_key = "sk-test"

    def __init__(self):
        self.responses = _FakeResponses()


class _FakeBedrockOpenAIFactory:
    def __init__(self):
        self.calls = []
        self.client = _FakeOpenAIClient()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.client


class _FakeHFMessage:
    content = '{"decision":"Buy"}'


class _FakeHFChoice:
    message = _FakeHFMessage()


class _FakeHFResponse:
    choices = [_FakeHFChoice()]

    def model_dump(self, mode="json"):
        return {"id": "chat_1", "usage": {"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18}, "choices": [{"message": {"content": '{"decision":"Buy"}'}}]}


class _FakeHFCompletions:
    def create(self, **payload):
        self.payload = payload
        return _FakeHFResponse()


class _FakeHFChat:
    def __init__(self):
        self.completions = _FakeHFCompletions()


class _FakeHFClient:
    api_key = "hf-test"

    def __init__(self):
        self.chat = _FakeHFChat()


def _payload():
    return response_payload(
        model="gpt-test",
        system_message="system",
        user_message="{{ item.text }}",
        json_schema={"type": "json_schema", "name": "decision_schema", "schema": {"type": "object"}},
        item={"text": "hello"},
    )


def test_llm_handler_routes_openai_and_logs_usage(tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "usage.jsonl"
    monkeypatch.setenv("OPENAI_USAGE_LOG_FILE", str(log_file))
    client = _FakeOpenAIClient()
    handler = LLMHandler({"openai": OpenAIResponsesProvider(client)})

    result = handler.predict(LLMRequest(provider="openai", model="gpt-test", payload=_payload(), usage_context={"purpose": "unit"}))

    assert result.parsed == {"decision": "Hold"}
    rows = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert rows[0]["status"] == "ok"
    assert rows[0]["model"] == "gpt-test"
    assert rows[0]["context"]["purpose"] == "unit"


def test_huggingface_provider_translates_responses_payload_to_chat_and_logs(tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "usage.jsonl"
    monkeypatch.setenv("OPENAI_USAGE_LOG_FILE", str(log_file))
    client = _FakeHFClient()
    handler = LLMHandler({"huggingface": HuggingFaceChatProvider(client=client)})

    result = handler.predict(LLMRequest(provider="hf", model="openai/gpt-oss-20b:cerebras", payload=_payload(), usage_context={"purpose": "hf_unit"}))

    assert result.parsed == {"decision": "Buy"}
    sent = client.chat.completions.payload
    assert sent["messages"][0] == {"role": "system", "content": "system"}
    assert sent["messages"][1] == {"role": "user", "content": "hello"}
    row = json.loads(log_file.read_text().splitlines()[0])
    assert row["provider"] == "huggingface"
    assert row["usage"]["total_tokens"] == 18


def test_bedrock_openai_provider_uses_bedrock_token_and_responses_api(tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "usage.jsonl"
    monkeypatch.setenv("OPENAI_USAGE_LOG_FILE", str(log_file))
    factory = _FakeBedrockOpenAIFactory()
    provider = BedrockOpenAIResponsesProvider(token_provider=lambda: "bedrock-short-term-token", openai_factory=factory)
    handler = LLMHandler({"bedrock": provider})

    result = handler.predict(LLMRequest(provider="bedrock_openai", model=DEFAULT_BEDROCK_OPENAI_MODEL, payload=_payload(), usage_context={"purpose": "bedrock_unit"}))

    assert result.parsed == {"decision": "Hold"}
    assert factory.calls == [{"api_key": "bedrock-short-term-token", "base_url": BEDROCK_OPENAI_BASE_URL}]
    assert factory.client.responses.payload["model"] == DEFAULT_BEDROCK_OPENAI_MODEL
    row = json.loads(log_file.read_text().splitlines()[0])
    assert row["provider"] == "bedrock"
    assert row["model"] == DEFAULT_BEDROCK_OPENAI_MODEL


def test_default_registry_has_future_provider_slots() -> None:
    providers = default_provider_registry(openai_client=_FakeOpenAIClient())

    assert {"openai", "huggingface", "bedrock", "llm_studio"}.issubset(set(providers))
    assert normalize_provider_name("HF") == "huggingface"
    assert normalize_provider_name("bedrock-openai") == "bedrock"


def test_llm_studio_provider_uses_openai_compatible_chat_endpoint(tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "usage.jsonl"
    monkeypatch.setenv("OPENAI_USAGE_LOG_FILE", str(log_file))
    client = _FakeHFClient()
    handler = LLMHandler({"llm_studio": OpenAICompatibleChatProvider(client=client, api_key="local-key", base_url="http://127.0.0.1:1234/v1")})

    result = handler.predict(LLMRequest(provider="lm_studio", model="local-trader", payload=_payload(), usage_context={"purpose": "local_unit"}))

    assert result.parsed == {"decision": "Buy"}
    sent = client.chat.completions.payload
    assert sent["model"] == "local-trader"
    assert sent["messages"][0] == {"role": "system", "content": "system"}
    row = json.loads(log_file.read_text().splitlines()[0])
    assert row["provider"] == "llm_studio"
    assert row["model"] == "local-trader"
