from __future__ import annotations

import json

import market_forecasting_engine.llm_handler as llm_handler
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
    reasoning_content = None


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
    assert sent["response_format"]["type"] == "json_schema"
    row = json.loads(log_file.read_text().splitlines()[0])
    assert row["provider"] == "llm_studio"
    assert row["model"] == "local-trader"


def test_llm_studio_provider_reads_structured_json_from_reasoning_content(tmp_path, monkeypatch) -> None:
    class Message:
        content = ""
        reasoning_content = '{"decision":"Hold"}'

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

        def model_dump(self, mode="json"):
            return {
                "id": "chat_reasoning_1",
                "choices": [{"message": {"content": "", "reasoning_content": '{"decision":"Hold"}'}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }

    class Completions:
        def create(self, **payload):
            self.payload = payload
            return Response()

    class Chat:
        def __init__(self):
            self.completions = Completions()

    class Client:
        api_key = "local-key"

        def __init__(self):
            self.chat = Chat()

    log_file = tmp_path / "usage.jsonl"
    monkeypatch.setenv("OPENAI_USAGE_LOG_FILE", str(log_file))
    client = Client()
    handler = LLMHandler({"llm_studio": OpenAICompatibleChatProvider(client=client, api_key="local-key", base_url="http://127.0.0.1:1234/v1")})

    result = handler.predict(LLMRequest(provider="llm_studio", model="local-trader", payload=_payload(), usage_context={"purpose": "local_reasoning_unit"}))

    assert result.parsed == {"decision": "Hold"}
    assert client.chat.completions.payload["response_format"]["type"] == "json_schema"


def test_llm_studio_provider_executes_local_web_search_tool_loop(tmp_path, monkeypatch) -> None:
    class ToolLoopCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **payload):
            self.calls.append(payload)
            if len(self.calls) == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "web_search",
                                            "arguments": json.dumps({"query": "J stock news today", "max_results": 3}),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                }
            return {
                "choices": [{"message": {"content": '{"decision":"Hold","freshness":"used_tool"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            }

    class Chat:
        def __init__(self):
            self.completions = ToolLoopCompletions()

    class Client:
        api_key = "local-key"

        def __init__(self):
            self.chat = Chat()

    monkeypatch.setattr(
        llm_handler,
        "local_web_search",
        lambda *, query, max_results=6: {"status": "ok", "query": query, "results": [{"title": "Jacobs news"}]},
    )
    log_file = tmp_path / "usage.jsonl"
    monkeypatch.setenv("OPENAI_USAGE_LOG_FILE", str(log_file))
    client = Client()
    provider = OpenAICompatibleChatProvider(client=client, api_key="local-key", base_url="http://127.0.0.1:1234/v1")
    payload = {
        **_payload(),
        "tools": [{"type": "web_search", "search_context_size": "low"}],
    }
    handler = LLMHandler({"llm_studio": provider})

    result = handler.predict(LLMRequest(provider="llm_studio", model="qwen3-vl-8b-instruct-mlx", payload=payload, usage_context={"purpose": "local_tool_unit"}))

    assert result.parsed == {"decision": "Hold", "freshness": "used_tool"}
    assert len(client.chat.completions.calls) == 2
    assert client.chat.completions.calls[0]["tools"][0]["function"]["name"] == "web_search"
    assert client.chat.completions.calls[1]["messages"][-1]["role"] == "tool"
    assert json.loads(client.chat.completions.calls[1]["messages"][-1]["content"])["results"][0]["title"] == "Jacobs news"
    row = json.loads(log_file.read_text().splitlines()[0])
    assert row["provider"] == "llm_studio"
    assert row["context"]["purpose"] == "local_tool_unit"
