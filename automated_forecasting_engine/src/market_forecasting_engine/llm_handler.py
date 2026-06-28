from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from market_forecasting_engine.llm_usage import log_llm_usage, log_openai_usage, monotonic_ms, new_llm_call_id
from market_forecasting_engine.llm_model_catalog import (
    DEFAULT_LLM_STUDIO_BASE_URL,
    LLMModelProfile,
    normalize_llm_provider,
    resolve_llm_model_profile,
)
from market_forecasting_engine.openai_models import BEDROCK_OPENAI_BASE_URL


@dataclass(frozen=True)
class LLMRequest:
    provider: str
    model: str
    payload: dict[str, Any]
    usage_context: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class LLMResult:
    provider: str
    model: str
    payload: dict[str, Any]
    response_data: dict[str, Any]
    parsed: dict[str, Any]
    output_text: str | None = None
    api_key: str | None = None


class LLMProvider(Protocol):
    name: str

    def generate(self, request: LLMRequest) -> LLMResult:
        ...


class LLMProviderNotConfigured(RuntimeError):
    pass


class OpenAIResponsesProvider:
    name = "openai"

    def __init__(self, client: Any):
        if client is None:
            raise ValueError("OpenAIResponsesProvider requires an OpenAI-compatible client.")
        self.client = client

    def generate(self, request: LLMRequest) -> LLMResult:
        payload = payload_with_model(request.payload, request.model)
        response = self.client.responses.create(**payload)
        return responses_result(
            provider=self.name,
            model=request.model,
            payload=payload,
            response=response,
            api_key=getattr(self.client, "api_key", None),
        )


class BedrockOpenAIResponsesProvider:
    name = "bedrock"

    def __init__(
        self,
        client: Any | None = None,
        *,
        base_url: str = BEDROCK_OPENAI_BASE_URL,
        token_provider: Callable[[], str] | None = None,
        openai_factory: Callable[..., Any] | None = None,
    ):
        self.client = client
        self.base_url = base_url
        self.token_provider = token_provider
        self.openai_factory = openai_factory

    def generate(self, request: LLMRequest) -> LLMResult:
        client, api_key = self._client()
        payload = payload_with_model(request.payload, request.model)
        response = client.responses.create(**payload)
        return responses_result(
            provider=self.name,
            model=request.model,
            payload=payload,
            response=response,
            api_key=api_key or getattr(client, "api_key", None),
        )

    def _client(self) -> tuple[Any, str | None]:
        if self.client is not None:
            return self.client, getattr(self.client, "api_key", None)
        token_provider = self.token_provider
        if token_provider is None:
            try:
                from aws_bedrock_token_generator import provide_token
            except ImportError as exc:
                raise LLMProviderNotConfigured("Install aws-bedrock-token-generator to use provider `bedrock`.") from exc
            token_provider = provide_token
        factory = self.openai_factory
        if factory is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMProviderNotConfigured("Install the openai package to use provider `bedrock`.") from exc
            factory = OpenAI
        token = token_provider()
        return factory(api_key=token, base_url=self.base_url), token


class HuggingFaceChatProvider:
    name = "huggingface"

    def __init__(self, client: Any | None = None, *, api_key: str | None = None, base_url: str = "https://router.huggingface.co/v1"):
        self.client = client
        self.api_key = api_key
        self.base_url = base_url

    def generate(self, request: LLMRequest) -> LLMResult:
        client = self.client or self._client()
        payload = payload_with_model(request.payload, request.model)
        if payload.get("tools"):
            raise LLMProviderNotConfigured("Hugging Face chat provider does not support Responses API tools in this framework yet.")
        messages = responses_payload_to_chat_messages(payload)
        if _is_huggingface_structured_output_unsupported(request.model):
            messages = _messages_with_json_instruction(messages, payload.get("text"))
        create_payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": payload.get("temperature"),
        }
        response_format = None if _is_huggingface_structured_output_unsupported(request.model) else _chat_response_format(payload.get("text"))
        if response_format is not None:
            create_payload["response_format"] = response_format
        create_payload = {key: value for key, value in create_payload.items() if value is not None}
        response = client.chat.completions.create(**create_payload)
        response_data = response_json(response)
        output_text = _chat_output_text(response, response_data)
        return LLMResult(
            provider=self.name,
            model=request.model,
            payload=create_payload,
            response_data=response_data,
            parsed=parse_json_object(output_text),
            output_text=output_text,
            api_key=getattr(client, "api_key", None) or self.api_key,
        )

    def _client(self) -> Any:
        api_key = self.api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise LLMProviderNotConfigured("HF_TOKEN or HUGGINGFACE_API_KEY is required for provider `huggingface`.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMProviderNotConfigured("Install the openai package to use the Hugging Face router provider.") from exc
        return OpenAI(api_key=api_key, base_url=self.base_url)


class OpenAICompatibleChatProvider:
    name = "llm_studio"

    def __init__(
        self,
        client: Any | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        provider_name: str = "llm_studio",
        use_response_format: bool | None = None,
    ):
        self.client = client
        self.api_key = api_key
        self.base_url = base_url or os.getenv("LLM_STUDIO_BASE_URL") or os.getenv("LOCAL_LLM_BASE_URL") or DEFAULT_LLM_STUDIO_BASE_URL
        self.name = normalize_provider_name(provider_name)
        self.use_response_format = use_response_format

    def generate(self, request: LLMRequest) -> LLMResult:
        client = self.client or self._client()
        payload = payload_with_model(request.payload, request.model)
        if payload.get("tools"):
            return self._generate_with_tools(client=client, request=request, payload=payload)
        messages = responses_payload_to_chat_messages(payload)
        if not self._should_use_response_format():
            messages = _messages_with_json_instruction(messages, payload.get("text"))
        create_payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": payload.get("temperature"),
        }
        if self._should_use_response_format():
            response_format = _chat_response_format(payload.get("text"))
            if response_format is not None:
                create_payload["response_format"] = response_format
        create_payload = {key: value for key, value in create_payload.items() if value is not None}
        response = client.chat.completions.create(**create_payload)
        response_data = response_json(response)
        output_text = _chat_output_text(response, response_data)
        return LLMResult(
            provider=self.name,
            model=request.model,
            payload=create_payload,
            response_data=response_data,
            parsed=parse_json_object(output_text),
            output_text=output_text,
            api_key=getattr(client, "api_key", None) or self.api_key or os.getenv("LLM_STUDIO_API_KEY") or "local",
        )

    def _generate_with_tools(self, *, client: Any, request: LLMRequest, payload: dict[str, Any]) -> LLMResult:
        chat_tools = _local_chat_tools_from_responses_tools(payload.get("tools") or [])
        if not chat_tools:
            raise LLMProviderNotConfigured(f"{self.name} provider received unsupported tools.")
        messages = responses_payload_to_chat_messages(payload)
        if not self._should_use_response_format():
            messages = _messages_with_json_instruction(messages, payload.get("text"))
        create_payload = self._chat_create_payload(model=request.model, messages=messages, payload=payload, tools=chat_tools)
        first_response = client.chat.completions.create(**create_payload)
        first_data = response_json(first_response)
        tool_calls = _chat_tool_calls(first_data)
        if not tool_calls:
            output_text = _chat_output_text(first_response, first_data)
            return LLMResult(
                provider=self.name,
                model=request.model,
                payload=create_payload,
                response_data=first_data,
                parsed=parse_json_object(output_text),
                output_text=output_text,
                api_key=getattr(client, "api_key", None) or self.api_key or os.getenv("LLM_STUDIO_API_KEY") or "local",
            )
        messages = [*messages, _assistant_tool_call_message(first_data)]
        tool_results = []
        for tool_call in tool_calls:
            tool_result = _execute_local_tool_call(tool_call)
            tool_results.append(tool_result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id") or tool_call.get("index") or "tool_call",
                    "content": json.dumps(tool_result, separators=(",", ":"), default=str),
                }
            )
        final_payload = self._chat_create_payload(model=request.model, messages=messages, payload=payload, tools=chat_tools)
        final_response = client.chat.completions.create(**final_payload)
        final_data = response_json(final_response)
        output_text = _chat_output_text(final_response, final_data)
        response_data = {
            "tool_loop": {
                "first_response": first_data,
                "tool_results": tool_results,
                "final_response": final_data,
            }
        }
        return LLMResult(
            provider=self.name,
            model=request.model,
            payload={**final_payload, "local_tool_loop": {"enabled": True, "tools": [tool.get("function", {}).get("name") for tool in chat_tools]}},
            response_data=response_data,
            parsed=parse_json_object(output_text),
            output_text=output_text,
            api_key=getattr(client, "api_key", None) or self.api_key or os.getenv("LLM_STUDIO_API_KEY") or "local",
        )

    def _chat_create_payload(self, *, model: str, messages: list[dict[str, Any]], payload: dict[str, Any], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        create_payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": payload.get("temperature"),
        }
        if tools:
            create_payload["tools"] = tools
            create_payload["tool_choice"] = "auto"
        if self._should_use_response_format():
            response_format = _chat_response_format(payload.get("text"))
            if response_format is not None:
                create_payload["response_format"] = response_format
        return {key: value for key, value in create_payload.items() if value is not None}

    def _client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMProviderNotConfigured("Install the openai package to use the OpenAI-compatible local provider.") from exc
        return OpenAI(api_key=self.api_key or os.getenv("LLM_STUDIO_API_KEY") or os.getenv("LOCAL_LLM_API_KEY") or "local", base_url=self.base_url)

    def _should_use_response_format(self) -> bool:
        if self.use_response_format is not None:
            return bool(self.use_response_format)
        value = os.getenv("LLM_STUDIO_USE_RESPONSE_FORMAT") or os.getenv("LOCAL_LLM_USE_RESPONSE_FORMAT")
        if value is None:
            return True
        return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


class NotConfiguredProvider:
    def __init__(self, name: str, install_hint: str):
        self.name = name
        self.install_hint = install_hint

    def generate(self, request: LLMRequest) -> LLMResult:
        raise LLMProviderNotConfigured(f"LLM provider `{self.name}` is not configured. {self.install_hint}")


class LLMHandler:
    """Provider router for every LLM call in this repo.

    New model vendors should be added as providers here, not by scattering
    direct SDK calls through trading code. Usage logging is part of the route.
    """

    def __init__(self, providers: dict[str, LLMProvider], fallback_provider: str | None = None):
        if not providers:
            raise ValueError("LLMHandler requires at least one provider.")
        self.providers = {normalize_provider_name(name): provider for name, provider in providers.items()}
        self.fallback_provider = normalize_provider_name(fallback_provider) if fallback_provider else None

    def predict(self, request: LLMRequest) -> LLMResult:
        provider_name = normalize_provider_name(request.provider)
        provider = self.providers.get(provider_name)
        if provider is None:
            available = ", ".join(sorted(self.providers))
            raise LLMProviderNotConfigured(f"LLM provider `{provider_name}` is not registered. Available providers: {available}.")
        try:
            return self._predict_with_provider(provider=provider, request=request)
        except Exception:
            if not self.fallback_provider or self.fallback_provider == provider_name:
                raise
            fallback = self.providers.get(self.fallback_provider)
            if fallback is None:
                raise
            fallback_request = LLMRequest(
                provider=self.fallback_provider,
                model=request.model,
                payload=request.payload,
                usage_context={**request.usage_context, "fallback_from_provider": provider_name},
                timeout_seconds=request.timeout_seconds,
            )
            return self._predict_with_provider(provider=fallback, request=fallback_request)

    def _predict_with_provider(self, *, provider: LLMProvider, request: LLMRequest) -> LLMResult:
        call_id = new_llm_call_id()
        started_ms = monotonic_ms()
        provider_name = normalize_provider_name(provider.name)
        try:
            result = provider.generate(request)
            log_provider_usage(
                call_id=call_id,
                provider=provider_name,
                model=request.model,
                payload=result.payload,
                response_data=result.response_data,
                started_ms=started_ms,
                status="ok",
                context=request.usage_context,
                api_key=result.api_key,
            )
            return result
        except Exception as exc:
            log_provider_usage(
                call_id=call_id,
                provider=provider_name,
                model=request.model,
                payload=request.payload,
                started_ms=started_ms,
                status="error",
                error=str(exc),
                context=request.usage_context,
            )
            raise


def default_provider_registry(
    *,
    openai_client: Any | None = None,
    huggingface_client: Any | None = None,
    bedrock_client: Any | None = None,
) -> dict[str, LLMProvider]:
    providers: dict[str, LLMProvider] = {
        "bedrock": BedrockOpenAIResponsesProvider(client=bedrock_client),
        "huggingface": HuggingFaceChatProvider(client=huggingface_client),
        "llm_studio": OpenAICompatibleChatProvider(),
    }
    if openai_client is not None:
        providers["openai"] = OpenAIResponsesProvider(openai_client)
    return providers


def default_llm_handler(
    *,
    openai_client: Any | None = None,
    huggingface_client: Any | None = None,
    bedrock_client: Any | None = None,
    fallback_provider: str | None = None,
) -> LLMHandler:
    return LLMHandler(
        default_provider_registry(openai_client=openai_client, huggingface_client=huggingface_client, bedrock_client=bedrock_client),
        fallback_provider=fallback_provider,
    )


def normalize_provider_name(value: str | None) -> str:
    return normalize_llm_provider(value)


def resolve_llm_client_profile(provider: str | None = None, model: str | None = None) -> LLMModelProfile:
    return resolve_llm_model_profile(provider=provider, model=model)


def response_json(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        data = response.model_dump(mode="json")
        return data if isinstance(data, dict) else {}
    if isinstance(response, dict):
        return response
    return {}


def payload_with_model(payload: dict[str, Any], model: str) -> dict[str, Any]:
    return {**payload, "model": model}


def responses_result(
    *,
    provider: str,
    model: str,
    payload: dict[str, Any],
    response: Any,
    api_key: str | None = None,
) -> LLMResult:
    response_data = response_json(response)
    output_text = str(getattr(response, "output_text", "") or "")
    parsed = parse_json_object(output_text)
    return LLMResult(
        provider=provider,
        model=model,
        payload=payload,
        response_data=response_data,
        parsed=parsed,
        output_text=output_text,
        api_key=api_key,
    )


def responses_payload_to_chat_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in payload.get("input") or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user")
        if role == "developer":
            role = "system"
        content_parts = item.get("content") or []
        text_parts: list[str] = []
        for part in content_parts if isinstance(content_parts, list) else [content_parts]:
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = part
            if text is not None:
                text_parts.append(str(text))
        messages.append({"role": role, "content": "\n".join(text_parts)})
    return messages


def parse_json_object(output_text: str) -> dict[str, Any]:
    text = str(output_text or "{}").strip() or "{}"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM structured output JSON must be an object.")
    return parsed


def _chat_response_format(text_config: Any) -> dict[str, Any] | None:
    if not isinstance(text_config, dict):
        return {"type": "json_object"}
    json_schema = text_config.get("format")
    if not isinstance(json_schema, dict):
        return {"type": "json_object"}
    chat_schema = dict(json_schema)
    chat_schema.pop("type", None)
    return {"type": "json_schema", "json_schema": chat_schema}


def _is_huggingface_structured_output_unsupported(model: str) -> bool:
    normalized = str(model or "").lower()
    return "kimi" in normalized


def _messages_with_json_instruction(messages: list[dict[str, str]], text_config: Any) -> list[dict[str, str]]:
    schema = None
    if isinstance(text_config, dict) and isinstance(text_config.get("format"), dict):
        schema = text_config.get("format", {}).get("schema")
    instruction = "Return exactly one valid JSON object. Do not include markdown, prose, or code fences."
    if schema:
        instruction += f"\nThe JSON object must follow this schema intent: {json.dumps(schema, sort_keys=True, default=str)}"
    if not messages:
        return [{"role": "system", "content": instruction}]
    updated = [dict(item) for item in messages]
    updated[0]["content"] = f"{updated[0].get('content', '')}\n\n{instruction}".strip()
    return updated


def _chat_output_text(response: Any, response_data: dict[str, Any]) -> str:
    choices = response_data.get("choices") if isinstance(response_data, dict) else None
    if isinstance(choices, list) and choices:
        message = (choices[0] or {}).get("message") if isinstance(choices[0], dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
        if content is not None:
            text = str(content)
            if text.strip():
                return text
        reasoning_content = message.get("reasoning_content") if isinstance(message, dict) else None
        if reasoning_content is not None:
            text = str(reasoning_content)
            if text.strip():
                return text
    try:
        return str(response.choices[0].message.content)
    except Exception:
        return ""


def _local_chat_tools_from_responses_tools(tools: list[Any]) -> list[dict[str, Any]]:
    chat_tools: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "web_search":
            chat_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search current web/news results for market, stock, company, macro, earnings, analyst, and regulatory context.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query."},
                                "max_results": {"type": "integer", "description": "Maximum number of results to return.", "default": 6},
                            },
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    },
                }
            )
    return chat_tools


def _chat_tool_calls(response_data: dict[str, Any]) -> list[dict[str, Any]]:
    choices = response_data.get("choices") if isinstance(response_data, dict) else None
    if not isinstance(choices, list) or not choices:
        return []
    message = (choices[0] or {}).get("message") if isinstance(choices[0], dict) else {}
    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
    return [call for call in tool_calls if isinstance(call, dict)] if isinstance(tool_calls, list) else []


def _assistant_tool_call_message(response_data: dict[str, Any]) -> dict[str, Any]:
    choices = response_data.get("choices") if isinstance(response_data, dict) else None
    message = (choices[0] or {}).get("message") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    if not isinstance(message, dict):
        message = {}
    return {
        "role": "assistant",
        "content": message.get("content") or "",
        "tool_calls": message.get("tool_calls") or [],
    }


def _execute_local_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    name = str(function.get("name") or tool_call.get("name") or "")
    raw_arguments = function.get("arguments") or tool_call.get("arguments") or "{}"
    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else dict(raw_arguments)
    except Exception:
        arguments = {}
    if name != "web_search":
        return {"status": "unsupported_tool", "tool": name}
    query = str(arguments.get("query") or "").strip()
    max_results = _bounded_int(arguments.get("max_results"), default=6, low=1, high=10)
    return local_web_search(query=query, max_results=max_results)


def local_web_search(*, query: str, max_results: int = 6) -> dict[str, Any]:
    if not query:
        return {"status": "error", "provider": "yahoo_news_rss", "query": query, "error": "empty_query", "results": []}
    url = "https://news.search.yahoo.com/rss?" + urlencode({"p": query})
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 market-forecasting-engine-local-tool/1.0"})
    try:
        with urlopen(request, timeout=12) as response:
            raw = response.read(400_000)
    except Exception as exc:
        return {"status": "error", "provider": "yahoo_news_rss", "query": query, "error": f"{type(exc).__name__}: {exc}", "results": []}
    try:
        root = ElementTree.fromstring(raw)
    except Exception as exc:
        return {"status": "parse_error", "provider": "yahoo_news_rss", "query": query, "error": f"{type(exc).__name__}: {exc}", "results": []}
    results = []
    for item_node in root.findall(".//item")[: max(1, int(max_results))]:
        results.append(
            {
                "title": _clean_text(item_node.findtext("title")),
                "link": _clean_text(item_node.findtext("link")),
                "published": _clean_text(item_node.findtext("pubDate")),
                "summary": _clean_text(item_node.findtext("description")),
            }
        )
    return {
        "status": "ok" if results else "empty",
        "provider": "yahoo_news_rss",
        "query": query,
        "results": results,
        "limitations": ["Bounded Yahoo News RSS search executed locally by Python for an LM Studio tool call."],
    }


def _bounded_int(value: Any, *, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(low, min(high, parsed))


def _clean_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def log_provider_usage(
    *,
    call_id: str,
    provider: str,
    model: str,
    payload: dict[str, Any],
    response_data: dict[str, Any] | None = None,
    started_ms: float | None = None,
    status: str = "ok",
    error: str | None = None,
    context: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> None:
    provider_name = normalize_provider_name(provider)
    if provider_name == "openai":
        log_openai_usage(
            call_id=call_id,
            model=model,
            payload=payload,
            response_data=response_data,
            started_ms=started_ms,
            status=status,
            error=error,
            context=context,
            api_key=api_key,
        )
        return
    log_llm_usage(
        call_id=call_id,
        provider=provider_name,
        model=model,
        payload=payload,
        response_data=response_data,
        started_ms=started_ms,
        status=status,
        error=error,
        context=context,
        api_key=api_key or os.getenv(f"{provider_name.upper()}_API_KEY"),
    )
