from __future__ import annotations

import json
from typing import Any

from market_forecasting_engine.llm_handler import LLMRequest, default_llm_handler, response_json
from market_forecasting_engine.openai_models import DEFAULT_REASONING_EFFORT, is_reasoning_model


def render_template(template: str, item: dict[str, Any]) -> str:
    rendered = template
    for key, value in item.items():
        rendered = rendered.replace(f"{{{{ item.{key} }}}}", str(value))
    return rendered


def response_payload(
    *,
    model: str,
    system_message: str,
    user_message: str,
    json_schema: dict[str, Any],
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    item: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "tool_choice": "auto",
        "text": {"format": json_schema, "verbosity": "medium"},
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": render_template(user_message, item or {})}],
            },
        ],
        "store": False,
        "tools": tools or [],
    }
    if is_reasoning_model(model):
        payload["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
    return payload


def parse_response_output_text(response: Any) -> dict[str, Any]:
    text = str(getattr(response, "output_text", "") or "").strip()
    parsed = json.loads(text or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("Responses API output JSON must be an object.")
    return parsed


def call_response(
    *,
    client: Any | None = None,
    provider: str = "openai",
    model: str,
    system_message: str,
    user_message: str,
    json_schema: dict[str, Any],
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    item: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    usage_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = response_payload(
        model=model,
        system_message=system_message,
        user_message=user_message,
        json_schema=json_schema,
        reasoning_effort=reasoning_effort,
        item=item,
        tools=tools,
    )
    result = default_llm_handler(openai_client=client if provider == "openai" else None).predict(
        LLMRequest(provider=provider, model=model, payload=payload, usage_context=usage_context or {})
    )
    return payload, result.response_data, result.parsed
