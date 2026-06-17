import json

from market_forecasting_engine.llm_handler import LLMRequest, default_llm_handler
from market_forecasting_engine.openai_models import DEFAULT_REASONING_EFFORT, is_reasoning_model


def render_template(template, item):
    rendered = template
    for key, value in item.items():
        rendered = rendered.replace(f"{{{{ item.{key} }}}}", str(value))
    return rendered


def response_payload(
    model,
    system_message,
    user_message,
    json_schema,
    reasoning_effort,
    item,
    use_web_search,
    search_context_size,
    require_web_search=False,
):
    tools = []
    if use_web_search:
        tools = [
            {
                "type": "web_search",
                "search_context_size": search_context_size,
                "external_web_access": True,
                "user_location": {
                    "type": "approximate",
                    "country": "US",
                    "timezone": "Europe/Amsterdam",
                },
            }
        ]
    payload = {
        "model": model,
        "tool_choice": "required" if use_web_search and require_web_search else "auto",
        "text": {"format": json_schema, "verbosity": "medium"},
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": render_template(user_message, item)}],
            },
        ],
        "store": False,
        "tools": tools,
    }
    if use_web_search:
        payload["include"] = ["web_search_call.action.sources"]
    if is_reasoning_model(model):
        payload["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
    return payload


def response_json(response):
    return response.model_dump(mode="json")


def call_response(
    model,
    system_message,
    user_message,
    json_schema,
    item,
    use_web_search,
    search_context_size,
    client=None,
    provider="openai",
    reasoning_effort=DEFAULT_REASONING_EFFORT,
    usage_context=None,
    require_web_search=False,
):
    payload = response_payload(
        model=model,
        system_message=system_message,
        user_message=user_message,
        json_schema=json_schema,
        reasoning_effort=reasoning_effort,
        item=item,
        use_web_search=use_web_search,
        search_context_size=search_context_size,
        require_web_search=require_web_search,
    )
    result = default_llm_handler(openai_client=client if provider == "openai" else None).predict(
        LLMRequest(provider=provider, model=model, payload=payload, usage_context=usage_context or {})
    )
    return payload, result.response_data, result.parsed
