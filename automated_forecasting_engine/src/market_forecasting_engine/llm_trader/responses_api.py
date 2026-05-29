import json

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
):
    tools = []
    if use_web_search:
        tools = [
            {
                "type": "web_search",
                "search_context_size": search_context_size,
                "user_location": {
                    "type": "approximate",
                    "country": "US",
                    "timezone": "Europe/Amsterdam",
                },
            }
        ]
    payload = {
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
                "content": [{"type": "input_text", "text": render_template(user_message, item)}],
            },
        ],
        "store": False,
        "tools": tools,
    }
    if is_reasoning_model(model):
        payload["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
    return payload


def response_json(response):
    return response.model_dump(mode="json")


def call_response(
    client,
    model,
    system_message,
    user_message,
    json_schema,
    item,
    use_web_search,
    search_context_size,
    reasoning_effort=DEFAULT_REASONING_EFFORT,
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
    )
    response = client.responses.create(**payload)
    data = response_json(response)
    parsed = json.loads(response.output_text)
    return payload, data, parsed
