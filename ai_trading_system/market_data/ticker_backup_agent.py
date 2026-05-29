import json

from dotenv import load_dotenv
from openai import OpenAI

from ingestion.cache import build_cache_key, get_cache_value, set_cache

load_dotenv()
client = OpenAI()


def resolve_ticker_with_backup_agent(
    sector: str,
    company_name: str,
    bad_ticker: str,
) -> str | None:
    cache_key = build_cache_key(
        "ticker_correction",
        sector,
        company_name,
        bad_ticker,
    )
    cached_value = get_cache_value(cache_key)
    if cached_value is not None:
        if cached_value == "__none__":
            return None
        return cached_value

    response = client.responses.create(
        model="gpt-4.1-mini",
        tools=[
            {
                "type": "web_search_preview",
            }
        ],
        input=[
            {
                "role": "system",
                "content": """
You are a ticker correction backup agent for market data ingestion.
Given a company name, sector, and possibly incorrect ticker, return the most likely Yahoo Finance ticker.
Rules:
- Return only one ticker symbol if high confidence.
- If uncertain, return null.
- Prefer ticker formats accepted by Yahoo Finance.
- Do not fabricate a ticker.
""",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "sector": sector,
                        "company_name": company_name,
                        "bad_ticker": bad_ticker,
                    }
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "ticker_correction",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "corrected_ticker": {
                            "type": ["string", "null"],
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "reason": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "corrected_ticker",
                        "confidence",
                        "reason",
                    ],
                },
            }
        },
        temperature=0,
    )

    try:
        data = json.loads(response.output_text)
    except Exception:
        set_cache(
            cache_key=cache_key,
            cache_type="ticker_correction",
            value="__none__",
            ttl_hours=24,
        )
        return None

    corrected = data.get("corrected_ticker")
    confidence = data.get("confidence", 0)

    if not corrected or not isinstance(corrected, str):
        set_cache(
            cache_key=cache_key,
            cache_type="ticker_correction",
            value="__none__",
            ttl_hours=24,
        )
        return None

    corrected = corrected.strip().upper()
    if confidence < 0.6 or corrected == bad_ticker.upper():
        set_cache(
            cache_key=cache_key,
            cache_type="ticker_correction",
            value="__none__",
            ttl_hours=24,
        )
        return None

    set_cache(
        cache_key=cache_key,
        cache_type="ticker_correction",
        value=corrected,
        ttl_hours=24 * 7,
    )
    return corrected
