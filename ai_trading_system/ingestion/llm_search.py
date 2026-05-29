import json

from openai import OpenAI
from dotenv import load_dotenv

from ingestion.schemas import (
    CompanyUniverse,
    NewsSearchResult,
    NewsEvent,
)

load_dotenv()
client = OpenAI()


def parse_json_response(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        print("\n--- INVALID JSON FROM LLM ---")
        print(text[:2000])
        print("\n--- JSON ERROR ---")
        print(error)
        raise


def discover_companies_by_sector(sector: str, limit: int = 40) -> CompanyUniverse:
    response = client.responses.create(
        model="gpt-4.1",
        max_output_tokens=16000,
        tools=[
            {
                "type": "web_search_preview"
            }
        ],
        input=[
            {
                "role": "system",
                "content": """
You are a financial market data ingestion agent.
Your task is to search the web and identify the most relevant publicly traded companies for a given sector.
Return only companies that are publicly traded and have a valid ticker symbol.
Rank them by relevance to the sector, prioritizing:
1. direct sector exposure
2. market importance
3. liquidity
4. global relevance
5. investor relevance
Do not include private companies.
Do not include ETFs.
Do not include funds.
Do not invent tickers.
"""
            },
            {
                "role": "user",
                "content": f"""
Sector: {sector}
Find the top {limit} most relevant publicly traded companies for this sector.
Return:
- rank
- ticker
- company_name
- exchange
- country
- industry
- relevance_reason
- source_url
"""
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "company_universe",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "sector": {
                            "type": "string"
                        },
                        "companies": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "rank": {
                                        "type": "integer",
                                        "minimum": 1
                                    },
                                    "ticker": {
                                        "type": "string"
                                    },
                                    "company_name": {
                                        "type": "string"
                                    },
                                    "exchange": {
                                        "type": ["string", "null"]
                                    },
                                    "country": {
                                        "type": ["string", "null"]
                                    },
                                    "industry": {
                                        "type": ["string", "null"]
                                    },
                                    "relevance_reason": {
                                        "type": "string"
                                    },
                                    "source_url": {
                                        "type": ["string", "null"]
                                    }
                                },
                                "required": [
                                    "rank",
                                    "ticker",
                                    "company_name",
                                    "exchange",
                                    "country",
                                    "industry",
                                    "relevance_reason",
                                    "source_url"
                                ]
                            }
                        }
                    },
                    "required": [
                        "sector",
                        "companies"
                    ]
                }
            }
        },
        temperature=0
    )
    data = json.loads(response.output_text)
    universe = CompanyUniverse.model_validate(data)
    universe.companies = universe.companies[:limit]
    return universe


def search_latest_news_for_universe(
    sector: str,
    companies: list[dict],
    max_articles_per_company: int = 2,
) -> NewsSearchResult:
    max_total_articles = len(companies) * max_articles_per_company
    response = client.responses.create(
        model="gpt-4.1",
        max_output_tokens=6000,
        tools=[
            {
                "type": "web_search_preview"
            }
        ],
        input=[
            {
                "role": "system",
                "content": """
You are a financial news ingestion agent.
Search the web for recent, relevant news for the provided public companies.
Return only articles that are relevant for investors or sector analysis.
Rules:
- Return at most the requested number of articles.
- Avoid duplicate articles.
- Avoid generic company pages.
- Avoid opinion-only content unless it contains market-relevant information.
- Each article must have a URL.
- raw_summary must be maximum 240 characters.
- Do not include full article text.
- Do not include markdown.
- Return only valid JSON matching the schema.
"""
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "sector": sector,
                        "max_articles_per_company": max_articles_per_company,
                        "max_total_articles": max_total_articles,
                        "companies": companies,
                    },
                    indent=2,
                )
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "news_search_result",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "sector": {
                            "type": "string"
                        },
                        "articles": {
                            "type": "array",
                            "maxItems": max_total_articles,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "ticker": {
                                        "type": "string"
                                    },
                                    "company_name": {
                                        "type": "string"
                                    },
                                    "source": {
                                        "type": ["string", "null"]
                                    },
                                    "title": {
                                        "type": "string",
                                        "maxLength": 300
                                    },
                                    "url": {
                                        "type": "string"
                                    },
                                    "published_at": {
                                        "type": ["string", "null"]
                                    },
                                    "raw_summary": {
                                        "type": ["string", "null"],
                                        "maxLength": 300
                                    }
                                },
                                "required": [
                                    "ticker",
                                    "company_name",
                                    "source",
                                    "title",
                                    "url",
                                    "published_at",
                                    "raw_summary"
                                ]
                            }
                        }
                    },
                    "required": [
                        "sector",
                        "articles"
                    ]
                }
            }
        },
        temperature=0
    )
    text_content = response.output_text
    if not text_content:
        raise ValueError("No text content found in response")
    data = parse_json_response(text_content)
    return NewsSearchResult.model_validate(data)


def structure_news_event(
    sector: str,
    ticker: str,
    company_name: str,
    title: str,
    raw_summary: str | None,
) -> NewsEvent:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": """
You are a trading data normalization agent.
Convert the raw news article metadata into a structured trading event.
Use only the provided title and summary.
Do not invent facts.
"""
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "sector": sector,
                        "ticker": ticker,
                        "company_name": company_name,
                        "title": title,
                        "raw_summary": raw_summary,
                    },
                    indent=2,
                )
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "news_event",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "summary": {
                            "type": "string"
                        },
                        "event_type": {
                            "type": "string",
                            "enum": [
                                "earnings",
                                "guidance",
                                "demand_growth",
                                "demand_weakness",
                                "regulation",
                                "export_restriction",
                                "supply_chain",
                                "product_launch",
                                "analyst_upgrade",
                                "analyst_downgrade",
                                "macro",
                                "geopolitical",
                                "management_change",
                                "legal",
                                "other"
                            ]
                        },
                        "affected_direction": {
                            "type": "string",
                            "enum": [
                                "positive",
                                "negative",
                                "mixed",
                                "neutral",
                                "unknown"
                            ]
                        },
                        "relevance_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        }
                    },
                    "required": [
                        "summary",
                        "event_type",
                        "affected_direction",
                        "relevance_score",
                        "confidence"
                    ]
                }
            }
        },
        temperature=0
    )
    # Extract text from response.output
    text_content = None
    for item in response.output:
        if hasattr(item, 'content') and isinstance(item.content, list):
            for content_item in item.content:
                if hasattr(content_item, 'text'):
                    text_content = content_item.text
                    break
        if text_content:
            break
    if not text_content:
        raise ValueError("No text content found in response")
    data = json.loads(text_content)
    return NewsEvent.model_validate(data)
