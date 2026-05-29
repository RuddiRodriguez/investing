import json

from openai import OpenAI
from dotenv import load_dotenv

from relevance.schemas import NewsEventInput, NewsRelevanceDecision

load_dotenv()
client = OpenAI()


def evaluate_news_relevance(event: NewsEventInput) -> NewsRelevanceDecision:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": """
You are a trading news relevance agent.
Your job is to decide whether a structured news event is relevant enough
to continue into sentiment analysis for a trading system.
You must be strict.
A news event is relevant if it may affect:
- revenue expectations
- earnings expectations
- margins
- demand
- supply
- regulation
- litigation
- management credibility
- product competitiveness
- analyst expectations
- sector positioning
- macro sensitivity
- investor risk perception
A news event is not relevant if it is:
- generic
- only weakly related to the company
- promotional without investor impact
- duplicate-looking
- vague
- not connected to future expectations
- too broad without clear company or sector effect
Use only the provided event fields.
Do not invent facts.
Return only the structured JSON.
"""
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "news_event_id": event.id,
                        "sector": event.sector,
                        "ticker": event.ticker,
                        "company_name": event.company_name,
                        "title": event.title,
                        "summary": event.summary,
                        "event_type": event.event_type,
                        "affected_direction": event.affected_direction,
                        "ingestion_relevance_score": event.relevance_score,
                        "ingestion_confidence": event.confidence,
                    },
                    indent=2,
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "news_relevance_decision",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "is_relevant": {
                            "type": "boolean"
                        },
                        "relevance_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "impact_horizon": {
                            "type": "string",
                            "enum": [
                                "intraday",
                                "short_term",
                                "medium_term",
                                "long_term",
                                "unknown"
                            ]
                        },
                        "affected_scope": {
                            "type": "string",
                            "enum": [
                                "single_company",
                                "sector",
                                "market",
                                "macro",
                                "unknown"
                            ]
                        },
                        "reason": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "is_relevant",
                        "relevance_score",
                        "impact_horizon",
                        "affected_scope",
                        "reason"
                    ]
                }
            }
        },
        temperature=0,
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
    return NewsRelevanceDecision.model_validate(data)
