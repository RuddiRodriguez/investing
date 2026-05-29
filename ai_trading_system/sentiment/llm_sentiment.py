import json

from openai import OpenAI
from dotenv import load_dotenv

from sentiment.schemas import (
    RelevantNewsEventInput,
    NewsSentimentDecision,
)

load_dotenv()
client = OpenAI()


def evaluate_news_sentiment(event: RelevantNewsEventInput) -> NewsSentimentDecision:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": """
You are a trading sentiment agent.
Your job is to evaluate the sentiment impact of a relevant news event.
This sentiment is not generic public opinion.
It is trading-relevant sentiment.
Evaluate whether the event is likely positive, negative, neutral, mixed, or unknown
for the specific company and ticker.
Use only the provided fields.
Do not invent facts.
Do not make a buy/sell recommendation.
Do not discuss portfolio decisions.
Do not discuss valuation unless the provided event mentions it.
Scoring rules:
- sentiment_score must be between -1 and 1.
- -1 means very negative.
- 0 means neutral.
- 1 means very positive.
- magnitude must be between 0 and 1.
- magnitude measures how strong the possible market impact is.
- confidence must be between 0 and 1.
- confidence should be lower if the information is vague, indirect, or weakly supported.
Sentiment label rules:
- positive: clear favorable effect on expectations, demand, revenue, margins, competitiveness, regulation, or investor perception.
- negative: clear unfavorable effect.
- neutral: relevant but no clear directional impact.
- mixed: contains both positive and negative implications.
- unknown: insufficient evidence.
Risk flags should be short strings.
Examples:
- "valuation_risk"
- "regulatory_risk"
- "demand_risk"
- "supply_chain_risk"
- "earnings_risk"
- "macro_risk"
- "geopolitical_risk"
- "competition_risk"
- "low_information_quality"
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
                        "event_relevance_score": event.event_relevance_score,
                        "event_confidence": event.event_confidence,
                        "relevance_agent_score": event.relevance_agent_score,
                        "impact_horizon": event.impact_horizon,
                        "affected_scope": event.affected_scope,
                        "relevance_reason": event.relevance_reason,
                    },
                    indent=2,
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "news_sentiment_decision",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "sentiment_score": {
                            "type": "number",
                            "minimum": -1,
                            "maximum": 1
                        },
                        "sentiment_label": {
                            "type": "string",
                            "enum": [
                                "positive",
                                "negative",
                                "neutral",
                                "mixed",
                                "unknown"
                            ]
                        },
                        "magnitude": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "time_horizon": {
                            "type": "string",
                            "enum": [
                                "intraday",
                                "short_term",
                                "medium_term",
                                "long_term",
                                "unknown"
                            ]
                        },
                        "main_driver": {
                            "type": "string"
                        },
                        "risk_flags": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "sentiment_score",
                        "sentiment_label",
                        "magnitude",
                        "confidence",
                        "time_horizon",
                        "main_driver",
                        "risk_flags"
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
    return NewsSentimentDecision.model_validate(data)
