import json

from dotenv import load_dotenv
from openai import OpenAI

from aggregation.schemas import SentimentEventInput

load_dotenv()
client = OpenAI()


def generate_ticker_signal_summary(
    ticker: str,
    company_name: str,
    sector: str,
    signal_score: float,
    signal_label: str,
    confidence: float,
    events: list[SentimentEventInput],
) -> str:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": """
You are a financial signal summarization agent.
Your job is to summarize the aggregated news sentiment signal for one ticker.
Rules:
- Do not make buy/sell recommendations.
- Do not mention portfolio allocation.
- Do not invent facts.
- Use only the provided events.
- Keep the summary short, factual, and trading-relevant.
- Mention the main positive and negative drivers if both exist.
""",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "ticker": ticker,
                        "company_name": company_name,
                        "sector": sector,
                        "signal_score": signal_score,
                        "signal_label": signal_label,
                        "confidence": confidence,
                        "events": [
                            {
                                "news_event_id": event.news_event_id,
                                "title": event.title,
                                "event_summary": event.event_summary,
                                "sentiment_score": event.sentiment_score,
                                "sentiment_label": event.sentiment_label,
                                "magnitude": event.magnitude,
                                "confidence": event.confidence,
                                "main_driver": event.main_driver,
                                "risk_flags": event.risk_flags,
                            }
                            for event in events
                        ],
                    },
                    indent=2,
                ),
            },
        ],
        temperature=0,
    )
    return response.output_text.strip()
