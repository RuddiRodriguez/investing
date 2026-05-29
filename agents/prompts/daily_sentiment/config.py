"""Prompts and schemas for daily_sentiment_agent"""

from typing import List, Literal

from pydantic import BaseModel, Field


# Prompts
SYSTEM_MESSAGE = (
    "You are a financial market sentiment analyst. "
    "Analyze all provided articles together as one group. "
    "Return one overall sentiment for the date, not sentiment per article. "
    "Focus only on likely stock-market impact for the sector. "
    "Use only the provided article data. "
    "Do not invent facts, companies, tickers, or URLs. "
    "Return structured output only."
)

USER_MESSAGE_TEMPLATE = (
    "Date: {date}\n"
    "Sector: {sector}\n\n"
    "Analyze the combined stock-market sentiment of these articles. "
    "Decide whether the day is all positive, mostly positive, balanced, "
    "mostly negative, all negative, or unclear.\n\n"
    "Articles:\n{articles_text}"
)


# Schemas
class DailyStockSentimentResult(BaseModel):
    date: str
    sector: str

    overall_sentiment: Literal[
        "all_positive",
        "mostly_positive",
        "balanced",
        "mostly_negative",
        "all_negative",
        "unclear",
    ]

    stock_market_impact: Literal[
        "bullish",
        "slightly_bullish",
        "neutral",
        "slightly_bearish",
        "bearish",
        "unclear",
    ]

    positive_signals: List[str]
    negative_signals: List[str]
    balance_explanation: str
    confidence: float
    notes: str
