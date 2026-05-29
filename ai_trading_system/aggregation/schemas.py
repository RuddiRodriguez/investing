from typing import Literal

from pydantic import BaseModel


class SentimentEventInput(BaseModel):
    news_event_id: int
    sector: str
    ticker: str
    company_name: str
    title: str
    event_type: str
    affected_direction: str
    event_summary: str
    sentiment_score: float
    sentiment_label: str
    magnitude: float
    confidence: float
    time_horizon: str
    main_driver: str
    risk_flags: list[str]


class TickerSentimentSignal(BaseModel):
    sector: str
    ticker: str
    company_name: str
    signal_score: float
    signal_label: Literal[
        "bullish",
        "bearish",
        "neutral",
        "mixed",
        "unknown",
    ]
    confidence: float
    event_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    mixed_count: int
    unknown_count: int
    strongest_positive_event_id: int | None
    strongest_negative_event_id: int | None
    summary: str
