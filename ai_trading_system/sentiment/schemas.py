from typing import Literal

from pydantic import BaseModel, Field


class RelevantNewsEventInput(BaseModel):
    id: int
    sector: str
    ticker: str
    company_name: str
    title: str
    summary: str
    event_type: str
    affected_direction: str
    event_relevance_score: float
    event_confidence: float
    relevance_agent_score: float
    impact_horizon: str
    affected_scope: str
    relevance_reason: str


class NewsSentimentDecision(BaseModel):
    sentiment_score: float = Field(ge=-1, le=1)
    sentiment_label: Literal[
        "positive",
        "negative",
        "neutral",
        "mixed",
        "unknown",
    ]
    magnitude: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    time_horizon: Literal[
        "intraday",
        "short_term",
        "medium_term",
        "long_term",
        "unknown",
    ]
    main_driver: str
    risk_flags: list[str]
