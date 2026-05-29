from typing import Literal

from pydantic import BaseModel, Field


class NewsEventInput(BaseModel):
    id: int
    sector: str
    ticker: str
    company_name: str
    title: str
    summary: str
    event_type: str
    affected_direction: str
    relevance_score: float
    confidence: float


class NewsRelevanceDecision(BaseModel):
    is_relevant: bool
    relevance_score: float = Field(ge=0, le=1)
    impact_horizon: Literal[
        "intraday",
        "short_term",
        "medium_term",
        "long_term",
        "unknown",
    ]
    affected_scope: Literal[
        "single_company",
        "sector",
        "market",
        "macro",
        "unknown",
    ]
    reason: str
