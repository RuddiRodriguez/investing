from typing import Literal
from pydantic import BaseModel, Field


class CompanyUniverseItem(BaseModel):
    rank: int = Field(ge=1)
    ticker: str
    company_name: str
    exchange: str | None = None
    country: str | None = None
    industry: str | None = None
    relevance_reason: str
    source_url: str | None = None


class CompanyUniverse(BaseModel):
    sector: str
    companies: list[CompanyUniverseItem]


class RawNewsItem(BaseModel):
    ticker: str
    company_name: str
    source: str | None = None
    title: str
    url: str
    published_at: str | None = None
    raw_summary: str | None = None


class NewsSearchResult(BaseModel):
    sector: str
    articles: list[RawNewsItem]


class NewsEvent(BaseModel):
    summary: str
    event_type: Literal[
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
    affected_direction: Literal[
        "positive",
        "negative",
        "mixed",
        "neutral",
        "unknown"
    ]
    relevance_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
