from typing import Literal

from pydantic import BaseModel, Field


class PortfolioInput(BaseModel):
    sector: str
    ticker: str
    company_name: str
    signal_date: str
    alpha_score: float
    alpha_label: str
    alpha_confidence: float
    llm_risk_score: float
    llm_risk_label: str
    position_size_multiplier: float
    risk_summary: str
    decision_reason: str
    chart_decision: str | None = None
    chart_score: float | None = None
    chart_confidence: float | None = None
    trend_reading: str | None = None
    breakout_status: str | None = None
    volume_confirmation: str | None = None
    entry_quality: str | None = None
    support_level: float | None = None
    resistance_level: float | None = None
    buy_trigger: str | None = None
    invalid_buy_reason: str | None = None
    reason_to_wait: str | None = None
    current_price_stop_7_pct: float | None = None
    current_price_stop_8_pct: float | None = None
    breakout_entry_stop_7_pct: float | None = None
    breakout_entry_stop_8_pct: float | None = None
    stop_loss_7_pct: float | None = None
    stop_loss_8_pct: float | None = None
    danger_level: float | None = None


class SuggestedPosition(BaseModel):
    suggested_direction: Literal["long", "short", "none"]
    suggested_position_size: float = Field(ge=0, le=1)


class LlmPortfolioDecision(BaseModel):
    llm_direction: Literal["long", "short", "none"]
    llm_portfolio_action: Literal["open", "skip", "watch", "reduce"]
    llm_position_size: float = Field(ge=0, le=1)
    llm_confidence: float = Field(ge=0, le=1)
    buy_probability: float = Field(ge=0, le=1)
    portfolio_reason: str
    portfolio_flags: list[str]


class PortfolioPosition(BaseModel):
    sector: str
    ticker: str
    company_name: str
    signal_date: str
    alpha_score: float
    alpha_label: str
    alpha_confidence: float
    llm_risk_score: float
    llm_risk_label: str
    position_size_multiplier: float
    suggested_direction: str
    suggested_position_size: float
    llm_direction: str
    llm_portfolio_action: str
    llm_position_size: float
    llm_confidence: float
    buy_probability: float
    portfolio_reason: str
    portfolio_flags: list[str]
