from typing import Literal

from pydantic import BaseModel, Field


class RiskInput(BaseModel):
    sector: str
    ticker: str
    company_name: str
    signal_date: str
    alpha_score: float
    alpha_label: str
    alpha_confidence: float
    sentiment_score: float | None
    sentiment_label: str | None
    sentiment_confidence: float | None
    technical_score: float | None
    technical_label: str | None
    technical_confidence: float | None
    volatility_20d: float | None
    max_drawdown_60d: float | None
    return_20d: float | None
    return_60d: float | None
    price_vs_sma_20: float | None
    price_vs_sma_50: float | None


class RuleBasedRiskResult(BaseModel):
    rule_based_risk_score: float = Field(ge=0, le=1)
    rule_based_risk_label: Literal["low", "medium", "high", "extreme", "unknown"]
    rule_based_trade_allowed: bool
    rule_based_rejection_reason: str | None


class LlmRiskDecision(BaseModel):
    llm_risk_score: float = Field(ge=0, le=1)
    llm_risk_label: Literal["low", "medium", "high", "extreme", "unknown"]
    llm_trade_allowed: bool
    position_size_multiplier: float = Field(ge=0, le=1)
    risk_flags: list[str]
    risk_summary: str
    decision_reason: str


class RiskDecision(BaseModel):
    sector: str
    ticker: str
    company_name: str
    signal_date: str
    alpha_score: float
    alpha_label: str
    alpha_confidence: float
    sentiment_score: float | None
    sentiment_label: str | None
    sentiment_confidence: float | None
    technical_score: float | None
    technical_label: str | None
    technical_confidence: float | None
    volatility_20d: float | None
    max_drawdown_60d: float | None
    return_20d: float | None
    return_60d: float | None
    price_vs_sma_20: float | None
    price_vs_sma_50: float | None
    rule_based_risk_score: float
    rule_based_risk_label: str
    rule_based_trade_allowed: bool
    rule_based_rejection_reason: str | None
    llm_risk_score: float
    llm_risk_label: str
    llm_trade_allowed: bool
    position_size_multiplier: float
    risk_flags: list[str]
    risk_summary: str
    decision_reason: str
