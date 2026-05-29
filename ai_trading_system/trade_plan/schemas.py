from typing import Literal

from pydantic import BaseModel, Field


class TradePlanInput(BaseModel):
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


class LlmTradePlanDecision(BaseModel):
    planned_side: Literal["buy", "sell_short", "none"]
    planned_order_type: Literal["market", "limit"]
    planned_time_in_force: Literal["day", "gtc"]
    execution_priority: Literal["low", "normal", "high"]
    max_slippage_pct: float = Field(ge=0, le=0.05)
    trade_plan_status: Literal["planned", "skipped"]
    trade_reason: str
    execution_notes: str


class TradePlan(BaseModel):
    sector: str
    ticker: str
    company_name: str
    signal_date: str
    llm_direction: str
    llm_portfolio_action: str
    llm_position_size: float
    buy_probability: float
    planned_side: str
    planned_order_type: str
    planned_time_in_force: str
    execution_priority: str
    max_slippage_pct: float
    trade_plan_status: str
    trade_reason: str
    execution_notes: str
