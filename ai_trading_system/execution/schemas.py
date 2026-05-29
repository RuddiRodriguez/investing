from typing import Literal

from pydantic import BaseModel, Field


class ExecutionInput(BaseModel):
    trade_plan_id: int
    sector: str
    ticker: str
    company_name: str
    signal_date: str
    planned_side: Literal["buy", "sell_short", "none"]
    planned_order_type: str
    planned_time_in_force: str
    execution_priority: str
    max_slippage_pct: float
    llm_position_size: float


class LatestPrice(BaseModel):
    ticker: str
    price: float
    timestamp: str


class ExecutionContext(BaseModel):
    trader_name: str
    profile_type: str
    trader_status: str
    cash: float
    invested_value: float
    total_portfolio_value: float
    open_positions_count: int
    max_position_size: float
    max_portfolio_exposure: float
    min_cash_reserve: float
    risk_tolerance: float
    trade_frequency: str
    ticker: str
    company_name: str
    planned_side: str
    planned_order_type: str
    planned_time_in_force: str
    execution_priority: str
    requested_position_size: float
    requested_value: float
    latest_price: float
    latest_price_date: str
    available_cash_after_reserve: float
    available_exposure_value: float
    max_executable_value: float


class LlmExecutionDecision(BaseModel):
    llm_execution_status: Literal["fill", "partial_fill", "reject", "skip"]
    llm_fill_ratio: float = Field(ge=0, le=1)
    llm_execution_confidence: float = Field(ge=0, le=1)
    llm_execution_reason: str
    llm_execution_flags: list[str]


class TradeExecution(BaseModel):
    trader_name: str
    trade_plan_id: int
    sector: str
    ticker: str
    company_name: str
    signal_date: str
    side: str
    order_type: str
    time_in_force: str
    requested_position_size: float
    execution_price: float
    quantity: float
    gross_value: float
    simulated_slippage_pct: float
    commission: float
    llm_execution_status: str
    llm_fill_ratio: float
    llm_execution_confidence: float
    llm_execution_reason: str
    llm_execution_flags: list[str]
    execution_status: Literal["filled", "partial_filled", "rejected", "skipped"]
    execution_reason: str
