from typing import Literal

from pydantic import BaseModel, Field


class TraderProfile(BaseModel):
    trader_name: str
    profile_type: Literal["aggressive", "medium", "conservative"]
    initial_cash: float = Field(gt=0)
    current_cash: float = Field(ge=0)
    max_position_size: float = Field(ge=0, le=1)
    max_portfolio_exposure: float = Field(ge=0, le=1)
    min_cash_reserve: float = Field(ge=0, le=1)
    trade_frequency: Literal["high", "medium", "low"]
    risk_tolerance: float = Field(ge=0, le=1)
    status: Literal["running", "stopped", "paused"]


class PortfolioState(BaseModel):
    trader_name: str
    cash: float
    invested_value: float
    total_portfolio_value: float
    open_positions_count: int


class PortfolioHolding(BaseModel):
    trader_name: str
    ticker: str
    company_name: str
    direction: Literal["long", "short"]
    quantity: float
    average_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    position_size: float


class TraderRunLog(BaseModel):
    trader_name: str
    event_type: str
    message: str
    metadata: dict | None = None


class AggregatePortfolioSnapshot(BaseModel):
    total_traders: int
    active_traders: int
    total_initial_cash: float
    total_current_cash: float
    total_invested_value: float
    total_portfolio_value: float
