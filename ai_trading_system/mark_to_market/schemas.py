from typing import Literal

from pydantic import BaseModel


class HoldingForValuation(BaseModel):
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


class LatestPrice(BaseModel):
    ticker: str
    price: float
    timestamp: str


class UpdatedHoldingValuation(BaseModel):
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


class PortfolioValuationSnapshot(BaseModel):
    trader_name: str
    cash: float
    invested_value: float
    total_portfolio_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    open_positions_count: int
    valuation_reason: str
