from typing import Literal

from pydantic import BaseModel, Field


class PriceBarInput(BaseModel):
    sector: str
    ticker: str
    timestamp: str
    close: float
    adjusted_close: float | None
    volume: int | None


class TechnicalSignal(BaseModel):
    sector: str
    ticker: str
    signal_date: str
    close: float
    return_5d: float | None
    return_20d: float | None
    return_60d: float | None
    volatility_20d: float | None
    sma_20: float | None
    sma_50: float | None
    price_vs_sma_20: float | None
    price_vs_sma_50: float | None
    max_drawdown_60d: float | None
    technical_score: float = Field(ge=-1, le=1)
    technical_label: Literal[
        "bullish",
        "bearish",
        "neutral",
        "mixed",
        "unknown",
    ]
    confidence: float = Field(ge=0, le=1)
