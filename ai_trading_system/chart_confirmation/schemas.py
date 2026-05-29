from typing import Literal

from pydantic import BaseModel, Field


class PriceBarForChart(BaseModel):
    sector: str
    ticker: str
    timestamp: str
    open: float | None
    high: float | None
    low: float | None
    close: float
    adjusted_close: float | None
    volume: int | None


class ChartMetrics(BaseModel):
    sector: str
    ticker: str
    company_name: str
    signal_date: str
    current_price: float
    open_price: float | None
    high_price: float | None
    low_price: float | None
    close_price: float | None
    latest_volume: float | None
    trend_status: Literal[
        "uptrend",
        "downtrend",
        "sideways",
        "unknown",
    ]
    trend_reading: Literal[
        "strong_upward",
        "weak_upward",
        "sideways",
        "weak_downward",
        "strong_downward",
        "unknown",
    ]
    base_status: Literal[
        "building_base",
        "no_base",
        "extended",
        "unknown",
    ]
    support_level: float | None
    resistance_level: float | None
    breakout_status: Literal[
        "confirmed_breakout",
        "near_breakout",
        "failed_breakout",
        "no_breakout",
        "unknown",
    ]
    breakout_price: float | None
    volume_confirmation: Literal[
        "strong_volume",
        "weak_volume",
        "distribution_volume",
        "unknown",
    ]
    volume_ratio: float | None
    entry_quality: Literal[
        "proper_entry",
        "too_early",
        "too_late",
        "overextended",
        "avoid",
    ]
    extension_pct: float | None
    buy_trigger: str
    invalid_buy_reason: str
    reason_to_wait: str
    current_price_stop_7_pct: float | None
    current_price_stop_8_pct: float | None
    breakout_entry_stop_7_pct: float | None
    breakout_entry_stop_8_pct: float | None
    danger_level: float | None
    sell_signal: Literal[
        "none",
        "stop_loss_triggered",
        "failed_breakout",
        "heavy_distribution",
        "overextended_reversal",
    ]


class LlmChartDecision(BaseModel):
    chart_decision: Literal[
        "BUY",
        "WAIT_FOR_BREAKOUT",
        "AVOID",
        "SELL",
        "HOLD",
    ]
    chart_score: float = Field(ge=0, le=1)
    chart_confidence: float = Field(ge=0, le=1)
    llm_chart_reason: str
    chart_flags: list[str]


class ChartConfirmationSignal(BaseModel):
    sector: str
    ticker: str
    company_name: str
    signal_date: str
    current_price: float
    open_price: float | None
    high_price: float | None
    low_price: float | None
    close_price: float | None
    latest_volume: float | None
    trend_status: str
    trend_reading: str
    base_status: str
    support_level: float | None
    resistance_level: float | None
    breakout_status: str
    breakout_price: float | None
    volume_confirmation: str
    volume_ratio: float | None
    entry_quality: str
    extension_pct: float | None
    buy_trigger: str
    invalid_buy_reason: str
    reason_to_wait: str
    current_price_stop_7_pct: float | None
    current_price_stop_8_pct: float | None
    breakout_entry_stop_7_pct: float | None
    breakout_entry_stop_8_pct: float | None
    danger_level: float | None
    sell_signal: str
    chart_decision: str
    chart_score: float
    chart_confidence: float
    llm_chart_reason: str
    chart_flags: list[str]
