from typing import Literal

from pydantic import BaseModel, Field


class AlphaInput(BaseModel):
    sector: str
    ticker: str
    company_name: str
    signal_date: str
    sentiment_score: float | None
    sentiment_label: str | None
    sentiment_confidence: float | None
    technical_score: float | None
    technical_label: str | None
    technical_confidence: float | None
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


class CombinedAlphaSignal(BaseModel):
    sector: str
    ticker: str
    company_name: str
    signal_date: str
    sentiment_score: float | None
    sentiment_label: str | None
    sentiment_confidence: float | None
    technical_score: float | None
    technical_label: str | None
    technical_confidence: float | None
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
    alpha_score: float = Field(ge=-1, le=1)
    alpha_label: Literal[
        "strong_bullish",
        "bullish",
        "neutral",
        "bearish",
        "strong_bearish",
        "mixed",
        "unknown",
    ]
    confidence: float = Field(ge=0, le=1)
    main_driver: str
