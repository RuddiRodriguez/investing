from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_curve_shape_analysis(
    prices: pd.DataFrame,
    *,
    current_price: float | None = None,
    lookback_rows: int = 120,
    short_window: int = 9,
    long_window: int = 21,
    impulse_bars: int = 8,
    range_lookback: int = 60,
) -> dict[str, Any]:
    """Classify recent price curve shape using only pre-decision data.

    These labels are intentionally deterministic and auditable. They are the
    first layer for later learning: outcome labels can be joined back to this
    pre-trade shape label after forecasts or trades mature.
    """

    if prices is None or prices.empty:
        return {"enabled": True, "status": "unavailable", "reason": "missing_price_history"}
    close = _close_series(prices).dropna()
    if len(close) < max(30, int(long_window) + 5):
        return {"enabled": True, "status": "unavailable", "reason": "not_enough_price_history", "rows": int(len(close))}

    window = close.tail(max(30, int(lookback_rows))).copy()
    latest = float(current_price if current_price is not None else window.iloc[-1])
    short = window.rolling(max(2, int(short_window)), min_periods=max(2, int(short_window))).mean()
    long = window.rolling(max(3, int(long_window)), min_periods=max(3, int(long_window))).mean()
    short_latest = _float_or_none(short.iloc[-1])
    long_latest = _float_or_none(long.iloc[-1])
    sma_state = _sma_state(short, long)
    impulse = _impulse_features(window, bars=int(impulse_bars))
    range_features = _range_features(window, lookback=int(range_lookback), latest=latest)
    reversal = _reversal_features(window, impulse=impulse)

    label, direction, confidence, reason, recommendation = _classify_shape(
        latest=latest,
        short_sma=short_latest,
        long_sma=long_latest,
        sma_state=sma_state,
        impulse=impulse,
        range_features=range_features,
        reversal=reversal,
    )

    return {
        "enabled": True,
        "status": "ok",
        "label": label,
        "direction": direction,
        "confidence": round(float(confidence), 4),
        "reason": reason,
        "recommended_option_bias": recommendation,
        "lookback_rows": int(len(window)),
        "current_price": latest,
        "features": {
            "sma_short_window": int(short_window),
            "sma_long_window": int(long_window),
            "sma_short": short_latest,
            "sma_long": long_latest,
            "sma_state": sma_state,
            "impulse": impulse,
            "range": range_features,
            "reversal": reversal,
        },
        "label_policy": "pre_trade_curve_shape_only_no_future_data",
    }


def _classify_shape(
    *,
    latest: float,
    short_sma: float | None,
    long_sma: float | None,
    sma_state: str,
    impulse: dict[str, Any],
    range_features: dict[str, Any],
    reversal: dict[str, Any],
) -> tuple[str, str, float, str, str]:
    impulse_direction = str(impulse.get("direction") or "none")
    move_pct = abs(float(impulse.get("move_pct") or 0.0))
    trend_strength = abs(float(range_features.get("trend_slope_pct") or 0.0)) + abs(float(range_features.get("ema_extension_pct") or 0.0))
    range_width_pct = float(range_features.get("range_width_pct") or 0.0)
    range_position = float(range_features.get("range_position") or 0.5)
    breakout = str(range_features.get("breakout") or "inside")

    if breakout == "breakout_up":
        return ("breakout_after_range", "bullish", 0.78, "price_broke_above_recent_range", "call")
    if breakout == "breakdown_down":
        return ("breakdown_after_range", "bearish", 0.78, "price_broke_below_recent_range", "put")

    if reversal.get("v_reversal_up"):
        return ("v_reversal_up", "bullish", 0.74, "sharp_drop_followed_by_strong_bounce", "call")
    if reversal.get("v_reversal_down"):
        return ("v_reversal_down", "bearish", 0.74, "sharp_rally_followed_by_strong_rejection", "put")

    if impulse_direction == "down" and reversal.get("bullish_reversal_bars") and range_position <= 0.25:
        return ("late_downtrend_exhaustion", "neutral_to_bullish", 0.72, "down_move_is_late_near_support_with_bounce_attempt", "hold")
    if impulse_direction == "up" and reversal.get("bearish_reversal_bars") and range_position >= 0.75:
        return ("late_uptrend_exhaustion", "neutral_to_bearish", 0.72, "up_move_is_late_near_resistance_with_rejection", "hold")

    if impulse_direction == "down" and move_pct >= 0.005 and sma_state in {"bearish", "bearish_cross_recent"}:
        return ("sharp_down_impulse", "bearish", 0.76, "recent_fast_down_move_with_bearish_sma_alignment", "put")
    if impulse_direction == "up" and move_pct >= 0.005 and sma_state in {"bullish", "bullish_cross_recent"}:
        return ("sharp_up_impulse", "bullish", 0.76, "recent_fast_up_move_with_bullish_sma_alignment", "call")

    if range_width_pct <= 0.012 and trend_strength <= 0.004:
        return ("range_chop", "neutral", 0.70, "narrow_range_without_directional_trend", "hold")

    if short_sma is not None and long_sma is not None and short_sma > long_sma and trend_strength > 0.004:
        return ("slow_grind_up", "bullish", 0.64, "fast_average_above_slow_average_with_positive_drift", "call")
    if short_sma is not None and long_sma is not None and short_sma < long_sma and trend_strength > 0.004:
        return ("slow_grind_down", "bearish", 0.64, "fast_average_below_slow_average_with_negative_drift", "put")

    return ("unclear_transition", "neutral", 0.50, "mixed_curve_shape_without_clear_edge", "hold")


def _impulse_features(close: pd.Series, *, bars: int) -> dict[str, Any]:
    lookback = max(3, min(int(bars), len(close) - 1))
    segment = close.tail(lookback + 1)
    start = float(segment.iloc[0])
    latest = float(segment.iloc[-1])
    move_pct = latest / max(abs(start), 1e-9) - 1.0
    diffs = segment.diff().dropna()
    up_bars = int((diffs > 0).sum())
    down_bars = int((diffs < 0).sum())
    direction = "up" if move_pct > 0 and up_bars >= down_bars else "down" if move_pct < 0 and down_bars >= up_bars else "none"
    return {
        "bars": int(lookback),
        "move_pct": round(float(move_pct), 6),
        "direction": direction,
        "up_bars": up_bars,
        "down_bars": down_bars,
        "directional_bar_ratio": round(max(up_bars, down_bars) / max(lookback, 1), 4),
    }


def _range_features(close: pd.Series, *, lookback: int, latest: float) -> dict[str, Any]:
    window = close.tail(max(20, min(int(lookback), len(close))))
    prior = window.iloc[:-1] if len(window) > 1 else window
    support = float(prior.min())
    resistance = float(prior.max())
    width = max(resistance - support, 0.0)
    range_width_pct = width / max(abs(latest), 1e-9)
    position = 0.5 if width <= 0 else max(0.0, min(1.0, (latest - support) / width))
    ema = close.ewm(span=min(9, len(close)), adjust=False).mean()
    ema_extension_pct = latest / max(float(ema.iloc[-1]), 1e-9) - 1.0
    slope_window = min(15, max(3, len(window) // 4))
    trend_slope_pct = float(window.iloc[-1] / max(float(window.iloc[-slope_window]), 1e-9) - 1.0)
    breakout_buffer = 0.0015
    breakout = "inside"
    if latest > resistance * (1.0 + breakout_buffer):
        breakout = "breakout_up"
    elif latest < support * (1.0 - breakout_buffer):
        breakout = "breakdown_down"
    return {
        "support": round(support, 6),
        "resistance": round(resistance, 6),
        "range_width_pct": round(float(range_width_pct), 6),
        "range_position": round(float(position), 4),
        "ema_extension_pct": round(float(ema_extension_pct), 6),
        "trend_slope_pct": round(float(trend_slope_pct), 6),
        "breakout": breakout,
    }


def _reversal_features(close: pd.Series, *, impulse: dict[str, Any]) -> dict[str, Any]:
    diffs = close.diff().dropna()
    last_two = diffs.tail(2)
    bullish_reversal_bars = bool(len(last_two) == 2 and (last_two > 0).all())
    bearish_reversal_bars = bool(len(last_two) == 2 and (last_two < 0).all())
    tail = close.tail(10)
    first_half = tail.head(5)
    second_half = tail.tail(5)
    first_move = float(first_half.iloc[-1] / max(float(first_half.iloc[0]), 1e-9) - 1.0) if len(first_half) >= 2 else 0.0
    second_move = float(second_half.iloc[-1] / max(float(second_half.iloc[0]), 1e-9) - 1.0) if len(second_half) >= 2 else 0.0
    v_reversal_up = first_move <= -0.004 and second_move >= 0.003
    v_reversal_down = first_move >= 0.004 and second_move <= -0.003
    return {
        "bullish_reversal_bars": bullish_reversal_bars,
        "bearish_reversal_bars": bearish_reversal_bars,
        "v_reversal_up": bool(v_reversal_up),
        "v_reversal_down": bool(v_reversal_down),
        "first_half_move_pct": round(float(first_move), 6),
        "second_half_move_pct": round(float(second_move), 6),
        "impulse_direction": impulse.get("direction"),
    }


def _sma_state(short: pd.Series, long: pd.Series) -> str:
    if short.isna().iloc[-1] or long.isna().iloc[-1]:
        return "unknown"
    latest_short = float(short.iloc[-1])
    latest_long = float(long.iloc[-1])
    prev_short = float(short.iloc[-2]) if len(short) >= 2 and not pd.isna(short.iloc[-2]) else latest_short
    prev_long = float(long.iloc[-2]) if len(long) >= 2 and not pd.isna(long.iloc[-2]) else latest_long
    if prev_short <= prev_long and latest_short > latest_long:
        return "bullish_cross_recent"
    if prev_short >= prev_long and latest_short < latest_long:
        return "bearish_cross_recent"
    if latest_short > latest_long:
        return "bullish"
    if latest_short < latest_long:
        return "bearish"
    return "neutral"


def _close_series(prices: pd.DataFrame) -> pd.Series:
    for column in ("close", "Close", "adj_close", "Adj Close"):
        if column in prices.columns:
            return pd.to_numeric(prices[column], errors="coerce")
    return pd.to_numeric(prices.iloc[:, 0], errors="coerce")


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None
