from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_options_market_regime(
    prices: pd.DataFrame,
    *,
    current_price: float,
    forecast: dict[str, Any],
    lookback_rows: int = 120,
    breakout_buffer_pct: float = 0.001,
    middle_zone_width: float = 0.30,
    min_trend_strength_pct: float = 0.003,
    allow_range_edge_reversal_entry: bool = False,
    enable_impulse_entry: bool = True,
    impulse_lookback_bars: int = 12,
    min_impulse_move_pct: float = 0.006,
    min_impulse_directional_bars: int = 7,
    enable_late_entry_filter: bool = True,
    max_late_entry_move_pct: float = 0.018,
    max_ema_extension_pct: float = 0.010,
    exhaustion_reversal_bars: int = 2,
) -> dict[str, Any]:
    if prices is None or prices.empty:
        return {"enabled": True, "status": "unavailable", "reason": "missing_price_history", "allow_directional_entry": False}
    close = _price_series(prices).dropna()
    if len(close) < 30:
        return {"enabled": True, "status": "unavailable", "reason": "not_enough_price_history", "rows": len(close), "allow_directional_entry": False}
    window = close.tail(max(30, int(lookback_rows)))
    prior = window.iloc[:-1] if len(window) > 1 else window
    support = float(prior.min())
    resistance = float(prior.max())
    latest = float(current_price)
    price_range = max(resistance - support, 0.0)
    range_width_pct = price_range / max(abs(latest), 1e-9)
    range_position = 0.5 if price_range <= 0 else max(0.0, min(1.0, (latest - support) / price_range))
    fast = window.ewm(span=min(9, len(window)), adjust=False).mean()
    slow = window.ewm(span=min(21, len(window)), adjust=False).mean()
    ema_spread_pct = float(fast.iloc[-1] / max(float(slow.iloc[-1]), 1e-9) - 1.0)
    slope_window = min(15, max(3, len(window) // 4))
    slope_pct = float(window.iloc[-1] / max(float(window.iloc[-slope_window]), 1e-9) - 1.0)
    trend_strength_pct = abs(ema_spread_pct) + abs(slope_pct)
    forecast_return = _float_or_none(forecast.get("expected_return")) or 0.0
    forecast_direction = "up" if forecast_return > 0 else "down" if forecast_return < 0 else "flat"
    breakout_up = latest > resistance * (1.0 + max(0.0, float(breakout_buffer_pct)))
    breakout_down = latest < support * (1.0 - max(0.0, float(breakout_buffer_pct)))
    aligned_up = forecast_direction == "up" and (breakout_up or (ema_spread_pct > 0 and slope_pct > 0 and range_position >= 0.60))
    aligned_down = forecast_direction == "down" and (breakout_down or (ema_spread_pct < 0 and slope_pct < 0 and range_position <= 0.40))
    strong_trend = trend_strength_pct >= max(0.0, float(min_trend_strength_pct))
    middle_half_width = max(0.0, min(1.0, float(middle_zone_width))) / 2.0
    middle_low = 0.5 - middle_half_width
    middle_high = 0.5 + middle_half_width
    in_middle = middle_low <= range_position <= middle_high
    near_support = range_position <= 0.20
    near_resistance = range_position >= 0.80
    impulse = (
        _options_impulse_signal(
            window,
            forecast_direction=forecast_direction,
            lookback_bars=int(impulse_lookback_bars),
            min_move_pct=float(min_impulse_move_pct),
            min_directional_bars=int(min_impulse_directional_bars),
        )
        if enable_impulse_entry
        else {"enabled": False, "allow_entry": False}
    )
    exhaustion = (
        _options_exhaustion_signal(
            window,
            forecast_direction=forecast_direction,
            support=support,
            resistance=resistance,
            lookback_bars=int(impulse_lookback_bars),
            max_late_entry_move_pct=float(max_late_entry_move_pct),
            max_ema_extension_pct=float(max_ema_extension_pct),
            reversal_bars=int(exhaustion_reversal_bars),
        )
        if enable_late_entry_filter
        else {"enabled": False, "late_entry_block": False}
    )
    range_edge_reversal = bool(
        allow_range_edge_reversal_entry
        and (
            (forecast_direction == "up" and near_support and slope_pct > 0)
            or (forecast_direction == "down" and near_resistance and slope_pct < 0)
        )
    )
    if breakout_up:
        regime = "trend_up"
        breakout_status = "confirmed_breakout"
    elif breakout_down:
        regime = "trend_down"
        breakout_status = "confirmed_breakdown"
    elif strong_trend and aligned_up:
        regime = "trend_up"
        breakout_status = "trend_continuation"
    elif strong_trend and aligned_down:
        regime = "trend_down"
        breakout_status = "trend_continuation"
    elif bool(impulse.get("allow_entry")) and forecast_direction == "up":
        regime = "trend_up"
        breakout_status = "impulse_up"
    elif bool(impulse.get("allow_entry")) and forecast_direction == "down":
        regime = "trend_down"
        breakout_status = "impulse_down"
    elif in_middle:
        regime = "range_bound"
        breakout_status = "inside_range_middle"
    else:
        regime = "range_edge"
        breakout_status = "inside_range_edge"
    allow_entry = bool(
        (forecast_direction == "up" and regime == "trend_up" and strong_trend)
        or (forecast_direction == "down" and regime == "trend_down" and strong_trend)
        or bool(impulse.get("allow_entry"))
        or range_edge_reversal
    )
    if allow_entry and bool(exhaustion.get("late_entry_block")):
        allow_entry = False
        regime = "late_trend"
        breakout_status = "late_entry_exhaustion"
    reason = (
        "directional_trend_confirmed"
        if allow_entry and not range_edge_reversal and not bool(impulse.get("allow_entry"))
        else "impulse_entry_confirmed"
        if bool(impulse.get("allow_entry"))
        else "range_edge_reversal_allowed"
        if allow_entry
        else "range_or_noise_without_confirmed_direction"
    )
    if bool(exhaustion.get("late_entry_block")):
        reason = str(exhaustion.get("reason") or "late_entry_exhaustion")
    if in_middle and not allow_entry:
        reason = "price_in_middle_of_range"
    return {
        "enabled": True,
        "status": "ok",
        "regime": regime,
        "reason": reason,
        "allow_directional_entry": allow_entry,
        "forecast_direction": forecast_direction,
        "support_level": support,
        "resistance_level": resistance,
        "range_position": round(float(range_position), 4),
        "range_width_pct": round(float(range_width_pct), 6),
        "breakout_status": breakout_status,
        "breakout_up": breakout_up,
        "breakout_down": breakout_down,
        "ema_spread_pct": round(float(ema_spread_pct), 6),
        "slope_pct": round(float(slope_pct), 6),
        "trend_strength_pct": round(float(trend_strength_pct), 6),
        "min_trend_strength_pct": float(min_trend_strength_pct),
        "middle_zone": [round(float(middle_low), 4), round(float(middle_high), 4)],
        "near_support": near_support,
        "near_resistance": near_resistance,
        "range_edge_reversal_allowed": range_edge_reversal,
        "impulse": impulse,
        "exhaustion": exhaustion,
        "lookback_rows": len(window),
    }


def _options_impulse_signal(
    close: pd.Series,
    *,
    forecast_direction: str,
    lookback_bars: int,
    min_move_pct: float,
    min_directional_bars: int,
) -> dict[str, Any]:
    if len(close) < max(4, int(lookback_bars) + 1):
        return {"enabled": True, "status": "insufficient_history", "allow_entry": False}
    lookback = max(3, int(lookback_bars))
    segment = close.tail(lookback + 1)
    start = float(segment.iloc[0])
    end = float(segment.iloc[-1])
    move_pct = end / max(abs(start), 1e-9) - 1.0
    diffs = segment.diff().dropna()
    up_bars = int((diffs > 0).sum())
    down_bars = int((diffs < 0).sum())
    required_bars = max(1, min(int(min_directional_bars), lookback))
    if move_pct >= float(min_move_pct) and up_bars >= required_bars:
        direction = "up"
    elif move_pct <= -float(min_move_pct) and down_bars >= required_bars:
        direction = "down"
    else:
        direction = "none"
    allow_entry = bool(direction != "none" and direction == forecast_direction)
    return {
        "enabled": True,
        "status": "ok",
        "direction": direction,
        "allow_entry": allow_entry,
        "move_pct": round(float(move_pct), 6),
        "min_move_pct": float(min_move_pct),
        "lookback_bars": lookback,
        "up_bars": up_bars,
        "down_bars": down_bars,
        "min_directional_bars": required_bars,
        "forecast_direction": forecast_direction,
    }


def _options_exhaustion_signal(
    close: pd.Series,
    *,
    forecast_direction: str,
    support: float,
    resistance: float,
    lookback_bars: int,
    max_late_entry_move_pct: float,
    max_ema_extension_pct: float,
    reversal_bars: int,
) -> dict[str, Any]:
    if len(close) < max(8, int(lookback_bars) + 1):
        return {"enabled": True, "status": "insufficient_history", "late_entry_block": False}
    lookback = max(3, int(lookback_bars))
    segment = close.tail(lookback + 1)
    start = float(segment.iloc[0])
    latest = float(segment.iloc[-1])
    move_pct = latest / max(abs(start), 1e-9) - 1.0
    ema = close.ewm(span=min(9, len(close)), adjust=False).mean()
    ema_extension_pct = latest / max(float(ema.iloc[-1]), 1e-9) - 1.0
    diffs = close.diff().dropna()
    recent = diffs.tail(max(1, int(reversal_bars)))
    reversal_up = bool(len(recent) >= max(1, int(reversal_bars)) and (recent > 0).all())
    reversal_down = bool(len(recent) >= max(1, int(reversal_bars)) and (recent < 0).all())
    support_distance_pct = abs(latest - float(support)) / max(abs(latest), 1e-9)
    resistance_distance_pct = abs(float(resistance) - latest) / max(abs(latest), 1e-9)
    late_reasons: list[str] = []
    if forecast_direction == "down":
        if move_pct <= -abs(float(max_late_entry_move_pct)):
            late_reasons.append("down_move_already_extended")
        if ema_extension_pct <= -abs(float(max_ema_extension_pct)):
            late_reasons.append("price_extended_below_fast_ema")
        if reversal_up:
            late_reasons.append("recent_bullish_reversal_bars")
        if support_distance_pct <= 0.0025:
            late_reasons.append("price_near_support_after_drop")
    elif forecast_direction == "up":
        if move_pct >= abs(float(max_late_entry_move_pct)):
            late_reasons.append("up_move_already_extended")
        if ema_extension_pct >= abs(float(max_ema_extension_pct)):
            late_reasons.append("price_extended_above_fast_ema")
        if reversal_down:
            late_reasons.append("recent_bearish_reversal_bars")
        if resistance_distance_pct <= 0.0025:
            late_reasons.append("price_near_resistance_after_rally")
    has_reversal_warning = "recent_bullish_reversal_bars" in late_reasons or "recent_bearish_reversal_bars" in late_reasons
    late_entry_block = len(late_reasons) >= 2 and has_reversal_warning
    return {
        "enabled": True,
        "status": "ok",
        "late_entry_block": late_entry_block,
        "reason": "late_entry_exhaustion" if late_entry_block else "not_exhausted",
        "late_reasons": late_reasons,
        "move_pct": round(float(move_pct), 6),
        "max_late_entry_move_pct": float(max_late_entry_move_pct),
        "ema_extension_pct": round(float(ema_extension_pct), 6),
        "max_ema_extension_pct": float(max_ema_extension_pct),
        "support_distance_pct": round(float(support_distance_pct), 6),
        "resistance_distance_pct": round(float(resistance_distance_pct), 6),
        "reversal_up": reversal_up,
        "reversal_down": reversal_down,
        "reversal_bars": max(1, int(reversal_bars)),
        "forecast_direction": forecast_direction,
    }


def _price_series(prices: pd.DataFrame) -> pd.Series:
    for column in ("close", "Close", "price", "Price"):
        if column in prices.columns:
            return pd.to_numeric(prices[column], errors="coerce")
    numeric = prices.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.Series(dtype=float)
    return pd.to_numeric(numeric.iloc[:, 0], errors="coerce")


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return float(parsed)
