from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_chart_pattern_analysis(
    prices: pd.DataFrame,
    *,
    current_price: float,
    forecast: dict[str, Any],
    lookback_rows: int = 240,
    pivot_order: int = 3,
    level_tolerance_pct: float = 0.006,
    breakout_buffer_pct: float = 0.0015,
    min_volume_ratio: float = 1.20,
) -> dict[str, Any]:
    """Detect practical crypto chart-pattern context for directional option entries.

    This is intentionally conservative: named patterns are used as confluence
    and conflict detection, not as standalone trading signals.
    """

    if prices is None or prices.empty:
        return {"enabled": True, "status": "unavailable", "reason": "missing_price_history"}
    close = _close_series(prices).dropna()
    if len(close) < 40:
        return {"enabled": True, "status": "unavailable", "reason": "not_enough_price_history", "rows": len(close)}
    window = close.tail(max(40, int(lookback_rows)))
    volume = _volume_series(prices).reindex(close.index).tail(len(window)) if _volume_series(prices) is not None else None
    expected_return = _float_or_none(forecast.get("expected_return")) or 0.0
    forecast_direction = "bullish" if expected_return > 0 else "bearish" if expected_return < 0 else "flat"
    pivots = _pivots(window, order=max(2, int(pivot_order)))
    patterns: list[dict[str, Any]] = []
    patterns.extend(
        _double_top_bottom_patterns(
            window,
            pivots=pivots,
            current_price=float(current_price),
            volume=volume,
            level_tolerance_pct=float(level_tolerance_pct),
            breakout_buffer_pct=float(breakout_buffer_pct),
            min_volume_ratio=float(min_volume_ratio),
        )
    )
    patterns.extend(
        _flag_patterns(
            window,
            current_price=float(current_price),
            volume=volume,
            min_volume_ratio=float(min_volume_ratio),
        )
    )
    patterns.extend(
        _wedge_patterns(
            window,
            current_price=float(current_price),
            min_volume_ratio=float(min_volume_ratio),
            volume=volume,
            breakout_buffer_pct=float(breakout_buffer_pct),
        )
    )
    patterns = sorted(patterns, key=lambda row: (-float(row.get("confidence") or 0.0), str(row.get("name") or "")))
    summary = _summarize_patterns(patterns, forecast_direction=forecast_direction)
    return {
        "enabled": True,
        "status": "ok",
        "source": "crypto_chart_patterns_caia_2025_structured_rules",
        "lookback_rows": len(window),
        "current_price": float(current_price),
        "forecast_direction": forecast_direction,
        "patterns": patterns[:8],
        "summary": summary,
        "parameters": {
            "pivot_order": max(2, int(pivot_order)),
            "level_tolerance_pct": float(level_tolerance_pct),
            "breakout_buffer_pct": float(breakout_buffer_pct),
            "min_volume_ratio": float(min_volume_ratio),
        },
    }


def _double_top_bottom_patterns(
    close: pd.Series,
    *,
    pivots: dict[str, list[tuple[Any, float]]],
    current_price: float,
    volume: pd.Series | None,
    level_tolerance_pct: float,
    breakout_buffer_pct: float,
    min_volume_ratio: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    highs = pivots["highs"][-4:]
    lows = pivots["lows"][-4:]
    if len(highs) >= 2 and lows:
        h1, h2 = highs[-2], highs[-1]
        level_diff = abs(h2[1] - h1[1]) / max(abs((h1[1] + h2[1]) / 2.0), 1e-9)
        middle_lows = [low for low in lows if h1[0] < low[0] < h2[0]]
        neckline = min((low[1] for low in middle_lows), default=min(low[1] for low in lows[-2:]))
        confirmed = current_price < neckline * (1.0 - max(0.0, breakout_buffer_pct))
        second_volume = _volume_at(volume, h2[0])
        first_volume = _volume_at(volume, h1[0])
        volume_fades = first_volume is not None and second_volume is not None and second_volume <= first_volume * 0.85
        volume_confirmed = _latest_volume_ratio(volume) >= min_volume_ratio if confirmed else volume_fades
        if level_diff <= level_tolerance_pct:
            rows.append(
                _pattern_row(
                    name="double_top",
                    direction="bearish",
                    status="confirmed" if confirmed else "forming",
                    confidence=0.72 + (0.08 if volume_confirmed else 0.0),
                    volume_confirmed=volume_confirmed,
                    reason="two_resistance_rejections_with_neckline_break" if confirmed else "two_resistance_rejections_forming",
                    entry_trigger="neckline_break",
                    stop_reference="highest_peak",
                    target_reference="pattern_height_or_127_fibonacci_extension",
                    levels={"peak_1": h1[1], "peak_2": h2[1], "neckline": neckline},
                )
            )
    if len(lows) >= 2 and highs:
        l1, l2 = lows[-2], lows[-1]
        level_diff = abs(l2[1] - l1[1]) / max(abs((l1[1] + l2[1]) / 2.0), 1e-9)
        middle_highs = [high for high in highs if l1[0] < high[0] < l2[0]]
        neckline = max((high[1] for high in middle_highs), default=max(high[1] for high in highs[-2:]))
        confirmed = current_price > neckline * (1.0 + max(0.0, breakout_buffer_pct))
        volume_confirmed = _latest_volume_ratio(volume) >= min_volume_ratio if confirmed else False
        if level_diff <= level_tolerance_pct:
            rows.append(
                _pattern_row(
                    name="double_bottom",
                    direction="bullish",
                    status="confirmed" if confirmed else "forming",
                    confidence=0.72 + (0.08 if volume_confirmed else 0.0),
                    volume_confirmed=volume_confirmed,
                    reason="two_support_tests_with_neckline_breakout" if confirmed else "two_support_tests_forming",
                    entry_trigger="neckline_breakout",
                    stop_reference="lowest_trough",
                    target_reference="pattern_height_or_161_fibonacci_extension",
                    levels={"trough_1": l1[1], "trough_2": l2[1], "neckline": neckline},
                )
            )
    return rows


def _flag_patterns(
    close: pd.Series,
    *,
    current_price: float,
    volume: pd.Series | None,
    min_volume_ratio: float,
) -> list[dict[str, Any]]:
    if len(close) < 30:
        return []
    pole = close.tail(24).head(10)
    flag = close.tail(14)
    pole_move = float(pole.iloc[-1] / max(float(pole.iloc[0]), 1e-9) - 1.0)
    flag_move = float(flag.iloc[-1] / max(float(flag.iloc[0]), 1e-9) - 1.0)
    retrace = abs(flag_move) / max(abs(pole_move), 1e-9)
    latest_ratio = _latest_volume_ratio(volume)
    flag_volume_ratio = _tail_volume_ratio(volume, tail=14)
    rows: list[dict[str, Any]] = []
    if pole_move >= 0.012 and retrace <= 0.50 and abs(flag_move) <= abs(pole_move) * 0.55:
        confirmed = current_price >= float(flag.max())
        rows.append(
            _pattern_row(
                name="bullish_flag",
                direction="bullish",
                status="confirmed" if confirmed and latest_ratio >= min_volume_ratio else "forming",
                confidence=0.66 + (0.08 if latest_ratio >= min_volume_ratio else 0.0) + (0.04 if flag_volume_ratio <= 0.80 else 0.0),
                volume_confirmed=latest_ratio >= min_volume_ratio,
                reason="sharp_up_pole_with_shallow_consolidation",
                entry_trigger="flag_breakout_with_expanding_volume",
                stop_reference="flag_low",
                target_reference="flagpole_extension",
                levels={"pole_move_pct": pole_move, "flag_retrace_ratio": retrace, "flag_high": float(flag.max()), "flag_low": float(flag.min())},
            )
        )
    if pole_move <= -0.012 and retrace <= 0.50 and abs(flag_move) <= abs(pole_move) * 0.55:
        confirmed = current_price <= float(flag.min())
        rows.append(
            _pattern_row(
                name="bearish_flag",
                direction="bearish",
                status="confirmed" if confirmed and latest_ratio >= min_volume_ratio else "forming",
                confidence=0.66 + (0.08 if latest_ratio >= min_volume_ratio else 0.0) + (0.04 if flag_volume_ratio <= 0.80 else 0.0),
                volume_confirmed=latest_ratio >= min_volume_ratio,
                reason="sharp_down_pole_with_shallow_bounce",
                entry_trigger="flag_breakdown_with_expanding_volume",
                stop_reference="flag_high",
                target_reference="flagpole_extension",
                levels={"pole_move_pct": pole_move, "flag_retrace_ratio": retrace, "flag_high": float(flag.max()), "flag_low": float(flag.min())},
            )
        )
    return rows


def _wedge_patterns(
    close: pd.Series,
    *,
    current_price: float,
    volume: pd.Series | None,
    min_volume_ratio: float,
    breakout_buffer_pct: float,
) -> list[dict[str, Any]]:
    if len(close) < 36:
        return []
    window = close.tail(36)
    chunks = np.array_split(window.to_numpy(dtype=float), 6)
    highs = np.array([float(np.max(chunk)) for chunk in chunks])
    lows = np.array([float(np.min(chunk)) for chunk in chunks])
    x = np.arange(len(highs), dtype=float)
    high_slope = float(np.polyfit(x, highs, 1)[0] / max(float(np.mean(highs)), 1e-9))
    low_slope = float(np.polyfit(x, lows, 1)[0] / max(float(np.mean(lows)), 1e-9))
    early_width = highs[:2].mean() - lows[:2].mean()
    late_width = highs[-2:].mean() - lows[-2:].mean()
    narrowing = late_width < early_width * 0.75
    latest_ratio = _latest_volume_ratio(volume)
    rows: list[dict[str, Any]] = []
    if narrowing and high_slope > 0 and low_slope > 0 and low_slope > high_slope:
        support = float(lows[-1])
        confirmed = current_price < support * (1.0 - breakout_buffer_pct)
        rows.append(
            _pattern_row(
                name="ascending_wedge",
                direction="bearish",
                status="confirmed" if confirmed and latest_ratio >= min_volume_ratio else "forming",
                confidence=0.68 + (0.08 if confirmed and latest_ratio >= min_volume_ratio else 0.0),
                volume_confirmed=latest_ratio >= min_volume_ratio if confirmed else False,
                reason="rising_narrowing_channel_warns_buyer_exhaustion",
                entry_trigger="lower_trendline_break",
                stop_reference="recent_wedge_high",
                target_reference="127_fibonacci_extension_or_wedge_height",
                levels={"support": support, "recent_high": float(highs[-1]), "width_contraction_ratio": late_width / max(early_width, 1e-9)},
            )
        )
    if narrowing and high_slope < 0 and low_slope < 0 and high_slope > low_slope:
        resistance = float(highs[-1])
        confirmed = current_price > resistance * (1.0 + breakout_buffer_pct)
        rows.append(
            _pattern_row(
                name="descending_wedge",
                direction="bullish",
                status="confirmed" if confirmed and latest_ratio >= min_volume_ratio else "forming",
                confidence=0.70 + (0.08 if confirmed and latest_ratio >= min_volume_ratio else 0.0),
                volume_confirmed=latest_ratio >= min_volume_ratio if confirmed else False,
                reason="falling_narrowing_channel_warns_selling_exhaustion",
                entry_trigger="upper_trendline_breakout",
                stop_reference="recent_wedge_low",
                target_reference="161_fibonacci_extension_or_wedge_height",
                levels={"resistance": resistance, "recent_low": float(lows[-1]), "width_contraction_ratio": late_width / max(early_width, 1e-9)},
            )
        )
    return rows


def _summarize_patterns(patterns: list[dict[str, Any]], *, forecast_direction: str) -> dict[str, Any]:
    confirmed = [row for row in patterns if row.get("status") == "confirmed"]
    active = confirmed or patterns[:3]
    supportive = [row for row in active if row.get("direction") == forecast_direction]
    conflicting = [row for row in active if forecast_direction != "flat" and row.get("direction") != forecast_direction]
    best = active[0] if active else None
    if supportive and not conflicting:
        permission = "supportive"
        reason = "chart_pattern_aligns_with_forecast"
    elif conflicting and not supportive:
        permission = "conflict"
        reason = "chart_pattern_conflicts_with_forecast"
    elif supportive and conflicting:
        permission = "mixed"
        reason = "mixed_chart_patterns"
    else:
        permission = "neutral"
        reason = "no_reliable_chart_pattern"
    return {
        "permission": permission,
        "reason": reason,
        "dominant_pattern": None if best is None else best.get("name"),
        "dominant_direction": None if best is None else best.get("direction"),
        "dominant_status": None if best is None else best.get("status"),
        "dominant_confidence": None if best is None else best.get("confidence"),
        "supportive_count": len(supportive),
        "conflicting_count": len(conflicting),
        "confirmed_count": len(confirmed),
    }


def _pattern_row(
    *,
    name: str,
    direction: str,
    status: str,
    confidence: float,
    volume_confirmed: bool,
    reason: str,
    entry_trigger: str,
    stop_reference: str,
    target_reference: str,
    levels: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "direction": direction,
        "status": status,
        "confidence": round(float(min(max(confidence, 0.0), 0.95)), 4),
        "volume_confirmed": bool(volume_confirmed),
        "reason": reason,
        "entry_trigger": entry_trigger,
        "stop_reference": stop_reference,
        "target_reference": target_reference,
        "levels": {key: round(float(value), 6) if isinstance(value, (int, float, np.floating)) else value for key, value in levels.items()},
    }


def _pivots(close: pd.Series, *, order: int) -> dict[str, list[tuple[Any, float]]]:
    highs: list[tuple[Any, float]] = []
    lows: list[tuple[Any, float]] = []
    values = close.to_numpy(dtype=float)
    index = list(close.index)
    for i in range(order, len(values) - order):
        segment = values[i - order : i + order + 1]
        value = values[i]
        if value == float(np.max(segment)) and value > float(np.min(segment)):
            highs.append((index[i], float(value)))
        if value == float(np.min(segment)) and value < float(np.max(segment)):
            lows.append((index[i], float(value)))
    return {"highs": highs, "lows": lows}


def _close_series(prices: pd.DataFrame) -> pd.Series:
    for column in ("close", "Close", "price", "last"):
        if column in prices.columns:
            return pd.to_numeric(prices[column], errors="coerce")
    numeric = prices.select_dtypes(include="number")
    if numeric.empty:
        return pd.Series(dtype=float)
    return pd.to_numeric(numeric.iloc[:, 0], errors="coerce")


def _volume_series(prices: pd.DataFrame) -> pd.Series | None:
    for column in ("volume", "Volume", "vol"):
        if column in prices.columns:
            return pd.to_numeric(prices[column], errors="coerce")
    return None


def _volume_at(volume: pd.Series | None, index: Any) -> float | None:
    if volume is None or index not in volume.index:
        return None
    return _float_or_none(volume.loc[index])


def _latest_volume_ratio(volume: pd.Series | None) -> float:
    if volume is None or len(volume.dropna()) < 20:
        return 1.0
    clean = volume.dropna()
    latest = float(clean.iloc[-1])
    baseline = float(clean.tail(60).iloc[:-1].mean()) if len(clean) > 1 else latest
    return latest / max(baseline, 1e-9)


def _tail_volume_ratio(volume: pd.Series | None, *, tail: int) -> float:
    if volume is None or len(volume.dropna()) < tail + 10:
        return 1.0
    clean = volume.dropna()
    recent = float(clean.tail(tail).mean())
    baseline = float(clean.tail(max(40, tail * 3)).mean())
    return recent / max(baseline, 1e-9)


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed
