from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PIVOT_LEFT_BARS = 3
PIVOT_RIGHT_BARS = 3
MIN_PATTERN_BARS = 35
BREAKOUT_PCT = 0.003
RETEST_PCT = 0.02
FLAT_SLOPE_PCT_PER_BAR = 0.0007
TREND_SLOPE_PCT_PER_BAR = 0.0009
MIN_WIDTH_COMPRESSION = 0.20
MIN_BOUNDARY_TOUCHES = 2
TRIANGLE_WINDOWS = {
    "daily": (45, 63, 90, 126),
    "weekly": (26, 39, 52, 78),
    "monthly": (18, 24, 36, 48),
    "chart": (35, 50, 70, 90),
}


def analyze_triangle_patterns(prices: pd.DataFrame, target_column: str = "close") -> dict[str, Any]:
    """Analyze Edwards/Magee Chapter 8 triangle formations."""

    target = target_column.lower()
    frames = {
        "daily": prices.copy(),
        "weekly": _resample_ohlcv(prices, "W-FRI"),
        "monthly": _resample_ohlcv(prices, "ME"),
    }
    timeframes = {
        name: _analyze_timeframe(frame=frame, timeframe=name, target_column=target)
        for name, frame in frames.items()
    }
    preferred = _choose_preferred(timeframes)
    return {
        "principle": (
            "Edwards/Magee Chapter 8: triangles are often consolidations, not reversals; "
            "direction should usually wait for a decisive breakout, with apex timing and volume used as reliability controls."
        ),
        "primary_timeframe": "weekly",
        "secondary_timeframe": "daily",
        "higher_order_timeframe": "monthly",
        "confirmation_rule": f"Close beyond a triangle boundary by at least {BREAKOUT_PCT:.2%}.",
        "preferred": preferred,
        "timeframes": timeframes,
        "technical_method_card": triangle_patterns_method_card(target_column=target),
    }


def latest_triangle_patterns(
    prices: pd.DataFrame,
    target_column: str = "close",
    timeframe: str = "chart",
) -> dict[str, Any]:
    """Return triangle diagnostics for a plotted bar set."""

    return _analyze_timeframe(
        frame=prices,
        timeframe=timeframe,
        target_column=target_column.lower(),
    )


def triangle_patterns_method_card(target_column: str = "close") -> dict[str, Any]:
    return {
        "name": "edwards_magee_triangle_patterns",
        "version": "chapter_8_alignment_v1",
        "target_column": target_column.lower(),
        "implemented_patterns": ["symmetrical_triangle", "ascending_triangle", "descending_triangle"],
        "pivot_confirmation": {
            "left_bars": PIVOT_LEFT_BARS,
            "right_bars": PIVOT_RIGHT_BARS,
            "rule": "triangle boundaries are fit from confirmed pivot highs and lows",
        },
        "pattern_rules": {
            "symmetrical_triangle": "falling upper boundary and rising lower boundary; no directional bias until breakout",
            "ascending_triangle": "flat resistance and rising lower boundary; bullish bias only after upside breakout",
            "descending_triangle": "falling upper boundary and flat support; bearish bias only after downside breakdown",
        },
        "apex_timing": {
            "best_zone": "roughly 50%-75% of the base-to-apex distance",
            "late_apex_warning": "breakouts after the 75% mark are lower reliability",
        },
        "volume": {
            "inside_pattern": "volume should usually contract during formation",
            "upside_breakout": "volume expansion is important on upside breakouts",
            "downside_breakout": "volume expansion strengthens, but is less mandatory on downside breaks",
        },
        "objective": "minimum target is the breakout price plus/minus the triangle height near the base",
        "decision_use": "triangles support or block model-driven actions; they are not standalone Buy/Sell signals",
    }


def _analyze_timeframe(frame: pd.DataFrame, timeframe: str, target_column: str) -> dict[str, Any]:
    clean = frame.copy().sort_index()
    if clean.empty or target_column not in clean.columns:
        return {
            "state": "InsufficientData",
            "timeframe": timeframe,
            "rows": int(len(clean)),
            "preferred": _empty_pattern(timeframe, "InsufficientData", "missing target column", len(clean)),
        }
    clean.index = pd.DatetimeIndex(clean.index)
    clean = clean.loc[pd.to_numeric(clean[target_column], errors="coerce").notna()]
    if len(clean) < MIN_PATTERN_BARS:
        return {
            "state": "InsufficientData",
            "timeframe": timeframe,
            "rows": int(len(clean)),
            "preferred": _empty_pattern(timeframe, "InsufficientData", "not enough bars for triangle analysis", len(clean)),
        }

    pattern = _find_triangle(clean, timeframe=timeframe, target_column=target_column)
    return {
        "state": "Measured",
        "timeframe": timeframe,
        "rows": int(len(clean)),
        "start_date": str(clean.index[0].date()),
        "end_date": str(clean.index[-1].date()),
        "preferred": pattern,
    }


def _find_triangle(frame: pd.DataFrame, timeframe: str, target_column: str) -> dict[str, Any]:
    close = pd.to_numeric(frame[target_column], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce") if "high" in frame.columns else close
    low = pd.to_numeric(frame["low"], errors="coerce") if "low" in frame.columns else close
    volume = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame.columns else None
    pivot_high, pivot_low = _confirmed_pivots(high=high, low=low)
    candidates = []
    for window in TRIANGLE_WINDOWS.get(timeframe, TRIANGLE_WINDOWS["chart"]):
        if len(frame) < window:
            continue
        candidate = _triangle_candidate(
            close=close,
            high=high,
            low=low,
            volume=volume,
            pivot_high=pivot_high,
            pivot_low=pivot_low,
            timeframe=timeframe,
            window=window,
        )
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return _empty_pattern(timeframe, "NoPattern", "no valid converging triangle geometry", len(frame))
    return max(candidates, key=_pattern_rank)


def _triangle_candidate(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series | None,
    pivot_high: pd.Series,
    pivot_low: pd.Series,
    timeframe: str,
    window: int,
) -> dict[str, Any] | None:
    scope = close.tail(window)
    start_position = len(close) - len(scope)
    positions = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    high_points = pivot_high.reindex(scope.index).dropna()
    low_points = pivot_low.reindex(scope.index).dropna()
    if len(high_points) < MIN_BOUNDARY_TOUCHES or len(low_points) < MIN_BOUNDARY_TOUCHES:
        return None

    upper = _fit_boundary(positions.reindex(high_points.index), high_points)
    lower = _fit_boundary(positions.reindex(low_points.index), low_points)
    if upper is None or lower is None:
        return None

    latest_position = len(close) - 1
    base_position = max(start_position, int(min(upper["first_position"], lower["first_position"])))
    upper_base = _line_value(upper, base_position)
    lower_base = _line_value(lower, base_position)
    upper_latest = _line_value(upper, latest_position)
    lower_latest = _line_value(lower, latest_position)
    if min(upper_base, lower_base, upper_latest, lower_latest) <= 0:
        return None
    if upper_base <= lower_base or upper_latest <= lower_latest:
        return None

    base_width = upper_base - lower_base
    latest_width = upper_latest - lower_latest
    width_compression = 1.0 - latest_width / base_width if base_width > 0 else 0.0
    if width_compression < MIN_WIDTH_COMPRESSION:
        return None

    average_price = float(scope.mean())
    upper_slope_pct = upper["slope"] / average_price if average_price else 0.0
    lower_slope_pct = lower["slope"] / average_price if average_price else 0.0
    pattern_type = _classify_triangle(upper_slope_pct, lower_slope_pct)
    if pattern_type is None:
        return None

    apex_position = _apex_position(upper, lower)
    if apex_position is None:
        return None
    base_to_apex = apex_position - base_position
    if base_to_apex <= 0:
        return None
    apex_progress = (latest_position - base_position) / base_to_apex
    if apex_progress > 1.15:
        return None

    break_info = _breakout_state(
        close=close,
        upper=upper,
        lower=lower,
        start_position=int(max(upper["last_position"], lower["last_position"])),
    )
    volume_state = _volume_state(volume=volume, start_position=base_position, break_position=break_info.get("position"))
    retest = _retest_state(
        close=close,
        upper=upper,
        lower=lower,
        break_position=break_info.get("position"),
        direction=break_info.get("direction"),
    )
    status = _status_from_break(pattern_type=pattern_type, break_info=break_info, retest=retest, apex_progress=apex_progress)
    direction = _direction(pattern_type=pattern_type, status=status)
    breakout_price = break_info.get("close") or float(close.iloc[-1])
    height = base_width
    objective = None
    if status in {"Breakout", "Breakdown", "Retest", "FailedBreakout", "FailedBreakdown"}:
        objective = float(breakout_price + height) if break_info.get("direction") == "up" else float(breakout_price - height)
        if objective <= 0:
            objective = None

    geometry_score = _geometry_score(
        width_compression=width_compression,
        apex_progress=apex_progress,
        high_touches=len(high_points),
        low_touches=len(low_points),
        pattern_type=pattern_type,
    )
    volume_score = float(volume_state.get("score", 0.5))
    status_score = _status_score(status)
    reliability = float(0.50 * geometry_score + 0.25 * volume_score + 0.25 * status_score)
    if status == "LateApex":
        reliability *= 0.65

    return {
        "pattern": pattern_type,
        "status": status,
        "direction": direction,
        "timeframe": timeframe,
        "score": _finite_or_none(reliability),
        "window_bars": int(window),
        "latest_close": _finite_or_none(close.iloc[-1]),
        "breakout_date": break_info.get("date"),
        "breakout_close": break_info.get("close"),
        "breakout_direction": break_info.get("direction"),
        "breakout_margin_pct": break_info.get("margin_pct"),
        "measured_objective": _finite_or_none(objective),
        "measured_move_pct": _finite_or_none(objective / float(close.iloc[-1]) - 1.0) if objective and float(close.iloc[-1]) else None,
        "apex": {
            "date": _position_to_date(close.index, apex_position),
            "position": _finite_or_none(apex_position),
            "progress_pct": _finite_or_none(apex_progress),
            "timing": _apex_timing(apex_progress),
        },
        "boundaries": {
            "start_date": _date_string(close.index[base_position]),
            "latest_date": _date_string(close.index[-1]),
            "upper_start": _finite_or_none(upper_base),
            "upper_latest": _finite_or_none(upper_latest),
            "upper_slope_per_bar": _finite_or_none(upper["slope"]),
            "upper_slope_pct_per_bar": _finite_or_none(upper_slope_pct),
            "lower_start": _finite_or_none(lower_base),
            "lower_latest": _finite_or_none(lower_latest),
            "lower_slope_per_bar": _finite_or_none(lower["slope"]),
            "lower_slope_pct_per_bar": _finite_or_none(lower_slope_pct),
            "base_width": _finite_or_none(base_width),
            "latest_width": _finite_or_none(latest_width),
            "width_compression_pct": _finite_or_none(width_compression),
            "upper_touch_count": int(len(high_points)),
            "lower_touch_count": int(len(low_points)),
        },
        "volume_confirmation": volume_state,
        "retest": retest,
        "reliability_notes": _reliability_notes(status=status, apex_progress=apex_progress, volume_state=volume_state),
    }


def _confirmed_pivots(high: pd.Series, low: pd.Series) -> tuple[pd.Series, pd.Series]:
    window = PIVOT_LEFT_BARS + PIVOT_RIGHT_BARS + 1
    raw_high = high == high.rolling(window=window, center=True).max()
    raw_low = low == low.rolling(window=window, center=True).min()
    confirmed_high = high.shift(PIVOT_RIGHT_BARS).where(raw_high.shift(PIVOT_RIGHT_BARS, fill_value=False), np.nan)
    confirmed_low = low.shift(PIVOT_RIGHT_BARS).where(raw_low.shift(PIVOT_RIGHT_BARS, fill_value=False), np.nan)
    return confirmed_high, confirmed_low


def _fit_boundary(position: pd.Series, values: pd.Series) -> dict[str, Any] | None:
    aligned = pd.concat([position, values], axis=1).dropna()
    if len(aligned) < MIN_BOUNDARY_TOUCHES:
        return None
    x = aligned.iloc[:, 0].to_numpy(dtype=float)
    y = aligned.iloc[:, 1].to_numpy(dtype=float)
    if len(np.unique(x)) < 2:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "first_position": float(x.min()),
        "last_position": float(x.max()),
    }


def _classify_triangle(upper_slope_pct: float, lower_slope_pct: float) -> str | None:
    upper_falling = upper_slope_pct <= -TREND_SLOPE_PCT_PER_BAR
    lower_rising = lower_slope_pct >= TREND_SLOPE_PCT_PER_BAR
    upper_flat = abs(upper_slope_pct) <= FLAT_SLOPE_PCT_PER_BAR
    lower_flat = abs(lower_slope_pct) <= FLAT_SLOPE_PCT_PER_BAR
    if upper_falling and lower_rising:
        return "SymmetricalTriangle"
    if upper_flat and lower_rising:
        return "AscendingTriangle"
    if upper_falling and lower_flat:
        return "DescendingTriangle"
    return None


def _line_value(line: dict[str, Any], position: float) -> float:
    return float(line["intercept"] + line["slope"] * position)


def _apex_position(upper: dict[str, Any], lower: dict[str, Any]) -> float | None:
    denominator = float(upper["slope"] - lower["slope"])
    if abs(denominator) < 1e-12:
        return None
    return float((lower["intercept"] - upper["intercept"]) / denominator)


def _breakout_state(close: pd.Series, upper: dict[str, Any], lower: dict[str, Any], start_position: int) -> dict[str, Any]:
    start = max(0, min(start_position, len(close) - 1))
    latest_position = len(close) - 1
    output = {"state": "Inside", "direction": None, "position": None, "date": None, "close": None, "margin_pct": None}
    for position in range(start, len(close)):
        close_value = close.iloc[position]
        if pd.isna(close_value):
            continue
        upper_value = _line_value(upper, position)
        lower_value = _line_value(lower, position)
        if close_value > upper_value * (1.0 + BREAKOUT_PCT):
            output = _break_payload(close, position, "up", close_value / upper_value - 1.0)
            break
        if close_value < lower_value * (1.0 - BREAKOUT_PCT):
            output = _break_payload(close, position, "down", close_value / lower_value - 1.0)
            break
    if output["state"] != "Inside":
        latest_close = float(close.iloc[-1])
        latest_upper = _line_value(upper, latest_position)
        latest_lower = _line_value(lower, latest_position)
        if output["direction"] == "up" and latest_close < latest_upper:
            output["state"] = "Failed"
        elif output["direction"] == "down" and latest_close > latest_lower:
            output["state"] = "Failed"
    return output


def _break_payload(close: pd.Series, position: int, direction: str, margin: float) -> dict[str, Any]:
    return {
        "state": "Breakout" if direction == "up" else "Breakdown",
        "direction": direction,
        "position": int(position),
        "date": _date_string(close.index[position]),
        "close": _finite_or_none(close.iloc[position]),
        "margin_pct": _finite_or_none(margin),
    }


def _retest_state(close: pd.Series, upper: dict[str, Any], lower: dict[str, Any], break_position: int | None, direction: str | None) -> dict[str, Any]:
    if break_position is None or direction not in {"up", "down"}:
        return {"state": "NotApplicable", "latest_retest": False, "last_retest_date": None}
    last_date = None
    latest = False
    for position in range(int(break_position) + 1, len(close)):
        close_value = close.iloc[position]
        boundary = _line_value(upper if direction == "up" else lower, position)
        if pd.isna(close_value) or boundary <= 0:
            continue
        if abs(float(close_value) / boundary - 1.0) <= RETEST_PCT:
            last_date = _date_string(close.index[position])
            latest = position >= len(close) - 5
    return {"state": "Observed" if last_date else "NotObserved", "latest_retest": bool(latest), "last_retest_date": last_date}


def _status_from_break(pattern_type: str, break_info: dict[str, Any], retest: dict[str, Any], apex_progress: float) -> str:
    if break_info.get("state") == "Failed":
        return "FailedBreakout" if break_info.get("direction") == "up" else "FailedBreakdown"
    if break_info.get("state") in {"Breakout", "Breakdown"}:
        if retest.get("latest_retest"):
            return "Retest"
        return str(break_info["state"])
    if apex_progress >= 0.75:
        return "LateApex"
    return "Candidate"


def _direction(pattern_type: str, status: str) -> str:
    if status in {"Breakout", "Retest"}:
        return "bullish"
    if status == "Breakdown":
        return "bearish"
    if status == "FailedBreakout":
        return "bearish"
    if status == "FailedBreakdown":
        return "bullish"
    if pattern_type == "AscendingTriangle":
        return "bullish_bias"
    if pattern_type == "DescendingTriangle":
        return "bearish_bias"
    return "undetermined"


def _volume_state(volume: pd.Series | None, start_position: int, break_position: int | None) -> dict[str, Any]:
    if volume is None or volume.dropna().empty:
        return {"state": "Unavailable", "score": 0.5}
    scope = volume.iloc[start_position:]
    if scope.dropna().empty:
        return {"state": "Unavailable", "score": 0.5}
    first_half = scope.iloc[: max(1, len(scope) // 2)]
    second_half = scope.iloc[max(1, len(scope) // 2) :]
    contraction = None
    if first_half.notna().any() and second_half.notna().any() and float(first_half.median()) > 0:
        contraction = bool(float(second_half.median()) <= float(first_half.median()) * 0.9)
    breakout_expansion = None
    break_volume = None
    if break_position is not None:
        break_volume = float(volume.iloc[int(break_position)]) if pd.notna(volume.iloc[int(break_position)]) else None
        baseline = volume.shift(1).rolling(20, min_periods=5).mean().iloc[int(break_position)]
        if break_volume is not None and pd.notna(baseline) and float(baseline) > 0:
            breakout_expansion = bool(break_volume >= float(baseline) * 1.2)
    score = 0.5
    if contraction is True:
        score += 0.2
    if breakout_expansion is True:
        score += 0.25
    elif breakout_expansion is False:
        score -= 0.10
    return {
        "state": "Measured",
        "volume_contracts_inside_pattern": contraction,
        "breakout_volume_expansion": breakout_expansion,
        "break_volume": _finite_or_none(break_volume),
        "score": _finite_or_none(min(max(score, 0.0), 1.0)),
    }


def _geometry_score(width_compression: float, apex_progress: float, high_touches: int, low_touches: int, pattern_type: str) -> float:
    compression_score = _clip01(width_compression / 0.65)
    touch_score = _clip01((min(high_touches, 5) + min(low_touches, 5) - 4) / 6)
    if 0.50 <= apex_progress <= 0.75:
        timing_score = 1.0
    elif 0.35 <= apex_progress < 0.50:
        timing_score = 0.75
    elif 0.75 < apex_progress <= 0.90:
        timing_score = 0.45
    else:
        timing_score = 0.25
    type_score = 0.9 if pattern_type in {"AscendingTriangle", "DescendingTriangle"} else 0.75
    return float(0.35 * compression_score + 0.25 * touch_score + 0.25 * timing_score + 0.15 * type_score)


def _status_score(status: str) -> float:
    return {
        "Breakout": 0.95,
        "Breakdown": 0.95,
        "Retest": 0.90,
        "Candidate": 0.55,
        "LateApex": 0.25,
        "FailedBreakout": 0.35,
        "FailedBreakdown": 0.35,
    }.get(status, 0.20)


def _apex_timing(apex_progress: float) -> str:
    if apex_progress < 0.50:
        return "Early"
    if apex_progress <= 0.75:
        return "PreferredZone"
    if apex_progress <= 1.0:
        return "LateApex"
    return "PastApex"


def _position_to_date(index: pd.Index, position: float) -> str | None:
    if not np.isfinite(position):
        return None
    rounded = int(round(position))
    if 0 <= rounded < len(index):
        return _date_string(index[rounded])
    if len(index) < 2:
        return None
    step = pd.Timestamp(index[-1]) - pd.Timestamp(index[-2])
    projected = pd.Timestamp(index[-1]) + step * max(0, rounded - len(index) + 1)
    return _date_string(projected)


def _reliability_notes(status: str, apex_progress: float, volume_state: dict[str, Any]) -> list[str]:
    notes = []
    if status == "LateApex":
        notes.append("Breakout has not occurred before the late-apex zone; reliability is reduced.")
    if apex_progress > 0.75:
        notes.append("Pattern is beyond the preferred half-to-three-quarter breakout zone.")
    if volume_state.get("breakout_volume_expansion") is False:
        notes.append("Breakout/breakdown lacks volume expansion.")
    if volume_state.get("volume_contracts_inside_pattern") is False:
        notes.append("Volume did not contract clearly inside the triangle.")
    return notes


def _choose_preferred(timeframes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    patterns = [
        payload.get("preferred", {})
        for payload in timeframes.values()
        if isinstance(payload.get("preferred"), dict)
    ]
    actionable = [pattern for pattern in patterns if pattern.get("status") not in {"NoPattern", "InsufficientData"}]
    if actionable:
        return _preferred_payload(max(actionable, key=_pattern_rank))
    for timeframe in ("monthly", "weekly", "daily"):
        pattern = timeframes.get(timeframe, {}).get("preferred", {})
        if pattern:
            return _preferred_payload(pattern)
    return _empty_pattern("unknown", "NoPattern", "pattern unavailable", 0)


def _pattern_rank(pattern: dict[str, Any]) -> tuple[float, float, str]:
    status_rank = {
        "Retest": 5.0,
        "Breakout": 4.5,
        "Breakdown": 4.5,
        "Candidate": 2.0,
        "LateApex": 1.0,
        "FailedBreakout": 0.8,
        "FailedBreakdown": 0.8,
        "NoPattern": 0.0,
        "InsufficientData": 0.0,
    }.get(str(pattern.get("status")), 0.0)
    timeframe_bonus = {"monthly": 0.30, "weekly": 0.15, "daily": 0.0}.get(str(pattern.get("timeframe")), 0.0)
    score = float(pattern.get("score") or 0.0)
    date = pattern.get("breakout_date") or pattern.get("boundaries", {}).get("latest_date") or ""
    return (status_rank + timeframe_bonus, score, str(date))


def _preferred_payload(pattern: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "pattern",
        "status",
        "direction",
        "timeframe",
        "score",
        "window_bars",
        "latest_close",
        "breakout_date",
        "breakout_close",
        "breakout_direction",
        "breakout_margin_pct",
        "measured_objective",
        "measured_move_pct",
        "apex",
        "boundaries",
        "volume_confirmation",
        "retest",
        "reliability_notes",
        "reason",
    ]
    return {key: pattern[key] for key in keys if key in pattern}


def _empty_pattern(timeframe: str, status: str, reason: str, rows: int) -> dict[str, Any]:
    return {
        "pattern": "NoTriangle",
        "status": status,
        "timeframe": timeframe,
        "rows": int(rows),
        "reason": reason,
    }


def _resample_ohlcv(prices: pd.DataFrame, rule: str) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()
    aggregations: dict[str, str] = {}
    if "open" in prices.columns:
        aggregations["open"] = "first"
    if "high" in prices.columns:
        aggregations["high"] = "max"
    if "low" in prices.columns:
        aggregations["low"] = "min"
    if "close" in prices.columns:
        aggregations["close"] = "last"
    if "volume" in prices.columns:
        aggregations["volume"] = "sum"
    for optional in ("dividends", "stock_splits"):
        if optional in prices.columns:
            aggregations[optional] = "sum"
    frame = prices.resample(rule).agg(aggregations)
    return frame.dropna(subset=["close"]) if "close" in frame.columns else frame.dropna(how="all")


def _date_string(value: Any) -> str:
    return str(pd.Timestamp(value).date())


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _finite_or_none(value: Any) -> float | None:
    try:
        output = float(value)
    except Exception:
        return None
    return output if np.isfinite(output) else None
