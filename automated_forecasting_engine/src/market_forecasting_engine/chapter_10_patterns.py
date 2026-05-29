from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PIVOT_LEFT_BARS = 3
PIVOT_RIGHT_BARS = 3
MIN_PATTERN_BARS = 45
BREAKOUT_PCT = 0.03
RETEST_PCT = 0.03
MIN_BOUNDARY_TOUCHES = 2
MIN_WIDTH_COMPRESSION = 0.18
MIN_SLOPE_PCT_PER_BAR = 0.00055
BROADENING_STEP_PCT = 0.01
FLAT_TOLERANCE_PCT = 0.03
WEDGE_WINDOWS = {
    "daily": (45, 63, 90, 126),
    "weekly": (26, 39, 52, 78),
    "monthly": (18, 24, 36, 48),
    "chart": (35, 50, 70, 90),
}
BROADENING_LOOKBACK = {"daily": 160, "weekly": 78, "monthly": 48, "chart": 130}
DIAMOND_LOOKBACK = {"daily": 140, "weekly": 70, "monthly": 42, "chart": 110}
PRIOR_TREND_LOOKBACK_BARS = {"daily": 126, "weekly": 52, "monthly": 36, "chart": 90}
PRIOR_TREND_MIN_RETURN = {"daily": 0.08, "weekly": 0.10, "monthly": 0.12, "chart": 0.06}


def analyze_chapter_10_patterns(prices: pd.DataFrame, target_column: str = "close") -> dict[str, Any]:
    """Analyze Edwards/Magee Chapter 10 reversal phenomena."""

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
    structural = {
        "principle": (
            "Edwards/Magee Chapter 10 structural reversals are warning-heavy formations: "
            "broadening tops imply disorder, wedges show tiring moves, and diamonds need clean breakout confirmation."
        ),
        "preferred": _choose_preferred_structural(timeframes),
        "timeframes": {
            name: {
                "state": payload.get("state"),
                "rows": payload.get("rows"),
                "broadening": payload.get("broadening", {}),
                "wedge": payload.get("wedge", {}),
                "diamond": payload.get("diamond", {}),
                "preferred": payload.get("structural_preferred", {}),
            }
            for name, payload in timeframes.items()
        },
    }
    short_term = {
        "principle": (
            "One-day reversals, spikes, runaway days, and selling climaxes are tactical events; "
            "they warn or block very near-term conflicts but should not replace validated forecasts."
        ),
        "preferred": _choose_preferred_event(timeframes),
        "timeframes": {
            name: {
                "state": payload.get("state"),
                "rows": payload.get("rows"),
                "events": payload.get("short_term_events", {}).get("events", []),
                "preferred": payload.get("short_term_events", {}).get("preferred", {}),
            }
            for name, payload in timeframes.items()
        },
    }
    return {
        "principle": (
            "Chapter 10 adds exhaustion and disorder controls. Structural confirmations can block conflicting "
            "model actions; one-day phenomena are mostly warnings unless they conflict with a fresh directional action."
        ),
        "primary_timeframe": "weekly",
        "secondary_timeframe": "daily",
        "higher_order_timeframe": "monthly",
        "confirmation_rule": f"Structural patterns require a close at least {BREAKOUT_PCT:.0%} beyond the relevant boundary.",
        "preferred": _choose_chapter_10_preferred(structural["preferred"], short_term["preferred"]),
        "structural_patterns": structural,
        "short_term_events": short_term,
        "timeframes": timeframes,
        "technical_method_card": chapter_10_patterns_method_card(target_column=target),
    }


def latest_chapter_10_patterns(
    prices: pd.DataFrame,
    target_column: str = "close",
    timeframe: str = "chart",
) -> dict[str, Any]:
    """Return Chapter 10 diagnostics for a plotted bar set."""

    return _analyze_timeframe(
        frame=prices,
        timeframe=timeframe,
        target_column=target_column.lower(),
    )


def chapter_10_patterns_method_card(target_column: str = "close") -> dict[str, Any]:
    return {
        "name": "edwards_magee_chapter_10_patterns",
        "version": "chapter_10_alignment_v1",
        "target_column": target_column.lower(),
        "implemented_patterns": [
            "broadening_top",
            "right_angled_broadening",
            "diamond",
            "rising_wedge",
            "falling_wedge",
            "one_day_reversal",
            "selling_climax",
            "spike",
            "runaway_day",
            "key_reversal_day",
        ],
        "structural_rules": {
            "broadening_top": "three higher highs with two lower intervening lows after a prior advance; bearish only after downside confirmation",
            "right_angled_broadening": "one horizontal boundary and one expanding boundary; horizontal-side breaks carry more force",
            "diamond": "early broadening followed by converging swings; require a clean boundary break",
            "wedge": "both boundaries slope in the same direction and converge; rising wedges are bearish after downside break, falling wedges are bullish after upside break",
            "confirmation": f"close at least {BREAKOUT_PCT:.0%} beyond the active boundary",
        },
        "short_term_rules": {
            "one_day_reversal": "wide-range exhaustion day after a strong move with close near the opposite side of the range",
            "selling_climax": "panic decline with downside gap, extreme range, extreme volume, and close recovery",
            "spike": "unusually prominent high or low that needs follow-through observation",
            "runaway_day": "unusually wide range with close near the direction of travel; invalidated if price returns to origin",
            "key_reversal_day": "new high then close below prior close, or new low then close above prior close",
        },
        "decision_use": "Chapter 10 patterns can block conflicting model actions but are not standalone Buy/Sell generators.",
    }


def _analyze_timeframe(frame: pd.DataFrame, timeframe: str, target_column: str) -> dict[str, Any]:
    clean = frame.copy().sort_index()
    if clean.empty or target_column not in clean.columns:
        return _empty_timeframe(timeframe, "InsufficientData", "missing target column", len(clean))
    clean.index = pd.DatetimeIndex(clean.index)
    clean = clean.loc[pd.to_numeric(clean[target_column], errors="coerce").notna()]
    if len(clean) < MIN_PATTERN_BARS:
        return _empty_timeframe(timeframe, "InsufficientData", "not enough bars for Chapter 10 analysis", len(clean))

    broadening = _find_broadening(clean, timeframe=timeframe, target_column=target_column)
    wedge = _find_wedge(clean, timeframe=timeframe, target_column=target_column)
    diamond = _find_diamond(clean, timeframe=timeframe, target_column=target_column)
    structural_preferred = max([broadening, wedge, diamond], key=_structural_rank)
    short_term_events = _short_term_events(clean, timeframe=timeframe, target_column=target_column)
    return {
        "state": "Measured",
        "timeframe": timeframe,
        "rows": int(len(clean)),
        "start_date": _date_string(clean.index[0]),
        "end_date": _date_string(clean.index[-1]),
        "broadening": broadening,
        "wedge": wedge,
        "diamond": diamond,
        "structural_preferred": _preferred_structural_payload(structural_preferred),
        "short_term_events": short_term_events,
    }


def _find_broadening(frame: pd.DataFrame, timeframe: str, target_column: str) -> dict[str, Any]:
    close = pd.to_numeric(frame[target_column], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce") if "high" in frame.columns else close
    low = pd.to_numeric(frame["low"], errors="coerce") if "low" in frame.columns else close
    volume = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame.columns else None
    pivot_high, pivot_low = _confirmed_pivots(high=high, low=low)
    lookback = min(len(frame), BROADENING_LOOKBACK.get(timeframe, BROADENING_LOOKBACK["chart"]))
    scope = close.tail(lookback)
    positions = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    high_points = pivot_high.reindex(scope.index).dropna()
    low_points = pivot_low.reindex(scope.index).dropna()
    candidates = []
    candidates.extend(_orthodox_broadening_candidates(close, high_points, low_points, positions, volume, timeframe))
    candidates.extend(_right_angled_broadening_candidates(close, high_points, low_points, positions, volume, timeframe))
    if not candidates:
        return _empty_pattern("NoBroadeningPattern", timeframe, "NoPattern", "no valid broadening reversal geometry", len(frame))
    return max(candidates, key=_structural_rank)


def _orthodox_broadening_candidates(
    close: pd.Series,
    high_points: pd.Series,
    low_points: pd.Series,
    positions: pd.Series,
    volume: pd.Series | None,
    timeframe: str,
) -> list[dict[str, Any]]:
    candidates = []
    highs = list(high_points.items())[-24:]
    lows = list(low_points.items())[-24:]
    for h1_idx in range(len(highs)):
        for h2_idx in range(h1_idx + 1, len(highs)):
            for h3_idx in range(h2_idx + 1, len(highs)):
                h1_date, h1 = highs[h1_idx]
                h2_date, h2 = highs[h2_idx]
                h3_date, h3 = highs[h3_idx]
                h1_pos, h2_pos, h3_pos = int(positions.loc[h1_date]), int(positions.loc[h2_date]), int(positions.loc[h3_date])
                between_1 = [(date, value) for date, value in lows if h1_pos < int(positions.loc[date]) < h2_pos]
                between_2 = [(date, value) for date, value in lows if h2_pos < int(positions.loc[date]) < h3_pos]
                if not between_1 or not between_2:
                    continue
                l1_date, l1 = min(between_1, key=lambda item: float(item[1]))
                l2_date, l2 = min(between_2, key=lambda item: float(item[1]))
                if h2 <= h1 * (1.0 + BROADENING_STEP_PCT) or h3 <= h2 * (1.0 + BROADENING_STEP_PCT):
                    continue
                if l2 >= l1 * (1.0 - BROADENING_STEP_PCT):
                    continue
                prior_trend = _prior_trend(close, h1_pos, timeframe=timeframe, require_advance=True)
                if prior_trend["state"] != "PriorAdvance":
                    continue
                candidate = _broadening_payload(
                    close=close,
                    volume=volume,
                    timeframe=timeframe,
                    pattern="BroadeningTop",
                    points=[
                        ("top_1", h1_date, float(h1)),
                        ("bottom_1", l1_date, float(l1)),
                        ("top_2", h2_date, float(h2)),
                        ("bottom_2", l2_date, float(l2)),
                        ("top_3", h3_date, float(h3)),
                    ],
                    upper_start=(h1_date, float(h1)),
                    upper_latest=(h3_date, float(h3)),
                    lower_start=(l1_date, float(l1)),
                    lower_latest=(l2_date, float(l2)),
                    confirmation_level=float(l2),
                    bullish_level=float(h3),
                    prior_trend=prior_trend,
                    start_position=h1_pos,
                    last_position=h3_pos,
                    bearish_default=True,
                )
                candidates.append(candidate)
    return candidates


def _right_angled_broadening_candidates(
    close: pd.Series,
    high_points: pd.Series,
    low_points: pd.Series,
    positions: pd.Series,
    volume: pd.Series | None,
    timeframe: str,
) -> list[dict[str, Any]]:
    candidates = []
    highs = list(high_points.items())[-18:]
    lows = list(low_points.items())[-18:]
    if len(highs) < 2 or len(lows) < 2:
        return candidates
    for first in range(len(highs) - 1):
        selected_highs = highs[first:first + 3]
        if len(selected_highs) < 2:
            continue
        high_values = [float(item[1]) for item in selected_highs]
        top_level = float(np.median(high_values))
        if top_level <= 0 or max(abs(value / top_level - 1.0) for value in high_values) > FLAT_TOLERANCE_PCT:
            continue
        high_positions = [int(positions.loc[date]) for date, _ in selected_highs]
        local_lows = [(date, value) for date, value in lows if min(high_positions) < int(positions.loc[date]) < max(high_positions) + 25]
        if len(local_lows) < 2:
            continue
        low_values = [float(value) for _, value in local_lows[-2:]]
        if low_values[-1] >= low_values[0] * (1.0 - BROADENING_STEP_PCT):
            continue
        prior_trend = _prior_trend(close, min(high_positions), timeframe=timeframe, require_advance=True)
        if prior_trend["state"] != "PriorAdvance":
            continue
        l1_date, l1 = local_lows[-2]
        l2_date, l2 = local_lows[-1]
        candidates.append(
            _broadening_payload(
                close=close,
                volume=volume,
                timeframe=timeframe,
                pattern="FlatToppedBroadening",
                points=[
                    ("top_1", selected_highs[0][0], float(selected_highs[0][1])),
                    ("bottom_1", l1_date, float(l1)),
                    ("top_2", selected_highs[-1][0], float(selected_highs[-1][1])),
                    ("bottom_2", l2_date, float(l2)),
                ],
                upper_start=(selected_highs[0][0], top_level),
                upper_latest=(selected_highs[-1][0], top_level),
                lower_start=(l1_date, float(l1)),
                lower_latest=(l2_date, float(l2)),
                confirmation_level=float(l2),
                bullish_level=top_level,
                prior_trend=prior_trend,
                start_position=min(high_positions),
                last_position=max(int(positions.loc[l2_date]), max(high_positions)),
                bearish_default=True,
            )
        )
    return candidates


def _broadening_payload(
    close: pd.Series,
    volume: pd.Series | None,
    timeframe: str,
    pattern: str,
    points: list[tuple[str, Any, float]],
    upper_start: tuple[Any, float],
    upper_latest: tuple[Any, float],
    lower_start: tuple[Any, float],
    lower_latest: tuple[Any, float],
    confirmation_level: float,
    bullish_level: float,
    prior_trend: dict[str, Any],
    start_position: int,
    last_position: int,
    bearish_default: bool,
) -> dict[str, Any]:
    break_info = _horizontal_breakout(close, lower=confirmation_level, upper=bullish_level, start_position=last_position)
    status = "Candidate"
    direction = "bearish_bias" if bearish_default else "undetermined"
    if break_info.get("direction") == "down":
        status = "Confirmed"
        direction = "bearish"
    elif break_info.get("direction") == "up":
        status = "UpsideBreakout"
        direction = "bullish"
    latest_close = float(close.iloc[-1])
    height = max(value for _, _, value in points) - min(value for _, _, value in points)
    objective = None
    if status == "Confirmed":
        objective = confirmation_level - height
        if objective <= 0:
            objective = None
        if objective is not None and latest_close <= objective:
            status = "ObjectiveReached"
    retest = _horizontal_retest(close, break_info.get("position"), confirmation_level if break_info.get("direction") == "down" else bullish_level)
    if status == "Confirmed" and retest.get("latest_retest"):
        status = "PullbackToBoundary"
    volume_state = _volume_state(volume=volume, start_position=start_position, break_position=break_info.get("position"))
    score = float(0.45 + 0.20 * float(volume_state.get("score", 0.5)) + 0.20 * _status_score(status) + 0.15 * _clip01(height / max(latest_close, 1e-9)))
    return {
        "pattern": pattern,
        "status": status,
        "direction": direction,
        "timeframe": timeframe,
        "score": _finite_or_none(score),
        "latest_close": _finite_or_none(latest_close),
        "confirmation_level": _finite_or_none(confirmation_level),
        "breakout_date": break_info.get("date"),
        "breakout_close": break_info.get("close"),
        "breakout_direction": break_info.get("direction"),
        "breakout_margin_pct": break_info.get("margin_pct"),
        "measured_objective": _finite_or_none(objective),
        "measured_move_pct": _finite_or_none(objective / latest_close - 1.0) if objective and latest_close else None,
        "prior_trend": prior_trend,
        "points": {name: {"date": _date_string(date), "price": _finite_or_none(value)} for name, date, value in points},
        "boundaries": {
            "start_date": _date_string(upper_start[0]),
            "latest_date": _date_string(max(upper_latest[0], lower_latest[0])),
            "upper_start": _finite_or_none(upper_start[1]),
            "upper_latest": _finite_or_none(upper_latest[1]),
            "lower_start": _finite_or_none(lower_start[1]),
            "lower_latest": _finite_or_none(lower_latest[1]),
            "height": _finite_or_none(height),
        },
        "volume_confirmation": volume_state,
        "retest": retest,
        "reliability_notes": _broadening_notes(status, volume_state),
    }


def _find_wedge(frame: pd.DataFrame, timeframe: str, target_column: str) -> dict[str, Any]:
    close = pd.to_numeric(frame[target_column], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce") if "high" in frame.columns else close
    low = pd.to_numeric(frame["low"], errors="coerce") if "low" in frame.columns else close
    volume = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame.columns else None
    pivot_high, pivot_low = _confirmed_pivots(high=high, low=low)
    candidates = []
    for window in WEDGE_WINDOWS.get(timeframe, WEDGE_WINDOWS["chart"]):
        if len(frame) < window:
            continue
        candidate = _wedge_candidate(close, pivot_high, pivot_low, volume, timeframe, window)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return _empty_pattern("NoWedge", timeframe, "NoPattern", "no valid converging same-direction boundaries", len(frame))
    return max(candidates, key=_structural_rank)


def _wedge_candidate(
    close: pd.Series,
    pivot_high: pd.Series,
    pivot_low: pd.Series,
    volume: pd.Series | None,
    timeframe: str,
    window: int,
) -> dict[str, Any] | None:
    scope = close.tail(window)
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
    base_position = max(0, int(min(upper["first_position"], lower["first_position"])))
    upper_base = _line_value(upper, base_position)
    lower_base = _line_value(lower, base_position)
    upper_latest = _line_value(upper, latest_position)
    lower_latest = _line_value(lower, latest_position)
    if min(upper_base, lower_base, upper_latest, lower_latest) <= 0 or upper_base <= lower_base or upper_latest <= lower_latest:
        return None
    base_width = upper_base - lower_base
    latest_width = upper_latest - lower_latest
    compression = 1.0 - latest_width / base_width if base_width > 0 else 0.0
    if compression < MIN_WIDTH_COMPRESSION:
        return None
    average_price = float(scope.mean())
    upper_slope_pct = upper["slope"] / average_price if average_price else 0.0
    lower_slope_pct = lower["slope"] / average_price if average_price else 0.0
    if upper_slope_pct >= MIN_SLOPE_PCT_PER_BAR and lower_slope_pct > upper_slope_pct + MIN_SLOPE_PCT_PER_BAR * 0.35:
        pattern = "RisingWedge"
        expected_break = "down"
        direction = "bearish_bias"
    elif lower_slope_pct <= -MIN_SLOPE_PCT_PER_BAR and upper_slope_pct < lower_slope_pct - MIN_SLOPE_PCT_PER_BAR * 0.35:
        pattern = "FallingWedge"
        expected_break = "up"
        direction = "bullish_bias"
    else:
        return None
    break_info = _line_breakout(close, upper, lower, int(max(upper["last_position"], lower["last_position"])))
    retest = _line_retest(close, upper, lower, break_info.get("position"), break_info.get("direction"))
    status = "Candidate"
    if break_info.get("direction") == expected_break:
        status = "Breakdown" if expected_break == "down" else "Breakout"
        direction = "bearish" if expected_break == "down" else "bullish"
    elif break_info.get("direction") in {"up", "down"}:
        status = "OppositeBreak"
        direction = "failed_bearish" if expected_break == "down" else "failed_bullish"
    if status in {"Breakout", "Breakdown"} and retest.get("latest_retest"):
        status = "Retest"
    latest_close = float(close.iloc[-1])
    objective = None
    if pattern == "RisingWedge" and status in {"Breakdown", "Retest"}:
        objective = float(lower_base)
    elif pattern == "FallingWedge" and status in {"Breakout", "Retest"}:
        objective = float(upper_base)
    volume_state = _volume_state(volume=volume, start_position=base_position, break_position=break_info.get("position"))
    geometry_score = float(0.45 * _clip01(compression / 0.60) + 0.25 * _clip01((len(high_points) + len(low_points) - 4) / 6) + 0.30)
    score = float(0.50 * geometry_score + 0.25 * float(volume_state.get("score", 0.5)) + 0.25 * _status_score(status))
    return {
        "pattern": pattern,
        "status": status,
        "direction": direction,
        "timeframe": timeframe,
        "score": _finite_or_none(score),
        "window_bars": int(window),
        "latest_close": _finite_or_none(latest_close),
        "breakout_date": break_info.get("date"),
        "breakout_close": break_info.get("close"),
        "breakout_direction": break_info.get("direction"),
        "breakout_margin_pct": break_info.get("margin_pct"),
        "measured_objective": _finite_or_none(objective),
        "measured_move_pct": _finite_or_none(objective / latest_close - 1.0) if objective and latest_close else None,
        "boundaries": {
            "start_date": _date_string(close.index[base_position]),
            "latest_date": _date_string(close.index[-1]),
            "upper_start": _finite_or_none(upper_base),
            "upper_latest": _finite_or_none(upper_latest),
            "upper_slope_pct_per_bar": _finite_or_none(upper_slope_pct),
            "lower_start": _finite_or_none(lower_base),
            "lower_latest": _finite_or_none(lower_latest),
            "lower_slope_pct_per_bar": _finite_or_none(lower_slope_pct),
            "base_width": _finite_or_none(base_width),
            "latest_width": _finite_or_none(latest_width),
            "width_compression_pct": _finite_or_none(compression),
            "upper_touch_count": int(len(high_points)),
            "lower_touch_count": int(len(low_points)),
        },
        "volume_confirmation": volume_state,
        "retest": retest,
        "reliability_notes": _wedge_notes(pattern, status, volume_state),
    }


def _find_diamond(frame: pd.DataFrame, timeframe: str, target_column: str) -> dict[str, Any]:
    close = pd.to_numeric(frame[target_column], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce") if "high" in frame.columns else close
    low = pd.to_numeric(frame["low"], errors="coerce") if "low" in frame.columns else close
    volume = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame.columns else None
    pivot_high, pivot_low = _confirmed_pivots(high=high, low=low)
    lookback = min(len(frame), DIAMOND_LOOKBACK.get(timeframe, DIAMOND_LOOKBACK["chart"]))
    scope = close.tail(lookback)
    positions = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    high_points = pivot_high.reindex(scope.index).dropna()
    low_points = pivot_low.reindex(scope.index).dropna()
    if len(high_points) < 3 or len(low_points) < 3:
        return _empty_pattern("Diamond", timeframe, "NoPattern", "not enough pivots for diamond analysis", len(frame))
    midpoint = int(positions.reindex(scope.index).median())
    early_high = high_points[positions.reindex(high_points.index) <= midpoint]
    early_low = low_points[positions.reindex(low_points.index) <= midpoint]
    late_high = high_points[positions.reindex(high_points.index) > midpoint]
    late_low = low_points[positions.reindex(low_points.index) > midpoint]
    if len(early_high) < 2 or len(early_low) < 2 or len(late_high) < 2 or len(late_low) < 2:
        return _empty_pattern("Diamond", timeframe, "NoPattern", "diamond requires broadening then converging pivots", len(frame))
    early_expands = float(early_high.iloc[-1]) > float(early_high.iloc[0]) and float(early_low.iloc[-1]) < float(early_low.iloc[0])
    late_contracts = float(late_high.iloc[-1]) < float(late_high.iloc[0]) and float(late_low.iloc[-1]) > float(late_low.iloc[0])
    if not early_expands or not late_contracts:
        return _empty_pattern("Diamond", timeframe, "NoPattern", "no broadening-to-converging diamond geometry", len(frame))
    upper = _fit_boundary(positions.reindex(late_high.index), late_high)
    lower = _fit_boundary(positions.reindex(late_low.index), late_low)
    if upper is None or lower is None:
        return _empty_pattern("Diamond", timeframe, "NoPattern", "late diamond boundaries unavailable", len(frame))
    start_position = int(min(positions.reindex(early_high.index).min(), positions.reindex(early_low.index).min()))
    latest_position = len(close) - 1
    upper_start = _line_value(upper, start_position)
    lower_start = _line_value(lower, start_position)
    upper_latest = _line_value(upper, latest_position)
    lower_latest = _line_value(lower, latest_position)
    break_info = _line_breakout(close, upper, lower, int(max(upper["last_position"], lower["last_position"])))
    status = "Candidate"
    direction = "undetermined"
    if break_info.get("direction") == "up":
        status = "Breakout"
        direction = "bullish"
    elif break_info.get("direction") == "down":
        status = "Breakdown"
        direction = "bearish"
    retest = _line_retest(close, upper, lower, break_info.get("position"), break_info.get("direction"))
    if status in {"Breakout", "Breakdown"} and retest.get("latest_retest"):
        status = "Retest"
    latest_close = float(close.iloc[-1])
    width = float(max(high_points.max() - low_points.min(), 0.0))
    breakout_close = break_info.get("close") or latest_close
    objective = None
    if status in {"Breakout", "Retest"} and direction == "bullish":
        objective = float(breakout_close + width)
    elif status in {"Breakdown", "Retest"} and direction == "bearish":
        objective = float(breakout_close - width)
        if objective <= 0:
            objective = None
    volume_state = _volume_state(volume=volume, start_position=start_position, break_position=break_info.get("position"))
    score = float(0.45 + 0.25 * float(volume_state.get("score", 0.5)) + 0.30 * _status_score(status))
    return {
        "pattern": "Diamond",
        "status": status,
        "direction": direction,
        "timeframe": timeframe,
        "score": _finite_or_none(score),
        "latest_close": _finite_or_none(latest_close),
        "breakout_date": break_info.get("date"),
        "breakout_close": break_info.get("close"),
        "breakout_direction": break_info.get("direction"),
        "breakout_margin_pct": break_info.get("margin_pct"),
        "measured_objective": _finite_or_none(objective),
        "measured_move_pct": _finite_or_none(objective / latest_close - 1.0) if objective and latest_close else None,
        "boundaries": {
            "start_date": _date_string(close.index[start_position]),
            "latest_date": _date_string(close.index[-1]),
            "upper_start": _finite_or_none(upper_start),
            "upper_latest": _finite_or_none(upper_latest),
            "lower_start": _finite_or_none(lower_start),
            "lower_latest": _finite_or_none(lower_latest),
            "maximum_width": _finite_or_none(width),
        },
        "volume_confirmation": volume_state,
        "retest": retest,
        "reliability_notes": _diamond_notes(status, volume_state),
    }


def _short_term_events(frame: pd.DataFrame, timeframe: str, target_column: str) -> dict[str, Any]:
    close = pd.to_numeric(frame[target_column], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce") if "high" in frame.columns else close
    low = pd.to_numeric(frame["low"], errors="coerce") if "low" in frame.columns else close
    open_price = pd.to_numeric(frame["open"], errors="coerce") if "open" in frame.columns else close.shift(1)
    volume = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame.columns else pd.Series(np.nan, index=frame.index)
    if len(close) < 25:
        event = _empty_pattern("NoShortTermEvent", timeframe, "InsufficientData", "not enough rows for one-day events", len(frame))
        return {"preferred": event, "events": []}
    latest = len(close) - 1
    prior_close = close.shift(1)
    prior_high_20 = high.shift(1).rolling(20).max()
    prior_low_20 = low.shift(1).rolling(20).min()
    true_range = _true_range(high, low, close)
    range_pct = _safe_divide(true_range, close)
    range_baseline = range_pct.shift(1).rolling(20).mean()
    volume_baseline = volume.shift(1).rolling(20).mean()
    close_location = _safe_divide(close - low, high - low).clip(lower=0, upper=1)
    prior_return_20 = close.pct_change(20)
    idx = close.index[latest]
    events = []
    wide = bool(range_pct.iloc[latest] > range_baseline.iloc[latest] * 1.6) if pd.notna(range_baseline.iloc[latest]) else False
    very_wide = bool(range_pct.iloc[latest] > range_baseline.iloc[latest] * 2.2) if pd.notna(range_baseline.iloc[latest]) else False
    volume_extreme = bool(volume.iloc[latest] > volume_baseline.iloc[latest] * 1.8) if pd.notna(volume_baseline.iloc[latest]) else False
    volume_multiple = float(volume.iloc[latest] / volume_baseline.iloc[latest]) if pd.notna(volume_baseline.iloc[latest]) and volume_baseline.iloc[latest] else None
    loc = float(close_location.iloc[latest]) if pd.notna(close_location.iloc[latest]) else 0.5
    prior_ret = float(prior_return_20.iloc[latest]) if pd.notna(prior_return_20.iloc[latest]) else 0.0
    new_high = bool(high.iloc[latest] >= prior_high_20.iloc[latest] * 1.0025) if pd.notna(prior_high_20.iloc[latest]) else False
    new_low = bool(low.iloc[latest] <= prior_low_20.iloc[latest] * 0.9975) if pd.notna(prior_low_20.iloc[latest]) else False
    gap_down = bool(open_price.iloc[latest] < low.shift(1).iloc[latest] * 0.99) if pd.notna(low.shift(1).iloc[latest]) else False
    gap_up = bool(open_price.iloc[latest] > high.shift(1).iloc[latest] * 1.01) if pd.notna(high.shift(1).iloc[latest]) else False

    if new_high and close.iloc[latest] < prior_close.iloc[latest]:
        events.append(_event_payload("KeyReversalTop", "bearish", idx, frame, latest, timeframe, range_pct, volume_multiple, prior_ret, 0.62 + 0.16 * wide + 0.10 * volume_extreme))
    if new_low and close.iloc[latest] > prior_close.iloc[latest]:
        events.append(_event_payload("KeyReversalBottom", "bullish", idx, frame, latest, timeframe, range_pct, volume_multiple, prior_ret, 0.62 + 0.16 * wide + 0.10 * volume_extreme))
    if prior_ret > 0.10 and new_high and wide and loc <= 0.25 and close.iloc[latest] < open_price.iloc[latest]:
        events.append(_event_payload("OneDayReversalTop", "bearish", idx, frame, latest, timeframe, range_pct, volume_multiple, prior_ret, 0.70 + 0.12 * volume_extreme))
    if prior_ret < -0.10 and new_low and wide and loc >= 0.75 and close.iloc[latest] > open_price.iloc[latest]:
        events.append(_event_payload("OneDayReversalBottom", "bullish", idx, frame, latest, timeframe, range_pct, volume_multiple, prior_ret, 0.70 + 0.12 * volume_extreme))
    if prior_ret < -0.15 and gap_down and very_wide and volume_extreme and loc >= 0.65:
        events.append(_event_payload("SellingClimax", "bullish", idx, frame, latest, timeframe, range_pct, volume_multiple, prior_ret, 0.88))
    if new_high and very_wide and loc <= 0.25:
        events.append(_event_payload("SpikeTop", "bearish", idx, frame, latest, timeframe, range_pct, volume_multiple, prior_ret, 0.68 + 0.10 * volume_extreme))
    if new_low and very_wide and loc >= 0.75:
        events.append(_event_payload("SpikeBottom", "bullish", idx, frame, latest, timeframe, range_pct, volume_multiple, prior_ret, 0.68 + 0.10 * volume_extreme))
    if wide and gap_up and loc >= 0.80:
        events.append(_event_payload("RunawayDayUp", "bullish_tactical", idx, frame, latest, timeframe, range_pct, volume_multiple, prior_ret, 0.58 + 0.10 * volume_extreme))
    if wide and gap_down and loc <= 0.20:
        events.append(_event_payload("RunawayDayDown", "bearish_tactical", idx, frame, latest, timeframe, range_pct, volume_multiple, prior_ret, 0.58 + 0.10 * volume_extreme))

    events = sorted(events, key=lambda item: float(item.get("score") or 0.0), reverse=True)
    preferred = events[0] if events else _empty_pattern("NoShortTermEvent", timeframe, "NoPattern", "no latest one-day reversal event", len(frame))
    return {"preferred": preferred, "events": events[:5]}


def _event_payload(
    pattern: str,
    direction: str,
    date: Any,
    frame: pd.DataFrame,
    position: int,
    timeframe: str,
    range_pct: pd.Series,
    volume_multiple: float | None,
    prior_return_20: float,
    score: float,
) -> dict[str, Any]:
    row = frame.iloc[position]
    high = float(row["high"]) if "high" in frame.columns else float(row["close"])
    low = float(row["low"]) if "low" in frame.columns else float(row["close"])
    close = float(row["close"])
    origin = low if direction.startswith("bullish") else high
    invalidation = origin
    return {
        "pattern": pattern,
        "status": "Observed",
        "direction": direction,
        "timeframe": timeframe,
        "score": _finite_or_none(min(score, 0.99)),
        "date": _date_string(date),
        "open": _finite_or_none(row.get("open", close)),
        "high": _finite_or_none(high),
        "low": _finite_or_none(low),
        "close": _finite_or_none(close),
        "range_pct": _finite_or_none(range_pct.iloc[position]),
        "volume_multiple_20d": _finite_or_none(volume_multiple),
        "prior_return_20d": _finite_or_none(prior_return_20),
        "invalidation_level": _finite_or_none(invalidation),
        "reliability_notes": _event_notes(pattern),
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


def _line_value(line: dict[str, Any], position: float) -> float:
    return float(line["intercept"] + line["slope"] * position)


def _horizontal_breakout(close: pd.Series, lower: float, upper: float, start_position: int) -> dict[str, Any]:
    for position in range(max(0, int(start_position)), len(close)):
        value = close.iloc[position]
        if pd.isna(value):
            continue
        if value <= lower * (1.0 - BREAKOUT_PCT):
            return _break_payload(close, position, "down", value / lower - 1.0)
        if value >= upper * (1.0 + BREAKOUT_PCT):
            return _break_payload(close, position, "up", value / upper - 1.0)
    return {"state": "Inside", "direction": None, "position": None, "date": None, "close": None, "margin_pct": None}


def _line_breakout(close: pd.Series, upper: dict[str, Any], lower: dict[str, Any], start_position: int) -> dict[str, Any]:
    for position in range(max(0, int(start_position)), len(close)):
        value = close.iloc[position]
        if pd.isna(value):
            continue
        upper_value = _line_value(upper, position)
        lower_value = _line_value(lower, position)
        if value >= upper_value * (1.0 + BREAKOUT_PCT):
            return _break_payload(close, position, "up", value / upper_value - 1.0)
        if value <= lower_value * (1.0 - BREAKOUT_PCT):
            return _break_payload(close, position, "down", value / lower_value - 1.0)
    return {"state": "Inside", "direction": None, "position": None, "date": None, "close": None, "margin_pct": None}


def _break_payload(close: pd.Series, position: int, direction: str, margin: float) -> dict[str, Any]:
    return {
        "state": "Breakout" if direction == "up" else "Breakdown",
        "direction": direction,
        "position": int(position),
        "date": _date_string(close.index[position]),
        "close": _finite_or_none(close.iloc[position]),
        "margin_pct": _finite_or_none(margin),
    }


def _horizontal_retest(close: pd.Series, break_position: int | None, boundary: float) -> dict[str, Any]:
    if break_position is None or boundary <= 0:
        return {"state": "NotApplicable", "latest_retest": False, "last_retest_date": None}
    last_date = None
    latest = False
    for position in range(int(break_position) + 1, len(close)):
        value = close.iloc[position]
        if pd.notna(value) and abs(float(value) / boundary - 1.0) <= RETEST_PCT:
            last_date = _date_string(close.index[position])
            latest = position >= len(close) - 5
    return {"state": "Observed" if last_date else "NotObserved", "latest_retest": bool(latest), "last_retest_date": last_date}


def _line_retest(close: pd.Series, upper: dict[str, Any], lower: dict[str, Any], break_position: int | None, direction: str | None) -> dict[str, Any]:
    if break_position is None or direction not in {"up", "down"}:
        return {"state": "NotApplicable", "latest_retest": False, "last_retest_date": None}
    line = upper if direction == "up" else lower
    last_date = None
    latest = False
    for position in range(int(break_position) + 1, len(close)):
        value = close.iloc[position]
        boundary = _line_value(line, position)
        if pd.notna(value) and boundary > 0 and abs(float(value) / boundary - 1.0) <= RETEST_PCT:
            last_date = _date_string(close.index[position])
            latest = position >= len(close) - 5
    return {"state": "Observed" if last_date else "NotObserved", "latest_retest": bool(latest), "last_retest_date": last_date}


def _prior_trend(close: pd.Series, first_position: int, timeframe: str, require_advance: bool) -> dict[str, Any]:
    lookback = PRIOR_TREND_LOOKBACK_BARS.get(timeframe, PRIOR_TREND_LOOKBACK_BARS["chart"])
    start = max(0, int(first_position) - lookback)
    if int(first_position) <= start:
        return {"state": "InsufficientData", "lookback_bars": int(first_position - start), "return_pct": None}
    start_value = float(close.iloc[start])
    end_value = float(close.iloc[int(first_position)])
    if start_value <= 0:
        return {"state": "InsufficientData", "lookback_bars": int(first_position - start), "return_pct": None}
    prior_return = end_value / start_value - 1.0
    threshold = PRIOR_TREND_MIN_RETURN.get(timeframe, PRIOR_TREND_MIN_RETURN["chart"])
    state = "PriorAdvance" if prior_return >= threshold else "NoPriorAdvance"
    if not require_advance:
        state = "PriorDecline" if prior_return <= -threshold else "NoPriorDecline"
    return {
        "state": state,
        "lookback_bars": int(first_position - start),
        "return_pct": _finite_or_none(prior_return),
        "threshold_pct": threshold,
    }


def _volume_state(volume: pd.Series | None, start_position: int, break_position: int | None) -> dict[str, Any]:
    if volume is None or volume.dropna().empty:
        return {"state": "Unavailable", "score": 0.5}
    scope = volume.iloc[start_position:]
    if scope.dropna().empty:
        return {"state": "Unavailable", "score": 0.5}
    median = float(scope.median()) if scope.notna().any() else 0.0
    irregularity = float(scope.std() / median) if median > 0 and len(scope.dropna()) > 2 else None
    first_half = scope.iloc[: max(1, len(scope) // 2)]
    second_half = scope.iloc[max(1, len(scope) // 2) :]
    contraction = None
    if first_half.notna().any() and second_half.notna().any() and float(first_half.median()) > 0:
        contraction = bool(float(second_half.median()) <= float(first_half.median()) * 0.9)
    breakout_expansion = None
    if break_position is not None:
        break_volume = float(volume.iloc[int(break_position)]) if pd.notna(volume.iloc[int(break_position)]) else None
        baseline = volume.shift(1).rolling(20, min_periods=5).mean().iloc[int(break_position)]
        if break_volume is not None and pd.notna(baseline) and float(baseline) > 0:
            breakout_expansion = bool(break_volume >= float(baseline) * 1.2)
    score = 0.5
    if contraction is True:
        score += 0.15
    if breakout_expansion is True:
        score += 0.20
    if irregularity is not None and irregularity > 0.45:
        score += 0.10
    return {
        "state": "Measured",
        "volume_contracts_inside_pattern": contraction,
        "breakout_volume_expansion": breakout_expansion,
        "volume_irregularity": _finite_or_none(irregularity),
        "score": _finite_or_none(min(max(score, 0.0), 1.0)),
    }


def _status_score(status: str) -> float:
    return {
        "Confirmed": 0.95,
        "PullbackToBoundary": 0.90,
        "Breakdown": 0.95,
        "Breakout": 0.95,
        "Retest": 0.90,
        "Candidate": 0.45,
        "UpsideBreakout": 0.55,
        "OppositeBreak": 0.30,
        "ObjectiveReached": 0.40,
        "Observed": 0.60,
    }.get(status, 0.20)


def _structural_rank(pattern: dict[str, Any]) -> tuple[float, float, str]:
    status_rank = {
        "PullbackToBoundary": 5.0,
        "Retest": 4.8,
        "Confirmed": 4.6,
        "Breakdown": 4.6,
        "Breakout": 4.6,
        "UpsideBreakout": 2.4,
        "Candidate": 1.5,
        "OppositeBreak": 1.0,
        "ObjectiveReached": 0.8,
        "NoPattern": 0.0,
        "InsufficientData": 0.0,
    }.get(str(pattern.get("status")), 0.0)
    pattern_bonus = {"BroadeningTop": 0.30, "RisingWedge": 0.20, "FallingWedge": 0.20, "Diamond": 0.15}.get(str(pattern.get("pattern")), 0.0)
    timeframe_bonus = {"monthly": 0.30, "weekly": 0.15, "daily": 0.0}.get(str(pattern.get("timeframe")), 0.0)
    score = float(pattern.get("score") or 0.0)
    date = pattern.get("breakout_date") or pattern.get("boundaries", {}).get("latest_date") or ""
    return (status_rank + pattern_bonus + timeframe_bonus, score, str(date))


def _event_rank(event: dict[str, Any]) -> tuple[float, float, str]:
    pattern_rank = {
        "SellingClimax": 4.5,
        "OneDayReversalTop": 4.0,
        "OneDayReversalBottom": 4.0,
        "KeyReversalTop": 3.5,
        "KeyReversalBottom": 3.5,
        "SpikeTop": 3.0,
        "SpikeBottom": 3.0,
        "RunawayDayUp": 2.0,
        "RunawayDayDown": 2.0,
    }.get(str(event.get("pattern")), 0.0)
    return (pattern_rank, float(event.get("score") or 0.0), str(event.get("date") or ""))


def _choose_preferred_structural(timeframes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    patterns = [
        payload.get("structural_preferred", {})
        for payload in timeframes.values()
        if isinstance(payload.get("structural_preferred"), dict)
    ]
    actionable = [pattern for pattern in patterns if pattern.get("status") not in {"NoPattern", "InsufficientData"}]
    if actionable:
        return _preferred_structural_payload(max(actionable, key=_structural_rank))
    for timeframe in ("monthly", "weekly", "daily"):
        pattern = timeframes.get(timeframe, {}).get("structural_preferred", {})
        if pattern:
            return _preferred_structural_payload(pattern)
    return _empty_pattern("NoChapter10StructuralPattern", "unknown", "NoPattern", "pattern unavailable", 0)


def _choose_preferred_event(timeframes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    events = []
    for payload in timeframes.values():
        preferred = payload.get("short_term_events", {}).get("preferred", {})
        if preferred.get("status") not in {"NoPattern", "InsufficientData"}:
            events.append(preferred)
    if events:
        return max(events, key=_event_rank)
    return _empty_pattern("NoShortTermEvent", "daily", "NoPattern", "no one-day event detected", 0)


def _choose_chapter_10_preferred(structural: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    candidates = [structural, event]
    candidates = [candidate for candidate in candidates if candidate.get("status") not in {"NoPattern", "InsufficientData"}]
    if not candidates:
        return {"pattern": "NoChapter10Pattern", "status": "NoPattern", "reason": "no Chapter 10 pattern detected"}
    return max(candidates, key=lambda item: max(_structural_rank(item), _event_rank(item)))


def _preferred_structural_payload(pattern: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "pattern",
        "status",
        "direction",
        "timeframe",
        "score",
        "window_bars",
        "latest_close",
        "confirmation_level",
        "breakout_date",
        "breakout_close",
        "breakout_direction",
        "breakout_margin_pct",
        "measured_objective",
        "measured_move_pct",
        "prior_trend",
        "points",
        "boundaries",
        "volume_confirmation",
        "retest",
        "reliability_notes",
        "reason",
    ]
    return {key: pattern[key] for key in keys if key in pattern}


def _empty_timeframe(timeframe: str, status: str, reason: str, rows: int) -> dict[str, Any]:
    empty_structural = _empty_pattern("NoChapter10StructuralPattern", timeframe, status, reason, rows)
    empty_event = _empty_pattern("NoShortTermEvent", timeframe, status, reason, rows)
    return {
        "state": status,
        "timeframe": timeframe,
        "rows": int(rows),
        "broadening": _empty_pattern("NoBroadeningPattern", timeframe, status, reason, rows),
        "wedge": _empty_pattern("NoWedge", timeframe, status, reason, rows),
        "diamond": _empty_pattern("Diamond", timeframe, status, reason, rows),
        "structural_preferred": empty_structural,
        "short_term_events": {"preferred": empty_event, "events": []},
    }


def _empty_pattern(pattern: str, timeframe: str, status: str, reason: str, rows: int) -> dict[str, Any]:
    return {
        "pattern": pattern,
        "status": status,
        "timeframe": timeframe,
        "rows": int(rows),
        "reason": reason,
    }


def _broadening_notes(status: str, volume_state: dict[str, Any]) -> list[str]:
    notes = []
    if status == "Candidate":
        notes.append("Broadening formation is unconfirmed but implies disorder after an advance.")
    if status == "UpsideBreakout":
        notes.append("Upside broadening break is possible but less common; require follow-through.")
    if volume_state.get("volume_irregularity") is not None and float(volume_state["volume_irregularity"]) > 0.45:
        notes.append("Volume is irregular, consistent with broadening-formation instability.")
    return notes


def _wedge_notes(pattern: str, status: str, volume_state: dict[str, Any]) -> list[str]:
    notes = []
    if status == "Candidate":
        notes.append(f"{pattern} is forming but remains unconfirmed until boundary breakout.")
    if pattern == "RisingWedge" and status in {"Breakdown", "Retest"}:
        notes.append("Rising Wedge broke down; Chapter 10 treats this as weakening demand.")
    if pattern == "FallingWedge" and status in {"Breakout", "Retest"}:
        notes.append("Falling Wedge broke upward; recovery can be slower than a rising-wedge decline.")
    if volume_state.get("volume_contracts_inside_pattern") is False:
        notes.append("Volume did not contract clearly inside the wedge.")
    return notes


def _diamond_notes(status: str, volume_state: dict[str, Any]) -> list[str]:
    notes = []
    if status == "Candidate":
        notes.append("Diamond is warning-only until the converging half breaks decisively.")
    if volume_state.get("volume_contracts_inside_pattern") is False:
        notes.append("Diamond's converging half lacks clear volume contraction.")
    return notes


def _event_notes(pattern: str) -> list[str]:
    if pattern == "SellingClimax":
        return ["Selling climax can block panic selling, but it is not a dependable major-bottom signal."]
    if pattern.startswith("RunawayDay"):
        return ["Runaway day needs follow-through; return to the day's origin invalidates the event."]
    if pattern.startswith("Spike"):
        return ["Spike requires subsequent bars for confirmation; treat as short-term warning."]
    return ["One-day events are tactical warnings and should not override longer-horizon validation by themselves."]


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


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prior_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prior_close).abs(),
            (low - prior_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0.0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


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
