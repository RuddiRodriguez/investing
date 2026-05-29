from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PIVOT_LEFT_BARS = 3
PIVOT_RIGHT_BARS = 3
MIN_PATTERN_BARS = 45
RECTANGLE_WINDOWS = {
    "daily": (45, 63, 90, 126),
    "weekly": (26, 39, 52, 78),
    "monthly": (18, 24, 36, 48),
    "chart": (35, 50, 70, 90),
}
MULTI_TOP_BOTTOM_LOOKBACK = {"daily": 378, "weekly": 156, "monthly": 96, "chart": 220}
MIN_RECTANGLE_TOUCHES = 2
MIN_RECTANGLE_WIDTH_PCT = 0.035
MAX_RECTANGLE_WIDTH_PCT = 0.35
RECTANGLE_LEVEL_TOLERANCE_PCT = 0.025
BREAKOUT_PCT = 0.03
RETEST_PCT = 0.03
EQUAL_LEVEL_TOLERANCE_PCT = 0.03
MIN_REACTION_DEPTH_PCT = 0.05
MIN_DOUBLE_SEPARATION = {"daily": 35, "weekly": 10, "monthly": 5, "chart": 25}
MIN_TRIPLE_SEPARATION = {"daily": 20, "weekly": 7, "monthly": 4, "chart": 15}
PRIOR_TREND_LOOKBACK_BARS = {"daily": 126, "weekly": 52, "monthly": 36, "chart": 90}
PRIOR_TREND_MIN_RETURN = {"daily": 0.08, "weekly": 0.10, "monthly": 0.12, "chart": 0.06}


def analyze_chapter_9_patterns(prices: pd.DataFrame, target_column: str = "close") -> dict[str, Any]:
    """Analyze Edwards/Magee Chapter 9 rectangles and double/triple formations."""

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
    rectangles = {
        "principle": (
            "Edwards/Magee Chapter 9: rectangles mark horizontal supply and demand; "
            "direction is normally governed by a confirmed break, with volume contraction and pullbacks tracked."
        ),
        "preferred": _choose_preferred(timeframes, "rectangle"),
        "timeframes": {
            name: {"state": payload.get("state"), "rows": payload.get("rows"), "preferred": payload.get("rectangle", {})}
            for name, payload in timeframes.items()
        },
    }
    multi_top_bottom = {
        "principle": (
            "Edwards/Magee Chapter 9: double/triple tops and bottoms are rare and remain warning-only "
            "until the intervening valley or peak is decisively broken."
        ),
        "preferred": _choose_preferred(timeframes, "multi_top_bottom"),
        "timeframes": {
            name: {
                "state": payload.get("state"),
                "rows": payload.get("rows"),
                "double_top": payload.get("double_top", {}),
                "double_bottom": payload.get("double_bottom", {}),
                "triple_top": payload.get("triple_top", {}),
                "triple_bottom": payload.get("triple_bottom", {}),
                "preferred": payload.get("multi_top_bottom", {}),
            }
            for name, payload in timeframes.items()
        },
    }
    return {
        "principle": (
            "Chapter 9 adds horizontal range governance: rectangles can delay directional action until a break, "
            "while double/triple formations require confirmation before they can block a model signal."
        ),
        "primary_timeframe": "weekly",
        "secondary_timeframe": "daily",
        "higher_order_timeframe": "monthly",
        "confirmation_rule": f"Close at least {BREAKOUT_PCT:.0%} beyond the rectangle boundary or intervening confirmation level.",
        "preferred": _choose_chapter_9_preferred(rectangles["preferred"], multi_top_bottom["preferred"]),
        "rectangle_patterns": rectangles,
        "multi_top_bottom_patterns": multi_top_bottom,
        "timeframes": timeframes,
        "technical_method_card": chapter_9_patterns_method_card(target_column=target),
    }


def latest_chapter_9_patterns(
    prices: pd.DataFrame,
    target_column: str = "close",
    timeframe: str = "chart",
) -> dict[str, Any]:
    """Return Chapter 9 diagnostics for a plotted bar set."""

    return _analyze_timeframe(
        frame=prices,
        timeframe=timeframe,
        target_column=target_column.lower(),
    )


def chapter_9_patterns_method_card(target_column: str = "close") -> dict[str, Any]:
    return {
        "name": "edwards_magee_chapter_9_patterns",
        "version": "chapter_9_alignment_v1",
        "target_column": target_column.lower(),
        "implemented_patterns": [
            "rectangle",
            "double_top",
            "double_bottom",
            "triple_top",
            "triple_bottom",
        ],
        "rectangle": {
            "geometry": "near-horizontal resistance and support from confirmed pivot highs/lows",
            "minimum_touches": {
                "upper_boundary": MIN_RECTANGLE_TOUCHES,
                "lower_boundary": MIN_RECTANGLE_TOUCHES,
            },
            "width_filter": f"{MIN_RECTANGLE_WIDTH_PCT:.1%} to {MAX_RECTANGLE_WIDTH_PCT:.0%} of price",
            "confirmation": f"close at least {BREAKOUT_PCT:.0%} outside the boundary",
            "volume": "volume should generally contract through the range; breakout expansion improves reliability",
            "objective": "project rectangle height from the confirmed breakout or breakdown boundary",
            "decision_use": "unbroken rectangles can hold directional actions; confirmed breaks can support or block model direction",
        },
        "double_triple_tops_bottoms": {
            "rarity_control": "patterns are suspected until confirmation; suspected patterns warn but do not block",
            "equal_level_tolerance": f"{EQUAL_LEVEL_TOLERANCE_PCT:.0%}",
            "minimum_reaction_depth": f"{MIN_REACTION_DEPTH_PCT:.0%}",
            "confirmation": "double/triple tops confirm below the intervening valley; bottoms confirm above the intervening peak",
            "volume": "tops prefer declining volume across peaks; bottoms prefer accumulation and breakout expansion",
            "objective": "project the pattern height from the confirmation level",
            "decision_use": "confirmed tops can block fresh Buy actions; confirmed bottoms can block fresh Sell actions",
        },
        "timeframes": {
            "primary": "weekly",
            "secondary": "daily",
            "monthly": "useful for broad major reversals",
        },
    }


def _analyze_timeframe(frame: pd.DataFrame, timeframe: str, target_column: str) -> dict[str, Any]:
    clean = frame.copy().sort_index()
    empty = _empty_timeframe(timeframe, "InsufficientData", "missing target column", len(clean))
    if clean.empty or target_column not in clean.columns:
        return empty
    clean.index = pd.DatetimeIndex(clean.index)
    clean = clean.loc[pd.to_numeric(clean[target_column], errors="coerce").notna()]
    if len(clean) < MIN_PATTERN_BARS:
        return _empty_timeframe(timeframe, "InsufficientData", "not enough bars for Chapter 9 analysis", len(clean))

    rectangle = _find_rectangle(clean, timeframe=timeframe, target_column=target_column)
    double_top = _find_multi_top_bottom(clean, timeframe=timeframe, target_column=target_column, kind="double_top")
    double_bottom = _find_multi_top_bottom(clean, timeframe=timeframe, target_column=target_column, kind="double_bottom")
    triple_top = _find_multi_top_bottom(clean, timeframe=timeframe, target_column=target_column, kind="triple_top")
    triple_bottom = _find_multi_top_bottom(clean, timeframe=timeframe, target_column=target_column, kind="triple_bottom")
    multi_candidates = [double_top, double_bottom, triple_top, triple_bottom]
    multi_preferred = max(multi_candidates, key=_multi_pattern_rank)
    return {
        "state": "Measured",
        "timeframe": timeframe,
        "rows": int(len(clean)),
        "start_date": _date_string(clean.index[0]),
        "end_date": _date_string(clean.index[-1]),
        "rectangle": rectangle,
        "double_top": double_top,
        "double_bottom": double_bottom,
        "triple_top": triple_top,
        "triple_bottom": triple_bottom,
        "multi_top_bottom": _preferred_multi_payload(multi_preferred),
    }


def _find_rectangle(frame: pd.DataFrame, timeframe: str, target_column: str) -> dict[str, Any]:
    close = pd.to_numeric(frame[target_column], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce") if "high" in frame.columns else close
    low = pd.to_numeric(frame["low"], errors="coerce") if "low" in frame.columns else close
    volume = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame.columns else None
    pivot_high, pivot_low = _confirmed_pivots(high=high, low=low)
    candidates = []
    for window in RECTANGLE_WINDOWS.get(timeframe, RECTANGLE_WINDOWS["chart"]):
        if len(frame) < window:
            continue
        candidate = _rectangle_candidate(
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
        return _empty_pattern("Rectangle", timeframe, "NoPattern", "no valid horizontal supply/demand range", len(frame))
    return max(candidates, key=_rectangle_rank)


def _rectangle_candidate(
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
    positions = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    high_points = pivot_high.reindex(scope.index).dropna()
    low_points = pivot_low.reindex(scope.index).dropna()
    if len(high_points) < MIN_RECTANGLE_TOUCHES or len(low_points) < MIN_RECTANGLE_TOUCHES:
        return None

    resistance = float(high_points.median())
    support = float(low_points.median())
    latest_close = float(close.iloc[-1])
    if not all(np.isfinite(value) and value > 0 for value in (resistance, support, latest_close)):
        return None
    if resistance <= support:
        return None

    midline = (resistance + support) / 2.0
    width = resistance - support
    width_pct = width / midline if midline else 0.0
    if width_pct < MIN_RECTANGLE_WIDTH_PCT or width_pct > MAX_RECTANGLE_WIDTH_PCT:
        return None

    upper_touches = high_points[(high_points / resistance - 1.0).abs() <= RECTANGLE_LEVEL_TOLERANCE_PCT]
    lower_touches = low_points[(low_points / support - 1.0).abs() <= RECTANGLE_LEVEL_TOLERANCE_PCT]
    if len(upper_touches) < MIN_RECTANGLE_TOUCHES or len(lower_touches) < MIN_RECTANGLE_TOUCHES:
        return None

    start_position = int(min(positions.reindex(upper_touches.index).min(), positions.reindex(lower_touches.index).min()))
    last_touch_position = int(max(positions.reindex(upper_touches.index).max(), positions.reindex(lower_touches.index).max()))
    if len(close) - start_position < MIN_PATTERN_BARS // 2:
        return None

    range_prices = close.iloc[start_position : last_touch_position + 1]
    if range_prices.empty:
        return None
    outside_share = float(((range_prices > resistance * (1.0 + BREAKOUT_PCT)) | (range_prices < support * (1.0 - BREAKOUT_PCT))).mean())
    if outside_share > 0.18:
        return None

    break_info = _rectangle_breakout_state(close=close, resistance=resistance, support=support, start_position=last_touch_position)
    volume_state = _volume_state(volume=volume, start_position=start_position, break_position=break_info.get("position"))
    retest = _rectangle_retest_state(
        close=close,
        resistance=resistance,
        support=support,
        break_position=break_info.get("position"),
        direction=break_info.get("direction"),
    )
    status = _rectangle_status(break_info=break_info, retest=retest, latest_close=latest_close, resistance=resistance, support=support, width=width)
    direction = _rectangle_direction(status)
    objective = _rectangle_objective(status=status, break_info=break_info, resistance=resistance, support=support, width=width)

    touch_score = _clip01((min(len(upper_touches), 5) + min(len(lower_touches), 5) - 4) / 6)
    level_score = 1.0 - _clip01(
        (
            float((upper_touches / resistance - 1.0).abs().mean())
            + float((lower_touches / support - 1.0).abs().mean())
        )
        / (2 * RECTANGLE_LEVEL_TOLERANCE_PCT)
    )
    width_score = 1.0 if 0.07 <= width_pct <= 0.18 else 0.70
    volume_score = float(volume_state.get("score", 0.5))
    status_score = _rectangle_status_score(status)
    reliability = float(0.35 * touch_score + 0.25 * level_score + 0.15 * width_score + 0.15 * volume_score + 0.10 * status_score)

    return {
        "pattern": "Rectangle",
        "status": status,
        "direction": direction,
        "timeframe": timeframe,
        "score": _finite_or_none(reliability),
        "window_bars": int(window),
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
            "resistance": _finite_or_none(resistance),
            "support": _finite_or_none(support),
            "midline": _finite_or_none(midline),
            "height": _finite_or_none(width),
            "width_pct": _finite_or_none(width_pct),
            "upper_touch_count": int(len(upper_touches)),
            "lower_touch_count": int(len(lower_touches)),
            "last_touch_date": _date_string(close.index[last_touch_position]),
        },
        "volume_confirmation": volume_state,
        "retest": retest,
        "reliability_notes": _rectangle_notes(status=status, volume_state=volume_state, width_pct=width_pct),
    }


def _find_multi_top_bottom(frame: pd.DataFrame, timeframe: str, target_column: str, kind: str) -> dict[str, Any]:
    close = pd.to_numeric(frame[target_column], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce") if "high" in frame.columns else close
    low = pd.to_numeric(frame["low"], errors="coerce") if "low" in frame.columns else close
    volume = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame.columns else None
    pivot_high, pivot_low = _confirmed_pivots(high=high, low=low)
    lookback = min(len(frame), MULTI_TOP_BOTTOM_LOOKBACK.get(timeframe, MULTI_TOP_BOTTOM_LOOKBACK["chart"]))
    scope_index = close.tail(lookback).index
    positions = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    is_top = kind.endswith("top")
    count = 3 if kind.startswith("triple") else 2
    pivots = pivot_high.reindex(scope_index).dropna() if is_top else pivot_low.reindex(scope_index).dropna()
    if len(pivots) < count:
        return _empty_pattern(_pattern_name(kind), timeframe, "NoPattern", "not enough confirmed pivots", len(frame))

    candidates: list[dict[str, Any]] = []
    pivot_items = list(pivots.items())
    for combo in _recent_combinations(pivot_items, count=count, limit=70):
        candidate = _multi_candidate(
            close=close,
            high=high,
            low=low,
            volume=volume,
            positions=positions,
            combo=combo,
            kind=kind,
            timeframe=timeframe,
        )
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return _empty_pattern(_pattern_name(kind), timeframe, "NoPattern", "no confirmed multi-top/bottom geometry", len(frame))
    return max(candidates, key=_multi_pattern_rank)


def _multi_candidate(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series | None,
    positions: pd.Series,
    combo: list[tuple[Any, float]],
    kind: str,
    timeframe: str,
) -> dict[str, Any] | None:
    is_top = kind.endswith("top")
    count = len(combo)
    dates = [item[0] for item in combo]
    values = [float(item[1]) for item in combo]
    pivot_positions = [int(positions.loc[date]) for date in dates]
    if pivot_positions != sorted(pivot_positions):
        return None
    min_separation = (MIN_TRIPLE_SEPARATION if count == 3 else MIN_DOUBLE_SEPARATION).get(timeframe, 20)
    if min(np.diff(pivot_positions)) < min_separation:
        return None

    reference_level = float(np.median(values))
    if reference_level <= 0:
        return None
    equal_error = float(max(abs(value / reference_level - 1.0) for value in values))
    if equal_error > EQUAL_LEVEL_TOLERANCE_PCT:
        return None

    confirmation_level, reaction_depth = _confirmation_level_and_depth(
        high=high,
        low=low,
        pivot_positions=pivot_positions,
        reference_level=reference_level,
        is_top=is_top,
    )
    if confirmation_level is None or reaction_depth < MIN_REACTION_DEPTH_PCT:
        return None

    prior_trend = _prior_trend(close=close, first_position=pivot_positions[0], timeframe=timeframe, is_top=is_top)
    if prior_trend["state"] in {"NoPriorAdvance", "NoPriorDecline"}:
        return None

    break_info = _multi_breakout_state(
        close=close,
        start_position=pivot_positions[-1],
        confirmation_level=confirmation_level,
        is_top=is_top,
    )
    status = _multi_status(
        close=close,
        break_info=break_info,
        confirmation_level=confirmation_level,
        reference_level=reference_level,
        is_top=is_top,
    )
    direction = "bearish" if is_top and status in {"Confirmed", "PullbackToConfirmation", "ObjectiveReached"} else (
        "bullish" if (not is_top and status in {"Confirmed", "PullbackToConfirmation", "ObjectiveReached"}) else "warning_only"
    )
    objective = None
    if status in {"Confirmed", "PullbackToConfirmation", "ObjectiveReached"}:
        height = abs(reference_level - confirmation_level)
        objective = confirmation_level - height if is_top else confirmation_level + height
        if objective <= 0:
            objective = None

    volume_state = _multi_volume_state(volume=volume, pivot_positions=pivot_positions, break_position=break_info.get("position"), is_top=is_top)
    geometry_score = float(
        0.40 * (1.0 - _clip01(equal_error / EQUAL_LEVEL_TOLERANCE_PCT))
        + 0.30 * _clip01(reaction_depth / 0.18)
        + 0.30 * _clip01((min(np.diff(pivot_positions)) / max(min_separation, 1)))
    )
    status_score = _multi_status_score(status)
    reliability = float(0.45 * geometry_score + 0.25 * float(volume_state.get("score", 0.5)) + 0.30 * status_score)
    latest_close = float(close.iloc[-1])
    points = {
        f"{'peak' if is_top else 'low'}_{idx + 1}": {
            "date": _date_string(date),
            "price": _finite_or_none(value),
        }
        for idx, (date, value) in enumerate(combo)
    }
    return {
        "pattern": _pattern_name(kind),
        "status": status,
        "direction": direction,
        "timeframe": timeframe,
        "score": _finite_or_none(reliability),
        "latest_close": _finite_or_none(latest_close),
        "level": _finite_or_none(reference_level),
        "confirmation_level": _finite_or_none(confirmation_level),
        "confirmation_date": break_info.get("date"),
        "confirmation_close": break_info.get("close"),
        "confirmation_margin_pct": break_info.get("margin_pct"),
        "measured_objective": _finite_or_none(objective),
        "measured_move_pct": _finite_or_none(objective / latest_close - 1.0) if objective and latest_close else None,
        "equal_level_error_pct": _finite_or_none(equal_error),
        "reaction_depth_pct": _finite_or_none(reaction_depth),
        "prior_trend": prior_trend,
        "points": points,
        "volume_confirmation": volume_state,
        "reliability_notes": _multi_notes(status=status, volume_state=volume_state, is_top=is_top),
    }


def _confirmed_pivots(high: pd.Series, low: pd.Series) -> tuple[pd.Series, pd.Series]:
    window = PIVOT_LEFT_BARS + PIVOT_RIGHT_BARS + 1
    raw_high = high == high.rolling(window=window, center=True).max()
    raw_low = low == low.rolling(window=window, center=True).min()
    confirmed_high = high.shift(PIVOT_RIGHT_BARS).where(raw_high.shift(PIVOT_RIGHT_BARS, fill_value=False), np.nan)
    confirmed_low = low.shift(PIVOT_RIGHT_BARS).where(raw_low.shift(PIVOT_RIGHT_BARS, fill_value=False), np.nan)
    return confirmed_high, confirmed_low


def _rectangle_breakout_state(close: pd.Series, resistance: float, support: float, start_position: int) -> dict[str, Any]:
    output = {"state": "Inside", "direction": None, "position": None, "date": None, "close": None, "margin_pct": None}
    start = max(0, min(int(start_position), len(close) - 1))
    for position in range(start, len(close)):
        value = close.iloc[position]
        if pd.isna(value):
            continue
        if value >= resistance * (1.0 + BREAKOUT_PCT):
            return _break_payload(close, position, "up", value / resistance - 1.0)
        if value <= support * (1.0 - BREAKOUT_PCT):
            return _break_payload(close, position, "down", value / support - 1.0)
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


def _rectangle_retest_state(
    close: pd.Series,
    resistance: float,
    support: float,
    break_position: int | None,
    direction: str | None,
) -> dict[str, Any]:
    if break_position is None or direction not in {"up", "down"}:
        return {"state": "NotApplicable", "latest_retest": False, "last_retest_date": None}
    boundary = resistance if direction == "up" else support
    last_date = None
    latest = False
    for position in range(int(break_position) + 1, len(close)):
        value = close.iloc[position]
        if pd.isna(value) or boundary <= 0:
            continue
        if abs(float(value) / boundary - 1.0) <= RETEST_PCT:
            last_date = _date_string(close.index[position])
            latest = position >= len(close) - 5
    return {"state": "Observed" if last_date else "NotObserved", "latest_retest": bool(latest), "last_retest_date": last_date}


def _rectangle_status(
    break_info: dict[str, Any],
    retest: dict[str, Any],
    latest_close: float,
    resistance: float,
    support: float,
    width: float,
) -> str:
    direction = break_info.get("direction")
    if direction not in {"up", "down"}:
        return "Candidate"
    if direction == "up":
        if latest_close <= support * (1.0 - BREAKOUT_PCT):
            return "FalseBreakout"
        if support < latest_close < resistance:
            return "PrematureBreakout"
        if latest_close >= resistance + width:
            return "ObjectiveReached"
        if retest.get("latest_retest"):
            return "Retest"
        return "Breakout"
    if latest_close >= resistance * (1.0 + BREAKOUT_PCT):
        return "FalseBreakdown"
    if support < latest_close < resistance:
        return "PrematureBreakdown"
    if latest_close <= support - width:
        return "ObjectiveReached"
    if retest.get("latest_retest"):
        return "Retest"
    return "Breakdown"


def _rectangle_direction(status: str) -> str:
    if status in {"Breakout", "Retest", "ObjectiveReached", "FalseBreakdown"}:
        return "bullish"
    if status in {"Breakdown", "FalseBreakout"}:
        return "bearish"
    if status == "PrematureBreakout":
        return "failed_bullish"
    if status == "PrematureBreakdown":
        return "failed_bearish"
    return "undetermined"


def _rectangle_objective(status: str, break_info: dict[str, Any], resistance: float, support: float, width: float) -> float | None:
    direction = break_info.get("direction")
    if direction == "up" and status not in {"FalseBreakout", "PrematureBreakout"}:
        return float(resistance + width)
    if direction == "down" and status not in {"FalseBreakdown", "PrematureBreakdown"}:
        objective = float(support - width)
        return objective if objective > 0 else None
    return None


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
    elif contraction is False:
        score -= 0.1
    if breakout_expansion is True:
        score += 0.25
    elif breakout_expansion is False:
        score -= 0.1
    return {
        "state": "Measured",
        "volume_contracts_inside_pattern": contraction,
        "breakout_volume_expansion": breakout_expansion,
        "break_volume": _finite_or_none(break_volume),
        "score": _finite_or_none(min(max(score, 0.0), 1.0)),
    }


def _recent_combinations(items: list[tuple[Any, float]], count: int, limit: int) -> list[list[tuple[Any, float]]]:
    items = items[-limit:]
    if count == 2:
        return [[items[i], items[j]] for i in range(len(items)) for j in range(i + 1, len(items))]
    return [
        [items[i], items[j], items[k]]
        for i in range(len(items))
        for j in range(i + 1, len(items))
        for k in range(j + 1, len(items))
    ]


def _confirmation_level_and_depth(
    high: pd.Series,
    low: pd.Series,
    pivot_positions: list[int],
    reference_level: float,
    is_top: bool,
) -> tuple[float | None, float]:
    levels = []
    for left, right in zip(pivot_positions, pivot_positions[1:]):
        segment = low.iloc[left:right + 1] if is_top else high.iloc[left:right + 1]
        if segment.dropna().empty:
            return None, 0.0
        levels.append(float(segment.min() if is_top else segment.max()))
    confirmation_level = min(levels) if is_top else max(levels)
    if is_top:
        depth = (reference_level - confirmation_level) / reference_level if reference_level else 0.0
    else:
        depth = (confirmation_level - reference_level) / reference_level if reference_level else 0.0
    return confirmation_level, float(depth)


def _prior_trend(close: pd.Series, first_position: int, timeframe: str, is_top: bool) -> dict[str, Any]:
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
    if is_top:
        state = "PriorAdvance" if prior_return >= threshold else "NoPriorAdvance"
    else:
        state = "PriorDecline" if prior_return <= -threshold else "NoPriorDecline"
    return {
        "state": state,
        "lookback_bars": int(first_position - start),
        "return_pct": _finite_or_none(prior_return),
        "threshold_pct": threshold,
    }


def _multi_breakout_state(close: pd.Series, start_position: int, confirmation_level: float, is_top: bool) -> dict[str, Any]:
    for position in range(max(0, int(start_position)), len(close)):
        value = close.iloc[position]
        if pd.isna(value) or confirmation_level <= 0:
            continue
        if is_top and value <= confirmation_level * (1.0 - BREAKOUT_PCT):
            return _multi_break_payload(close, position, value / confirmation_level - 1.0)
        if not is_top and value >= confirmation_level * (1.0 + BREAKOUT_PCT):
            return _multi_break_payload(close, position, value / confirmation_level - 1.0)
    return {"state": "Unconfirmed", "position": None, "date": None, "close": None, "margin_pct": None}


def _multi_break_payload(close: pd.Series, position: int, margin: float) -> dict[str, Any]:
    return {
        "state": "Confirmed",
        "position": int(position),
        "date": _date_string(close.index[position]),
        "close": _finite_or_none(close.iloc[position]),
        "margin_pct": _finite_or_none(margin),
    }


def _multi_status(
    close: pd.Series,
    break_info: dict[str, Any],
    confirmation_level: float,
    reference_level: float,
    is_top: bool,
) -> str:
    if break_info.get("state") != "Confirmed":
        return "Suspected"
    latest = float(close.iloc[-1])
    height = abs(reference_level - confirmation_level)
    if is_top and latest <= confirmation_level - height:
        return "ObjectiveReached"
    if not is_top and latest >= confirmation_level + height:
        return "ObjectiveReached"
    if abs(latest / confirmation_level - 1.0) <= RETEST_PCT:
        return "PullbackToConfirmation"
    return "Confirmed"


def _multi_volume_state(
    volume: pd.Series | None,
    pivot_positions: list[int],
    break_position: int | None,
    is_top: bool,
) -> dict[str, Any]:
    if volume is None or volume.dropna().empty:
        return {"state": "Unavailable", "score": 0.5}
    pivot_volumes = [float(volume.iloc[position]) for position in pivot_positions if pd.notna(volume.iloc[position])]
    if not pivot_volumes:
        return {"state": "Unavailable", "score": 0.5}
    sequence_declines = None
    if len(pivot_volumes) >= 2:
        sequence_declines = all(right <= left for left, right in zip(pivot_volumes, pivot_volumes[1:]))
    breakout_expansion = None
    if break_position is not None:
        break_volume = float(volume.iloc[int(break_position)]) if pd.notna(volume.iloc[int(break_position)]) else None
        baseline = volume.shift(1).rolling(20, min_periods=5).mean().iloc[int(break_position)]
        if break_volume is not None and pd.notna(baseline) and float(baseline) > 0:
            breakout_expansion = bool(break_volume >= float(baseline) * 1.2)
    score = 0.5
    if is_top and sequence_declines is True:
        score += 0.20
    elif is_top and sequence_declines is False:
        score -= 0.10
    if not is_top and breakout_expansion is True:
        score += 0.25
    elif not is_top and breakout_expansion is False:
        score -= 0.10
    return {
        "state": "Measured",
        "pivot_volumes_decline": sequence_declines,
        "breakout_volume_expansion": breakout_expansion,
        "pivot_volumes": [_finite_or_none(value) for value in pivot_volumes],
        "score": _finite_or_none(min(max(score, 0.0), 1.0)),
    }


def _rectangle_status_score(status: str) -> float:
    return {
        "Breakout": 0.95,
        "Breakdown": 0.95,
        "Retest": 0.90,
        "Candidate": 0.55,
        "ObjectiveReached": 0.45,
        "FalseBreakout": 0.35,
        "FalseBreakdown": 0.35,
        "PrematureBreakout": 0.30,
        "PrematureBreakdown": 0.30,
    }.get(status, 0.20)


def _multi_status_score(status: str) -> float:
    return {
        "Confirmed": 0.95,
        "PullbackToConfirmation": 0.90,
        "ObjectiveReached": 0.55,
        "Suspected": 0.40,
        "NoPattern": 0.0,
        "InsufficientData": 0.0,
    }.get(status, 0.20)


def _rectangle_notes(status: str, volume_state: dict[str, Any], width_pct: float) -> list[str]:
    notes = []
    if status == "Candidate":
        notes.append("Rectangle is unbroken; direction should wait for boundary confirmation.")
    if status in {"PrematureBreakout", "PrematureBreakdown"}:
        notes.append("Initial break returned inside the rectangle, so the move is treated as premature.")
    if status in {"FalseBreakout", "FalseBreakdown"}:
        notes.append("Initial break reversed through the opposite side of the rectangle.")
    if volume_state.get("volume_contracts_inside_pattern") is False:
        notes.append("Volume did not contract clearly inside the range.")
    if volume_state.get("breakout_volume_expansion") is False:
        notes.append("Boundary break lacks volume expansion.")
    if width_pct < 0.06:
        notes.append("Rectangle is relatively narrow; measured objective may be less useful.")
    return notes


def _multi_notes(status: str, volume_state: dict[str, Any], is_top: bool) -> list[str]:
    notes = []
    if status == "Suspected":
        notes.append("Pattern is not confirmed; wait for the intervening valley or peak break.")
    if is_top and volume_state.get("pivot_volumes_decline") is False:
        notes.append("Volume did not decline across the candidate peaks.")
    if not is_top and volume_state.get("breakout_volume_expansion") is False:
        notes.append("Bottom confirmation lacks upside volume expansion.")
    return notes


def _choose_preferred(timeframes: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    patterns = []
    for payload in timeframes.values():
        if key == "rectangle":
            pattern = payload.get("rectangle", {})
            if isinstance(pattern, dict):
                patterns.append(pattern)
        else:
            pattern = payload.get("multi_top_bottom", {})
            if isinstance(pattern, dict):
                patterns.append(pattern)
    actionable = [pattern for pattern in patterns if pattern.get("status") not in {"NoPattern", "InsufficientData"}]
    if actionable:
        ranker = _rectangle_rank if key == "rectangle" else _multi_pattern_rank
        payload = max(actionable, key=ranker)
        return _preferred_rectangle_payload(payload) if key == "rectangle" else _preferred_multi_payload(payload)
    for timeframe in ("monthly", "weekly", "daily"):
        payload = timeframes.get(timeframe, {}).get("rectangle" if key == "rectangle" else "multi_top_bottom", {})
        if payload:
            return _preferred_rectangle_payload(payload) if key == "rectangle" else _preferred_multi_payload(payload)
    return _empty_pattern("Rectangle" if key == "rectangle" else "NoMultiTopBottom", "unknown", "NoPattern", "pattern unavailable", 0)


def _choose_chapter_9_preferred(rectangle: dict[str, Any], multi: dict[str, Any]) -> dict[str, Any]:
    candidates = [rectangle, multi]
    candidates = [candidate for candidate in candidates if candidate.get("status") not in {"NoPattern", "InsufficientData"}]
    if not candidates:
        return {"pattern": "NoChapter9Pattern", "status": "NoPattern", "reason": "no Chapter 9 pattern detected"}
    return max(candidates, key=lambda item: max(_rectangle_rank(item), _multi_pattern_rank(item)))


def _rectangle_rank(pattern: dict[str, Any]) -> tuple[float, float, str]:
    status_rank = {
        "Retest": 5.0,
        "Breakout": 4.7,
        "Breakdown": 4.7,
        "FalseBreakout": 3.4,
        "FalseBreakdown": 3.4,
        "PrematureBreakout": 3.0,
        "PrematureBreakdown": 3.0,
        "Candidate": 2.0,
        "ObjectiveReached": 1.3,
        "NoPattern": 0.0,
        "InsufficientData": 0.0,
    }.get(str(pattern.get("status")), 0.0)
    timeframe_bonus = {"monthly": 0.30, "weekly": 0.15, "daily": 0.0}.get(str(pattern.get("timeframe")), 0.0)
    score = float(pattern.get("score") or 0.0)
    date = pattern.get("breakout_date") or pattern.get("boundaries", {}).get("latest_date") or ""
    return (status_rank + timeframe_bonus, score, str(date))


def _multi_pattern_rank(pattern: dict[str, Any]) -> tuple[float, float, str]:
    status_rank = {
        "PullbackToConfirmation": 5.0,
        "Confirmed": 4.8,
        "ObjectiveReached": 2.0,
        "Suspected": 1.0,
        "NoPattern": 0.0,
        "InsufficientData": 0.0,
    }.get(str(pattern.get("status")), 0.0)
    pattern_bonus = 0.25 if str(pattern.get("pattern", "")).startswith("Triple") else 0.0
    timeframe_bonus = {"monthly": 0.30, "weekly": 0.15, "daily": 0.0}.get(str(pattern.get("timeframe")), 0.0)
    score = float(pattern.get("score") or 0.0)
    date = pattern.get("confirmation_date") or ""
    return (status_rank + pattern_bonus + timeframe_bonus, score, str(date))


def _preferred_rectangle_payload(pattern: dict[str, Any]) -> dict[str, Any]:
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
        "boundaries",
        "volume_confirmation",
        "retest",
        "reliability_notes",
        "reason",
    ]
    return {key: pattern[key] for key in keys if key in pattern}


def _preferred_multi_payload(pattern: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "pattern",
        "status",
        "direction",
        "timeframe",
        "score",
        "latest_close",
        "level",
        "confirmation_level",
        "confirmation_date",
        "confirmation_close",
        "confirmation_margin_pct",
        "measured_objective",
        "measured_move_pct",
        "equal_level_error_pct",
        "reaction_depth_pct",
        "prior_trend",
        "points",
        "volume_confirmation",
        "reliability_notes",
        "reason",
    ]
    return {key: pattern[key] for key in keys if key in pattern}


def _empty_timeframe(timeframe: str, status: str, reason: str, rows: int) -> dict[str, Any]:
    return {
        "state": status,
        "timeframe": timeframe,
        "rows": int(rows),
        "rectangle": _empty_pattern("Rectangle", timeframe, status, reason, rows),
        "double_top": _empty_pattern("DoubleTop", timeframe, status, reason, rows),
        "double_bottom": _empty_pattern("DoubleBottom", timeframe, status, reason, rows),
        "triple_top": _empty_pattern("TripleTop", timeframe, status, reason, rows),
        "triple_bottom": _empty_pattern("TripleBottom", timeframe, status, reason, rows),
        "multi_top_bottom": _empty_pattern("NoMultiTopBottom", timeframe, status, reason, rows),
    }


def _empty_pattern(pattern: str, timeframe: str, status: str, reason: str, rows: int) -> dict[str, Any]:
    return {
        "pattern": pattern,
        "status": status,
        "timeframe": timeframe,
        "rows": int(rows),
        "reason": reason,
    }


def _pattern_name(kind: str) -> str:
    return {
        "double_top": "DoubleTop",
        "double_bottom": "DoubleBottom",
        "triple_top": "TripleTop",
        "triple_bottom": "TripleBottom",
    }[kind]


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
