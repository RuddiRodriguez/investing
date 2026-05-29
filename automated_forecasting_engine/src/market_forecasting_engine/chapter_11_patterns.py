from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PIVOT_LEFT_BARS = 3
PIVOT_RIGHT_BARS = 3
MIN_PATTERN_BARS = 55
BREAKOUT_PCT = 0.025
RETEST_PCT = 0.025
MIN_MAST_RETURN = {"daily": 0.08, "weekly": 0.10, "monthly": 0.12, "chart": 0.07}
MAX_CONSOLIDATION_BARS = {"daily": 20, "weekly": 4, "monthly": 2, "chart": 20}
CONSOLIDATION_WINDOWS = {
    "daily": (6, 8, 13, 18, 21),
    "weekly": (3, 4, 5),
    "monthly": (),
    "chart": (6, 8, 13, 18, 21),
}
MAST_WINDOWS = {
    "daily": (8, 13, 21, 34),
    "weekly": (4, 6, 8, 13),
    "monthly": (),
    "chart": (8, 13, 21, 34),
}
MIN_CONTRA_SLOPE_PCT = 0.0004
MAX_CONSOLIDATION_TO_MAST = 0.75
MIN_PENNANT_COMPRESSION = 0.22
PRIOR_TREND_LOOKBACK = {"daily": 126, "weekly": 52, "monthly": 36, "chart": 90}


def analyze_chapter_11_patterns(prices: pd.DataFrame, target_column: str = "close") -> dict[str, Any]:
    """Analyze Edwards/Magee Chapter 11 consolidation and continuation formations."""

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
    continuation = {
        "principle": (
            "Edwards/Magee Chapter 11: flags and pennants are short-lived continuation formations after sharp moves; "
            "they need volume contraction, timely breakout, and a half-mast objective."
        ),
        "preferred": _choose_preferred(timeframes, "continuation_preferred"),
        "timeframes": {
            name: {
                "state": payload.get("state"),
                "rows": payload.get("rows"),
                "flag": payload.get("flag", {}),
                "pennant": payload.get("pennant", {}),
                "preferred": payload.get("continuation_preferred", {}),
            }
            for name, payload in timeframes.items()
        },
    }
    hs_continuation = {
        "principle": (
            "Head-and-Shoulders consolidations are continuation patterns: H&S Bottom shape in an advance, "
            "or H&S Top shape in a decline."
        ),
        "preferred": _choose_preferred(timeframes, "head_and_shoulders_continuation"),
        "timeframes": {
            name: {
                "state": payload.get("state"),
                "rows": payload.get("rows"),
                "preferred": payload.get("head_and_shoulders_continuation", {}),
            }
            for name, payload in timeframes.items()
        },
    }
    scallops = {
        "principle": "Scallops are optional context for repeated saucer-like continuation advances, not an automatic signal.",
        "preferred": _choose_preferred(timeframes, "scallop_context"),
        "timeframes": {
            name: {
                "state": payload.get("state"),
                "rows": payload.get("rows"),
                "preferred": payload.get("scallop_context", {}),
            }
            for name, payload in timeframes.items()
        },
    }
    return {
        "principle": (
            "Chapter 11 adds continuation evidence. Clean flags and pennants can support model direction; "
            "stale, high-volume, failed, or conflicting continuations can block fresh directional action."
        ),
        "primary_timeframe": "daily",
        "secondary_timeframe": "weekly",
        "monthly_rule": "Do not treat monthly flag-like shapes as true flags or pennants.",
        "confirmation_rule": f"Close at least {BREAKOUT_PCT:.1%} beyond the flag or pennant boundary in the mast direction.",
        "preferred": _choose_chapter_11_preferred(
            continuation["preferred"],
            hs_continuation["preferred"],
            scallops["preferred"],
        ),
        "continuation_patterns": continuation,
        "head_and_shoulders_continuation": hs_continuation,
        "scallop_context": scallops,
        "timeframes": timeframes,
        "technical_method_card": chapter_11_patterns_method_card(target_column=target),
    }


def latest_chapter_11_patterns(
    prices: pd.DataFrame,
    target_column: str = "close",
    timeframe: str = "chart",
) -> dict[str, Any]:
    """Return Chapter 11 diagnostics for a plotted bar set."""

    return _analyze_timeframe(
        frame=prices,
        timeframe=timeframe,
        target_column=target_column.lower(),
    )


def chapter_11_patterns_method_card(target_column: str = "close") -> dict[str, Any]:
    return {
        "name": "edwards_magee_chapter_11_continuation_patterns",
        "version": "chapter_11_alignment_v1",
        "target_column": target_column.lower(),
        "implemented_patterns": [
            "flag",
            "pennant",
            "head_and_shoulders_continuation",
            "scallop_context_optional",
        ],
        "flag_pennant_rules": {
            "mast_required": "sharp straight-line move immediately before the consolidation",
            "volume": "volume should contract during the consolidation and expand on breakout",
            "duration": "daily patterns should normally break within 3-4 weeks; weekly/monthly flag-like shapes are suspect",
            "objective": "half-mast objective: project the prior mast distance from the breakout point",
        },
        "head_and_shoulders_continuation": {
            "bullish": "H&S Bottom contour inside an existing uptrend",
            "bearish": "H&S Top contour inside an existing downtrend",
            "decision_use": "supports matching model direction; blocks only when confirmed continuation conflicts with the model",
        },
        "scallops": {
            "decision_use": "optional context only; useful for slow repeated saucer advances, not a hard action gate",
        },
        "decision_use": "Chapter 11 can add supporting reasons when continuation agrees with the model and block stale/failed/conflicting continuations.",
    }


def _analyze_timeframe(frame: pd.DataFrame, timeframe: str, target_column: str) -> dict[str, Any]:
    clean = frame.copy().sort_index()
    if clean.empty or target_column not in clean.columns:
        return _empty_timeframe(timeframe, "InsufficientData", "missing target column", len(clean))
    clean.index = pd.DatetimeIndex(clean.index)
    clean = clean.loc[pd.to_numeric(clean[target_column], errors="coerce").notna()]
    if len(clean) < MIN_PATTERN_BARS:
        return _empty_timeframe(timeframe, "InsufficientData", "not enough bars for Chapter 11 analysis", len(clean))

    flag, pennant = _find_flag_or_pennant(clean, timeframe=timeframe, target_column=target_column)
    continuation_preferred = max([flag, pennant], key=_continuation_rank)
    hs = _find_hs_continuation(clean, timeframe=timeframe, target_column=target_column)
    scallop = _find_scallop_context(clean, timeframe=timeframe, target_column=target_column)
    return {
        "state": "Measured",
        "timeframe": timeframe,
        "rows": int(len(clean)),
        "start_date": _date_string(clean.index[0]),
        "end_date": _date_string(clean.index[-1]),
        "flag": flag,
        "pennant": pennant,
        "continuation_preferred": _preferred_payload(continuation_preferred),
        "head_and_shoulders_continuation": hs,
        "scallop_context": scallop,
    }


def _find_flag_or_pennant(frame: pd.DataFrame, timeframe: str, target_column: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not CONSOLIDATION_WINDOWS.get(timeframe):
        empty = _empty_pattern("NoFlagPennant", timeframe, "NoPattern", "true flags/pennants are not evaluated on this timeframe", len(frame))
        return _empty_pattern("Flag", timeframe, "NoPattern", "monthly flag-like shapes are not trusted", len(frame)), empty
    close = pd.to_numeric(frame[target_column], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce") if "high" in frame.columns else close
    low = pd.to_numeric(frame["low"], errors="coerce") if "low" in frame.columns else close
    volume = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame.columns else None
    candidates: list[dict[str, Any]] = []
    for consolidation_window in CONSOLIDATION_WINDOWS[timeframe]:
        for mast_window in MAST_WINDOWS[timeframe]:
            # end is exclusive for the consolidation; len(close)-1 lets the latest bar be a breakout.
            for end in (len(close), len(close) - 1):
                start = end - consolidation_window
                mast_start = start - mast_window
                if mast_start < 0 or start <= 1 or end > len(close) or end <= start:
                    continue
                candidate = _flag_pennant_candidate(
                    close=close,
                    high=high,
                    low=low,
                    volume=volume,
                    timeframe=timeframe,
                    pattern_window=consolidation_window,
                    mast_window=mast_window,
                    mast_start=mast_start,
                    start=start,
                    end=end,
                )
                if candidate is not None:
                    candidates.append(candidate)
    flags = [candidate for candidate in candidates if candidate["pattern"] == "Flag"]
    pennants = [candidate for candidate in candidates if candidate["pattern"] == "Pennant"]
    flag = max(flags, key=_continuation_rank) if flags else _empty_pattern("Flag", timeframe, "NoPattern", "no valid flag geometry", len(frame))
    pennant = max(pennants, key=_continuation_rank) if pennants else _empty_pattern("Pennant", timeframe, "NoPattern", "no valid pennant geometry", len(frame))
    return flag, pennant


def _flag_pennant_candidate(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series | None,
    timeframe: str,
    pattern_window: int,
    mast_window: int,
    mast_start: int,
    start: int,
    end: int,
) -> dict[str, Any] | None:
    scope_index = close.index[start:end]
    if len(scope_index) < 4:
        return None
    mast_base = float(close.iloc[mast_start])
    mast_peak = float(close.iloc[start - 1])
    if mast_base <= 0:
        return None
    mast_return = mast_peak / mast_base - 1.0
    min_mast = MIN_MAST_RETURN.get(timeframe, MIN_MAST_RETURN["chart"])
    if abs(mast_return) < min_mast:
        return None
    expected_direction = "bullish" if mast_return > 0 else "bearish"
    expected_break = "up" if expected_direction == "bullish" else "down"
    scope_high = high.iloc[start:end]
    scope_low = low.iloc[start:end]
    positions = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    upper = _fit_boundary(positions.reindex(scope_index), scope_high)
    lower = _fit_boundary(positions.reindex(scope_index), scope_low)
    if upper is None or lower is None:
        return None
    latest_pattern_position = end - 1
    upper_start = _line_value(upper, start)
    lower_start = _line_value(lower, start)
    upper_end = _line_value(upper, latest_pattern_position)
    lower_end = _line_value(lower, latest_pattern_position)
    if min(upper_start, lower_start, upper_end, lower_end) <= 0 or upper_start <= lower_start or upper_end <= lower_end:
        return None
    pattern_range = float(scope_high.max() - scope_low.min())
    mast_distance = abs(mast_peak - mast_base)
    if mast_distance <= 0 or pattern_range / mast_distance > MAX_CONSOLIDATION_TO_MAST:
        return None

    average_price = float(close.iloc[start:end].mean())
    upper_slope_pct = upper["slope"] / average_price if average_price else 0.0
    lower_slope_pct = lower["slope"] / average_price if average_price else 0.0
    base_width = upper_start - lower_start
    latest_width = upper_end - lower_end
    compression = 1.0 - latest_width / base_width if base_width > 0 else 0.0
    parallel_gap = abs(upper_slope_pct - lower_slope_pct)
    slopes_against_mast = (
        upper_slope_pct < -MIN_CONTRA_SLOPE_PCT and lower_slope_pct < MIN_CONTRA_SLOPE_PCT
        if expected_direction == "bullish"
        else upper_slope_pct > -MIN_CONTRA_SLOPE_PCT and lower_slope_pct > MIN_CONTRA_SLOPE_PCT
    )
    is_flag = parallel_gap <= 0.0013 and slopes_against_mast
    is_pennant = compression >= MIN_PENNANT_COMPRESSION and (
        (upper_slope_pct < -MIN_CONTRA_SLOPE_PCT and lower_slope_pct > MIN_CONTRA_SLOPE_PCT)
        or (expected_direction == "bearish" and upper_slope_pct < -MIN_CONTRA_SLOPE_PCT and lower_slope_pct > MIN_CONTRA_SLOPE_PCT)
    )
    if not is_flag and not is_pennant:
        return None
    pattern = "Flag" if is_flag else "Pennant"
    break_info = _breakout_state(close, upper, lower, start_position=end)
    volume_state = _volume_state(
        volume,
        start_position=start,
        end_position=end,
        break_position=break_info.get("position"),
    )
    status = _status_from_break(pattern_window, expected_break, break_info, volume_state, timeframe)
    direction = expected_direction if status in {"Breakout", "Breakdown", "Candidate", "Stale"} else f"failed_{expected_direction}"
    breakout_close = break_info.get("close") or float(close.iloc[-1])
    objective = None
    if status in {"Breakout", "Breakdown"}:
        objective = float(breakout_close + mast_distance) if expected_direction == "bullish" else float(breakout_close - mast_distance)
        if objective <= 0:
            objective = None
        latest_close = float(close.iloc[-1])
        if objective is not None and ((expected_direction == "bullish" and latest_close >= objective) or (expected_direction == "bearish" and latest_close <= objective)):
            status = "ObjectiveReached"
    else:
        latest_close = float(close.iloc[-1])

    geometry_score = float(
        0.35 * _clip01(abs(mast_return) / 0.20)
        + 0.25 * _clip01(1.0 - pattern_range / max(mast_distance, 1e-9))
        + 0.20 * (1.0 if pattern == "Flag" else _clip01(compression / 0.55))
        + 0.20 * _clip01((MAX_CONSOLIDATION_BARS.get(timeframe, 20) + 1 - pattern_window) / max(MAX_CONSOLIDATION_BARS.get(timeframe, 20), 1))
    )
    score = float(0.45 * geometry_score + 0.25 * float(volume_state.get("score", 0.5)) + 0.30 * _status_score(status))
    return {
        "pattern": pattern,
        "status": status,
        "direction": direction,
        "expected_breakout_direction": expected_break,
        "timeframe": timeframe,
        "score": _finite_or_none(score),
        "latest_close": _finite_or_none(latest_close),
        "window_bars": int(pattern_window),
        "mast": {
            "start_date": _date_string(close.index[mast_start]),
            "end_date": _date_string(close.index[start - 1]),
            "start_price": _finite_or_none(mast_base),
            "end_price": _finite_or_none(mast_peak),
            "return_pct": _finite_or_none(mast_return),
            "height": _finite_or_none(mast_distance),
        },
        "breakout_date": break_info.get("date"),
        "breakout_close": break_info.get("close"),
        "breakout_direction": break_info.get("direction"),
        "breakout_margin_pct": break_info.get("margin_pct"),
        "measured_objective": _finite_or_none(objective),
        "measured_move_pct": _finite_or_none(objective / latest_close - 1.0) if objective and latest_close else None,
        "boundaries": {
            "start_date": _date_string(close.index[start]),
            "latest_date": _date_string(close.index[end - 1]),
            "upper_start": _finite_or_none(upper_start),
            "upper_latest": _finite_or_none(upper_end),
            "upper_slope_pct_per_bar": _finite_or_none(upper_slope_pct),
            "lower_start": _finite_or_none(lower_start),
            "lower_latest": _finite_or_none(lower_end),
            "lower_slope_pct_per_bar": _finite_or_none(lower_slope_pct),
            "base_width": _finite_or_none(base_width),
            "latest_width": _finite_or_none(latest_width),
            "width_compression_pct": _finite_or_none(compression),
            "pattern_range_to_mast_pct": _finite_or_none(pattern_range / mast_distance),
        },
        "volume_confirmation": volume_state,
        "reliability_notes": _continuation_notes(pattern, status, volume_state, pattern_window, timeframe),
    }


def _find_hs_continuation(frame: pd.DataFrame, timeframe: str, target_column: str) -> dict[str, Any]:
    close = pd.to_numeric(frame[target_column], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce") if "high" in frame.columns else close
    low = pd.to_numeric(frame["low"], errors="coerce") if "low" in frame.columns else close
    pivot_high, pivot_low = _confirmed_pivots(high, low)
    positions = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    candidates = []
    candidates.extend(_hs_continuation_candidates(close, pivot_high, pivot_low, positions, timeframe, kind="bullish"))
    candidates.extend(_hs_continuation_candidates(close, pivot_high, pivot_low, positions, timeframe, kind="bearish"))
    if not candidates:
        return _empty_pattern("HeadAndShouldersContinuation", timeframe, "NoPattern", "no valid H&S continuation geometry", len(frame))
    return max(candidates, key=_continuation_rank)


def _hs_continuation_candidates(
    close: pd.Series,
    pivot_high: pd.Series,
    pivot_low: pd.Series,
    positions: pd.Series,
    timeframe: str,
    kind: str,
) -> list[dict[str, Any]]:
    lows = list(pivot_low.dropna().items())[-18:]
    highs = list(pivot_high.dropna().items())[-18:]
    candidates = []
    if kind == "bullish":
        for i in range(len(lows) - 2):
            left_date, left = lows[i]
            head_date, head = lows[i + 1]
            right_date, right = lows[i + 2]
            lp, hp, rp = int(positions.loc[left_date]), int(positions.loc[head_date]), int(positions.loc[right_date])
            if not (left > head and right > head and abs(left / right - 1.0) <= 0.12):
                continue
            neck_points = [(date, value) for date, value in highs if lp < int(positions.loc[date]) < rp]
            if len(neck_points) < 2:
                continue
            prior = _prior_trend(close, lp, timeframe, require_advance=True)
            if prior["state"] != "PriorAdvance":
                continue
            neckline = float(np.mean([float(value) for _, value in neck_points[:2]]))
            break_info = _horizontal_breakout(close, lower=0.0, upper=neckline, start_position=rp)
            candidates.append(_hs_payload(close, timeframe, kind, left_date, left, head_date, head, right_date, right, neckline, break_info, prior))
    else:
        for i in range(len(highs) - 2):
            left_date, left = highs[i]
            head_date, head = highs[i + 1]
            right_date, right = highs[i + 2]
            lp, hp, rp = int(positions.loc[left_date]), int(positions.loc[head_date]), int(positions.loc[right_date])
            if not (left < head and right < head and abs(left / right - 1.0) <= 0.12):
                continue
            neck_points = [(date, value) for date, value in lows if lp < int(positions.loc[date]) < rp]
            if len(neck_points) < 2:
                continue
            prior = _prior_trend(close, lp, timeframe, require_advance=False)
            if prior["state"] != "PriorDecline":
                continue
            neckline = float(np.mean([float(value) for _, value in neck_points[:2]]))
            break_info = _horizontal_breakout(close, lower=neckline, upper=np.inf, start_position=rp)
            candidates.append(_hs_payload(close, timeframe, kind, left_date, left, head_date, head, right_date, right, neckline, break_info, prior))
    return candidates


def _hs_payload(
    close: pd.Series,
    timeframe: str,
    kind: str,
    left_date: Any,
    left: float,
    head_date: Any,
    head: float,
    right_date: Any,
    right: float,
    neckline: float,
    break_info: dict[str, Any],
    prior: dict[str, Any],
) -> dict[str, Any]:
    bullish = kind == "bullish"
    confirmed = break_info.get("direction") == ("up" if bullish else "down")
    status = "Confirmed" if confirmed else "Candidate"
    direction = "bullish" if bullish else "bearish"
    latest_close = float(close.iloc[-1])
    height = abs(neckline - float(head))
    objective = None
    if confirmed:
        objective = float(break_info.get("close") + height) if bullish else float(break_info.get("close") - height)
        if objective <= 0:
            objective = None
    score = float(0.55 + 0.25 * _status_score(status) + 0.20 * _clip01(height / max(latest_close, 1e-9)))
    return {
        "pattern": "HeadAndShouldersContinuation",
        "status": status,
        "direction": direction,
        "timeframe": timeframe,
        "score": _finite_or_none(min(score, 0.99)),
        "latest_close": _finite_or_none(latest_close),
        "neckline": _finite_or_none(neckline),
        "breakout_date": break_info.get("date"),
        "breakout_close": break_info.get("close"),
        "breakout_direction": break_info.get("direction"),
        "measured_objective": _finite_or_none(objective),
        "measured_move_pct": _finite_or_none(objective / latest_close - 1.0) if objective and latest_close else None,
        "prior_trend": prior,
        "points": {
            "left_shoulder": {"date": _date_string(left_date), "price": _finite_or_none(left)},
            "head": {"date": _date_string(head_date), "price": _finite_or_none(head)},
            "right_shoulder": {"date": _date_string(right_date), "price": _finite_or_none(right)},
        },
        "reliability_notes": ["Continuation H&S uses context opposite to reversal H&S; validate against the prior trend."],
    }


def _find_scallop_context(frame: pd.DataFrame, timeframe: str, target_column: str) -> dict[str, Any]:
    if timeframe not in {"daily", "chart"}:
        return _empty_pattern("ScallopContext", timeframe, "NoPattern", "scallops are evaluated on detailed daily-style charts", len(frame))
    close = pd.to_numeric(frame[target_column], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce") if "high" in frame.columns else close
    low = pd.to_numeric(frame["low"], errors="coerce") if "low" in frame.columns else close
    pivot_high, pivot_low = _confirmed_pivots(high, low)
    lows = list(pivot_low.dropna().items())[-8:]
    highs = list(pivot_high.dropna().items())[-8:]
    if len(lows) < 2 or len(highs) < 2:
        return _empty_pattern("ScallopContext", timeframe, "NoPattern", "not enough repeated saucer pivots", len(frame))
    recent_lows = lows[-3:]
    recent_highs = highs[-3:]
    low_values = [float(value) for _, value in recent_lows]
    high_values = [float(value) for _, value in recent_highs]
    higher_lows = all(right >= left * 0.98 for left, right in zip(low_values, low_values[1:]))
    higher_highs = all(right >= left * 1.02 for left, right in zip(high_values, high_values[1:]))
    if not higher_lows or not higher_highs:
        return _empty_pattern("ScallopContext", timeframe, "NoPattern", "no repeated rising saucer sequence", len(frame))
    latest_close = float(close.iloc[-1])
    score = 0.55 + 0.10 * min(len(recent_lows), 3)
    return {
        "pattern": "ScallopContext",
        "status": "PossibleSequence",
        "direction": "bullish_context",
        "timeframe": timeframe,
        "score": _finite_or_none(score),
        "latest_close": _finite_or_none(latest_close),
        "swing_count": int(min(len(recent_lows), len(recent_highs))),
        "points": {
            "recent_lows": [{"date": _date_string(date), "price": _finite_or_none(value)} for date, value in recent_lows],
            "recent_highs": [{"date": _date_string(date), "price": _finite_or_none(value)} for date, value in recent_highs],
        },
        "decision_use": "context_only",
        "reliability_notes": ["Scallops are optional accumulation context and should not block actions by themselves."],
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
    if len(aligned) < 2:
        return None
    x = aligned.iloc[:, 0].to_numpy(dtype=float)
    y = aligned.iloc[:, 1].to_numpy(dtype=float)
    if len(np.unique(x)) < 2:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    return {"slope": float(slope), "intercept": float(intercept)}


def _line_value(line: dict[str, Any], position: float) -> float:
    return float(line["intercept"] + line["slope"] * position)


def _breakout_state(close: pd.Series, upper: dict[str, Any], lower: dict[str, Any], start_position: int) -> dict[str, Any]:
    for position in range(max(0, int(start_position)), len(close)):
        value = close.iloc[position]
        if pd.isna(value):
            continue
        upper_value = _line_value(upper, position)
        lower_value = _line_value(lower, position)
        if np.isfinite(upper_value) and upper_value > 0 and value >= upper_value * (1.0 + BREAKOUT_PCT):
            return _break_payload(close, position, "up", value / upper_value - 1.0)
        if np.isfinite(lower_value) and lower_value > 0 and value <= lower_value * (1.0 - BREAKOUT_PCT):
            return _break_payload(close, position, "down", value / lower_value - 1.0)
    return {"state": "Inside", "direction": None, "position": None, "date": None, "close": None, "margin_pct": None}


def _horizontal_breakout(close: pd.Series, lower: float, upper: float, start_position: int) -> dict[str, Any]:
    for position in range(max(0, int(start_position)), len(close)):
        value = close.iloc[position]
        if pd.isna(value):
            continue
        if np.isfinite(upper) and upper > 0 and value >= upper * (1.0 + BREAKOUT_PCT):
            return _break_payload(close, position, "up", value / upper - 1.0)
        if np.isfinite(lower) and lower > 0 and value <= lower * (1.0 - BREAKOUT_PCT):
            return _break_payload(close, position, "down", value / lower - 1.0)
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


def _status_from_break(
    pattern_window: int,
    expected_break: str,
    break_info: dict[str, Any],
    volume_state: dict[str, Any],
    timeframe: str,
) -> str:
    if pattern_window > MAX_CONSOLIDATION_BARS.get(timeframe, 20):
        return "Stale"
    if break_info.get("direction") is None:
        return "Candidate"
    if break_info.get("direction") != expected_break:
        return "FailedBreakout" if expected_break == "up" else "FailedBreakdown"
    if volume_state.get("breakout_volume_expansion") is False:
        return "Breakout" if expected_break == "up" else "Breakdown"
    return "Breakout" if expected_break == "up" else "Breakdown"


def _volume_state(
    volume: pd.Series | None,
    start_position: int,
    end_position: int,
    break_position: int | None,
) -> dict[str, Any]:
    if volume is None or volume.dropna().empty:
        return {"state": "Unavailable", "score": 0.5}
    scope = volume.iloc[start_position:end_position]
    first_half = scope.iloc[: max(1, len(scope) // 2)]
    second_half = scope.iloc[max(1, len(scope) // 2) :]
    contraction = None
    if first_half.notna().any() and second_half.notna().any() and float(first_half.median()) > 0:
        contraction = bool(float(second_half.median()) <= float(first_half.median()) * 0.85)
    breakout_expansion = None
    if break_position is not None:
        break_volume = float(volume.iloc[int(break_position)]) if pd.notna(volume.iloc[int(break_position)]) else None
        baseline = volume.shift(1).rolling(20, min_periods=5).mean().iloc[int(break_position)]
        if break_volume is not None and pd.notna(baseline) and float(baseline) > 0:
            breakout_expansion = bool(break_volume >= float(baseline) * 1.20)
    score = 0.5
    if contraction is True:
        score += 0.25
    elif contraction is False:
        score -= 0.15
    if breakout_expansion is True:
        score += 0.20
    elif breakout_expansion is False:
        score -= 0.05
    return {
        "state": "Measured",
        "volume_contracts_inside_pattern": contraction,
        "breakout_volume_expansion": breakout_expansion,
        "score": _finite_or_none(min(max(score, 0.0), 1.0)),
    }


def _prior_trend(close: pd.Series, first_position: int, timeframe: str, require_advance: bool) -> dict[str, Any]:
    lookback = PRIOR_TREND_LOOKBACK.get(timeframe, PRIOR_TREND_LOOKBACK["chart"])
    start = max(0, int(first_position) - lookback)
    if int(first_position) <= start:
        return {"state": "InsufficientData", "return_pct": None}
    start_value = float(close.iloc[start])
    end_value = float(close.iloc[int(first_position)])
    if start_value <= 0:
        return {"state": "InsufficientData", "return_pct": None}
    prior_return = end_value / start_value - 1.0
    threshold = MIN_MAST_RETURN.get(timeframe, MIN_MAST_RETURN["chart"])
    if require_advance:
        state = "PriorAdvance" if prior_return >= threshold else "NoPriorAdvance"
    else:
        state = "PriorDecline" if prior_return <= -threshold else "NoPriorDecline"
    return {"state": state, "return_pct": _finite_or_none(prior_return), "threshold_pct": threshold}


def _status_score(status: str) -> float:
    return {
        "Breakout": 0.95,
        "Breakdown": 0.95,
        "Confirmed": 0.90,
        "Candidate": 0.50,
        "Stale": 0.20,
        "FailedBreakout": 0.20,
        "FailedBreakdown": 0.20,
        "ObjectiveReached": 0.45,
        "PossibleSequence": 0.45,
    }.get(status, 0.20)


def _continuation_rank(pattern: dict[str, Any]) -> tuple[float, float, str]:
    status_rank = {
        "Breakout": 5.0,
        "Breakdown": 5.0,
        "Confirmed": 4.5,
        "Candidate": 2.0,
        "PossibleSequence": 1.0,
        "ObjectiveReached": 0.8,
        "Stale": 0.5,
        "FailedBreakout": 0.3,
        "FailedBreakdown": 0.3,
        "NoPattern": 0.0,
        "InsufficientData": 0.0,
    }.get(str(pattern.get("status")), 0.0)
    pattern_bonus = {"Flag": 0.25, "Pennant": 0.25, "HeadAndShouldersContinuation": 0.15}.get(str(pattern.get("pattern")), 0.0)
    timeframe_bonus = {"daily": 0.20, "weekly": 0.05, "chart": 0.10}.get(str(pattern.get("timeframe")), 0.0)
    score = float(pattern.get("score") or 0.0)
    date = pattern.get("breakout_date") or pattern.get("boundaries", {}).get("latest_date") or ""
    return (status_rank + pattern_bonus + timeframe_bonus, score, str(date))


def _choose_preferred(timeframes: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    patterns = [payload.get(key, {}) for payload in timeframes.values() if isinstance(payload.get(key), dict)]
    actionable = [pattern for pattern in patterns if pattern.get("status") not in {"NoPattern", "InsufficientData"}]
    if actionable:
        return _preferred_payload(max(actionable, key=_continuation_rank))
    for timeframe in ("daily", "weekly", "monthly"):
        pattern = timeframes.get(timeframe, {}).get(key, {})
        if pattern:
            return _preferred_payload(pattern)
    return _empty_pattern("NoChapter11Pattern", "unknown", "NoPattern", "pattern unavailable", 0)


def _choose_chapter_11_preferred(continuation: dict[str, Any], hs: dict[str, Any], scallop: dict[str, Any]) -> dict[str, Any]:
    candidates = [continuation, hs, scallop]
    candidates = [candidate for candidate in candidates if candidate.get("status") not in {"NoPattern", "InsufficientData"}]
    if not candidates:
        return {"pattern": "NoChapter11Pattern", "status": "NoPattern", "reason": "no Chapter 11 continuation pattern detected"}
    return _preferred_payload(max(candidates, key=_continuation_rank))


def _preferred_payload(pattern: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "pattern",
        "status",
        "direction",
        "expected_breakout_direction",
        "timeframe",
        "score",
        "latest_close",
        "window_bars",
        "mast",
        "neckline",
        "breakout_date",
        "breakout_close",
        "breakout_direction",
        "breakout_margin_pct",
        "measured_objective",
        "measured_move_pct",
        "boundaries",
        "volume_confirmation",
        "prior_trend",
        "points",
        "swing_count",
        "decision_use",
        "reliability_notes",
        "reason",
    ]
    return {key: pattern[key] for key in keys if key in pattern}


def _empty_timeframe(timeframe: str, status: str, reason: str, rows: int) -> dict[str, Any]:
    return {
        "state": status,
        "timeframe": timeframe,
        "rows": int(rows),
        "flag": _empty_pattern("Flag", timeframe, status, reason, rows),
        "pennant": _empty_pattern("Pennant", timeframe, status, reason, rows),
        "continuation_preferred": _empty_pattern("NoFlagPennant", timeframe, status, reason, rows),
        "head_and_shoulders_continuation": _empty_pattern("HeadAndShouldersContinuation", timeframe, status, reason, rows),
        "scallop_context": _empty_pattern("ScallopContext", timeframe, status, reason, rows),
    }


def _empty_pattern(pattern: str, timeframe: str, status: str, reason: str, rows: int) -> dict[str, Any]:
    return {"pattern": pattern, "status": status, "timeframe": timeframe, "rows": int(rows), "reason": reason}


def _continuation_notes(pattern: str, status: str, volume_state: dict[str, Any], pattern_window: int, timeframe: str) -> list[str]:
    notes = []
    if status == "Candidate":
        notes.append(f"{pattern} is forming but has not broken in the mast direction.")
    if status == "Stale" or pattern_window > MAX_CONSOLIDATION_BARS.get(timeframe, 20):
        notes.append("Pattern has run longer than the Chapter 11 reliability window.")
    if status in {"FailedBreakout", "FailedBreakdown"}:
        notes.append("Pattern broke against the expected continuation direction.")
    if volume_state.get("volume_contracts_inside_pattern") is False:
        notes.append("Volume did not contract during the consolidation.")
    if volume_state.get("breakout_volume_expansion") is False:
        notes.append("Breakout lacks volume expansion.")
    return notes


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
