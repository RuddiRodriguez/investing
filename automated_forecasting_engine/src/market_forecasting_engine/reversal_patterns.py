from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PIVOT_LEFT_BARS = 3
PIVOT_RIGHT_BARS = 3
NECKLINE_BREAK_PCT = 0.03
NECKLINE_TEST_PCT = 0.03
MIN_HEAD_PROMINENCE_PCT = 0.015
MAX_SHOULDER_IMBALANCE_PCT = 0.22
MIN_PATTERN_BARS = 45
PATTERN_LOOKBACK_BARS = {"daily": 378, "weekly": 156, "monthly": 84, "chart": 180}
PRIOR_TREND_LOOKBACK_BARS = {"daily": 126, "weekly": 52, "monthly": 36, "chart": 90}
PRIOR_TREND_MIN_RETURN = {"daily": 0.08, "weekly": 0.10, "monthly": 0.12, "chart": 0.06}
DORMANT_BASE_BARS = {"daily": 120, "weekly": 52, "monthly": 24, "chart": 90}
DORMANT_BREAKOUT_BARS = {"daily": 20, "weekly": 8, "monthly": 4, "chart": 15}


def analyze_reversal_patterns(prices: pd.DataFrame, target_column: str = "close") -> dict[str, Any]:
    """Analyze Edwards/Magee Chapter 6-7 reversal patterns."""

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
            "Edwards/Magee reversal diagnostics: a reversal must have a prior trend to reverse; "
            "tops emphasize distribution and downside neckline breaks, while bottoms require accumulation "
            "evidence and strong upside breakout volume."
        ),
        "primary_timeframe": "weekly",
        "secondary_timeframe": "daily",
        "higher_order_timeframe": "monthly",
        "confirmation_rule": f"Close at least {NECKLINE_BREAK_PCT:.0%} beyond the neckline after the right shoulder is confirmed.",
        "preferred": preferred,
        "preferred_top": _choose_pattern(timeframes, "head_and_shoulders_top"),
        "preferred_bottom": _choose_pattern(timeframes, "head_and_shoulders_bottom"),
        "timeframes": timeframes,
        "optional_methods": {
            "complex_head_and_shoulders": {
                "use": "Warning-only context for less clean multiple H&S reversals; do not use as an automatic trade trigger.",
                "preferred": _choose_optional(timeframes, "complex_head_and_shoulders"),
                "timeframes": _optional_by_timeframe(timeframes, "complex_head_and_shoulders"),
            },
            "dormant_bottoms": {
                "use": "Optional long-horizon accumulation screen for low-liquidity or forgotten issues after a large decline.",
                "preferred": _choose_optional(timeframes, "dormant_bottom"),
                "timeframes": _optional_by_timeframe(timeframes, "dormant_bottom"),
            },
        },
        "technical_method_card": reversal_patterns_method_card(target_column=target),
    }


def latest_head_and_shoulders_top(
    prices: pd.DataFrame,
    target_column: str = "close",
    timeframe: str = "chart",
) -> dict[str, Any]:
    """Return the latest Head-and-Shoulders Top pattern for a plotted bar set."""

    return latest_reversal_patterns(prices, target_column=target_column, timeframe=timeframe).get("head_and_shoulders_top", {})


def latest_reversal_patterns(
    prices: pd.DataFrame,
    target_column: str = "close",
    timeframe: str = "chart",
) -> dict[str, Any]:
    """Return reversal diagnostics for a plotted bar set."""

    return _analyze_timeframe(
        frame=prices,
        timeframe=timeframe,
        target_column=target_column.lower(),
    )


def reversal_patterns_method_card(target_column: str = "close") -> dict[str, Any]:
    return {
        "name": "edwards_magee_reversal_patterns",
        "version": "chapter_7_alignment_v1",
        "target_column": target_column.lower(),
        "implemented_patterns": [
            "head_and_shoulders_top",
            "head_and_shoulders_bottom",
            "complex_head_and_shoulders_warning",
            "dormant_bottom_optional",
        ],
        "shared_rules": {
            "prior_trend_required": "tops require a prior advance; bottoms require a prior decline",
            "primary_timeframes": ["weekly", "monthly"],
            "secondary_timeframe": "daily",
            "pivot_confirmation": {
                "left_bars": PIVOT_LEFT_BARS,
                "right_bars": PIVOT_RIGHT_BARS,
                "rule": "a pivot is eligible only after the right-side bars confirm it",
            },
        },
        "head_and_shoulders_top": {
            "geometry": "left shoulder, reaction low, higher head, second reaction low, lower right shoulder",
            "confirmation": {
                "neckline": "line through the two reaction lows",
                "break_rule": f"close <= neckline * (1 - {NECKLINE_BREAK_PCT:.2f})",
                "pullback_rule": f"post-break close within +/- {NECKLINE_TEST_PCT:.2f} of the neckline",
            },
            "volume": "right-shoulder volume should contract; break volume strengthens but is not mandatory for downside breaks",
            "objective": "project the head-to-neckline height downward from the neckline break",
            "decision_use": "confirmed active tops can block fresh Buy actions",
        },
        "head_and_shoulders_bottom": {
            "geometry": "left shoulder low, reaction high, lower head, second reaction high, higher right shoulder",
            "confirmation": {
                "neckline": "line through the two reaction highs",
                "break_rule": f"close >= neckline * (1 + {NECKLINE_BREAK_PCT:.2f})",
                "throwback_rule": f"post-break close within +/- {NECKLINE_TEST_PCT:.2f} of the neckline",
            },
            "volume": "upside neckline break should expand versus baseline volume before treating the bottom as confirmed",
            "objective": "project the neckline-to-head height upward from the neckline break",
            "decision_use": "confirmed active bottoms can block fresh Sell actions and support bullish interpretation",
        },
        "complex_head_and_shoulders": {
            "decision_use": "warning-only; complex formations are less clean and should not block actions by themselves",
            "outer_neckline": "prefer the outer neckline when inner and outer boundaries coexist",
        },
        "dormant_bottoms": {
            "decision_use": "optional context only; useful for low-liquidity accumulation screens, not liquid mega-cap timing",
            "requirements": [
                "large prior decline",
                "extended flat low-volatility base",
                "low or fading turnover in the base",
                "recent upside breakout with volume expansion",
            ],
        },
    }


def _analyze_timeframe(frame: pd.DataFrame, timeframe: str, target_column: str) -> dict[str, Any]:
    clean = frame.copy().sort_index()
    empty_payload = {
        "head_and_shoulders_top": _empty_pattern("HeadAndShouldersTop", timeframe, "InsufficientData", "missing target column", len(clean)),
        "head_and_shoulders_bottom": _empty_pattern("HeadAndShouldersBottom", timeframe, "InsufficientData", "missing target column", len(clean)),
        "complex_head_and_shoulders": _empty_pattern("ComplexHeadAndShoulders", timeframe, "InsufficientData", "missing target column", len(clean)),
        "dormant_bottom": _empty_pattern("DormantBottom", timeframe, "InsufficientData", "missing target column", len(clean)),
    }
    if clean.empty or target_column not in clean.columns:
        return {"state": "InsufficientData", "timeframe": timeframe, "rows": int(len(clean)), **empty_payload}
    clean.index = pd.DatetimeIndex(clean.index)
    clean = clean.loc[pd.to_numeric(clean[target_column], errors="coerce").notna()]
    if len(clean) < MIN_PATTERN_BARS:
        payload = {
            "head_and_shoulders_top": _empty_pattern("HeadAndShouldersTop", timeframe, "InsufficientData", "not enough bars for reversal analysis", len(clean)),
            "head_and_shoulders_bottom": _empty_pattern("HeadAndShouldersBottom", timeframe, "InsufficientData", "not enough bars for reversal analysis", len(clean)),
            "complex_head_and_shoulders": _empty_pattern("ComplexHeadAndShoulders", timeframe, "InsufficientData", "not enough bars for reversal analysis", len(clean)),
            "dormant_bottom": _empty_pattern("DormantBottom", timeframe, "InsufficientData", "not enough bars for dormant-bottom analysis", len(clean)),
        }
        return {"state": "InsufficientData", "timeframe": timeframe, "rows": int(len(clean)), **payload}

    top = _find_head_and_shoulders(clean, target_column=target_column, timeframe=timeframe, kind="top")
    bottom = _find_head_and_shoulders(clean, target_column=target_column, timeframe=timeframe, kind="bottom")
    return {
        "state": "Measured",
        "timeframe": timeframe,
        "rows": int(len(clean)),
        "start_date": str(clean.index[0].date()),
        "end_date": str(clean.index[-1].date()),
        "head_and_shoulders_top": top,
        "head_and_shoulders_bottom": bottom,
        "complex_head_and_shoulders": _find_complex_head_and_shoulders(top, bottom),
        "dormant_bottom": _find_dormant_bottom(clean, target_column=target_column, timeframe=timeframe),
    }


def _find_head_and_shoulders(frame: pd.DataFrame, target_column: str, timeframe: str, kind: str) -> dict[str, Any]:
    close = pd.to_numeric(frame[target_column], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce") if "high" in frame.columns else close
    low = pd.to_numeric(frame["low"], errors="coerce") if "low" in frame.columns else close
    volume = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame.columns else None
    events = _confirmed_pivot_events(high=high, low=low, volume=volume)
    pattern_name = "HeadAndShouldersTop" if kind == "top" else "HeadAndShouldersBottom"
    if len(events) < 5:
        return _empty_pattern(pattern_name, timeframe, "NoPattern", "not enough confirmed pivots", len(frame))

    alternating = _alternating_events(events)
    target_types = ["high", "low", "high", "low", "high"] if kind == "top" else ["low", "high", "low", "high", "low"]
    lookback = PATTERN_LOOKBACK_BARS.get(timeframe, PATTERN_LOOKBACK_BARS["chart"])
    earliest_right_shoulder = max(0, len(frame) - lookback)
    candidates = []
    for index in range(0, len(alternating) - 4):
        sequence = alternating[index : index + 5]
        if [event["type"] for event in sequence] != target_types:
            continue
        if int(sequence[-1]["position"]) < earliest_right_shoulder:
            continue
        candidate = _score_head_and_shoulders_candidate(
            sequence=sequence,
            close=close,
            volume=volume,
            timeframe=timeframe,
            kind=kind,
        )
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return _empty_pattern(pattern_name, timeframe, "NoPattern", f"no valid {pattern_name} geometry", len(frame))
    return max(candidates, key=_pattern_rank)


def _score_head_and_shoulders_candidate(
    sequence: list[dict[str, Any]],
    close: pd.Series,
    volume: pd.Series | None,
    timeframe: str,
    kind: str,
) -> dict[str, Any] | None:
    first, neckline_1, head, neckline_2, right = sequence
    span = int(right["position"]) - int(first["position"])
    if span < 10:
        return None

    first_price = float(first["price"])
    head_price = float(head["price"])
    right_price = float(right["price"])
    neckline_1_price = float(neckline_1["price"])
    neckline_2_price = float(neckline_2["price"])
    if min(first_price, head_price, right_price, neckline_1_price, neckline_2_price) <= 0:
        return None

    if kind == "top":
        pattern_name = "HeadAndShouldersTop"
        head_prominence = min(head_price / first_price - 1.0, head_price / right_price - 1.0)
        trough_depth = min(first_price - neckline_1_price, right_price - neckline_2_price) / head_price
        neckline = _neckline_function(neckline_1, neckline_2)
        neckline_at_head = neckline(int(head["position"]))
        pattern_height = head_price - neckline_at_head if neckline_at_head else None
    else:
        pattern_name = "HeadAndShouldersBottom"
        head_prominence = min(first_price / head_price - 1.0, right_price / head_price - 1.0)
        trough_depth = min(neckline_1_price - first_price, neckline_2_price - right_price) / neckline_1_price
        neckline = _neckline_function(neckline_1, neckline_2)
        neckline_at_head = neckline(int(head["position"]))
        pattern_height = neckline_at_head - head_price if neckline_at_head else None

    shoulder_imbalance = abs(first_price - right_price) / max(first_price, right_price, head_price)
    neckline_imbalance = abs(neckline_1_price - neckline_2_price) / max(neckline_1_price, neckline_2_price, head_price)
    if head_prominence < MIN_HEAD_PROMINENCE_PCT:
        return None
    if shoulder_imbalance > MAX_SHOULDER_IMBALANCE_PCT:
        return None
    if trough_depth < 0.015 or pattern_height is None or pattern_height <= 0:
        return None

    latest_position = len(close) - 1
    latest_close = float(close.iloc[-1])
    latest_neckline = neckline(latest_position)
    if latest_neckline is None or latest_neckline <= 0:
        return None

    prior_trend = _prior_trend(close=close, first_position=int(first["position"]), timeframe=timeframe, kind=kind)
    if not prior_trend["valid_reversal_context"]:
        return None

    break_info = _find_neckline_break(
        close=close,
        neckline=neckline,
        start_position=int(right["confirmed_position"]),
        direction="down" if kind == "top" else "up",
    )
    retest = _neckline_retest_state(close=close, neckline=neckline, break_position=break_info.get("position"))
    break_neckline = break_info.get("neckline_price") or latest_neckline
    measured_objective = float(break_neckline - pattern_height) if kind == "top" else float(break_neckline + pattern_height)
    if measured_objective <= 0:
        measured_objective = None

    status = _pattern_status(
        kind=kind,
        break_state=break_info["state"],
        latest_close=latest_close,
        latest_neckline=latest_neckline,
        measured_objective=measured_objective,
        retest=retest,
    )
    volume_diagnostics = _volume_diagnostics(
        volume=volume,
        left_shoulder=first,
        head=head,
        right_shoulder=right,
        break_position=break_info.get("position"),
        kind=kind,
    )
    geometry_score = _geometry_score(
        head_prominence=head_prominence,
        shoulder_imbalance=shoulder_imbalance,
        trough_imbalance=neckline_imbalance,
        trough_depth=trough_depth,
    )
    if geometry_score < 0.45:
        return None
    volume_score = float(volume_diagnostics.get("score", 0.5))
    confirmation_score = _confirmation_score(status=status, latest_close=latest_close, latest_neckline=latest_neckline)
    trend_score = float(prior_trend.get("score", 0.5))
    score = float(0.45 * geometry_score + 0.25 * volume_score + 0.20 * confirmation_score + 0.10 * trend_score)

    latest_margin = latest_close / latest_neckline - 1.0
    objective_key = "measured_downside_pct" if kind == "top" else "measured_upside_pct"
    objective_return = measured_objective / latest_close - 1.0 if measured_objective and latest_close else None
    return {
        "pattern": pattern_name,
        "direction": "bearish" if kind == "top" else "bullish",
        "status": status,
        "timeframe": timeframe,
        "score": _finite_or_none(score),
        "latest_close": _finite_or_none(latest_close),
        "latest_neckline_margin_pct": _finite_or_none(latest_margin),
        "neckline_break_date": break_info.get("date"),
        "neckline_break_close": break_info.get("close"),
        "neckline_break_price": break_info.get("neckline_price"),
        "neckline_break_margin_pct": break_info.get("margin_pct"),
        "measured_objective": _finite_or_none(measured_objective),
        objective_key: _finite_or_none(objective_return),
        "objective_status": _objective_status(kind=kind, latest_close=latest_close, objective=measured_objective),
        "neckline_test": bool(abs(latest_margin) <= NECKLINE_TEST_PCT),
        "neckline_retest": retest,
        "pullback": retest if kind == "top" else None,
        "throwback": retest if kind == "bottom" else None,
        "prior_trend": prior_trend,
        "points": {
            "left_shoulder": _event_payload(first),
            "head": _event_payload(head),
            "right_shoulder": _event_payload(right),
            "low_1" if kind == "top" else "high_1": _event_payload(neckline_1),
            "low_2" if kind == "top" else "high_2": _event_payload(neckline_2),
        },
        "neckline": {
            "start_date": _date_string(neckline_1["date"]),
            "start_price": _finite_or_none(neckline_1_price),
            "second_point_date": _date_string(neckline_2["date"]),
            "second_point_price": _finite_or_none(neckline_2_price),
            "latest_date": _date_string(close.index[-1]),
            "latest_price": _finite_or_none(latest_neckline),
            "slope_per_bar": _finite_or_none(_neckline_slope(neckline_1, neckline_2)),
        },
        "geometry": {
            "span_bars": int(span),
            "head_prominence_pct": _finite_or_none(head_prominence),
            "shoulder_imbalance_pct": _finite_or_none(shoulder_imbalance),
            "neckline_point_imbalance_pct": _finite_or_none(neckline_imbalance),
            "shoulder_depth_pct": _finite_or_none(trough_depth),
            "score": _finite_or_none(geometry_score),
        },
        "volume_confirmation": volume_diagnostics,
    }


def _find_complex_head_and_shoulders(top: dict[str, Any], bottom: dict[str, Any]) -> dict[str, Any]:
    active = [pattern for pattern in (top, bottom) if pattern.get("status") not in {"NoPattern", "InsufficientData"}]
    if not active:
        timeframe = top.get("timeframe") or bottom.get("timeframe") or "unknown"
        return _empty_pattern("ComplexHeadAndShoulders", timeframe, "NoPattern", "no simple H&S base pattern to extend", 0)

    base = max(active, key=_pattern_rank)
    geometry = base.get("geometry", {})
    shoulder_imbalance = float(geometry.get("shoulder_imbalance_pct") or 1.0)
    neckline_imbalance = float(geometry.get("neckline_point_imbalance_pct") or 1.0)
    if shoulder_imbalance <= 0.08 and neckline_imbalance <= 0.08 and float(geometry.get("span_bars") or 0) >= 35:
        return {
            "pattern": "ComplexHeadAndShoulders",
            "status": "WarningOnly",
            "timeframe": base.get("timeframe"),
            "direction": base.get("direction"),
            "score": _finite_or_none(min(0.75, float(base.get("score") or 0.0) + 0.05)),
            "base_pattern": base.get("pattern"),
            "outer_neckline": base.get("neckline"),
            "reason": "Pattern is broad and symmetrical enough to monitor as a possible complex H&S; it is not an action blocker.",
        }
    return _empty_pattern("ComplexHeadAndShoulders", base.get("timeframe", "unknown"), "NoPattern", "base pattern is not broad/symmetrical enough", 0)


def _find_dormant_bottom(frame: pd.DataFrame, target_column: str, timeframe: str) -> dict[str, Any]:
    close = pd.to_numeric(frame[target_column], errors="coerce").dropna()
    volume = pd.to_numeric(frame["volume"], errors="coerce").reindex(close.index) if "volume" in frame.columns else None
    base_bars = DORMANT_BASE_BARS.get(timeframe, DORMANT_BASE_BARS["chart"])
    breakout_bars = DORMANT_BREAKOUT_BARS.get(timeframe, DORMANT_BREAKOUT_BARS["chart"])
    minimum_rows = base_bars + breakout_bars + max(20, base_bars // 2)
    if len(close) < minimum_rows:
        return _empty_pattern("DormantBottom", timeframe, "InsufficientData", "not enough rows for dormant-bottom window", len(close))

    base = close.iloc[-(base_bars + breakout_bars) : -breakout_bars]
    recent = close.iloc[-breakout_bars:]
    prior = close.iloc[: -(base_bars + breakout_bars)]
    base_mean = float(base.mean())
    if base_mean <= 0:
        return _empty_pattern("DormantBottom", timeframe, "NoPattern", "invalid base price", len(close))

    base_high = float(base.max())
    base_low = float(base.min())
    base_range_pct = (base_high - base_low) / base_mean
    prior_decline = base_mean / float(prior.iloc[0]) - 1.0 if len(prior) else 0.0
    recent_breakout = float(recent.max()) / base_high - 1.0 if base_high else 0.0
    latest_close = float(close.iloc[-1])
    recent_low_above_base_mid = float(recent.tail(max(2, breakout_bars // 3)).min()) >= base_mean

    volume_state = _dormant_volume_state(volume=volume, base_index=base.index, recent_index=recent.index, prior_index=prior.index)
    flat_base = base_range_pct <= 0.18
    large_prior_decline = prior_decline <= -0.20
    breakout = recent_breakout >= 0.08 and recent_low_above_base_mid
    base_is_quiet = volume_state.get("base_quiet") in {True, None}
    breakout_volume = volume_state.get("breakout_volume_expansion") in {True, None}

    if flat_base and large_prior_decline and breakout and base_is_quiet:
        status = "Breakout" if breakout_volume else "Candidate"
    elif flat_base and large_prior_decline:
        status = "BaseForming"
    else:
        return _empty_pattern("DormantBottom", timeframe, "NoPattern", "no extended quiet base after a large decline", len(close))

    score = 0.25
    score += 0.20 if flat_base else 0.0
    score += 0.20 if large_prior_decline else 0.0
    score += 0.20 if breakout else 0.0
    score += 0.15 if breakout_volume else 0.05 if breakout_volume is None else 0.0
    return {
        "pattern": "DormantBottom",
        "status": status,
        "timeframe": timeframe,
        "score": _finite_or_none(min(score, 1.0)),
        "optional": True,
        "decision_use": "context_only",
        "latest_close": _finite_or_none(latest_close),
        "base_start_date": _date_string(base.index[0]),
        "base_end_date": _date_string(base.index[-1]),
        "base_bars": int(len(base)),
        "base_low": _finite_or_none(base_low),
        "base_high": _finite_or_none(base_high),
        "base_range_pct": _finite_or_none(base_range_pct),
        "prior_decline_pct": _finite_or_none(prior_decline),
        "recent_breakout_pct": _finite_or_none(recent_breakout),
        "volume_confirmation": volume_state,
        "reason": "Optional dormant-bottom accumulation diagnostic; use for long-horizon review before relying on forecast signals.",
    }


def _prior_trend(close: pd.Series, first_position: int, timeframe: str, kind: str) -> dict[str, Any]:
    lookback = PRIOR_TREND_LOOKBACK_BARS.get(timeframe, PRIOR_TREND_LOOKBACK_BARS["chart"])
    threshold = PRIOR_TREND_MIN_RETURN.get(timeframe, PRIOR_TREND_MIN_RETURN["chart"])
    start = max(0, first_position - lookback)
    if first_position <= start:
        return {"state": "Unavailable", "return_pct": None, "valid_reversal_context": True, "score": 0.5}
    prior = close.iloc[start : first_position + 1].dropna()
    if len(prior) < 10:
        return {"state": "Unavailable", "return_pct": None, "valid_reversal_context": True, "score": 0.5}
    total_return = float(prior.iloc[-1] / prior.iloc[0] - 1.0)
    if kind == "top":
        valid = total_return >= threshold
        state = "PriorAdvance" if valid else "NoMaterialAdvance"
        score = _clip01(total_return / max(threshold * 2, 1e-9))
    else:
        valid = total_return <= -threshold
        state = "PriorDecline" if valid else "NoMaterialDecline"
        score = _clip01(abs(total_return) / max(threshold * 2, 1e-9))
    return {
        "state": state,
        "lookback_bars": int(len(prior)),
        "return_pct": _finite_or_none(total_return),
        "threshold_pct": _finite_or_none(threshold),
        "valid_reversal_context": bool(valid),
        "score": _finite_or_none(score),
    }


def _confirmed_pivot_events(
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series | None,
    left: int = PIVOT_LEFT_BARS,
    right: int = PIVOT_RIGHT_BARS,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    values_high = high.to_numpy(dtype=float)
    values_low = low.to_numpy(dtype=float)
    for position in range(left, len(high) - right):
        high_window = values_high[position - left : position + right + 1]
        low_window = values_low[position - left : position + right + 1]
        high_value = values_high[position]
        low_value = values_low[position]
        if np.isfinite(high_value) and high_value == np.nanmax(high_window):
            events.append(_pivot_event(high, volume, position, right, "high", high_value))
        if np.isfinite(low_value) and low_value == np.nanmin(low_window):
            events.append(_pivot_event(low, volume, position, right, "low", low_value))
    return sorted(events, key=lambda item: (int(item["position"]), 0 if item["type"] == "low" else 1))


def _pivot_event(
    series: pd.Series,
    volume: pd.Series | None,
    position: int,
    right: int,
    event_type: str,
    value: float,
) -> dict[str, Any]:
    confirmed_position = min(position + right, len(series) - 1)
    event_volume = None
    if volume is not None:
        volume_value = volume.iloc[position]
        if pd.notna(volume_value) and np.isfinite(float(volume_value)):
            event_volume = float(volume_value)
    return {
        "type": event_type,
        "date": pd.Timestamp(series.index[position]),
        "confirmed_date": pd.Timestamp(series.index[confirmed_position]),
        "position": int(position),
        "confirmed_position": int(confirmed_position),
        "price": float(value),
        "volume": event_volume,
    }


def _alternating_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alternating: list[dict[str, Any]] = []
    for event in events:
        if not alternating or alternating[-1]["type"] != event["type"]:
            alternating.append(event)
            continue
        previous = alternating[-1]
        if event["type"] == "high" and float(event["price"]) > float(previous["price"]):
            alternating[-1] = event
        elif event["type"] == "low" and float(event["price"]) < float(previous["price"]):
            alternating[-1] = event
    return alternating


def _neckline_function(point_1: dict[str, Any], point_2: dict[str, Any]) -> Any:
    pos_1 = int(point_1["position"])
    pos_2 = int(point_2["position"])
    price_1 = float(point_1["price"])
    price_2 = float(point_2["price"])
    if pos_2 == pos_1:
        return lambda _position: None
    slope = (price_2 - price_1) / (pos_2 - pos_1)

    def neckline(position: int) -> float:
        return float(price_1 + slope * (int(position) - pos_1))

    return neckline


def _neckline_slope(point_1: dict[str, Any], point_2: dict[str, Any]) -> float | None:
    pos_1 = int(point_1["position"])
    pos_2 = int(point_2["position"])
    if pos_2 == pos_1:
        return None
    return float((float(point_2["price"]) - float(point_1["price"])) / (pos_2 - pos_1))


def _find_neckline_break(close: pd.Series, neckline: Any, start_position: int, direction: str) -> dict[str, Any]:
    if start_position >= len(close):
        return {"state": "Unconfirmed", "position": None, "date": None, "close": None, "neckline_price": None, "margin_pct": None}
    for position in range(max(0, start_position), len(close)):
        close_value = close.iloc[position]
        neckline_value = neckline(position)
        if pd.isna(close_value) or neckline_value is None or neckline_value <= 0:
            continue
        margin = float(close_value) / float(neckline_value) - 1.0
        if (direction == "down" and margin <= -NECKLINE_BREAK_PCT) or (direction == "up" and margin >= NECKLINE_BREAK_PCT):
            return {
                "state": "Confirmed",
                "position": int(position),
                "date": _date_string(close.index[position]),
                "close": _finite_or_none(close_value),
                "neckline_price": _finite_or_none(neckline_value),
                "margin_pct": _finite_or_none(margin),
            }
    return {"state": "Unconfirmed", "position": None, "date": None, "close": None, "neckline_price": None, "margin_pct": None}


def _neckline_retest_state(close: pd.Series, neckline: Any, break_position: int | None) -> dict[str, Any]:
    if break_position is None:
        return {"state": "NotApplicable", "latest_retest_to_neckline": False, "last_retest_date": None}
    last_retest_date = None
    latest_retest = False
    for position in range(int(break_position) + 1, len(close)):
        neckline_value = neckline(position)
        close_value = close.iloc[position]
        if pd.isna(close_value) or neckline_value is None or neckline_value <= 0:
            continue
        margin = float(close_value) / float(neckline_value) - 1.0
        if abs(margin) <= NECKLINE_TEST_PCT:
            last_retest_date = _date_string(close.index[position])
            if position >= len(close) - 5:
                latest_retest = True
    return {
        "state": "Observed" if last_retest_date else "NotObserved",
        "latest_retest_to_neckline": bool(latest_retest),
        "last_retest_date": last_retest_date,
    }


def _pattern_status(
    kind: str,
    break_state: str,
    latest_close: float,
    latest_neckline: float,
    measured_objective: float | None,
    retest: dict[str, Any],
) -> str:
    if break_state != "Confirmed":
        return "Candidate"
    if kind == "top":
        if latest_close > latest_neckline * (1.0 + NECKLINE_TEST_PCT):
            return "Failed"
        if measured_objective is not None and latest_close <= measured_objective:
            return "ObjectiveReached"
        if retest["latest_retest_to_neckline"]:
            return "PullbackToNeckline"
        return "Confirmed"
    if latest_close < latest_neckline * (1.0 - NECKLINE_TEST_PCT):
        return "Failed"
    if measured_objective is not None and latest_close >= measured_objective:
        return "ObjectiveReached"
    if retest["latest_retest_to_neckline"]:
        return "ThrowbackToNeckline"
    return "Confirmed"


def _volume_diagnostics(
    volume: pd.Series | None,
    left_shoulder: dict[str, Any],
    head: dict[str, Any],
    right_shoulder: dict[str, Any],
    break_position: int | None,
    kind: str,
) -> dict[str, Any]:
    if volume is None or volume.dropna().empty:
        return {"state": "Unavailable", "score": 0.5}

    rolling = volume.rolling(20, min_periods=5).mean()
    left_volume = _local_volume(volume, int(left_shoulder["position"]))
    head_volume = _local_volume(volume, int(head["position"]))
    right_volume = _local_volume(volume, int(right_shoulder["position"]))
    break_volume = _local_volume(volume, int(break_position)) if break_position is not None else None
    right_quieter = _is_quieter(right_volume, left_volume, head_volume)
    break_expands = None
    if break_position is not None and break_volume is not None:
        baseline = rolling.iloc[int(break_position)]
        break_expands = bool(pd.notna(baseline) and float(baseline) > 0 and break_volume >= float(baseline) * 1.1)
    score = 0.5
    if right_quieter is True:
        score += 0.25
    if kind == "bottom":
        if break_expands is True:
            score += 0.30
        elif break_expands is False:
            score -= 0.20
    else:
        if break_expands is True:
            score += 0.15
        elif break_position is None:
            score += 0.05

    return {
        "state": "Measured",
        "left_shoulder_volume": _finite_or_none(left_volume),
        "head_volume": _finite_or_none(head_volume),
        "right_shoulder_volume": _finite_or_none(right_volume),
        "break_volume": _finite_or_none(break_volume),
        "right_shoulder_quieter": right_quieter,
        "break_volume_expansion": break_expands,
        "confirms_distribution": bool(kind == "top" and right_quieter is True and (break_expands is True or break_expands is None)),
        "confirms_accumulation": bool(kind == "bottom" and break_expands is True),
        "score": _finite_or_none(min(max(score, 0.0), 1.0)),
    }


def _dormant_volume_state(
    volume: pd.Series | None,
    base_index: pd.Index,
    recent_index: pd.Index,
    prior_index: pd.Index,
) -> dict[str, Any]:
    if volume is None or volume.dropna().empty:
        return {"state": "Unavailable", "base_quiet": None, "breakout_volume_expansion": None}
    base_volume = volume.reindex(base_index)
    recent_volume = volume.reindex(recent_index)
    prior_volume = volume.reindex(prior_index)
    base_median = float(base_volume.median()) if base_volume.notna().any() else None
    recent_median = float(recent_volume.median()) if recent_volume.notna().any() else None
    prior_median = float(prior_volume.tail(max(10, len(base_index) // 2)).median()) if prior_volume.notna().any() else None
    base_quiet = None
    if base_median is not None and prior_median is not None and prior_median > 0:
        base_quiet = bool(base_median <= prior_median * 0.75)
    breakout_expansion = None
    if base_median is not None and recent_median is not None and base_median > 0:
        breakout_expansion = bool(recent_median >= base_median * 1.5)
    zero_volume_ratio = float((base_volume.fillna(0.0) <= 0.0).mean()) if len(base_volume) else 0.0
    return {
        "state": "Measured",
        "base_median_volume": _finite_or_none(base_median),
        "recent_median_volume": _finite_or_none(recent_median),
        "prior_median_volume": _finite_or_none(prior_median),
        "base_quiet": base_quiet,
        "breakout_volume_expansion": breakout_expansion,
        "base_zero_volume_ratio": _finite_or_none(zero_volume_ratio),
    }


def _local_volume(volume: pd.Series, position: int, radius: int = 1) -> float | None:
    start = max(0, position - radius)
    end = min(len(volume), position + radius + 1)
    value = volume.iloc[start:end].mean()
    return float(value) if pd.notna(value) and np.isfinite(float(value)) else None


def _is_quieter(right_volume: float | None, left_volume: float | None, head_volume: float | None) -> bool | None:
    reference = max(value for value in (left_volume, head_volume) if value is not None) if any(value is not None for value in (left_volume, head_volume)) else None
    if right_volume is None or reference is None or reference <= 0:
        return None
    return bool(right_volume <= reference * 0.9)


def _geometry_score(
    head_prominence: float,
    shoulder_imbalance: float,
    trough_imbalance: float,
    trough_depth: float,
) -> float:
    prominence_score = _clip01((head_prominence - MIN_HEAD_PROMINENCE_PCT) / 0.10)
    shoulder_score = 1.0 - _clip01(shoulder_imbalance / MAX_SHOULDER_IMBALANCE_PCT)
    trough_score = 1.0 - _clip01(trough_imbalance / 0.16)
    depth_score = _clip01(trough_depth / 0.10)
    return float(0.35 * prominence_score + 0.30 * shoulder_score + 0.15 * trough_score + 0.20 * depth_score)


def _confirmation_score(status: str, latest_close: float, latest_neckline: float) -> float:
    if status in {"Confirmed", "PullbackToNeckline", "ThrowbackToNeckline"}:
        return 1.0
    if status == "ObjectiveReached":
        return 0.75
    if status == "Failed":
        return 0.15
    if latest_neckline > 0 and abs(latest_close / latest_neckline - 1.0) <= NECKLINE_TEST_PCT:
        return 0.45
    return 0.25


def _objective_status(kind: str, latest_close: float, objective: float | None) -> str:
    if objective is None:
        return "Unavailable"
    if kind == "top" and latest_close <= objective:
        return "Reached"
    if kind == "bottom" and latest_close >= objective:
        return "Reached"
    return "Open"


def _choose_preferred(timeframes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    patterns = []
    for pattern_key in ("head_and_shoulders_top", "head_and_shoulders_bottom"):
        for payload in timeframes.values():
            pattern = payload.get(pattern_key, {})
            if pattern.get("status") not in {"NoPattern", "InsufficientData"}:
                patterns.append(pattern)
    if patterns:
        return _preferred_payload(max(patterns, key=_pattern_rank))
    top = _choose_pattern(timeframes, "head_and_shoulders_top")
    if top.get("status") != "NoPattern":
        return top
    return _choose_pattern(timeframes, "head_and_shoulders_bottom")


def _choose_pattern(timeframes: dict[str, dict[str, Any]], pattern_key: str) -> dict[str, Any]:
    patterns = [
        payload.get(pattern_key, {})
        for payload in timeframes.values()
        if isinstance(payload.get(pattern_key), dict)
    ]
    actionable = [pattern for pattern in patterns if pattern.get("status") not in {"NoPattern", "InsufficientData"}]
    if actionable:
        return _preferred_payload(max(actionable, key=_pattern_rank))
    for timeframe in ("monthly", "weekly", "daily"):
        pattern = timeframes.get(timeframe, {}).get(pattern_key, {})
        if pattern:
            return _preferred_payload(pattern)
    pattern_name = "HeadAndShouldersTop" if pattern_key.endswith("_top") else "HeadAndShouldersBottom"
    return _empty_pattern(pattern_name, "unknown", "NoPattern", "pattern unavailable", 0)


def _choose_optional(timeframes: dict[str, dict[str, Any]], pattern_key: str) -> dict[str, Any]:
    patterns = [
        payload.get(pattern_key, {})
        for payload in timeframes.values()
        if isinstance(payload.get(pattern_key), dict)
    ]
    actionable = [pattern for pattern in patterns if pattern.get("status") not in {"NoPattern", "InsufficientData"}]
    if actionable:
        return _preferred_payload(max(actionable, key=_optional_rank))
    for timeframe in ("monthly", "weekly", "daily"):
        pattern = timeframes.get(timeframe, {}).get(pattern_key, {})
        if pattern:
            return _preferred_payload(pattern)
    return {}


def _optional_by_timeframe(timeframes: dict[str, dict[str, Any]], pattern_key: str) -> dict[str, Any]:
    return {
        timeframe: payload.get(pattern_key, {})
        for timeframe, payload in timeframes.items()
    }


def _pattern_rank(pattern: dict[str, Any]) -> tuple[float, float, str]:
    status_rank = {
        "PullbackToNeckline": 5.0,
        "ThrowbackToNeckline": 5.0,
        "Confirmed": 4.0,
        "Candidate": 2.0,
        "ObjectiveReached": 1.0,
        "Failed": 0.5,
        "NoPattern": 0.0,
        "InsufficientData": 0.0,
    }.get(str(pattern.get("status")), 0.0)
    timeframe_bonus = {"monthly": 0.30, "weekly": 0.15, "daily": 0.0}.get(str(pattern.get("timeframe")), 0.0)
    score = float(pattern.get("score") or 0.0)
    right_date = pattern.get("points", {}).get("right_shoulder", {}).get("date") or ""
    return (status_rank + timeframe_bonus, score, str(right_date))


def _optional_rank(pattern: dict[str, Any]) -> tuple[float, float, str]:
    status_rank = {
        "Breakout": 4.0,
        "WarningOnly": 3.0,
        "Candidate": 2.0,
        "BaseForming": 1.0,
        "NoPattern": 0.0,
        "InsufficientData": 0.0,
    }.get(str(pattern.get("status")), 0.0)
    timeframe_bonus = {"monthly": 0.30, "weekly": 0.15, "daily": 0.0}.get(str(pattern.get("timeframe")), 0.0)
    score = float(pattern.get("score") or 0.0)
    date = pattern.get("base_end_date") or pattern.get("timeframe") or ""
    return (status_rank + timeframe_bonus, score, str(date))


def _preferred_payload(pattern: dict[str, Any]) -> dict[str, Any]:
    if not pattern:
        return {"pattern": "NoPattern", "status": "NoPattern"}
    keys = [
        "pattern",
        "direction",
        "status",
        "timeframe",
        "score",
        "optional",
        "decision_use",
        "latest_close",
        "latest_neckline_margin_pct",
        "neckline_break_date",
        "neckline_break_close",
        "neckline_break_price",
        "neckline_break_margin_pct",
        "measured_objective",
        "measured_downside_pct",
        "measured_upside_pct",
        "objective_status",
        "neckline_test",
        "neckline_retest",
        "pullback",
        "throwback",
        "prior_trend",
        "points",
        "neckline",
        "geometry",
        "volume_confirmation",
        "base_pattern",
        "outer_neckline",
        "base_start_date",
        "base_end_date",
        "base_bars",
        "base_low",
        "base_high",
        "base_range_pct",
        "prior_decline_pct",
        "recent_breakout_pct",
        "reason",
    ]
    return {key: pattern[key] for key in keys if key in pattern}


def _empty_pattern(pattern: str, timeframe: str, status: str, reason: str, rows: int) -> dict[str, Any]:
    return {
        "pattern": pattern,
        "status": status,
        "timeframe": timeframe,
        "rows": int(rows),
        "reason": reason,
    }


def _event_payload(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": _date_string(event["date"]),
        "confirmed_date": _date_string(event["confirmed_date"]),
        "price": _finite_or_none(event["price"]),
        "volume": _finite_or_none(event.get("volume")),
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
