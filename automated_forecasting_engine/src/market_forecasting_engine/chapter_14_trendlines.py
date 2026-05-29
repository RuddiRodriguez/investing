from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PIVOT_LEFT_BARS = 3
PIVOT_RIGHT_BARS = 3
MIN_PATTERN_BARS = 70
LOOKBACK_BARS = {"daily": 504, "weekly": 260, "monthly": 180, "chart": 504}
MIN_PIVOT_SPACING = {"daily": 8, "weekly": 4, "monthly": 3, "chart": 8}
TOUCH_TOLERANCE_PCT = {"daily": 0.015, "weekly": 0.024, "monthly": 0.035, "chart": 0.015}
DECISIVE_PENETRATION_PCT = 0.03
BORDERLINE_PENETRATION_PCT = 0.02
RETURN_LINE_FAILURE_PCT = 0.035
VOLUME_CONFIRMATION_RATIO = 1.25


def analyze_chapter_14_trendlines(prices: pd.DataFrame, target_column: str = "close") -> dict[str, Any]:
    """Analyze Edwards/Magee Chapter 14 trendlines, channels, and fan lines."""

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
    return {
        "principle": (
            "Chapter 14 treats trendlines as pivot-confirmed delimiters of an intermediate trend. "
            "A decisive close beyond a valid line, normally around 3%, is more important than an intraday shakeout."
        ),
        "primary_timeframe": "weekly",
        "secondary_timeframe": "daily",
        "decision_rule": (
            "Use authoritative trendline breaks, channel deterioration, pullbacks to broken lines, and corrective fan lines "
            "as governance context around the model forecast."
        ),
        "preferred": _choose_preferred(timeframes, "preferred_trendline"),
        "trendlines": {
            "preferred": _choose_preferred(timeframes, "preferred_trendline"),
            "strongest": _collect_strongest(timeframes, "trendlines"),
        },
        "channels": {
            "preferred": _choose_preferred(timeframes, "channel"),
            "timeframes": {
                name: payload.get("channel", {})
                for name, payload in timeframes.items()
            },
        },
        "fan_lines": {
            "preferred": _choose_preferred(timeframes, "fan_lines"),
            "timeframes": {
                name: payload.get("fan_lines", {})
                for name, payload in timeframes.items()
            },
        },
        "timeframes": timeframes,
        "technical_method_card": chapter_14_trendlines_method_card(target_column=target),
    }


def latest_chapter_14_trendlines(
    prices: pd.DataFrame,
    target_column: str = "close",
    timeframe: str = "chart",
) -> dict[str, Any]:
    """Return Chapter 14 diagnostics for a plotted bar set."""

    return _analyze_timeframe(
        frame=prices,
        timeframe=timeframe,
        target_column=target_column.lower(),
    )


def chapter_14_trendlines_method_card(target_column: str = "close") -> dict[str, Any]:
    return {
        "name": "edwards_magee_chapter_14_trendlines_channels",
        "version": "chapter_14_alignment_v1",
        "target_column": target_column.lower(),
        "implemented_controls": [
            "pivot_confirmed_basic_trendlines",
            "trendline_authority_score",
            "three_percent_decisive_penetration",
            "volume_confirmed_borderline_penetration",
            "intraday_shakeout_warning",
            "double_trendline_outer_line_test",
            "return_line_trend_channel",
            "failure_to_reach_return_line",
            "pullback_to_broken_trendline",
            "three_fan_principle_for_corrective_moves",
        ],
        "authority_inputs": {
            "touches": "more confirmed pivot contacts increase authority",
            "duration": "longer lines from independent pivots carry more weight",
            "slope": "very steep lines are downgraded because they are easier to break without ending the trend",
            "confirmation": "third and fourth touches are explicit authority evidence",
        },
        "penetration_rules": {
            "decisive": "close beyond the basic or outer double trendline by roughly 3%",
            "borderline": "2% close beyond the line with expanded turnover",
            "shakeout": "intraday penetration without a confirming close",
            "pullback": "return to the broken line from the other side without reclaiming it",
        },
        "decision_use": (
            "Decisive uptrend-line breaks can block fresh Buy actions; decisive downtrend-line breaks can block fresh Sell actions. "
            "Active authoritative lines and fan-line reversals provide supporting or warning evidence."
        ),
    }


def _analyze_timeframe(frame: pd.DataFrame, timeframe: str, target_column: str) -> dict[str, Any]:
    clean = frame.copy().sort_index()
    if clean.empty or target_column not in clean.columns:
        return _empty_timeframe(timeframe, "InsufficientData", "missing target column", len(clean))
    clean.index = pd.DatetimeIndex(clean.index)
    clean = clean.loc[pd.to_numeric(clean[target_column], errors="coerce").notna()]
    if len(clean) < MIN_PATTERN_BARS:
        return _empty_timeframe(timeframe, "InsufficientData", "not enough bars for Chapter 14 trendline analysis", len(clean))

    lookback = min(LOOKBACK_BARS.get(timeframe, LOOKBACK_BARS["chart"]), len(clean))
    scope = clean.tail(lookback)
    trendlines = _build_trendlines(scope, timeframe=timeframe, target_column=target_column)
    preferred = _choose_timeframe_line(trendlines, timeframe)
    channel = _build_channel(scope, preferred, timeframe=timeframe, target_column=target_column)
    fan_lines = _build_fan_lines(scope, timeframe=timeframe, target_column=target_column)
    return {
        "state": "Measured",
        "timeframe": timeframe,
        "rows": int(len(clean)),
        "lookback_bars": int(lookback),
        "start_date": _date_string(clean.index[0]),
        "end_date": _date_string(clean.index[-1]),
        "latest_close": _finite_or_none(clean[target_column].iloc[-1]),
        "trendlines": [_preferred_payload(line) for line in trendlines[:6]],
        "preferred_trendline": _preferred_payload(preferred),
        "channel": _preferred_payload(channel),
        "fan_lines": _preferred_payload(fan_lines),
    }


def _build_trendlines(frame: pd.DataFrame, timeframe: str, target_column: str) -> list[dict[str, Any]]:
    close = pd.to_numeric(frame[target_column], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce") if "high" in frame.columns else close
    low = pd.to_numeric(frame["low"], errors="coerce") if "low" in frame.columns else close
    volume = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame.columns else pd.Series(np.nan, index=frame.index)
    pivot_high, pivot_low = _confirmed_pivots(high, low)
    low_events = _pivot_events(frame, pivot_low, "low")
    high_events = _pivot_events(frame, pivot_high, "high")
    candidates: list[dict[str, Any]] = []
    candidates.extend(
        _candidate_lines(
            frame=frame,
            close=close,
            high=high,
            low=low,
            volume=volume,
            pivot_events=low_events,
            kind="uptrend",
            timeframe=timeframe,
        )
    )
    candidates.extend(
        _candidate_lines(
            frame=frame,
            close=close,
            high=high,
            low=low,
            volume=volume,
            pivot_events=high_events,
            kind="downtrend",
            timeframe=timeframe,
        )
    )
    return sorted(candidates, key=_line_rank, reverse=True)


def _candidate_lines(
    frame: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    pivot_events: list[dict[str, Any]],
    kind: str,
    timeframe: str,
) -> list[dict[str, Any]]:
    if len(pivot_events) < 2:
        return []
    spacing = MIN_PIVOT_SPACING.get(timeframe, MIN_PIVOT_SPACING["chart"])
    tolerance = TOUCH_TOLERANCE_PCT.get(timeframe, TOUCH_TOLERANCE_PCT["chart"])
    events = pivot_events[-24:]
    candidates: list[dict[str, Any]] = []
    for first_index, first in enumerate(events[:-1]):
        for second in events[first_index + 1 :]:
            if int(second["position"]) - int(first["position"]) < spacing:
                continue
            if kind == "uptrend" and float(second["price"]) <= float(first["price"]) * 1.002:
                continue
            if kind == "downtrend" and float(second["price"]) >= float(first["price"]) * 0.998:
                continue
            line = _line_from_points(first, second)
            if line is None:
                continue
            payload = _line_payload(
                frame=frame,
                close=close,
                high=high,
                low=low,
                volume=volume,
                pivot_events=events,
                line=line,
                first=first,
                second=second,
                kind=kind,
                timeframe=timeframe,
                tolerance=tolerance,
            )
            if payload.get("authority_score", 0.0) >= 0.25:
                candidates.append(payload)
    return candidates


def _line_payload(
    frame: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    pivot_events: list[dict[str, Any]],
    line: dict[str, float],
    first: dict[str, Any],
    second: dict[str, Any],
    kind: str,
    timeframe: str,
    tolerance: float,
) -> dict[str, Any]:
    latest_position = len(frame) - 1
    latest_close = float(close.iloc[-1])
    current_value = _line_value(line, latest_position)
    touches = _line_touches(pivot_events, line, tolerance=tolerance, start_position=int(first["position"]))
    duration = latest_position - int(first["position"])
    volume_ratio = _latest_volume_ratio(volume)
    break_info = _break_info(
        frame=frame,
        close=close,
        high=high,
        low=low,
        line=line,
        start_position=int(second["position"]) + 1,
        kind=kind,
        volume_ratio=volume_ratio,
    )
    double_trendline = _double_trendline(
        close=close,
        pivot_events=pivot_events,
        line=line,
        kind=kind,
        tolerance=tolerance,
        start_position=int(first["position"]),
    )
    if double_trendline.get("status") == "DoubleTrendline":
        outer_line = double_trendline.get("line", {})
        outer_intercept = _safe_float(outer_line.get("intercept"))
        outer_slope = _safe_float(outer_line.get("slope_log_per_bar"))
        if outer_intercept is not None and outer_slope is not None:
            outer = {"intercept": outer_intercept, "slope": outer_slope}
            outer_line.update(
                {
                    "start_date": _date_string(frame.index[int(first["position"])]),
                    "end_date": _date_string(frame.index[-1]),
                    "start_value": _finite_or_none(_line_value(outer, int(first["position"]))),
                    "current_value": _finite_or_none(_line_value(outer, latest_position)),
                }
            )
    status = str(break_info["status"])
    effective_decisive_break = bool(break_info["effective_decisive_break"])
    break_scope = "basic_line"
    if double_trendline.get("status") == "DoubleTrendline" and status in {"DecisiveBreak", "BorderlineBreak"}:
        if double_trendline.get("outer_decisive_break"):
            break_scope = "outer_double_line"
        else:
            status = "InnerLineBreak"
            effective_decisive_break = False
            break_scope = "inner_line_only"

    direction = _line_direction(kind, status)
    authority = _authority_score(
        touch_count=len(touches),
        duration_bars=duration,
        lookback_bars=len(frame),
        slope_pct_per_bar=float(np.expm1(line["slope"])),
        pivot_spacing=int(second["position"]) - int(first["position"]),
        confirmed_touches=len(touches) >= 3,
        double_trendline=double_trendline.get("status") == "DoubleTrendline",
    )
    return {
        "pattern": "Chapter14Trendline",
        "status": status,
        "direction": direction,
        "kind": kind,
        "timeframe": timeframe,
        "score": _finite_or_none(authority),
        "authority_score": _finite_or_none(authority),
        "effective_decisive_break": bool(effective_decisive_break),
        "break_scope": break_scope,
        "latest_close": _finite_or_none(latest_close),
        "current_value": _finite_or_none(current_value),
        "distance_to_line_pct": _finite_or_none(_distance_to_line(latest_close, current_value, kind)),
        "latest_penetration_pct": _finite_or_none(break_info["latest_penetration_pct"]),
        "volume_ratio": _finite_or_none(volume_ratio),
        "volume_confirmed": bool(volume_ratio is not None and volume_ratio >= VOLUME_CONFIRMATION_RATIO),
        "first_break": break_info.get("first_break"),
        "pullback": break_info.get("pullback", {}),
        "intraday_shakeout": bool(break_info.get("intraday_shakeout")),
        "touch_count": int(len(touches)),
        "touches": touches[-6:],
        "duration_bars": int(duration),
        "anchor_points": [
            _anchor_payload(first),
            _anchor_payload(second),
        ],
        "line": {
            "start_date": _date_string(frame.index[int(first["position"])]),
            "second_date": _date_string(frame.index[int(second["position"])]),
            "end_date": _date_string(frame.index[-1]),
            "start_value": _finite_or_none(_line_value(line, int(first["position"]))),
            "second_value": _finite_or_none(_line_value(line, int(second["position"]))),
            "current_value": _finite_or_none(current_value),
            "slope_log_per_bar": _finite_or_none(line["slope"]),
            "slope_pct_per_bar": _finite_or_none(np.expm1(line["slope"])),
            "intercept": _finite_or_none(line["intercept"]),
        },
        "double_trendline": double_trendline,
        "reliability_notes": _line_notes(kind, status, authority, double_trendline, break_info),
    }


def _break_info(
    frame: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    line: dict[str, float],
    start_position: int,
    kind: str,
    volume_ratio: float | None,
) -> dict[str, Any]:
    latest_position = len(frame) - 1
    current_value = _line_value(line, latest_position)
    latest_close = float(close.iloc[-1])
    latest_penetration = _penetration_pct(latest_close, current_value, kind)
    volume_confirmed = volume_ratio is not None and volume_ratio >= VOLUME_CONFIRMATION_RATIO
    first_break = None
    for position in range(max(0, start_position), len(close)):
        value = _line_value(line, position)
        penetration = _penetration_pct(float(close.iloc[position]), value, kind)
        if penetration >= DECISIVE_PENETRATION_PCT:
            first_break = {
                "date": _date_string(frame.index[position]),
                "position": int(position),
                "close": _finite_or_none(close.iloc[position]),
                "line_value": _finite_or_none(value),
                "penetration_pct": _finite_or_none(penetration),
            }
            break
    intraday_shakeout = _intraday_shakeout(high, low, close, line, kind, latest_position)
    if latest_penetration >= DECISIVE_PENETRATION_PCT:
        status = "DecisiveBreak"
        effective_decisive_break = True
    elif latest_penetration >= BORDERLINE_PENETRATION_PCT and volume_confirmed:
        status = "BorderlineBreak"
        effective_decisive_break = True
    elif intraday_shakeout:
        status = "ShakeoutWarning"
        effective_decisive_break = False
    elif first_break is not None:
        pullback = _pullback_to_broken_line(frame, close, high, low, line, kind, int(first_break["position"]))
        if pullback.get("observed"):
            status = "PullbackToBrokenLine"
        else:
            status = "Broken"
        effective_decisive_break = False
        return {
            "status": status,
            "effective_decisive_break": effective_decisive_break,
            "latest_penetration_pct": latest_penetration,
            "first_break": first_break,
            "pullback": pullback,
            "intraday_shakeout": intraday_shakeout,
        }
    else:
        status = "Active"
        effective_decisive_break = False
    pullback = _pullback_to_broken_line(frame, close, high, low, line, kind, int(first_break["position"])) if first_break else {}
    return {
        "status": status,
        "effective_decisive_break": effective_decisive_break,
        "latest_penetration_pct": latest_penetration,
        "first_break": first_break,
        "pullback": pullback,
        "intraday_shakeout": intraday_shakeout,
    }


def _build_channel(frame: pd.DataFrame, trendline: dict[str, Any], timeframe: str, target_column: str) -> dict[str, Any]:
    if trendline.get("pattern") != "Chapter14Trendline":
        return _empty_pattern("Chapter14Channel", timeframe, "NoPattern", "no valid basic trendline")
    close = pd.to_numeric(frame[target_column], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce") if "high" in frame.columns else close
    low = pd.to_numeric(frame["low"], errors="coerce") if "low" in frame.columns else close
    line_data = trendline.get("line", {})
    slope = _safe_float(line_data.get("slope_log_per_bar"))
    intercept = _safe_float(line_data.get("intercept"))
    if slope is None or intercept is None:
        return _empty_pattern("Chapter14Channel", timeframe, "NoPattern", "basic trendline missing line parameters")
    line = {"slope": slope, "intercept": intercept}
    kind = str(trendline.get("kind"))
    first_position = int(trendline.get("anchor_points", [{}])[0].get("position", 0))
    x = np.arange(len(frame), dtype=float)
    basic_log = intercept + slope * x
    if kind == "uptrend":
        opposite = np.log(high.replace(0, np.nan).to_numpy(dtype=float))
        residual = opposite - basic_log
    else:
        opposite = np.log(low.replace(0, np.nan).to_numpy(dtype=float))
        residual = basic_log - opposite
    scoped = residual[first_position:]
    scoped = scoped[np.isfinite(scoped) & (scoped > 0)]
    if len(scoped) < 8:
        return _empty_pattern("Chapter14Channel", timeframe, "NoPattern", "not enough opposite pivots to establish return line")
    offset = float(np.nanquantile(scoped, 0.88))
    if offset <= TOUCH_TOLERANCE_PCT.get(timeframe, TOUCH_TOLERANCE_PCT["chart"]) * 1.2:
        return _empty_pattern("Chapter14Channel", timeframe, "NoPattern", "return-line offset too small")

    latest_position = len(frame) - 1
    basic_current = _line_value(line, latest_position)
    return_current = float(np.exp((intercept + slope * latest_position) + offset)) if kind == "uptrend" else float(np.exp((intercept + slope * latest_position) - offset))
    latest_close = float(close.iloc[-1])
    if kind == "uptrend":
        position = (np.log(latest_close) - np.log(basic_current)) / offset if latest_close > 0 and basic_current > 0 else np.nan
        return_break = latest_close > return_current * (1.0 + DECISIVE_PENETRATION_PCT)
    else:
        position = (np.log(basic_current) - np.log(latest_close)) / offset if latest_close > 0 and basic_current > 0 else np.nan
        return_break = latest_close < return_current * (1.0 - DECISIVE_PENETRATION_PCT)
    failure = _return_line_failure(frame, high, low, line, offset, kind, first_position)
    status = "ReturnLineBreakout" if return_break else "ReturnLineFailure" if failure.get("failure_to_reach") else "ActiveChannel"
    direction = "bullish_acceleration" if kind == "uptrend" and return_break else "bearish_acceleration" if kind == "downtrend" and return_break else "deterioration_warning" if failure.get("failure_to_reach") else "trend_context"
    return {
        "pattern": "Chapter14Channel",
        "status": status,
        "direction": direction,
        "kind": kind,
        "timeframe": timeframe,
        "score": _finite_or_none(float(trendline.get("authority_score") or 0.0) * (0.85 if failure.get("failure_to_reach") else 1.0)),
        "basic_line_current": _finite_or_none(basic_current),
        "return_line_current": _finite_or_none(return_current),
        "channel_width_pct": _finite_or_none(float(np.expm1(offset))),
        "channel_position": _finite_or_none(position),
        "failure_to_reach_return_line": bool(failure.get("failure_to_reach")),
        "failure_margin_pct": _finite_or_none(failure.get("failure_margin_pct")),
        "failure_projection": {
            "principle": "Chapter 14 notes that the failure margin can approximate the later break beyond the basic line.",
            "projected_basic_line_penetration_pct": _finite_or_none(failure.get("failure_margin_pct")),
        },
        "line": {
            "start_date": line_data.get("start_date"),
            "end_date": line_data.get("end_date"),
            "basic_start_value": line_data.get("start_value"),
            "basic_current_value": _finite_or_none(basic_current),
            "return_start_value": _finite_or_none(float(np.exp(intercept + offset)) if kind == "uptrend" else float(np.exp(intercept - offset))),
            "return_current_value": _finite_or_none(return_current),
            "slope_log_per_bar": _finite_or_none(slope),
            "intercept": _finite_or_none(intercept),
            "return_intercept": _finite_or_none(intercept + offset if kind == "uptrend" else intercept - offset),
        },
        "reliability_notes": _channel_notes(status, kind, failure),
    }


def _build_fan_lines(frame: pd.DataFrame, timeframe: str, target_column: str) -> dict[str, Any]:
    close = pd.to_numeric(frame[target_column], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce") if "high" in frame.columns else close
    low = pd.to_numeric(frame["low"], errors="coerce") if "low" in frame.columns else close
    pivot_high, pivot_low = _confirmed_pivots(high, low)
    high_events = _pivot_events(frame, pivot_high, "high")
    low_events = _pivot_events(frame, pivot_low, "low")
    bullish = _fan_candidate(frame, close, high_events, "bullish", timeframe)
    bearish = _fan_candidate(frame, close, low_events, "bearish", timeframe)
    candidates = [
        item
        for item in (bullish, bearish)
        if item.get("status") not in {None, "NoPattern", "InsufficientData"}
    ]
    if not candidates:
        return _empty_pattern("Chapter14FanLines", timeframe, "NoPattern", "no valid three-fan corrective structure")
    return _preferred_payload(max(candidates, key=_fan_rank))


def _fan_candidate(
    frame: pd.DataFrame,
    close: pd.Series,
    events: list[dict[str, Any]],
    direction: str,
    timeframe: str,
) -> dict[str, Any]:
    if len(events) < 4:
        return _empty_pattern("Chapter14FanLines", timeframe, "NoPattern", "not enough pivots for fan lines")
    scoped = events[-18:]
    if direction == "bullish":
        origin = max(scoped[:-3], key=lambda item: float(item["price"]))
        later = [event for event in scoped if int(event["position"]) > int(origin["position"]) and float(event["price"]) < float(origin["price"]) * 0.995]
    else:
        origin = min(scoped[:-3], key=lambda item: float(item["price"]))
        later = [event for event in scoped if int(event["position"]) > int(origin["position"]) and float(event["price"]) > float(origin["price"]) * 1.005]
    if len(later) < 3:
        return _empty_pattern("Chapter14FanLines", timeframe, "NoPattern", "three fan pivots unavailable")
    fan_points = later[:3]
    lines = []
    for point in fan_points:
        line = _line_from_points(origin, point)
        if line is None:
            return _empty_pattern("Chapter14FanLines", timeframe, "NoPattern", "invalid fan line")
        lines.append(line)
    slopes = [float(line["slope"]) for line in lines]
    if direction == "bullish" and not (slopes[0] < slopes[1] < slopes[2] < 0):
        return _empty_pattern("Chapter14FanLines", timeframe, "NoPattern", "fan lines are not progressively flattening")
    if direction == "bearish" and not (slopes[0] > slopes[1] > slopes[2] > 0):
        return _empty_pattern("Chapter14FanLines", timeframe, "NoPattern", "fan lines are not progressively flattening")
    latest_position = len(frame) - 1
    third_value = _line_value(lines[2], latest_position)
    latest_close = float(close.iloc[-1])
    if direction == "bullish":
        broken = latest_close > third_value * 1.01
        status = "ThirdFanBreakUpside" if broken else "FanLinesDeveloping"
    else:
        broken = latest_close < third_value * 0.99
        status = "ThirdFanBreakDownside" if broken else "FanLinesDeveloping"
    return {
        "pattern": "Chapter14FanLines",
        "status": status,
        "direction": direction,
        "timeframe": timeframe,
        "score": 0.72 if broken else 0.48,
        "origin": _anchor_payload(origin),
        "fan_points": [_anchor_payload(point) for point in fan_points],
        "third_fan_value": _finite_or_none(third_value),
        "latest_close": _finite_or_none(latest_close),
        "third_fan_broken": bool(broken),
        "lines": [
            {
                "name": f"F{index + 1}",
                "start_date": _date_string(frame.index[int(origin["position"])]),
                "end_date": _date_string(frame.index[int(point["position"])]),
                "current_value": _finite_or_none(_line_value(line, latest_position)),
                "start_value": _finite_or_none(_line_value(line, int(origin["position"]))),
                "end_value": _finite_or_none(_line_value(line, int(point["position"]))),
                "slope_log_per_bar": _finite_or_none(line["slope"]),
                "intercept": _finite_or_none(line["intercept"]),
            }
            for index, (point, line) in enumerate(zip(fan_points, lines))
        ],
        "reliability_notes": [
            "Three-Fan Principle is used only for corrective moves, not as a general reversal rule.",
            "Third fan-line break suggests the corrective move may be ending." if broken else "Fan lines are still developing and need a third-line break.",
        ],
    }


def _choose_timeframe_line(trendlines: list[dict[str, Any]], timeframe: str) -> dict[str, Any]:
    if trendlines:
        return _preferred_payload(trendlines[0])
    return _empty_pattern("Chapter14Trendline", timeframe, "NoPattern", "no valid Chapter 14 trendline")


def _choose_preferred(timeframes: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    candidates = []
    for name in ("weekly", "monthly", "daily"):
        payload = timeframes.get(name, {}).get(key, {})
        if payload.get("status") not in {None, "NoPattern", "InsufficientData"}:
            candidates.append(payload)
    if candidates:
        return _preferred_payload(max(candidates, key=_generic_rank))
    return {"pattern": "NoChapter14Pattern", "status": "NoPattern", "reason": f"{key} unavailable"}


def _collect_strongest(timeframes: dict[str, dict[str, Any]], key: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for name in ("monthly", "weekly", "daily"):
        value = timeframes.get(name, {}).get(key, [])
        if isinstance(value, list):
            items.extend(value[:4])
    items = [item for item in items if item.get("status") not in {None, "NoPattern", "InsufficientData"}]
    return [_preferred_payload(item) for item in sorted(items, key=_generic_rank, reverse=True)[:8]]


def _generic_rank(item: dict[str, Any]) -> tuple[float, float, str]:
    status_rank = {
        "DecisiveBreak": 1.1,
        "BorderlineBreak": 0.95,
        "PullbackToBrokenLine": 0.85,
        "ThirdFanBreakUpside": 0.85,
        "ThirdFanBreakDownside": 0.85,
        "InnerLineBreak": 0.70,
        "ReturnLineFailure": 0.68,
        "ReturnLineBreakout": 0.66,
        "ActiveChannel": 0.55,
        "Active": 0.55,
        "ShakeoutWarning": 0.45,
        "FanLinesDeveloping": 0.35,
        "Broken": 0.30,
    }.get(str(item.get("status")), 0.0)
    timeframe_rank = {"monthly": 0.18, "weekly": 0.16, "daily": 0.08, "chart": 0.04}.get(str(item.get("timeframe")), 0.0)
    score = float(item.get("authority_score") or item.get("score") or 0.0)
    date = item.get("line", {}).get("end_date") or item.get("end_date") or ""
    return (status_rank + timeframe_rank + score, score, str(date))


def _line_rank(line: dict[str, Any]) -> tuple[float, float, int]:
    rank = _generic_rank(line)
    return (rank[0], rank[1], int(line.get("duration_bars") or 0))


def _fan_rank(item: dict[str, Any]) -> tuple[float, float, str]:
    return _generic_rank(item)


def _line_from_points(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float] | None:
    first_position = int(first["position"])
    second_position = int(second["position"])
    first_price = float(first["price"])
    second_price = float(second["price"])
    if second_position <= first_position or first_price <= 0 or second_price <= 0:
        return None
    slope = (np.log(second_price) - np.log(first_price)) / (second_position - first_position)
    intercept = np.log(first_price) - slope * first_position
    return {"slope": float(slope), "intercept": float(intercept)}


def _line_value(line: dict[str, float], position: int) -> float:
    return float(np.exp(float(line["intercept"]) + float(line["slope"]) * position))


def _line_touches(
    pivot_events: list[dict[str, Any]],
    line: dict[str, float],
    tolerance: float,
    start_position: int,
) -> list[dict[str, Any]]:
    touches = []
    for event in pivot_events:
        position = int(event["position"])
        if position < start_position:
            continue
        line_value = _line_value(line, position)
        distance = abs(float(event["price"]) - line_value) / line_value if line_value else np.nan
        if np.isfinite(distance) and distance <= tolerance:
            touches.append(
                {
                    "date": event["date"],
                    "position": position,
                    "price": _finite_or_none(event["price"]),
                    "line_value": _finite_or_none(line_value),
                    "distance_pct": _finite_or_none(distance),
                }
            )
    return touches


def _double_trendline(
    close: pd.Series,
    pivot_events: list[dict[str, Any]],
    line: dict[str, float],
    kind: str,
    tolerance: float,
    start_position: int,
) -> dict[str, Any]:
    residuals = []
    events = []
    for event in pivot_events:
        position = int(event["position"])
        if position < start_position:
            continue
        line_value = _line_value(line, position)
        residual = float(event["price"] / line_value - 1.0) if line_value else np.nan
        if np.isfinite(residual):
            residuals.append(residual)
            events.append(event)
    if len(residuals) < 4:
        return {"status": "NoDoubleTrendline", "reason": "not enough pivot contacts"}
    residual_array = np.array(residuals, dtype=float)
    if kind == "uptrend":
        outer_residuals = residual_array[(residual_array < -tolerance * 0.5) & (residual_array > -0.12)]
    else:
        outer_residuals = residual_array[(residual_array > tolerance * 0.5) & (residual_array < 0.12)]
    if len(outer_residuals) < 2:
        return {"status": "NoDoubleTrendline", "reason": "outer-line contacts unavailable"}
    offset = float(np.median(outer_residuals))
    if abs(offset) <= tolerance * 0.75:
        return {"status": "NoDoubleTrendline", "reason": "outer-line offset too small"}
    outer_intercept = float(line["intercept"] + np.log1p(offset))
    outer_line = {"slope": float(line["slope"]), "intercept": outer_intercept}
    latest_position = len(close) - 1
    latest_close = float(close.iloc[-1])
    outer_value = _line_value(outer_line, latest_position)
    outer_penetration = _penetration_pct(latest_close, outer_value, kind)
    outer_decisive_break = outer_penetration >= DECISIVE_PENETRATION_PCT
    return {
        "status": "DoubleTrendline",
        "outer_line_role": "lower_outer_line" if kind == "uptrend" else "upper_outer_line",
        "outer_offset_pct": _finite_or_none(offset),
        "outer_current_value": _finite_or_none(outer_value),
        "outer_penetration_pct": _finite_or_none(outer_penetration),
        "outer_decisive_break": bool(outer_decisive_break),
        "rule": "In a double trendline, Chapter 14 waits for decisive penetration of the outer line.",
        "line": {
            "current_value": _finite_or_none(outer_value),
            "slope_log_per_bar": _finite_or_none(outer_line["slope"]),
            "intercept": _finite_or_none(outer_line["intercept"]),
        },
    }


def _return_line_failure(
    frame: pd.DataFrame,
    high: pd.Series,
    low: pd.Series,
    line: dict[str, float],
    offset: float,
    kind: str,
    start_position: int,
) -> dict[str, Any]:
    pivot_high, pivot_low = _confirmed_pivots(high, low)
    events = _pivot_events(frame, pivot_high if kind == "uptrend" else pivot_low, "high" if kind == "uptrend" else "low")
    events = [event for event in events if int(event["position"]) > start_position]
    if not events:
        return {"failure_to_reach": False}
    event = events[-1]
    position = int(event["position"])
    basic_log = float(line["intercept"] + line["slope"] * position)
    return_value = float(np.exp(basic_log + offset)) if kind == "uptrend" else float(np.exp(basic_log - offset))
    if kind == "uptrend":
        margin = (return_value - float(event["price"])) / return_value
    else:
        margin = (float(event["price"]) - return_value) / float(event["price"])
    failure = np.isfinite(margin) and margin >= RETURN_LINE_FAILURE_PCT
    return {
        "failure_to_reach": bool(failure),
        "failure_date": event["date"] if failure else None,
        "failure_pivot_price": _finite_or_none(event["price"]) if failure else None,
        "return_line_value": _finite_or_none(return_value) if failure else None,
        "failure_margin_pct": _finite_or_none(margin) if failure else None,
    }


def _pullback_to_broken_line(
    frame: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    line: dict[str, float],
    kind: str,
    break_position: int,
) -> dict[str, Any]:
    end = min(len(frame), break_position + 45)
    for position in range(break_position + 1, end):
        value = _line_value(line, position)
        if kind == "uptrend":
            touched = float(high.iloc[position]) >= value * 0.985
            failed_reclaim = touched and float(close.iloc[position]) < value
        else:
            touched = float(low.iloc[position]) <= value * 1.015
            failed_reclaim = touched and float(close.iloc[position]) > value
        if touched:
            return {
                "observed": True,
                "date": _date_string(frame.index[position]),
                "line_value": _finite_or_none(value),
                "close": _finite_or_none(close.iloc[position]),
                "failed_reclaim": bool(failed_reclaim),
            }
    return {"observed": False}


def _authority_score(
    touch_count: int,
    duration_bars: int,
    lookback_bars: int,
    slope_pct_per_bar: float,
    pivot_spacing: int,
    confirmed_touches: bool,
    double_trendline: bool,
) -> float:
    touch_component = 0.34 * _clip01(touch_count / 4.0)
    duration_component = 0.22 * _clip01(duration_bars / max(lookback_bars * 0.55, 1.0))
    spacing_component = 0.14 * _clip01(pivot_spacing / max(lookback_bars * 0.18, 1.0))
    abs_slope = abs(float(slope_pct_per_bar))
    if abs_slope <= 0.0002:
        slope_component = 0.07
    elif abs_slope <= 0.004:
        slope_component = 0.16
    elif abs_slope <= 0.008:
        slope_component = 0.10
    else:
        slope_component = 0.04
    confirmation_component = 0.10 if confirmed_touches else 0.0
    double_component = 0.04 if double_trendline else 0.0
    return float(min(0.99, 0.10 + touch_component + duration_component + spacing_component + slope_component + confirmation_component + double_component))


def _penetration_pct(close_value: float, line_value: float, kind: str) -> float:
    if line_value <= 0:
        return 0.0
    if kind == "uptrend":
        return float(max(0.0, (line_value - close_value) / line_value))
    return float(max(0.0, (close_value - line_value) / line_value))


def _distance_to_line(close_value: float, line_value: float, kind: str) -> float:
    if line_value <= 0:
        return np.nan
    if kind == "uptrend":
        return float((close_value - line_value) / close_value)
    return float((line_value - close_value) / close_value)


def _intraday_shakeout(high: pd.Series, low: pd.Series, close: pd.Series, line: dict[str, float], kind: str, position: int) -> bool:
    value = _line_value(line, position)
    if kind == "uptrend":
        return bool(float(low.iloc[position]) < value * 0.99 and float(close.iloc[position]) >= value * 0.995)
    return bool(float(high.iloc[position]) > value * 1.01 and float(close.iloc[position]) <= value * 1.005)


def _latest_volume_ratio(volume: pd.Series) -> float | None:
    latest = volume.iloc[-1] if len(volume) else np.nan
    baseline = volume.shift(1).rolling(20, min_periods=5).mean().iloc[-1] if len(volume) else np.nan
    if pd.notna(latest) and pd.notna(baseline) and float(baseline) > 0:
        return float(latest) / float(baseline)
    return None


def _line_direction(kind: str, status: str) -> str:
    if kind == "uptrend" and status in {"DecisiveBreak", "BorderlineBreak", "PullbackToBrokenLine", "Broken"}:
        return "bearish_warning"
    if kind == "downtrend" and status in {"DecisiveBreak", "BorderlineBreak", "PullbackToBrokenLine", "Broken"}:
        return "bullish_warning"
    if kind == "uptrend":
        return "bullish_context"
    if kind == "downtrend":
        return "bearish_context"
    return "neutral"


def _line_notes(
    kind: str,
    status: str,
    authority: float,
    double_trendline: dict[str, Any],
    break_info: dict[str, Any],
) -> list[str]:
    notes = []
    if authority >= 0.70:
        notes.append("Trendline has high Chapter 14 authority from touches, duration, and slope.")
    elif authority < 0.45:
        notes.append("Trendline is tentative and should be treated as context.")
    if status == "InnerLineBreak" and double_trendline.get("status") == "DoubleTrendline":
        notes.append("Inner line is broken, but Chapter 14 waits for the outer double trendline.")
    if status == "ShakeoutWarning":
        notes.append("Intraday penetration did not receive closing confirmation.")
    if break_info.get("pullback", {}).get("failed_reclaim"):
        notes.append("Pullback to the broken trendline failed to reclaim it.")
    if kind == "uptrend":
        notes.append("Uptrend lines are drawn across confirmed wave lows.")
    else:
        notes.append("Downtrend lines are drawn across confirmed rally highs.")
    return notes[:5]


def _channel_notes(status: str, kind: str, failure: dict[str, Any]) -> list[str]:
    notes = []
    if status == "ReturnLineFailure":
        notes.append("Failure to reach the return line is Chapter 14 deterioration evidence.")
    if status == "ReturnLineBreakout":
        notes.append("Break beyond the return line indicates acceleration, not support/resistance from the return line.")
    if failure.get("failure_margin_pct") is not None:
        notes.append("Failure margin is recorded because Chapter 14 uses it as a rough follow-through yardstick.")
    notes.append("Trend channels are parallel return-line channels built from the basic trendline.")
    if kind == "downtrend":
        notes.append("Downtrend return lines are less reliable than uptrend channels in Chapter 14.")
    return notes[:5]


def _pivot_events(frame: pd.DataFrame, pivots: pd.Series, kind: str) -> list[dict[str, Any]]:
    events = []
    for date, value in pivots.dropna().items():
        price = _finite_or_none(value)
        if price is None or price <= 0:
            continue
        events.append(
            {
                "date": _date_string(date),
                "position": int(frame.index.get_loc(date)),
                "price": price,
                "kind": kind,
            }
        )
    return events


def _confirmed_pivots(high: pd.Series, low: pd.Series) -> tuple[pd.Series, pd.Series]:
    window = PIVOT_LEFT_BARS + PIVOT_RIGHT_BARS + 1
    raw_high = high == high.rolling(window=window, center=True).max()
    raw_low = low == low.rolling(window=window, center=True).min()
    confirmed_high = high.shift(PIVOT_RIGHT_BARS).where(raw_high.shift(PIVOT_RIGHT_BARS, fill_value=False), np.nan)
    confirmed_low = low.shift(PIVOT_RIGHT_BARS).where(raw_low.shift(PIVOT_RIGHT_BARS, fill_value=False), np.nan)
    return confirmed_high, confirmed_low


def _anchor_payload(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": event.get("date"),
        "position": int(event.get("position", 0)),
        "price": _finite_or_none(event.get("price")),
        "kind": event.get("kind"),
    }


def _preferred_payload(item: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in item.items():
        if isinstance(value, dict):
            output[key] = _preferred_payload(value)
        elif isinstance(value, list):
            output[key] = [_preferred_payload(entry) if isinstance(entry, dict) else entry for entry in value]
        elif isinstance(value, (np.floating, float)):
            output[key] = _finite_or_none(value)
        elif isinstance(value, (np.integer, int)):
            output[key] = int(value)
        elif isinstance(value, (np.bool_, bool)):
            output[key] = bool(value)
        else:
            output[key] = value
    return output


def _empty_timeframe(timeframe: str, status: str, reason: str, rows: int) -> dict[str, Any]:
    return {
        "state": status,
        "timeframe": timeframe,
        "rows": int(rows),
        "preferred_trendline": _empty_pattern("Chapter14Trendline", timeframe, status, reason),
        "trendlines": [],
        "channel": _empty_pattern("Chapter14Channel", timeframe, status, reason),
        "fan_lines": _empty_pattern("Chapter14FanLines", timeframe, status, reason),
    }


def _empty_pattern(pattern: str, timeframe: str, status: str, reason: str) -> dict[str, Any]:
    return {
        "pattern": pattern,
        "status": status,
        "direction": "neutral",
        "timeframe": timeframe,
        "score": 0.0,
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
    try:
        return str(pd.Timestamp(value).date())
    except Exception:
        return str(value)


def _safe_float(value: object) -> float | None:
    try:
        output = float(value)
    except Exception:
        return None
    return output if pd.notna(output) else None


def _finite_or_none(value: Any) -> float | None:
    try:
        output = float(value)
    except Exception:
        return None
    return output if np.isfinite(output) else None


def _clip01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))
