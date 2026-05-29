from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


MIN_PATTERN_BARS = 40
MIN_GAP_PCT = {"daily": 0.006, "weekly": 0.012, "monthly": 0.020, "chart": 0.006}
MIN_GAP_ATR_MULTIPLE = 0.45
RECENT_GAP_LOOKBACK = {"daily": 90, "weekly": 52, "monthly": 36, "chart": 90}
HABITUAL_GAP_FREQUENCY = {"daily": 0.08, "weekly": 0.14, "monthly": 0.20, "chart": 0.08}
RUNAWAY_LOOKBACK = {"daily": 30, "weekly": 13, "monthly": 6, "chart": 30}
BREAKOUT_LOOKBACK = {"daily": 63, "weekly": 26, "monthly": 18, "chart": 63}
EXHAUSTION_FILL_BARS = {"daily": 5, "weekly": 3, "monthly": 2, "chart": 5}
ISLAND_MAX_BARS = {"daily": 8, "weekly": 4, "monthly": 2, "chart": 8}


def analyze_chapter_12_gaps(prices: pd.DataFrame, target_column: str = "close") -> dict[str, Any]:
    """Analyze Edwards/Magee Chapter 12 gap classes and gap-zone behavior."""

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
    classified_gaps = {
        "principle": (
            "Classify true price-range gaps before using them. Common gaps are mostly context; "
            "breakaway and runaway gaps can confirm trend; exhaustion and island reversals are risk warnings."
        ),
        "preferred": _choose_preferred(timeframes, "preferred_gap"),
        "timeframes": {
            name: {
                "state": payload.get("state"),
                "rows": payload.get("rows"),
                "preferred": payload.get("preferred_gap", {}),
                "recent_gaps": payload.get("recent_gaps", []),
            }
            for name, payload in timeframes.items()
        },
    }
    islands = {
        "principle": (
            "An Island Reversal is a compact trading area isolated by an exhaustion-style gap and "
            "an opposite breakaway gap at overlapping levels."
        ),
        "preferred": _choose_preferred(timeframes, "island_reversal"),
        "timeframes": {
            name: {
                "state": payload.get("state"),
                "rows": payload.get("rows"),
                "preferred": payload.get("island_reversal", {}),
            }
            for name, payload in timeframes.items()
        },
    }
    return {
        "principle": (
            "Chapter 12 adds gap classification and gap-zone governance. It rejects the rule that every gap "
            "must close before a trend can be trusted."
        ),
        "primary_timeframe": "daily",
        "secondary_timeframe": "weekly",
        "confirmation_rule": "Use true range gaps where adjacent bars do not overlap; filter corporate-action and habitual thin-issue gaps.",
        "preferred": _choose_chapter_12_preferred(classified_gaps["preferred"], islands["preferred"]),
        "classified_gaps": classified_gaps,
        "island_reversals": islands,
        "timeframes": timeframes,
        "technical_method_card": chapter_12_gaps_method_card(target_column=target),
    }


def latest_chapter_12_gaps(
    prices: pd.DataFrame,
    target_column: str = "close",
    timeframe: str = "chart",
) -> dict[str, Any]:
    """Return Chapter 12 diagnostics for a plotted bar set."""

    return _analyze_timeframe(
        frame=prices,
        timeframe=timeframe,
        target_column=target_column.lower(),
    )


def chapter_12_gaps_method_card(target_column: str = "close") -> dict[str, Any]:
    return {
        "name": "edwards_magee_chapter_12_gaps",
        "version": "chapter_12_alignment_v1",
        "target_column": target_column.lower(),
        "implemented_patterns": [
            "common_area_gap",
            "breakaway_gap",
            "runaway_measuring_gap",
            "exhaustion_gap",
            "island_reversal",
        ],
        "gap_definition": "true range gap: current low above prior high, or current high below prior low",
        "filters": {
            "significance": "gap width must exceed a minimum percent threshold or ATR multiple",
            "corporate_actions": "dividend and split rows are excluded from directional interpretation",
            "thin_issues": "habitual high gap frequency is marked as low reliability",
        },
        "classification_rules": {
            "common": "gap inside congestion or without breakout/fast-move context",
            "breakaway": "gap through support/resistance or out of congestion",
            "runaway": "gap in a fast straight-line move with remaining objective potential",
            "exhaustion": "gap after an extended move with quick fill, weak follow-through, or extreme turnover",
            "island": "opposite gaps at overlapping levels isolate a compact trading area",
        },
        "decision_use": (
            "Breakaway and runaway gaps can support matching model direction; exhaustion gaps and island reversals "
            "can block fresh directional action; common gaps do not drive action."
        ),
    }


def _analyze_timeframe(frame: pd.DataFrame, timeframe: str, target_column: str) -> dict[str, Any]:
    clean = frame.copy().sort_index()
    if clean.empty or target_column not in clean.columns:
        return _empty_timeframe(timeframe, "InsufficientData", "missing target column", len(clean))
    clean.index = pd.DatetimeIndex(clean.index)
    clean = clean.loc[pd.to_numeric(clean[target_column], errors="coerce").notna()]
    if len(clean) < MIN_PATTERN_BARS:
        return _empty_timeframe(timeframe, "InsufficientData", "not enough bars for Chapter 12 gap analysis", len(clean))
    if "high" not in clean.columns or "low" not in clean.columns:
        return _empty_timeframe(timeframe, "InsufficientData", "high/low columns are required for true gap analysis", len(clean))

    gaps = _classify_gaps(clean, timeframe=timeframe, target_column=target_column)
    lookback = RECENT_GAP_LOOKBACK.get(timeframe, RECENT_GAP_LOOKBACK["chart"])
    recent_gaps = [gap for gap in gaps if int(gap.get("position", 0)) >= len(clean) - lookback]
    visible_gaps = [gap for gap in recent_gaps if gap.get("pattern") not in {"InsignificantGap", "CorporateActionGap"}]
    preferred_gap = _choose_gap(visible_gaps, timeframe=timeframe)
    island = _find_island_reversal(gaps, clean, timeframe=timeframe, target_column=target_column)

    return {
        "state": "Measured",
        "timeframe": timeframe,
        "rows": int(len(clean)),
        "start_date": _date_string(clean.index[0]),
        "end_date": _date_string(clean.index[-1]),
        "recent_gaps": [_preferred_payload(gap) for gap in visible_gaps[-8:]],
        "preferred_gap": _preferred_payload(preferred_gap),
        "island_reversal": _preferred_payload(island),
        "habitual_gap_frequency": _finite_or_none(_gap_frequency(gaps, len(clean), len(clean) - 1, lookback)),
    }


def _classify_gaps(frame: pd.DataFrame, timeframe: str, target_column: str) -> list[dict[str, Any]]:
    close = pd.to_numeric(frame[target_column], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    open_price = pd.to_numeric(frame["open"], errors="coerce") if "open" in frame.columns else close.shift(1)
    volume = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame.columns else pd.Series(np.nan, index=frame.index)
    dividends = pd.to_numeric(frame["dividends"], errors="coerce").fillna(0.0) if "dividends" in frame.columns else pd.Series(0.0, index=frame.index)
    splits = pd.to_numeric(frame["stock_splits"], errors="coerce").fillna(0.0) if "stock_splits" in frame.columns else pd.Series(0.0, index=frame.index)
    true_range_pct = _safe_divide(_true_range(high=high, low=low, close=close), close)
    volume_mean = volume.shift(1).rolling(20, min_periods=5).mean()
    gaps: list[dict[str, Any]] = []
    raw_gap_positions: list[int] = []

    for position in range(1, len(frame)):
        prior_high = float(high.iloc[position - 1])
        prior_low = float(low.iloc[position - 1])
        prior_close = float(close.iloc[position - 1])
        if not np.isfinite(prior_high) or not np.isfinite(prior_low) or prior_close <= 0:
            continue

        gap_direction = None
        lower = upper = None
        if float(low.iloc[position]) > prior_high:
            gap_direction = "up"
            lower = prior_high
            upper = float(low.iloc[position])
        elif float(high.iloc[position]) < prior_low:
            gap_direction = "down"
            lower = float(high.iloc[position])
            upper = prior_low
        if gap_direction is None or lower is None or upper is None or upper <= lower:
            continue

        raw_gap_positions.append(position)
        gap_width = float(upper - lower)
        gap_pct = gap_width / prior_close
        atr = float(true_range_pct.shift(1).rolling(20, min_periods=5).mean().iloc[position] or np.nan)
        atr_multiple = gap_pct / atr if np.isfinite(atr) and atr > 0 else np.nan
        significant = gap_pct >= MIN_GAP_PCT.get(timeframe, MIN_GAP_PCT["chart"]) or (
            np.isfinite(atr_multiple) and atr_multiple >= MIN_GAP_ATR_MULTIPLE
        )
        corporate_action = bool(float(dividends.iloc[position]) != 0.0 or float(splits.iloc[position]) != 0.0)
        gap_frequency = _gap_frequency_from_positions(raw_gap_positions, position, RECENT_GAP_LOOKBACK.get(timeframe, 90))
        habitual = gap_frequency >= HABITUAL_GAP_FREQUENCY.get(timeframe, HABITUAL_GAP_FREQUENCY["chart"])
        context = _gap_context(
            close=close,
            high=high,
            low=low,
            open_price=open_price,
            volume=volume,
            volume_mean=volume_mean,
            position=position,
            gap_direction=gap_direction,
            timeframe=timeframe,
        )
        fill = _gap_fill_state(
            high=high,
            low=low,
            position=position,
            gap_direction=gap_direction,
            lower=lower,
            upper=upper,
        )
        gap = _gap_payload(
            frame=frame,
            close=close,
            position=position,
            gap_direction=gap_direction,
            lower=lower,
            upper=upper,
            gap_width=gap_width,
            gap_pct=gap_pct,
            atr_multiple=atr_multiple,
            significant=significant,
            corporate_action=corporate_action,
            habitual=habitual,
            gap_frequency=gap_frequency,
            context=context,
            fill=fill,
            prior_same_direction_gaps=_prior_same_direction_gap_count(gaps, position, gap_direction, lookback=30),
            timeframe=timeframe,
        )
        gaps.append(gap)
    return gaps


def _gap_payload(
    frame: pd.DataFrame,
    close: pd.Series,
    position: int,
    gap_direction: str,
    lower: float,
    upper: float,
    gap_width: float,
    gap_pct: float,
    atr_multiple: float,
    significant: bool,
    corporate_action: bool,
    habitual: bool,
    gap_frequency: float,
    context: dict[str, Any],
    fill: dict[str, Any],
    prior_same_direction_gaps: int,
    timeframe: str,
) -> dict[str, Any]:
    if corporate_action:
        pattern = "CorporateActionGap"
        status = "Excluded"
        direction = "neutral"
    elif not significant:
        pattern = "InsignificantGap"
        status = "Ignored"
        direction = "neutral"
    else:
        pattern, direction = _gap_classification(
            gap_direction=gap_direction,
            context=context,
            fill=fill,
            prior_same_direction_gaps=prior_same_direction_gaps,
        )
        status = _gap_status(pattern, fill)

    measured_objective = None
    objective_reached = None
    move_start = _move_start(close, position, gap_direction, timeframe)
    gap_midpoint = float((lower + upper) / 2.0)
    if pattern == "RunawayGap" and move_start.get("price") is not None:
        start_price = float(move_start["price"])
        if gap_direction == "up":
            measured_objective = gap_midpoint + (gap_midpoint - start_price)
            objective_reached = bool(close.iloc[position:].max() >= measured_objective)
        else:
            measured_objective = gap_midpoint - (start_price - gap_midpoint)
            objective_reached = bool(measured_objective > 0 and close.iloc[position:].min() <= measured_objective)
        if measured_objective is not None and measured_objective <= 0:
            measured_objective = None
            objective_reached = None

    score = _gap_score(
        pattern=pattern,
        gap_pct=gap_pct,
        atr_multiple=atr_multiple,
        context=context,
        fill=fill,
        habitual=habitual,
        objective_reached=objective_reached,
    )
    return {
        "pattern": pattern,
        "status": status,
        "direction": direction,
        "gap_direction": gap_direction,
        "timeframe": timeframe,
        "score": _finite_or_none(score),
        "position": int(position),
        "date": _date_string(frame.index[position]),
        "latest_close": _finite_or_none(close.iloc[-1]),
        "gap_zone": {
            "lower": _finite_or_none(lower),
            "upper": _finite_or_none(upper),
            "midpoint": _finite_or_none(gap_midpoint),
            "width": _finite_or_none(gap_width),
            "width_pct": _finite_or_none(gap_pct),
            "atr_multiple": _finite_or_none(atr_multiple),
        },
        "fill_state": fill,
        "context": context,
        "move_start": move_start,
        "prior_same_direction_gaps": int(prior_same_direction_gaps),
        "habitual_gap_frequency": _finite_or_none(gap_frequency),
        "habitual_gap_warning": bool(habitual),
        "corporate_action_excluded": bool(corporate_action),
        "measured_objective": _finite_or_none(measured_objective),
        "objective_reached": objective_reached,
        "reliability_notes": _gap_notes(pattern, status, context, fill, habitual, objective_reached),
    }


def _gap_classification(
    gap_direction: str,
    context: dict[str, Any],
    fill: dict[str, Any],
    prior_same_direction_gaps: int,
) -> tuple[str, str]:
    bullish = gap_direction == "up"
    extended = bool(context.get("extended_prior_move"))
    breakout = bool(context.get("breakout_context"))
    congestion = bool(context.get("prior_congestion"))
    quick_fill = fill.get("fill_bars") is not None and int(fill["fill_bars"]) <= EXHAUSTION_FILL_BARS.get(str(context.get("timeframe")), 5)
    weak_followthrough = bool(context.get("weak_followthrough"))
    extreme_volume = bool(context.get("volume_extreme"))

    if extended and (quick_fill or weak_followthrough or (extreme_volume and prior_same_direction_gaps >= 1)):
        return "ExhaustionGap", "bearish_warning" if bullish else "bullish_warning"
    if breakout:
        return "BreakawayGap", "bullish" if bullish else "bearish"
    if extended:
        return "RunawayGap", "bullish" if bullish else "bearish"
    if congestion:
        return "CommonGap", "neutral"
    return "CommonGap", "neutral"


def _gap_status(pattern: str, fill: dict[str, Any]) -> str:
    if pattern == "CommonGap":
        return fill.get("state", "Open")
    if pattern == "ExhaustionGap":
        return "ClosedQuickly" if fill.get("fill_bars") is not None and int(fill["fill_bars"]) <= 5 else "ExhaustionWarning"
    if fill.get("state") == "Filled":
        return "Filled"
    if fill.get("state") == "PartiallyFilled":
        return "PartiallyFilled"
    return "Open"


def _gap_context(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    open_price: pd.Series,
    volume: pd.Series,
    volume_mean: pd.Series,
    position: int,
    gap_direction: str,
    timeframe: str,
) -> dict[str, Any]:
    lookback = RUNAWAY_LOOKBACK.get(timeframe, RUNAWAY_LOOKBACK["chart"])
    breakout_window = BREAKOUT_LOOKBACK.get(timeframe, BREAKOUT_LOOKBACK["chart"])
    prior_start = max(0, position - lookback)
    prior_return = None
    if prior_start < position and float(close.iloc[prior_start]) > 0:
        prior_return = float(close.iloc[position - 1] / close.iloc[prior_start] - 1.0)
    extended_prior_move = bool(
        (gap_direction == "up" and prior_return is not None and prior_return >= 0.12)
        or (gap_direction == "down" and prior_return is not None and prior_return <= -0.12)
    )

    prior_high = high.shift(1).rolling(breakout_window, min_periods=max(5, min(20, breakout_window))).max().iloc[position]
    prior_low = low.shift(1).rolling(breakout_window, min_periods=max(5, min(20, breakout_window))).min().iloc[position]
    prior_range_20 = high.shift(1).rolling(20, min_periods=10).max() - low.shift(1).rolling(20, min_periods=10).min()
    prior_range_126 = high.shift(1).rolling(126, min_periods=30).max() - low.shift(1).rolling(126, min_periods=30).min()
    range_20_pct = float(prior_range_20.iloc[position] / close.iloc[position - 1]) if close.iloc[position - 1] else np.nan
    range_126_pct = float(prior_range_126.iloc[position] / close.iloc[position - 1]) if close.iloc[position - 1] else np.nan
    prior_congestion = bool(
        np.isfinite(range_20_pct)
        and (
            range_20_pct <= 0.08
            or (np.isfinite(range_126_pct) and range_20_pct <= range_126_pct * 0.45)
        )
    )
    breakout_context = bool(
        (gap_direction == "up" and np.isfinite(prior_high) and float(low.iloc[position]) > float(prior_high) * 1.002)
        or (gap_direction == "down" and np.isfinite(prior_low) and float(high.iloc[position]) < float(prior_low) * 0.998)
    )
    if prior_congestion and not breakout_context:
        breakout_context = bool(
            (gap_direction == "up" and np.isfinite(prior_high) and float(close.iloc[position]) > float(prior_high) * 1.002)
            or (gap_direction == "down" and np.isfinite(prior_low) and float(close.iloc[position]) < float(prior_low) * 0.998)
        )

    current_volume = float(volume.iloc[position]) if pd.notna(volume.iloc[position]) else np.nan
    baseline_volume = float(volume_mean.iloc[position]) if pd.notna(volume_mean.iloc[position]) else np.nan
    volume_ratio = current_volume / baseline_volume if np.isfinite(current_volume) and np.isfinite(baseline_volume) and baseline_volume > 0 else np.nan
    day_range = float(high.iloc[position] - low.iloc[position])
    close_location = float((close.iloc[position] - low.iloc[position]) / day_range) if day_range > 0 else np.nan
    next_close = float(close.iloc[position + 1]) if position + 1 < len(close) else np.nan
    weak_followthrough = bool(
        (gap_direction == "up" and ((np.isfinite(close_location) and close_location <= 0.35) or (np.isfinite(next_close) and next_close < close.iloc[position])))
        or (gap_direction == "down" and ((np.isfinite(close_location) and close_location >= 0.65) or (np.isfinite(next_close) and next_close > close.iloc[position])))
    )
    return {
        "timeframe": timeframe,
        "prior_return_pct": _finite_or_none(prior_return),
        "extended_prior_move": extended_prior_move,
        "breakout_context": breakout_context,
        "prior_congestion": prior_congestion,
        "prior_resistance": _finite_or_none(prior_high),
        "prior_support": _finite_or_none(prior_low),
        "prior_range_20_pct": _finite_or_none(range_20_pct),
        "volume_ratio": _finite_or_none(volume_ratio),
        "volume_extreme": bool(np.isfinite(volume_ratio) and volume_ratio >= 1.8),
        "close_location": _finite_or_none(close_location),
        "weak_followthrough": weak_followthrough,
        "opening_gap_pct": _finite_or_none((open_price.iloc[position] - close.iloc[position - 1]) / close.iloc[position - 1]),
    }


def _gap_fill_state(
    high: pd.Series,
    low: pd.Series,
    position: int,
    gap_direction: str,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    partial_date = fill_date = None
    partial_bars = fill_bars = None
    for future_position in range(position + 1, len(high)):
        if gap_direction == "up":
            if partial_date is None and float(low.iloc[future_position]) <= upper:
                partial_date = _date_string(low.index[future_position])
                partial_bars = future_position - position
            if float(low.iloc[future_position]) <= lower:
                fill_date = _date_string(low.index[future_position])
                fill_bars = future_position - position
                break
        else:
            if partial_date is None and float(high.iloc[future_position]) >= lower:
                partial_date = _date_string(high.index[future_position])
                partial_bars = future_position - position
            if float(high.iloc[future_position]) >= upper:
                fill_date = _date_string(high.index[future_position])
                fill_bars = future_position - position
                break
    if fill_date is not None:
        state = "Filled"
    elif partial_date is not None:
        state = "PartiallyFilled"
    else:
        state = "Open"
    return {
        "state": state,
        "partial_fill_date": partial_date,
        "partial_fill_bars": partial_bars,
        "fill_date": fill_date,
        "fill_bars": fill_bars,
        "age_bars": int(len(high) - 1 - position) if fill_date is None else int(fill_bars),
    }


def _find_island_reversal(
    gaps: list[dict[str, Any]],
    frame: pd.DataFrame,
    timeframe: str,
    target_column: str,
) -> dict[str, Any]:
    eligible = [
        gap
        for gap in gaps
        if gap.get("pattern") not in {"InsignificantGap", "CorporateActionGap"}
        and gap.get("gap_direction") in {"up", "down"}
    ]
    candidates = []
    max_bars = ISLAND_MAX_BARS.get(timeframe, ISLAND_MAX_BARS["chart"])
    close = pd.to_numeric(frame[target_column], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame.columns else pd.Series(np.nan, index=frame.index)
    for first in eligible:
        for second in eligible:
            first_pos = int(first.get("position", -1))
            second_pos = int(second.get("position", -1))
            if second_pos <= first_pos or second_pos - first_pos > max_bars + 1:
                continue
            if first.get("gap_direction") == second.get("gap_direction"):
                continue
            first_zone = first.get("gap_zone", {})
            second_zone = second.get("gap_zone", {})
            overlap_lower = max(float(first_zone.get("lower") or np.nan), float(second_zone.get("lower") or np.nan))
            overlap_upper = min(float(first_zone.get("upper") or np.nan), float(second_zone.get("upper") or np.nan))
            if not np.isfinite(overlap_lower) or not np.isfinite(overlap_upper) or overlap_upper < overlap_lower:
                continue
            island_slice = close.iloc[first_pos:second_pos]
            volume_slice = volume.iloc[first_pos:second_pos]
            direction = "bearish" if first.get("gap_direction") == "up" else "bullish"
            prior_start = max(0, first_pos - RUNAWAY_LOOKBACK.get(timeframe, 30))
            retracement_objective = float(close.iloc[prior_start]) if prior_start < first_pos else None
            score = 0.72
            if volume_slice.notna().any():
                baseline = volume.shift(1).rolling(20, min_periods=5).mean().iloc[first_pos]
                if pd.notna(baseline) and float(baseline) > 0 and float(volume_slice.max()) >= float(baseline) * 1.4:
                    score += 0.12
            score += 0.10 if first.get("pattern") == "ExhaustionGap" or second.get("pattern") == "BreakawayGap" else 0.0
            candidates.append(
                {
                    "pattern": "IslandReversal",
                    "status": "Confirmed",
                    "direction": direction,
                    "timeframe": timeframe,
                    "score": _finite_or_none(min(score, 0.98)),
                    "start_date": _date_string(frame.index[first_pos]),
                    "end_date": _date_string(frame.index[second_pos]),
                    "rows": int(second_pos - first_pos),
                    "latest_close": _finite_or_none(close.iloc[-1]),
                    "entry_gap": _preferred_payload(first),
                    "exit_gap": _preferred_payload(second),
                    "gap_overlap": {
                        "lower": _finite_or_none(overlap_lower),
                        "upper": _finite_or_none(overlap_upper),
                    },
                    "island_range": {
                        "high": _finite_or_none(island_slice.max()),
                        "low": _finite_or_none(island_slice.min()),
                    },
                    "measured_objective": _finite_or_none(retracement_objective),
                    "reliability_notes": [
                        "Island Reversal is a tactical reversal warning; use it as alert and risk control, not a standalone long-term forecast."
                    ],
                }
            )
    recent = [
        candidate
        for candidate in candidates
        if _parse_position_date(frame, candidate.get("end_date")) is not None
        and int(frame.index.get_loc(_parse_position_date(frame, candidate.get("end_date")))) >= len(frame) - RECENT_GAP_LOOKBACK.get(timeframe, 90)
    ]
    if recent:
        return max(recent, key=_gap_rank)
    return _empty_pattern("IslandReversal", timeframe, "NoPattern", "no compact area isolated by opposite overlapping gaps", len(frame))


def _choose_gap(gaps: list[dict[str, Any]], timeframe: str) -> dict[str, Any]:
    actionable = [gap for gap in gaps if gap.get("pattern") not in {"NoChapter12Gap", "InsignificantGap", "CorporateActionGap"}]
    if actionable:
        return max(actionable, key=_gap_rank)
    return _empty_pattern("NoChapter12Gap", timeframe, "NoPattern", "no significant recent true gap", 0)


def _choose_preferred(timeframes: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    patterns = [payload.get(key, {}) for payload in timeframes.values() if isinstance(payload.get(key), dict)]
    actionable = [pattern for pattern in patterns if pattern.get("status") not in {"NoPattern", "InsufficientData", "Ignored", "Excluded"}]
    if actionable:
        return _preferred_payload(max(actionable, key=_gap_rank))
    for timeframe in ("daily", "weekly", "monthly"):
        pattern = timeframes.get(timeframe, {}).get(key, {})
        if pattern:
            return _preferred_payload(pattern)
    return _empty_pattern("NoChapter12Gap", "unknown", "NoPattern", "gap unavailable", 0)


def _choose_chapter_12_preferred(gap: dict[str, Any], island: dict[str, Any]) -> dict[str, Any]:
    candidates = [gap, island]
    actionable = [candidate for candidate in candidates if candidate.get("status") not in {"NoPattern", "InsufficientData", "Ignored", "Excluded"}]
    if actionable:
        return _preferred_payload(max(actionable, key=_gap_rank))
    return {"pattern": "NoChapter12Gap", "status": "NoPattern", "reason": "no Chapter 12 gap detected"}


def _gap_rank(pattern: dict[str, Any]) -> tuple[float, float, str]:
    pattern_rank = {
        "IslandReversal": 6.0,
        "ExhaustionGap": 5.0,
        "RunawayGap": 4.0,
        "BreakawayGap": 3.5,
        "CommonGap": 1.0,
        "CorporateActionGap": 0.2,
        "InsignificantGap": 0.1,
        "NoChapter12Gap": 0.0,
        "IslandReversalNoPattern": 0.0,
    }.get(str(pattern.get("pattern")), 0.0)
    status_rank = {
        "Confirmed": 1.0,
        "Open": 0.8,
        "ExhaustionWarning": 0.8,
        "ClosedQuickly": 0.7,
        "PartiallyFilled": 0.5,
        "Filled": 0.3,
        "NoPattern": 0.0,
    }.get(str(pattern.get("status")), 0.2)
    score = float(pattern.get("score") or 0.0)
    date = pattern.get("date") or pattern.get("end_date") or ""
    return (pattern_rank + status_rank, score, str(date))


def _gap_score(
    pattern: str,
    gap_pct: float,
    atr_multiple: float,
    context: dict[str, Any],
    fill: dict[str, Any],
    habitual: bool,
    objective_reached: bool | None,
) -> float:
    if pattern in {"InsignificantGap", "CorporateActionGap"}:
        return 0.0
    gap_component = 0.25 * _clip01(gap_pct / 0.05)
    atr_component = 0.20 * _clip01(float(atr_multiple) / 2.5) if np.isfinite(atr_multiple) else 0.08
    volume_ratio = float(context.get("volume_ratio") or 1.0)
    volume_component = 0.15 * _clip01(volume_ratio / 2.0)
    context_component = 0.0
    if context.get("breakout_context"):
        context_component += 0.18
    if context.get("extended_prior_move"):
        context_component += 0.18
    if context.get("prior_congestion"):
        context_component += 0.08
    if pattern == "ExhaustionGap":
        context_component += 0.18 if fill.get("fill_bars") is not None else 0.06
    if pattern == "RunawayGap" and objective_reached is True:
        context_component -= 0.10
    if habitual:
        context_component -= 0.20
    return float(min(max(0.20 + gap_component + atr_component + volume_component + context_component, 0.0), 0.99))


def _gap_notes(
    pattern: str,
    status: str,
    context: dict[str, Any],
    fill: dict[str, Any],
    habitual: bool,
    objective_reached: bool | None,
) -> list[str]:
    notes: list[str] = []
    if pattern == "CommonGap":
        notes.append("Common/area gap has little directional forecasting value.")
    if pattern == "BreakawayGap":
        notes.append("Breakaway gap emphasizes the breakout; high far-side volume lowers near-term fill odds.")
    if pattern == "RunawayGap":
        notes.append("Runaway gap is treated as a measuring gap; use the objective more for exit/risk than entry.")
    if pattern == "ExhaustionGap":
        notes.append("Exhaustion gap is a stop/warning condition, not a standalone long-term reversal forecast.")
    if status == "Filled" or status == "ClosedQuickly":
        notes.append("Gap zone has been covered.")
    if fill.get("fill_bars") is not None and int(fill["fill_bars"]) <= 5:
        notes.append("Quick gap closure supports exhaustion rather than continuation.")
    if context.get("volume_extreme"):
        notes.append("Volume is extreme relative to the recent baseline.")
    if habitual:
        notes.append("This issue/timeframe gaps frequently; downgrade single-gap significance.")
    if objective_reached is True:
        notes.append("Runaway gap measured objective has already been reached.")
    return notes


def _move_start(close: pd.Series, position: int, gap_direction: str, timeframe: str) -> dict[str, Any]:
    lookback = RUNAWAY_LOOKBACK.get(timeframe, RUNAWAY_LOOKBACK["chart"])
    start = max(0, position - lookback)
    scope = close.iloc[start:position]
    if scope.empty:
        return {"date": None, "price": None}
    if gap_direction == "up":
        index = scope.idxmin()
    else:
        index = scope.idxmax()
    return {"date": _date_string(index), "price": _finite_or_none(scope.loc[index])}


def _prior_same_direction_gap_count(gaps: list[dict[str, Any]], position: int, gap_direction: str, lookback: int) -> int:
    return int(
        sum(
            1
            for gap in gaps
            if gap.get("gap_direction") == gap_direction
            and int(gap.get("position", -9999)) < position
            and int(gap.get("position", -9999)) >= position - lookback
            and gap.get("pattern") not in {"InsignificantGap", "CorporateActionGap"}
        )
    )


def _gap_frequency(gaps: list[dict[str, Any]], rows: int, position: int, lookback: int) -> float:
    if rows <= 1:
        return 0.0
    count = sum(1 for gap in gaps if int(gap.get("position", -1)) >= max(1, position - lookback))
    return float(count / max(1, min(lookback, rows - 1)))


def _gap_frequency_from_positions(positions: list[int], position: int, lookback: int) -> float:
    count = sum(1 for candidate in positions if candidate >= max(1, position - lookback))
    return float(count / max(1, min(lookback, position)))


def _preferred_payload(pattern: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "pattern",
        "status",
        "direction",
        "gap_direction",
        "timeframe",
        "score",
        "position",
        "date",
        "start_date",
        "end_date",
        "rows",
        "latest_close",
        "gap_zone",
        "fill_state",
        "context",
        "move_start",
        "entry_gap",
        "exit_gap",
        "gap_overlap",
        "island_range",
        "prior_same_direction_gaps",
        "habitual_gap_frequency",
        "habitual_gap_warning",
        "corporate_action_excluded",
        "measured_objective",
        "objective_reached",
        "reliability_notes",
        "reason",
    ]
    return {key: pattern[key] for key in keys if key in pattern}


def _empty_timeframe(timeframe: str, status: str, reason: str, rows: int) -> dict[str, Any]:
    return {
        "state": status,
        "timeframe": timeframe,
        "rows": int(rows),
        "recent_gaps": [],
        "preferred_gap": _empty_pattern("NoChapter12Gap", timeframe, status, reason, rows),
        "island_reversal": _empty_pattern("IslandReversal", timeframe, status, reason, rows),
    }


def _empty_pattern(pattern: str, timeframe: str, status: str, reason: str, rows: int) -> dict[str, Any]:
    return {"pattern": pattern, "status": status, "timeframe": timeframe, "rows": int(rows), "reason": reason}


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


def _parse_position_date(frame: pd.DataFrame, value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    timestamp = pd.Timestamp(value)
    return timestamp if timestamp in frame.index else None


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
