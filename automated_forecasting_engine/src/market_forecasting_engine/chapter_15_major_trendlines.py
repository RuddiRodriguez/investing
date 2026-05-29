from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


CONTEXT_PREFIXES = ("benchmark_", "sector_", "market_", "index_")
MIN_MONTHLY_BARS = 36
LOOKBACK_MONTHS = 180
PIVOT_LEFT_BARS = 2
PIVOT_RIGHT_BARS = 2
MIN_PIVOT_SPACING = 6
STOCK_BREAK_PCT = 0.03
INDEX_BREAK_PCT = 0.02
TOUCH_TOLERANCE_PCT = 0.035


def analyze_chapter_15_major_trendlines(prices: pd.DataFrame, target_column: str = "close") -> dict[str, Any]:
    """Analyze Edwards/Magee Chapter 15 major trendlines.

    Benchmark/index context is optional. When supplied as columns with
    benchmark_, sector_, market_, or index_ prefixes, those series are analyzed
    as broad-market confirmation. When absent, confirmation is marked
    unavailable and the stock-only diagnostics still run.
    """

    target = target_column.lower()
    stock = _analyze_series(
        prices=prices,
        column=target,
        label="stock",
        series_type="stock",
        break_pct=STOCK_BREAK_PCT,
    )
    context = {
        str(column): _analyze_series(
            prices=prices,
            column=str(column),
            label=str(column),
            series_type="index",
            break_pct=INDEX_BREAK_PCT,
        )
        for column in _context_columns(prices.columns, target)
    }
    confirmation = _broad_market_confirmation(stock, context)
    return {
        "principle": (
            "Chapter 15 treats major trendlines as long-range monthly perspective. "
            "Scale choice matters more than in intermediate trends, and broad indexes normally produce cleaner trendlines than individual stocks."
        ),
        "primary_timeframe": "monthly",
        "decision_rule": (
            "Use major trendlines as long-horizon regime and risk controls. "
            "Benchmark confirmation is used only when benchmark/index data is supplied."
        ),
        "preferred": stock.get("major_trendline", {}),
        "stock_major_trend": stock,
        "scale_comparison": stock.get("scale_comparison", {}),
        "broad_market_confirmation": confirmation,
        "context_major_trends": context,
        "technical_method_card": chapter_15_major_trendlines_method_card(target_column=target),
    }


def latest_chapter_15_major_trendlines(prices: pd.DataFrame, target_column: str = "close") -> dict[str, Any]:
    """Return stock-only Chapter 15 diagnostics for chart overlays."""

    return _analyze_series(
        prices=prices,
        column=target_column.lower(),
        label="stock",
        series_type="stock",
        break_pct=STOCK_BREAK_PCT,
    )


def chapter_15_major_trendlines_method_card(target_column: str = "close") -> dict[str, Any]:
    return {
        "name": "edwards_magee_chapter_15_major_trendlines",
        "version": "chapter_15_alignment_v1",
        "target_column": target_column.lower(),
        "implemented_controls": [
            "monthly_major_trendline_analysis",
            "log_vs_linear_scale_comparison",
            "major_trend_shape_classification",
            "post_base_anchor_selection",
            "stock_three_percent_major_break_rule",
            "index_two_percent_major_break_rule",
            "optional_broad_market_confirmation",
            "major_bear_trendline_warning_only",
        ],
        "scale_rules": {
            "speculative_or_accelerating": "semilog lines often give earlier and better major-trend warnings",
            "investment_or_straight_arithmetic": "linear scale can fit some high-grade issues better",
            "mixed": "record both fits and avoid overconfidence",
        },
        "anchor_rules": {
            "major_bull": "prefer the first meaningful intermediate bottom after the accumulation/base low",
            "major_bear": "major bear trendlines are less dependable and are warning-only unless confirmed elsewhere",
        },
        "benchmark_policy": (
            "Benchmark/sector/index confirmation is optional. Missing context is reported as Unavailable and never blocks execution."
        ),
        "decision_use": (
            "Major uptrend-line breaks can block aggressive Buy actions; active major bull trendlines can block premature Sell actions; "
            "broad-market divergence is a warning unless context data is supplied."
        ),
    }


def _analyze_series(
    prices: pd.DataFrame,
    column: str,
    label: str,
    series_type: str,
    break_pct: float,
) -> dict[str, Any]:
    if column not in prices.columns:
        return _empty_series(label, series_type, "InsufficientData", f"missing `{column}` column")
    monthly = _monthly_frame(prices, column)
    if len(monthly) < MIN_MONTHLY_BARS:
        return _empty_series(
            label,
            series_type,
            "InsufficientData",
            f"need at least {MIN_MONTHLY_BARS} monthly bars for Chapter 15 major trendlines",
            rows=len(monthly),
        )
    monthly = monthly.tail(min(LOOKBACK_MONTHS, len(monthly)))
    close = monthly["close"].astype(float)
    regime = _major_regime(close)
    scale_comparison = _scale_comparison(close)
    scale = str(scale_comparison.get("preferred_scale") or "log")
    major_line = _major_trendline(
        monthly=monthly,
        regime=regime,
        scale=scale,
        label=label,
        series_type=series_type,
        break_pct=break_pct,
    )
    return {
        "label": label,
        "series_type": series_type,
        "state": "Measured",
        "timeframe": "monthly",
        "rows": int(len(monthly)),
        "start_date": _date_string(monthly.index[0]),
        "end_date": _date_string(monthly.index[-1]),
        "latest_close": _finite_or_none(close.iloc[-1]),
        "major_regime": regime,
        "scale_comparison": scale_comparison,
        "major_trendline": major_line,
    }


def _major_regime(close: pd.Series) -> dict[str, Any]:
    latest = float(close.iloc[-1])
    lookback = min(60, len(close) - 1)
    prior = float(close.iloc[-lookback - 1]) if lookback > 0 else float(close.iloc[0])
    total_return = latest / float(close.iloc[0]) - 1.0 if float(close.iloc[0]) > 0 else 0.0
    lookback_return = latest / prior - 1.0 if prior > 0 else 0.0
    sma_12 = float(close.rolling(12, min_periods=6).mean().iloc[-1])
    sma_36 = float(close.rolling(36, min_periods=18).mean().iloc[-1])
    drawdown = latest / float(close.max()) - 1.0 if float(close.max()) > 0 else 0.0
    up_votes = int(lookback_return >= 0.15) + int(total_return > 0.25) + int(sma_12 > sma_36) + int(drawdown > -0.18)
    down_votes = int(lookback_return <= -0.15) + int(total_return < -0.20) + int(sma_12 < sma_36) + int(drawdown <= -0.25)
    if up_votes >= 3 and up_votes > down_votes:
        state = "Bullish"
    elif down_votes >= 2 and down_votes > up_votes:
        state = "Bearish"
    else:
        state = "Neutral"
    return {
        "state": state,
        "lookback_months": int(lookback),
        "lookback_return": _finite_or_none(lookback_return),
        "full_period_return": _finite_or_none(total_return),
        "drawdown_from_major_high": _finite_or_none(drawdown),
        "sma_12": _finite_or_none(sma_12),
        "sma_36": _finite_or_none(sma_36),
        "up_votes": int(up_votes),
        "down_votes": int(down_votes),
    }


def _scale_comparison(close: pd.Series) -> dict[str, Any]:
    values = close.dropna().astype(float)
    x = np.arange(len(values), dtype=float)
    linear_r2 = _fit_r2(x, values.to_numpy(dtype=float))
    log_values = np.log(values.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    log_r2 = _fit_r2(np.arange(len(log_values), dtype=float), log_values.to_numpy(dtype=float)) if len(log_values) >= 3 else 0.0
    thirds = np.array_split(values.to_numpy(dtype=float), 3)
    linear_slopes = [_simple_slope(chunk) for chunk in thirds if len(chunk) >= 3]
    log_thirds = np.array_split(np.log(values.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float), 3)
    log_slopes = [_simple_slope(chunk) for chunk in log_thirds if len(chunk) >= 3]
    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0) if float(values.iloc[0]) > 0 else 0.0
    preferred = "log" if log_r2 >= linear_r2 + 0.02 else "linear" if linear_r2 >= log_r2 + 0.04 else "log"
    shape = "mixed_or_unclear"
    if total_return > 0.35 and log_r2 >= linear_r2 + 0.03:
        shape = "accelerating_speculative_semilog_preferred"
    elif total_return > 0.20 and linear_r2 >= log_r2 + 0.04:
        shape = "straight_arithmetic_investment_type"
    elif len(log_slopes) == 3 and total_return > 0.10 and log_slopes[-1] < log_slopes[0] * 0.55:
        shape = "decelerating_investment_preferred_style"
    return {
        "preferred_scale": preferred,
        "major_trend_shape": shape,
        "linear_fit_r2": _finite_or_none(linear_r2),
        "log_fit_r2": _finite_or_none(log_r2),
        "linear_slope_early_mid_late": [_finite_or_none(value) for value in linear_slopes],
        "log_slope_early_mid_late": [_finite_or_none(value) for value in log_slopes],
        "reason": _scale_reason(preferred, shape),
    }


def _major_trendline(
    monthly: pd.DataFrame,
    regime: dict[str, Any],
    scale: str,
    label: str,
    series_type: str,
    break_pct: float,
) -> dict[str, Any]:
    close = monthly["close"].astype(float)
    high = monthly["high"].astype(float)
    low = monthly["low"].astype(float)
    state = str(regime.get("state", "Neutral"))
    if state == "Bearish":
        pivots = _pivot_events(monthly, _confirmed_pivots(high, low)[0], "high")
        kind = "major_downtrend"
    else:
        pivots = _pivot_events(monthly, _confirmed_pivots(high, low)[1], "low")
        kind = "major_uptrend"
    if len(pivots) < 2:
        return _empty_pattern(label, series_type, "NoPattern", "not enough confirmed monthly pivots")
    anchors = _select_anchors(pivots, kind)
    if anchors is None:
        return _empty_pattern(label, series_type, "NoPattern", "no independent major trendline anchors")
    first, second = anchors
    line = _line_from_points(first, second, scale=scale)
    if line is None:
        return _empty_pattern(label, series_type, "NoPattern", "invalid major trendline anchors")
    latest_position = len(monthly) - 1
    latest_close = float(close.iloc[-1])
    current_value = _line_value(line, latest_position, scale=scale)
    penetration = _penetration_pct(latest_close, current_value, kind)
    effective_break = penetration >= break_pct
    status = "MajorTrendlineBreak" if effective_break else "ActiveMajorTrendline"
    if kind == "major_downtrend" and effective_break:
        status = "MajorBearTrendlineBreakWarning"
    touches = _line_touches(pivots, line, scale=scale, tolerance=TOUCH_TOLERANCE_PCT, start_position=int(first["position"]))
    authority = _authority_score(len(touches), latest_position - int(first["position"]), len(monthly), scale, series_type)
    return {
        "pattern": "Chapter15MajorTrendline",
        "status": status,
        "direction": _line_direction(kind, effective_break),
        "kind": kind,
        "series_type": series_type,
        "label": label,
        "timeframe": "monthly",
        "scale": scale,
        "score": _finite_or_none(authority),
        "authority_score": _finite_or_none(authority),
        "break_threshold_pct": _finite_or_none(break_pct),
        "effective_major_break": bool(effective_break),
        "latest_close": _finite_or_none(latest_close),
        "current_value": _finite_or_none(current_value),
        "latest_penetration_pct": _finite_or_none(penetration),
        "major_bear_warning_only": bool(kind == "major_downtrend"),
        "anchor_selection": "post_base_intermediate_bottom" if kind == "major_uptrend" else "major_bear_rally_high",
        "anchor_points": [_anchor_payload(first), _anchor_payload(second)],
        "touch_count": int(len(touches)),
        "touches": touches[-6:],
        "line": {
            "start_date": first["date"],
            "second_date": second["date"],
            "end_date": _date_string(monthly.index[-1]),
            "start_value": _finite_or_none(_line_value(line, int(first["position"]), scale=scale)),
            "second_value": _finite_or_none(_line_value(line, int(second["position"]), scale=scale)),
            "current_value": _finite_or_none(current_value),
            "slope_per_bar": _finite_or_none(line["slope"]),
            "intercept": _finite_or_none(line["intercept"]),
        },
        "reliability_notes": _line_notes(kind, effective_break, series_type, scale, authority),
    }


def _select_anchors(events: list[dict[str, Any]], kind: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    ordered = sorted(events, key=lambda item: int(item["position"]))
    if len(ordered) < 2:
        return None
    if kind == "major_uptrend":
        absolute = min(ordered, key=lambda item: float(item["price"]))
        later = [
            event
            for event in ordered
            if int(event["position"]) >= int(absolute["position"]) + MIN_PIVOT_SPACING
            and float(event["price"]) >= float(absolute["price"]) * 1.02
        ]
        first = later[0] if later else absolute
        seconds = [
            event
            for event in ordered
            if int(event["position"]) >= int(first["position"]) + MIN_PIVOT_SPACING
            and float(event["price"]) > float(first["price"]) * 1.01
        ]
    else:
        first = max(ordered[:-1], key=lambda item: float(item["price"]))
        seconds = [
            event
            for event in ordered
            if int(event["position"]) >= int(first["position"]) + MIN_PIVOT_SPACING
            and float(event["price"]) < float(first["price"]) * 0.99
        ]
    if not seconds:
        return None
    return first, seconds[0]


def _broad_market_confirmation(stock: dict[str, Any], context: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not context:
        return {
            "status": "Unavailable",
            "reason": "No benchmark, sector, market, or index series supplied.",
            "confirming_contexts": [],
            "conflicting_contexts": [],
            "neutral_contexts": [],
            "confirmation_ratio": None,
        }
    stock_state = stock.get("major_regime", {}).get("state", "Neutral")
    confirming = []
    conflicting = []
    neutral = []
    for name, diagnostics in context.items():
        context_state = diagnostics.get("major_regime", {}).get("state", "Neutral")
        line = diagnostics.get("major_trendline", {})
        entry = {
            "name": name,
            "state": context_state,
            "trendline_status": line.get("status"),
            "scale": line.get("scale"),
        }
        if context_state == stock_state and context_state != "Neutral":
            confirming.append(entry)
        elif context_state != "Neutral" and stock_state != "Neutral" and context_state != stock_state:
            conflicting.append(entry)
        else:
            neutral.append(entry)
    status = "Confirmed" if confirming and not conflicting else "Divergent" if conflicting else "Mixed"
    return {
        "status": status,
        "stock_major_state": stock_state,
        "confirming_contexts": confirming,
        "conflicting_contexts": conflicting,
        "neutral_contexts": neutral,
        "confirmation_ratio": float(len(confirming) / len(context)) if context else None,
    }


def _monthly_frame(prices: pd.DataFrame, column: str) -> pd.DataFrame:
    frame = prices.copy().sort_index()
    frame.index = pd.DatetimeIndex(frame.index)
    close = pd.to_numeric(frame[column], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce") if column == "close" and "high" in frame.columns else close
    low = pd.to_numeric(frame["low"], errors="coerce") if column == "close" and "low" in frame.columns else close
    monthly = pd.DataFrame({"close": close, "high": high, "low": low}).resample("ME").agg(
        {"close": "last", "high": "max", "low": "min"}
    )
    return monthly.dropna(subset=["close"])


def _confirmed_pivots(high: pd.Series, low: pd.Series) -> tuple[pd.Series, pd.Series]:
    window = PIVOT_LEFT_BARS + PIVOT_RIGHT_BARS + 1
    raw_high = high == high.rolling(window=window, center=True).max()
    raw_low = low == low.rolling(window=window, center=True).min()
    confirmed_high = high.shift(PIVOT_RIGHT_BARS).where(raw_high.shift(PIVOT_RIGHT_BARS, fill_value=False), np.nan)
    confirmed_low = low.shift(PIVOT_RIGHT_BARS).where(raw_low.shift(PIVOT_RIGHT_BARS, fill_value=False), np.nan)
    return confirmed_high, confirmed_low


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


def _line_from_points(first: dict[str, Any], second: dict[str, Any], scale: str) -> dict[str, float] | None:
    first_position = int(first["position"])
    second_position = int(second["position"])
    first_price = float(first["price"])
    second_price = float(second["price"])
    if second_position <= first_position or first_price <= 0 or second_price <= 0:
        return None
    if scale == "linear":
        slope = (second_price - first_price) / (second_position - first_position)
        intercept = first_price - slope * first_position
    else:
        slope = (np.log(second_price) - np.log(first_price)) / (second_position - first_position)
        intercept = np.log(first_price) - slope * first_position
    return {"slope": float(slope), "intercept": float(intercept)}


def _line_value(line: dict[str, float], position: int, scale: str) -> float:
    value = float(line["intercept"] + line["slope"] * position)
    return value if scale == "linear" else float(np.exp(value))


def _line_touches(
    pivot_events: list[dict[str, Any]],
    line: dict[str, float],
    scale: str,
    tolerance: float,
    start_position: int,
) -> list[dict[str, Any]]:
    touches = []
    for event in pivot_events:
        position = int(event["position"])
        if position < start_position:
            continue
        line_value = _line_value(line, position, scale=scale)
        distance = abs(float(event["price"]) - line_value) / line_value if line_value > 0 else np.nan
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


def _penetration_pct(close_value: float, line_value: float, kind: str) -> float:
    if line_value <= 0:
        return 0.0
    if kind == "major_uptrend":
        return float(max(0.0, (line_value - close_value) / line_value))
    return float(max(0.0, (close_value - line_value) / line_value))


def _authority_score(touch_count: int, duration_bars: int, lookback_bars: int, scale: str, series_type: str) -> float:
    touch_component = 0.30 * _clip01(touch_count / 4.0)
    duration_component = 0.30 * _clip01(duration_bars / max(lookback_bars * 0.65, 1.0))
    scale_component = 0.12 if scale == "log" else 0.09
    context_component = 0.12 if series_type == "index" else 0.06
    return float(min(0.99, 0.18 + touch_component + duration_component + scale_component + context_component))


def _line_direction(kind: str, effective_break: bool) -> str:
    if kind == "major_uptrend" and effective_break:
        return "bearish_major_warning"
    if kind == "major_downtrend" and effective_break:
        return "bullish_major_warning"
    if kind == "major_uptrend":
        return "bullish_major_context"
    return "bearish_major_context"


def _line_notes(kind: str, effective_break: bool, series_type: str, scale: str, authority: float) -> list[str]:
    notes = []
    if scale == "log":
        notes.append("Semilog scale is used because Chapter 15 treats percentage geometry as important for major trends.")
    else:
        notes.append("Linear scale fit is better for this long-term series.")
    if kind == "major_downtrend":
        notes.append("Major bear trendlines are warning-only; Chapter 15 treats them as less dependable.")
    if effective_break:
        notes.append("Latest close exceeds the Chapter 15 major trendline break threshold.")
    if series_type == "index":
        notes.append("Index/average series use the tighter 2% break threshold.")
    if authority >= 0.70:
        notes.append("Major trendline authority is high for the available monthly history.")
    return notes[:5]


def _scale_reason(preferred: str, shape: str) -> str:
    if shape == "accelerating_speculative_semilog_preferred":
        return "Long-term advance is more coherent on semilog scale, consistent with speculative accelerating issues."
    if shape == "straight_arithmetic_investment_type":
        return "Long-term advance is more coherent on linear scale, consistent with investment-type issues."
    if shape == "decelerating_investment_preferred_style":
        return "Long-term advance is decelerating; treat major trendline interpretation conservatively."
    return f"{preferred} scale selected by relative fit; trend shape is mixed."


def _fit_r2(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return 0.0
    x = x[finite]
    y = y[finite]
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _simple_slope(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 3:
        return 0.0
    x = np.arange(len(finite), dtype=float)
    slope, _ = np.polyfit(x, finite, 1)
    return float(slope)


def _context_columns(columns: pd.Index, target: str) -> list[str]:
    result = []
    base = {"open", "high", "low", target, "volume", "dividends", "stock_splits"}
    for column in columns:
        clean = str(column).lower()
        if clean in base:
            continue
        if clean.startswith(CONTEXT_PREFIXES):
            result.append(str(column))
    return result


def _empty_series(label: str, series_type: str, status: str, reason: str, rows: int = 0) -> dict[str, Any]:
    return {
        "label": label,
        "series_type": series_type,
        "state": status,
        "timeframe": "monthly",
        "rows": int(rows),
        "reason": reason,
        "major_regime": {"state": "Neutral", "reason": reason},
        "scale_comparison": {"status": status, "reason": reason},
        "major_trendline": _empty_pattern(label, series_type, status, reason),
    }


def _empty_pattern(label: str, series_type: str, status: str, reason: str) -> dict[str, Any]:
    return {
        "pattern": "Chapter15MajorTrendline",
        "status": status,
        "direction": "neutral",
        "label": label,
        "series_type": series_type,
        "timeframe": "monthly",
        "score": 0.0,
        "reason": reason,
    }


def _anchor_payload(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": event.get("date"),
        "position": int(event.get("position", 0)),
        "price": _finite_or_none(event.get("price")),
        "kind": event.get("kind"),
    }


def _date_string(value: Any) -> str:
    try:
        return str(pd.Timestamp(value).date())
    except Exception:
        return str(value)


def _finite_or_none(value: Any) -> float | None:
    try:
        output = float(value)
    except Exception:
        return None
    return output if np.isfinite(output) else None


def _clip01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))
