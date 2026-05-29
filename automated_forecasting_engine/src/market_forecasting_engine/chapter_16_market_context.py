from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DONCHIAN_ENTRY_WINDOWS = (20, 55)
DONCHIAN_EXIT_WINDOWS = (10, 20)
SEASONALITY_MIN_YEARS = 3


def analyze_chapter_16_market_context(
    prices: pd.DataFrame,
    target_column: str = "close",
    ticker: str | None = None,
    security_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze Edwards/Magee Chapter 16 instrument and market-context diagnostics.

    This layer is always report-only. It records whether the market behaves like
    a trading or trending instrument, adds Donchian/Turtle-style context, and
    captures optional futures/commodity data such as open interest when present.
    """

    target = target_column.lower()
    frame = _normalized_frame(prices, target)
    if frame.empty or target not in frame.columns:
        return _empty_context(target, "missing target price column")

    asset_context = _asset_context(frame, ticker=ticker, security_metadata=security_metadata)
    market_character = _market_character(frame, target)
    donchian = _donchian_context(frame, target)
    seasonality = _seasonality_context(frame, target)
    open_interest = _open_interest_context(frame)
    futures_risk = _futures_risk_context(frame, target, asset_context)
    reliability = _pattern_reliability_context(asset_context)
    warnings = _chapter_16_warnings(asset_context, market_character, donchian, seasonality, open_interest, futures_risk)

    return {
        "principle": (
            "Chapter 16 applies classical chart analysis to commodities and futures while recognizing instrument-specific risks "
            "such as contract life, hedging flow, seasonality, open interest, leverage, and fast regime changes."
        ),
        "state": "Measured",
        "decision_policy": {
            "mode": "report_only",
            "influences_final_action": False,
            "reason": "Chapter 16 context is logged and plotted, but it does not vote, block, or alter Buy/Hold/Sell decisions.",
        },
        "asset_context": asset_context,
        "market_character": market_character,
        "donchian_context": donchian,
        "seasonality": seasonality,
        "open_interest": open_interest,
        "futures_risk_context": futures_risk,
        "pattern_reliability_context": reliability,
        "warnings": warnings,
        "technical_method_card": chapter_16_market_context_method_card(target_column=target),
    }


def latest_chapter_16_market_context(prices: pd.DataFrame, target_column: str = "close") -> dict[str, Any]:
    return analyze_chapter_16_market_context(prices, target_column=target_column)


def chapter_16_donchian_history(prices: pd.DataFrame, target_column: str = "close") -> pd.DataFrame:
    frame = _normalized_frame(prices, target_column.lower())
    if frame.empty:
        return pd.DataFrame(index=prices.index)
    high = frame["high"] if "high" in frame.columns else frame[target_column.lower()]
    low = frame["low"] if "low" in frame.columns else frame[target_column.lower()]
    history = pd.DataFrame(index=frame.index)
    for window in DONCHIAN_ENTRY_WINDOWS:
        history[f"donchian_high_{window}"] = high.shift(1).rolling(window).max()
        history[f"donchian_low_{window}"] = low.shift(1).rolling(window).min()
    return history


def chapter_16_market_context_method_card(target_column: str = "close") -> dict[str, Any]:
    return {
        "name": "edwards_magee_chapter_16_market_context",
        "version": "chapter_16_report_only_v1",
        "target_column": target_column.lower(),
        "decision_policy": "report_only_no_action_filter",
        "implemented_controls": [
            "asset_class_inference",
            "trending_vs_trading_market_diagnostic",
            "donchian_20_55_breakout_context",
            "commodity_futures_seasonality_context",
            "optional_open_interest_context",
            "atr_and_contract_risk_context",
            "limit_like_gap_warning",
            "pattern_reliability_notes_by_instrument_type",
        ],
        "optional_columns": [
            "open_interest",
            "contract_multiplier",
            "margin_requirement",
            "days_to_expiration",
            "roll_adjusted",
        ],
        "non_goal": "This method card does not define a Buy/Hold/Sell action filter.",
    }


def _normalized_frame(prices: pd.DataFrame, target: str) -> pd.DataFrame:
    frame = prices.copy().sort_index()
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame.loc[frame.index.notna()]
    output = pd.DataFrame(index=pd.DatetimeIndex(frame.index))
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().any():
            output[str(column).lower()] = converted
    return output.dropna(subset=[target]) if target in output.columns else output


def _asset_context(
    frame: pd.DataFrame,
    ticker: str | None,
    security_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = security_metadata or {}
    metadata_asset = _clean_label(
        metadata.get("asset_class")
        or metadata.get("instrument_type")
        or metadata.get("security_type")
        or metadata.get("quote_type")
        or ""
    )
    ticker_text = str(ticker or metadata.get("ticker") or metadata.get("symbol") or "").upper()
    futures_columns = any(
        column in frame.columns
        for column in ("open_interest", "oi", "contract_multiplier", "margin_requirement", "days_to_expiration")
    )
    if metadata_asset:
        asset_class = metadata_asset
    elif futures_columns or ticker_text.endswith("=F") or ticker_text.startswith(("@", "/")):
        asset_class = "future"
    elif ticker_text.startswith("^"):
        asset_class = "index"
    else:
        asset_class = "equity_or_etf"
    commodity_like = asset_class in {"commodity", "future", "futures", "commodity_future", "commodity_futures"}
    return {
        "asset_class": asset_class,
        "commodity_or_futures_like": bool(commodity_like),
        "ticker": ticker_text or None,
        "source": "security_metadata" if metadata_asset else "inferred_from_columns_or_ticker",
        "has_contract_specific_columns": bool(futures_columns),
        "report_only": True,
    }


def _market_character(frame: pd.DataFrame, target: str) -> dict[str, Any]:
    close = frame[target].astype(float)
    returns = np.log(close.replace(0, np.nan)).diff()
    horizon = min(63, max(20, len(close) // 4))
    if len(close) <= horizon:
        return {
            "status": "InsufficientData",
            "state": "Unknown",
            "reason": "not enough bars for Chapter 16 trending/trading classification",
        }
    recent = close.tail(horizon)
    net_move = abs(float(recent.iloc[-1] / recent.iloc[0] - 1.0)) if float(recent.iloc[0]) > 0 else 0.0
    path = float(recent.pct_change().abs().sum())
    efficiency = net_move / path if path > 0 else 0.0
    realized_vol = float(returns.tail(horizon).std() * np.sqrt(252)) if returns.tail(horizon).notna().sum() > 2 else 0.0
    slope = _rolling_slope(np.log(close.replace(0, np.nan)).tail(horizon))
    latest_return = float(close.iloc[-1] / close.iloc[-horizon] - 1.0) if float(close.iloc[-horizon]) > 0 else 0.0
    if efficiency >= 0.38 and latest_return > 0.06:
        state = "TrendingUp"
    elif efficiency >= 0.38 and latest_return < -0.06:
        state = "TrendingDown"
    elif efficiency <= 0.18:
        state = "TradingRange"
    else:
        state = "Mixed"
    return {
        "status": "Measured",
        "state": state,
        "lookback_bars": int(horizon),
        "efficiency_ratio": _finite_or_none(efficiency),
        "lookback_return": _finite_or_none(latest_return),
        "annualized_volatility": _finite_or_none(realized_vol),
        "log_slope_per_bar": _finite_or_none(slope),
        "interpretation": _market_character_interpretation(state),
    }


def _donchian_context(frame: pd.DataFrame, target: str) -> dict[str, Any]:
    close = frame[target].astype(float)
    high = frame["high"].astype(float) if "high" in frame.columns else close
    low = frame["low"].astype(float) if "low" in frame.columns else close
    latest_close = float(close.iloc[-1])
    channels: dict[str, Any] = {}
    states = []
    for window in DONCHIAN_ENTRY_WINDOWS:
        if len(close) <= window:
            channels[str(window)] = {"status": "InsufficientData"}
            continue
        upper = high.shift(1).rolling(window).max().iloc[-1]
        lower = low.shift(1).rolling(window).min().iloc[-1]
        width = float((upper - lower) / latest_close) if pd.notna(upper) and pd.notna(lower) and latest_close else np.nan
        position = float((latest_close - lower) / (upper - lower)) if pd.notna(upper) and pd.notna(lower) and upper > lower else np.nan
        if pd.notna(upper) and latest_close > float(upper):
            state = "LongBreakout"
        elif pd.notna(lower) and latest_close < float(lower):
            state = "ShortBreakout"
        else:
            state = "InsideChannel"
        states.append(state)
        channels[str(window)] = {
            "status": "Measured",
            "window": int(window),
            "upper": _finite_or_none(upper),
            "lower": _finite_or_none(lower),
            "channel_width_pct": _finite_or_none(width),
            "channel_position": _finite_or_none(position),
            "state": state,
        }
    primary = channels.get("20", {})
    long_breaks = sum(1 for state in states if state == "LongBreakout")
    short_breaks = sum(1 for state in states if state == "ShortBreakout")
    if long_breaks:
        overall = "LongBreakout"
    elif short_breaks:
        overall = "ShortBreakout"
    else:
        overall = "NoBreakout"
    exits = _donchian_exit_levels(high, low)
    return {
        "status": "Measured" if channels else "InsufficientData",
        "overall_state": overall,
        "primary_window": 20,
        "primary": primary,
        "channels": channels,
        "exit_reference": exits,
        "interpretation": _donchian_interpretation(overall),
    }


def _donchian_exit_levels(high: pd.Series, low: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for window in DONCHIAN_EXIT_WINDOWS:
        if len(high) <= window:
            result[str(window)] = {"status": "InsufficientData"}
            continue
        result[str(window)] = {
            "status": "Measured",
            "long_exit_stop": _finite_or_none(low.shift(1).rolling(window).min().iloc[-1]),
            "short_exit_stop": _finite_or_none(high.shift(1).rolling(window).max().iloc[-1]),
        }
    return result


def _seasonality_context(frame: pd.DataFrame, target: str) -> dict[str, Any]:
    close = frame[target].astype(float)
    index = pd.DatetimeIndex(frame.index)
    years = int(index.year.max() - index.year.min() + 1) if len(index) else 0
    monthly = close.resample("ME").last().pct_change().dropna()
    if years < SEASONALITY_MIN_YEARS or monthly.empty:
        return {
            "status": "Unavailable",
            "reason": f"need at least {SEASONALITY_MIN_YEARS} calendar years for a simple seasonal profile",
            "years": years,
        }
    by_month = monthly.groupby(monthly.index.month)
    profile = {
        str(month): {
            "mean_return": _finite_or_none(values.mean()),
            "median_return": _finite_or_none(values.median()),
            "positive_rate": _finite_or_none((values > 0).mean()),
            "sample_count": int(values.count()),
        }
        for month, values in by_month
    }
    current_month = int(index[-1].month)
    current = profile.get(str(current_month), {})
    positive_rate = current.get("positive_rate")
    if positive_rate is None:
        bias = "Unavailable"
    elif positive_rate >= 0.62:
        bias = "SeasonallyBullish"
    elif positive_rate <= 0.38:
        bias = "SeasonallyBearish"
    else:
        bias = "Mixed"
    strength = float(np.nanstd([item["mean_return"] for item in profile.values() if item["mean_return"] is not None]))
    return {
        "status": "Measured",
        "years": years,
        "current_month": current_month,
        "current_month_bias": bias,
        "seasonality_strength": _finite_or_none(strength),
        "monthly_profile": profile,
    }


def _open_interest_context(frame: pd.DataFrame) -> dict[str, Any]:
    column = _first_existing_column(frame, ("open_interest", "openinterest", "oi"))
    if column is None:
        return {
            "status": "Unavailable",
            "reason": "open interest column was not supplied",
            "required_for_equities": False,
        }
    open_interest = pd.to_numeric(frame[column], errors="coerce")
    latest = open_interest.dropna().iloc[-1] if open_interest.notna().any() else np.nan
    change_20 = open_interest.pct_change(20).iloc[-1] if len(open_interest) > 20 else np.nan
    volume_column = _first_existing_column(frame, ("volume",))
    volume_to_oi = None
    if volume_column is not None and pd.notna(latest) and latest > 0:
        latest_volume = pd.to_numeric(frame[volume_column], errors="coerce").iloc[-1]
        volume_to_oi = float(latest_volume / latest) if pd.notna(latest_volume) else None
    return {
        "status": "Measured",
        "column": column,
        "latest_open_interest": _finite_or_none(latest),
        "open_interest_change_20d": _finite_or_none(change_20),
        "volume_to_open_interest": _finite_or_none(volume_to_oi),
        "interpretation": _open_interest_interpretation(change_20),
    }


def _futures_risk_context(frame: pd.DataFrame, target: str, asset_context: dict[str, Any]) -> dict[str, Any]:
    close = frame[target].astype(float)
    high = frame["high"].astype(float) if "high" in frame.columns else close
    low = frame["low"].astype(float) if "low" in frame.columns else close
    true_range = _true_range(high, low, close)
    atr_20 = true_range.rolling(20).mean().iloc[-1] if len(true_range) >= 20 else np.nan
    latest_close = close.iloc[-1]
    atr_pct = float(atr_20 / latest_close) if pd.notna(atr_20) and latest_close else np.nan
    stop_distance_pct = max(0.03, 2.0 * atr_pct) if np.isfinite(atr_pct) else None
    multiplier = _latest_optional(frame, ("contract_multiplier", "multiplier", "point_value"))
    margin = _latest_optional(frame, ("margin_requirement", "initial_margin", "margin"))
    risk_per_unit = float(latest_close * stop_distance_pct) if stop_distance_pct is not None else None
    risk_per_contract = risk_per_unit * multiplier if risk_per_unit is not None and multiplier is not None else None
    margin_risk_ratio = risk_per_contract / margin if risk_per_contract is not None and margin not in (None, 0) else None
    limit_like_gap = _limit_like_gap(frame, target, atr_20)
    if atr_pct is None or not np.isfinite(atr_pct):
        risk_state = "Unavailable"
    elif atr_pct >= 0.06:
        risk_state = "HighVolatility"
    elif atr_pct >= 0.03:
        risk_state = "ElevatedVolatility"
    else:
        risk_state = "NormalVolatility"
    return {
        "status": "Measured",
        "asset_class": asset_context.get("asset_class"),
        "latest_close": _finite_or_none(latest_close),
        "atr_20": _finite_or_none(atr_20),
        "atr_20_pct": _finite_or_none(atr_pct),
        "risk_state": risk_state,
        "suggested_two_atr_stop_distance_pct": _finite_or_none(stop_distance_pct),
        "contract_multiplier": _finite_or_none(multiplier),
        "margin_requirement": _finite_or_none(margin),
        "estimated_risk_per_unit": _finite_or_none(risk_per_unit),
        "estimated_risk_per_contract": _finite_or_none(risk_per_contract),
        "risk_to_margin_ratio": _finite_or_none(margin_risk_ratio),
        "limit_like_gap": limit_like_gap,
        "interpretation": "Risk is informational only and does not change the engine action.",
    }


def _pattern_reliability_context(asset_context: dict[str, Any]) -> dict[str, Any]:
    commodity_like = bool(asset_context.get("commodity_or_futures_like"))
    if commodity_like:
        notes = [
            "Trendlines and simple reversal formations remain useful in futures/commodities.",
            "Long-term support/resistance is weaker for a single expiring contract than for a stock.",
            "Triangles, rectangles, flags, and ordinary gaps should be treated more conservatively.",
            "Stops and position sizing are more important because leverage accelerates losses.",
        ]
        adjustment = "commodity_futures_caution"
    else:
        notes = [
            "Chapter 16 commodity/futures caveats are logged for completeness.",
            "No commodity-specific downgrade is applied to this equity/index-style run.",
        ]
        adjustment = "equity_default"
    return {
        "adjustment": adjustment,
        "chart_features_still_relevant": [
            "trendlines",
            "head_and_shoulders",
            "rounding_patterns",
            "breakouts",
            "gaps",
        ],
        "notes": notes,
    }


def _chapter_16_warnings(
    asset_context: dict[str, Any],
    market_character: dict[str, Any],
    donchian: dict[str, Any],
    seasonality: dict[str, Any],
    open_interest: dict[str, Any],
    futures_risk: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if market_character.get("state") == "TradingRange":
        warnings.append("Chapter 16 context says the market is range-like; mechanical breakout systems are more vulnerable to whipsaw.")
    if donchian.get("overall_state") in {"LongBreakout", "ShortBreakout"}:
        warnings.append("Donchian breakout context is active; this is logged as context, not used as an action filter.")
    if asset_context.get("commodity_or_futures_like") and open_interest.get("status") == "Unavailable":
        warnings.append("Futures/commodity-style run has no open interest column, so hedging-flow confirmation is unavailable.")
    if seasonality.get("status") == "Unavailable":
        warnings.append("Seasonality context is unavailable or weak because the history is too short.")
    if futures_risk.get("risk_state") in {"HighVolatility", "ElevatedVolatility"}:
        warnings.append("ATR risk context is elevated; position sizing should be reviewed outside this forecast engine.")
    if futures_risk.get("limit_like_gap", {}).get("detected"):
        warnings.append("Recent gap/range behavior looks limit-like; ordinary gap interpretation should be conservative.")
    return warnings[:8]


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prior_close = close.shift(1)
    return pd.concat([high - low, (high - prior_close).abs(), (low - prior_close).abs()], axis=1).max(axis=1)


def _limit_like_gap(frame: pd.DataFrame, target: str, atr_20: float | None) -> dict[str, Any]:
    close = frame[target].astype(float)
    high = frame["high"].astype(float) if "high" in frame.columns else close
    low = frame["low"].astype(float) if "low" in frame.columns else close
    open_price = frame["open"].astype(float) if "open" in frame.columns else close.shift(1)
    if len(frame) < 2 or atr_20 is None or not np.isfinite(atr_20) or atr_20 <= 0:
        return {"detected": False, "reason": "insufficient ATR context"}
    prior_high = high.shift(1).iloc[-1]
    prior_low = low.shift(1).iloc[-1]
    gap_up = low.iloc[-1] > prior_high
    gap_down = high.iloc[-1] < prior_low
    gap_size = min(abs(open_price.iloc[-1] - close.shift(1).iloc[-1]), abs(close.iloc[-1] - close.shift(1).iloc[-1]))
    detected = bool((gap_up or gap_down) and gap_size > 1.75 * atr_20)
    return {
        "detected": detected,
        "gap_direction": "up" if gap_up else "down" if gap_down else "none",
        "gap_size_to_atr": _finite_or_none(gap_size / atr_20),
    }


def _market_character_interpretation(state: str) -> str:
    if state == "TrendingUp":
        return "Trend-following context is favorable on the recent lookback."
    if state == "TrendingDown":
        return "Downtrend-following context is favorable on the recent lookback."
    if state == "TradingRange":
        return "Breakout systems are more exposed to whipsaw in this range-like context."
    return "Recent market character is mixed; use chart context as descriptive evidence."


def _donchian_interpretation(state: str) -> str:
    if state == "LongBreakout":
        return "Price is above a prior Donchian channel high; Turtle-style context is bullish but report-only."
    if state == "ShortBreakout":
        return "Price is below a prior Donchian channel low; Turtle-style context is bearish but report-only."
    return "Price remains inside the monitored Donchian channels."


def _open_interest_interpretation(change_20: Any) -> str:
    value = _finite_or_none(change_20)
    if value is None:
        return "Open interest is present but recent change is unavailable."
    if value > 0.10:
        return "Open interest is rising, which can indicate increasing participation."
    if value < -0.10:
        return "Open interest is falling, which can indicate liquidation or fading participation."
    return "Open interest is relatively stable."


def _rolling_slope(series: pd.Series) -> float:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < 3:
        return 0.0
    x = np.arange(len(values), dtype=float)
    slope, _ = np.polyfit(x, values, 1)
    return float(slope)


def _first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    columns = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def _latest_optional(frame: pd.DataFrame, candidates: tuple[str, ...]) -> float | None:
    column = _first_existing_column(frame, candidates)
    if column is None:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return _finite_or_none(values.iloc[-1])


def _empty_context(target_column: str, reason: str) -> dict[str, Any]:
    return {
        "state": "InsufficientData",
        "decision_policy": {
            "mode": "report_only",
            "influences_final_action": False,
            "reason": "Chapter 16 context is unavailable and cannot influence final action.",
        },
        "reason": reason,
        "technical_method_card": chapter_16_market_context_method_card(target_column=target_column),
    }


def _clean_label(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _finite_or_none(value: Any) -> float | None:
    try:
        output = float(value)
    except Exception:
        return None
    return output if np.isfinite(output) else None
