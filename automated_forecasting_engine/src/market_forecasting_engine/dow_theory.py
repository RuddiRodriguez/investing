from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


CONTEXT_PREFIXES = ("benchmark_", "sector_", "market_", "index_")
PRIMARY_LOOKBACK_WEEKS = 52
PRIMARY_RETURN_THRESHOLD = 0.08
SECONDARY_LOOKBACK_DAYS = 63
SECONDARY_RETURN_THRESHOLD = 0.04
MINOR_LOOKBACK_DAYS = 10
MINOR_RETURN_THRESHOLD = 0.015
LINE_WIDTH_THRESHOLD = 0.05
SIGNAL_TRANSACTION_COST_BPS = 5.0


def analyze_dow_theory(
    prices: pd.DataFrame,
    target_column: str = "close",
    transaction_cost_bps: float = SIGNAL_TRANSACTION_COST_BPS,
) -> dict[str, Any]:
    """Build Dow Theory-inspired market-action diagnostics.

    This is a modernized diagnostic layer: it keeps Dow's ideas of trend
    hierarchy, confirmation, closing-price signals, volume evidence, and lines,
    while using benchmark/sector context instead of the original rail/industrial
    averages.
    """

    frame = prices.copy().sort_index()
    frame.index = pd.DatetimeIndex(frame.index)
    target = target_column.lower()
    if target not in frame.columns:
        raise ValueError(f"Dow Theory diagnostics require `{target}` prices.")

    close = pd.to_numeric(frame[target], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce") if "high" in frame.columns else close
    low = pd.to_numeric(frame["low"], errors="coerce") if "low" in frame.columns else close
    volume = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame.columns else None

    stock = _analyze_price_series(close=close, high=high, low=low, volume=volume, label="stock")
    context = {
        str(column): _analyze_price_series(
            close=pd.to_numeric(frame[column], errors="coerce"),
            high=None,
            low=None,
            volume=None,
            label=str(column),
        )
        for column in _context_columns(frame.columns, target)
    }
    confirmation = _trend_confirmation(stock, context)
    primary_regime = _rolling_primary_regime(close)
    signal_lag = _signal_lag_diagnostics(close=close, regime=primary_regime)
    sensitivity = _sensitivity_analysis(close)
    regime_backtest = _technical_regime_backtest(
        close=close,
        regime=primary_regime,
        transaction_cost_bps=transaction_cost_bps,
    )
    method_card = technical_method_card(target_column=target)

    return {
        "principle": "Dow Theory-inspired diagnostics using closes, trend hierarchy, confirmation, volume, lines, and continuation-until-reversal.",
        "primary_trend": stock["primary_trend"],
        "secondary_trend": stock["secondary_trend"],
        "minor_trend": stock["minor_trend"],
        "swing_structure": stock["swing_structure"],
        "retracement": stock["retracement"],
        "volume_confirmation": stock["volume_confirmation"],
        "line_pattern": stock["line_pattern"],
        "close_confirmed_signals": stock["close_confirmed_signals"],
        "trend_confirmation": confirmation,
        "context_trends": context,
        "continuation_rule": _continuation_rule(stock, confirmation),
        "chapter_4_defect_diagnostics": {
            "principle": "Dow-style signals are treated as useful but late, interpretive, and imperfect; they must be tested against full-cycle risk.",
            "signal_lag": signal_lag,
            "sensitivity_analysis": sensitivity,
            "ambiguity_score": sensitivity["ambiguity_score"],
            "regime_backtest": regime_backtest,
            "known_limitations": [
                "Confirmed trend signals can arrive after a meaningful part of the move has already occurred.",
                "Trend interpretation can change when lookbacks or thresholds are varied.",
                "Sideways line/range periods may require a wait state instead of a directional action.",
                "Primary-trend diagnostics are a regime filter, not a complete stock-selection model.",
            ],
        },
        "technical_method_card": method_card,
    }


def _analyze_price_series(
    close: pd.Series,
    high: pd.Series | None,
    low: pd.Series | None,
    volume: pd.Series | None,
    label: str,
) -> dict[str, Any]:
    clean_close = close.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean_close) < 40:
        empty_trend = _empty_trend(label)
        return {
            "label": label,
            "primary_trend": empty_trend,
            "secondary_trend": empty_trend,
            "minor_trend": empty_trend,
            "swing_structure": {"state": "InsufficientData"},
            "retracement": {"state": "InsufficientData"},
            "volume_confirmation": {"state": "Unavailable"},
            "line_pattern": {"state": "InsufficientData"},
            "close_confirmed_signals": {"state": "InsufficientData"},
        }

    primary = _trend_state(
        clean_close.resample("W-FRI").last().dropna(),
        lookback=PRIMARY_LOOKBACK_WEEKS,
        threshold=PRIMARY_RETURN_THRESHOLD,
        label="primary",
    )
    secondary = _trend_state(
        clean_close,
        lookback=SECONDARY_LOOKBACK_DAYS,
        threshold=SECONDARY_RETURN_THRESHOLD,
        label="secondary",
    )
    minor = _trend_state(
        clean_close,
        lookback=MINOR_LOOKBACK_DAYS,
        threshold=MINOR_RETURN_THRESHOLD,
        label="minor",
    )
    swing_structure = _swing_structure(clean_close)
    retracement = _retracement_state(clean_close)
    line_pattern = _line_pattern(clean_close)
    close_signals = _close_confirmed_signals(
        close=clean_close,
        high=high.reindex(clean_close.index) if high is not None else None,
        low=low.reindex(clean_close.index) if low is not None else None,
    )
    volume_confirmation = _volume_confirmation(clean_close, volume.reindex(clean_close.index) if volume is not None else None, primary["state"])
    return {
        "label": label,
        "primary_trend": primary,
        "secondary_trend": secondary,
        "minor_trend": minor,
        "swing_structure": swing_structure,
        "retracement": retracement,
        "volume_confirmation": volume_confirmation,
        "line_pattern": line_pattern,
        "close_confirmed_signals": close_signals,
    }


def _trend_state(close: pd.Series, lookback: int, threshold: float, label: str) -> dict[str, Any]:
    values = close.dropna()
    if len(values) < max(lookback, 10):
        return _empty_trend(label)
    latest = float(values.iloc[-1])
    prior = float(values.iloc[-lookback])
    total_return = latest / prior - 1 if prior else 0.0
    fast_window = max(3, min(20, lookback // 3))
    slow_window = max(fast_window + 1, min(50, lookback))
    fast = float(values.rolling(fast_window).mean().iloc[-1])
    slow = float(values.rolling(slow_window).mean().iloc[-1])
    swing = _swing_structure(values, lookback=max(lookback, 30))

    up_votes = int(total_return > threshold) + int(fast > slow) + int(latest > slow) + int(swing["state"] == "HigherHighsHigherLows")
    down_votes = int(total_return < -threshold) + int(fast < slow) + int(latest < slow) + int(swing["state"] == "LowerHighsLowerLows")
    if up_votes >= 2 and up_votes > down_votes:
        state = "Bullish"
    elif down_votes >= 2 and down_votes > up_votes:
        state = "Bearish"
    else:
        state = "Neutral"

    return {
        "label": label,
        "state": state,
        "lookback_bars": int(lookback),
        "lookback_return": float(total_return),
        "fast_average": fast,
        "slow_average": slow,
        "swing_state": swing["state"],
        "up_votes": int(up_votes),
        "down_votes": int(down_votes),
    }


def _swing_structure(close: pd.Series, lookback: int = 126, left: int = 3, right: int = 3) -> dict[str, Any]:
    pivot_high, pivot_low = _confirmed_close_pivots(close, left=left, right=right)
    scope = close.tail(lookback)
    highs = pivot_high.reindex(scope.index).dropna()
    lows = pivot_low.reindex(scope.index).dropna()
    high_values = highs[highs > 0]
    low_values = lows[lows > 0]

    higher_high = _last_is_higher(high_values)
    higher_low = _last_is_higher(low_values)
    lower_high = _last_is_lower(high_values)
    lower_low = _last_is_lower(low_values)

    if higher_high and higher_low:
        state = "HigherHighsHigherLows"
    elif lower_high and lower_low:
        state = "LowerHighsLowerLows"
    elif len(high_values) >= 2 and len(low_values) >= 2:
        state = "Mixed"
    else:
        state = "InsufficientPivots"

    return {
        "state": state,
        "lookback_bars": int(lookback),
        "last_pivot_high": _pivot_payload(high_values.iloc[-1:], "high") if len(high_values) else None,
        "previous_pivot_high": _pivot_payload(high_values.iloc[-2:-1], "high") if len(high_values) >= 2 else None,
        "last_pivot_low": _pivot_payload(low_values.iloc[-1:], "low") if len(low_values) else None,
        "previous_pivot_low": _pivot_payload(low_values.iloc[-2:-1], "low") if len(low_values) >= 2 else None,
        "higher_high": bool(higher_high),
        "higher_low": bool(higher_low),
        "lower_high": bool(lower_high),
        "lower_low": bool(lower_low),
    }


def _retracement_state(close: pd.Series) -> dict[str, Any]:
    pivot_high, pivot_low = _confirmed_close_pivots(close, left=3, right=3)
    pivots = []
    for date, value in pivot_high[pivot_high > 0].items():
        pivots.append((pd.Timestamp(date), "high", float(value)))
    for date, value in pivot_low[pivot_low > 0].items():
        pivots.append((pd.Timestamp(date), "low", float(value)))
    pivots = sorted(pivots, key=lambda item: item[0])
    if len(pivots) < 2:
        return {"state": "InsufficientPivots"}

    last = pivots[-1]
    previous = next((pivot for pivot in reversed(pivots[:-1]) if pivot[1] != last[1]), None)
    if previous is None:
        return {"state": "InsufficientOppositePivots"}

    latest_close = float(close.iloc[-1])
    move = abs(last[2] - previous[2])
    if move <= 0:
        return {"state": "FlatPriorSwing"}

    if previous[1] == "low" and last[1] == "high":
        depth = (last[2] - latest_close) / move
        direction = "DownRetracementAfterAdvance"
    else:
        depth = (latest_close - last[2]) / move
        direction = "UpRetracementAfterDecline"
    depth = float(max(0.0, depth))
    if depth < 1 / 3:
        state = "Shallow"
    elif depth <= 2 / 3:
        state = "NormalSecondary"
    elif depth <= 1.0:
        state = "Deep"
    else:
        state = "ReversalRisk"

    duration = int(len(close.loc[last[0] :]))
    return {
        "state": state,
        "direction": direction,
        "depth": depth,
        "duration_bars": duration,
        "secondary_candidate": bool(duration >= 15 and depth >= 1 / 3),
        "prior_swing_start": {"date": str(previous[0].date()), "type": previous[1], "price": previous[2]},
        "prior_swing_end": {"date": str(last[0].date()), "type": last[1], "price": last[2]},
    }


def _volume_confirmation(close: pd.Series, volume: pd.Series | None, primary_state: str) -> dict[str, Any]:
    if volume is None or volume.dropna().empty:
        return {"state": "Unavailable", "reason": "Volume data is missing."}
    returns = close.diff()
    recent_volume = volume.tail(63)
    recent_returns = returns.reindex(recent_volume.index)
    up_volume = float(recent_volume[recent_returns > 0].mean()) if (recent_returns > 0).any() else np.nan
    down_volume = float(recent_volume[recent_returns < 0].mean()) if (recent_returns < 0).any() else np.nan
    ratio = up_volume / down_volume if np.isfinite(up_volume) and np.isfinite(down_volume) and down_volume else np.nan
    if primary_state == "Bullish":
        confirms = bool(np.isfinite(ratio) and ratio > 1.05)
    elif primary_state == "Bearish":
        confirms = bool(np.isfinite(ratio) and ratio < 0.95)
    else:
        confirms = False
    return {
        "state": "ConfirmsTrend" if confirms else "DoesNotConfirmTrend",
        "lookback_bars": 63,
        "up_day_average_volume": _finite_or_none(up_volume),
        "down_day_average_volume": _finite_or_none(down_volume),
        "up_down_volume_ratio": _finite_or_none(ratio),
        "confirms_primary_trend": confirms,
    }


def _line_pattern(close: pd.Series) -> dict[str, Any]:
    latest = float(close.iloc[-1])
    for window in (63, 30, 15):
        if len(close) <= window + 1:
            continue
        prior = close.iloc[-window - 1 : -1]
        mean = float(prior.mean())
        if mean == 0 or not np.isfinite(mean):
            continue
        upper = float(prior.max())
        lower = float(prior.min())
        width = (upper - lower) / mean
        if width <= LINE_WIDTH_THRESHOLD:
            if latest > upper:
                state = "BullishLineBreakout"
            elif latest < lower:
                state = "BearishLineBreakdown"
            else:
                state = "ActiveLine"
            return {
                "state": state,
                "lookback_bars": int(window),
                "range_width_pct": float(width),
                "upper_boundary": upper,
                "lower_boundary": lower,
                "latest_close": latest,
                "significance": "High" if window >= 30 and width <= 0.04 else "Medium",
            }
    return {"state": "NoCompactLine", "range_width_threshold": LINE_WIDTH_THRESHOLD}


def technical_method_card(target_column: str = "close") -> dict[str, Any]:
    """Governance metadata for the technical rules used in Dow diagnostics."""

    return {
        "name": "dow_theory_inspired_market_action",
        "version": "chapter_4_alignment_v1",
        "target_column": target_column.lower(),
        "price_basis": "closing prices for signals; OHLC only used for intraday context.",
        "primary_trend": {
            "bar_frequency": "weekly Friday close",
            "lookback_bars": PRIMARY_LOOKBACK_WEEKS,
            "return_threshold": PRIMARY_RETURN_THRESHOLD,
            "votes": ["lookback_return", "fast_average_above_slow_average", "close_above_slow_average", "swing_structure"],
            "minimum_directional_votes": 2,
        },
        "secondary_trend": {
            "bar_frequency": "daily",
            "lookback_bars": SECONDARY_LOOKBACK_DAYS,
            "return_threshold": SECONDARY_RETURN_THRESHOLD,
        },
        "minor_trend": {
            "bar_frequency": "daily",
            "lookback_bars": MINOR_LOOKBACK_DAYS,
            "return_threshold": MINOR_RETURN_THRESHOLD,
        },
        "pivots": {
            "left_bars": 3,
            "right_bars": 3,
            "confirmation": "pivot is not known until the right-side bars have elapsed",
        },
        "line_pattern": {
            "candidate_windows": [63, 30, 15],
            "max_range_width_pct": LINE_WIDTH_THRESHOLD,
        },
        "confirmation": {
            "context_prefixes": list(CONTEXT_PREFIXES),
            "volume_lookback_bars": 63,
            "close_signal_lookback_bars": 63,
        },
        "chapter_4_controls": {
            "signal_lag": "measures the move from the prior swing reference to the primary-regime signal date",
            "sensitivity_analysis": "re-runs primary trend classification across alternative lookbacks and thresholds",
            "regime_backtest": "compares long-only and long/short primary-regime signals with buy-and-hold",
        },
    }


def _rolling_primary_regime(close: pd.Series) -> pd.DataFrame:
    clean = close.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 80:
        return pd.DataFrame({"state": "Neutral", "signal": 0.0}, index=clean.index)

    weekly = clean.resample("W-FRI").last().dropna()
    weekly_return = weekly / weekly.shift(PRIMARY_LOOKBACK_WEEKS) - 1
    fast = weekly.rolling(max(3, PRIMARY_LOOKBACK_WEEKS // 3)).mean()
    slow = weekly.rolling(max(4, min(50, PRIMARY_LOOKBACK_WEEKS))).mean()
    up_votes = (
        (weekly_return > PRIMARY_RETURN_THRESHOLD).astype(int)
        + (fast > slow).astype(int)
        + (weekly > slow).astype(int)
    )
    down_votes = (
        (weekly_return < -PRIMARY_RETURN_THRESHOLD).astype(int)
        + (fast < slow).astype(int)
        + (weekly < slow).astype(int)
    )
    weekly_signal = pd.Series(0.0, index=weekly.index)
    weekly_signal[(up_votes >= 2) & (up_votes > down_votes)] = 1.0
    weekly_signal[(down_votes >= 2) & (down_votes > up_votes)] = -1.0
    daily_signal = weekly_signal.reindex(clean.index, method="ffill").fillna(0.0)
    state = daily_signal.map({1.0: "Bullish", -1.0: "Bearish", 0.0: "Neutral"})
    return pd.DataFrame({"state": state, "signal": daily_signal}, index=clean.index)


def _signal_lag_diagnostics(close: pd.Series, regime: pd.DataFrame) -> dict[str, Any]:
    clean = close.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty or regime.empty:
        return {"state": "InsufficientData"}

    signal = regime["signal"].reindex(clean.index).fillna(0.0)
    latest_signal = float(signal.iloc[-1])
    if latest_signal == 0:
        return {
            "state": "NoActivePrimarySignal",
            "latest_regime": "Neutral",
            "reason": "The primary-regime proxy is neutral, so no current signal lag is measured.",
        }

    change_dates = signal.index[(signal == latest_signal) & (signal.shift(1).fillna(0.0) != latest_signal)]
    if len(change_dates) == 0:
        return {"state": "UnknownSignalStart", "latest_regime": _signal_label(latest_signal)}
    signal_date = pd.Timestamp(change_dates[-1])
    signal_price = float(clean.loc[signal_date])

    pivot_high, pivot_low = _confirmed_close_pivots(clean, left=3, right=3)
    if latest_signal > 0:
        references = pivot_low[(pivot_low > 0) & (pivot_low.index < signal_date)]
        reference_kind = "prior_confirmed_swing_low"
        fallback = clean.loc[:signal_date].tail(126).min()
        reference_date = pd.Timestamp(references.index[-1]) if len(references) else pd.Timestamp(clean.loc[:signal_date].tail(126).idxmin())
        reference_price = float(references.iloc[-1]) if len(references) else float(fallback)
        move_before_signal = signal_price / reference_price - 1 if reference_price else 0.0
        move_since_signal = float(clean.iloc[-1] / signal_price - 1) if signal_price else 0.0
        total_move = float(clean.iloc[-1] / reference_price - 1) if reference_price else 0.0
    else:
        references = pivot_high[(pivot_high > 0) & (pivot_high.index < signal_date)]
        reference_kind = "prior_confirmed_swing_high"
        fallback = clean.loc[:signal_date].tail(126).max()
        reference_date = pd.Timestamp(references.index[-1]) if len(references) else pd.Timestamp(clean.loc[:signal_date].tail(126).idxmax())
        reference_price = float(references.iloc[-1]) if len(references) else float(fallback)
        move_before_signal = reference_price / signal_price - 1 if signal_price else 0.0
        move_since_signal = float(signal_price / clean.iloc[-1] - 1) if clean.iloc[-1] else 0.0
        total_move = float(reference_price / clean.iloc[-1] - 1) if clean.iloc[-1] else 0.0

    missed_fraction = move_before_signal / total_move if total_move > 1e-12 else None
    return {
        "state": "Measured",
        "latest_regime": _signal_label(latest_signal),
        "signal_date": str(signal_date.date()),
        "signal_price": signal_price,
        "reference_kind": reference_kind,
        "reference_date": str(reference_date.date()),
        "reference_price": reference_price,
        "days_from_reference_to_signal": int(len(clean.loc[reference_date:signal_date]) - 1),
        "move_before_signal_pct": float(max(move_before_signal, 0.0)),
        "move_since_signal_pct": float(move_since_signal),
        "total_reference_to_latest_move_pct": float(total_move),
        "missed_move_fraction": _finite_or_none(missed_fraction),
        "interpretation": "This estimates how much of the current primary move had already occurred before confirmation.",
    }


def _sensitivity_analysis(close: pd.Series) -> dict[str, Any]:
    clean = close.replace([np.inf, -np.inf], np.nan).dropna().resample("W-FRI").last().dropna()
    lookbacks = [42, PRIMARY_LOOKBACK_WEEKS, 63]
    thresholds = [0.06, PRIMARY_RETURN_THRESHOLD, 0.10]
    variants = []
    for lookback in lookbacks:
        for threshold in thresholds:
            diagnostics = _trend_state(clean, lookback=lookback, threshold=threshold, label="primary")
            variants.append(
                {
                    "lookback_bars": int(lookback),
                    "return_threshold": float(threshold),
                    "state": diagnostics["state"],
                    "up_votes": diagnostics["up_votes"],
                    "down_votes": diagnostics["down_votes"],
                }
            )
    state_counts = {
        state: int(sum(1 for variant in variants if variant["state"] == state))
        for state in ("Bullish", "Neutral", "Bearish", "InsufficientData")
    }
    active_counts = {state: count for state, count in state_counts.items() if count > 0}
    consensus_state = max(active_counts, key=active_counts.get) if active_counts else "InsufficientData"
    consensus_ratio = float(active_counts.get(consensus_state, 0) / max(len(variants), 1))
    ambiguity_score = float(1.0 - consensus_ratio)
    return {
        "state": "Stable" if ambiguity_score <= 0.25 else "Mixed",
        "variant_count": int(len(variants)),
        "consensus_state": consensus_state,
        "consensus_ratio": consensus_ratio,
        "ambiguity_score": ambiguity_score,
        "state_counts": state_counts,
        "variants": variants,
    }


def _technical_regime_backtest(
    close: pd.Series,
    regime: pd.DataFrame,
    transaction_cost_bps: float,
) -> dict[str, Any]:
    clean = close.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 80 or regime.empty:
        return {"state": "InsufficientData", "rows": int(len(clean))}

    signal = regime["signal"].reindex(clean.index).fillna(0.0)
    returns = np.log(clean).diff().fillna(0.0)
    benchmark = _equity_curve(returns)
    cost = float(transaction_cost_bps) / 10_000.0

    long_only_signal = (signal > 0).astype(float)
    long_short_signal = signal.astype(float)
    long_only = _signal_backtest(returns, long_only_signal, cost)
    long_short = _signal_backtest(returns, long_short_signal, cost)
    benchmark_metrics = _curve_metrics(benchmark, returns=np.expm1(returns.to_numpy(dtype=float)), periods_per_year=252)
    return {
        "state": "Measured",
        "method": "Primary-regime proxy, signals shifted one bar to avoid same-close execution.",
        "rows": int(len(clean)),
        "start_date": str(clean.index[0].date()),
        "end_date": str(clean.index[-1].date()),
        "transaction_cost_bps": float(transaction_cost_bps),
        "buy_and_hold": benchmark_metrics,
        "long_only_regime": long_only,
        "long_short_regime": long_short,
    }


def _signal_backtest(log_returns: pd.Series, signal: pd.Series, transaction_cost: float) -> dict[str, Any]:
    aligned_signal = signal.reindex(log_returns.index).fillna(0.0)
    executable_signal = aligned_signal.shift(1).fillna(0.0)
    signal_change = executable_signal.diff().abs().fillna(executable_signal.abs())
    strategy_log_returns = executable_signal * log_returns - signal_change * transaction_cost
    equity = _equity_curve(strategy_log_returns)
    simple_returns = np.expm1(strategy_log_returns.to_numpy(dtype=float))
    metrics = _curve_metrics(equity, returns=simple_returns, periods_per_year=252)
    metrics.update(
        {
            "trades": int((signal_change > 0).sum()),
            "turnover": float(signal_change.mean()),
            "long_exposure": float((executable_signal > 0).mean()),
            "short_exposure": float((executable_signal < 0).mean()),
            "cash_exposure": float((executable_signal == 0).mean()),
            "days_out_of_market": int((executable_signal == 0).sum()),
        }
    )
    return metrics


def _equity_curve(log_returns: pd.Series) -> pd.Series:
    return np.exp(log_returns.cumsum())


def _curve_metrics(equity: pd.Series, returns: np.ndarray, periods_per_year: int) -> dict[str, Any]:
    final_equity = float(equity.iloc[-1]) if len(equity) else 1.0
    cumulative_return = final_equity - 1
    annualized_return = final_equity ** (periods_per_year / max(len(equity), 1)) - 1 if final_equity > 0 else -1.0
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(np.mean(returns) / std * np.sqrt(periods_per_year)) if std > 1e-12 else 0.0
    running_high = equity.cummax()
    drawdown = equity / running_high - 1
    return {
        "cumulative_return": _finite_or_none(cumulative_return),
        "annualized_return": _finite_or_none(annualized_return),
        "sharpe_ratio": _finite_or_none(sharpe),
        "max_drawdown": _finite_or_none(float(drawdown.min()) if len(drawdown) else 0.0),
    }


def _signal_label(signal: float) -> str:
    if signal > 0:
        return "Bullish"
    if signal < 0:
        return "Bearish"
    return "Neutral"


def _close_confirmed_signals(close: pd.Series, high: pd.Series | None, low: pd.Series | None) -> dict[str, Any]:
    latest = float(close.iloc[-1])
    prior_high_close = close.shift(1).rolling(63).max().iloc[-1]
    prior_low_close = close.shift(1).rolling(63).min().iloc[-1]
    close_breakout = bool(pd.notna(prior_high_close) and latest > prior_high_close)
    close_breakdown = bool(pd.notna(prior_low_close) and latest < prior_low_close)
    intraday_breakout = None
    intraday_breakdown = None
    if high is not None:
        prior_intraday_high = high.shift(1).rolling(63).max().iloc[-1]
        intraday_breakout = bool(pd.notna(prior_intraday_high) and high.iloc[-1] > prior_intraday_high)
    if low is not None:
        prior_intraday_low = low.shift(1).rolling(63).min().iloc[-1]
        intraday_breakdown = bool(pd.notna(prior_intraday_low) and low.iloc[-1] < prior_intraday_low)
    return {
        "state": "CloseBreakout" if close_breakout else "CloseBreakdown" if close_breakdown else "NoCloseSignal",
        "lookback_bars": 63,
        "close_breakout": close_breakout,
        "close_breakdown": close_breakdown,
        "intraday_breakout": intraday_breakout,
        "intraday_breakdown": intraday_breakdown,
        "prior_close_resistance": _finite_or_none(prior_high_close),
        "prior_close_support": _finite_or_none(prior_low_close),
        "latest_close": latest,
    }


def _trend_confirmation(stock: dict[str, Any], context: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stock_state = stock.get("primary_trend", {}).get("state", "Neutral")
    if not context:
        return {
            "status": "Unavailable",
            "reason": "No benchmark or sector context series supplied.",
            "stock_primary_trend": stock_state,
            "confirming_contexts": [],
            "conflicting_contexts": [],
        }
    confirming = []
    conflicting = []
    neutral = []
    for name, diagnostics in context.items():
        state = diagnostics.get("primary_trend", {}).get("state", "Neutral")
        if stock_state != "Neutral" and state == stock_state:
            confirming.append(name)
        elif stock_state != "Neutral" and state in {"Bullish", "Bearish"} and state != stock_state:
            conflicting.append(name)
        else:
            neutral.append(name)
    if stock_state == "Neutral":
        status = "StockTrendNeutral"
    elif confirming and not conflicting:
        status = "Confirmed"
    elif confirming and conflicting:
        status = "MixedConfirmation"
    elif conflicting:
        status = "Divergent"
    else:
        status = "Unconfirmed"
    return {
        "status": status,
        "stock_primary_trend": stock_state,
        "confirming_contexts": confirming,
        "conflicting_contexts": conflicting,
        "neutral_contexts": neutral,
        "confirmation_ratio": float(len(confirming) / len(context)) if context else 0.0,
    }


def _continuation_rule(stock: dict[str, Any], confirmation: dict[str, Any]) -> dict[str, Any]:
    primary = stock.get("primary_trend", {}).get("state", "Neutral")
    close_signal = stock.get("close_confirmed_signals", {}).get("state", "NoCloseSignal")
    swing = stock.get("swing_structure", {}).get("state", "Mixed")
    if close_signal == "CloseBreakdown" and primary == "Bullish":
        instruction = "WatchForConfirmedReversal"
    elif close_signal == "CloseBreakout" and primary == "Bearish":
        instruction = "WatchForConfirmedReversal"
    elif confirmation.get("status") in {"Divergent", "MixedConfirmation"}:
        instruction = "TreatTrendAsDoubtful"
    elif primary in {"Bullish", "Bearish"} and swing in {"HigherHighsHigherLows", "LowerHighsLowerLows"}:
        instruction = "AssumeTrendContinues"
    else:
        instruction = "WaitForClearConfirmation"
    return {
        "state": instruction,
        "primary_trend": primary,
        "close_signal": close_signal,
        "swing_structure": swing,
        "confirmation_status": confirmation.get("status"),
    }


def _confirmed_close_pivots(close: pd.Series, left: int, right: int) -> tuple[pd.Series, pd.Series]:
    window = left + right + 1
    centered_high = close == close.rolling(window=window, center=True).max()
    centered_low = close == close.rolling(window=window, center=True).min()
    confirmed_high = centered_high.shift(right, fill_value=False).astype(bool)
    confirmed_low = centered_low.shift(right, fill_value=False).astype(bool)
    pivot_high = close.shift(right).where(confirmed_high, 0.0)
    pivot_low = close.shift(right).where(confirmed_low, 0.0)
    return pivot_high.fillna(0.0), pivot_low.fillna(0.0)


def _last_is_higher(values: pd.Series) -> bool:
    return bool(len(values) >= 2 and float(values.iloc[-1]) > float(values.iloc[-2]))


def _last_is_lower(values: pd.Series) -> bool:
    return bool(len(values) >= 2 and float(values.iloc[-1]) < float(values.iloc[-2]))


def _pivot_payload(values: pd.Series, kind: str) -> dict[str, Any]:
    if values.empty:
        return {}
    return {"date": str(pd.Timestamp(values.index[-1]).date()), "type": kind, "price": float(values.iloc[-1])}


def _context_columns(columns: pd.Index, target: str) -> list[str]:
    result = []
    for column in columns:
        clean = str(column).lower()
        if clean == target:
            continue
        if clean.startswith(CONTEXT_PREFIXES):
            result.append(str(column))
    return result


def _empty_trend(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "state": "InsufficientData",
        "lookback_bars": 0,
        "lookback_return": None,
        "fast_average": None,
        "slow_average": None,
        "swing_state": "InsufficientData",
        "up_votes": 0,
        "down_votes": 0,
    }


def _finite_or_none(value: Any) -> float | None:
    try:
        output = float(value)
    except Exception:
        return None
    return output if np.isfinite(output) else None
