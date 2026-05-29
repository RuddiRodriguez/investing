from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PIVOT_LEFT_BARS = 3
PIVOT_RIGHT_BARS = 3
DEFAULT_TRANSACTION_COST_BPS = 5.0


def analyze_basing_points(
    prices: pd.DataFrame,
    target_column: str = "close",
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
) -> dict[str, Any]:
    """Analyze John Magee-inspired basing-point stops on daily and weekly bars."""

    target = target_column.lower()
    frames = {
        "daily": prices.copy(),
        "weekly": _resample_ohlcv(prices, "W-FRI"),
    }
    timeframes = {
        name: _analyze_timeframe(
            frame=frame,
            timeframe=name,
            target_column=target,
            bars_per_year=252 if name == "daily" else 52,
            transaction_cost_bps=transaction_cost_bps,
        )
        for name, frame in frames.items()
    }
    preferred = timeframes["weekly"] if timeframes["weekly"].get("state") == "Measured" else timeframes["daily"]
    return {
        "principle": "Magee basing-points procedure: use confirmed wave highs/lows to create mechanical stairstep stops and trend-control signals.",
        "primary_timeframe": "weekly",
        "confirmation_rule": f"{PIVOT_RIGHT_BARS}-bars-away confirmed wave highs/lows",
        "preferred": _preferred_payload(preferred),
        "timeframes": timeframes,
        "technical_method_card": basing_points_method_card(target_column=target),
    }


def magee_basing_point_history(prices: pd.DataFrame, target_column: str = "close") -> pd.DataFrame:
    """Return basing-point stop and signal histories for already-selected bars."""

    frame = prices.copy().sort_index()
    if frame.empty:
        return pd.DataFrame(index=frame.index)
    target = target_column.lower()
    close = pd.to_numeric(frame[target], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce") if "high" in frame.columns else close
    low = pd.to_numeric(frame["low"], errors="coerce") if "low" in frame.columns else close
    pivot_high, pivot_low = _confirmed_wave_points(high=high, low=low, left=PIVOT_LEFT_BARS, right=PIVOT_RIGHT_BARS)
    variant_1 = _variant_history(close=close, wave_high=pivot_high, wave_low=pivot_low, variant="variant_1")
    variant_2 = _variant_history(close=close, wave_high=pivot_high, wave_low=pivot_low, variant="variant_2")
    return pd.DataFrame(
        {
            "magee_wave_high": pivot_high,
            "magee_wave_low": pivot_low,
            "magee_variant_1_stop": variant_1["stop"],
            "magee_variant_1_signal": variant_1["signal"],
            "magee_variant_2_stop": variant_2["stop"],
            "magee_variant_2_signal": variant_2["signal"],
        },
        index=frame.index,
    )


def basing_points_method_card(target_column: str = "close") -> dict[str, Any]:
    return {
        "name": "magee_basing_points",
        "version": "chapter_5_alignment_v1",
        "target_column": target_column.lower(),
        "price_basis": "confirmed wave highs/lows from high/low bars; closing price used for stop breaks.",
        "primary_timeframe": "weekly",
        "secondary_timeframe": "daily",
        "pivot_confirmation": {
            "left_bars": PIVOT_LEFT_BARS,
            "right_bars": PIVOT_RIGHT_BARS,
            "rule": "a wave point is recognized only after the right-side bars have elapsed",
        },
        "variant_1": {
            "description": "Long-side basing stops rise from confirmed wave lows.",
            "stop_update": "raise stop when a newly confirmed wave low is above the active stop",
            "exit_signal": "close below active basing stop",
            "reentry_signal": "close above last confirmed wave high",
        },
        "variant_2": {
            "description": "Wave-low stops are ratified by subsequent wave highs.",
            "stop_update": "store confirmed wave lows, then raise stop after a new wave high confirms the advance",
            "exit_signal": "close below active basing stop",
            "reentry_signal": "close above last confirmed wave high",
        },
        "risk_controls": {
            "stop_distance_pct": "distance from latest close to active basing stop",
            "backtests": ["long_only", "long_short"],
            "metrics": ["cumulative_return", "annualized_return", "sharpe_ratio", "max_drawdown", "trades", "exposure"],
        },
    }


def _analyze_timeframe(
    frame: pd.DataFrame,
    timeframe: str,
    target_column: str,
    bars_per_year: int,
    transaction_cost_bps: float,
) -> dict[str, Any]:
    clean = frame.copy().sort_index()
    if clean.empty or target_column not in clean.columns:
        return {"state": "InsufficientData", "timeframe": timeframe, "rows": int(len(clean))}
    clean.index = pd.DatetimeIndex(clean.index)
    close = pd.to_numeric(clean[target_column], errors="coerce")
    high = pd.to_numeric(clean["high"], errors="coerce") if "high" in clean.columns else close
    low = pd.to_numeric(clean["low"], errors="coerce") if "low" in clean.columns else close
    valid = close.dropna()
    if len(valid) < 30:
        return {"state": "InsufficientData", "timeframe": timeframe, "rows": int(len(valid))}

    history = magee_basing_point_history(clean, target_column=target_column).reindex(valid.index)
    variant_payloads = {}
    for variant in ("variant_1", "variant_2"):
        variant_payloads[variant] = _variant_payload(
            close=valid,
            stop=history[f"magee_{variant}_stop"],
            signal=history[f"magee_{variant}_signal"],
            bars_per_year=bars_per_year,
            transaction_cost_bps=transaction_cost_bps,
        )

    preferred_variant = "variant_2" if variant_payloads["variant_2"]["state"] != "InsufficientData" else "variant_1"
    latest = variant_payloads[preferred_variant]["latest"]
    return {
        "state": "Measured",
        "timeframe": timeframe,
        "rows": int(len(valid)),
        "start_date": str(valid.index[0].date()),
        "end_date": str(valid.index[-1].date()),
        "preferred_variant": preferred_variant,
        "latest": latest,
        "wave_points": _wave_point_summary(history),
        "variants": variant_payloads,
    }


def _variant_payload(
    close: pd.Series,
    stop: pd.Series,
    signal: pd.Series,
    bars_per_year: int,
    transaction_cost_bps: float,
) -> dict[str, Any]:
    aligned_stop = stop.reindex(close.index)
    aligned_signal = signal.reindex(close.index).fillna(0.0)
    if aligned_stop.dropna().empty:
        return {"state": "InsufficientData", "latest": {}, "backtests": {}}

    latest_close = float(close.iloc[-1])
    latest_stop = _last_finite(aligned_stop)
    latest_signal = float(aligned_signal.iloc[-1])
    stop_distance = (latest_close - latest_stop) / latest_close if latest_stop is not None and latest_close else None
    transitions = aligned_signal[aligned_signal.diff().fillna(aligned_signal).abs() > 0]
    last_transition = transitions.index[-1] if len(transitions) else None
    latest_state = "Long" if latest_signal > 0 else "Short" if latest_signal < 0 else "Neutral"
    if latest_stop is None:
        stop_status = "NoActiveStop"
    elif latest_close >= latest_stop:
        stop_status = "AboveBasingStop"
    else:
        stop_status = "BelowBasingStop"

    return {
        "state": "Measured",
        "latest": {
            "trend_state": latest_state,
            "stop_status": stop_status,
            "latest_close": latest_close,
            "active_basing_stop": latest_stop,
            "stop_distance_pct": _finite_or_none(stop_distance),
            "last_signal_date": str(pd.Timestamp(last_transition).date()) if last_transition is not None else None,
            "last_signal": latest_state if last_transition is not None else None,
        },
        "backtests": {
            "long_only": _signal_backtest(
                close=close,
                signal=(aligned_signal > 0).astype(float),
                transaction_cost_bps=transaction_cost_bps,
                bars_per_year=bars_per_year,
            ),
            "long_short": _signal_backtest(
                close=close,
                signal=aligned_signal,
                transaction_cost_bps=transaction_cost_bps,
                bars_per_year=bars_per_year,
            ),
        },
        "stop_history_tail": _history_tail(close=close, stop=aligned_stop, signal=aligned_signal, rows=8),
    }


def _variant_history(close: pd.Series, wave_high: pd.Series, wave_low: pd.Series, variant: str) -> pd.DataFrame:
    index = close.index
    active_stop = np.nan
    active_stop_date: pd.Timestamp | None = None
    last_high = np.nan
    last_low = np.nan
    pending_low = np.nan
    pending_low_date: pd.Timestamp | None = None
    signal = 0.0
    stops: list[float] = []
    signals: list[float] = []

    for date in index:
        close_value = float(close.loc[date]) if pd.notna(close.loc[date]) else np.nan
        high_value = float(wave_high.loc[date]) if date in wave_high.index and pd.notna(wave_high.loc[date]) else np.nan
        low_value = float(wave_low.loc[date]) if date in wave_low.index and pd.notna(wave_low.loc[date]) else np.nan

        if np.isfinite(high_value):
            last_high = high_value
            if variant == "variant_2" and np.isfinite(pending_low):
                if not np.isfinite(active_stop) or pending_low > active_stop:
                    active_stop = pending_low
                    active_stop_date = pending_low_date
                pending_low = np.nan
                pending_low_date = None

        if np.isfinite(low_value):
            last_low = low_value
            if variant == "variant_1":
                if not np.isfinite(active_stop) or low_value > active_stop:
                    active_stop = low_value
                    active_stop_date = pd.Timestamp(date)
            else:
                pending_low = low_value
                pending_low_date = pd.Timestamp(date)
                if not np.isfinite(active_stop):
                    active_stop = low_value
                    active_stop_date = pd.Timestamp(date)

        if np.isfinite(close_value):
            if signal >= 0 and np.isfinite(active_stop) and close_value < active_stop:
                signal = -1.0
            elif signal <= 0 and np.isfinite(last_high) and close_value > last_high:
                signal = 1.0
                if np.isfinite(last_low) and (not np.isfinite(active_stop) or last_low > active_stop):
                    active_stop = last_low
                    active_stop_date = pd.Timestamp(date)
            elif signal == 0 and np.isfinite(active_stop) and close_value > active_stop:
                signal = 1.0

        stops.append(float(active_stop) if np.isfinite(active_stop) else np.nan)
        signals.append(signal)

    return pd.DataFrame(
        {
            "stop": pd.Series(stops, index=index),
            "signal": pd.Series(signals, index=index),
        }
    )


def _signal_backtest(
    close: pd.Series,
    signal: pd.Series,
    transaction_cost_bps: float,
    bars_per_year: int,
) -> dict[str, Any]:
    clean_close = close.dropna()
    if len(clean_close) < 2:
        return {"state": "InsufficientData", "rows": int(len(clean_close))}
    log_returns = np.log(clean_close).diff().fillna(0.0)
    executable_signal = signal.reindex(clean_close.index).fillna(0.0).shift(1).fillna(0.0)
    signal_change = executable_signal.diff().abs().fillna(executable_signal.abs())
    cost = float(transaction_cost_bps) / 10_000.0
    strategy_log_returns = executable_signal * log_returns - signal_change * cost
    equity = np.exp(strategy_log_returns.cumsum())
    simple_returns = np.expm1(strategy_log_returns.to_numpy(dtype=float))
    metrics = _curve_metrics(equity, returns=simple_returns, bars_per_year=bars_per_year)
    metrics.update(
        {
            "state": "Measured",
            "rows": int(len(clean_close)),
            "trades": int((signal_change > 0).sum()),
            "turnover": float(signal_change.mean()),
            "long_exposure": float((executable_signal > 0).mean()),
            "short_exposure": float((executable_signal < 0).mean()),
            "cash_exposure": float((executable_signal == 0).mean()),
            "days_out_of_market": int((executable_signal == 0).sum()),
            "transaction_cost_bps": float(transaction_cost_bps),
        }
    )
    return metrics


def _curve_metrics(equity: pd.Series, returns: np.ndarray, bars_per_year: int) -> dict[str, Any]:
    final_equity = float(equity.iloc[-1]) if len(equity) else 1.0
    cumulative_return = final_equity - 1
    annualized_return = final_equity ** (bars_per_year / max(len(equity), 1)) - 1 if final_equity > 0 else -1.0
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(np.mean(returns) / std * np.sqrt(bars_per_year)) if std > 1e-12 else 0.0
    running_high = equity.cummax()
    drawdown = equity / running_high - 1
    return {
        "cumulative_return": _finite_or_none(cumulative_return),
        "annualized_return": _finite_or_none(annualized_return),
        "sharpe_ratio": _finite_or_none(sharpe),
        "max_drawdown": _finite_or_none(float(drawdown.min()) if len(drawdown) else 0.0),
    }


def _confirmed_wave_points(high: pd.Series, low: pd.Series, left: int, right: int) -> tuple[pd.Series, pd.Series]:
    window = left + right + 1
    raw_high = high == high.rolling(window=window, center=True).max()
    raw_low = low == low.rolling(window=window, center=True).min()
    confirmed_high_mask = raw_high.shift(right, fill_value=False).astype(bool)
    confirmed_low_mask = raw_low.shift(right, fill_value=False).astype(bool)
    confirmed_high = high.shift(right).where(confirmed_high_mask, np.nan)
    confirmed_low = low.shift(right).where(confirmed_low_mask, np.nan)
    return confirmed_high, confirmed_low


def _wave_point_summary(history: pd.DataFrame) -> dict[str, Any]:
    wave_high = history["magee_wave_high"].dropna()
    wave_low = history["magee_wave_low"].dropna()
    return {
        "confirmed_high_count": int(len(wave_high)),
        "confirmed_low_count": int(len(wave_low)),
        "last_confirmed_high": _point_payload(wave_high, "high"),
        "last_confirmed_low": _point_payload(wave_low, "low"),
    }


def _point_payload(series: pd.Series, kind: str) -> dict[str, Any] | None:
    if series.empty:
        return None
    return {
        "date": str(pd.Timestamp(series.index[-1]).date()),
        "type": kind,
        "price": float(series.iloc[-1]),
    }


def _history_tail(close: pd.Series, stop: pd.Series, signal: pd.Series, rows: int) -> list[dict[str, Any]]:
    tail_index = close.tail(rows).index
    output = []
    for date in tail_index:
        output.append(
            {
                "date": str(pd.Timestamp(date).date()),
                "close": _finite_or_none(close.loc[date]),
                "stop": _finite_or_none(stop.loc[date]) if date in stop.index else None,
                "signal": _signal_label(float(signal.loc[date])) if date in signal.index else "Neutral",
            }
        )
    return output


def _preferred_payload(timeframe_payload: dict[str, Any]) -> dict[str, Any]:
    if timeframe_payload.get("state") != "Measured":
        return {"state": timeframe_payload.get("state", "InsufficientData")}
    return {
        "timeframe": timeframe_payload["timeframe"],
        "variant": timeframe_payload["preferred_variant"],
        **timeframe_payload["latest"],
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


def _last_finite(series: pd.Series) -> float | None:
    values = series.replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.iloc[-1]) if len(values) else None


def _signal_label(signal: float) -> str:
    if signal > 0:
        return "Long"
    if signal < 0:
        return "Short"
    return "Neutral"


def _finite_or_none(value: Any) -> float | None:
    try:
        output = float(value)
    except Exception:
        return None
    return output if np.isfinite(output) else None
