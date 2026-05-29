from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def backtest_validation_signals(
    validation_records: list[dict[str, Any]],
    horizon_days: int,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 0.0,
) -> dict[str, Any]:
    """Backtest validation-period forecast signs as long/short/flat signals."""

    frame = pd.DataFrame(validation_records)
    if frame.empty:
        return _empty_backtest()

    frame["validation_date"] = pd.to_datetime(frame["validation_date"])
    frame = frame.sort_values("validation_date")
    frame["signal"] = np.sign(frame["predicted_log_return"]).astype(float)
    frame.loc[frame["predicted_log_return"].abs() < 1e-8, "signal"] = 0.0
    frame["gross_log_return"] = frame["signal"] * frame["actual_log_return"]
    frame["signal_change"] = frame["signal"].diff().abs().fillna(frame["signal"].abs())
    cost = (transaction_cost_bps + slippage_bps) / 10_000.0
    frame["net_log_return"] = frame["gross_log_return"] - frame["signal_change"] * cost
    frame["benchmark_log_return"] = frame["actual_log_return"]
    frame["equity"] = np.exp(frame["net_log_return"].cumsum())
    frame["benchmark_equity"] = np.exp(frame["benchmark_log_return"].cumsum())

    returns = np.expm1(frame["net_log_return"].to_numpy(dtype=float))
    benchmark_returns = np.expm1(frame["benchmark_log_return"].to_numpy(dtype=float))
    periods_per_year = 252 / max(horizon_days, 1)

    summary = {
        "rows": int(len(frame)),
        "start_date": str(frame["validation_date"].iloc[0].date()),
        "end_date": str(frame["validation_date"].iloc[-1].date()),
        "transaction_cost_bps": float(transaction_cost_bps),
        "slippage_bps": float(slippage_bps),
        "execution_timing": "vectorized_same_period_close_to_close",
        "trades": int((frame["signal_change"] > 0).sum()),
        "turnover": float(frame["signal_change"].mean()),
        "cumulative_return": float(frame["equity"].iloc[-1] - 1),
        "benchmark_cumulative_return": float(frame["benchmark_equity"].iloc[-1] - 1),
        "annualized_return": _annualized_return(frame["equity"].iloc[-1], len(frame), periods_per_year),
        "sharpe_ratio": _sharpe(returns, periods_per_year),
        "max_drawdown": _max_drawdown(frame["equity"]),
        "hit_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "profit_factor": _profit_factor(returns),
        "benchmark_sharpe_ratio": _sharpe(benchmark_returns, periods_per_year),
        "benchmark_max_drawdown": _max_drawdown(frame["benchmark_equity"]),
        "trade_ledger_summary": _trade_ledger_summary(frame),
        "slippage_sensitivity": _slippage_sensitivity(
            frame=frame,
            horizon_days=horizon_days,
            transaction_cost_bps=transaction_cost_bps,
        ),
    }
    return {key: _json_float(value) for key, value in summary.items()}


def _slippage_sensitivity(
    *,
    frame: pd.DataFrame,
    horizon_days: int,
    transaction_cost_bps: float,
) -> list[dict[str, Any]]:
    rows = []
    periods_per_year = 252 / max(horizon_days, 1)
    for slippage_bps in [0.0, 2.5, 5.0, 10.0, 25.0]:
        cost = (transaction_cost_bps + slippage_bps) / 10_000.0
        net_log_return = frame["gross_log_return"] - frame["signal_change"] * cost
        equity = np.exp(net_log_return.cumsum())
        returns = np.expm1(net_log_return.to_numpy(dtype=float))
        rows.append(
            {
                "slippage_bps": float(slippage_bps),
                "total_cost_bps": float(transaction_cost_bps + slippage_bps),
                "cumulative_return": _json_float(float(equity.iloc[-1] - 1.0)),
                "sharpe_ratio": _json_float(_sharpe(returns, periods_per_year)),
                "max_drawdown": _json_float(_max_drawdown(equity)),
            }
        )
    return rows


def _trade_ledger_summary(frame: pd.DataFrame) -> dict[str, Any]:
    changes = frame[frame["signal_change"] > 0]
    if changes.empty:
        return {
            "trade_count": 0,
            "first_trade_date": None,
            "last_trade_date": None,
            "long_entries": 0,
            "short_entries": 0,
            "flat_entries": 0,
        }
    return {
        "trade_count": int(len(changes)),
        "first_trade_date": str(changes["validation_date"].iloc[0].date()),
        "last_trade_date": str(changes["validation_date"].iloc[-1].date()),
        "long_entries": int((changes["signal"] > 0).sum()),
        "short_entries": int((changes["signal"] < 0).sum()),
        "flat_entries": int((changes["signal"] == 0).sum()),
    }


def _annualized_return(final_equity: float, periods: int, periods_per_year: float) -> float:
    if periods <= 0 or final_equity <= 0:
        return 0.0
    return float(final_equity ** (periods_per_year / periods) - 1)


def _sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(periods_per_year))


def _max_drawdown(equity: pd.Series) -> float:
    running_high = equity.cummax()
    drawdown = equity / running_high - 1
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _profit_factor(returns: np.ndarray) -> float:
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    if losses < 0:
        return float(gains / abs(losses))
    return float("inf") if gains > 0 else 0.0


def _json_float(value: Any) -> Any:
    if isinstance(value, (str, int)):
        return value
    if value is None:
        return None
    if isinstance(value, list):
        return [_json_float(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_float(item) for key, item in value.items()}
    if np.isfinite(value):
        return float(value)
    return 999.0 if value > 0 else -999.0


def _empty_backtest() -> dict[str, Any]:
    return {
        "rows": 0,
        "transaction_cost_bps": 0.0,
        "slippage_bps": 0.0,
        "execution_timing": "vectorized_same_period_close_to_close",
        "trades": 0,
        "turnover": 0.0,
        "cumulative_return": 0.0,
        "benchmark_cumulative_return": 0.0,
        "annualized_return": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "hit_rate": 0.0,
        "profit_factor": 0.0,
        "trade_ledger_summary": {
            "trade_count": 0,
            "first_trade_date": None,
            "last_trade_date": None,
            "long_entries": 0,
            "short_entries": 0,
            "flat_entries": 0,
        },
        "slippage_sensitivity": [],
    }
