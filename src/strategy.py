"""Core strategy logic for a simple medium-term ETF rotation model.

The model is intentionally small and explainable:
- use trailing total-return momentum over a configurable lookback window
- require the ETF to be above its long-term moving average
- rank eligible ETFs and hold the top N
- fall back to cash when no ETF passes the trend filter

This is educational software. It is not financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters controlling the ETF allocation model."""

    lookback_months: int = 6
    moving_average_days: int = 200
    top_n: int = 3
    rebalance_frequency: str = "Q"
    cash_ticker: str = "CASH"
    min_history_days: int = 260


def validate_prices(prices: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Validate and clean an adjusted-close price matrix.

    The input is expected to have:
    - DatetimeIndex
    - one column per ticker
    - positive numeric prices
    """

    if prices.empty:
        raise ValueError("No price data was provided.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("Price data must use a DatetimeIndex.")

    clean = prices.sort_index().copy()
    clean = clean.apply(pd.to_numeric, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).ffill().dropna(how="all")

    if len(clean) < config.min_history_days:
        raise ValueError(
            f"Need at least {config.min_history_days} trading days of history "
            f"to run the model reliably. Only found {len(clean)} rows."
        )

    if (clean <= 0).any().any():
        raise ValueError("Prices must be strictly positive.")

    return clean


def compute_indicators(prices: pd.DataFrame, config: StrategyConfig) -> dict[str, pd.DataFrame]:
    """Compute the indicators used by the model."""

    trading_days_per_month = 21
    lookback_days = config.lookback_months * trading_days_per_month

    momentum = prices.pct_change(lookback_days)
    moving_average = prices.rolling(config.moving_average_days).mean()
    above_trend = prices > moving_average

    return {
        "momentum": momentum,
        "moving_average": moving_average,
        "above_trend": above_trend,
    }


def _rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """Return the last available trading date in each rebalance period."""

    periods = index.to_period(frequency)
    return pd.DatetimeIndex(index.to_series().groupby(periods).max().tolist())


def build_weights(prices: pd.DataFrame, config: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a rebalance schedule and daily target weights."""

    clean = validate_prices(prices, config)
    indicators = compute_indicators(clean, config)
    rebalance_dates = _rebalance_dates(clean.index, config.rebalance_frequency)

    holdings = []
    for rebalance_date in rebalance_dates:
        momentum_today = indicators["momentum"].loc[rebalance_date].dropna()
        above_trend_today = indicators["above_trend"].loc[rebalance_date]

        eligible = momentum_today[above_trend_today.loc[momentum_today.index].fillna(False)]
        ranked = eligible.sort_values(ascending=False).head(config.top_n)

        row = pd.Series(0.0, index=list(clean.columns) + [config.cash_ticker], name=rebalance_date)
        if ranked.empty:
            row[config.cash_ticker] = 1.0
        else:
            weight = 1.0 / len(ranked)
            row.loc[ranked.index] = weight
        holdings.append(row)

    rebalance_weights = pd.DataFrame(holdings).fillna(0.0)
    daily_weights = rebalance_weights.reindex(clean.index, method="ffill").fillna(0.0)

    return rebalance_weights, daily_weights


def backtest(prices: pd.DataFrame, config: StrategyConfig) -> dict[str, pd.DataFrame | pd.Series]:
    """Run a simple end-of-period rebalanced backtest."""

    clean = validate_prices(prices, config)
    rebalance_weights, daily_weights = build_weights(clean, config)

    asset_returns = clean.pct_change().fillna(0.0)
    lagged_weights = daily_weights.shift(1).fillna(0.0)
    portfolio_returns = (lagged_weights[clean.columns] * asset_returns).sum(axis=1)
    equity_curve = (1.0 + portfolio_returns).cumprod()

    drawdown = equity_curve / equity_curve.cummax() - 1.0
    metrics = pd.Series(
        {
            "Total Return": equity_curve.iloc[-1] - 1.0,
            "Annualized Return": annualized_return(equity_curve),
            "Annualized Volatility": portfolio_returns.std() * np.sqrt(252),
            "Sharpe (rf=0)": sharpe_ratio(portfolio_returns),
            "Max Drawdown": drawdown.min(),
            "Win Rate": (portfolio_returns > 0).mean(),
        }
    )

    latest_date = rebalance_weights.index[-1]
    latest_allocation = rebalance_weights.loc[latest_date]
    latest_allocation = latest_allocation[latest_allocation > 0].sort_values(ascending=False)
    indicators = compute_indicators(clean, config)

    latest_signal_frame = pd.DataFrame(
        {
            "price": clean.loc[latest_date],
            "momentum": indicators["momentum"].loc[latest_date],
            "moving_average": indicators["moving_average"].loc[latest_date],
            "above_trend": indicators["above_trend"].loc[latest_date],
        }
    ).sort_values("momentum", ascending=False)

    return {
        "clean_prices": clean,
        "rebalance_weights": rebalance_weights,
        "daily_weights": daily_weights,
        "portfolio_returns": portfolio_returns,
        "equity_curve": equity_curve,
        "drawdown": drawdown,
        "metrics": metrics,
        "latest_allocation": latest_allocation,
        "latest_signal_frame": latest_signal_frame,
    }


def annualized_return(equity_curve: pd.Series) -> float:
    """Compute annualized return from an equity curve."""

    total_periods = len(equity_curve)
    if total_periods < 2:
        return 0.0

    years = total_periods / 252
    if years <= 0:
        return 0.0
    return float(equity_curve.iloc[-1] ** (1 / years) - 1)


def sharpe_ratio(returns: pd.Series) -> float:
    """Compute a simple annualized Sharpe ratio with zero risk-free rate."""

    volatility = returns.std()
    if volatility == 0 or np.isnan(volatility):
        return 0.0
    return float((returns.mean() / volatility) * np.sqrt(252))
