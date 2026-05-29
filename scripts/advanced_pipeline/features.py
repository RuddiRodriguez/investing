"""Feature and target construction for the standalone advanced pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig


def validate_prices(prices: pd.DataFrame, min_history_days: int = 0) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("No price data was provided.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("Price data must use a DatetimeIndex.")

    clean = prices.sort_index().copy()
    clean.columns = [str(column).upper() for column in clean.columns]
    clean = clean.apply(pd.to_numeric, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).ffill().dropna(how="all")
    clean = clean.loc[:, clean.notna().sum() > 0]
    if len(clean) < min_history_days:
        raise ValueError(f"Need at least {min_history_days} trading days. Found {len(clean)}.")
    if (clean <= 0).any().any():
        raise ValueError("Prices must be positive.")
    return clean


def equal_weight_benchmark(prices: pd.DataFrame) -> pd.Series:
    returns = prices.pct_change().fillna(0.0)
    benchmark = (1.0 + returns.mean(axis=1)).cumprod()
    benchmark.name = "EQUAL_WEIGHT"
    return benchmark


def build_feature_panel(
    prices: pd.DataFrame,
    config: PipelineConfig,
    benchmark: pd.Series | None = None,
    fundamentals: pd.DataFrame | None = None,
    news: pd.DataFrame | None = None,
) -> pd.DataFrame:
    clean = validate_prices(prices)
    benchmark = _align_benchmark(clean, benchmark)
    returns = clean.pct_change()
    frames = []

    for ticker in clean.columns:
        price = clean[ticker]
        asset_return = returns[ticker]
        frame = pd.DataFrame(index=clean.index)
        frame["ticker"] = ticker
        frame["price"] = price
        for window in (1, 5, 10, 20, 60, 120):
            frame[f"return_{window}d"] = price.pct_change(window)
            frame[f"relative_return_{window}d"] = frame[f"return_{window}d"] - benchmark.pct_change(window)
        for window in (20, 60, 120):
            vol = asset_return.rolling(window).std() * np.sqrt(252)
            frame[f"volatility_{window}d"] = vol
            frame[f"trend_strength_{window}d"] = frame[f"return_{window}d"] / (vol * np.sqrt(window / 252))
        for window in (50, 100, 200):
            moving_average = price.rolling(window).mean()
            frame[f"ma_distance_{window}d"] = price / moving_average - 1.0
        for window in (60, 252):
            high = price.rolling(window).max()
            low = price.rolling(window).min()
            frame[f"drawdown_{window}d"] = price / high - 1.0
            frame[f"range_position_{window}d"] = (price - low) / (high - low)
        downside = asset_return.clip(upper=0.0)
        frame["downside_volatility_60d"] = downside.rolling(60).std() * np.sqrt(252)
        frame["var_95_252d"] = asset_return.rolling(252).quantile(0.05)
        frame["expected_shortfall_95_252d"] = asset_return.rolling(252).apply(_expected_shortfall, raw=False)
        frames.append(frame)

    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.set_index("ticker", append=True).sort_index()
    panel.index = panel.index.set_names(["date", "ticker"])
    panel = _attach_static_features(panel, fundamentals, "fundamental")
    panel = _attach_static_features(panel, news, "news")
    panel["technical_signal"] = technical_signal(panel)
    panel["fundamental_signal"] = fundamental_signal(panel)
    panel["news_signal"] = news_signal(panel)
    panel["graph_signal"] = graph_signal(panel, clean)
    return panel.replace([np.inf, -np.inf], np.nan)


def add_forward_targets(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    config: PipelineConfig,
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    clean = validate_prices(prices)
    benchmark = _align_benchmark(clean, benchmark)
    output = features.copy()
    for horizon in config.horizons:
        asset_forward = clean.shift(-horizon) / clean - 1.0
        benchmark_forward = benchmark.shift(-horizon) / benchmark - 1.0
        excess = asset_forward.sub(benchmark_forward, axis=0)
        long = excess.stack(future_stack=True).rename(f"target_excess_return_{horizon}d")
        long.index = long.index.set_names(["date", "ticker"])
        output = output.join(long)
        output[f"target_outperform_{horizon}d"] = (output[f"target_excess_return_{horizon}d"] > 0).astype(float)
    return output


def technical_signal(panel: pd.DataFrame) -> pd.Series:
    vol = panel["volatility_60d"]
    vol_fill = vol.dropna().median() if not vol.dropna().empty else 0.20
    raw = (
        panel["relative_return_20d"].fillna(0.0) * 2.0
        + panel["relative_return_60d"].fillna(0.0)
        + panel["ma_distance_100d"].fillna(0.0)
        - vol.fillna(vol_fill) * 0.10
    )
    return np.tanh(cross_sectional_zscore(raw))


def fundamental_signal(panel: pd.DataFrame) -> pd.Series:
    positive = [
        "fundamental_revenue_growth",
        "fundamental_earnings_growth",
        "fundamental_gross_margin",
        "fundamental_operating_margin",
        "fundamental_profit_margin",
        "fundamental_return_on_equity",
        "fundamental_free_cashflow_yield",
    ]
    negative = ["fundamental_debt_to_equity", "fundamental_forward_pe", "fundamental_trailing_pe"]
    return _composite_signal(panel, positive, negative)


def news_signal(panel: pd.DataFrame) -> pd.Series:
    positive = ["news_news_sentiment", "news_positive_news_intensity"]
    negative = ["news_negative_news_intensity"]
    return _composite_signal(panel, positive, negative)


def graph_signal(panel: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    momentum = panel["return_20d"].unstack("ticker")
    correlation = prices.pct_change().tail(252).corr().clip(lower=0.0)
    if correlation.empty:
        return pd.Series(0.0, index=panel.index)
    np.fill_diagonal(correlation.values, 0.0)
    weighted = momentum.dot(correlation.T)
    weights = correlation.sum(axis=1).replace(0.0, np.nan)
    graph = weighted.div(weights, axis=1)
    long = graph.stack(future_stack=True).rename("graph_signal_raw")
    long.index = long.index.set_names(["date", "ticker"])
    return np.tanh(cross_sectional_zscore(long.reindex(panel.index).fillna(0.0)))


def cross_sectional_zscore(values: pd.Series) -> pd.Series:
    def scale(group: pd.Series) -> pd.Series:
        std = group.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=group.index)
        return (group - group.mean()) / std

    return values.groupby(level="date", group_keys=False).apply(scale).clip(-3.0, 3.0)


def _align_benchmark(prices: pd.DataFrame, benchmark: pd.Series | None) -> pd.Series:
    if benchmark is None:
        return equal_weight_benchmark(prices)
    aligned = benchmark.sort_index().reindex(prices.index).ffill()
    if aligned.isna().all():
        raise ValueError("Benchmark has no overlap with asset prices.")
    return aligned


def _attach_static_features(panel: pd.DataFrame, frame: pd.DataFrame | None, prefix: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return panel
    data = frame.copy()
    if "ticker" in data.columns:
        data["ticker"] = data["ticker"].astype(str).str.upper()
        data = data.set_index("ticker")
    data.index = data.index.astype(str).str.upper()
    data = data.add_prefix(f"{prefix}_")
    merged = panel.reset_index().merge(data, left_on="ticker", right_index=True, how="left")
    return merged.set_index(["date", "ticker"]).sort_index()


def _composite_signal(panel: pd.DataFrame, positive: list[str], negative: list[str]) -> pd.Series:
    parts = []
    for column in positive:
        if column in panel:
            values = pd.to_numeric(panel[column], errors="coerce")
            parts.append(cross_sectional_zscore(values.fillna(0.0)))
    for column in negative:
        if column in panel:
            values = pd.to_numeric(panel[column], errors="coerce")
            parts.append(-cross_sectional_zscore(values.fillna(0.0)))
    if not parts:
        return pd.Series(0.0, index=panel.index)
    return np.tanh(pd.concat(parts, axis=1).mean(axis=1).fillna(0.0))


def _expected_shortfall(values: pd.Series) -> float:
    valid = values.dropna()
    if valid.empty:
        return np.nan
    var = valid.quantile(0.05)
    return float(valid[valid <= var].mean())
