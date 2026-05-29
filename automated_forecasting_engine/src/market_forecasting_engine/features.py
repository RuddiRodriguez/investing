from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from market_forecasting_engine.technical_structure import build_technical_structure_features


BASE_PRICE_COLUMNS = {"open", "high", "low", "close", "volume", "dividends", "stock_splits"}


def _clean_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0.0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return _safe_divide(series - rolling_mean, rolling_std)


def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(lambda values: pd.Series(values).rank(pct=True).iloc[-1], raw=False)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    relative_strength = _safe_divide(gain, loss)
    return 100 - (100 / (1 + relative_strength))


def build_feature_frame(prices: pd.DataFrame, target_column: str = "close") -> pd.DataFrame:
    """Build time-series features available as of each row's timestamp."""

    target = target_column.lower()
    close = prices[target].astype(float)
    log_close = np.log(close.replace(0, np.nan))
    log_return = log_close.diff()

    features = pd.DataFrame(index=prices.index)
    features["log_return_1d"] = log_return

    for window in (2, 3, 5, 10, 21, 63):
        features[f"log_return_{window}d"] = log_close.diff(window)
        features[f"momentum_{window}d"] = close.pct_change(window)

    for window in (5, 10, 20, 50, 100, 200):
        moving_average = close.rolling(window).mean()
        features[f"close_to_sma_{window}"] = _safe_divide(close, moving_average) - 1

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    features["close_to_ema_12"] = _safe_divide(close, ema_12) - 1
    features["close_to_ema_26"] = _safe_divide(close, ema_26) - 1
    features["macd"] = _safe_divide(macd, close)
    features["macd_signal"] = _safe_divide(macd_signal, close)
    features["macd_hist"] = _safe_divide(macd - macd_signal, close)

    for window in (5, 20, 63):
        features[f"volatility_{window}d"] = log_return.rolling(window).std() * np.sqrt(252)
    features["volatility_252d"] = log_return.rolling(252).std() * np.sqrt(252)

    high = prices["high"].astype(float) if "high" in prices.columns else close
    low = prices["low"].astype(float) if "low" in prices.columns else close
    open_price = prices["open"].astype(float) if "open" in prices.columns else close.shift(1)
    true_range = _true_range(high=high, low=low, close=close)
    features["true_range_pct"] = _safe_divide(true_range, close)
    features["atr_14_pct"] = _safe_divide(true_range.rolling(14).mean(), close)
    features["atr_20_pct"] = _safe_divide(true_range.rolling(20).mean(), close)
    features["parkinson_volatility_20d"] = _parkinson_volatility(high=high, low=low, window=20)
    features["garman_klass_volatility_20d"] = _garman_klass_volatility(
        open_price=open_price,
        high=high,
        low=low,
        close=close,
        window=20,
    )
    features["realized_skew_20d"] = log_return.rolling(20).skew()
    features["realized_kurtosis_20d"] = log_return.rolling(20).kurt()
    features["drawdown_63d"] = _safe_divide(close, close.rolling(63).max()) - 1
    features["drawdown_252d"] = _safe_divide(close, close.rolling(252).max()) - 1
    features["distance_to_52w_high"] = _safe_divide(close, close.rolling(252).max()) - 1
    features["distance_to_52w_low"] = _safe_divide(close, close.rolling(252).min()) - 1
    features["volatility_regime_20_252"] = _safe_divide(features["volatility_20d"], features["volatility_252d"]) - 1

    bollinger_mid = close.rolling(20).mean()
    bollinger_std = close.rolling(20).std()
    features["bollinger_z_20"] = _safe_divide(close - bollinger_mid, bollinger_std)
    features["rsi_14"] = _rsi(close, 14)

    for lag in (1, 2, 3, 5, 10):
        features[f"lagged_log_return_{lag}d"] = log_return.shift(lag)

    if "volume" in prices.columns:
        volume = prices["volume"].astype(float)
        features["volume_change_1d"] = volume.pct_change()
        features["volume_to_sma_20"] = _safe_divide(volume, volume.rolling(20).mean()) - 1
        features["volume_z_20"] = _rolling_zscore(volume, 20)
        dollar_volume = close * volume
        features["dollar_volume_log"] = np.log1p(dollar_volume.clip(lower=0))
        features["dollar_volume_to_sma_20"] = _safe_divide(dollar_volume, dollar_volume.rolling(20).mean()) - 1
        features["dollar_volume_z_20"] = _rolling_zscore(dollar_volume, 20)
        features["average_dollar_volume_20d_log"] = np.log1p(dollar_volume.rolling(20).mean().clip(lower=0))
        features["average_dollar_volume_63d_log"] = np.log1p(dollar_volume.rolling(63).mean().clip(lower=0))
        features["volume_volatility_20d"] = volume.pct_change().rolling(20).std()
        features["amihud_illiquidity_20d"] = _safe_divide(log_return.abs(), dollar_volume).rolling(20).mean()
        features["on_balance_volume_z_63"] = _rolling_zscore(_on_balance_volume(close, volume), 63)
        features["accumulation_distribution_z_63"] = _rolling_zscore(
            _accumulation_distribution_line(high=high, low=low, close=close, volume=volume),
            63,
        )
        features["money_flow_index_14"] = _money_flow_index(high=high, low=low, close=close, volume=volume, window=14)
        volume_breakout = (volume > volume.rolling(20).mean() * 1.5).astype(float)
        features["volume_breakout_persistence_5d"] = volume_breakout.rolling(5).mean()

    calendar_index = pd.DatetimeIndex(prices.index)
    features["day_of_week"] = calendar_index.dayofweek.astype(float)
    features["month"] = calendar_index.month.astype(float)
    features["is_month_end"] = calendar_index.is_month_end.astype(float)
    features["is_quarter_end"] = calendar_index.is_quarter_end.astype(float)

    relative_features: dict[str, pd.Series] = {}
    for column in _relative_strength_columns(prices.columns, target_column=target):
        series = prices[column].astype(float)
        clean = _clean_name(column)
        context_log = np.log(series.replace(0, np.nan))
        context_return = context_log.diff()
        ratio = np.log(_safe_divide(close, series).replace(0, np.nan))
        relative_features[f"relative_{clean}_price_ratio_log"] = ratio
        relative_features[f"relative_{clean}_price_ratio_z_63"] = _rolling_zscore(ratio, 63)
        relative_features[f"relative_{clean}_return_1d"] = log_return - context_return
        for window in (5, 21, 63):
            relative_features[f"relative_{clean}_return_{window}d"] = log_close.diff(window) - context_log.diff(window)
        beta = _rolling_beta(log_return, context_return, window=63)
        relative_features[f"relative_{clean}_beta_63d"] = beta
        relative_features[f"relative_{clean}_corr_63d"] = log_return.rolling(63).corr(context_return)
        residual_return = log_return - beta * context_return
        relative_features[f"relative_{clean}_residual_return_1d"] = residual_return
        relative_features[f"relative_{clean}_residual_return_z_63"] = _rolling_zscore(residual_return, 63)
        relative_features[f"relative_{clean}_strength_percentile_252d"] = _rolling_percentile_rank(ratio, 252)

    if relative_features:
        features = pd.concat([features, pd.DataFrame(relative_features, index=prices.index)], axis=1)

    external_features: dict[str, pd.Series] = {}
    for column in _external_feature_columns(prices.columns, target_column=target):
        series = prices[column].astype(float)
        clean = _clean_name(column)
        external_features[f"exo_{clean}"] = series
        external_features[f"exo_{clean}_change_1d"] = series.pct_change()
        external_features[f"exo_{clean}_z_20"] = _rolling_zscore(series, 20)

    if external_features:
        features = pd.concat([features, pd.DataFrame(external_features, index=prices.index)], axis=1)

    structure_features = build_technical_structure_features(prices, target_column=target)
    features = pd.concat([features, structure_features], axis=1)
    return features.replace([np.inf, -np.inf], np.nan)


def add_forward_return_targets(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: Iterable[int],
    target_column: str = "close",
) -> pd.DataFrame:
    """Append forward log-return targets for each forecast horizon."""

    close = prices[target_column.lower()].astype(float)
    high = prices["high"].astype(float) if "high" in prices.columns else close
    low = prices["low"].astype(float) if "low" in prices.columns else close
    log_close = np.log(close.replace(0, np.nan))
    supervised = features.copy()
    for horizon in horizons:
        future_log_return = log_close.shift(-horizon) - log_close
        prior_support = low.shift(1).rolling(63).min()
        prior_resistance = high.shift(1).rolling(63).max()
        future_high = _future_rolling_max(high, horizon)
        future_low = _future_rolling_min(low, horizon)
        upside_room = _safe_divide(prior_resistance - close, close).clip(lower=0)
        downside_room = _safe_divide(close - prior_support, close).clip(lower=0)

        supervised[f"target_log_return_{horizon}d"] = future_log_return
        supervised[f"target_price_{horizon}d"] = close.shift(-horizon)
        supervised[f"target_direction_{horizon}d"] = np.where(future_log_return.notna(), (future_log_return > 0).astype(float), np.nan)
        supervised[f"target_upside_breakout_{horizon}d"] = np.where(
            future_high.notna() & prior_resistance.notna(),
            (future_high > prior_resistance * 1.0025).astype(float),
            np.nan,
        )
        supervised[f"target_downside_breakdown_{horizon}d"] = np.where(
            future_low.notna() & prior_support.notna(),
            (future_low < prior_support * 0.9975).astype(float),
            np.nan,
        )
        supervised[f"target_reward_to_risk_{horizon}d"] = _safe_divide(future_log_return, downside_room.clip(lower=0.005))
        supervised[f"target_upside_room_{horizon}d"] = upside_room
        supervised[f"target_downside_room_{horizon}d"] = downside_room
    return supervised


def _future_rolling_max(series: pd.Series, horizon: int) -> pd.Series:
    return series.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).max().iloc[::-1]


def _future_rolling_min(series: pd.Series, horizon: int) -> pd.Series:
    return series.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1]


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


def _parkinson_volatility(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    log_range = np.log(_safe_divide(high, low).replace(0, np.nan))
    variance = (log_range**2).rolling(window).mean() / (4 * np.log(2))
    return np.sqrt(variance.clip(lower=0) * 252)


def _garman_klass_volatility(
    open_price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    log_hl = np.log(_safe_divide(high, low).replace(0, np.nan))
    log_co = np.log(_safe_divide(close, open_price).replace(0, np.nan))
    variance = (0.5 * log_hl**2) - ((2 * np.log(2) - 1) * log_co**2)
    return np.sqrt(variance.clip(lower=0).rolling(window).mean() * 252)


def _on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume.fillna(0.0)).cumsum()


def _accumulation_distribution_line(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    close_location = _safe_divide((close - low) - (high - close), high - low).fillna(0.0)
    return (close_location * volume.fillna(0.0)).cumsum()


def _money_flow_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int,
) -> pd.Series:
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    direction = typical_price.diff()
    positive_flow = raw_money_flow.where(direction > 0, 0.0).rolling(window).sum()
    negative_flow = raw_money_flow.where(direction < 0, 0.0).rolling(window).sum().abs()
    money_ratio = _safe_divide(positive_flow, negative_flow)
    return 100 - (100 / (1 + money_ratio))


def _rolling_beta(target_return: pd.Series, context_return: pd.Series, window: int) -> pd.Series:
    covariance = target_return.rolling(window).cov(context_return)
    variance = context_return.rolling(window).var()
    return _safe_divide(covariance, variance)


def _external_feature_columns(columns: Iterable[str], target_column: str) -> list[str]:
    excluded = set(BASE_PRICE_COLUMNS)
    excluded.add(target_column.lower())
    result = []
    for column in columns:
        clean = _clean_name(str(column))
        if clean not in excluded:
            result.append(column)
    return result


def _relative_strength_columns(columns: Iterable[str], target_column: str) -> list[str]:
    result = []
    target = target_column.lower()
    for column in columns:
        clean = _clean_name(str(column))
        if clean == target:
            continue
        if clean.startswith(("benchmark_", "sector_", "market_", "index_")):
            result.append(column)
    return result
