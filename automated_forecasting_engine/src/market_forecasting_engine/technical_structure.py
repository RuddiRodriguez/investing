from __future__ import annotations

import numpy as np
import pandas as pd


def build_technical_structure_features(prices: pd.DataFrame, target_column: str = "close") -> pd.DataFrame:
    """Build chart-structure features without using future data."""

    close = prices[target_column].astype(float)
    high = prices["high"].astype(float) if "high" in prices.columns else close
    low = prices["low"].astype(float) if "low" in prices.columns else close
    open_price = prices["open"].astype(float) if "open" in prices.columns else close.shift(1)
    volume = prices["volume"].astype(float) if "volume" in prices.columns else pd.Series(np.nan, index=prices.index)

    features = pd.DataFrame(index=prices.index)
    prior_close = close.shift(1)
    prior_high = high.shift(1)
    prior_low = low.shift(1)

    volume_sma_20 = volume.shift(1).rolling(20).mean()
    volume_sma_60 = volume.shift(1).rolling(60).mean()
    features["structure_volume_confirmation_20"] = _safe_divide(volume, volume_sma_20) - 1
    features["structure_volume_confirmation_60"] = _safe_divide(volume, volume_sma_60) - 1

    for window in (20, 63, 126):
        support = prior_low.rolling(window).min()
        resistance = prior_high.rolling(window).max()
        features[f"structure_support_{window}d"] = support
        features[f"structure_resistance_{window}d"] = resistance
        features[f"structure_close_to_support_{window}d"] = _safe_divide(close - support, close)
        features[f"structure_close_to_resistance_{window}d"] = _safe_divide(resistance - close, close)
        features[f"structure_range_position_{window}d"] = _safe_divide(close - support, resistance - support)
        features[f"structure_breakout_{window}d"] = (close > resistance * 1.0025).astype(float)
        features[f"structure_breakdown_{window}d"] = (close < support * 0.9975).astype(float)
        features[f"structure_breakout_volume_confirmed_{window}d"] = (
            (features[f"structure_breakout_{window}d"] > 0) & (volume > volume_sma_20 * 1.25)
        ).astype(float)
        features[f"structure_breakdown_volume_confirmed_{window}d"] = (
            (features[f"structure_breakdown_{window}d"] > 0) & (volume > volume_sma_20 * 1.25)
        ).astype(float)

    features["structure_gap_up"] = (open_price > prior_high * 1.003).astype(float)
    features["structure_gap_down"] = (open_price < prior_low * 0.997).astype(float)
    features["structure_gap_size"] = _safe_divide(open_price - prior_close, prior_close)
    true_gap_up = low > prior_high * 1.001
    true_gap_down = high < prior_low * 0.999
    true_gap_width = pd.Series(np.nan, index=prices.index)
    true_gap_width = true_gap_width.mask(true_gap_up, low - prior_high)
    true_gap_width = true_gap_width.mask(true_gap_down, prior_low - high)
    features["structure_true_gap_up"] = true_gap_up.astype(float)
    features["structure_true_gap_down"] = true_gap_down.astype(float)
    features["structure_true_gap_size_pct"] = _safe_divide(true_gap_width, prior_close)
    gap_up_size = open_price - prior_high
    gap_down_size = prior_low - open_price
    features["structure_gap_fill_pct"] = np.where(
        features["structure_gap_up"] > 0,
        _safe_divide(open_price - low, gap_up_size).clip(lower=0, upper=1),
        np.where(
            features["structure_gap_down"] > 0,
            _safe_divide(high - open_price, gap_down_size).clip(lower=0, upper=1),
            np.nan,
        ),
    )
    features["structure_gap_filled_same_day"] = (
        ((features["structure_gap_up"] > 0) & (low <= prior_high))
        | ((features["structure_gap_down"] > 0) & (high >= prior_low))
    ).astype(float)
    true_range_pct = _safe_divide(_true_range(high=high, low=low, close=close), close)
    features["structure_true_gap_atr_multiple"] = _safe_divide(true_gap_width, true_range_pct.shift(1).rolling(20).mean() * prior_close)
    corporate_action = pd.Series(False, index=prices.index)
    if "dividends" in prices.columns:
        corporate_action = corporate_action | (pd.to_numeric(prices["dividends"], errors="coerce").fillna(0.0) != 0.0)
    if "stock_splits" in prices.columns:
        corporate_action = corporate_action | (pd.to_numeric(prices["stock_splits"], errors="coerce").fillna(0.0) != 0.0)
    features["structure_gap_corporate_action"] = (
        ((features["structure_true_gap_up"] > 0) | (features["structure_true_gap_down"] > 0)) & corporate_action
    ).astype(float)
    prior_range_mean_20 = true_range_pct.shift(1).rolling(20).mean()
    close_location = _safe_divide(close - low, high - low).clip(lower=0, upper=1)
    features["structure_gap_breakaway"] = (
        ((features["structure_gap_up"] > 0) | (features["structure_gap_down"] > 0))
        & (features["structure_gap_size"].abs() > true_range_pct.shift(1).rolling(20).mean() * 1.5)
    ).astype(float)
    prior_high_20 = high.shift(1).rolling(20).max()
    prior_low_20 = low.shift(1).rolling(20).min()
    prior_return_20 = close.pct_change(20)
    wide_range = true_range_pct > prior_range_mean_20 * 1.6
    very_wide_range = true_range_pct > prior_range_mean_20 * 2.2
    volume_extreme = volume > volume_sma_20 * 1.8
    features["structure_key_reversal_top"] = (
        (high >= prior_high_20 * 1.0025) & (close < prior_close)
    ).astype(float)
    features["structure_key_reversal_bottom"] = (
        (low <= prior_low_20 * 0.9975) & (close > prior_close)
    ).astype(float)
    features["structure_one_day_reversal_top"] = (
        (prior_return_20 > 0.10)
        & (high >= prior_high_20 * 1.0025)
        & wide_range
        & (close_location <= 0.25)
        & (close < open_price)
    ).astype(float)
    features["structure_one_day_reversal_bottom"] = (
        (prior_return_20 < -0.10)
        & (low <= prior_low_20 * 0.9975)
        & wide_range
        & (close_location >= 0.75)
        & (close > open_price)
    ).astype(float)
    features["structure_selling_climax"] = (
        (prior_return_20 < -0.15)
        & (open_price < prior_low * 0.99)
        & very_wide_range
        & volume_extreme
        & (close_location >= 0.65)
    ).astype(float)
    features["structure_spike_top"] = (
        (high >= prior_high_20 * 1.0025) & very_wide_range & (close_location <= 0.25)
    ).astype(float)
    features["structure_spike_bottom"] = (
        (low <= prior_low_20 * 0.9975) & very_wide_range & (close_location >= 0.75)
    ).astype(float)
    features["structure_runaway_day_up"] = (
        (features["structure_gap_up"] > 0) & wide_range & (close_location >= 0.80)
    ).astype(float)
    features["structure_runaway_day_down"] = (
        (features["structure_gap_down"] > 0) & wide_range & (close_location <= 0.20)
    ).astype(float)

    for window in (20, 50, 100):
        slope, channel_position, channel_width = _rolling_trend_channel(np.log(close.replace(0, np.nan)), window)
        features[f"structure_trend_slope_{window}d"] = slope
        features[f"structure_channel_position_{window}d"] = channel_position
        features[f"structure_channel_width_{window}d"] = channel_width

    pivot_high, pivot_low = _confirmed_pivots(high=high, low=low, left=3, right=3)
    features["structure_pivot_high_confirmed"] = pivot_high.astype(float)
    features["structure_pivot_low_confirmed"] = pivot_low.astype(float)
    pivot_resistance = high.shift(3).where(pivot_high > 0).ffill()
    pivot_support = low.shift(3).where(pivot_low > 0).ffill()
    features["structure_pivot_resistance"] = pivot_resistance
    features["structure_pivot_support"] = pivot_support
    features["structure_close_to_pivot_resistance"] = _safe_divide(pivot_resistance - close, close)
    features["structure_close_to_pivot_support"] = _safe_divide(close - pivot_support, close)
    resistance_touch = (_safe_divide((high - pivot_resistance).abs(), close) <= 0.005).astype(float)
    support_touch = (_safe_divide((low - pivot_support).abs(), close) <= 0.005).astype(float)
    features["structure_resistance_touch_count_63d"] = resistance_touch.rolling(63).sum()
    features["structure_support_touch_count_63d"] = support_touch.rolling(63).sum()
    features["structure_pivot_resistance_age"] = _age_since_signal(pivot_high)
    features["structure_pivot_support_age"] = _age_since_signal(pivot_low)
    recent_breakout = features["structure_breakout_63d"].shift(1).rolling(10).max()
    recent_breakdown = features["structure_breakdown_63d"].shift(1).rolling(10).max()
    features["structure_breakout_retest_63d"] = (
        (recent_breakout > 0) & (_safe_divide((low - prior_high.rolling(63).max()).abs(), close) <= 0.01)
    ).astype(float)
    features["structure_breakdown_retest_63d"] = (
        (recent_breakdown > 0) & (_safe_divide((high - prior_low.rolling(63).min()).abs(), close) <= 0.01)
    ).astype(float)
    features["structure_failed_breakout_63d"] = (
        (recent_breakout > 0) & (close < prior_high.rolling(63).max())
    ).astype(float)
    features["structure_failed_breakdown_63d"] = (
        (recent_breakdown > 0) & (close > prior_low.rolling(63).min())
    ).astype(float)

    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    features["structure_trend_state_short"] = np.sign(sma_20 - sma_50)
    features["structure_trend_state_long"] = np.sign(sma_50 - sma_200)
    features["structure_close_above_sma_20"] = (close > sma_20).astype(float)
    features["structure_close_above_sma_50"] = (close > sma_50).astype(float)
    features["structure_close_above_sma_200"] = (close > sma_200).astype(float)
    features["structure_gap_continuation"] = (
        ((features["structure_gap_up"] > 0) & (features["structure_trend_state_short"] > 0))
        | ((features["structure_gap_down"] > 0) & (features["structure_trend_state_short"] < 0))
    ).astype(float)
    features["structure_rectangle_consolidation_20d"] = (
        _safe_divide(high.rolling(20).max() - low.rolling(20).min(), close)
        < _safe_divide(high.rolling(126).max() - low.rolling(126).min(), close).rolling(126).median() * 0.60
    ).astype(float)
    features["structure_channel_compression_20_100"] = _safe_divide(
        features["structure_channel_width_20d"],
        features["structure_channel_width_100d"],
    )

    return features.replace([np.inf, -np.inf], np.nan)


def latest_structure_snapshot(features: pd.DataFrame) -> dict[str, float]:
    """Return a compact latest structure snapshot for reports."""

    if features.empty:
        return {}
    latest = features.iloc[-1]
    keys = [
        "structure_close_to_support_63d",
        "structure_close_to_resistance_63d",
        "structure_range_position_63d",
        "structure_breakout_63d",
        "structure_breakdown_63d",
        "structure_breakout_volume_confirmed_63d",
        "structure_breakdown_volume_confirmed_63d",
        "structure_gap_up",
        "structure_gap_down",
        "structure_true_gap_up",
        "structure_true_gap_down",
        "structure_true_gap_size_pct",
        "structure_true_gap_atr_multiple",
        "structure_gap_corporate_action",
        "structure_gap_breakaway",
        "structure_gap_filled_same_day",
        "structure_key_reversal_top",
        "structure_key_reversal_bottom",
        "structure_one_day_reversal_top",
        "structure_one_day_reversal_bottom",
        "structure_selling_climax",
        "structure_spike_top",
        "structure_spike_bottom",
        "structure_runaway_day_up",
        "structure_runaway_day_down",
        "structure_trend_slope_50d",
        "structure_channel_position_50d",
        "structure_trend_state_short",
        "structure_trend_state_long",
        "structure_close_to_pivot_resistance",
        "structure_close_to_pivot_support",
        "structure_failed_breakout_63d",
        "structure_failed_breakdown_63d",
    ]
    return {key: float(latest[key]) for key in keys if key in latest.index and pd.notna(latest[key])}


def _rolling_trend_channel(log_close: pd.Series, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_demeaned = x - x_mean
    denominator = float(np.sum(x_demeaned**2))

    slopes = []
    positions = []
    widths = []
    values = log_close.to_numpy(dtype=float)
    for end in range(len(values)):
        start = end - window + 1
        if start < 0:
            slopes.append(np.nan)
            positions.append(np.nan)
            widths.append(np.nan)
            continue
        y = values[start : end + 1]
        if np.isnan(y).any():
            slopes.append(np.nan)
            positions.append(np.nan)
            widths.append(np.nan)
            continue
        y_mean = y.mean()
        slope = float(np.sum(x_demeaned * (y - y_mean)) / denominator)
        intercept = y_mean - slope * x_mean
        fitted = intercept + slope * x
        residuals = y - fitted
        residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        lower = fitted[-1] - 2 * residual_std
        upper = fitted[-1] + 2 * residual_std
        positions.append(float((y[-1] - lower) / (upper - lower)) if upper > lower else 0.5)
        widths.append(float(upper - lower))
        slopes.append(slope)

    return (
        pd.Series(slopes, index=log_close.index),
        pd.Series(positions, index=log_close.index),
        pd.Series(widths, index=log_close.index),
    )


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0.0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


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


def _confirmed_pivots(high: pd.Series, low: pd.Series, left: int, right: int) -> tuple[pd.Series, pd.Series]:
    window = left + right + 1
    raw_pivot_high = (high == high.rolling(window=window, center=True).max()).astype(float)
    raw_pivot_low = (low == low.rolling(window=window, center=True).min()).astype(float)
    return raw_pivot_high.shift(right).fillna(0.0), raw_pivot_low.shift(right).fillna(0.0)


def _age_since_signal(signal: pd.Series) -> pd.Series:
    values = signal.fillna(0.0).to_numpy(dtype=float)
    ages = []
    age = np.nan
    for value in values:
        if value > 0:
            age = 0.0
        elif np.isnan(age):
            age = np.nan
        else:
            age += 1.0
        ages.append(age)
    return pd.Series(ages, index=signal.index)
