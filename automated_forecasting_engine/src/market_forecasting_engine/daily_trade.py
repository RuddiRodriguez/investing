from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from market_forecasting_engine.data import normalize_price_frame

REGULAR_SESSION_OPEN = (9, 30)
REGULAR_SESSION_CLOSE = (16, 0)


@dataclass(frozen=True)
class DailyTradeConfig:
    """Configuration for one same-session trading plan."""

    ticker: str
    interval: str = "5m"
    target_column: str = "close"
    opening_range_bars: int = 6
    fast_ema_bars: int = 9
    slow_ema_bars: int = 21
    atr_bars: int = 14
    minimum_session_bars: int = 20
    minimum_score_to_trade: float = 2.0
    risk_reward: float = 1.8
    stop_atr_multiple: float = 1.2
    max_hold_bars: int = 24
    forecast_hours: tuple[float, ...] = (1.0, 2.0, 4.0)
    transaction_cost_bps: float = 2.0
    model_version: str = "0.1.0-intraday"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_daily_trade_plan(prices: pd.DataFrame, config: DailyTradeConfig) -> dict[str, Any]:
    """Build a same-session trade plan from intraday OHLCV bars.

    The plan is deliberately rule-based. It is meant to be auditable and usable
    before adding a broker or learned intraday model.
    """

    frame = normalize_price_frame(prices, target_column=config.target_column)
    if len(frame) < config.minimum_session_bars:
        raise ValueError(
            f"Need at least {config.minimum_session_bars} intraday rows; received {len(frame)}."
        )

    interval_minutes = infer_bar_interval_minutes(frame.index)
    has_intraday_data = _has_intraday_timestamps(frame.index)
    latest_session = _latest_session(frame)
    if len(latest_session) < config.minimum_session_bars:
        latest_session = frame.tail(config.minimum_session_bars)

    target = config.target_column.lower()
    close = latest_session[target].astype(float)
    high = latest_session["high"].astype(float) if "high" in latest_session.columns else close
    low = latest_session["low"].astype(float) if "low" in latest_session.columns else close
    volume = (
        latest_session["volume"].astype(float)
        if "volume" in latest_session.columns
        else pd.Series(1.0, index=latest_session.index)
    )
    typical_price = (high + low + close) / 3.0
    vwap = (typical_price * volume).cumsum() / volume.replace(0.0, np.nan).cumsum()
    fast_ema = close.ewm(span=config.fast_ema_bars, adjust=False).mean()
    slow_ema = close.ewm(span=config.slow_ema_bars, adjust=False).mean()
    true_range = _true_range(high, low, close)
    atr = true_range.rolling(config.atr_bars, min_periods=max(2, config.atr_bars // 2)).mean()
    opening = latest_session.head(min(config.opening_range_bars, len(latest_session)))
    opening_high = float(opening["high"].max() if "high" in opening.columns else opening[target].max())
    opening_low = float(opening["low"].min() if "low" in opening.columns else opening[target].min())

    latest_price = float(close.iloc[-1])
    latest_vwap = float(vwap.iloc[-1])
    latest_atr = float(atr.dropna().iloc[-1]) if atr.notna().any() else float(true_range.tail(5).mean())
    if not np.isfinite(latest_atr) or latest_atr <= 0:
        latest_atr = max(latest_price * 0.0025, 0.01)

    signals = _score_intraday_setup(
        latest_price=latest_price,
        latest_vwap=latest_vwap,
        fast_ema=float(fast_ema.iloc[-1]),
        slow_ema=float(slow_ema.iloc[-1]),
        opening_high=opening_high,
        opening_low=opening_low,
        close=close,
        volume=volume,
    )
    direction = _direction_from_score(signals["score"], config.minimum_score_to_trade)
    trade_plan = _trade_plan_for_direction(
        direction=direction,
        latest_price=latest_price,
        latest_atr=latest_atr,
        config=config,
        interval_minutes=interval_minutes,
    )
    forecasts = _hourly_forecasts(
        close=close,
        latest_price=latest_price,
        latest_atr=latest_atr,
        latest_timestamp=pd.Timestamp(latest_session.index[-1]),
        interval_minutes=interval_minutes,
        signal_score=float(signals["score"]),
        forecast_hours=config.forecast_hours,
    )

    return {
        "ticker": config.ticker.upper(),
        "as_of": latest_session.index[-1].isoformat(),
        "mode": "same_session_intraday_trade",
        "requires_intraday_data": True,
        "has_intraday_data": has_intraday_data,
        "interval_minutes": interval_minutes,
        "source_rows": int(len(frame)),
        "session_rows": int(len(latest_session)),
        "latest_price": latest_price,
        "vwap": latest_vwap,
        "opening_range": {
            "bars": int(len(opening)),
            "high": opening_high,
            "low": opening_low,
        },
        "signals": signals,
        "forecasts": forecasts,
        "trade_plan": trade_plan,
        "data_warning": None
        if has_intraday_data
        else "Input looks like daily/end-of-day data. Use 1m, 5m, or 15m bars for same-session trading.",
        "config": config.to_dict(),
    }


def _hourly_forecasts(
    *,
    close: pd.Series,
    latest_price: float,
    latest_atr: float,
    latest_timestamp: pd.Timestamp,
    interval_minutes: float | None,
    signal_score: float,
    forecast_hours: tuple[float, ...],
) -> list[dict[str, Any]]:
    if interval_minutes is None or interval_minutes <= 0:
        interval_minutes = 5.0
    returns = np.log(close.replace(0, np.nan)).diff().dropna()
    recent_drift_per_bar = float(returns.tail(12).mean()) if not returns.empty else 0.0
    score_tilt_per_bar = np.clip(signal_score, -4.0, 4.0) * 0.00015
    expected_return_per_bar = recent_drift_per_bar + float(score_tilt_per_bar)
    volatility_per_bar = float(returns.tail(24).std()) if len(returns) >= 2 else 0.0
    atr_return = latest_atr / latest_price if latest_price else 0.0
    volatility_per_bar = max(volatility_per_bar, atr_return * 0.35, 0.0005)

    forecasts: list[dict[str, Any]] = []
    for hours in forecast_hours:
        horizon_bars = max(1, int(round((float(hours) * 60.0) / interval_minutes)))
        expected_log_return = expected_return_per_bar * horizon_bars
        interval_width = 1.28 * volatility_per_bar * np.sqrt(horizon_bars)
        predicted_price = latest_price * float(np.exp(expected_log_return))
        lower_price = latest_price * float(np.exp(expected_log_return - interval_width))
        upper_price = latest_price * float(np.exp(expected_log_return + interval_width))
        forecast_timestamp = add_trading_bars(close.index, latest_timestamp, horizon_bars, interval_minutes)
        forecasts.append(
            {
                "horizon_hours": float(hours),
                "horizon_bars": int(horizon_bars),
                "forecast_timestamp": forecast_timestamp.isoformat(),
                "expected_log_return": float(expected_log_return),
                "expected_return": float(np.expm1(expected_log_return)),
                "predicted_price": float(predicted_price),
                "lower_price": float(lower_price),
                "upper_price": float(upper_price),
                "method": "recent_intraday_drift_plus_signal_tilt",
            }
        )
    return forecasts


def build_intraday_feature_frame(prices: pd.DataFrame, target_column: str = "close") -> pd.DataFrame:
    """Build features that only make sense for intraday bars."""

    frame = normalize_price_frame(prices, target_column=target_column)
    target = target_column.lower()
    close = frame[target].astype(float)
    high = frame["high"].astype(float) if "high" in frame.columns else close
    low = frame["low"].astype(float) if "low" in frame.columns else close
    open_price = frame["open"].astype(float) if "open" in frame.columns else close.shift(1)
    volume = frame["volume"].astype(float) if "volume" in frame.columns else pd.Series(1.0, index=frame.index)
    index = pd.DatetimeIndex(frame.index)
    session_key = index.date
    log_close = np.log(close.replace(0, np.nan))
    returns = log_close.diff()
    typical_price = (high + low + close) / 3.0
    grouped_volume = volume.where(volume != 0.0).groupby(session_key).cumsum()
    vwap = (typical_price * volume).groupby(session_key).cumsum() / grouped_volume

    features = pd.DataFrame(index=frame.index)
    for window in (1, 3, 6, 12, 24, 48):
        features[f"intraday_log_return_{window}_bars"] = log_close.diff(window)
        features[f"intraday_momentum_{window}_bars"] = close.pct_change(window)
    for window in (6, 12, 24, 48):
        features[f"intraday_realized_vol_{window}_bars"] = returns.rolling(window).std()
        features[f"intraday_volume_z_{window}_bars"] = _rolling_zscore(volume, window)

    session_open = close.groupby(session_key).transform("first")
    opening_high = high.groupby(session_key).transform(lambda values: values.head(6).max())
    opening_low = low.groupby(session_key).transform(lambda values: values.head(6).min())
    session_high = high.groupby(session_key).cummax()
    session_low = low.groupby(session_key).cummin()
    minute_of_day = index.hour * 60 + index.minute
    open_minutes = REGULAR_SESSION_OPEN[0] * 60 + REGULAR_SESSION_OPEN[1]
    close_minutes = REGULAR_SESSION_CLOSE[0] * 60 + REGULAR_SESSION_CLOSE[1]
    session_progress = np.clip((minute_of_day - open_minutes) / max(close_minutes - open_minutes, 1), 0.0, 1.0)

    features["intraday_close_to_vwap"] = close / vwap - 1
    features["intraday_open_gap"] = open_price / close.shift(1) - 1
    features["intraday_close_to_session_open"] = close / session_open - 1
    features["intraday_close_to_opening_high"] = close / opening_high - 1
    features["intraday_close_to_opening_low"] = close / opening_low - 1
    features["intraday_session_range_position"] = (close - session_low) / (session_high - session_low).replace(0.0, np.nan)
    features["intraday_ema_9_to_21"] = close.ewm(span=9, adjust=False).mean() / close.ewm(span=21, adjust=False).mean() - 1
    features["intraday_range_pct"] = (high - low) / close
    features["intraday_relative_volume_24_bars"] = volume / volume.rolling(24).mean() - 1
    features["intraday_minute_of_day"] = minute_of_day.astype(float)
    features["intraday_session_progress"] = session_progress.astype(float)
    features["intraday_day_of_week"] = index.dayofweek.astype(float)
    features["intraday_is_opening_hour"] = (session_progress <= 60 / max(close_minutes - open_minutes, 1)).astype(float)
    features["intraday_is_closing_hour"] = (session_progress >= 1 - 60 / max(close_minutes - open_minutes, 1)).astype(float)
    return features.replace([np.inf, -np.inf], np.nan).ffill()


def build_intraday_risk_context(prices: pd.DataFrame, target_column: str = "close") -> dict[str, Any]:
    frame = normalize_price_frame(prices, target_column=target_column)
    target = target_column.lower()
    close = frame[target].astype(float)
    returns = np.log(close.replace(0, np.nan)).diff().replace([np.inf, -np.inf], np.nan)
    recent = returns.tail(78).dropna()
    downside = recent[recent < 0]
    realized_vol = float(recent.std() * np.sqrt(max(len(recent), 1))) if len(recent) >= 2 else 0.0
    downside_vol = float(downside.std() * np.sqrt(max(len(recent), 1))) if len(downside) >= 2 else realized_vol
    var_95 = float(recent.quantile(0.05)) if len(recent) else 0.0
    expected_shortfall = float(recent[recent <= var_95].mean()) if len(recent[recent <= var_95]) else var_95
    drawdown = float(close.iloc[-1] / close.tail(78).max() - 1.0) if len(close) else 0.0
    tail_loss = abs(expected_shortfall)
    uncertainty = max(0.0, realized_vol * 0.65 + tail_loss * 0.25)
    risk_score = float(
        np.clip(
            (realized_vol / 0.035) * 0.35
            + (downside_vol / 0.030) * 0.25
            + (abs(drawdown) / 0.060) * 0.20
            + (tail_loss / 0.015) * 0.20,
            0.0,
            1.0,
        )
    )
    return {
        "realized_session_volatility": realized_vol,
        "downside_session_volatility": downside_vol,
        "var_95_log_return": var_95,
        "expected_shortfall_95_log_return": expected_shortfall,
        "drawdown_from_session_high": drawdown,
        "uncertainty": uncertainty,
        "risk_score": risk_score,
        "risk_penalty": risk_score * 0.006 + uncertainty * 0.20,
    }


def build_intraday_chart_confirmation(prices: pd.DataFrame, target_column: str = "close") -> dict[str, Any]:
    frame = normalize_price_frame(prices, target_column=target_column)
    target = target_column.lower()
    close = frame[target].astype(float)
    high = frame["high"].astype(float) if "high" in frame.columns else close
    low = frame["low"].astype(float) if "low" in frame.columns else close
    volume = frame["volume"].astype(float) if "volume" in frame.columns else pd.Series(1.0, index=frame.index)
    if len(close) < 20:
        return {"status": "insufficient_history"}
    lookback = min(78, max(20, len(close) - 1))
    prior_high = float(high.shift(1).tail(lookback).max())
    prior_low = float(low.shift(1).tail(lookback).min())
    latest_price = float(close.iloc[-1])
    volume_ratio = float(volume.iloc[-1] / volume.tail(20).mean()) if volume.tail(20).mean() else 1.0
    trend = "uptrend" if close.iloc[-1] > close.ewm(span=21, adjust=False).mean().iloc[-1] else "downtrend"
    if latest_price > prior_high * 1.001:
        breakout_status = "confirmed_breakout"
    elif latest_price < prior_low * 0.999:
        breakout_status = "confirmed_breakdown"
    elif latest_price >= prior_high * 0.995:
        breakout_status = "near_breakout"
    elif latest_price <= prior_low * 1.005:
        breakout_status = "near_breakdown"
    else:
        breakout_status = "inside_range"
    volume_confirmation = "strong_volume" if volume_ratio >= 1.4 else "weak_volume" if volume_ratio < 1.1 else "normal_volume"
    return {
        "status": "ok",
        "trend_status": trend,
        "support_level": prior_low,
        "resistance_level": prior_high,
        "breakout_status": breakout_status,
        "volume_confirmation": volume_confirmation,
        "volume_ratio": volume_ratio,
    }


def add_trading_minutes(timestamp: pd.Timestamp, minutes: float) -> pd.Timestamp:
    """Add regular-session US equity minutes, skipping nights and weekends."""

    remaining = float(minutes)
    current = _coerce_regular_session_timestamp(pd.Timestamp(timestamp))
    while remaining > 1e-9:
        close_time = _session_close(current)
        available = max(0.0, (close_time - current).total_seconds() / 60.0)
        if remaining <= available:
            return current + pd.Timedelta(minutes=remaining)
        remaining -= available
        current = _next_session_open(current + pd.Timedelta(days=1))
    return current


def add_trading_bars(
    index: pd.DatetimeIndex,
    timestamp: pd.Timestamp,
    bars: int,
    interval_minutes: float | None = None,
) -> pd.Timestamp:
    """Return the timestamp of the Nth future regular-session bar.

    The session clock is inferred from the observed intraday data. This avoids
    hardcoding an exchange timezone after providers normalize timestamps.
    """

    if bars <= 0:
        return pd.Timestamp(timestamp)
    observed = pd.DatetimeIndex(index).sort_values()
    if len(observed) < 2:
        return add_trading_minutes(timestamp, float(interval_minutes or 5.0) * bars)
    interval = float(interval_minutes or infer_bar_interval_minutes(observed) or 5.0)
    session_template = _observed_session_template(observed)
    if session_template is None:
        return add_trading_minutes(timestamp, interval * bars)

    current = pd.Timestamp(timestamp)
    future_count = 0
    day = current.normalize()
    trades_weekends = _observed_has_weekend_sessions(observed)
    while True:
        if int(day.dayofweek) >= 5 and not trades_weekends:
            day += pd.Timedelta(days=1)
            continue
        session_times = [
            day + pd.Timedelta(minutes=minute)
            for minute in session_template
            if day + pd.Timedelta(minutes=minute) > current
        ]
        for candidate in session_times:
            future_count += 1
            if future_count >= bars:
                return candidate
        day += pd.Timedelta(days=1)


def _observed_session_template(index: pd.DatetimeIndex) -> list[float] | None:
    frame = pd.DataFrame({"timestamp": index})
    frame["date"] = frame["timestamp"].dt.date
    grouped = frame.groupby("date")["timestamp"].apply(list)
    if grouped.empty:
        return None
    longest = max(grouped, key=len)
    if len(longest) < 2:
        return None
    return [ts.hour * 60 + ts.minute + ts.second / 60.0 for ts in longest]


def _observed_has_weekend_sessions(index: pd.DatetimeIndex) -> bool:
    return bool((pd.DatetimeIndex(index).dayofweek >= 5).any())


def infer_bar_interval_minutes(index: pd.DatetimeIndex) -> float | None:
    if len(index) < 2:
        return None
    deltas = pd.Series(index.sort_values()).diff().dropna()
    if deltas.empty:
        return None
    minutes = deltas.dt.total_seconds() / 60.0
    intraday_minutes = minutes[minutes < 18 * 60]
    if intraday_minutes.empty:
        return float(minutes.median())
    return float(intraday_minutes.median())


def _latest_session(frame: pd.DataFrame) -> pd.DataFrame:
    index = pd.DatetimeIndex(frame.index)
    latest_date = index[-1].date()
    return frame[index.date == latest_date]


def _has_intraday_timestamps(index: pd.DatetimeIndex) -> bool:
    if len(index) < 2:
        return False
    midnight = pd.Timestamp("00:00:00").time()
    if any(timestamp.time() != midnight for timestamp in index):
        return True
    interval = infer_bar_interval_minutes(index)
    return bool(interval is not None and interval < 18 * 60)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / rolling_std.replace(0.0, np.nan)


def _coerce_regular_session_timestamp(timestamp: pd.Timestamp) -> pd.Timestamp:
    current = timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
    open_time = _session_open(current)
    close_time = _session_close(current)
    if not _is_trading_day(current):
        return _next_session_open(current)
    if current < open_time:
        return open_time
    if current >= close_time:
        return _next_session_open(current + pd.Timedelta(days=1))
    return current


def _is_trading_day(timestamp: pd.Timestamp) -> bool:
    return int(timestamp.dayofweek) < 5


def _session_open(timestamp: pd.Timestamp) -> pd.Timestamp:
    return timestamp.normalize() + pd.Timedelta(hours=REGULAR_SESSION_OPEN[0], minutes=REGULAR_SESSION_OPEN[1])


def _session_close(timestamp: pd.Timestamp) -> pd.Timestamp:
    return timestamp.normalize() + pd.Timedelta(hours=REGULAR_SESSION_CLOSE[0], minutes=REGULAR_SESSION_CLOSE[1])


def _next_session_open(timestamp: pd.Timestamp) -> pd.Timestamp:
    current = timestamp.normalize()
    while not _is_trading_day(current):
        current += pd.Timedelta(days=1)
    return _session_open(current)


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


def _score_intraday_setup(
    *,
    latest_price: float,
    latest_vwap: float,
    fast_ema: float,
    slow_ema: float,
    opening_high: float,
    opening_low: float,
    close: pd.Series,
    volume: pd.Series,
) -> dict[str, Any]:
    score = 0.0
    reasons: list[str] = []

    if latest_price > latest_vwap:
        score += 1.0
        reasons.append("price_above_vwap")
    else:
        score -= 1.0
        reasons.append("price_below_vwap")

    if fast_ema > slow_ema:
        score += 1.0
        reasons.append("fast_ema_above_slow_ema")
    else:
        score -= 1.0
        reasons.append("fast_ema_below_slow_ema")

    if latest_price > opening_high:
        score += 1.0
        reasons.append("opening_range_breakout")
    elif latest_price < opening_low:
        score -= 1.0
        reasons.append("opening_range_breakdown")
    else:
        reasons.append("inside_opening_range")

    recent_return = float(close.pct_change(3).iloc[-1]) if len(close) >= 4 else 0.0
    if recent_return > 0:
        score += 0.5
        reasons.append("positive_recent_momentum")
    elif recent_return < 0:
        score -= 0.5
        reasons.append("negative_recent_momentum")

    volume_sma = volume.rolling(20, min_periods=5).mean()
    relative_volume = (
        float(volume.iloc[-1] / volume_sma.iloc[-1])
        if volume_sma.notna().any() and volume_sma.iloc[-1]
        else 1.0
    )
    if relative_volume >= 1.2:
        score += 0.5 if score >= 0 else -0.5
        reasons.append("volume_confirms_move")

    return {
        "score": float(score),
        "reasons": reasons,
        "recent_return_3_bars": recent_return,
        "relative_volume": relative_volume,
    }


def _direction_from_score(score: float, threshold: float) -> str:
    if score >= threshold:
        return "long"
    if score <= -threshold:
        return "short"
    return "no_trade"


def _trade_plan_for_direction(
    *,
    direction: str,
    latest_price: float,
    latest_atr: float,
    config: DailyTradeConfig,
    interval_minutes: float | None,
) -> dict[str, Any]:
    hold_minutes = None if interval_minutes is None else int(round(config.max_hold_bars * interval_minutes))
    if direction == "no_trade":
        return {
            "action": "no_trade",
            "reason": "Intraday score did not clear the configured threshold.",
            "max_hold_minutes": hold_minutes,
        }

    stop_distance = latest_atr * config.stop_atr_multiple
    target_distance = stop_distance * config.risk_reward
    if direction == "long":
        stop = latest_price - stop_distance
        target = latest_price + target_distance
    else:
        stop = latest_price + stop_distance
        target = latest_price - target_distance

    return {
        "action": direction,
        "entry_reference": latest_price,
        "stop": float(stop),
        "take_profit": float(target),
        "stop_distance": float(stop_distance),
        "target_distance": float(target_distance),
        "risk_reward": float(config.risk_reward),
        "max_hold_bars": int(config.max_hold_bars),
        "max_hold_minutes": hold_minutes,
        "transaction_cost_bps": float(config.transaction_cost_bps),
    }
