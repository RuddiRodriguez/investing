from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from market_forecasting_engine.data import normalize_price_frame


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
        forecast_timestamp = latest_timestamp + pd.Timedelta(minutes=interval_minutes * horizon_bars)
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
