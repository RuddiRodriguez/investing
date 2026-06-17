from __future__ import annotations

import math
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd


_PIPELINE_CACHE: dict[str, Any] = {}
_FORECAST_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def build_chronos_forecast(
    *,
    price_bars: list[dict[str, Any]],
    forecast_hours: tuple[float, ...],
    data_interval: str,
    model_name: str,
    context_rows: int,
    num_samples: int,
    refresh_seconds: int,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False, "status": "disabled"}
    cache_key = "|".join(
        [
            model_name,
            data_interval,
            ",".join(str(value) for value in forecast_hours),
            str(context_rows),
            str(num_samples),
            _last_bar_signature(price_bars),
        ]
    )
    now_monotonic = time.monotonic()
    cached = _FORECAST_CACHE.get(cache_key)
    if cached and now_monotonic - cached[0] < max(1, int(refresh_seconds)):
        return {**cached[1], "cache": "hit"}
    try:
        forecast = _build_chronos_forecast_uncached(
            price_bars=price_bars,
            forecast_hours=forecast_hours,
            data_interval=data_interval,
            model_name=model_name,
            context_rows=context_rows,
            num_samples=num_samples,
        )
    except Exception as exc:
        forecast = {
            "enabled": True,
            "status": "failed",
            "model": model_name,
            "reason": str(exc),
            "note": "Chronos forecast is optional evidence for the LLM trader; LLM decision still runs without it.",
        }
    _FORECAST_CACHE[cache_key] = (now_monotonic, forecast)
    return {**forecast, "cache": "miss"}


def _build_chronos_forecast_uncached(
    *,
    price_bars: list[dict[str, Any]],
    forecast_hours: tuple[float, ...],
    data_interval: str,
    model_name: str,
    context_rows: int,
    num_samples: int,
) -> dict[str, Any]:
    import torch

    closes = [_float_or_none(row.get("close")) for row in price_bars]
    timestamps = [_parse_timestamp(row.get("timestamp")) for row in price_bars]
    series = [(ts, close) for ts, close in zip(timestamps, closes, strict=False) if ts is not None and close is not None]
    if len(series) < 24:
        return {"enabled": True, "status": "skipped", "model": model_name, "reason": "not_enough_price_bars", "rows": len(series)}
    series = series[-max(24, int(context_rows)) :]
    last_timestamp = series[-1][0]
    last_price = float(series[-1][1])
    interval_minutes = _interval_minutes(data_interval)
    horizon_steps = _horizon_steps(forecast_hours, interval_minutes)
    prediction_length = max(horizon_steps.values())
    context = torch.tensor([float(close) for _, close in series], dtype=torch.float32)
    pipeline = _pipeline(model_name)
    if "chronos-bolt" in model_name.lower() or "chronos-2" in model_name.lower():
        samples = pipeline.predict(context, prediction_length=prediction_length)
        if samples.ndim == 3:
            samples = samples[0]
        samples = samples.float()
        if samples.shape[0] >= 3:
            quantiles = samples[:3]
        else:
            quantiles = torch.vstack([samples[0], samples[0], samples[-1]])
    else:
        samples = pipeline.predict(context, prediction_length=prediction_length, num_samples=max(8, int(num_samples)))
        if samples.ndim == 3:
            samples = samples[0]
        quantiles = torch.quantile(samples.float(), torch.tensor([0.1, 0.5, 0.9]), dim=0)
    forecast_path = []
    for step in range(1, prediction_length + 1):
        forecast_path.append(
            {
                "step": step,
                "timestamp": (last_timestamp + timedelta(minutes=interval_minutes * step)).isoformat(),
                "lower_price": _finite_float(quantiles[0, step - 1].item()),
                "median_price": _finite_float(quantiles[1, step - 1].item()),
                "upper_price": _finite_float(quantiles[2, step - 1].item()),
            }
        )
    horizon_points = []
    for hours, step in horizon_steps.items():
        point = forecast_path[step - 1]
        median = point["median_price"]
        lower = point["lower_price"]
        upper = point["upper_price"]
        horizon_points.append(
            {
                "horizon_hours": hours,
                "step": step,
                "timestamp": point["timestamp"],
                "predicted_price": median,
                "lower_price": lower,
                "upper_price": upper,
                "predicted_move_pct": None if median is None or last_price == 0 else median / last_price - 1.0,
                "direction": _direction(last_price, median),
            }
        )
    signal_points = _short_horizon_signal_points(
        series=series,
        forecast_hours=forecast_hours,
        interval_minutes=interval_minutes,
    )
    chronos_collapsed = _forecast_collapsed(horizon_points)
    return {
        "enabled": True,
        "status": "ok",
        "provider": "huggingface",
        "model": model_name,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "context_rows": len(series),
        "data_interval": data_interval,
        "interval_minutes": interval_minutes,
        "num_samples": max(8, int(num_samples)),
        "as_of_timestamp": last_timestamp.isoformat(),
        "as_of_price": last_price,
        "forecast_path": forecast_path,
        "horizon_points": horizon_points,
        "short_horizon_signal_points": signal_points,
        "preferred_horizon_points": signal_points if chronos_collapsed else horizon_points,
        "preferred_source": "oscillation_aware_short_horizon_signal" if chronos_collapsed else "chronos_median",
        "chronos_collapsed": chronos_collapsed,
        "interpretation": "Chronos forecast is a numeric time-series signal supplied to the LLM as evidence, not as the final trade decision.",
    }


def _pipeline(model_name: str):
    if model_name in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[model_name]
    import torch
    from chronos import ChronosBoltPipeline, ChronosPipeline

    kwargs = {"device_map": "cpu"}
    try:
        kwargs["dtype"] = torch.float32
        pipeline_class = ChronosBoltPipeline if "chronos-bolt" in model_name.lower() or "chronos-2" in model_name.lower() else ChronosPipeline
        pipeline = pipeline_class.from_pretrained(model_name, **kwargs)
    except TypeError:
        pipeline_class = ChronosBoltPipeline if "chronos-bolt" in model_name.lower() or "chronos-2" in model_name.lower() else ChronosPipeline
        pipeline = pipeline_class.from_pretrained(model_name)
    _PIPELINE_CACHE[model_name] = pipeline
    return pipeline


def _horizon_steps(forecast_hours: tuple[float, ...], interval_minutes: int) -> dict[float, int]:
    steps = {}
    for hours in forecast_hours:
        steps[float(hours)] = max(1, int(round(float(hours) * 60.0 / max(1, interval_minutes))))
    return steps


def _interval_minutes(interval: str) -> int:
    value = str(interval).strip().lower()
    if value.endswith("m"):
        return max(1, int(float(value[:-1])))
    if value.endswith("h"):
        return max(1, int(float(value[:-1]) * 60))
    if value.endswith("d"):
        return max(1, int(float(value[:-1]) * 1440))
    return 1


def _parse_timestamp(value: Any) -> datetime | None:
    try:
        parsed = pd.Timestamp(value).to_pydatetime()
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _float_or_none(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _finite_float(value: Any) -> float | None:
    output = _float_or_none(value)
    return None if output is None else float(output)


def _direction(last_price: float, predicted_price: float | None) -> str:
    if predicted_price is None:
        return "unknown"
    move = predicted_price / last_price - 1.0 if last_price else 0.0
    if move > 0.001:
        return "up"
    if move < -0.001:
        return "down"
    return "flat"


def _forecast_collapsed(points: list[dict[str, Any]]) -> bool:
    values = [_float_or_none(point.get("predicted_price")) for point in points]
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return False
    return max(values) - min(values) <= max(0.01, abs(values[-1]) * 0.00005)


def _short_horizon_signal_points(
    *,
    series: list[tuple[datetime, float]],
    forecast_hours: tuple[float, ...],
    interval_minutes: int,
) -> list[dict[str, Any]]:
    prices = [float(close) for _, close in series]
    if len(prices) < 30:
        return []
    last_timestamp = series[-1][0]
    last_price = prices[-1]
    window = prices[-90:]
    mean_30 = sum(prices[-30:]) / 30.0
    mean_90 = sum(window) / len(window)
    std_30 = _std(prices[-30:]) or max(last_price * 0.0005, 0.01)
    slope_9 = _linear_slope(prices[-9:])
    slope_21 = _linear_slope(prices[-21:])
    momentum_5 = last_price - prices[-6] if len(prices) >= 6 else 0.0
    momentum_15 = last_price - prices[-16] if len(prices) >= 16 else momentum_5
    oscillation_score = max(0.0, min(1.0, 1.0 - abs(slope_21) / max(std_30 / 8.0, 0.01)))
    trend_score = 1.0 - oscillation_score
    points: list[dict[str, Any]] = []
    for hours in forecast_hours:
        steps = max(1, int(round(float(hours) * 60.0 / max(1, interval_minutes))))
        horizon_damping = min(1.0, steps / 30.0)
        momentum_component = (0.45 * momentum_5 + 0.25 * momentum_15 + 0.30 * slope_9 * min(steps, 12)) * (0.45 + 0.55 * trend_score)
        reversion_target = (0.65 * mean_30) + (0.35 * mean_90)
        mean_reversion_component = (reversion_target - last_price) * oscillation_score * horizon_damping
        predicted = last_price + momentum_component + mean_reversion_component
        uncertainty = max(std_30 * (steps ** 0.5) * 0.55, last_price * 0.0007)
        tape_direction = _tape_direction(momentum_5=momentum_5, momentum_15=momentum_15, slope_9=slope_9)
        scalp_bias = _scalp_bias(
            tape_direction=tape_direction,
            momentum_component=momentum_component,
            mean_reversion_component=mean_reversion_component,
            oscillation_score=oscillation_score,
        )
        points.append(
            {
                "horizon_hours": float(hours),
                "step": steps,
                "timestamp": (last_timestamp + timedelta(minutes=interval_minutes * steps)).isoformat(),
                "predicted_price": _finite_float(predicted),
                "lower_price": _finite_float(predicted - uncertainty),
                "upper_price": _finite_float(predicted + uncertainty),
                "predicted_move_pct": None if last_price == 0 else predicted / last_price - 1.0,
                "direction": _direction(last_price, predicted),
                "tape_direction": tape_direction,
                "scalp_bias": scalp_bias,
                "signal_components": {
                    "momentum_component": _finite_float(momentum_component),
                    "mean_reversion_component": _finite_float(mean_reversion_component),
                    "oscillation_score": _finite_float(oscillation_score),
                    "trend_score": _finite_float(trend_score),
                    "sma_30": _finite_float(mean_30),
                    "sma_90": _finite_float(mean_90),
                    "slope_9_per_bar": _finite_float(slope_9),
                    "slope_21_per_bar": _finite_float(slope_21),
                    "momentum_5_bars": _finite_float(momentum_5),
                    "momentum_15_bars": _finite_float(momentum_15),
                    "local_std_30": _finite_float(std_30),
                },
            }
        )
    return points


def _linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    n = len(values)
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    denom = sum((i - x_mean) ** 2 for i in range(n))
    if denom == 0:
        return 0.0
    return sum((i - x_mean) * (value - y_mean) for i, value in enumerate(values)) / denom


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _tape_direction(*, momentum_5: float, momentum_15: float, slope_9: float) -> str:
    positives = sum(1 for value in (momentum_5, momentum_15, slope_9) if value > 0)
    negatives = sum(1 for value in (momentum_5, momentum_15, slope_9) if value < 0)
    if positives >= 2:
        return "short_term_up"
    if negatives >= 2:
        return "short_term_down"
    return "mixed"


def _scalp_bias(*, tape_direction: str, momentum_component: float, mean_reversion_component: float, oscillation_score: float) -> str:
    if tape_direction == "short_term_up" and momentum_component > abs(mean_reversion_component) * 0.55:
        return "call_bias_short_up_inside_chop" if oscillation_score >= 0.60 else "call_bias_directional"
    if tape_direction == "short_term_down" and momentum_component < -abs(mean_reversion_component) * 0.55:
        return "put_bias_short_down_inside_chop" if oscillation_score >= 0.60 else "put_bias_directional"
    if oscillation_score >= 0.75:
        return "mean_reversion_chop_wait_for_extreme"
    return "neutral_mixed"


def _last_bar_signature(price_bars: list[dict[str, Any]]) -> str:
    if not price_bars:
        return "empty"
    last = price_bars[-1]
    return f"{last.get('timestamp')}:{last.get('close')}"
