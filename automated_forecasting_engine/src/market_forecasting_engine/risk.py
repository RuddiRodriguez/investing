from __future__ import annotations

import math

import numpy as np
import pandas as pd


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.where(np.abs(actual) < 1e-12, np.nan, np.abs(actual))
    return float(np.nanmean(np.abs((actual - predicted) / denominator)))


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.abs(actual) + np.abs(predicted)
    denominator = np.where(denominator < 1e-12, np.nan, denominator)
    return float(np.nanmean(2 * np.abs(predicted - actual) / denominator))


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.sign(actual) != 0
    if not valid.any():
        return 0.0
    return float((np.sign(actual[valid]) == np.sign(predicted[valid])).mean())


def information_criteria(actual: np.ndarray, predicted: np.ndarray, parameter_count: int) -> tuple[float, float]:
    n_obs = max(len(actual), 1)
    residual_sum_squares = float(np.sum(np.square(actual - predicted)))
    sigma2 = max(residual_sum_squares / n_obs, 1e-12)
    k = max(parameter_count, 1)
    aic = n_obs * math.log(sigma2) + 2 * k
    bic = n_obs * math.log(sigma2) + math.log(n_obs) * k
    return float(aic), float(bic)


def evaluate_signal_risk(actual_log_returns: np.ndarray, predicted_log_returns: np.ndarray, horizon_days: int) -> dict[str, float]:
    signals = np.sign(predicted_log_returns)
    strategy_log_returns = signals * actual_log_returns
    strategy_simple_returns = np.expm1(strategy_log_returns)

    if len(strategy_simple_returns) == 0:
        return {
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "hit_rate": 0.0,
            "profit_factor": 0.0,
        }

    periods_per_year = 252 / max(horizon_days, 1)
    std = float(np.std(strategy_simple_returns, ddof=1)) if len(strategy_simple_returns) > 1 else 0.0
    sharpe = 0.0 if std == 0 else float(np.mean(strategy_simple_returns) / std * math.sqrt(periods_per_year))

    equity = pd.Series(np.cumprod(1 + strategy_simple_returns))
    running_high = equity.cummax()
    drawdown = (equity / running_high) - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    positive = strategy_simple_returns[strategy_simple_returns > 0].sum()
    negative = strategy_simple_returns[strategy_simple_returns < 0].sum()
    profit_factor = float(positive / abs(negative)) if negative < 0 else float("inf") if positive > 0 else 0.0
    hit_rate = float((strategy_simple_returns > 0).mean())

    return {
        "sharpe_ratio": _finite(sharpe),
        "max_drawdown": _finite(max_drawdown),
        "hit_rate": _finite(hit_rate),
        "profit_factor": _finite(profit_factor),
    }


def risk_level(validation_metrics: dict[str, float], directional_confidence: float) -> str:
    score = 0
    if validation_metrics.get("directional_accuracy", 0.0) < 0.50:
        score += 1
    if validation_metrics.get("sharpe_ratio", 0.0) < 0.0:
        score += 1
    if validation_metrics.get("max_drawdown", 0.0) < -0.20:
        score += 1
    if directional_confidence < 0.55:
        score += 1
    if validation_metrics.get("mae", 1.0) > 0.08:
        score += 1

    if score >= 3:
        return "High"
    if score >= 1:
        return "Medium"
    return "Low"


def suggested_action(forecasts: list[dict[str, float | str]], risk: str) -> str:
    if not forecasts:
        return "Hold"

    preferred = next((forecast for forecast in forecasts if forecast["horizon_days"] == 5), forecasts[0])
    expected_return = float(preferred["expected_return"])
    confidence = float(preferred["directional_confidence"])
    validation_mae = float(preferred["validation_metrics"].get("mae", 0.03))  # type: ignore[union-attr]
    threshold = max(0.02, validation_mae)

    if risk == "High" or confidence < 0.55:
        return "Hold"
    if expected_return > threshold:
        return "Buy"
    if expected_return < -threshold:
        return "Sell"
    return "Hold"


def normal_directional_confidence(expected_log_return: float, residual_std: float) -> float:
    if residual_std <= 1e-12:
        return 0.50 if abs(expected_log_return) < 1e-12 else 0.99
    z_score = abs(expected_log_return) / residual_std
    confidence = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))
    return float(min(0.99, max(0.50, confidence)))


def z_value(confidence_level: float) -> float:
    if confidence_level >= 0.95:
        return 1.96
    if confidence_level >= 0.90:
        return 1.6448536269514722
    if confidence_level >= 0.80:
        return 1.2815515655446004
    return 1.0


def _finite(value: float) -> float:
    if math.isfinite(value):
        return float(value)
    return 999.0 if value > 0 else -999.0
