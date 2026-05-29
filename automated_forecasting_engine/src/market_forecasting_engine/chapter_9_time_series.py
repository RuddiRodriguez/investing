from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


TIME_SERIES_FAMILIES = {"classical_forecasting", "volatility_model", "multivariate_time_series", "state_space_model"}


def analyze_chapter_9_time_series(
    *,
    prices: pd.DataFrame,
    forecasts: list[dict[str, Any]],
    candidate_results: dict[str, list[dict[str, Any]]],
    selected_validation_predictions: dict[str, list[dict[str, Any]]],
    model_cards: dict[str, Any],
    horizons: tuple[int, ...],
    target_column: str = "close",
) -> dict[str, Any]:
    """Audit existing time-series models and residual behavior."""

    close = pd.to_numeric(prices[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    log_price = np.log(close.replace(0, np.nan)).dropna()
    log_return = log_price.diff().dropna()
    series_diagnostics = {
        "price_stationarity": _stationarity_test(log_price, "log_price"),
        "return_stationarity": _stationarity_test(log_return, "log_return"),
        "return_autocorrelation": _autocorrelation_summary(log_return),
        "volatility_clustering": _volatility_clustering(log_return),
    }

    horizon_reports = {}
    selected_ts_count = 0
    residual_warning_count = 0
    for horizon in horizons:
        key = str(horizon)
        forecast = _forecast_for_horizon(forecasts, horizon)
        model_family = str(forecast.get("selected_model_family", ""))
        selected_is_ts = model_family in TIME_SERIES_FAMILIES
        selected_ts_count += int(selected_is_ts)
        residual_audit = _residual_white_noise_audit(selected_validation_predictions.get(key, []))
        residual_warning_count += int(residual_audit["status"] != "pass")
        horizon_reports[key] = {
            "horizon_days": int(horizon),
            "selected_model": forecast.get("selected_model"),
            "selected_model_family": model_family,
            "selected_model_is_time_series": bool(selected_is_ts),
            "time_series_candidate_comparison": _time_series_candidate_comparison(candidate_results.get(key, [])),
            "residual_white_noise_audit": residual_audit,
            "selected_model_diagnostics": _selected_time_series_model_diagnostics(model_cards.get(key, {})),
        }

    promotion_policy = _promotion_policy(
        horizon_count=len(horizons),
        selected_ts_count=selected_ts_count,
        residual_warning_count=residual_warning_count,
    )
    return {
        "chapter": 9,
        "name": "Time-Series Models for Volatility Forecasts and Statistical Arbitrage",
        "status": "pass" if promotion_policy["candidate_for_promotion"] else "warn",
        "decision_policy": {
            "influences_final_action": False,
            "influences_model_selection": True,
            "mode": "model_selection_adjustment",
            "reason": "Chapter 9 diagnostics adjust candidate ranking before selection, but do not directly override the final action gate.",
        },
        "model_policy": {
            "new_arima_garch_models_added": False,
            "reason": "ARIMA, SARIMA, VAR, and GARCH candidates already exist; diagnostics are improved first.",
            "selection_adjustment": "enabled",
            "pairs_trading_module": "available_as_market_forecasting_engine.pairs_trading",
        },
        "series_diagnostics": series_diagnostics,
        "horizons": horizon_reports,
        "promotion_policy": promotion_policy,
        "technical_method_card": chapter_9_time_series_method_card(),
    }


def chapter_9_time_series_method_card() -> dict[str, Any]:
    return {
        "name": "jansen_ml4t_chapter_9_time_series",
        "version": "chapter_9_time_series_v1",
        "source": "Machine Learning for Algorithmic Trading, 2nd ed., Chapter 9",
        "purpose": "Audit stationarity, autocorrelation, volatility clustering, and existing time-series model residual quality.",
        "decision_policy": "diagnostic_only",
        "implemented_components": [
            "adf_stationarity_test_when_statsmodels_available",
            "acf_pacf_style_autocorrelation_summary",
            "ljung_box_residual_white_noise_test",
            "volatility_clustering_score",
            "existing_arima_sarima_var_garch_candidate_audit",
            "pairs_trading_utility_module",
        ],
        "not_implemented": [
            "duplicate_arima_garch_candidate_families",
            "automatic_pairs_trade_execution",
            "cointegration_action_gate",
        ],
    }


def _time_series_candidate_comparison(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for row in candidates:
        family = str(row.get("model_family", ""))
        name = str(row.get("model_name", ""))
        if family not in TIME_SERIES_FAMILIES and not name.startswith(("arima_", "sarima_", "var_", "garch_")):
            continue
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
        rows.append(
            {
                "model_name": name,
                "model_family": family,
                "mae": _finite(_number(metrics.get("mae"))),
                "rmse": _finite(_number(metrics.get("rmse"))),
                "directional_accuracy": _finite(_number(metrics.get("directional_accuracy"))),
                "sharpe_ratio": _finite(_number(metrics.get("sharpe_ratio"))),
                "deflated_sharpe_ratio": _finite(_number(metrics.get("deflated_sharpe_ratio"))),
                "chapter_9_selection_penalty": _finite(_number(metrics.get("chapter_9_selection_penalty"))),
                "chapter_9_adjusted_mae": _finite(_number(metrics.get("chapter_9_adjusted_mae"))),
                "chapter_9_adjusted_rmse": _finite(_number(metrics.get("chapter_9_adjusted_rmse"))),
                "chapter_9_residual_autocorrelation": _finite(_number(metrics.get("chapter_9_residual_autocorrelation"))),
            }
        )
    if not rows:
        return {"status": "no_time_series_candidates", "candidate_count": 0, "best_by_mae": None}
    return {
        "status": "available",
        "candidate_count": int(len(rows)),
        "best_by_mae": min(rows, key=lambda item: item["mae"]),
        "candidates": sorted(rows, key=lambda item: item["mae"]),
    }


def _selected_time_series_model_diagnostics(model_card: dict[str, Any]) -> dict[str, Any]:
    diagnostics = model_card.get("model_diagnostics", {}) if isinstance(model_card, dict) else {}
    parameters = model_card.get("model_parameters", {}) if isinstance(model_card, dict) else {}
    return {
        "status": "available" if diagnostics or parameters else "not_available",
        "parameters": parameters,
        "diagnostics": diagnostics,
    }


def _residual_white_noise_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(records)
    if frame.empty or "actual_log_return" not in frame or "predicted_log_return" not in frame:
        return {"status": "not_available", "rows": 0, "reason": "No validation predictions."}
    actual = pd.to_numeric(frame["actual_log_return"], errors="coerce")
    predicted = pd.to_numeric(frame["predicted_log_return"], errors="coerce")
    residuals = (actual - predicted).replace([np.inf, -np.inf], np.nan).dropna()
    if len(residuals) < 20:
        return {"status": "warn", "rows": int(len(residuals)), "reason": "Too few residuals for a reliable white-noise audit."}
    lb = _ljung_box(residuals, lags=min(10, max(1, len(residuals) // 5)))
    acf = _autocorrelation_values(residuals, max_lag=5)
    max_abs_acf = max([abs(value) for value in acf.values()], default=0.0)
    status = "pass" if lb["p_value"] >= 0.05 and max_abs_acf < 0.25 else "warn"
    return {
        "status": status,
        "rows": int(len(residuals)),
        "ljung_box": lb,
        "residual_autocorrelation": acf,
        "max_abs_residual_acf_lag_1_5": _finite(max_abs_acf),
        "reason": (
            "Residuals look close to white noise."
            if status == "pass"
            else "Residuals retain autocorrelation; time-series model assumptions remain imperfect."
        ),
    }


def _stationarity_test(series: pd.Series, label: str) -> dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 30:
        return {"status": "insufficient_rows", "series": label, "rows": int(len(clean))}
    try:
        from statsmodels.tsa.stattools import adfuller

        statistic, p_value, used_lag, nobs, *_ = adfuller(clean.to_numpy(dtype=float), autolag="AIC")
        stationary = bool(p_value < 0.05)
        return {
            "status": "stationary" if stationary else "unit_root_risk",
            "series": label,
            "rows": int(nobs),
            "adf_statistic": _finite(float(statistic)),
            "p_value": _finite(float(p_value)),
            "used_lag": int(used_lag),
        }
    except Exception:
        acf1 = clean.autocorr(lag=1)
        return {
            "status": "unit_root_risk" if _number(acf1) > 0.90 else "likely_stationary",
            "series": label,
            "rows": int(len(clean)),
            "lag_1_autocorrelation": _finite(_number(acf1)),
            "method": "acf_fallback",
        }


def _autocorrelation_summary(series: pd.Series, max_lag: int = 10) -> dict[str, Any]:
    values = _autocorrelation_values(series, max_lag=max_lag)
    high_lags = [lag for lag, value in values.items() if abs(value) >= 0.20]
    return {
        "status": "autocorrelated" if high_lags else "low_autocorrelation",
        "acf": values,
        "high_autocorrelation_lags": high_lags,
    }


def _volatility_clustering(series: pd.Series) -> dict[str, Any]:
    squared = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna() ** 2
    if len(squared) < 30:
        return {"status": "insufficient_rows", "score": 0.0}
    acf1 = _number(squared.autocorr(lag=1))
    acf5 = _number(squared.autocorr(lag=5)) if len(squared) > 10 else 0.0
    score = max(abs(acf1), abs(acf5))
    return {
        "status": "clustered_volatility" if score >= 0.15 else "low_clustering",
        "score": _finite(score),
        "squared_return_acf_1": _finite(acf1),
        "squared_return_acf_5": _finite(acf5),
    }


def _ljung_box(series: pd.Series, lags: int) -> dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < lags + 5:
        return {"statistic": 0.0, "p_value": 1.0, "lags": int(lags)}
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        result = acorr_ljungbox(clean, lags=[lags], return_df=True)
        return {
            "statistic": _finite(float(result["lb_stat"].iloc[-1])),
            "p_value": _finite(float(result["lb_pvalue"].iloc[-1])),
            "lags": int(lags),
        }
    except Exception:
        values = _autocorrelation_values(clean, max_lag=lags)
        n = len(clean)
        statistic = n * (n + 2) * sum((value**2) / max(n - lag, 1) for lag, value in values.items())
        return {"statistic": _finite(statistic), "p_value": _chi2_survival_rough(statistic), "lags": int(lags)}


def _autocorrelation_values(series: pd.Series, max_lag: int) -> dict[int, float]:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return {lag: _finite(_number(clean.autocorr(lag=lag))) for lag in range(1, max_lag + 1) if len(clean) > lag + 2}


def _promotion_policy(*, horizon_count: int, selected_ts_count: int, residual_warning_count: int) -> dict[str, Any]:
    candidate = horizon_count > 0 and selected_ts_count == horizon_count and residual_warning_count == 0
    return {
        "status": "candidate_for_time_series_gate_research" if candidate else "diagnostic_only",
        "candidate_for_promotion": bool(candidate),
        "time_series_selected_horizon_count": int(selected_ts_count),
        "residual_warning_horizon_count": int(residual_warning_count),
        "tested_horizon_count": int(horizon_count),
        "recommended_action": (
            "Research a time-series model preference gate before enabling action changes."
            if candidate
            else "Keep Chapter 9 diagnostic-only; do not prefer ARIMA/GARCH automatically."
        ),
    }


def _forecast_for_horizon(forecasts: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    for forecast in forecasts:
        if int(forecast.get("horizon_days", -1)) == int(horizon):
            return forecast
    return {}


def _chi2_survival_rough(value: float) -> float:
    return _finite(float(np.exp(-max(value, 0.0) / 2.0)))


def _number(value: Any) -> float:
    try:
        number = float(value if value is not None else 0.0)
    except Exception:
        return 0.0
    return number if np.isfinite(number) else 0.0


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0
