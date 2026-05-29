from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd


def analyze_chapter_10_bayesian_ml(
    *,
    forecasts: list[dict[str, Any]],
    candidate_results: dict[str, list[dict[str, Any]]],
    selected_validation_predictions: dict[str, list[dict[str, Any]]],
    model_cards: dict[str, Any],
    horizons: tuple[int, ...],
    enable_heavy_bayesian: bool = False,
    mcmc_draws: int = 300,
    mcmc_tune: int = 300,
) -> dict[str, Any]:
    """Return Chapter 10 Bayesian ML diagnostics and model-selection consequences."""

    horizon_reports = {}
    penalized_count = 0
    for horizon in horizons:
        key = str(horizon)
        forecast = _forecast_for_horizon(forecasts, horizon)
        candidate_rows = candidate_results.get(key, [])
        selected_metrics = forecast.get("validation_metrics", {}) if isinstance(forecast, dict) else {}
        candidate_comparison = _candidate_bayesian_comparison(candidate_rows)
        if _number(selected_metrics.get("chapter_10_bayesian_selection_penalty")) > 0:
            penalized_count += 1
        horizon_reports[key] = {
            "horizon_days": int(horizon),
            "selected_model": forecast.get("selected_model") if isinstance(forecast, dict) else None,
            "selected_posterior": _selected_posterior_summary(selected_metrics),
            "candidate_bayesian_comparison": candidate_comparison,
            "posterior_predictive_confidence": _posterior_predictive_confidence(forecast, model_cards.get(key, {})),
            "heavy_bayesian_path": _heavy_bayesian_path(
                records=selected_validation_predictions.get(key, []),
                enabled=enable_heavy_bayesian,
                draws=mcmc_draws,
                tune=mcmc_tune,
            ),
        }

    return {
        "chapter": 10,
        "name": "Bayesian ML - Dynamic Sharpe Ratios and Pairs Trading",
        "status": "pass" if penalized_count == 0 else "warn",
        "decision_policy": {
            "influences_final_action": False,
            "influences_model_selection": True,
            "influences_forecast_confidence": True,
            "mode": "bayesian_selection_and_uncertainty_adjustment",
            "reason": "Bayesian Sharpe and uncertainty diagnostics adjust model selection and confidence, but do not directly override Buy/Hold/Sell.",
        },
        "heavy_bayesian_policy": {
            "enabled": bool(enable_heavy_bayesian),
            "default_enabled": False,
            "reason": (
                "Heavy PyMC/MCMC path was requested for this run."
                if enable_heavy_bayesian
                else "Conservative default avoids PyMC/MCMC cost during normal ticker runs."
            ),
            "direct_trading_gate_enabled": False,
            "stochastic_volatility_model_selection": "diagnostic_only_until_stable_improvement",
        },
        "horizons": horizon_reports,
        "technical_method_card": chapter_10_bayesian_ml_method_card(),
    }


def chapter_10_bayesian_ml_method_card() -> dict[str, Any]:
    return {
        "name": "jansen_ml4t_chapter_10_bayesian_ml",
        "version": "chapter_10_bayesian_ml_v1",
        "source": "Machine Learning for Algorithmic Trading, 2nd ed., Chapter 10",
        "purpose": "Use Bayesian Sharpe uncertainty and posterior predictive uncertainty to adjust model selection and confidence.",
        "decision_policy": "model_selection_and_confidence_adjustment",
        "implemented_components": [
            "bayesian_sharpe_probability",
            "candidate_selection_penalty",
            "posterior_predictive_confidence_multiplier",
            "optional_pymc_mcmc_path",
            "pairs_trading_dynamic_hedge_ratio_policy",
        ],
        "not_implemented_by_default": [
            "heavy_mcmc_every_run",
            "direct_bayesian_buy_sell_gate",
            "automatic_stochastic_volatility_model_promotion",
        ],
    }


def _candidate_bayesian_comparison(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for row in candidates:
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
        rows.append(
            {
                "model_name": row.get("model_name"),
                "model_family": row.get("model_family"),
                "sharpe_ratio": _finite(_number(metrics.get("sharpe_ratio"))),
                "bayesian_prob_sharpe_positive": _finite(_number(metrics.get("chapter_10_prob_sharpe_positive"))),
                "bayesian_selection_penalty": _finite(_number(metrics.get("chapter_10_bayesian_selection_penalty"))),
                "confidence_multiplier": _finite(_number(metrics.get("chapter_10_confidence_multiplier"))),
            }
        )
    penalized = [row for row in rows if _number(row["bayesian_selection_penalty"]) > 0]
    return {
        "candidate_count": int(len(rows)),
        "penalized_candidate_count": int(len(penalized)),
        "candidates": sorted(rows, key=lambda item: _number(item["bayesian_selection_penalty"]))[:20],
    }


def _selected_posterior_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "bayesian_selection_penalty": _finite(_number(metrics.get("chapter_10_bayesian_selection_penalty"))),
        "probability_sharpe_positive": _finite(_number(metrics.get("chapter_10_prob_sharpe_positive"))),
        "posterior_sharpe_mean": _finite(_number(metrics.get("chapter_10_posterior_sharpe_mean"))),
        "posterior_sharpe_std": _finite(_number(metrics.get("chapter_10_posterior_sharpe_std"))),
        "confidence_multiplier": _finite(_number(metrics.get("chapter_10_confidence_multiplier"))),
    }


def _posterior_predictive_confidence(forecast: dict[str, Any], model_card: dict[str, Any]) -> dict[str, Any]:
    metrics = forecast.get("validation_metrics", {}) if isinstance(forecast, dict) else {}
    interval = model_card.get("confidence_interval", {}) if isinstance(model_card, dict) else {}
    multiplier = _number(metrics.get("chapter_10_confidence_multiplier"))
    return {
        "status": "available" if multiplier > 0 else "not_available",
        "confidence_multiplier": _finite(multiplier),
        "calibration_sample_size": int(interval.get("sample_size", forecast.get("calibration_sample_size", 0)) or 0),
        "reason": "Small samples and uncertain Sharpe widen uncertainty by lowering directional confidence.",
    }


def _heavy_bayesian_path(
    *,
    records: list[dict[str, Any]],
    enabled: bool,
    draws: int,
    tune: int,
) -> dict[str, Any]:
    if not enabled:
        return {"status": "disabled", "reason": "Enable with --enable-bayesian-heavy."}
    frame = pd.DataFrame(records)
    if frame.empty or "actual_log_return" not in frame or "predicted_log_return" not in frame:
        return {"status": "not_available", "reason": "No selected validation predictions."}
    strategy = np.sign(pd.to_numeric(frame["predicted_log_return"], errors="coerce").fillna(0.0)) * pd.to_numeric(
        frame["actual_log_return"],
        errors="coerce",
    ).fillna(0.0)
    strategy = strategy.replace([np.inf, -np.inf], np.nan).dropna()
    if len(strategy) < 30:
        return {"status": "not_available", "reason": "Too few validation returns for MCMC."}
    try:
        os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
        os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
        os.environ.setdefault("PYTENSOR_FLAGS", "base_compiledir=/private/tmp/pytensor")
        import pymc as pm

        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0.0, sigma=max(float(strategy.std(ddof=1)), 1e-4))
            sigma = pm.HalfNormal("sigma", sigma=max(float(strategy.std(ddof=1)) * 2, 1e-4))
            pm.Normal("returns", mu=mu, sigma=sigma, observed=strategy.to_numpy(dtype=float))
            trace = pm.sample(
                draws=max(int(draws), 50),
                tune=max(int(tune), 50),
                chains=2,
                cores=1,
                progressbar=False,
                compute_convergence_checks=False,
                random_seed=42,
            )
        mu_samples = np.asarray(trace.posterior["mu"]).reshape(-1)
        sigma_samples = np.asarray(trace.posterior["sigma"]).reshape(-1)
        sharpe_samples = np.divide(mu_samples, np.maximum(sigma_samples, 1e-12)) * np.sqrt(252)
        return {
            "status": "executed",
            "backend": "pymc",
            "draws": int(max(int(draws), 50)),
            "tune": int(max(int(tune), 50)),
            "posterior_mean_return": _finite(float(np.mean(mu_samples))),
            "posterior_volatility": _finite(float(np.mean(sigma_samples))),
            "probability_sharpe_positive": _finite(float((sharpe_samples > 0).mean())),
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "backend": "pymc",
            "reason": str(exc),
        }


def _forecast_for_horizon(forecasts: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    for forecast in forecasts:
        if int(forecast.get("horizon_days", -1)) == int(horizon):
            return forecast
    return {}


def _number(value: Any) -> float:
    try:
        number = float(value if value is not None else 0.0)
    except Exception:
        return 0.0
    return number if np.isfinite(number) else 0.0


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0
