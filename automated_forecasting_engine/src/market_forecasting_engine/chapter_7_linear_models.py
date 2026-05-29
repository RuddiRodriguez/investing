from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


LINEAR_PREFIXES = ("ridge_", "lasso_", "elastic_net_")
LOWER_IS_BETTER = {"rmse", "mae", "mape", "smape", "aic", "bic", "composite"}
HIGHER_IS_BETTER = {"directional_accuracy", "sharpe_ratio", "hit_rate", "profit_factor"}


def analyze_chapter_7_linear_models(
    *,
    forecasts: list[dict[str, Any]],
    candidate_results: dict[str, list[dict[str, Any]]],
    selected_validation_predictions: dict[str, list[dict[str, Any]]],
    model_cards: dict[str, Any],
    horizons: tuple[int, ...],
    selection_metric: str,
) -> dict[str, Any]:
    """Evaluate Chapter 7 linear-model diagnostics without overriding selection."""

    horizon_reports = {}
    linear_wins = 0
    residual_warnings = 0
    for horizon in horizons:
        key = str(horizon)
        forecast = _forecast_for_horizon(forecasts, horizon)
        candidates = candidate_results.get(key, [])
        linear_comparison = _linear_candidate_comparison(
            candidates=candidates,
            selected_model=str(forecast.get("selected_model", "")),
            selection_metric=selection_metric,
        )
        residual_diagnostics = _residual_diagnostics(selected_validation_predictions.get(key, []))
        prediction_ic = _prediction_ic(selected_validation_predictions.get(key, []))
        coefficients = _linear_coefficients(model_cards.get(key, {}))
        linear_wins += int(linear_comparison.get("selected_model_is_linear", False))
        residual_warnings += int(residual_diagnostics["status"] in {"autocorrelated", "fat_tailed", "unstable"})
        horizon_reports[key] = {
            "horizon_days": int(horizon),
            "selected_model": forecast.get("selected_model"),
            "selected_model_family": forecast.get("selected_model_family"),
            "linear_candidate_comparison": linear_comparison,
            "prediction_information_coefficient": prediction_ic,
            "residual_diagnostics": residual_diagnostics,
            "coefficient_explainability": coefficients,
        }

    promotion_policy = _promotion_policy(
        horizon_count=len(horizons),
        linear_wins=linear_wins,
        residual_warnings=residual_warnings,
    )
    return {
        "chapter": 7,
        "name": "Linear Models - From Risk Factors to Return Forecasts",
        "status": "pass" if promotion_policy["candidate_for_promotion"] else "warn",
        "decision_policy": {
            "influences_final_action": False,
            "mode": "diagnostic_only",
            "reason": "Chapter 7 adds linear baselines and diagnostics but does not discard existing models or override model selection.",
        },
        "model_registry_policy": {
            "ridge": "kept_distinct_l2_shrinkage",
            "lasso": "kept_distinct_l1_feature_selection",
            "elastic_net": "kept_distinct_mixed_l1_l2_shrinkage",
            "duplication_check": "No existing fixed Ridge or Lasso candidates were present; ElasticNet is not a duplicate.",
        },
        "horizons": horizon_reports,
        "promotion_policy": promotion_policy,
        "technical_method_card": chapter_7_linear_models_method_card(),
    }


def chapter_7_linear_models_method_card() -> dict[str, Any]:
    return {
        "name": "jansen_ml4t_chapter_7_linear_models",
        "version": "chapter_7_linear_models_v1",
        "source": "Machine Learning for Algorithmic Trading, 2nd ed., Chapter 7",
        "purpose": "Treat linear models as explicit baselines, evaluate regularized linear candidates, and diagnose residual and prediction-signal quality.",
        "decision_policy": "diagnostic_only",
        "implemented_components": [
            "ridge_lasso_elastic_net_candidate_comparison",
            "selected_prediction_information_coefficient",
            "residual_fat_tail_and_autocorrelation_diagnostics",
            "linear_coefficient_explainability",
            "linear_model_registry_duplication_policy",
        ],
        "not_implemented": [
            "OLS_p_values_as_trade_rules",
            "automatic_replacement_of_tree_or_boosting_models",
            "Quandl_Wiki_cross_section_universe",
        ],
    }


def _forecast_for_horizon(forecasts: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    for forecast in forecasts:
        if int(forecast.get("horizon_days", -1)) == int(horizon):
            return forecast
    return {}


def _linear_candidate_comparison(
    *,
    candidates: list[dict[str, Any]],
    selected_model: str,
    selection_metric: str,
) -> dict[str, Any]:
    rows = []
    metric = selection_metric.lower()
    for row in candidates:
        name = str(row.get("model_name", ""))
        family = str(row.get("model_family", ""))
        if not (name.startswith(LINEAR_PREFIXES) or family == "linear_regularized"):
            continue
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
        rows.append(
            {
                "model_name": name,
                "model_family": family,
                "selection_metric": metric,
                "metric_value": _finite(_number(metrics.get(metric))),
                "mae": _finite(_number(metrics.get("mae"))),
                "rmse": _finite(_number(metrics.get("rmse"))),
                "directional_accuracy": _finite(_number(metrics.get("directional_accuracy"))),
                "deflated_sharpe_ratio": _finite(_number(metrics.get("deflated_sharpe_ratio"))),
            }
        )
    if not rows:
        return {
            "status": "no_linear_candidates",
            "selected_model_is_linear": False,
            "best_linear_model": None,
            "linear_candidates": [],
        }
    reverse = metric in HIGHER_IS_BETTER
    best = sorted(rows, key=lambda item: item["metric_value"], reverse=reverse)[0]
    selected_linear = selected_model.startswith(LINEAR_PREFIXES)
    return {
        "status": "available",
        "selected_model_is_linear": bool(selected_linear),
        "best_linear_model": best,
        "linear_candidate_count": int(len(rows)),
        "linear_candidates": sorted(rows, key=lambda item: item["metric_value"], reverse=reverse),
        "reason": (
            "The selected model is a regularized linear model."
            if selected_linear
            else "Linear models remain explicit baselines but did not win this horizon."
        ),
    }


def _prediction_ic(records: list[dict[str, Any]]) -> dict[str, Any]:
    frame = _validation_frame(records)
    if frame.empty:
        return {"status": "no_validation_records", "spearman_ic": 0.0, "pearson_ic": 0.0}
    spearman = frame["predicted"].corr(frame["actual"], method="spearman")
    pearson = frame["predicted"].corr(frame["actual"], method="pearson")
    rolling = (
        frame["predicted"]
        .rolling(21)
        .corr(frame["actual"], pairwise=False)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    return {
        "status": "available",
        "rows": int(len(frame)),
        "spearman_ic": _finite(_number(spearman)),
        "pearson_ic": _finite(_number(pearson)),
        "rolling_21_mean_pearson_ic": _finite(float(rolling.mean())) if not rolling.empty else 0.0,
        "rolling_21_positive_rate": _finite(float((rolling > 0).mean())) if not rolling.empty else 0.0,
        "interpretation": "Positive IC means higher predicted returns tended to align with higher realized returns.",
    }


def _residual_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    frame = _validation_frame(records)
    if frame.empty:
        return {"status": "no_validation_records", "rows": 0}
    residuals = (frame["actual"] - frame["predicted"]).to_numpy(dtype=float)
    jb = _jarque_bera(residuals)
    dw = _durbin_watson(residuals)
    fat_tailed = bool(jb["p_value"] < 0.05 and jb["kurtosis"] > 4.0)
    autocorrelated = bool(dw < 1.5 or dw > 2.5)
    if fat_tailed and autocorrelated:
        status = "unstable"
    elif fat_tailed:
        status = "fat_tailed"
    elif autocorrelated:
        status = "autocorrelated"
    else:
        status = "acceptable"
    return {
        "status": status,
        "rows": int(len(residuals)),
        "mean_residual": _finite(float(np.mean(residuals))),
        "residual_std": _finite(float(np.std(residuals, ddof=1))) if len(residuals) > 1 else 0.0,
        "durbin_watson": _finite(dw),
        "jarque_bera": jb,
        "reason": (
            "Residuals show fat-tail and autocorrelation warnings."
            if status == "unstable"
            else "Residual diagnostics are usable as warning evidence, not an automatic trading rule."
        ),
    }


def _linear_coefficients(model_card: dict[str, Any]) -> dict[str, Any]:
    diagnostics = model_card.get("model_diagnostics", {}) if isinstance(model_card, dict) else {}
    coefficients = diagnostics.get("linear_coefficients", {}) if isinstance(diagnostics, dict) else {}
    if not coefficients:
        return {
            "status": "not_available",
            "reason": "Selected model is not a fitted linear estimator with coefficients.",
        }
    return {
        "status": "available",
        **coefficients,
        "interpretation": "Coefficients are standardized pipeline coefficients and should be treated as model explainability, not causal truth.",
    }


def _promotion_policy(*, horizon_count: int, linear_wins: int, residual_warnings: int) -> dict[str, Any]:
    candidate = horizon_count > 0 and linear_wins == horizon_count and residual_warnings == 0
    return {
        "status": "candidate_for_linear_gate_research" if candidate else "diagnostic_only",
        "candidate_for_promotion": bool(candidate),
        "linear_selected_horizon_count": int(linear_wins),
        "tested_horizon_count": int(horizon_count),
        "residual_warning_horizon_count": int(residual_warnings),
        "recommended_action": (
            "Research a linear-model ensemble gate before enabling any action changes."
            if candidate
            else "Keep Chapter 7 as diagnostics; do not discard non-linear or statistical candidates."
        ),
    }


def _validation_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty or "actual_log_return" not in frame or "predicted_log_return" not in frame:
        return pd.DataFrame()
    if "validation_date" in frame:
        frame["validation_date"] = pd.to_datetime(frame["validation_date"], errors="coerce")
        frame = frame.sort_values("validation_date")
    actual = pd.to_numeric(frame["actual_log_return"], errors="coerce")
    predicted = pd.to_numeric(frame["predicted_log_return"], errors="coerce")
    return pd.DataFrame({"actual": actual, "predicted": predicted}).replace([np.inf, -np.inf], np.nan).dropna()


def _jarque_bera(values: np.ndarray) -> dict[str, float]:
    clean = values[np.isfinite(values)]
    if len(clean) < 8:
        return {"statistic": 0.0, "p_value": 1.0, "skew": 0.0, "kurtosis": 0.0}
    centered = clean - float(np.mean(clean))
    std = float(np.std(centered, ddof=0))
    if std <= 1e-12:
        return {"statistic": 0.0, "p_value": 1.0, "skew": 0.0, "kurtosis": 0.0}
    skew = float(np.mean((centered / std) ** 3))
    kurtosis = float(np.mean((centered / std) ** 4))
    statistic = len(clean) / 6.0 * (skew**2 + 0.25 * (kurtosis - 3.0) ** 2)
    p_value = _chi2_2_survival(statistic)
    return {
        "statistic": _finite(statistic),
        "p_value": _finite(p_value),
        "skew": _finite(skew),
        "kurtosis": _finite(kurtosis),
    }


def _durbin_watson(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if len(clean) < 2:
        return 2.0
    denominator = float(np.sum(clean**2))
    if denominator <= 1e-12:
        return 2.0
    return float(np.sum(np.diff(clean) ** 2) / denominator)


def _chi2_2_survival(value: float) -> float:
    return float(np.exp(-max(value, 0.0) / 2.0))


def _number(value: Any) -> float:
    try:
        number = float(value if value is not None else 0.0)
    except Exception:
        return 0.0
    return number if np.isfinite(number) else 0.0


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0
