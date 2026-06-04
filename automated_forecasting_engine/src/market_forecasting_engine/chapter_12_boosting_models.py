from __future__ import annotations

from typing import Any

import numpy as np


BOOSTING_FAMILIES = {"boosting_model"}
BOOSTING_NAME_TOKENS = ("boosting", "lightgbm", "xgboost")
HIGHER_IS_BETTER = {"directional_accuracy", "sharpe_ratio", "hit_rate", "profit_factor"}


def analyze_chapter_12_boosting_models(
    *,
    forecasts: list[dict[str, Any]],
    candidate_results: dict[str, list[dict[str, Any]]],
    model_cards: dict[str, Any],
    horizons: tuple[int, ...],
    selection_metric: str,
) -> dict[str, Any]:
    """Evaluate boosted models as active forecast candidates with overfit controls."""

    horizon_reports = {}
    boosting_selected_count = 0
    penalized_count = 0
    for horizon in horizons:
        key = str(horizon)
        forecast = _forecast_for_horizon(forecasts, horizon)
        comparison = _boosting_candidate_comparison(
            candidates=candidate_results.get(key, []),
            selected_model=str(forecast.get("selected_model", "")),
            selected_family=str(forecast.get("selected_model_family", "")),
            selection_metric=selection_metric,
        )
        boosting_selected_count += int(comparison.get("selected_model_is_boosting", False))
        penalized_count += int(comparison.get("penalized_boosting_candidate_count", 0) or 0)
        horizon_reports[key] = {
            "horizon_days": int(horizon),
            "selected_model": forecast.get("selected_model"),
            "selected_model_family": forecast.get("selected_model_family"),
            "boosting_candidate_comparison": comparison,
            "selected_boosting_diagnostics": _selected_boosting_diagnostics(model_cards.get(key, {})),
        }

    return {
        "chapter": 12,
        "name": "Boosting Your Trading Strategy",
        "status": "active" if boosting_selected_count else "available",
        "decision_policy": {
            "influences_final_action": False,
            "influences_model_selection": True,
            "mode": "active_boosted_candidates_with_selection_adjustment",
            "reason": "Boosted models compete in adjusted walk-forward model selection; they do not directly override Buy/Hold/Sell.",
        },
        "model_registry_policy": {
            "gradient_boosting": "kept_as_conservative_sklearn_boosting_candidate",
            "hist_gradient_boosting": "added_as_regularized_sklearn_boosting_candidate",
            "lightgbm": "kept_when_enabled_and_dependency_available",
            "xgboost": "required_dependency_and_added_as_conservative_boosting_candidate",
            "selection_adjustment": "enabled_for_holdout_degradation_prediction_volatility_and_rolling_ic_instability",
        },
        "horizons": horizon_reports,
        "summary": {
            "boosting_selected_horizon_count": int(boosting_selected_count),
            "tested_horizon_count": int(len(horizons)),
            "penalized_boosting_candidate_count": int(penalized_count),
        },
        "technical_method_card": chapter_12_boosting_models_method_card(),
    }


def chapter_12_boosting_models_method_card() -> dict[str, Any]:
    return {
        "name": "jansen_ml4t_chapter_12_boosting_models",
        "version": "chapter_12_boosting_models_v1",
        "source": "Machine Learning for Algorithmic Trading, 2nd ed., Chapter 12",
        "purpose": "Use conservative boosted-tree models as active forecast candidates and penalize unstable boosting evidence before selection.",
        "decision_policy": "active_model_selection_no_trade_gate_override",
        "implemented_components": [
            "gradient_boosting_candidate",
            "hist_gradient_boosting_candidate",
            "lightgbm_candidate_when_enabled",
            "xgboost_candidate",
            "boosting_holdout_degradation_penalty",
            "boosting_prediction_volatility_penalty",
            "boosting_rolling_ic_stability_penalty",
        ],
        "not_implemented": [
            "boosted_classifier_as_direct_buy_sell_gate",
            "aggressive_deep_boosting_defaults",
            "shuffled_cross_validation",
        ],
    }


def _boosting_candidate_comparison(
    *,
    candidates: list[dict[str, Any]],
    selected_model: str,
    selected_family: str,
    selection_metric: str,
) -> dict[str, Any]:
    metric = selection_metric.lower()
    adjusted_metric = f"chapter_9_adjusted_{metric}"
    rows = []
    for row in candidates:
        name = str(row.get("model_name", ""))
        family = str(row.get("model_family", ""))
        if not _is_boosting_candidate(name=name, family=family):
            continue
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
        rows.append(
            {
                "model_name": name,
                "model_family": family,
                "selection_metric": metric,
                "metric_value": _finite(_number(metrics.get(metric))),
                "adjusted_metric_value": _finite(_number(metrics.get(adjusted_metric, metrics.get(metric)))),
                "mae": _finite(_number(metrics.get("mae"))),
                "holdout_mae": _finite(_number(metrics.get("holdout_mae"))),
                "directional_accuracy": _finite(_number(metrics.get("directional_accuracy"))),
                "deflated_sharpe_ratio": _finite(_number(metrics.get("deflated_sharpe_ratio"))),
                "chapter_12_boosting_selection_penalty": _finite(_number(metrics.get("chapter_12_boosting_selection_penalty"))),
                "prediction_volatility_ratio": _finite(_number(metrics.get("chapter_12_boosting_prediction_volatility_ratio"))),
                "rolling_ic_positive_rate": _finite(_number(metrics.get("chapter_12_boosting_rolling_ic_positive_rate"))),
            }
        )
    if not rows:
        return {"status": "no_boosting_candidates", "selected_model_is_boosting": False, "boosting_candidates": []}

    reverse = metric in HIGHER_IS_BETTER
    best = sorted(rows, key=lambda item: item["adjusted_metric_value"], reverse=reverse)[0]
    selected_is_boosting = _is_boosting_candidate(name=selected_model, family=selected_family)
    return {
        "status": "available",
        "selected_model_is_boosting": bool(selected_is_boosting),
        "best_boosting_model": best,
        "boosting_candidate_count": int(len(rows)),
        "penalized_boosting_candidate_count": int(sum(float(row["chapter_12_boosting_selection_penalty"] or 0.0) > 0.0 for row in rows)),
        "boosting_candidates": sorted(rows, key=lambda item: item["adjusted_metric_value"], reverse=reverse),
        "model_selection_consequence": (
            "A boosted model won the adjusted walk-forward selection metric for this horizon."
            if selected_is_boosting
            else "Boosted models contributed to candidate competition but did not win this horizon."
        ),
    }


def _selected_boosting_diagnostics(model_card: dict[str, Any]) -> dict[str, Any]:
    selected = str(model_card.get("selected_model", "") or "")
    family = str(model_card.get("selected_model_family", "") or "")
    if not _is_boosting_candidate(name=selected, family=family):
        return {"status": "not_available", "reason": "Selected model is not a boosted model."}
    diagnostics = model_card.get("model_diagnostics", {}) if isinstance(model_card, dict) else {}
    return {"status": "available", **diagnostics}


def _forecast_for_horizon(forecasts: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    for forecast in forecasts:
        if int(forecast.get("horizon_days", -1)) == int(horizon):
            return forecast
    return {}


def _is_boosting_candidate(*, name: str, family: str) -> bool:
    lowered = name.lower()
    return family in BOOSTING_FAMILIES or any(token in lowered for token in BOOSTING_NAME_TOKENS)


def _number(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0
