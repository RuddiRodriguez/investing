from __future__ import annotations

from typing import Any

import numpy as np


TREE_FAMILIES = {"tree_model", "tree_ensemble"}
TREE_NAME_TOKENS = ("tree", "forest")
LOWER_IS_BETTER = {"rmse", "mae", "mape", "smape", "aic", "bic", "composite"}
HIGHER_IS_BETTER = {"directional_accuracy", "sharpe_ratio", "hit_rate", "profit_factor"}


def analyze_chapter_11_tree_models(
    *,
    forecasts: list[dict[str, Any]],
    candidate_results: dict[str, list[dict[str, Any]]],
    model_cards: dict[str, Any],
    horizons: tuple[int, ...],
    selection_metric: str,
) -> dict[str, Any]:
    """Evaluate tree-based model candidates as active model-selection contributors."""

    horizon_reports = {}
    tree_selected_count = 0
    penalized_count = 0
    for horizon in horizons:
        key = str(horizon)
        forecast = _forecast_for_horizon(forecasts, horizon)
        candidates = candidate_results.get(key, [])
        comparison = _tree_candidate_comparison(
            candidates=candidates,
            selected_model=str(forecast.get("selected_model", "")),
            selected_family=str(forecast.get("selected_model_family", "")),
            selection_metric=selection_metric,
        )
        selected_is_tree = bool(comparison.get("selected_model_is_tree", False))
        tree_selected_count += int(selected_is_tree)
        penalized_count += int(comparison.get("penalized_tree_candidate_count", 0) or 0)
        horizon_reports[key] = {
            "horizon_days": int(horizon),
            "selected_model": forecast.get("selected_model"),
            "selected_model_family": forecast.get("selected_model_family"),
            "tree_candidate_comparison": comparison,
            "selected_model_tree_diagnostics": _selected_tree_diagnostics(model_cards.get(key, {})),
        }

    return {
        "chapter": 11,
        "name": "Decision Trees and Random Forests",
        "status": "active" if tree_selected_count else "available",
        "decision_policy": {
            "influences_final_action": False,
            "influences_model_selection": True,
            "mode": "active_model_candidates_with_selection_adjustment",
            "reason": "Tree models are validated as forecast candidates and can win model selection; they do not directly override the trading decision gate.",
        },
        "model_registry_policy": {
            "decision_tree": "added_as_shallow_interpretable_tree_candidate",
            "random_forest": "kept_and_strengthened_as_tree_ensemble_candidate",
            "extra_trees": "added_as_randomized_tree_ensemble_candidate",
            "gradient_boosting": "kept_as_boosted_tree_candidate",
            "lightgbm": "kept_optional_when_dependency_available",
            "selection_adjustment": "enabled_for_tree_overfit_or_instability",
        },
        "horizons": horizon_reports,
        "summary": {
            "tree_selected_horizon_count": int(tree_selected_count),
            "tested_horizon_count": int(len(horizons)),
            "penalized_tree_candidate_count": int(penalized_count),
        },
        "technical_method_card": chapter_11_tree_models_method_card(),
    }


def chapter_11_tree_models_method_card() -> dict[str, Any]:
    return {
        "name": "jansen_ml4t_chapter_11_tree_models",
        "version": "chapter_11_tree_models_v1",
        "source": "Machine Learning for Algorithmic Trading, 2nd ed., Chapter 11",
        "purpose": "Use decision trees and tree ensembles as active forecast candidates, with walk-forward validation and overfit diagnostics.",
        "decision_policy": "active_model_selection_no_trade_gate_override",
        "implemented_components": [
            "decision_tree_candidate",
            "random_forest_candidate",
            "extra_trees_candidate",
            "gradient_boosting_candidate",
            "optional_lightgbm_candidate",
            "tree_feature_importance_diagnostics",
            "tree_holdout_degradation_selection_penalty",
        ],
        "not_implemented": [
            "classification_trees_as_direct_buy_sell_gate",
            "shuffled_cross_validation",
            "unvalidated_tree_rule_trading",
        ],
    }


def _tree_candidate_comparison(
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
        if not _is_tree_candidate(name=name, family=family):
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
                "chapter_11_tree_selection_penalty": _finite(_number(metrics.get("chapter_11_tree_selection_penalty"))),
            }
        )
    if not rows:
        return {
            "status": "no_tree_candidates",
            "selected_model_is_tree": False,
            "tree_candidates": [],
        }
    reverse = metric in HIGHER_IS_BETTER
    sort_key = "adjusted_metric_value"
    best = sorted(rows, key=lambda item: item[sort_key], reverse=reverse)[0]
    selected_is_tree = _is_tree_candidate(name=selected_model, family=selected_family)
    return {
        "status": "available",
        "selected_model_is_tree": bool(selected_is_tree),
        "best_tree_model": best,
        "tree_candidate_count": int(len(rows)),
        "penalized_tree_candidate_count": int(sum(float(row["chapter_11_tree_selection_penalty"] or 0.0) > 0.0 for row in rows)),
        "tree_candidates": sorted(rows, key=lambda item: item[sort_key], reverse=reverse),
        "model_selection_consequence": (
            "A tree model won the adjusted walk-forward selection metric for this horizon."
            if selected_is_tree
            else "Tree models contributed to candidate competition but did not win this horizon."
        ),
    }


def _selected_tree_diagnostics(model_card: dict[str, Any]) -> dict[str, Any]:
    diagnostics = model_card.get("model_diagnostics", {}) if isinstance(model_card, dict) else {}
    tree = diagnostics.get("tree_model", {}) if isinstance(diagnostics, dict) else {}
    if not tree:
        return {
            "status": "not_available",
            "reason": "Selected model is not a fitted tree model with feature importances.",
        }
    return {
        "status": "available",
        **tree,
    }


def _forecast_for_horizon(forecasts: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    for forecast in forecasts:
        if int(forecast.get("horizon_days", -1)) == int(horizon):
            return forecast
    return {}


def _is_tree_candidate(*, name: str, family: str) -> bool:
    lowered = name.lower()
    return family in TREE_FAMILIES or any(token in lowered for token in TREE_NAME_TOKENS)


def _number(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0
