from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def analyze_chapter_1_ml4t_workflow(
    *,
    features: pd.DataFrame,
    supervised: pd.DataFrame,
    forecasts: list[dict[str, Any]],
    factor_evaluation: dict[str, list[dict[str, Any]]],
    candidate_results: dict[str, list[dict[str, object]]],
    selected_validation_predictions: dict[str, list[dict[str, Any]]],
    backtests: dict[str, dict[str, Any]],
    data_manifest: dict[str, Any],
    data_quality_report: dict[str, Any],
    final_action: str,
) -> dict[str, Any]:
    """Summarize the end-to-end ML4T workflow from idea to governed action."""

    horizons = [str(item.get("horizon_days")) for item in forecasts]
    workflow_stages = {
        "data": _stage(
            bool(data_manifest.get("row_count") or data_quality_report.get("rows")),
            f"{int(data_manifest.get('row_count') or data_quality_report.get('rows') or 0)} model-ready rows.",
        ),
        "features": _stage(
            len(features.columns) > 0,
            f"{len(features.columns)} point-in-time feature columns.",
        ),
        "targets": _stage(
            any(column.startswith("target_log_return_") for column in supervised.columns),
            f"{sum(column.startswith('target_log_return_') for column in supervised.columns)} forward-return target columns.",
        ),
        "model_selection": _stage(
            bool(candidate_results) and all(candidate_results.get(horizon) for horizon in horizons),
            f"Candidate validation is available for {sum(bool(candidate_results.get(horizon)) for horizon in horizons)} horizons.",
        ),
        "signals": _stage(
            bool(forecasts),
            f"{len(forecasts)} horizon forecasts converted to directional signals.",
        ),
        "backtest": _stage(
            bool(backtests) and all(backtests.get(horizon) for horizon in horizons),
            f"Signal backtests are available for {sum(bool(backtests.get(horizon)) for horizon in horizons)} horizons.",
        ),
        "risk_action": _stage(
            final_action in {"Buy", "Hold", "Sell"},
            f"Final governed action is {final_action}.",
        ),
    }
    complete = all(item["status"] == "pass" for item in workflow_stages.values())

    return {
        "chapter": 1,
        "name": "Machine Learning for Trading - From Idea to Execution",
        "status": "pass" if complete else "warn",
        "decision_policy": {
            "influences_final_action": False,
            "mode": "diagnostic_only",
            "reason": "Chapter 1 diagnostics audit the ML4T process but do not override trading decisions.",
        },
        "workflow_stages": workflow_stages,
        "workflow_complete": complete,
        "forecast_skill": _forecast_skill_summary(forecasts, factor_evaluation),
        "strategy_breadth": _strategy_breadth_summary(
            forecasts=forecasts,
            selected_validation_predictions=selected_validation_predictions,
            data_manifest=data_manifest,
        ),
        "hypothesis": _strategy_hypothesis(features, factor_evaluation, forecasts),
        "forecast_to_action_separation": {
            "forecast_layer": "Expected returns, prices, intervals, model validation, and factor quality.",
            "signal_layer": "Directional forecast confidence and trade-quality diagnostics.",
            "decision_layer": "Technical filters, risk gates, tactical plan, and optional LLM review.",
            "final_action": final_action,
        },
        "technical_method_card": chapter_1_ml4t_workflow_method_card(),
    }


def chapter_1_ml4t_workflow_method_card() -> dict[str, Any]:
    return {
        "name": "jansen_ml4t_chapter_1_workflow_audit",
        "version": "chapter_1_ml4t_workflow_v1",
        "source": "Machine Learning for Algorithmic Trading, 2nd ed., Chapter 1",
        "purpose": "Audit whether a forecast run covers the full ML4T path from data to model, signal, backtest, risk, and action.",
        "decision_policy": "diagnostic_only",
    }


def _stage(passed: bool, detail: str) -> dict[str, str]:
    return {"status": "pass" if passed else "warn", "detail": detail}


def _forecast_skill_summary(
    forecasts: list[dict[str, Any]],
    factor_evaluation: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    horizons = []
    for forecast in forecasts:
        horizon = str(forecast.get("horizon_days"))
        metrics = forecast.get("validation_metrics", {}) or {}
        factor_rows = factor_evaluation.get(horizon, [])
        top_rank_ic = max((abs(float(row.get("rank_ic", 0.0))) for row in factor_rows), default=0.0)
        horizons.append(
            {
                "horizon_days": int(forecast.get("horizon_days", 0)),
                "selected_model": forecast.get("selected_model"),
                "directional_accuracy": _finite(metrics.get("directional_accuracy", 0.0)),
                "sharpe_ratio": _finite(metrics.get("sharpe_ratio", 0.0)),
                "deflated_sharpe_ratio": _finite(metrics.get("deflated_sharpe_ratio", 0.0)),
                "mae": _finite(metrics.get("mae", 0.0)),
                "top_abs_rank_ic": _finite(top_rank_ic),
                "validation_edge": _edge_label(metrics, top_rank_ic),
            }
        )
    return {
        "principle": "Forecast quality is the core input to alpha; rank IC, directional accuracy, and validation Sharpe are reported together.",
        "horizons": horizons,
    }


def _strategy_breadth_summary(
    *,
    forecasts: list[dict[str, Any]],
    selected_validation_predictions: dict[str, list[dict[str, Any]]],
    data_manifest: dict[str, Any],
) -> dict[str, Any]:
    universe = data_manifest.get("universe", {}) or {}
    universe_count = int(universe.get("ticker_count") or universe.get("count") or 1)
    validation_bets = {
        horizon: int(len(records)) for horizon, records in selected_validation_predictions.items()
    }
    horizon_count = len(forecasts)
    independent_breadth_proxy = max(universe_count, 1) * max(horizon_count, 1)
    warnings = []
    if universe_count <= 1:
        warnings.append("Single-ticker runs have limited cross-sectional breadth.")
    if horizon_count < 2:
        warnings.append("Only one forecast horizon reduces independent signal breadth.")
    return {
        "principle": "Expected active value depends on forecast skill and the breadth of independent bets.",
        "universe_count": universe_count,
        "horizon_count": horizon_count,
        "independent_breadth_proxy": int(independent_breadth_proxy),
        "validation_bets_by_horizon": validation_bets,
        "warnings": warnings,
    }


def _strategy_hypothesis(
    features: pd.DataFrame,
    factor_evaluation: dict[str, list[dict[str, Any]]],
    forecasts: list[dict[str, Any]],
) -> dict[str, Any]:
    top_features = []
    for horizon, rows in factor_evaluation.items():
        for row in rows[:5]:
            top_features.append(str(row.get("feature", "")))
    families = sorted({_feature_family(column) for column in top_features if column})
    directions = sorted({str(item.get("expected_direction")) for item in forecasts if item.get("expected_direction")})
    return {
        "primary_feature_families": families,
        "available_feature_count": int(len(features.columns)),
        "forecast_directions": directions,
        "hypothesis_label": "+".join(families[:4]) if families else "unclassified",
        "description": "Run hypothesis is inferred from the strongest evaluated feature families and current forecast directions.",
    }


def _feature_family(column: str) -> str:
    name = column.lower()
    if "momentum" in name or "return_" in name or "relative" in name:
        return "momentum_relative_strength"
    if "volatility" in name or "atr" in name or "range" in name:
        return "volatility_regime"
    if "volume" in name or "liquidity" in name or "dollar" in name:
        return "liquidity_volume"
    if "support" in name or "resistance" in name or "breakout" in name or "trend" in name:
        return "technical_structure"
    if name.startswith("exo_") or name.startswith("macro_"):
        return "external_context"
    return "statistical_price_behavior"


def _edge_label(metrics: dict[str, Any], top_rank_ic: float) -> str:
    directional_accuracy = float(metrics.get("directional_accuracy", 0.0) or 0.0)
    deflated_sharpe = float(metrics.get("deflated_sharpe_ratio", 0.0) or 0.0)
    if directional_accuracy >= 0.55 and deflated_sharpe > 0 and abs(top_rank_ic) >= 0.03:
        return "validated_edge"
    if directional_accuracy >= 0.50 or abs(top_rank_ic) >= 0.02:
        return "weak_or_unstable_edge"
    return "no_clear_edge"


def _finite(value: Any) -> float:
    numeric = float(value or 0.0)
    return numeric if np.isfinite(numeric) else 0.0
