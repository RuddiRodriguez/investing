from __future__ import annotations

from typing import Any


CNN_MODEL_NAMES = {"temporal_cnn_research"}


def analyze_chapter_18_cnn(
    *,
    forecasts: list[dict[str, Any]],
    candidate_results: dict[str, list[dict[str, Any]]],
    model_cards: dict[str, Any],
    horizons: tuple[int, ...],
    deep_learning_profile: str,
    selection_metric: str,
) -> dict[str, Any]:
    """Audit Jansen Chapter 18 CNN candidates without creating a trading gate."""

    horizon_views = {}
    selected_count = 0
    validated_count = 0
    for horizon in horizons:
        key = str(horizon)
        forecast = next((row for row in forecasts if int(row.get("horizon_days", -1)) == int(horizon)), {})
        view = _horizon_view(
            forecast=forecast,
            candidates=candidate_results.get(key, []),
            model_card=model_cards.get(key, {}),
            selection_metric=selection_metric,
        )
        horizon_views[key] = view
        selected_count += int(view["selected_cnn"])
        validated_count += int(view["cnn_candidate_count"])

    enabled = deep_learning_profile == "research"
    promoted = bool(enabled and selected_count == len(horizons) and validated_count >= len(horizons))
    return {
        "status": "enabled" if enabled else "disabled",
        "chapter": "Jansen ML4T Chapter 18 - CNNs for Financial Time Series and Satellite Images",
        "profile": deep_learning_profile,
        "decision_policy": {
            "influences_final_action": False,
            "influences_model_selection": True,
            "influences_model_fitting": True,
            "mode": "research_candidate_only",
            "reason": "CNNs are optional forecast candidates only; they can affect the selected forecast only by beating baseline candidates in walk-forward validation.",
        },
        "model_policy": {
            "default_in_watch_agents": False,
            "cnn_trading_gate": False,
            "satellite_image_pipeline": False,
            "enabled_profile": "research",
            "baseline_requirement": "A CNN is used only if its adjusted validation metric beats the non-CNN candidates for that horizon.",
        },
        "validated_cnn_candidate_count": int(validated_count),
        "selected_cnn_horizon_count": int(selected_count),
        "promotion_policy": {
            "status": "candidate_for_research_profile_use" if promoted else "research_only",
            "candidate_for_promotion": promoted,
            "reason": (
                "CNN won model selection across all requested horizons."
                if promoted
                else "Keep CNN research-only until it beats baseline validation stably across requested horizons."
            ),
        },
        "horizons": horizon_views,
        "technical_method_card": {
            "name": "jansen_ml4t_chapter_18_cnn_time_series",
            "purpose": "Register optional temporal CNN candidates for financial time-series tensors and keep image/satellite CNNs out until real image data exists.",
            "inputs": ["engineered_feature_windows", "candidate_validation_results", "selected_forecasts"],
            "outputs": ["cnn_candidate_audit", "selection_consequence_policy"],
            "limitations": [
                "No CNN trading gate.",
                "No CNN default in watch agents.",
                "No satellite image model without real image data.",
                "CNN forecasts must win walk-forward validation to be selected.",
            ],
        },
    }


def _horizon_view(
    *,
    forecast: dict[str, Any],
    candidates: list[dict[str, Any]],
    model_card: dict[str, Any],
    selection_metric: str,
) -> dict[str, Any]:
    cnn_rows = [row for row in candidates if row.get("model_name") in CNN_MODEL_NAMES]
    selected_model = str(forecast.get("selected_model", ""))
    ranked = [_candidate_summary(row, selection_metric) for row in cnn_rows]
    ranked = sorted(ranked, key=lambda row: row["selection_value"])
    return {
        "selected_model": selected_model,
        "selected_cnn": selected_model in CNN_MODEL_NAMES,
        "cnn_candidate_count": int(len(cnn_rows)),
        "best_cnn_candidate": ranked[0] if ranked else None,
        "model_card_cnn_diagnostics": _cnn_model_card_entries(model_card),
    }


def _candidate_summary(row: dict[str, Any], selection_metric: str) -> dict[str, Any]:
    metrics = row.get("metrics", {})
    metric = selection_metric.lower()
    adjusted_key = f"chapter_9_adjusted_{metric}"
    selection_key = adjusted_key if adjusted_key in metrics else metric
    return {
        "model_name": row.get("model_name"),
        "selection_metric": selection_key,
        "selection_value": _finite(metrics.get(selection_key, metrics.get("mae", float("inf"))), default=float("inf")),
        "mae": _finite(metrics.get("mae")),
        "holdout_mae": _finite(metrics.get("holdout_mae")),
        "directional_accuracy": _finite(metrics.get("directional_accuracy")),
        "chapter_17_selection_penalty": _finite(metrics.get("chapter_17_deep_learning_selection_penalty")),
        "chapter_17_note_count": _finite(metrics.get("chapter_17_deep_learning_note_count")),
    }


def _cnn_model_card_entries(model_card: dict[str, Any]) -> dict[str, Any]:
    diagnostics = model_card.get("model_diagnostics", {})
    return diagnostics.get("chapter_18_cnn", {}) if isinstance(diagnostics, dict) else {}


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number and abs(number) != float("inf") else default
