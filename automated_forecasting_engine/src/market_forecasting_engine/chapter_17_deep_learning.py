from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEEP_FAMILIES = {"deep_learning"}


def analyze_chapter_17_deep_learning(
    *,
    features: pd.DataFrame,
    forecasts: list[dict[str, Any]],
    candidate_results: dict[str, list[dict[str, Any]]],
    model_cards: dict[str, Any],
    horizons: tuple[int, ...],
    deep_learning_profile: str,
    include_lstm: bool,
    selection_metric: str,
) -> dict[str, Any]:
    """Audit Jansen Chapter 17 deep-learning candidates and their selector consequences."""

    horizon_views = {}
    selected_deep_count = 0
    validated_deep_count = 0
    promotable_horizons = 0
    for horizon in horizons:
        key = str(horizon)
        forecast = next((row for row in forecasts if int(row.get("horizon_days", -1)) == int(horizon)), {})
        candidates = candidate_results.get(key, [])
        view = _horizon_view(
            forecast=forecast,
            candidates=candidates,
            model_card=model_cards.get(key, {}),
            selection_metric=selection_metric,
        )
        horizon_views[key] = view
        selected_deep_count += int(bool(view["selected_deep_learning"]))
        validated_deep_count += int(view["deep_learning_candidate_count"])
        promotable_horizons += int(view["promotion_candidate"])

    readiness = _readiness(features)
    enabled = deep_learning_profile != "off" or include_lstm
    promoted = bool(enabled and selected_deep_count == len(horizons) and promotable_horizons == len(horizons))
    return {
        "status": "enabled" if enabled else "disabled",
        "chapter": "Jansen ML4T Chapter 17 - Deep Learning for Trading",
        "profile": deep_learning_profile,
        "legacy_include_lstm": bool(include_lstm),
        "decision_policy": {
            "influences_final_action": False,
            "influences_model_selection": True,
            "influences_model_fitting": True,
            "mode": "optional_candidate_family_with_selection_penalties",
            "reason": "Deep-learning candidates may be selected only through walk-forward validation after explicit Chapter 17 penalties; they never override Buy/Hold/Sell directly.",
        },
        "model_policy": {
            "default_in_live_agents": False,
            "gpu_required": False,
            "deep_learning_buy_sell_gate": False,
            "transformer_or_cnn_stack": False,
            "available_profiles": ["off", "fast", "research"],
            "fast_profile": "bounded tabular MLP only",
            "research_profile": "bounded tabular MLP plus optional slower LSTM path",
        },
        "readiness": readiness,
        "validated_deep_learning_candidate_count": int(validated_deep_count),
        "selected_deep_learning_horizon_count": int(selected_deep_count),
        "promotion_policy": {
            "status": "candidate_for_model_selector_use" if promoted else "research_only",
            "candidate_for_promotion": promoted,
            "reason": (
                "Deep-learning candidates won model selection with stable validation across all requested horizons."
                if promoted
                else "Keep deep learning optional until it wins validation stably across requested horizons and runtime remains bounded."
            ),
        },
        "horizons": horizon_views,
        "technical_method_card": {
            "name": "jansen_ml4t_chapter_17_deep_learning",
            "purpose": "Register optional neural-network candidates, audit readiness, and make deep-learning diagnostics affect model selection without creating a trading gate.",
            "inputs": ["feature_matrix", "candidate_validation_results", "selected_forecasts", "model_cards"],
            "outputs": ["deep_learning_candidate_audit", "readiness_diagnostics", "selection_consequence_policy"],
            "limitations": [
                "No GPU is required or assumed.",
                "No deep-learning model directly changes Buy/Hold/Sell.",
                "No transformer/CNN stack is promoted at this stage.",
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
    deep_rows = [row for row in candidates if row.get("model_family") in DEEP_FAMILIES]
    selected_model = str(forecast.get("selected_model", ""))
    selected_family = str(forecast.get("selected_model_family", ""))
    selected_deep = selected_family in DEEP_FAMILIES
    ranked = [_candidate_summary(row, selection_metric) for row in deep_rows]
    ranked = sorted(ranked, key=lambda row: row["selection_value"])
    best = ranked[0] if ranked else None
    penalties = [float(row.get("metrics", {}).get("chapter_17_deep_learning_selection_penalty", 0.0) or 0.0) for row in deep_rows]
    overfit = [float(row.get("metrics", {}).get("chapter_17_deep_learning_overfit_ratio", 0.0) or 0.0) for row in deep_rows]
    note_counts = [float(row.get("metrics", {}).get("chapter_17_deep_learning_note_count", 0.0) or 0.0) for row in deep_rows]
    promotion_candidate = bool(best and max(penalties or [1.0]) <= 0.08 and max(note_counts or [1.0]) <= 1.0)
    return {
        "selected_model": selected_model,
        "selected_deep_learning": bool(selected_deep),
        "deep_learning_candidate_count": int(len(deep_rows)),
        "best_deep_learning_candidate": best,
        "max_selection_penalty": float(max(penalties, default=0.0)),
        "max_overfit_ratio": float(max(overfit, default=0.0)),
        "promotion_candidate": promotion_candidate,
        "model_card_deep_learning_diagnostics": _deep_model_card_entries(model_card),
    }


def _candidate_summary(row: dict[str, Any], selection_metric: str) -> dict[str, Any]:
    metrics = row.get("metrics", {})
    metric = selection_metric.lower()
    adjusted_key = f"chapter_9_adjusted_{metric}"
    selection_key = adjusted_key if adjusted_key in metrics else metric
    value = metrics.get(selection_key)
    if value is None:
        value = metrics.get("mae", float("inf"))
    return {
        "model_name": row.get("model_name"),
        "selection_metric": selection_key,
        "selection_value": _finite(value, default=float("inf")),
        "mae": _finite(metrics.get("mae")),
        "holdout_mae": _finite(metrics.get("holdout_mae")),
        "directional_accuracy": _finite(metrics.get("directional_accuracy")),
        "selection_penalty": _finite(metrics.get("chapter_17_deep_learning_selection_penalty")),
        "selection_notes": metrics.get("chapter_17_deep_learning_notes", ""),
    }


def _deep_model_card_entries(model_card: dict[str, Any]) -> dict[str, Any]:
    diagnostics = model_card.get("model_diagnostics", {})
    return diagnostics.get("chapter_17_deep_learning", {}) if isinstance(diagnostics, dict) else {}


def _readiness(features: pd.DataFrame) -> dict[str, Any]:
    numeric = features.select_dtypes(include=[np.number])
    rows = int(len(numeric))
    columns = int(numeric.shape[1])
    feature_sample_ratio = float(columns / max(rows, 1))
    return {
        "rows": rows,
        "numeric_feature_count": columns,
        "feature_sample_ratio": feature_sample_ratio,
        "status": "ready" if rows >= 220 and feature_sample_ratio <= 0.50 else "limited",
        "reason": (
            "Enough observations and bounded feature/sample ratio for optional neural candidates."
            if rows >= 220 and feature_sample_ratio <= 0.50
            else "Deep learning should remain optional/research because sample size or feature/sample ratio is weak."
        ),
    }


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default
