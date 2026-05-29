from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def analyze_chapter_6_ml_process(
    *,
    features: pd.DataFrame,
    supervised: pd.DataFrame,
    forecasts: list[dict[str, Any]],
    candidate_results: dict[str, list[dict[str, Any]]],
    selected_validation_predictions: dict[str, list[dict[str, Any]]],
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    """Evaluate Chapter 6 ML process diagnostics and model-selection consequences."""

    horizon_reports = {}
    promotable_horizons = 0
    leakage_warnings = 0
    overfit_warnings = 0
    for horizon in horizons:
        key = str(horizon)
        forecast = _forecast_for_horizon(forecasts, horizon)
        validation_metrics = forecast.get("validation_metrics", {}) if isinstance(forecast, dict) else {}
        classification = _directional_classification(selected_validation_predictions.get(key, []))
        threshold = _threshold_diagnostics(
            selected_validation_predictions.get(key, []),
            validation_mae=_number(validation_metrics.get("mae")),
        )
        bias_variance = _bias_variance_diagnostic(
            validation_metrics=validation_metrics,
            candidates=candidate_results.get(key, []),
        )
        leakage = _validation_leakage_audit(validation_metrics=validation_metrics, horizon_days=int(horizon))
        feature_information = _mutual_information_ranking(
            features=features,
            supervised=supervised,
            target_column=f"target_log_return_{horizon}d",
        )
        gate_candidate = _horizon_gate_candidate(
            classification=classification,
            threshold=threshold,
            bias_variance=bias_variance,
            leakage=leakage,
        )
        promotable_horizons += int(gate_candidate["candidate_for_gate"])
        leakage_warnings += int(leakage["risk_level"] == "high")
        overfit_warnings += int(bias_variance["status"] == "overfit_risk")
        horizon_reports[key] = {
            "horizon_days": int(horizon),
            "selected_model": forecast.get("selected_model") if isinstance(forecast, dict) else None,
            "regression_metrics": _regression_metrics(validation_metrics),
            "directional_classification": classification,
            "threshold_diagnostics": threshold,
            "bias_variance_diagnostic": bias_variance,
            "validation_leakage_audit": leakage,
            "feature_information": feature_information,
            "gate_candidate": gate_candidate,
        }

    promotion_policy = _promotion_policy(
        horizon_count=len(horizons),
        promotable_horizons=promotable_horizons,
        leakage_warnings=leakage_warnings,
        overfit_warnings=overfit_warnings,
    )
    return {
        "chapter": 6,
        "name": "The Machine Learning Process",
        "status": "pass" if promotion_policy["candidate_for_promotion"] else "warn",
        "decision_policy": {
            "influences_final_action": False,
            "influences_model_selection": True,
            "mode": "model_selection_adjustment",
            "reason": "Chapter 6 process diagnostics penalize weak holdout, deflated-Sharpe, and no-edge candidates before model selection; they do not directly override Buy/Hold/Sell.",
        },
        "horizons": horizon_reports,
        "promotion_policy": promotion_policy,
        "technical_method_card": chapter_6_ml_process_method_card(),
    }


def chapter_6_ml_process_method_card() -> dict[str, Any]:
    return {
        "name": "jansen_ml4t_chapter_6_ml_process",
        "version": "chapter_6_ml_process_v1",
        "source": "Machine Learning for Algorithmic Trading, 2nd ed., Chapter 6",
        "purpose": "Audit model-process quality, feature information, classification behavior, overfit risk, and finance-specific validation design.",
        "decision_policy": "model_selection_adjustment",
        "implemented_components": [
            "mutual_information_feature_ranking",
            "directional_classification_metrics",
            "validation_threshold_diagnostics",
            "bias_variance_holdout_diagnostic",
            "walk_forward_purge_embargo_leakage_audit",
            "gate_promotion_candidate_policy",
        ],
        "not_implemented": [
            "KNN tutorial model",
            "generic IID cross_validation_for_financial_labels",
            "automatic Buy/Sell threshold override",
        ],
    }


def _forecast_for_horizon(forecasts: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    for forecast in forecasts:
        if int(forecast.get("horizon_days", -1)) == int(horizon):
            return forecast
    return {}


def _regression_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    keys = [
        "mae",
        "rmse",
        "mape",
        "smape",
        "residual_std",
        "holdout_mae",
        "holdout_rmse",
        "aic",
        "bic",
        "deflated_sharpe_ratio",
    ]
    return {key: _finite(_number(metrics.get(key))) for key in keys if key in metrics}


def _directional_classification(records: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(records)
    if frame.empty or "actual_log_return" not in frame or "predicted_log_return" not in frame:
        return _empty_classification("no_validation_records")
    actual = pd.to_numeric(frame["actual_log_return"], errors="coerce")
    predicted = pd.to_numeric(frame["predicted_log_return"], errors="coerce")
    valid = pd.DataFrame({"actual": actual, "predicted": predicted}).replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid[(valid["actual"] != 0) & (valid["predicted"] != 0)]
    if valid.empty:
        return _empty_classification("no_directional_records")
    actual_up = valid["actual"].to_numpy(dtype=float) > 0
    predicted_up = valid["predicted"].to_numpy(dtype=float) > 0
    tp = int(np.sum(predicted_up & actual_up))
    fp = int(np.sum(predicted_up & ~actual_up))
    tn = int(np.sum(~predicted_up & ~actual_up))
    fn = int(np.sum(~predicted_up & actual_up))
    precision_up = _safe_div(tp, tp + fp)
    recall_up = _safe_div(tp, tp + fn)
    precision_down = _safe_div(tn, tn + fn)
    recall_down = _safe_div(tn, tn + fp)
    f1_up = _f1(precision_up, recall_up)
    f1_down = _f1(precision_down, recall_down)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    balanced_accuracy = (recall_up + recall_down) / 2
    return {
        "status": "available",
        "rows": int(len(valid)),
        "confusion_matrix": {
            "true_positive_up": tp,
            "false_positive_up": fp,
            "true_negative_down": tn,
            "false_negative_down": fn,
        },
        "accuracy": _finite(accuracy),
        "balanced_accuracy": _finite(balanced_accuracy),
        "precision_up": _finite(precision_up),
        "recall_up": _finite(recall_up),
        "f1_up": _finite(f1_up),
        "precision_down": _finite(precision_down),
        "recall_down": _finite(recall_down),
        "f1_down": _finite(f1_down),
        "weak_side_precision": _finite(min(precision_up, precision_down)),
    }


def _threshold_diagnostics(records: list[dict[str, Any]], validation_mae: float) -> dict[str, Any]:
    frame = pd.DataFrame(records)
    if frame.empty or "actual_log_return" not in frame or "predicted_log_return" not in frame:
        return {"status": "no_validation_records", "best_threshold": None, "thresholds": []}
    actual = pd.to_numeric(frame["actual_log_return"], errors="coerce")
    predicted = pd.to_numeric(frame["predicted_log_return"], errors="coerce")
    valid = pd.DataFrame({"actual": actual, "predicted": predicted}).replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return {"status": "no_directional_records", "best_threshold": None, "thresholds": []}
    base = max(abs(validation_mae), 0.001)
    rows = []
    for multiplier in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
        threshold = base * multiplier
        active = valid[valid["predicted"].abs() >= threshold]
        if active.empty:
            rows.append(_threshold_row(multiplier, threshold, 0, len(valid), 0.0, 0.0, 0.0))
            continue
        signal_return = np.sign(active["predicted"].to_numpy(dtype=float)) * active["actual"].to_numpy(dtype=float)
        directional_hit = signal_return > 0
        rows.append(
            _threshold_row(
                multiplier,
                threshold,
                int(len(active)),
                int(len(valid)),
                float(directional_hit.mean()),
                float(np.mean(signal_return)),
                float(np.sum(signal_return)),
            )
        )
    eligible = [row for row in rows if row["coverage"] >= 0.10]
    best = max(eligible, key=lambda row: (row["directional_precision"], row["mean_signed_log_return"]), default=None)
    return {
        "status": "available",
        "base_validation_mae": _finite(base),
        "best_threshold": best,
        "thresholds": rows,
        "candidate_gate": bool(best and best["directional_precision"] >= 0.55 and best["mean_signed_log_return"] > 0),
        "reason": (
            "A validation threshold reached useful precision and positive signed return."
            if best and best["directional_precision"] >= 0.55 and best["mean_signed_log_return"] > 0
            else "No validation threshold was strong enough to justify an automatic action gate."
        ),
    }


def _threshold_row(
    multiplier: float,
    threshold: float,
    active_rows: int,
    total_rows: int,
    precision: float,
    mean_signed_return: float,
    cumulative_signed_return: float,
) -> dict[str, float]:
    return {
        "mae_multiplier": _finite(multiplier),
        "threshold": _finite(threshold),
        "active_rows": float(active_rows),
        "coverage": _finite(_safe_div(active_rows, total_rows)),
        "directional_precision": _finite(precision),
        "mean_signed_log_return": _finite(mean_signed_return),
        "cumulative_signed_log_return": _finite(cumulative_signed_return),
    }


def _bias_variance_diagnostic(validation_metrics: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    mae = _number(validation_metrics.get("mae"))
    holdout_mae = _number(validation_metrics.get("holdout_mae"))
    sharpe = _number(validation_metrics.get("sharpe_ratio"))
    deflated_sharpe = _number(validation_metrics.get("deflated_sharpe_ratio"))
    degradation = _safe_div(holdout_mae - mae, abs(mae)) if mae > 0 and holdout_mae > 0 else 0.0
    candidate_maes = [
        _number(row.get("metrics", {}).get("mae"))
        for row in candidates
        if isinstance(row, dict) and isinstance(row.get("metrics"), dict)
    ]
    candidate_maes = [value for value in candidate_maes if value > 0]
    dispersion = float(np.std(candidate_maes, ddof=1)) if len(candidate_maes) > 1 else 0.0
    status = "balanced"
    reasons = []
    if holdout_mae > 0 and mae > 0 and degradation > 0.25:
        status = "overfit_risk"
        reasons.append("Holdout MAE degraded more than 25% versus validation MAE.")
    if sharpe > 0 and deflated_sharpe < 0:
        status = "overfit_risk"
        reasons.append("Deflated Sharpe is negative after multiple-testing adjustment.")
    if _number(validation_metrics.get("directional_accuracy")) < 0.51 and sharpe <= 0:
        status = "underfit_or_no_edge"
        reasons.append("Directional accuracy and validation Sharpe do not show edge.")
    if not reasons:
        reasons.append("Holdout, direction, and multiple-testing diagnostics do not show a clear process failure.")
    return {
        "status": status,
        "holdout_mae_degradation_ratio": _finite(degradation),
        "candidate_mae_dispersion": _finite(dispersion),
        "candidate_count": int(len(candidate_maes)),
        "reason": " ".join(reasons),
    }


def _validation_leakage_audit(validation_metrics: dict[str, Any], horizon_days: int) -> dict[str, Any]:
    purge = _number(validation_metrics.get("purge_window"))
    embargo = _number(validation_metrics.get("embargo_window"))
    holdout_rows = _number(validation_metrics.get("final_holdout_rows"))
    if purge >= horizon_days and holdout_rows > 0:
        risk = "low"
    elif purge > 0 and holdout_rows > 0:
        risk = "medium"
    else:
        risk = "high"
    return {
        "risk_level": risk,
        "purge_window": _finite(purge),
        "embargo_window": _finite(embargo),
        "final_holdout_rows": _finite(holdout_rows),
        "horizon_days": int(horizon_days),
        "point_in_time_assumption": "features are lagged/rolling and validation uses chronological walk-forward splits",
        "reason": (
            "Purge covers the forecast horizon and a final holdout is present."
            if risk == "low"
            else "Validation design does not fully cover the forecast horizon with purge plus holdout evidence."
        ),
    }


def _mutual_information_ranking(
    *,
    features: pd.DataFrame,
    supervised: pd.DataFrame,
    target_column: str,
    max_features: int = 80,
    max_rows: int = 1200,
) -> dict[str, Any]:
    if target_column not in supervised.columns:
        return {"status": "missing_target", "target_column": target_column, "top_features": []}
    feature_columns = [column for column in features.columns if column in supervised.columns and not str(column).startswith("target_")]
    frame = supervised[feature_columns + [target_column]].replace([np.inf, -np.inf], np.nan).dropna(subset=[target_column])
    if len(frame) < 40:
        return {"status": "insufficient_rows", "target_column": target_column, "top_features": []}
    numeric_features = frame[feature_columns].select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    if numeric_features.empty:
        return {"status": "no_numeric_features", "target_column": target_column, "top_features": []}
    variances = numeric_features.var(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    selected_columns = list(variances.sort_values(ascending=False).head(max_features).index)
    sample = frame[selected_columns + [target_column]].tail(max_rows).copy()
    x = sample[selected_columns].replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    y = pd.to_numeric(sample[target_column], errors="coerce").fillna(0.0)
    if len(x) < 40 or float(y.std()) <= 1e-12:
        return {"status": "insufficient_target_variance", "target_column": target_column, "top_features": []}
    try:
        scores = mutual_info_regression(x, y, random_state=42, n_neighbors=min(5, max(3, len(x) // 20)))
    except Exception as exc:
        return {"status": "failed", "target_column": target_column, "error": str(exc), "top_features": []}
    ranked = sorted(
        (
            {"feature": str(column), "mutual_information": _finite(float(score))}
            for column, score in zip(selected_columns, scores)
        ),
        key=lambda row: row["mutual_information"],
        reverse=True,
    )
    useful = [row for row in ranked if row["mutual_information"] > 0]
    return {
        "status": "available",
        "target_column": target_column,
        "rows": int(len(x)),
        "tested_feature_count": int(len(selected_columns)),
        "positive_mi_feature_count": int(len(useful)),
        "top_features": ranked[:15],
    }


def _horizon_gate_candidate(
    *,
    classification: dict[str, Any],
    threshold: dict[str, Any],
    bias_variance: dict[str, Any],
    leakage: dict[str, Any],
) -> dict[str, Any]:
    candidate = (
        classification.get("status") == "available"
        and classification.get("weak_side_precision", 0.0) >= 0.52
        and bool(threshold.get("candidate_gate"))
        and bias_variance.get("status") != "overfit_risk"
        and leakage.get("risk_level") in {"low", "medium"}
    )
    return {
        "candidate_for_gate": bool(candidate),
        "gate_type": "threshold_action_filter",
        "reason": (
            "Directional precision, threshold evidence, bias/variance, and leakage diagnostics are acceptable for this horizon."
            if candidate
            else "This horizon does not yet justify promoting Chapter 6 into an action gate."
        ),
    }


def _promotion_policy(
    *,
    horizon_count: int,
    promotable_horizons: int,
    leakage_warnings: int,
    overfit_warnings: int,
) -> dict[str, Any]:
    candidate = horizon_count > 0 and promotable_horizons == horizon_count and leakage_warnings == 0 and overfit_warnings == 0
    return {
        "status": "candidate_for_gate_promotion" if candidate else "report_only",
        "candidate_for_promotion": bool(candidate),
        "promotable_horizon_count": int(promotable_horizons),
        "tested_horizon_count": int(horizon_count),
        "high_leakage_horizon_count": int(leakage_warnings),
        "overfit_warning_horizon_count": int(overfit_warnings),
        "recommended_action": (
            "Run a controlled before/after action-gate backtest before enabling the gate."
            if candidate
            else "Keep Chapter 6 diagnostic-only until saved-run stability improves."
        ),
    }


def _empty_classification(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "rows": 0,
        "accuracy": 0.0,
        "balanced_accuracy": 0.0,
        "precision_up": 0.0,
        "recall_up": 0.0,
        "f1_up": 0.0,
        "precision_down": 0.0,
        "recall_down": 0.0,
        "f1_down": 0.0,
        "weak_side_precision": 0.0,
    }


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(numerator / denominator)


def _number(value: Any) -> float:
    try:
        number = float(value if value is not None else 0.0)
    except Exception:
        return 0.0
    return number if np.isfinite(number) else 0.0


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0
