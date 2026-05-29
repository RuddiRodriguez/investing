from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np
import pandas as pd

from market_forecasting_engine.models import ForecastCandidate
from market_forecasting_engine.risk import (
    directional_accuracy,
    evaluate_signal_risk,
    information_criteria,
    mae,
    mape,
    rmse,
    smape,
)
from market_forecasting_engine.schema import CandidateValidation


LOWER_IS_BETTER = {"rmse", "mae", "mape", "smape", "aic", "bic", "composite"}
HIGHER_IS_BETTER = {"directional_accuracy", "sharpe_ratio", "hit_rate", "profit_factor"}
TIME_SERIES_FAMILIES = {"classical_forecasting", "volatility_model", "multivariate_time_series", "state_space_model"}


def make_walk_forward_splits(
    n_rows: int,
    min_training_rows: int,
    validation_window: int,
    step_size: int,
    max_splits: int,
    purge_window: int = 0,
    embargo_window: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build expanding-window validation splits."""

    if n_rows < 30:
        raise ValueError("At least 30 supervised rows are required for validation.")

    min_train = min(min_training_rows, max(20, int(n_rows * 0.6)))
    val_window = min(validation_window, max(10, int(n_rows * 0.2)))
    step = max(step_size, 1)

    starts = list(range(min_train, n_rows - val_window + 1, step))
    if not starts:
        split = max(20, int(n_rows * 0.75))
        starts = [split]
        val_window = n_rows - split

    starts = starts[-max_splits:]
    splits = []
    for start in starts:
        stop = min(start + val_window, n_rows)
        train_stop = max(0, start - max(purge_window, 0) - max(embargo_window, 0))
        train_idx = np.arange(0, train_stop)
        validation_idx = np.arange(start, stop)
        if len(validation_idx) > 0 and len(train_idx) >= 20:
            splits.append((train_idx, validation_idx))
    return splits


def validate_candidate(
    candidate: ForecastCandidate,
    features: pd.DataFrame,
    target: pd.Series,
    horizon_days: int,
    min_training_rows: int,
    validation_window: int,
    step_size: int,
    max_splits: int,
    purge_window: int = 0,
    embargo_window: int = 0,
    final_holdout_fraction: float = 0.15,
) -> tuple[CandidateValidation, pd.DataFrame]:
    holdout_rows = _holdout_rows(len(target), validation_window, final_holdout_fraction)
    cv_features = features.iloc[:-holdout_rows] if holdout_rows else features
    cv_target = target.iloc[:-holdout_rows] if holdout_rows else target
    holdout_features = features.iloc[-holdout_rows:] if holdout_rows else features.iloc[0:0]
    holdout_target = target.iloc[-holdout_rows:] if holdout_rows else target.iloc[0:0]

    splits = make_walk_forward_splits(
        n_rows=len(cv_target),
        min_training_rows=min_training_rows,
        validation_window=validation_window,
        step_size=step_size,
        max_splits=max_splits,
        purge_window=purge_window,
        embargo_window=embargo_window,
    )
    if not splits:
        raise ValueError("No valid walk-forward splits after purge/embargo settings.")

    predictions = []
    for train_idx, validation_idx in splits:
        model = candidate.clone()
        train_features = cv_features.iloc[train_idx]
        train_target = cv_target.iloc[train_idx]
        validation_features = cv_features.iloc[validation_idx]
        validation_target = cv_target.iloc[validation_idx]

        model.fit(train_features, train_target)
        predicted = model.predict(validation_features)
        fold = pd.DataFrame(
            {
                "actual": validation_target.to_numpy(dtype=float),
                "predicted": predicted,
                "split_train_end": str(train_target.index[-1].date()),
            },
            index=validation_target.index,
        )
        predictions.append(fold)

    validation_predictions = pd.concat(predictions).sort_index()
    validation_predictions = validation_predictions[~validation_predictions.index.duplicated(keep="last")]

    actual = validation_predictions["actual"].to_numpy(dtype=float)
    predicted = validation_predictions["predicted"].to_numpy(dtype=float)
    metrics = _prediction_metrics(actual, predicted, candidate.parameter_count(features.shape[1]), horizon_days)
    metrics["purge_window"] = float(purge_window)
    metrics["embargo_window"] = float(embargo_window)
    metrics["final_holdout_rows"] = float(len(holdout_target))

    if len(holdout_target) > 0:
        holdout_model = candidate.clone().fit(cv_features, cv_target)
        holdout_predicted = holdout_model.predict(holdout_features)
        holdout_metrics = _prediction_metrics(
            holdout_target.to_numpy(dtype=float),
            holdout_predicted,
            candidate.parameter_count(features.shape[1]),
            horizon_days,
        )
        for key, value in holdout_metrics.items():
            metrics[f"holdout_{key}"] = value

    fitted_for_parameters = candidate.clone().fit(features, target)

    summary = CandidateValidation(
        model_name=candidate.name,
        model_family=candidate.family,
        model_parameters=fitted_for_parameters.parameters(),
        train_rows=int(max(train_idx[-1] + 1 for train_idx, _ in splits)),
        validation_rows=int(len(validation_predictions)),
        train_start_date=str(cv_target.index[0].date()),
        train_end_date=str(cv_target.iloc[: splits[-1][0][-1] + 1].index[-1].date()),
        validation_start_date=str(validation_predictions.index[0].date()),
        validation_end_date=str(validation_predictions.index[-1].date()),
        metrics={key: float(value) for key, value in metrics.items()},
    )
    return summary, validation_predictions


def validate_candidates(
    candidates: list[ForecastCandidate],
    features: pd.DataFrame,
    target: pd.Series,
    horizon_days: int,
    min_training_rows: int,
    validation_window: int,
    step_size: int,
    max_splits: int,
    purge_window: int = 0,
    embargo_window: int = 0,
    final_holdout_fraction: float = 0.15,
) -> list[tuple[ForecastCandidate, CandidateValidation, pd.DataFrame]]:
    results = []
    for candidate in candidates:
        try:
            summary, predictions = validate_candidate(
                candidate=candidate,
                features=features,
                target=target,
                horizon_days=horizon_days,
                min_training_rows=min_training_rows,
                validation_window=validation_window,
                step_size=step_size,
                max_splits=max_splits,
                purge_window=purge_window,
                embargo_window=embargo_window,
                final_holdout_fraction=final_holdout_fraction,
            )
        except Exception:
            continue
        results.append((candidate, summary, predictions))
    _add_multiple_testing_metrics(results)
    return results


def select_candidate(
    validation_results: list[tuple[ForecastCandidate, CandidateValidation, pd.DataFrame]],
    selection_metric: str,
) -> tuple[ForecastCandidate, CandidateValidation, pd.DataFrame]:
    if not validation_results:
        raise ValueError("No model candidates were validated.")

    metric = selection_metric.lower()
    if metric == "composite":
        return min(validation_results, key=lambda item: _composite_score(item[1]))

    if metric in LOWER_IS_BETTER:
        return min(validation_results, key=lambda item: item[1].metrics.get(_selection_metric_key(item[1], metric), float("inf")))

    if metric in HIGHER_IS_BETTER:
        return max(validation_results, key=lambda item: item[1].metrics.get(_selection_metric_key(item[1], metric), float("-inf")))

    raise ValueError(
        f"Unsupported selection metric `{selection_metric}`. "
        f"Use one of {sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER)}."
    )


def validation_summaries_as_dict(
    validation_results: list[tuple[ForecastCandidate, CandidateValidation, pd.DataFrame]],
) -> list[dict[str, object]]:
    return [asdict(summary) for _, summary, _ in validation_results]


def apply_ml4t_selection_adjustments(
    validation_results: list[tuple[ForecastCandidate, CandidateValidation, pd.DataFrame]],
    *,
    target: pd.Series,
    selection_metric: str,
) -> dict[str, object]:
    """Apply ML4T diagnostics as model-selection adjustments."""

    returns = pd.to_numeric(target, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    target_diagnostics = {
        "return_autocorrelation": _max_abs_autocorrelation(returns, max_lag=5),
        "volatility_clustering": _volatility_clustering_score(returns),
        "stationarity_risk": _stationarity_risk(returns),
    }
    adjusted = 0
    for candidate, summary, predictions in validation_results:
        metrics = summary.metrics
        chapter_6_penalty, chapter_6_notes = _chapter_6_process_penalty(metrics)
        chapter_8_penalty, chapter_8_notes = _chapter_8_backtest_penalty(metrics)
        chapter_10_penalty, chapter_10_notes = _chapter_10_bayesian_penalty(metrics, predictions)
        chapter_9_penalty, chapter_9_notes = _chapter_9_penalty(
            candidate=candidate,
            predictions=predictions,
            target_diagnostics=target_diagnostics,
        )
        penalty = min(chapter_6_penalty + chapter_8_penalty + chapter_9_penalty + chapter_10_penalty, 0.75)
        metrics["chapter_9_selection_penalty"] = float(penalty)
        metrics["chapter_6_process_selection_penalty"] = float(chapter_6_penalty)
        metrics["chapter_8_backtest_selection_penalty"] = float(chapter_8_penalty)
        metrics["chapter_9_time_series_selection_penalty"] = float(chapter_9_penalty)
        metrics["chapter_10_bayesian_selection_penalty"] = float(chapter_10_penalty)
        metrics["chapter_9_residual_autocorrelation"] = _max_abs_autocorrelation(
            pd.to_numeric(predictions["actual"], errors="coerce") - pd.to_numeric(predictions["predicted"], errors="coerce"),
            max_lag=5,
        )
        metrics["chapter_9_volatility_clustering_score"] = float(target_diagnostics["volatility_clustering"])
        metrics["chapter_9_return_autocorrelation_score"] = float(target_diagnostics["return_autocorrelation"])
        metrics["chapter_9_selection_adjustment_applied"] = 1.0
        metrics["chapter_9_selection_note_count"] = float(
            len(chapter_6_notes) + len(chapter_8_notes) + len(chapter_9_notes) + len(chapter_10_notes)
        )
        _add_adjusted_selection_metric(metrics, selection_metric.lower(), penalty)
        adjusted += 1
    return {
        "adjusted_candidate_count": int(adjusted),
        "selection_metric": selection_metric,
        "target_diagnostics": target_diagnostics,
        "policy": "ML4T diagnostics affect candidate ranking through explicit metric adjustments before model selection.",
    }


def apply_time_series_selection_adjustments(
    validation_results: list[tuple[ForecastCandidate, CandidateValidation, pd.DataFrame]],
    *,
    target: pd.Series,
    selection_metric: str,
) -> dict[str, object]:
    return apply_ml4t_selection_adjustments(
        validation_results,
        target=target,
        selection_metric=selection_metric,
    )


def _composite_score(summary: CandidateValidation) -> float:
    metrics = summary.metrics
    error_score = metrics.get("mae", 1.0) + metrics.get("rmse", 1.0)
    direction_penalty = 1.0 - metrics.get("directional_accuracy", 0.0)
    sharpe_penalty = max(0.0, -metrics.get("sharpe_ratio", 0.0)) * 0.05
    drawdown_penalty = abs(min(0.0, metrics.get("max_drawdown", 0.0))) * 0.25
    multiple_testing_penalty = max(0.0, -metrics.get("deflated_sharpe_ratio", 0.0)) * 0.01
    chapter_9_penalty = metrics.get("chapter_9_selection_penalty", 0.0)
    return float(error_score + direction_penalty * 0.02 + sharpe_penalty + drawdown_penalty + multiple_testing_penalty + chapter_9_penalty)


def _selection_metric_key(summary: CandidateValidation, metric: str) -> str:
    adjusted_key = f"chapter_9_adjusted_{metric}"
    return adjusted_key if adjusted_key in summary.metrics else metric


def _add_adjusted_selection_metric(metrics: dict[str, float], metric: str, penalty: float) -> None:
    if metric == "composite":
        metrics["chapter_9_adjusted_composite"] = _composite_score_for_metrics(metrics)
        return
    if metric in LOWER_IS_BETTER:
        base = float(metrics.get(metric, float("inf")))
        scale = max(abs(base), float(metrics.get("residual_std", 0.0)), 1e-6)
        metrics[f"chapter_9_adjusted_{metric}"] = float(base + penalty * scale)
    elif metric in HIGHER_IS_BETTER:
        base = float(metrics.get(metric, float("-inf")))
        metrics[f"chapter_9_adjusted_{metric}"] = float(base - penalty)


def _composite_score_for_metrics(metrics: dict[str, float]) -> float:
    error_score = metrics.get("mae", 1.0) + metrics.get("rmse", 1.0)
    direction_penalty = 1.0 - metrics.get("directional_accuracy", 0.0)
    sharpe_penalty = max(0.0, -metrics.get("sharpe_ratio", 0.0)) * 0.05
    drawdown_penalty = abs(min(0.0, metrics.get("max_drawdown", 0.0))) * 0.25
    multiple_testing_penalty = max(0.0, -metrics.get("deflated_sharpe_ratio", 0.0)) * 0.01
    chapter_9_penalty = metrics.get("chapter_9_selection_penalty", 0.0)
    return float(error_score + direction_penalty * 0.02 + sharpe_penalty + drawdown_penalty + multiple_testing_penalty + chapter_9_penalty)


def _chapter_9_penalty(
    *,
    candidate: ForecastCandidate,
    predictions: pd.DataFrame,
    target_diagnostics: dict[str, float],
) -> tuple[float, list[str]]:
    actual = pd.to_numeric(predictions["actual"], errors="coerce")
    predicted = pd.to_numeric(predictions["predicted"], errors="coerce")
    residuals = (actual - predicted).replace([np.inf, -np.inf], np.nan).dropna()
    residual_acf = _max_abs_autocorrelation(residuals, max_lag=5)
    penalty = 0.0
    notes = []
    if residual_acf >= 0.30:
        penalty += 0.18
        notes.append("residual_autocorrelation_high")
    elif residual_acf >= 0.20:
        penalty += 0.08
        notes.append("residual_autocorrelation_moderate")

    family = str(candidate.family)
    name = str(candidate.name).lower()
    is_time_series = family in TIME_SERIES_FAMILIES or name.startswith(("arima_", "sarima_", "var_", "garch_"))
    is_volatility = family == "volatility_model" or "garch" in name
    if is_time_series and target_diagnostics["stationarity_risk"] >= 0.5:
        penalty += 0.10
        notes.append("time_series_candidate_stationarity_risk")
    if target_diagnostics["volatility_clustering"] >= 0.15 and not is_volatility:
        penalty += 0.04
        notes.append("volatility_clustering_without_volatility_model")
    if is_volatility and target_diagnostics["volatility_clustering"] < 0.05:
        penalty += 0.04
        notes.append("volatility_model_without_clustering")
    return float(min(penalty, 0.40)), notes


def _chapter_6_process_penalty(metrics: dict[str, float]) -> tuple[float, list[str]]:
    penalty = 0.0
    notes = []
    mae = float(metrics.get("mae", 0.0) or 0.0)
    holdout_mae = float(metrics.get("holdout_mae", 0.0) or 0.0)
    if mae > 0 and holdout_mae > 0:
        degradation = (holdout_mae - mae) / abs(mae)
        if degradation > 0.50:
            penalty += 0.16
            notes.append("large_holdout_degradation")
        elif degradation > 0.25:
            penalty += 0.08
            notes.append("moderate_holdout_degradation")
    if float(metrics.get("sharpe_ratio", 0.0) or 0.0) > 0 and float(metrics.get("deflated_sharpe_ratio", 0.0) or 0.0) < 0:
        penalty += 0.08
        notes.append("negative_deflated_sharpe")
    if float(metrics.get("directional_accuracy", 0.0) or 0.0) < 0.50 and float(metrics.get("sharpe_ratio", 0.0) or 0.0) <= 0:
        penalty += 0.06
        notes.append("weak_direction_and_sharpe")
    return float(min(penalty, 0.25)), notes


def _chapter_8_backtest_penalty(metrics: dict[str, float]) -> tuple[float, list[str]]:
    penalty = 0.0
    notes = []
    max_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)
    profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
    hit_rate = float(metrics.get("hit_rate", 0.0) or 0.0)
    if max_drawdown < -0.20:
        penalty += 0.10
        notes.append("large_validation_drawdown")
    elif max_drawdown < -0.10:
        penalty += 0.05
        notes.append("moderate_validation_drawdown")
    if profit_factor < 1.0:
        penalty += 0.06
        notes.append("profit_factor_below_one")
    if hit_rate < 0.45:
        penalty += 0.04
        notes.append("low_hit_rate")
    return float(min(penalty, 0.20)), notes


def _chapter_10_bayesian_penalty(metrics: dict[str, float], predictions: pd.DataFrame) -> tuple[float, list[str]]:
    actual = pd.to_numeric(predictions["actual"], errors="coerce")
    predicted = pd.to_numeric(predictions["predicted"], errors="coerce")
    strategy = (np.sign(predicted.fillna(0.0)) * actual.fillna(0.0)).replace([np.inf, -np.inf], np.nan).dropna()
    posterior = _bayesian_sharpe_posterior(strategy, horizon_days=int(metrics.get("horizon_days", 1) or 1))
    prob_positive = posterior["probability_sharpe_positive"]
    width = posterior["credible_interval_width"]
    penalty = 0.0
    notes = []
    if prob_positive < 0.55:
        penalty += 0.12
        notes.append("low_probability_positive_sharpe")
    elif prob_positive < 0.65:
        penalty += 0.05
        notes.append("marginal_probability_positive_sharpe")
    if width > 2.0:
        penalty += 0.06
        notes.append("wide_sharpe_posterior")
    metrics["chapter_10_prob_sharpe_positive"] = float(prob_positive)
    metrics["chapter_10_posterior_sharpe_mean"] = float(posterior["posterior_sharpe_mean"])
    metrics["chapter_10_posterior_sharpe_std"] = float(posterior["posterior_sharpe_std"])
    metrics["chapter_10_sharpe_credible_interval_width"] = float(width)
    metrics["chapter_10_confidence_multiplier"] = float(max(0.55, min(1.0, prob_positive)))
    return float(min(penalty, 0.20)), notes


def _bayesian_sharpe_posterior(strategy_returns: pd.Series, horizon_days: int) -> dict[str, float]:
    clean = pd.to_numeric(strategy_returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3:
        return {
            "probability_sharpe_positive": 0.5,
            "posterior_sharpe_mean": 0.0,
            "posterior_sharpe_std": 2.0,
            "credible_interval_width": 4.0,
        }
    values = clean.to_numpy(dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    periods = 252 / max(int(horizon_days), 1)
    if std <= 1e-12:
        sharpe_mean = 0.0 if abs(mean) <= 1e-12 else np.sign(mean) * 4.0
        sharpe_std = 1.0 / np.sqrt(max(len(values), 1))
    else:
        sharpe_mean = mean / std * np.sqrt(periods)
        sharpe_std = float(np.sqrt((1.0 + 0.5 * sharpe_mean**2) / max(len(values) - 1, 1)))
    z = sharpe_mean / max(sharpe_std, 1e-12)
    prob_positive = 0.5 * (1.0 + math.erf(z / np.sqrt(2.0)))
    width = 3.92 * sharpe_std
    return {
        "probability_sharpe_positive": float(max(0.0, min(1.0, prob_positive))),
        "posterior_sharpe_mean": float(sharpe_mean) if np.isfinite(sharpe_mean) else 0.0,
        "posterior_sharpe_std": float(sharpe_std) if np.isfinite(sharpe_std) else 2.0,
        "credible_interval_width": float(width) if np.isfinite(width) else 4.0,
    }


def _max_abs_autocorrelation(series: pd.Series, max_lag: int) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    values = [abs(float(clean.autocorr(lag=lag) or 0.0)) for lag in range(1, max_lag + 1) if len(clean) > lag + 2]
    finite = [value for value in values if np.isfinite(value)]
    return float(max(finite, default=0.0))


def _volatility_clustering_score(series: pd.Series) -> float:
    squared = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna() ** 2
    return _max_abs_autocorrelation(squared, max_lag=5)


def _stationarity_risk(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 30:
        return 0.0
    try:
        from statsmodels.tsa.stattools import adfuller

        _, p_value, *_ = adfuller(clean.to_numpy(dtype=float), autolag="AIC")
        return 1.0 if float(p_value) >= 0.05 else 0.0
    except Exception:
        return 1.0 if abs(float(clean.autocorr(lag=1) or 0.0)) >= 0.90 else 0.0


def _holdout_rows(n_rows: int, validation_window: int, final_holdout_fraction: float) -> int:
    if final_holdout_fraction <= 0:
        return 0
    rows = int(round(n_rows * final_holdout_fraction))
    rows = max(min(validation_window, n_rows // 5), rows)
    return min(max(rows, 0), max(0, n_rows - 30))


def _prediction_metrics(actual: np.ndarray, predicted: np.ndarray, parameter_count: int, horizon_days: int) -> dict[str, float]:
    residuals = actual - predicted
    aic, bic = information_criteria(actual, predicted, parameter_count)
    signal_risk = evaluate_signal_risk(actual, predicted, horizon_days=horizon_days)
    return {
        "rmse": rmse(actual, predicted),
        "mae": mae(actual, predicted),
        "mape": mape(actual, predicted),
        "smape": smape(actual, predicted),
        "directional_accuracy": directional_accuracy(actual, predicted),
        "residual_std": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0,
        "aic": aic,
        "bic": bic,
        **signal_risk,
    }


def _add_multiple_testing_metrics(
    results: list[tuple[ForecastCandidate, CandidateValidation, pd.DataFrame]],
) -> None:
    candidate_count = max(len(results), 1)
    for _, summary, _ in results:
        validation_rows = max(summary.validation_rows, 2)
        sharpe = float(summary.metrics.get("sharpe_ratio", 0.0))
        sharpe_se = float(np.sqrt((1 + 0.5 * sharpe**2) / max(validation_rows - 1, 1)))
        multiple_testing_haircut = float(np.sqrt(2 * np.log(candidate_count)) * sharpe_se)
        summary.metrics["candidate_count"] = float(candidate_count)
        summary.metrics["multiple_testing_haircut"] = multiple_testing_haircut
        summary.metrics["deflated_sharpe_ratio"] = sharpe - multiple_testing_haircut
