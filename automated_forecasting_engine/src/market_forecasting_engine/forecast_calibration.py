from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MIN_ACTIVE_SAMPLES = 5
FULL_ACTIVE_SAMPLES = 15
MAX_BIAS_CORRECTION_FRACTION = 0.35
MAX_BAND_SCALE = 2.25
MIN_BAND_SCALE = 0.75


@dataclass(frozen=True)
class ForecastCalibrationSettings:
    enabled: bool = True
    ledger_path: str | None = None
    min_active_samples: int = MIN_ACTIVE_SAMPLES
    full_active_samples: int = FULL_ACTIVE_SAMPLES
    max_bias_correction_fraction: float = MAX_BIAS_CORRECTION_FRACTION
    lookback_matured_rows: int = 60
    decay_halflife_rows: float = 20.0


def apply_forecast_calibration(
    report: dict[str, Any],
    *,
    settings: ForecastCalibrationSettings | None = None,
) -> dict[str, Any]:
    settings = settings or ForecastCalibrationSettings()
    feedback = build_calibration_feedback(
        ledger_path=Path(settings.ledger_path) if settings.ledger_path else None,
        ticker=str(report.get("ticker") or ""),
        settings=settings,
    )
    report.setdefault("diagnostics", {})["forecast_calibration"] = feedback
    report.setdefault("operations_view", {})["forecast_calibration"] = feedback
    if not settings.enabled or feedback.get("status") not in {"active", "partial"}:
        return report

    current_price = _float(report.get("current_price"))
    calibrated_count = 0
    for forecast in report.get("forecasts", []) or []:
        if not isinstance(forecast, dict):
            continue
        horizon_key = _horizon_key(forecast)
        horizon_feedback = (feedback.get("by_horizon") or {}).get(horizon_key)
        if not isinstance(horizon_feedback, dict) or horizon_feedback.get("status") != "active":
            forecast["forecast_calibration"] = horizon_feedback or {
                "status": "insufficient_matured_samples",
                "horizon_key": horizon_key,
            }
            continue
        _apply_horizon_calibration(forecast, current_price=current_price, horizon_feedback=horizon_feedback, settings=settings)
        calibrated_count += 1

    feedback["calibrated_forecast_count"] = int(calibrated_count)
    return report


def build_calibration_feedback(
    *,
    ledger_path: Path | None,
    ticker: str,
    settings: ForecastCalibrationSettings | None = None,
) -> dict[str, Any]:
    settings = settings or ForecastCalibrationSettings()
    if not settings.enabled:
        return {"status": "disabled", "policy": _policy(settings)}
    if ledger_path is None:
        return {"status": "missing_ledger_path", "policy": _policy(settings)}
    if not ledger_path.exists():
        return {"status": "missing_ledger", "ledger_path": str(ledger_path), "policy": _policy(settings)}
    try:
        ledger = pd.read_csv(ledger_path)
    except Exception as exc:
        return {"status": "ledger_read_failed", "ledger_path": str(ledger_path), "error": str(exc), "policy": _policy(settings)}
    matured = _matured_rows(ledger, ticker=ticker)
    if matured.empty:
        return {
            "status": "no_matured_rows",
            "ledger_path": str(ledger_path),
            "matured_rows": 0,
            "policy": _policy(settings),
        }
    by_horizon: dict[str, dict[str, Any]] = {}
    for horizon_key, frame in matured.groupby("horizon_key", sort=True):
        by_horizon[str(horizon_key)] = _summarize_horizon_feedback(frame, settings=settings)
    active_count = sum(1 for item in by_horizon.values() if item.get("status") == "active")
    return {
        "status": "active" if active_count else "partial",
        "ledger_path": str(ledger_path),
        "matured_rows": int(len(matured)),
        "active_horizon_count": int(active_count),
        "by_horizon": by_horizon,
        "policy": _policy(settings),
    }


def update_forecast_ledger(
    report: dict[str, Any],
    prices: pd.DataFrame,
    *,
    output_dir: Path,
    target_column: str,
) -> dict[str, str]:
    ledger_path = output_dir / "forecast_ledger.csv"
    existing = pd.read_csv(ledger_path) if ledger_path.exists() else pd.DataFrame()
    now = datetime.now(UTC).isoformat()
    run_id = f"{report.get('ticker', 'TICKER')}_{report.get('as_of_timestamp', report.get('as_of_date', now))}"
    rows = []
    for forecast in report.get("forecasts", []) or []:
        if not isinstance(forecast, dict):
            continue
        gate = forecast.get("production_gate", forecast.get("validation_gate", {}))
        rows.append(
            {
                "run_id": run_id,
                "ticker": report.get("ticker"),
                "as_of_timestamp": report.get("as_of_timestamp", report.get("as_of_date")),
                "created_at_utc": now,
                "forecast_timestamp": forecast.get("forecast_date"),
                "horizon_bars": forecast.get("horizon_days"),
                "horizon_hours": forecast.get("horizon_hours"),
                "current_price": report.get("current_price"),
                "predicted_price": forecast.get("predicted_price"),
                "raw_predicted_price": (forecast.get("raw_forecast") or {}).get("predicted_price"),
                "lower_price": forecast.get("lower_price"),
                "upper_price": forecast.get("upper_price"),
                "expected_direction": forecast.get("expected_direction"),
                "selected_model": forecast.get("selected_model"),
                "trade_allowed": forecast.get("trade_allowed"),
                "gate_status": gate.get("status") if isinstance(gate, dict) else None,
                "forecast_calibration_status": (forecast.get("forecast_calibration") or {}).get("status"),
                "actual_price": np.nan,
                "actual_timestamp": "",
                "error": np.nan,
                "absolute_error": np.nan,
                "absolute_pct_error": np.nan,
                "direction_correct": np.nan,
                "status": "pending",
            }
        )
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if not existing.empty else pd.DataFrame(rows)
    combined = score_matured_ledger_rows(combined, prices, target_column)
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["run_id", "horizon_bars"], keep="last")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(ledger_path, index=False)
    summary_path = output_dir / "forecast_ledger_summary.json"
    summary_path.write_text(json.dumps(ledger_summary(combined), indent=2, default=str) + "\n", encoding="utf-8")
    return {"forecast_ledger": str(ledger_path), "forecast_ledger_summary": str(summary_path)}


def score_matured_ledger_rows(ledger: pd.DataFrame, prices: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    output = ledger.copy()
    price_series = prices[target_column].astype(float).sort_index()
    price_index = pd.DatetimeIndex(price_series.index)
    output["forecast_timestamp"] = pd.to_datetime(output["forecast_timestamp"], errors="coerce")
    for index, row in output.iterrows():
        if str(row.get("status", "")) == "matured" and pd.notna(row.get("actual_price")):
            continue
        forecast_timestamp = row.get("forecast_timestamp")
        if pd.isna(forecast_timestamp):
            output.at[index, "status"] = "invalid_timestamp"
            continue
        future_positions = np.where(price_index >= pd.Timestamp(forecast_timestamp))[0]
        if len(future_positions) == 0:
            output.at[index, "status"] = "pending"
            continue
        actual_idx = int(future_positions[0])
        actual_timestamp = price_index[actual_idx]
        actual_price = float(price_series.iloc[actual_idx])
        predicted_price = _float(row.get("predicted_price"))
        current_price = _float(row.get("current_price"))
        output.at[index, "actual_price"] = actual_price
        output.at[index, "actual_timestamp"] = actual_timestamp.isoformat()
        if predicted_price is None:
            output.at[index, "status"] = "invalid_prediction"
            continue
        error = actual_price - predicted_price
        output.at[index, "error"] = error
        output.at[index, "absolute_error"] = abs(error)
        output.at[index, "absolute_pct_error"] = abs(predicted_price / actual_price - 1.0) if actual_price else np.nan
        predicted_direction = np.sign(predicted_price - current_price) if current_price is not None else np.nan
        actual_direction = np.sign(actual_price - current_price) if current_price is not None else np.nan
        output.at[index, "direction_correct"] = bool(predicted_direction == actual_direction) if np.isfinite(predicted_direction) and np.isfinite(actual_direction) else np.nan
        output.at[index, "status"] = "matured"
    return output


def ledger_summary(ledger: pd.DataFrame) -> dict[str, Any]:
    matured = ledger[ledger.get("status") == "matured"] if "status" in ledger else pd.DataFrame()
    if matured.empty:
        return {"rows": int(len(ledger)), "matured_rows": 0}
    matured = matured.copy()
    matured["horizon_key"] = matured.apply(_ledger_horizon_key, axis=1)
    return {
        "rows": int(len(ledger)),
        "matured_rows": int(len(matured)),
        "mean_absolute_error": float(pd.to_numeric(matured["absolute_error"], errors="coerce").mean()),
        "mean_absolute_pct_error": float(pd.to_numeric(matured["absolute_pct_error"], errors="coerce").mean()),
        "directional_accuracy": float(pd.to_numeric(matured["direction_correct"], errors="coerce").mean()),
        "by_horizon": matured.groupby("horizon_key")
        .agg(
            matured_rows=("status", "count"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_absolute_pct_error=("absolute_pct_error", "mean"),
            directional_accuracy=("direction_correct", "mean"),
            bias=("error", "mean"),
        )
        .reset_index()
        .to_dict(orient="records"),
    }


def _apply_horizon_calibration(
    forecast: dict[str, Any],
    *,
    current_price: float | None,
    horizon_feedback: dict[str, Any],
    settings: ForecastCalibrationSettings,
) -> None:
    predicted = _float(forecast.get("predicted_price"))
    lower = _float(forecast.get("lower_price"))
    upper = _float(forecast.get("upper_price"))
    if predicted is None:
        return
    bias = float(horizon_feedback.get("weighted_bias") or 0.0)
    strength = float(horizon_feedback.get("calibration_strength") or 0.0)
    correction_cap = abs(predicted) * float(settings.max_bias_correction_fraction)
    correction = float(np.clip(bias * strength, -correction_cap, correction_cap))
    band_scale = float(horizon_feedback.get("band_scale") or 1.0)
    calibrated_prediction = predicted + correction
    raw = {
        "predicted_price": predicted,
        "lower_price": lower,
        "upper_price": upper,
        "expected_return": forecast.get("expected_return"),
        "expected_direction": forecast.get("expected_direction"),
        "directional_confidence": forecast.get("directional_confidence"),
    }
    forecast["raw_forecast"] = raw
    forecast["predicted_price"] = calibrated_prediction
    if lower is not None and upper is not None:
        half_width = max((upper - lower) / 2.0, abs(predicted) * 0.0005)
        forecast["lower_price"] = calibrated_prediction - half_width * band_scale
        forecast["upper_price"] = calibrated_prediction + half_width * band_scale
    if current_price is not None and current_price > 0:
        expected_return = calibrated_prediction / current_price - 1.0
        forecast["expected_return"] = float(expected_return)
        forecast["expected_log_return"] = float(np.log1p(expected_return)) if expected_return > -1 else forecast.get("expected_log_return")
        forecast["expected_direction"] = "Up" if expected_return > 0 else "Down" if expected_return < 0 else "Flat"
    direction_accuracy = _float(horizon_feedback.get("directional_accuracy"))
    old_confidence = _float(forecast.get("directional_confidence")) or 0.5
    if direction_accuracy is not None:
        reliability = max(0.35, min(1.0, direction_accuracy / 0.55))
        forecast["directional_confidence"] = float(min(0.95, max(0.35, 0.5 + (old_confidence - 0.5) * reliability)))
    forecast["confidence_interval_method"] = f"{forecast.get('confidence_interval_method', 'model')}_feedback_calibrated"
    forecast["forecast_calibration"] = {
        "status": "active",
        "horizon_key": horizon_feedback.get("horizon_key"),
        "sample_size": horizon_feedback.get("sample_size"),
        "calibration_strength": strength,
        "weighted_bias": bias,
        "applied_bias_correction": correction,
        "directional_accuracy": horizon_feedback.get("directional_accuracy"),
        "band_scale": band_scale,
        "policy": "Calibrated from matured forecast errors for the same ticker and horizon before downstream action/CEO decision.",
    }


def _summarize_horizon_feedback(frame: pd.DataFrame, *, settings: ForecastCalibrationSettings) -> dict[str, Any]:
    frame = frame.sort_values("actual_timestamp").tail(int(settings.lookback_matured_rows)).copy()
    sample_size = int(len(frame))
    horizon_key = str(frame["horizon_key"].iloc[0]) if sample_size else "unknown"
    errors = pd.to_numeric(frame["error"], errors="coerce")
    absolute_errors = pd.to_numeric(frame["absolute_error"], errors="coerce")
    direction = pd.to_numeric(frame["direction_correct"], errors="coerce")
    weights = _decay_weights(sample_size, settings.decay_halflife_rows)
    weighted_bias = _weighted_mean(errors, weights)
    weighted_mae = _weighted_mean(absolute_errors, weights)
    directional_accuracy = _weighted_mean(direction, weights)
    coverage = _coverage(frame)
    status = "active" if sample_size >= int(settings.min_active_samples) else "insufficient_matured_samples"
    strength = 0.0
    if status == "active":
        span = max(int(settings.full_active_samples) - int(settings.min_active_samples), 1)
        strength = min(1.0, max(0.25, (sample_size - int(settings.min_active_samples) + 1) / span))
    band_scale = _band_scale(coverage, sample_size, settings=settings)
    return {
        "status": status,
        "horizon_key": horizon_key,
        "sample_size": sample_size,
        "weighted_bias": weighted_bias,
        "weighted_mae": weighted_mae,
        "directional_accuracy": directional_accuracy,
        "coverage_rate": coverage,
        "calibration_strength": strength,
        "band_scale": band_scale,
        "latest_actual_timestamp": str(frame["actual_timestamp"].iloc[-1]) if sample_size and "actual_timestamp" in frame else None,
    }


def _matured_rows(ledger: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
    if ledger.empty or "status" not in ledger:
        return pd.DataFrame()
    output = ledger[ledger["status"].astype(str).str.lower() == "matured"].copy()
    if ticker and "ticker" in output:
        output = output[output["ticker"].astype(str).str.upper() == ticker.upper()]
    if output.empty:
        return output
    output["predicted_price"] = pd.to_numeric(output["predicted_price"], errors="coerce")
    output["actual_price"] = pd.to_numeric(output["actual_price"], errors="coerce")
    output["current_price"] = pd.to_numeric(output["current_price"], errors="coerce")
    output["lower_price"] = pd.to_numeric(output.get("lower_price"), errors="coerce")
    output["upper_price"] = pd.to_numeric(output.get("upper_price"), errors="coerce")
    output["error"] = pd.to_numeric(output.get("error"), errors="coerce")
    missing_error = output["error"].isna()
    output.loc[missing_error, "error"] = output.loc[missing_error, "actual_price"] - output.loc[missing_error, "predicted_price"]
    output["absolute_error"] = pd.to_numeric(output.get("absolute_error"), errors="coerce")
    output.loc[output["absolute_error"].isna(), "absolute_error"] = output.loc[output["absolute_error"].isna(), "error"].abs()
    if "direction_correct" not in output:
        output["direction_correct"] = np.nan
    output["direction_correct"] = output["direction_correct"].map(_boolish)
    missing_direction = output["direction_correct"].isna()
    predicted_direction = np.sign(output.loc[missing_direction, "predicted_price"] - output.loc[missing_direction, "current_price"])
    actual_direction = np.sign(output.loc[missing_direction, "actual_price"] - output.loc[missing_direction, "current_price"])
    output.loc[missing_direction, "direction_correct"] = predicted_direction.eq(actual_direction)
    output["horizon_key"] = output.apply(_ledger_horizon_key, axis=1)
    output["actual_timestamp"] = pd.to_datetime(output.get("actual_timestamp"), errors="coerce")
    return output.dropna(subset=["predicted_price", "actual_price", "error", "absolute_error", "horizon_key"])


def _ledger_horizon_key(row: pd.Series) -> str:
    hours = _float(row.get("horizon_hours"))
    if hours is not None and hours > 0:
        return f"{hours:g}h"
    bars = _float(row.get("horizon_bars"))
    if bars is not None:
        return f"{bars:g}b"
    return "unknown"


def _horizon_key(forecast: dict[str, Any]) -> str:
    hours = _float(forecast.get("horizon_hours"))
    if hours is not None and hours > 0:
        return f"{hours:g}h"
    bars = _float(forecast.get("horizon_days"))
    if bars is not None:
        return f"{bars:g}b"
    return "unknown"


def _coverage(frame: pd.DataFrame) -> float | None:
    if not {"actual_price", "lower_price", "upper_price"}.issubset(frame.columns):
        return None
    valid = frame.dropna(subset=["actual_price", "lower_price", "upper_price"])
    if valid.empty:
        return None
    return float(((valid["actual_price"] >= valid["lower_price"]) & (valid["actual_price"] <= valid["upper_price"])).mean())


def _band_scale(coverage: float | None, sample_size: int, *, settings: ForecastCalibrationSettings) -> float:
    if coverage is None or sample_size < settings.min_active_samples:
        return 1.0
    if coverage < 0.55:
        return MAX_BAND_SCALE
    if coverage < 0.70:
        return 1.5
    if coverage > 0.92 and sample_size >= settings.full_active_samples:
        return MIN_BAND_SCALE
    return 1.0


def _decay_weights(size: int, halflife: float) -> np.ndarray:
    if size <= 0:
        return np.array([])
    positions = np.arange(size, dtype=float)
    age = size - 1 - positions
    return np.power(0.5, age / max(float(halflife), 1.0))


def _weighted_mean(series: pd.Series, weights: np.ndarray) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights)
    if not mask.any():
        return None
    return float(np.average(values[mask], weights=weights[mask]))


def _boolish(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return 1.0
    if text in {"false", "0", "no"}:
        return 0.0
    return np.nan


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _policy(settings: ForecastCalibrationSettings) -> dict[str, Any]:
    return {
        "enabled": bool(settings.enabled),
        "min_active_samples": int(settings.min_active_samples),
        "full_active_samples": int(settings.full_active_samples),
        "lookback_matured_rows": int(settings.lookback_matured_rows),
        "decay_halflife_rows": float(settings.decay_halflife_rows),
        "max_bias_correction_fraction": float(settings.max_bias_correction_fraction),
        "sample_gate": "Fewer than min_active_samples is report-only; active calibration starts at min_active_samples and scales toward full_active_samples.",
    }
