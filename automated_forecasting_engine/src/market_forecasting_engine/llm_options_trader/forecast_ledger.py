from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def forecast_ledger_path(output_dir: Path, currency: str) -> Path:
    return output_dir / "forecasts" / f"{currency.upper()}_forecast_ledger.json"


def update_forecast_ledger(
    *,
    output_dir: Path,
    currency: str,
    forecast: dict[str, Any],
    price_bars: list[dict[str, Any]],
    max_records: int = 600,
) -> dict[str, Any]:
    path = forecast_ledger_path(output_dir, currency)
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger = _read_ledger(path, currency)
    records = ledger.get("records") if isinstance(ledger.get("records"), list) else []
    records = _mature_records(records, price_bars)
    records = _append_current_forecasts(records, forecast)
    records = sorted(records, key=lambda item: str(item.get("target_timestamp") or ""))[-max(1, int(max_records)) :]
    payload = {
        "status": "ok",
        "currency": currency.upper(),
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "record_count": len(records),
        "summary": summarize_forecast_records(records),
        "records": records,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return compact_forecast_validation(payload)


def load_forecast_validation(output_dir: Path, currency: str) -> dict[str, Any]:
    path = forecast_ledger_path(output_dir, currency)
    if not path.exists():
        return {"status": "empty", "summary": "No matured forecast validation is available yet.", "recent_matured": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "unreadable", "summary": "Forecast ledger exists but could not be parsed.", "recent_matured": []}
    return compact_forecast_validation(payload)


def compact_forecast_validation(payload: dict[str, Any], *, max_recent: int = 12) -> dict[str, Any]:
    records = payload.get("records") if isinstance(payload.get("records"), list) else []
    matured = [record for record in records if record.get("status") == "matured"]
    pending = [record for record in records if record.get("status") == "pending"]
    return {
        "status": payload.get("status") or "ok",
        "updated_at_utc": payload.get("updated_at_utc"),
        "record_count": len(records),
        "matured_count": len(matured),
        "pending_count": len(pending),
        "summary": summarize_forecast_records(records),
        "by_horizon": _by_horizon(matured),
        "recent_matured": matured[-max(1, int(max_recent)) :],
        "error_feedback": forecast_error_feedback(matured[-max(1, int(max_recent)) :]),
    }


def forecast_error_feedback(matured_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not matured_records:
        return {
            "status": "empty",
            "instruction": "No matured forecast errors yet. Treat the current forecast as unvalidated evidence, not a command.",
        }
    recent = matured_records[-min(8, len(matured_records)) :]
    errors = [_float(record.get("error")) for record in recent]
    errors = [value for value in errors if value is not None]
    abs_errors = [abs(value) for value in errors]
    directions = [record for record in recent if record.get("direction_correct") is not None]
    direction_accuracy = None if not directions else sum(1 for record in directions if record.get("direction_correct")) / len(directions)
    mean_error = None if not errors else sum(errors) / len(errors)
    mae = None if not abs_errors else sum(abs_errors) / len(abs_errors)
    latest = recent[-1]
    latest_error = _float(latest.get("error"))
    bias = "unknown"
    if mean_error is not None:
        if mean_error > 0:
            bias = "under_predicted_actual_price"
        elif mean_error < 0:
            bias = "over_predicted_actual_price"
        else:
            bias = "unbiased_recently"
    reliability = "unvalidated"
    if direction_accuracy is not None:
        if direction_accuracy < 0.4:
            reliability = "poor_directional_reliability"
        elif direction_accuracy < 0.6:
            reliability = "mixed_directional_reliability"
        else:
            reliability = "acceptable_directional_reliability"
    instruction = _forecast_error_instruction(bias=bias, reliability=reliability, mean_error=mean_error, latest_error=latest_error)
    return {
        "status": "ok",
        "sample_size": len(recent),
        "bias": bias,
        "mean_error_actual_minus_predicted": mean_error,
        "mae": mae,
        "directional_accuracy": direction_accuracy,
        "latest_error_actual_minus_predicted": latest_error,
        "latest_direction_correct": latest.get("direction_correct"),
        "reliability": reliability,
        "instruction": instruction,
    }


def _forecast_error_instruction(*, bias: str, reliability: str, mean_error: float | None, latest_error: float | None) -> str:
    parts: list[str] = []
    if reliability == "poor_directional_reliability":
        parts.append("Recent forecast direction has been poor; do not let it veto tape, MA, A/D, StochRSI, or option-chain evidence.")
    elif reliability == "mixed_directional_reliability":
        parts.append("Recent forecast direction is mixed; use it only as weak evidence and demand agreement with live tape.")
    else:
        parts.append("Recent forecast direction is usable but still secondary to executable option edge and live tape.")
    if bias == "under_predicted_actual_price":
        parts.append("Forecasts have recently under-read actual price; adjust upward/upside scenarios and be careful rejecting calls only because the median forecast is flat or low.")
    elif bias == "over_predicted_actual_price":
        parts.append("Forecasts have recently over-read actual price; adjust downward/downside scenarios and be careful rejecting puts only because the median forecast is flat or high.")
    if latest_error is not None and mean_error is not None and (latest_error > 0) != (mean_error > 0):
        parts.append("Latest error differs from the recent bias, so treat the forecast as unstable and lean more on current price action.")
    return " ".join(parts)


def summarize_forecast_records(records: list[dict[str, Any]]) -> str:
    matured = [record for record in records if record.get("status") == "matured"]
    if not matured:
        pending = sum(1 for record in records if record.get("status") == "pending")
        return f"No matured forecast points yet. Pending forecast points: {pending}."
    abs_errors = [_float(record.get("abs_error")) for record in matured]
    abs_errors = [value for value in abs_errors if value is not None]
    direction = [record for record in matured if record.get("direction_correct") is not None]
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else None
    directional_accuracy = sum(1 for record in direction if record.get("direction_correct")) / len(direction) if direction else None
    bias_values = [_float(record.get("error")) for record in matured]
    bias_values = [value for value in bias_values if value is not None]
    bias = sum(bias_values) / len(bias_values) if bias_values else None
    return (
        f"Matured forecasts: {len(matured)}. "
        f"MAE: {_fmt(mae)}. Bias actual-minus-predicted: {_fmt(bias)}. "
        f"Directional accuracy: {_fmt_pct(directional_accuracy)}."
    )


def _read_ledger(path: Path, currency: str) -> dict[str, Any]:
    if not path.exists():
        return {"status": "ok", "currency": currency.upper(), "records": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "ok", "currency": currency.upper(), "records": []}
    if not isinstance(payload, dict):
        return {"status": "ok", "currency": currency.upper(), "records": []}
    return payload


def _append_current_forecasts(records: list[dict[str, Any]], forecast: dict[str, Any]) -> list[dict[str, Any]]:
    points = forecast.get("preferred_horizon_points") or forecast.get("horizon_points") or []
    if not isinstance(points, list):
        return records
    as_of_timestamp = str(forecast.get("as_of_timestamp") or "")
    as_of_price = _float(forecast.get("as_of_price"))
    source = str(forecast.get("preferred_source") or "chronos_median")
    existing = {str(record.get("forecast_id")) for record in records}
    for point in points:
        if not isinstance(point, dict):
            continue
        horizon = _float(point.get("horizon_hours"))
        target = str(point.get("timestamp") or "")
        predicted = _float(point.get("predicted_price"))
        if horizon is None or not target or predicted is None:
            continue
        forecast_id = f"{as_of_timestamp}|{target}|{horizon:.6f}|{source}"
        if forecast_id in existing:
            continue
        records.append(
            {
                "forecast_id": forecast_id,
                "status": "pending",
                "source": source,
                "as_of_timestamp": as_of_timestamp,
                "as_of_price": as_of_price,
                "horizon_hours": horizon,
                "target_timestamp": target,
                "predicted_price": predicted,
                "lower_price": _float(point.get("lower_price")),
                "upper_price": _float(point.get("upper_price")),
                "predicted_direction": point.get("direction"),
                "scalp_bias": point.get("scalp_bias"),
                "tape_direction": point.get("tape_direction"),
            }
        )
        existing.add(forecast_id)
    return records


def _mature_records(records: list[dict[str, Any]], price_bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bars = _clean_bars(price_bars)
    if not bars:
        return records
    latest_ts = bars[-1][0]
    matured: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") == "matured":
            matured.append(record)
            continue
        target = _parse_timestamp(record.get("target_timestamp"))
        if target is None or target > latest_ts:
            matured.append(record)
            continue
        actual = _actual_at_or_after(bars, target)
        if actual is None:
            matured.append(record)
            continue
        actual_ts, actual_price = actual
        predicted = _float(record.get("predicted_price"))
        as_of_price = _float(record.get("as_of_price"))
        error = None if predicted is None else actual_price - predicted
        abs_error = None if error is None else abs(error)
        direction_correct = None
        if predicted is not None and as_of_price is not None:
            predicted_move = predicted - as_of_price
            actual_move = actual_price - as_of_price
            if abs(predicted_move) < max(0.01, abs(as_of_price) * 0.00005):
                direction_correct = abs(actual_move) < max(0.01, abs(as_of_price) * 0.0003)
            else:
                direction_correct = (predicted_move >= 0 and actual_move >= 0) or (predicted_move < 0 and actual_move < 0)
        matured.append(
            {
                **record,
                "status": "matured",
                "actual_timestamp": actual_ts.isoformat(),
                "actual_price": actual_price,
                "error": error,
                "abs_error": abs_error,
                "pct_error": None if predicted in (None, 0) else error / predicted,
                "direction_correct": direction_correct,
            }
        )
    return matured


def _clean_bars(price_bars: list[dict[str, Any]]) -> list[tuple[datetime, float]]:
    rows: list[tuple[datetime, float]] = []
    for row in price_bars:
        if not isinstance(row, dict):
            continue
        ts = _parse_timestamp(row.get("timestamp"))
        close = _float(row.get("close"))
        if ts is not None and close is not None:
            rows.append((ts, close))
    rows.sort(key=lambda item: item[0])
    return rows


def _actual_at_or_after(bars: list[tuple[datetime, float]], target: datetime) -> tuple[datetime, float] | None:
    for ts, close in bars:
        if ts >= target:
            return ts, close
    return None


def _by_horizon(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = f"{_float(record.get('horizon_hours')) or 0:g}h"
        grouped.setdefault(key, []).append(record)
    output: dict[str, dict[str, Any]] = {}
    for key, rows in grouped.items():
        abs_errors = [_float(row.get("abs_error")) for row in rows]
        abs_errors = [value for value in abs_errors if value is not None]
        directions = [row for row in rows if row.get("direction_correct") is not None]
        output[key] = {
            "count": len(rows),
            "mae": None if not abs_errors else sum(abs_errors) / len(abs_errors),
            "directional_accuracy": None if not directions else sum(1 for row in directions if row.get("direction_correct")) / len(directions),
            "latest": rows[-1] if rows else None,
        }
    return output


def _parse_timestamp(value: Any) -> datetime | None:
    try:
        parsed = pd.Timestamp(value).to_pydatetime()
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"
