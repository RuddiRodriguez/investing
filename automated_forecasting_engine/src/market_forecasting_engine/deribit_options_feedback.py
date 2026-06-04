from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


def update_feedback_loop(
    *,
    output_dir: Path,
    currency: str,
    now: datetime,
    actual_price: float | None,
    min_matured: int = 5,
    min_direction_accuracy: float = 0.45,
    max_abs_pct_error: float = 0.06,
    window: int = 50,
) -> dict[str, Any]:
    ledger_path = _ledger_path(output_dir, currency)
    rows = _read_jsonl(ledger_path)
    changed = _normalize_horizon_target_times(rows)
    if actual_price is not None and actual_price > 0:
        for row in rows:
            for horizon in row.get("horizons") or []:
                if horizon.get("matured"):
                    continue
                target_time = _parse_time(horizon.get("target_time"))
                as_of_price = _float_or_none(horizon.get("as_of_price"))
                predicted = _float_or_none(horizon.get("predicted_price"))
                if target_time is None or target_time > now or as_of_price is None or predicted is None:
                    continue
                expected_move = predicted - as_of_price
                actual_move = float(actual_price) - as_of_price
                horizon["matured"] = True
                horizon["matured_at"] = now.isoformat()
                horizon["actual_price"] = float(actual_price)
                horizon["actual_return"] = actual_move / max(abs(as_of_price), 1e-12)
                horizon["predicted_return"] = expected_move / max(abs(as_of_price), 1e-12)
                horizon["abs_pct_error"] = abs(float(actual_price) - predicted) / max(abs(as_of_price), 1e-12)
                horizon["direction_correct"] = _direction_correct(expected_move, actual_move)
                changed = True
    if changed:
        _write_jsonl(ledger_path, rows)
    metrics = summarize_feedback(rows, window=window)
    blocks = []
    if metrics["matured_horizon_count"] >= int(min_matured):
        direction_accuracy = metrics.get("direction_accuracy")
        avg_abs_pct_error = metrics.get("avg_abs_pct_error")
        if direction_accuracy is not None and direction_accuracy < float(min_direction_accuracy):
            blocks.append("feedback_direction_accuracy_below_min")
        if avg_abs_pct_error is not None and avg_abs_pct_error > float(max_abs_pct_error):
            blocks.append("feedback_price_error_above_max")
    return {
        "enabled": True,
        "ledger_path": str(ledger_path),
        "updated_matured_horizons": changed,
        "metrics": metrics,
        "blocks": blocks,
        "policy": {
            "min_matured": int(min_matured),
            "min_direction_accuracy": float(min_direction_accuracy),
            "max_abs_pct_error": float(max_abs_pct_error),
            "window": int(window),
        },
    }


def append_decision_to_ledger(*, output_dir: Path, currency: str, record: dict[str, Any]) -> dict[str, Any]:
    path = _ledger_path(output_dir, currency)
    selected = record.get("selected_forecast") or {}
    plan = record.get("forecast_plan") or {}
    checked_at = record.get("checked_at") or datetime.now(UTC).isoformat()
    as_of_price = _float_or_none(selected.get("spot")) or _float_or_none(plan.get("latest_price"))
    decision_id = f"{currency.upper()}-{_compact_time(checked_at)}"
    row = {
        "decision_id": decision_id,
        "checked_at": checked_at,
        "currency": currency.upper(),
        "venue": record.get("venue"),
        "forecast_created_at_utc": record.get("forecast_created_at_utc"),
        "as_of_price": as_of_price,
        "forecast_direction": selected.get("expected_direction"),
        "trade_action": (record.get("option_trade_plan") or {}).get("action"),
        "trade_reason": (record.get("option_trade_plan") or {}).get("reason"),
        "contract": ((record.get("option_trade_plan") or {}).get("selected_contract") or {}).get("instrument_name"),
        "option_type": (record.get("option_trade_plan") or {}).get("option_type"),
        "execution_blocks": record.get("execution_blocks") or [],
        "order_submitted": (record.get("order_result") or {}).get("submitted"),
        "order_state": (((record.get("order_result") or {}).get("order") or {}).get("order") or {}).get("order_state"),
        "feedback_context": record.get("feedback_context") or {},
        "horizons": _forecast_horizons(
            plan=plan,
            selected=selected,
            as_of_price=as_of_price,
            forecast_created_at_utc=record.get("forecast_created_at_utc") or checked_at,
        ),
        "trade_pl_base": _trade_pl_base(record),
        "fees_base": _fees_base(record),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")
    return {"decision_id": decision_id, "ledger_path": str(path), "horizon_count": len(row["horizons"])}


def summarize_feedback(rows: list[dict[str, Any]], *, window: int = 50) -> dict[str, Any]:
    matured = []
    for row in rows:
        for horizon in row.get("horizons") or []:
            if horizon.get("matured"):
                matured.append({**horizon, "decision_id": row.get("decision_id"), "checked_at": row.get("checked_at")})
    matured = matured[-max(1, int(window)) :]
    direction_values = [bool(item.get("direction_correct")) for item in matured if item.get("direction_correct") is not None]
    abs_errors = [_float_or_none(item.get("abs_pct_error")) for item in matured]
    abs_errors = [value for value in abs_errors if value is not None and np.isfinite(value)]
    submitted = [row for row in rows if row.get("order_submitted")]
    trade_pl = [_float_or_none(row.get("trade_pl_base")) or 0.0 for row in rows]
    fees = [_float_or_none(row.get("fees_base")) or 0.0 for row in rows]
    return {
        "decision_count": len(rows),
        "submitted_decision_count": len(submitted),
        "matured_horizon_count": len(matured),
        "direction_accuracy": None if not direction_values else sum(direction_values) / len(direction_values),
        "avg_abs_pct_error": None if not abs_errors else sum(abs_errors) / len(abs_errors),
        "median_abs_pct_error": None if not abs_errors else float(np.median(abs_errors)),
        "trade_pl_base_sum": round(sum(trade_pl), 8),
        "fees_base_sum": round(sum(fees), 8),
        "last_matured": matured[-5:],
    }


def _normalize_horizon_target_times(rows: list[dict[str, Any]]) -> bool:
    changed = False
    for row in rows:
        created_at = _parse_time(row.get("forecast_created_at_utc") or row.get("checked_at"))
        if created_at is None:
            continue
        for horizon in row.get("horizons") or []:
            horizon_hours = _float_or_none(horizon.get("horizon_hours"))
            target_time = _target_time_from_horizon(created_at=created_at, horizon_hours=horizon_hours)
            if target_time is None:
                continue
            target_iso = target_time.isoformat()
            existing_target = horizon.get("target_time")
            if existing_target != target_iso:
                if existing_target is not None and horizon.get("payload_target_time") is None:
                    horizon["payload_target_time"] = _iso_or_raw(existing_target)
                horizon["target_time"] = target_iso
                horizon["target_time_source"] = "forecast_created_at_plus_horizon_hours"
                changed = True
            elif horizon.get("target_time_source") is None:
                horizon["target_time_source"] = "forecast_created_at_plus_horizon_hours"
                changed = True
    return changed


def _forecast_horizons(
    *,
    plan: dict[str, Any],
    selected: dict[str, Any],
    as_of_price: float | None,
    forecast_created_at_utc: Any,
) -> list[dict[str, Any]]:
    forecasts = plan.get("forecasts") or []
    if not forecasts and selected:
        forecasts = [selected]
    rows = []
    created_at = _parse_time(forecast_created_at_utc)
    for forecast in forecasts:
        predicted = _float_or_none(forecast.get("predicted_price"))
        horizon_hours = _float_or_none(forecast.get("horizon_hours"))
        target_time = _target_time_from_horizon(created_at=created_at, horizon_hours=horizon_hours)
        if target_time is None:
            target_time = _parse_time(forecast.get("forecast_timestamp") or forecast.get("forecast_date"))
        if target_time is None or predicted is None:
            continue
        rows.append(
            {
                "horizon_hours": horizon_hours,
                "target_time": target_time.isoformat(),
                "target_time_source": "forecast_created_at_plus_horizon_hours"
                if created_at is not None and horizon_hours is not None
                else "forecast_payload_timestamp",
                "payload_target_time": _iso_or_raw(forecast.get("forecast_timestamp") or forecast.get("forecast_date"))
                if (forecast.get("forecast_timestamp") or forecast.get("forecast_date")) is not None
                else None,
                "as_of_price": as_of_price,
                "predicted_price": predicted,
                "lower_price": _float_or_none(forecast.get("lower_price")),
                "upper_price": _float_or_none(forecast.get("upper_price")),
                "matured": False,
            }
        )
    return rows


def _target_time_from_horizon(*, created_at: datetime | None, horizon_hours: float | None) -> datetime | None:
    if created_at is None or horizon_hours is None:
        return None
    return created_at + timedelta(hours=float(horizon_hours))


def _trade_pl_base(record: dict[str, Any]) -> float:
    return sum(_float_or_none(trade.get("profit_loss")) or 0.0 for trade in _record_trades(record))


def _fees_base(record: dict[str, Any]) -> float:
    return sum(_float_or_none(trade.get("fee")) or 0.0 for trade in _record_trades(record))


def _record_trades(record: dict[str, Any]) -> list[dict[str, Any]]:
    trades = []
    order_result = record.get("order_result") or {}
    trades.extend((order_result.get("order") or {}).get("trades") or [])
    for action in record.get("management_actions") or []:
        trades.extend((action.get("order") or {}).get("trades") or [])
    return [trade for trade in trades if isinstance(trade, dict)]


def _ledger_path(output_dir: Path, currency: str) -> Path:
    return output_dir / "feedback" / f"{currency.upper()}_decision_ledger.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _iso_or_raw(value: Any) -> str:
    parsed = _parse_time(value)
    return str(value) if parsed is None else parsed.isoformat()


def _compact_time(value: Any) -> str:
    parsed = _parse_time(value)
    if parsed is None:
        return str(value).replace(":", "").replace("-", "")[:24]
    return parsed.strftime("%Y%m%d%H%M%S%f")


def _direction_correct(expected_move: float, actual_move: float) -> bool | None:
    if abs(expected_move) < 1e-12 or abs(actual_move) < 1e-12:
        return None
    return expected_move * actual_move > 0


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None
