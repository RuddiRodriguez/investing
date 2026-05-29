from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def build_model_card(
    ticker: str,
    horizon_days: int,
    selected_model: str,
    selected_model_family: str,
    model_parameters: dict[str, Any],
    training_window: dict[str, Any],
    feature_columns: list[str],
    selection_metric: str,
    validation_metrics: dict[str, float],
    confidence_interval: dict[str, float],
    data_version: str,
    model_version: str,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "horizon_days": horizon_days,
        "selected_model": selected_model,
        "selected_model_family": selected_model_family,
        "model_parameters": model_parameters,
        "training_window": training_window,
        "feature_set": feature_columns,
        "selection_metric": selection_metric,
        "validation_metrics": validation_metrics,
        "confidence_interval": confidence_interval,
        "data_version": data_version,
        "model_version": model_version,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }


def write_audit_bundle(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output_path = Path(output_dir)
    governance_path = output_path / "governance"
    governance_path.mkdir(parents=True, exist_ok=True)

    report_file = output_path / "forecast_report.json"
    model_card_file = governance_path / f"model_card_{report['ticker']}.json"
    data_manifest_file = governance_path / f"data_manifest_{report['ticker']}.json"
    data_quality_file = governance_path / f"data_quality_{report['ticker']}.json"

    artifacts = {
        "forecast_report": str(report_file),
        "model_card": str(model_card_file),
    }
    if "data_manifest" in report:
        artifacts["data_manifest"] = str(data_manifest_file)
    data_quality = report.get("diagnostics", {}).get("data_quality")
    if data_quality:
        artifacts["data_quality"] = str(data_quality_file)

    report.setdefault("artifacts", {}).update(artifacts)
    _write_json(report_file, report)
    _write_json(model_card_file, report["governance"]["model_cards"])
    if "data_manifest" in report:
        _write_json(data_manifest_file, report["data_manifest"])
    if data_quality:
        _write_json(data_quality_file, data_quality)
    return artifacts


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
