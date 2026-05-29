from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_TECHNICAL_KEYS = (
    "trend_state",
    "dow_theory",
    "magee_basing_points",
    "chapter_17_governance_context",
    "chapter_18_tactical_problem",
    "decision_diagnostics",
)

REQUIRED_PLOT_KEYS = (
    "forecast_plotly",
    "technical_chart_plotly",
    "technical_clean_chart_plotly",
    "technical_daily_chart_plotly",
    "technical_weekly_chart_plotly",
    "technical_monthly_chart_plotly",
)


def analyze_chapter_19_validation(
    report: dict[str, Any],
    prices: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    target_column: str = "close",
) -> dict[str, Any]:
    """Validate the completed run as an operational routine.

    Chapter 19 is about repeatable charting practice, reliable data, clean
    records, and using computers as a disciplined workflow. This validation
    layer checks whether the report can be audited before it is used.
    """

    target = target_column.lower()
    input_action = _input_action(report)
    routines = {
        "data_routine": _data_routine(report, prices=prices, target_column=target),
        "forecast_routine": _forecast_routine(report),
        "technical_routine": _technical_routine(report),
        "artifact_routine": _artifact_routine(report, output_dir=output_dir),
        "governance_routine": _governance_routine(report),
    }
    checks = [check for routine in routines.values() for check in routine["checks"]]
    failures = [check for check in checks if check["status"] == "fail"]
    warnings = [check for check in checks if check["status"] == "warn"]
    not_requested = [check for check in checks if check["status"] == "not_requested"]
    hard_blocks = [check for check in failures if check.get("blocks_new_commitments")]
    status = "fail" if failures else "warn" if warnings else "pass"
    state = {
        "pass": "OperationallyReady",
        "warn": "ReadyWithWarnings",
        "fail": "NotReady",
    }[status]
    validated_action = "Hold" if hard_blocks and input_action in {"Buy", "Sell"} else input_action
    action_override = validated_action != input_action

    validation = {
        "principle": (
            "Chapter 19 validates the routine: data source, chart/update discipline, artifact records, "
            "portfolio accounting readiness, and reproducible governance before a trade decision is trusted."
        ),
        "state": state,
        "status": status,
        "decision_policy": {
            "mode": "conditional_operational_validation_gate",
            "influences_final_action": bool(action_override),
            "intended_consumer": "pipeline_and_human_reviewer",
            "reason": (
                "The validation layer is normally report-only, but a non-auditable run blocks fresh Buy/Sell commitments."
            ),
        },
        "action_gate": {
            "input_action": input_action,
            "validated_action": validated_action,
            "action_override": action_override,
            "hard_block_new_commitments": bool(hard_blocks),
            "hard_block_reasons": [str(check["message"]) for check in hard_blocks],
            "warning_reasons": [str(check["message"]) for check in warnings[:12]],
        },
        "routine_summary": {
            "total_checks": len(checks),
            "passed": sum(1 for check in checks if check["status"] == "pass"),
            "warnings": len(warnings),
            "failures": len(failures),
            "not_requested": len(not_requested),
            "output_dir": str(output_dir) if output_dir is not None else None,
        },
        **routines,
        "review_checklist": _review_checklist(routines),
        "previous_run_comparison": {
            "status": "not_supplied",
            "message": "No previous run was supplied to this single-run validation step.",
        },
        "technical_method_card": chapter_19_validation_method_card(target_column=target),
    }
    return validation


def apply_chapter_19_validation(
    report: dict[str, Any],
    prices: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    target_column: str = "close",
) -> dict[str, Any]:
    validation = analyze_chapter_19_validation(
        report,
        prices=prices,
        output_dir=output_dir,
        target_column=target_column,
    )
    action_gate = validation["action_gate"]
    input_action = action_gate["input_action"]
    validated_action = action_gate["validated_action"]

    if report.get("suggested_action") != validated_action:
        report.setdefault("pre_chapter_19_suggested_action", input_action)
        report["suggested_action"] = validated_action

    decision_view = report.setdefault("decision_view", {})
    decision_view["chapter_19_validation"] = validation
    decision_view["chapter_19_action_gate"] = action_gate
    decision_view["final_operational_action"] = validated_action

    report.setdefault("operations_view", {})["chapter_19_validation"] = validation
    report.setdefault("technical_view", {})["chapter_19_validation"] = validation
    report.setdefault("diagnostics", {})["chapter_19_validation"] = validation
    report.setdefault("governance", {}).setdefault("operational_method_cards", {})[
        "chapter_19_validation"
    ] = validation["technical_method_card"]
    return validation


def chapter_19_validation_method_card(target_column: str = "close") -> dict[str, Any]:
    return {
        "name": "edwards_magee_chapter_19_operational_validation",
        "version": "chapter_19_validation_v1",
        "target_column": target_column.lower(),
        "decision_policy": "conditional_hold_gate_only_on_non_auditable_runs",
        "implemented_controls": [
            "data_source_and_price_integrity_check",
            "forecast_validation_record_check",
            "technical_context_completeness_check",
            "artifact_existence_check_when_output_dir_is_used",
            "governance_record_check",
            "review_checklist",
        ],
        "chapter_19_alignment": [
            "regular_systematic_routine",
            "reliable_data_source",
            "charts_and_records_must_be_kept_up_to_date",
            "computer_as_tool_not_substitute_for_discipline",
            "portfolio_accounting_and_mark_to_market_readiness",
        ],
        "non_goal": "This validator does not create a technical signal or forecast price.",
    }


def _data_routine(
    report: dict[str, Any],
    prices: pd.DataFrame | None,
    target_column: str,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    data_quality = report.get("diagnostics", {}).get("data_quality", {})
    data_manifest = report.get("data_manifest", {})
    config = report.get("governance", {}).get("config", {})

    _add_check(
        checks,
        "data_quality_report_present",
        bool(data_quality),
        "Data-quality report is present.",
        "Data-quality report is missing.",
        fail_severity="high",
        blocks=True,
    )
    quality_status = data_quality.get("status")
    if quality_status == "fail":
        _add_status_check(
            checks,
            "data_quality_status",
            "fail",
            "Data-quality status is fail.",
            "high",
            blocks=True,
        )
    elif quality_status == "warn":
        _add_status_check(checks, "data_quality_status", "warn", "Data-quality status has warnings.", "medium")
    elif quality_status == "pass":
        _add_status_check(checks, "data_quality_status", "pass", "Data-quality status is pass.", "low")

    _add_check(
        checks,
        "data_manifest_present",
        bool(data_manifest),
        "Data manifest is present.",
        "Data manifest is missing.",
        fail_severity="high",
        blocks=True,
    )
    _add_check(
        checks,
        "data_version_present",
        bool(report.get("data_version")),
        "Data version hash is present.",
        "Data version hash is missing.",
        fail_severity="high",
        blocks=True,
    )

    if prices is None:
        _add_status_check(checks, "price_frame_available", "not_requested", "Raw price frame was not supplied.", "low")
    else:
        target_present = target_column in {str(column).lower() for column in prices.columns}
        _add_check(
            checks,
            "target_column_present",
            target_present,
            f"Target column `{target_column}` is present.",
            f"Target column `{target_column}` is missing.",
            fail_severity="high",
            blocks=True,
        )
        row_count = int(len(prices))
        min_rows = int(config.get("min_training_rows", 0) or 0)
        _add_check(
            checks,
            "enough_price_rows",
            row_count >= max(1, min_rows),
            f"Price frame has {row_count} rows.",
            f"Price frame has only {row_count} rows.",
            fail_severity="high",
            blocks=True,
        )
        if isinstance(prices.index, pd.DatetimeIndex):
            _add_check(
                checks,
                "price_index_monotonic",
                bool(prices.index.is_monotonic_increasing),
                "Price index is monotonic.",
                "Price index is not monotonic.",
                fail_severity="medium",
            )
            _add_check(
                checks,
                "price_index_unique",
                not bool(prices.index.duplicated().any()),
                "Price index has unique timestamps.",
                "Price index has duplicate timestamps.",
                fail_severity="high",
                blocks=True,
            )
        latest_price = _finite_or_none(report.get("current_price"))
        _add_check(
            checks,
            "current_price_valid",
            latest_price is not None and latest_price > 0,
            "Current price is finite and positive.",
            "Current price is missing, non-finite, or non-positive.",
            fail_severity="high",
            blocks=True,
        )

    return _routine("data_routine", checks)


def _forecast_routine(report: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    forecasts = report.get("forecasts", [])
    horizons = [str(item.get("horizon_days")) for item in forecasts if isinstance(item, dict)]
    backtests = report.get("backtests", {})
    candidate_results = report.get("candidate_results", {})
    selected_records = report.get("diagnostics", {}).get("selected_validation_predictions", {})

    _add_check(
        checks,
        "forecasts_present",
        bool(forecasts),
        f"{len(forecasts)} horizon forecasts are present.",
        "No horizon forecasts are present.",
        fail_severity="high",
        blocks=True,
    )
    for forecast in forecasts:
        horizon = str(forecast.get("horizon_days"))
        _add_check(
            checks,
            f"horizon_{horizon}_selected_model",
            bool(forecast.get("selected_model")),
            f"Horizon {horizon} has a selected model.",
            f"Horizon {horizon} is missing a selected model.",
            fail_severity="high",
            blocks=True,
        )
        metrics = forecast.get("validation_metrics", {})
        required_metrics = {"mae", "rmse", "directional_accuracy", "holdout_mae"}
        missing_metrics = sorted(required_metrics - set(metrics))
        _add_check(
            checks,
            f"horizon_{horizon}_validation_metrics",
            not missing_metrics,
            f"Horizon {horizon} has required validation metrics.",
            f"Horizon {horizon} is missing validation metrics: {', '.join(missing_metrics)}.",
            fail_severity="high",
            blocks=True,
        )
        confidence = _finite_or_none(forecast.get("directional_confidence"))
        _add_check(
            checks,
            f"horizon_{horizon}_confidence_valid",
            confidence is not None and 0.0 <= confidence <= 1.0,
            f"Horizon {horizon} has a finite confidence.",
            f"Horizon {horizon} confidence is invalid.",
            fail_severity="medium",
        )

    for horizon in horizons:
        _add_check(
            checks,
            f"horizon_{horizon}_candidate_results",
            bool(candidate_results.get(horizon)),
            f"Horizon {horizon} candidate validation records are present.",
            f"Horizon {horizon} candidate validation records are missing.",
            fail_severity="high",
            blocks=True,
        )
        _add_check(
            checks,
            f"horizon_{horizon}_backtest",
            bool(backtests.get(horizon)),
            f"Horizon {horizon} signal backtest is present.",
            f"Horizon {horizon} signal backtest is missing.",
            fail_severity="medium",
        )
        _add_check(
            checks,
            f"horizon_{horizon}_validation_predictions",
            bool(selected_records.get(horizon)),
            f"Horizon {horizon} selected validation predictions are present.",
            f"Horizon {horizon} selected validation predictions are missing.",
            fail_severity="medium",
        )

    return _routine("forecast_routine", checks)


def _technical_routine(report: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    technical_view = report.get("technical_view", {})
    decision_view = report.get("decision_view", {})

    for key in REQUIRED_TECHNICAL_KEYS:
        _add_check(
            checks,
            f"{key}_present",
            bool(technical_view.get(key)),
            f"`technical_view.{key}` is present.",
            f"`technical_view.{key}` is missing.",
            fail_severity="high" if key == "chapter_18_tactical_problem" else "medium",
            blocks=key == "chapter_18_tactical_problem",
        )
    _add_check(
        checks,
        "decision_view_present",
        bool(decision_view),
        "Decision view is present.",
        "Decision view is missing.",
        fail_severity="high",
        blocks=True,
    )
    _add_check(
        checks,
        "chapter_18_safety_gate_present",
        bool(decision_view.get("llm_safety_gate")),
        "Chapter 18 safety gate is present.",
        "Chapter 18 safety gate is missing.",
        fail_severity="high",
        blocks=True,
    )
    _add_check(
        checks,
        "suggested_action_valid",
        _input_action(report) in {"Buy", "Hold", "Sell"},
        "Suggested action is a valid action.",
        "Suggested action is not valid.",
        fail_severity="high",
        blocks=True,
    )

    return _routine("technical_routine", checks)


def _artifact_routine(report: dict[str, Any], output_dir: str | Path | None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    artifacts = report.get("artifacts", {})
    plots = artifacts.get("plots", {}) if isinstance(artifacts.get("plots"), dict) else {}

    if output_dir is None and not artifacts:
        _add_status_check(
            checks,
            "artifact_generation",
            "not_requested",
            "No output directory was supplied, so artifact validation is skipped.",
            "low",
        )
        return _routine("artifact_routine", checks)

    for key in REQUIRED_PLOT_KEYS:
        path = plots.get(key)
        if not path:
            _add_status_check(checks, f"{key}_artifact", "warn", f"Plot artifact `{key}` is not recorded.", "medium")
            continue
        _add_file_check(checks, f"{key}_artifact", path, blocks=False)

    for key in ("forecast_report", "model_card", "data_manifest", "data_quality"):
        path = artifacts.get(key)
        if path:
            _add_file_check(checks, f"{key}_artifact", path, blocks=key == "forecast_report")
        elif key in {"forecast_report", "model_card"}:
            _add_status_check(checks, f"{key}_artifact", "warn", f"Audit artifact `{key}` is not recorded yet.", "medium")

    return _routine("artifact_routine", checks)


def _governance_routine(report: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    governance = report.get("governance", {})
    model_cards = governance.get("model_cards", {})
    config = governance.get("config", {})
    method_cards = governance.get("technical_method_cards", {})
    tactical_cards = governance.get("tactical_method_cards", {})

    _add_check(
        checks,
        "config_present",
        bool(config),
        "Run config is recorded.",
        "Run config is missing.",
        fail_severity="high",
        blocks=True,
    )
    _add_check(
        checks,
        "model_cards_present",
        bool(model_cards),
        "Model cards are recorded.",
        "Model cards are missing.",
        fail_severity="high",
        blocks=True,
    )
    _add_check(
        checks,
        "feature_registry_present",
        bool(governance.get("feature_registry")),
        "Feature registry is recorded.",
        "Feature registry is missing.",
        fail_severity="medium",
    )
    _add_check(
        checks,
        "chapter_17_method_card_present",
        bool(method_cards.get("chapter_17_governance_context")),
        "Chapter 17 method card is recorded.",
        "Chapter 17 method card is missing.",
        fail_severity="medium",
    )
    _add_check(
        checks,
        "chapter_18_method_card_present",
        bool(tactical_cards.get("chapter_18_tactical_problem")),
        "Chapter 18 tactical method card is recorded.",
        "Chapter 18 tactical method card is missing.",
        fail_severity="high",
        blocks=True,
    )

    return _routine("governance_routine", checks)


def _review_checklist(routines: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "item": "data_source_available",
            "status": _check_status(routines["data_routine"], "data_manifest_present"),
            "reason": "The run records where the data came from and how it was prepared.",
        },
        {
            "item": "charts_generated_or_intentionally_skipped",
            "status": routines["artifact_routine"]["status"],
            "reason": "Computer charting is part of the repeatable routine when an output directory is used.",
        },
        {
            "item": "validation_records_present",
            "status": routines["forecast_routine"]["status"],
            "reason": "Forecasts must be backed by validation, backtest, and candidate records.",
        },
        {
            "item": "technical_decision_packet_complete",
            "status": routines["technical_routine"]["status"],
            "reason": "Chapter 17 and Chapter 18 packets must be present before interpretation.",
        },
        {
            "item": "governance_records_present",
            "status": routines["governance_routine"]["status"],
            "reason": "Config, model cards, and method cards make the run reproducible.",
        },
        {
            "item": "portfolio_accounting_inputs",
            "status": "not_supplied",
            "reason": "Single-ticker runs do not include shares, cost basis, account cash, or tax constraints.",
        },
    ]


def _routine(name: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    status = "fail" if any(check["status"] == "fail" for check in checks) else "warn" if any(check["status"] == "warn" for check in checks) else "pass"
    if checks and all(check["status"] == "not_requested" for check in checks):
        status = "not_requested"
    return {
        "name": name,
        "status": status,
        "checks": checks,
    }


def _add_check(
    checks: list[dict[str, Any]],
    name: str,
    condition: bool,
    pass_message: str,
    fail_message: str,
    fail_severity: str = "medium",
    blocks: bool = False,
) -> None:
    if condition:
        _add_status_check(checks, name, "pass", pass_message, "low", blocks=False)
    else:
        _add_status_check(checks, name, "fail", fail_message, fail_severity, blocks=blocks)


def _add_status_check(
    checks: list[dict[str, Any]],
    name: str,
    status: str,
    message: str,
    severity: str,
    blocks: bool = False,
) -> None:
    checks.append(
        {
            "name": name,
            "status": status,
            "severity": severity,
            "message": message,
            "blocks_new_commitments": bool(blocks and status == "fail"),
        }
    )


def _add_file_check(checks: list[dict[str, Any]], name: str, path: str | Path, blocks: bool) -> None:
    file_path = Path(path)
    exists = file_path.exists() and file_path.is_file() and file_path.stat().st_size > 0
    _add_check(
        checks,
        name,
        exists,
        f"Artifact `{file_path}` exists.",
        f"Artifact `{file_path}` is missing or empty.",
        fail_severity="high" if blocks else "medium",
        blocks=blocks,
    )


def _check_status(routine: dict[str, Any], name: str) -> str:
    for check in routine.get("checks", []):
        if check.get("name") == name:
            return str(check.get("status"))
    return "unknown"


def _input_action(report: dict[str, Any]) -> str:
    decision_view = report.get("decision_view", {})
    action = (
        decision_view.get("final_governed_action")
        or report.get("pre_chapter_19_suggested_action")
        or report.get("suggested_action")
        or "Hold"
    )
    return str(action) if str(action) in {"Buy", "Hold", "Sell"} else "Hold"


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None
