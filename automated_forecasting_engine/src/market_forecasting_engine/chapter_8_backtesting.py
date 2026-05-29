from __future__ import annotations

from typing import Any

import numpy as np


def analyze_chapter_8_backtesting(
    *,
    backtests: dict[str, dict[str, Any]],
    candidate_results: dict[str, list[dict[str, Any]]],
    selected_validation_predictions: dict[str, list[dict[str, Any]]],
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    """Audit ML4T backtest realism without adding a full event-driven engine."""

    horizon_reports = {}
    pass_count = 0
    for horizon in horizons:
        key = str(horizon)
        backtest = backtests.get(key, {})
        candidates = candidate_results.get(key, [])
        records = selected_validation_predictions.get(key, [])
        audit = _realism_audit(
            backtest=backtest,
            candidate_count=len(candidates),
            validation_rows=len(records),
            horizon_days=int(horizon),
        )
        if audit["status"] == "pass":
            pass_count += 1
        horizon_reports[key] = {
            "horizon_days": int(horizon),
            "backtest_type": "vectorized_validation_signal_backtest",
            "realism_audit": audit,
            "cost_slippage_sensitivity": backtest.get("slippage_sensitivity", []),
            "trade_ledger_summary": backtest.get("trade_ledger_summary", {}),
            "timing_policy": _timing_policy(backtest),
            "minimum_backtest_length": _minimum_backtest_length(
                validation_rows=len(records),
                candidate_count=len(candidates),
            ),
        }

    promotion_policy = _promotion_policy(horizon_count=len(horizons), pass_count=pass_count)
    return {
        "chapter": 8,
        "name": "The ML4T Workflow - From Model to Strategy Backtesting",
        "status": "pass" if promotion_policy["candidate_for_promotion"] else "warn",
        "decision_policy": {
            "influences_final_action": False,
            "influences_model_selection": True,
            "mode": "model_selection_adjustment",
            "reason": "Chapter 8 validation backtest diagnostics penalize candidates with poor signal risk before model selection; event-driven infrastructure is still future work.",
        },
        "engine_policy": {
            "zipline": "not_installed",
            "backtrader": "not_installed",
            "reason": "The current pipeline uses an auditable vectorized validation backtest; event-driven engines remain optional future infrastructure.",
        },
        "horizons": horizon_reports,
        "promotion_policy": promotion_policy,
        "technical_method_card": chapter_8_backtesting_method_card(),
    }


def chapter_8_backtesting_method_card() -> dict[str, Any]:
    return {
        "name": "jansen_ml4t_chapter_8_backtesting",
        "version": "chapter_8_backtesting_v1",
        "source": "Machine Learning for Algorithmic Trading, 2nd ed., Chapter 8",
        "purpose": "Audit signal-to-trade simulation realism, costs, timing, turnover, and multiple-testing pressure.",
        "decision_policy": "model_selection_adjustment",
        "implemented_components": [
            "vectorized_backtest_realism_audit",
            "signal_execution_timing_policy",
            "cost_and_slippage_sensitivity",
            "trade_ledger_summary",
            "minimum_backtest_length_warning",
            "candidate_count_multiple_testing_pressure",
        ],
        "not_implemented": [
            "Zipline_runtime_dependency",
            "Backtrader_runtime_dependency",
            "live_order_execution_adapter",
        ],
    }


def _realism_audit(
    *,
    backtest: dict[str, Any],
    candidate_count: int,
    validation_rows: int,
    horizon_days: int,
) -> dict[str, Any]:
    warnings = []
    turnover = _number(backtest.get("turnover"))
    rows = int(_number(backtest.get("rows")) or validation_rows)
    worst_slippage = _worst_slippage_row(backtest.get("slippage_sensitivity", []))
    if rows < max(60, candidate_count * 8):
        warnings.append("Validation backtest is short relative to the number of tested candidates.")
    if turnover > 0.75:
        warnings.append("Turnover is high; transaction costs and slippage can dominate edge.")
    if not backtest.get("slippage_sensitivity"):
        warnings.append("Slippage sensitivity is missing from this backtest record.")
    if worst_slippage and _number(worst_slippage.get("sharpe_ratio")) <= 0:
        warnings.append("Strategy Sharpe is not robust under the highest slippage stress.")
    execution_timing = str(backtest.get("execution_timing", "missing"))
    if execution_timing.startswith("vectorized_same_period") or execution_timing == "missing":
        warnings.append("Backtest is vectorized and does not prove next-open executable performance.")
    if horizon_days > 1 and rows < 100:
        warnings.append("Multi-day horizon has limited validation observations.")
    status = "pass" if not warnings else "warn"
    return {
        "status": status,
        "warnings": warnings,
        "validation_rows": int(rows),
        "candidate_count": int(candidate_count),
        "turnover": _finite(turnover),
        "highest_slippage_stress": worst_slippage,
        "interpretation": (
            "Backtest realism checks passed for this horizon."
            if status == "pass"
            else "Treat this as research evidence, not a deployable event-driven simulation."
        ),
    }


def _timing_policy(backtest: dict[str, Any]) -> dict[str, Any]:
    execution_timing = str(backtest.get("execution_timing", "unknown"))
    return {
        "execution_timing": execution_timing,
        "signal_timestamp_assumption": "validation prediction is known at the validation timestamp",
        "execution_assumption": (
            "same-period close-to-close research approximation"
            if execution_timing.startswith("vectorized_same_period")
            else "explicit execution timing supplied by backtest"
        ),
        "required_for_promotion": [
            "next_open_or_next_close_execution_prices",
            "slippage_model_by_liquidity",
            "event_order_ledger",
            "mark_to_market_position_accounting",
        ],
    }


def _minimum_backtest_length(validation_rows: int, candidate_count: int) -> dict[str, Any]:
    minimum_rows = max(60, int(candidate_count * 8))
    ratio = validation_rows / max(minimum_rows, 1)
    return {
        "validation_rows": int(validation_rows),
        "candidate_count": int(candidate_count),
        "minimum_rows_rule_of_thumb": int(minimum_rows),
        "coverage_ratio": _finite(ratio),
        "status": "adequate" if validation_rows >= minimum_rows else "short",
        "reason": "Controls multiple-testing pressure from trying many candidate models on the same sample.",
    }


def _promotion_policy(*, horizon_count: int, pass_count: int) -> dict[str, Any]:
    candidate = horizon_count > 0 and pass_count == horizon_count
    return {
        "status": "candidate_for_event_driven_research" if candidate else "diagnostic_only",
        "candidate_for_promotion": bool(candidate),
        "realism_pass_horizon_count": int(pass_count),
        "tested_horizon_count": int(horizon_count),
        "recommended_action": (
            "Research next-open/event-driven simulation before using backtest results for action gates."
            if candidate
            else "Keep Chapter 8 report-only; current evidence is research-grade rather than execution-grade."
        ),
    }


def _worst_slippage_row(rows: Any) -> dict[str, Any] | None:
    if not isinstance(rows, list) or not rows:
        return None
    valid = [row for row in rows if isinstance(row, dict)]
    if not valid:
        return None
    return max(valid, key=lambda row: _number(row.get("slippage_bps")))


def _number(value: Any) -> float:
    try:
        number = float(value if value is not None else 0.0)
    except Exception:
        return 0.0
    return number if np.isfinite(number) else 0.0


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0
