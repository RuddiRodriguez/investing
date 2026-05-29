from __future__ import annotations

from typing import Any

import numpy as np


def analyze_chapter_39_43_discipline_governance(report: dict[str, Any]) -> dict[str, Any]:
    """Build discipline/governance controls for Chapters 39 and 43."""

    trial_error = _chapter_39_trial_error(report)
    plan_discipline = _chapter_43_stick_to_plan(report)
    gate = _discipline_gate(report, trial_error=trial_error, plan_discipline=plan_discipline)
    status = "review_needed" if gate["warnings"] or gate["hard_blocks"] else "consistent"

    return {
        "principle": (
            "Chapters 39 and 43 treat trading as a recorded discipline: test rules through trial and error, "
            "then follow the accepted plan instead of changing methods after one uncomfortable result."
        ),
        "state": "ManualReviewNeeded" if status == "review_needed" else "PlanConsistent",
        "status": status,
        "decision_policy": {
            "mode": "discipline_governance_report_only",
            "influences_final_action": False,
            "intended_consumer": "human_or_llm_discipline_reviewer",
            "reason": "This layer audits consistency and process discipline; it warns but does not rewrite the forecast action.",
        },
        "chapter_39_trial_error": trial_error,
        "chapter_43_stick_to_plan": plan_discipline,
        "discipline_gate": gate,
        "review_triggers": _review_triggers(report, gate=gate),
        "llm_integration": {
            "status": "planned",
            "note": (
                "A later LLM discipline reviewer can explain rule deviations and suggest review steps, "
                "but it must not justify breaking stops, validation gates, sizing limits, or portfolio capital gates."
            ),
        },
        "technical_method_card": chapter_39_43_discipline_governance_method_card(),
    }


def apply_chapter_39_43_discipline_governance(report: dict[str, Any]) -> dict[str, Any]:
    discipline = analyze_chapter_39_43_discipline_governance(report)
    report.setdefault("discipline_view", {})["chapter_39_43_discipline_governance"] = discipline
    report.setdefault("technical_view", {})["chapter_39_43_discipline_governance"] = discipline
    report.setdefault("diagnostics", {})["chapter_39_43_discipline_governance"] = discipline
    report.setdefault("governance", {}).setdefault("discipline_method_cards", {})[
        "chapter_39_43_discipline_governance"
    ] = discipline["technical_method_card"]
    return discipline


def chapter_39_43_discipline_governance_method_card() -> dict[str, Any]:
    return {
        "name": "edwards_magee_chapters_39_43_discipline_governance",
        "version": "chapter_39_43_discipline_v1",
        "decision_policy": "report_only_discipline_audit_no_action_override",
        "implemented_controls": [
            "chapter_39_trial_error_evidence_record",
            "weak_horizon_and_backtest_review",
            "method_change_requires_logged_evidence",
            "chapter_43_plan_adherence_check",
            "capital_gate_and_trade_plan_consistency_check",
            "review_triggers_for_recalibration_or_manual_review",
        ],
        "chapter_alignment": [
            "use_trial_and_error_as_recorded_evidence_not_impulse",
            "do_not_change_methods_after_one_bad_trade_or_run",
            "follow_the_plan_after_rules_are_defined",
            "separate_forecast_opinion_from_capital_permission",
            "document_exceptions_before_action",
        ],
        "future_llm_integration": (
            "A later LLM can explain discipline exceptions only after rule-based validation, trade/risk, and capital gates run."
        ),
    }


def _chapter_39_trial_error(report: dict[str, Any]) -> dict[str, Any]:
    governance = report.get("governance", {})
    diagnostics = report.get("diagnostics", {})
    model_cards = governance.get("model_cards", {})
    backtests = report.get("backtests", {})
    candidate_results = report.get("candidate_results", {})
    validation_predictions = diagnostics.get("selected_validation_predictions", {})
    weak_horizons = _weak_backtest_horizons(backtests)
    evidence_status = "measured" if model_cards and backtests and validation_predictions else "incomplete"
    return {
        "status": evidence_status,
        "evidence_counts": {
            "model_cards": len(model_cards) if isinstance(model_cards, dict) else 0,
            "candidate_result_horizons": len(candidate_results) if isinstance(candidate_results, dict) else 0,
            "backtest_horizons": len(backtests) if isinstance(backtests, dict) else 0,
            "validation_prediction_horizons": len(validation_predictions) if isinstance(validation_predictions, dict) else 0,
        },
        "validation_design": diagnostics.get("validation_design", {}),
        "weak_horizons": weak_horizons,
        "method_change_policy": {
            "rule": "Do not alter models, filters, stops, or selection rules because of one disliked result.",
            "requires": [
                "logged_before_after_backtests",
                "ablation_or_holdout_evidence",
                "unchanged_data_version_or_explicit_data_change_note",
                "method_card_version_update",
            ],
            "current_recommendation": (
                "Review weak horizons, but keep the current method unless repeated evidence supports a rule change."
                if weak_horizons
                else "No method-change trigger from the available validation/backtest record."
            ),
        },
        "trial_error_record": {
            "data_version": report.get("data_version"),
            "model_version": report.get("model_version"),
            "selection_metric": report.get("selection_metric"),
            "suggested_action": report.get("suggested_action"),
            "risk_level": report.get("risk_level"),
        },
    }


def _chapter_43_stick_to_plan(report: dict[str, Any]) -> dict[str, Any]:
    chapter_18 = report.get("decision_view", {}).get("chapter_18_tactical_problem", {})
    chapter_19 = report.get("operations_view", {}).get("chapter_19_validation", {})
    chapter_21 = report.get("selection_view", {}).get("chapter_21_chart_selection", {})
    trade_risk = report.get("trade_risk_view", {}).get("chapter_23_30_trade_risk_plan", {})
    portfolio_risk = report.get("portfolio_view", {}).get("chapter_31_42_portfolio_capital_risk", {})
    suggested_action = report.get("suggested_action")
    chapter_18_final = chapter_18.get("final_action") or report.get("decision_view", {}).get("final_governed_action")
    chapter_19_gate = chapter_19.get("action_gate", {})
    chapter_19_validated = chapter_19_gate.get("validated_action")
    chapter_21_bucket = chapter_21.get("chart_selection", {}).get("chart_book_bucket")
    commitment = trade_risk.get("commitment", {})
    portfolio_gate = portfolio_risk.get("portfolio_capital_gate", {})
    warnings = _plan_warnings(
        suggested_action=suggested_action,
        chapter_18_final=chapter_18_final,
        chapter_19=chapter_19,
        chapter_19_validated=chapter_19_validated,
        chapter_21_bucket=chapter_21_bucket,
        commitment_type=commitment.get("commitment_type"),
        portfolio_allocation_status=portfolio_gate.get("allocation_status"),
    )
    return {
        "status": "needs_manual_review" if warnings else "consistent",
        "plan_adherence": "needs_manual_review" if warnings else "consistent",
        "current_plan_state": {
            "suggested_action": suggested_action,
            "chapter_18_final_action": chapter_18_final,
            "chapter_19_validated_action": chapter_19_validated,
            "chapter_19_status": chapter_19.get("status"),
            "chapter_21_bucket": chapter_21_bucket,
            "trade_risk_commitment_type": commitment.get("commitment_type"),
            "trade_risk_entry_plan": commitment.get("entry_plan"),
            "portfolio_allocation_status": portfolio_gate.get("allocation_status"),
            "portfolio_capital_state": portfolio_gate.get("state"),
        },
        "warnings": warnings,
        "discipline_policy": (
            "Execute only the plan that survives validation, chart selection, trade/risk controls, and portfolio capital gates. "
            "When those layers disagree, pause for review instead of improvising."
        ),
    }


def _discipline_gate(
    report: dict[str, Any],
    trial_error: dict[str, Any],
    plan_discipline: dict[str, Any],
) -> dict[str, Any]:
    state = plan_discipline.get("current_plan_state", {})
    action = state.get("suggested_action") or report.get("suggested_action")
    commitment_type = state.get("trade_risk_commitment_type")
    portfolio_status = state.get("portfolio_allocation_status")
    chapter_19_status = state.get("chapter_19_status")
    warnings = list(plan_discipline.get("warnings", []))
    hard_blocks = []
    if chapter_19_status == "fail":
        hard_blocks.append("Chapter 19 validation failed; do not allocate new capital.")
    if portfolio_status == "blocked_pending_inputs":
        hard_blocks.append("Portfolio capital gate is blocked until account/diversification inputs are supplied.")
    if trial_error.get("status") == "incomplete":
        warnings.append("Trial/error evidence is incomplete; review model and backtest records before changing rules.")
    if action in {"Buy", "Sell"} and commitment_type in {
        "no_new_commitment",
        "active_review_no_new_commitment",
        "watchlist_no_new_commitment",
        "monitor_only",
    }:
        warnings.append("Forecast action is directional, but the trade/risk layer does not permit new capital commitment.")
    new_capital_policy = "do_not_allocate" if hard_blocks else "no_new_capital_commitment"
    if not hard_blocks and commitment_type in {"candidate_long_commitment", "candidate_short_commitment"}:
        new_capital_policy = "may_allocate_only_if_portfolio_gate_is_ready"
    return {
        "status": "blocked" if hard_blocks else "review" if warnings else "clear",
        "plan_adherence": "needs_manual_review" if hard_blocks or warnings else "consistent",
        "new_capital_policy": new_capital_policy,
        "hard_blocks": hard_blocks,
        "warnings": _unique(warnings),
        "rule": "Warnings require review; hard blocks prohibit new capital allocation. The final forecast action is not overwritten here.",
    }


def _plan_warnings(
    suggested_action: Any,
    chapter_18_final: Any,
    chapter_19: dict[str, Any],
    chapter_19_validated: Any,
    chapter_21_bucket: Any,
    commitment_type: Any,
    portfolio_allocation_status: Any,
) -> list[str]:
    warnings = []
    action = str(suggested_action) if suggested_action is not None else None
    if chapter_19_validated and action != chapter_19_validated:
        warnings.append("Final suggested action does not match Chapter 19 validated action.")
    if chapter_18_final and action != chapter_18_final and not chapter_19.get("action_gate", {}).get("action_override"):
        warnings.append("Final suggested action differs from Chapter 18 without a Chapter 19 override.")
    if chapter_19.get("status") == "fail" and action in {"Buy", "Sell"}:
        warnings.append("Chapter 19 failed but final action remains directional.")
    if chapter_21_bucket == "excluded" and action in {"Buy", "Sell"}:
        warnings.append("Chapter 21 excludes this ticker while the final action remains directional.")
    if commitment_type == "candidate_long_commitment" and action != "Buy":
        warnings.append("Trade/risk plan is a long candidate but final action is not Buy.")
    if commitment_type == "candidate_short_commitment" and action != "Sell":
        warnings.append("Trade/risk plan is a short candidate but final action is not Sell.")
    if portfolio_allocation_status == "blocked_pending_inputs" and commitment_type in {
        "candidate_long_commitment",
        "candidate_short_commitment",
    }:
        warnings.append("Trade/risk layer permits a candidate commitment, but portfolio capital inputs are incomplete.")
    return _unique(warnings)


def _weak_backtest_horizons(backtests: Any) -> list[dict[str, Any]]:
    if not isinstance(backtests, dict):
        return []
    weak = []
    for horizon, metrics in backtests.items():
        if not isinstance(metrics, dict):
            continue
        reasons = []
        cumulative_return = _finite_or_none(metrics.get("cumulative_return"))
        sharpe_ratio = _finite_or_none(metrics.get("sharpe_ratio"))
        hit_rate = _finite_or_none(metrics.get("hit_rate"))
        max_drawdown = _finite_or_none(metrics.get("max_drawdown"))
        if cumulative_return is not None and cumulative_return < 0:
            reasons.append("negative_validation_strategy_return")
        if sharpe_ratio is not None and sharpe_ratio < 0:
            reasons.append("negative_validation_sharpe")
        if hit_rate is not None and hit_rate < 0.45:
            reasons.append("hit_rate_below_45pct")
        if max_drawdown is not None and max_drawdown <= -0.20:
            reasons.append("drawdown_worse_than_20pct")
        if reasons:
            weak.append(
                {
                    "horizon": str(horizon),
                    "reasons": reasons,
                    "cumulative_return": _round(cumulative_return),
                    "sharpe_ratio": _round(sharpe_ratio),
                    "hit_rate": _round(hit_rate),
                    "max_drawdown": _round(max_drawdown),
                }
            )
    return weak


def _review_triggers(report: dict[str, Any], gate: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "trigger": "new_price_data",
            "action": "Rerun the full pipeline before changing the action.",
            "active": True,
        },
        {
            "trigger": "chapter_19_fail",
            "action": "Fix audit/data/artifact issues before allocating capital.",
            "active": report.get("operations_view", {}).get("chapter_19_validation", {}).get("status") == "fail",
        },
        {
            "trigger": "portfolio_gate_blocked",
            "action": "Supply account equity, current position value, and diversification context.",
            "active": gate.get("new_capital_policy") == "do_not_allocate",
        },
        {
            "trigger": "stop_or_invalidation_hit",
            "action": "Follow the stop plan; do not widen risk after entry.",
            "active": False,
        },
        {
            "trigger": "support_resistance_or_trendline_break",
            "action": "Review Chapters 13-15 and update trade/risk plan.",
            "active": False,
        },
        {
            "trigger": "method_change_request",
            "action": "Require logged before/after backtests and method-card version change.",
            "active": False,
        },
    ]


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _round(value: Any, digits: int = 4) -> float | None:
    numeric = _finite_or_none(value)
    if numeric is None:
        return None
    return round(numeric, digits)


def _unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
