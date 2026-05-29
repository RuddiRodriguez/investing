from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_CHART_ARTIFACTS = (
    "forecast_plotly",
    "technical_chart_plotly",
    "technical_clean_chart_plotly",
    "technical_daily_chart_plotly",
    "technical_weekly_chart_plotly",
    "technical_monthly_chart_plotly",
)


def analyze_chapter_21_chart_selection(report: dict[str, Any]) -> dict[str, Any]:
    """Select whether a completed ticker belongs in the active chart list.

    Chapter 21 consumes the completed forecast report, especially Chapter 18,
    Chapter 19, and Chapter 20. It does not create a stronger trade action.
    """

    chapter_20 = report.get("selection_view", {}).get("chapter_20_ticker_suitability", {})
    profile_fit = chapter_20.get("profile_fit", {})
    component_scores = chapter_20.get("component_scores", {})
    chapter_19 = report.get("operations_view", {}).get("chapter_19_validation", {})
    chart_artifacts = _chart_artifact_readiness(report)
    score = _selection_priority_score(
        report=report,
        profile_fit=profile_fit,
        component_scores=component_scores,
        chapter_19=chapter_19,
        chart_artifacts=chart_artifacts,
    )
    selection = _chart_selection(report, profile_fit=profile_fit, chapter_19=chapter_19, priority_score=score)
    review_plan = _review_plan(selection, report=report, chart_artifacts=chart_artifacts)

    return {
        "principle": (
            "Chapter 21 turns completed ticker analysis into chart-list selection. "
            "It decides what deserves active review, watchlist attention, monitoring, or exclusion."
        ),
        "state": selection["state"],
        "status": selection["status"],
        "decision_policy": {
            "mode": "chart_selection_report_only",
            "influences_final_action": False,
            "intended_consumer": "chapter_22_diversification_and_human_or_llm_reviewer",
            "reason": "Chapter 21 selects chart-book attention; it does not override Chapter 18 action or Chapter 19 validation.",
        },
        "chart_selection": selection,
        "priority_score": score,
        "selection_inputs": {
            "chapter_20_primary_profile": profile_fit.get("primary_profile"),
            "chapter_20_classification": profile_fit.get("classification"),
            "chapter_20_selection_hint": profile_fit.get("selection_hint"),
            "chapter_20_suitability_score": _round(profile_fit.get("suitability_score")),
            "chapter_19_status": chapter_19.get("status"),
            "suggested_action": report.get("suggested_action"),
            "risk_level": report.get("risk_level"),
        },
        "chart_artifact_readiness": chart_artifacts,
        "review_plan": review_plan,
        "chart_book_row": _chart_book_row(report, selection=selection, score=score, review_plan=review_plan),
        "rule_interpretation": _rule_interpretation(selection, score=score, profile_fit=profile_fit, chapter_19=chapter_19),
        "chapter_22_readiness": {
            "status": "requires_universe_context",
            "message": "Chapter 21 single-ticker selection is ready; Chapter 22 must compare sector, correlation, and concentration across tickers.",
        },
        "llm_integration": {
            "status": "planned",
            "note": (
                "A later LLM reviewer can compare multiple Chapter 21 chart-book rows and explain the active list. "
                "It should not promote excluded or non-auditable reports into trade candidates."
            ),
        },
        "technical_method_card": chapter_21_chart_selection_method_card(),
    }


def apply_chapter_21_chart_selection(report: dict[str, Any]) -> dict[str, Any]:
    selection = analyze_chapter_21_chart_selection(report)
    report.setdefault("selection_view", {})["chapter_21_chart_selection"] = selection
    report.setdefault("technical_view", {})["chapter_21_chart_selection"] = selection
    report.setdefault("diagnostics", {})["chapter_21_chart_selection"] = selection
    report.setdefault("governance", {}).setdefault("selection_method_cards", {})[
        "chapter_21_chart_selection"
    ] = selection["technical_method_card"]
    return selection


def chapter_21_chart_selection_method_card() -> dict[str, Any]:
    return {
        "name": "edwards_magee_chapter_21_chart_selection",
        "version": "chapter_21_chart_book_selection_v1",
        "decision_policy": "report_only_chart_book_selection_no_trade_override",
        "implemented_controls": [
            "chapter_20_suitability_consumption",
            "chapter_19_operational_gate_consumption",
            "chapter_18_tactical_action_consumption",
            "chart_artifact_readiness_check",
            "chart_book_bucket_assignment",
            "review_cadence_assignment",
            "chapter_22_diversification_handoff",
        ],
        "chapter_21_alignment": [
            "select_stocks_to_chart_after_analysis_exists",
            "separate_chart_attention_from_trade_execution",
            "keep_active_chart_list_focused",
            "use_watchlist_and_monitor_buckets_for_incomplete_or_weaker_cases",
            "prepare_multi_ticker_selection_for_diversification_review",
        ],
        "future_llm_integration": (
            "A later LLM layer may rank and explain chart-book candidates across a universe, "
            "but it must stay subordinate to Chapter 18 tactical and Chapter 19 operational gates."
        ),
    }


def _selection_priority_score(
    report: dict[str, Any],
    profile_fit: dict[str, Any],
    component_scores: dict[str, Any],
    chapter_19: dict[str, Any],
    chart_artifacts: dict[str, Any],
) -> dict[str, Any]:
    suitability = _finite_or_default(profile_fit.get("suitability_score"), 0.0)
    operational = {"pass": 1.0, "warn": 0.75, "fail": 0.0}.get(str(chapter_19.get("status")), 0.50)
    tactical = _finite_or_default(component_scores.get("tactical_readiness", {}).get("score"), _tactical_fallback(report))
    urgency = _selection_urgency(report, profile_fit)
    artifact = _finite_or_default(chart_artifacts.get("score"), 0.50)
    score = (
        0.40 * suitability
        + 0.20 * operational
        + 0.20 * tactical
        + 0.10 * urgency
        + 0.10 * artifact
    )
    return {
        "score": _round(score),
        "level": "High" if score >= 0.70 else "Medium" if score >= 0.50 else "Low",
        "components": {
            "chapter_20_suitability": _round(suitability),
            "chapter_19_operational": _round(operational),
            "chapter_18_tactical_readiness": _round(tactical),
            "selection_urgency": _round(urgency),
            "chart_artifact_readiness": _round(artifact),
        },
        "weights": {
            "chapter_20_suitability": 0.40,
            "chapter_19_operational": 0.20,
            "chapter_18_tactical_readiness": 0.20,
            "selection_urgency": 0.10,
            "chart_artifact_readiness": 0.10,
        },
    }


def _chart_selection(
    report: dict[str, Any],
    profile_fit: dict[str, Any],
    chapter_19: dict[str, Any],
    priority_score: dict[str, Any],
) -> dict[str, Any]:
    score = float(priority_score.get("score") or 0.0)
    action = str(report.get("suggested_action") or "Hold")
    hint = str(profile_fit.get("selection_hint") or "avoid_for_now")
    chapter_19_gate = chapter_19.get("action_gate", {})
    if chapter_19.get("status") == "fail" or chapter_19_gate.get("hard_block_new_commitments"):
        return _selection_payload(
            state="Excluded",
            status="exclude",
            bucket="excluded",
            action="ExcludeForNow",
            reason="Chapter 19 validation did not clear the operational gate.",
            active=False,
            trade_candidate=False,
            watchlist=False,
        )
    if hint == "avoid_for_now":
        return _selection_payload(
            state="Excluded",
            status="exclude",
            bucket="excluded",
            action="ExcludeForNow",
            reason="Chapter 20 suitability says avoid for now.",
            active=False,
            trade_candidate=False,
            watchlist=False,
        )
    if bool(profile_fit.get("trade_candidate_eligible")) and action in {"Buy", "Sell"} and score >= 0.60:
        return _selection_payload(
            state="Selected",
            status="trade_candidate",
            bucket="trade_candidates",
            action="AddToActiveChartBook",
            reason="Directional action and Chapter 20 trade eligibility justify active chart-book review.",
            active=True,
            trade_candidate=True,
            watchlist=True,
        )
    if bool(profile_fit.get("active_review_eligible")) and score >= 0.55:
        return _selection_payload(
            state="Selected",
            status="active_review",
            bucket="active_review",
            action="AddToActiveChartBook",
            reason="Ticker is suitable for active chart review even though it is not a fresh trade candidate.",
            active=True,
            trade_candidate=False,
            watchlist=True,
        )
    if bool(profile_fit.get("watchlist_eligible")) and score >= 0.40:
        return _selection_payload(
            state="Watchlist",
            status="watchlist",
            bucket="watchlist",
            action="KeepInWatchlist",
            reason="Ticker is suitable enough to keep watching but not strong enough for active chart review.",
            active=False,
            trade_candidate=False,
            watchlist=True,
        )
    if score >= 0.30:
        return _selection_payload(
            state="Monitor",
            status="monitor_only",
            bucket="monitor_only",
            action="MonitorOnly",
            reason="Ticker has weak but non-zero suitability; monitor only.",
            active=False,
            trade_candidate=False,
            watchlist=True,
        )
    return _selection_payload(
        state="Excluded",
        status="exclude",
        bucket="excluded",
        action="ExcludeForNow",
        reason="Priority score is too weak for chart-book attention.",
        active=False,
        trade_candidate=False,
        watchlist=False,
    )


def _selection_payload(
    state: str,
    status: str,
    bucket: str,
    action: str,
    reason: str,
    active: bool,
    trade_candidate: bool,
    watchlist: bool,
) -> dict[str, Any]:
    return {
        "state": state,
        "status": status,
        "chart_book_bucket": bucket,
        "chart_book_action": action,
        "active_chart_book": active,
        "trade_candidate": trade_candidate,
        "watchlist": watchlist,
        "reason": reason,
    }


def _review_plan(
    selection: dict[str, Any],
    report: dict[str, Any],
    chart_artifacts: dict[str, Any],
) -> dict[str, Any]:
    status = selection.get("status")
    if status == "trade_candidate":
        cadence = "daily_until_resolved"
        timeframes = ["daily", "weekly", "monthly"]
    elif status == "active_review":
        cadence = "daily_or_twice_weekly"
        timeframes = ["daily", "weekly", "monthly"]
    elif status == "watchlist":
        cadence = "weekly"
        timeframes = ["weekly", "daily"]
    elif status == "monitor_only":
        cadence = "monthly_or_after_material_move"
        timeframes = ["monthly", "weekly"]
    else:
        cadence = "rerun_only_after_new_evidence"
        timeframes = []
    return {
        "review_cadence": cadence,
        "required_timeframes": timeframes,
        "required_artifacts": list(EXPECTED_CHART_ARTIFACTS) if status in {"trade_candidate", "active_review", "watchlist"} else [],
        "artifact_status": chart_artifacts.get("status"),
        "latest_action": report.get("suggested_action"),
    }


def _chart_book_row(
    report: dict[str, Any],
    selection: dict[str, Any],
    score: dict[str, Any],
    review_plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ticker": report.get("ticker"),
        "as_of_date": report.get("as_of_date"),
        "current_price": report.get("current_price"),
        "suggested_action": report.get("suggested_action"),
        "risk_level": report.get("risk_level"),
        "chart_book_bucket": selection.get("chart_book_bucket"),
        "priority_score": score.get("score"),
        "priority_level": score.get("level"),
        "review_cadence": review_plan.get("review_cadence"),
        "reason": selection.get("reason"),
    }


def _chart_artifact_readiness(report: dict[str, Any]) -> dict[str, Any]:
    artifacts = report.get("artifacts", {})
    plots = artifacts.get("plots", {}) if isinstance(artifacts.get("plots"), dict) else {}
    if not plots:
        return {
            "status": "not_generated",
            "score": 0.50,
            "available": [],
            "missing": list(EXPECTED_CHART_ARTIFACTS),
            "reason": "No plot artifact bundle is recorded; this is expected for in-memory runs without an output directory.",
        }
    available = []
    missing = []
    empty = []
    for key in EXPECTED_CHART_ARTIFACTS:
        path = plots.get(key)
        if not path:
            missing.append(key)
            continue
        file_path = Path(path)
        if file_path.exists() and file_path.is_file() and file_path.stat().st_size > 0:
            available.append(key)
        else:
            empty.append(key)
    score = len(available) / len(EXPECTED_CHART_ARTIFACTS)
    status = "complete" if score == 1.0 else "partial" if available else "missing"
    return {
        "status": status,
        "score": _round(score),
        "available": available,
        "missing": missing,
        "empty_or_missing_files": empty,
        "reason": "Chart artifact readiness checks whether the expected Plotly charts were recorded and written.",
    }


def _selection_urgency(report: dict[str, Any], profile_fit: dict[str, Any]) -> float:
    action = str(report.get("suggested_action") or "Hold")
    hint = str(profile_fit.get("selection_hint") or "")
    if action in {"Buy", "Sell"} and hint == "active_review":
        return 1.0
    if action in {"Buy", "Sell"}:
        return 0.75
    if hint == "active_review":
        return 0.65
    if hint == "keep_watching":
        return 0.45
    return 0.25


def _tactical_fallback(report: dict[str, Any]) -> float:
    action = str(report.get("suggested_action") or "Hold")
    if action in {"Buy", "Sell"}:
        return 0.70
    return 0.40


def _rule_interpretation(
    selection: dict[str, Any],
    score: dict[str, Any],
    profile_fit: dict[str, Any],
    chapter_19: dict[str, Any],
) -> list[str]:
    notes = [
        f"Chapter 21 bucket is {selection.get('chart_book_bucket')}.",
        f"Priority level is {score.get('level')} with score {score.get('score')}.",
        f"Chapter 20 selection hint was {profile_fit.get('selection_hint')}.",
    ]
    if chapter_19.get("status") == "warn":
        notes.append("Chapter 19 has warnings, so the ticker can be followed but should remain auditable.")
    if selection.get("status") == "trade_candidate":
        notes.append("Trade-candidate status still requires Chapter 22 diversification and portfolio checks before allocation.")
    return notes


def _finite_or_default(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _round(value: Any, digits: int = 4) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return round(result, digits)
