from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def analyze_chapter_17_governance_context(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    forecasts: list[dict[str, Any]],
    raw_action: str,
    final_action: str,
    risk_level: str,
    decision_diagnostics: dict[str, Any],
    action_filters: dict[str, dict[str, Any]],
    technical_contexts: dict[str, dict[str, Any]],
    backtests: dict[str, dict[str, Any]],
    data_quality_report: dict[str, Any],
    technical_history_quality: dict[str, Any],
    market_feature_comparison: dict[str, Any],
    security_metadata: dict[str, Any] | None = None,
    data_manifest: dict[str, Any] | None = None,
    target_column: str = "close",
) -> dict[str, Any]:
    """Build Chapter 17 report-only meta-governance context.

    Chapter 17 is about using charts, computers, derivatives, portfolio tools,
    and discipline without treating any tool as an automatic decision maker.
    This module therefore creates context for review/LLM use and never mutates
    the engine action.
    """

    target = target_column.lower()
    preferred = _preferred_forecast(forecasts)
    evidence_matrix = _technical_evidence_matrix(preferred, raw_action, final_action, technical_contexts)
    filter_stack = _filter_stack_review(action_filters, raw_action=raw_action, final_action=final_action)
    method_conflict = _method_conflict_score(evidence_matrix)
    volume_context = _volume_context_summary(prices, target_column=target)
    portfolio_context = _portfolio_risk_context(features, security_metadata or {}, data_manifest or {})
    mark_to_market = _mark_to_market_context(prices, target_column=target)
    performance_ledger = _method_performance_ledger(backtests)
    fragility = _decision_fragility(
        preferred=preferred,
        raw_action=raw_action,
        final_action=final_action,
        risk_level=risk_level,
        decision_diagnostics=decision_diagnostics,
        filter_stack=filter_stack,
        method_conflict=method_conflict,
        data_quality_report=data_quality_report,
        technical_history_quality=technical_history_quality,
        market_feature_comparison=market_feature_comparison,
    )
    hedging_context = _hedging_context(
        final_action=final_action,
        risk_level=risk_level,
        portfolio_context=portfolio_context,
        method_conflict=method_conflict,
        technical_contexts=technical_contexts,
    )

    return {
        "principle": (
            "Chapter 17 treats computers, indicators, portfolio analysis, options, and futures as tools. "
            "The aim is disciplined interpretation, mark-to-market honesty, and loss control rather than automatic certainty."
        ),
        "state": "Measured",
        "decision_policy": {
            "mode": "report_only",
            "influences_final_action": False,
            "intended_consumer": "human_or_llm_decision_layer",
            "reason": "Chapter 17 packages context for interpretation but does not vote, block, or alter Buy/Hold/Sell decisions.",
        },
        "llm_decision_packet": _llm_decision_packet(
            preferred=preferred,
            raw_action=raw_action,
            final_action=final_action,
            risk_level=risk_level,
            decision_diagnostics=decision_diagnostics,
            fragility=fragility,
            method_conflict=method_conflict,
            filter_stack=filter_stack,
        ),
        "computer_humility": {
            "summary": "The model output is a decision aid, not a self-sufficient decision maker.",
            "decision_fragility": fragility,
            "method_conflict_score": method_conflict,
            "filter_stack_review": filter_stack,
            "anti_overfitting_notes": [
                "Treat extra indicators as evidence only when their validation and chart context agree.",
                "Avoid replacing one losing method with a newly fitted method only because recent trades were unfavorable.",
                "A series of losses can occur even when the long-run method is sound.",
            ],
        },
        "technical_evidence_matrix": evidence_matrix,
        "volume_context_summary": volume_context,
        "portfolio_risk_context": portfolio_context,
        "mark_to_market_discipline": mark_to_market,
        "method_performance_ledger": performance_ledger,
        "optional_hedging_context": hedging_context,
        "data_and_method_limitations": _data_and_method_limitations(
            data_quality_report=data_quality_report,
            technical_history_quality=technical_history_quality,
            market_feature_comparison=market_feature_comparison,
        ),
        "technical_method_card": chapter_17_governance_context_method_card(target_column=target),
    }


def chapter_17_governance_context_method_card(target_column: str = "close") -> dict[str, Any]:
    return {
        "name": "edwards_magee_chapter_17_governance_context",
        "version": "chapter_17_report_only_v1",
        "target_column": target_column.lower(),
        "decision_policy": "report_only_no_action_filter",
        "implemented_controls": [
            "llm_decision_packet",
            "decision_fragility_score",
            "filter_stack_review",
            "method_conflict_score",
            "volume_context_summary",
            "mark_to_market_discipline",
            "portfolio_risk_context",
            "method_performance_ledger",
            "optional_hedging_context",
            "computer_humility_notes",
        ],
        "chapter_17_alignment": [
            "computers_are_tools_not_decision_makers",
            "simple_methods_need_validation",
            "open_losses_must_be_marked_to_market",
            "portfolio_risk_and_correlation_matter",
            "derivatives_context_requires_special_training",
            "no_method_avoids_all_losses",
        ],
        "non_goal": "This method card does not define or apply a trading action filter.",
    }


def _preferred_forecast(forecasts: list[dict[str, Any]]) -> dict[str, Any]:
    if not forecasts:
        return {}
    return next((item for item in forecasts if int(item.get("horizon_days", 0)) == 5), forecasts[0])


def _llm_decision_packet(
    preferred: dict[str, Any],
    raw_action: str,
    final_action: str,
    risk_level: str,
    decision_diagnostics: dict[str, Any],
    fragility: dict[str, Any],
    method_conflict: dict[str, Any],
    filter_stack: dict[str, Any],
) -> dict[str, Any]:
    return {
        "engine_raw_action": raw_action,
        "engine_final_action": final_action,
        "risk_level": risk_level,
        "hold_reason": decision_diagnostics.get("hold_reason"),
        "preferred_horizon_days": _safe_int(preferred.get("horizon_days")),
        "preferred_expected_direction": preferred.get("expected_direction"),
        "preferred_expected_return": _finite_or_none(preferred.get("expected_return")),
        "preferred_directional_confidence": _finite_or_none(preferred.get("directional_confidence")),
        "edge_to_error_ratio": _finite_or_none(decision_diagnostics.get("edge_to_error_ratio")),
        "decision_fragility_level": fragility.get("level"),
        "method_conflict_level": method_conflict.get("level"),
        "filters_applied_count": filter_stack.get("applied_filter_count"),
        "top_blocking_reasons": list(decision_diagnostics.get("blocking_reasons", []))[:8],
        "top_supporting_reasons": list(decision_diagnostics.get("supporting_reasons", []))[:8],
        "missing_context_for_llm": [
            "portfolio weights and cost basis, unless supplied by a portfolio run",
            "tax constraints and liquidity needs",
            "actual options chain or futures hedge instruments",
            "human risk tolerance and maximum acceptable drawdown",
        ],
    }


def _decision_fragility(
    preferred: dict[str, Any],
    raw_action: str,
    final_action: str,
    risk_level: str,
    decision_diagnostics: dict[str, Any],
    filter_stack: dict[str, Any],
    method_conflict: dict[str, Any],
    data_quality_report: dict[str, Any],
    technical_history_quality: dict[str, Any],
    market_feature_comparison: dict[str, Any],
) -> dict[str, Any]:
    components: list[dict[str, Any]] = []

    confidence = _finite_or_none(preferred.get("directional_confidence")) or 0.0
    edge_ratio = _finite_or_none(decision_diagnostics.get("edge_to_error_ratio")) or 0.0
    if risk_level == "High":
        _add_component(components, "high_risk_level", 0.22, "Overall risk level is High.")
    elif risk_level == "Medium":
        _add_component(components, "medium_risk_level", 0.10, "Overall risk level is Medium.")
    if confidence < 0.55:
        _add_component(components, "weak_directional_confidence", 0.20, "Preferred forecast confidence is below the action gate.")
    elif confidence < 0.60:
        _add_component(components, "thin_directional_confidence", 0.10, "Preferred forecast confidence barely clears the action gate.")
    if edge_ratio < 1.0:
        _add_component(components, "expected_edge_below_error", 0.22, "Expected return is smaller than validation-error threshold.")
    elif edge_ratio < 1.5:
        _add_component(components, "thin_edge_vs_error", 0.10, "Expected edge is close to validation-error threshold.")
    if raw_action != final_action:
        _add_component(components, "action_changed_by_filters", 0.14, "Sequential governance changed the raw model action.")
    applied = int(filter_stack.get("applied_filter_count", 0) or 0)
    if applied >= 3:
        _add_component(components, "many_filters_applied", 0.14, "Several technical filters applied pressure to the model signal.")
    elif applied:
        _add_component(components, "filter_pressure", 0.06, "At least one technical filter applied pressure to the model signal.")
    conflict_score = _finite_or_none(method_conflict.get("score")) or 0.0
    if conflict_score >= 0.35:
        _add_component(components, "conflicting_methods", 0.16, "Directional technical evidence is materially split.")
    if data_quality_report.get("status") == "fail":
        _add_component(components, "data_quality_fail", 0.18, "Data quality has high-severity warnings.")
    elif data_quality_report.get("status") == "warn":
        _add_component(components, "data_quality_warn", 0.08, "Data quality has warning-level issues.")
    if not technical_history_quality.get("sufficient_for_classical_technical_analysis", False):
        _add_component(components, "technical_history_limited", 0.08, "Classical chart background is limited.")
    if market_feature_comparison.get("status") == "compared":
        degradation = _market_feature_degradation(market_feature_comparison)
        if degradation > 0.0:
            _add_component(components, "enriched_features_not_clearly_better", min(0.10, degradation), "External/enriched features did not clearly improve validation.")

    score = min(1.0, sum(float(item["weight"]) for item in components))
    level = "High" if score >= 0.55 else "Medium" if score >= 0.25 else "Low"
    return {
        "score": _finite_or_none(score),
        "level": level,
        "components": components,
        "interpretation": _fragility_interpretation(level),
    }


def _filter_stack_review(action_filters: dict[str, dict[str, Any]], raw_action: str, final_action: str) -> dict[str, Any]:
    rows = []
    for name, payload in action_filters.items():
        input_action = payload.get("raw_action", payload.get("input_action", raw_action))
        filtered_action = payload.get("filtered_action", input_action)
        rows.append(
            {
                "name": name,
                "input_action": input_action,
                "filtered_action": filtered_action,
                "filter_applied": bool(payload.get("filter_applied")),
                "status": payload.get("status") or payload.get("primary_trend") or payload.get("trend_state") or payload.get("pattern"),
                "blocking_reasons": list(payload.get("blocking_reasons", []))[:5],
                "warnings": list(payload.get("warnings", []))[:5],
            }
        )
    applied_count = sum(1 for row in rows if row["filter_applied"])
    return {
        "raw_action": raw_action,
        "final_action": final_action,
        "filter_count": int(len(rows)),
        "applied_filter_count": int(applied_count),
        "filter_overload_warning": bool(raw_action in {"Buy", "Sell"} and final_action == "Hold" and applied_count >= 3),
        "permanent_hold_risk": bool(final_action == "Hold" and applied_count >= 4),
        "filters": rows,
    }


def _technical_evidence_matrix(
    preferred: dict[str, Any],
    raw_action: str,
    final_action: str,
    technical_contexts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reference_direction = _forecast_direction(preferred)
    entries = [
        _entry("model_forecast", reference_direction, _finite_or_none(preferred.get("directional_confidence")), preferred.get("expected_direction"), "Preferred selected-model forecast."),
        _entry("engine_raw_action", _action_direction(raw_action), 0.70, raw_action, "Raw action before technical governance filters."),
        _entry("engine_final_action", _action_direction(final_action), 0.70, final_action, "Final action after technical governance filters."),
    ]
    entries.extend(_technical_context_entries(technical_contexts))
    for item in entries:
        direction = item.get("direction")
        item["supports_reference_direction"] = bool(reference_direction != "neutral" and direction == reference_direction)
        item["conflicts_reference_direction"] = bool(
            reference_direction != "neutral" and direction in {"bullish", "bearish"} and direction != reference_direction
        )
    return {
        "comparison_basis": "preferred_forecast_direction",
        "reference_direction": reference_direction,
        "entries": entries,
    }


def _technical_context_entries(contexts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    trend = contexts.get("trend_state", {})
    entries.append(_entry("trend_state", _state_direction(trend.get("state")), trend.get("confidence"), trend.get("state"), "Composite trend view."))
    dow = contexts.get("dow_theory", {})
    entries.append(_entry("dow_primary_trend", _state_direction(dow.get("primary_trend", {}).get("state")), dow.get("primary_trend", {}).get("confidence"), dow.get("primary_trend", {}).get("state"), "Dow-style primary trend."))
    magee = contexts.get("magee_basing_points", {}).get("preferred", {})
    entries.append(_entry("magee_basing_points", _long_short_direction(magee.get("trend_state")), magee.get("score"), magee.get("trend_state"), "Magee basing-points stop context."))
    reversal = contexts.get("reversal_patterns", {}).get("preferred", {})
    entries.append(_entry("reversal_patterns", _pattern_direction(reversal), reversal.get("score"), _pattern_label(reversal), "Major reversal pattern context."))
    triangle = contexts.get("triangle_patterns", {}).get("preferred", {})
    entries.append(_entry("triangle_patterns", _pattern_direction(triangle), triangle.get("score"), _pattern_label(triangle), "Triangle pattern context."))
    chapter_9 = contexts.get("chapter_9_patterns", {})
    entries.append(_entry("chapter_9_rectangle", _pattern_direction(chapter_9.get("rectangle_patterns", {}).get("preferred", {})), None, _pattern_label(chapter_9.get("rectangle_patterns", {}).get("preferred", {})), "Chapter 9 rectangle context."))
    entries.append(_entry("chapter_9_multi_top_bottom", _pattern_direction(chapter_9.get("multi_top_bottom_patterns", {}).get("preferred", {})), None, _pattern_label(chapter_9.get("multi_top_bottom_patterns", {}).get("preferred", {})), "Chapter 9 double/triple top-bottom context."))
    chapter_10 = contexts.get("chapter_10_patterns", {})
    entries.append(_entry("chapter_10_structural", _pattern_direction(chapter_10.get("structural_patterns", {}).get("preferred", {})), None, _pattern_label(chapter_10.get("structural_patterns", {}).get("preferred", {})), "Chapter 10 structural reversal context."))
    entries.append(_entry("chapter_10_event", _pattern_direction(chapter_10.get("short_term_events", {}).get("preferred", {})), None, _pattern_label(chapter_10.get("short_term_events", {}).get("preferred", {})), "Chapter 10 one-day event context."))
    chapter_11 = contexts.get("chapter_11_patterns", {})
    entries.append(_entry("chapter_11_continuation", _pattern_direction(chapter_11.get("continuation_patterns", {}).get("preferred", {})), None, _pattern_label(chapter_11.get("continuation_patterns", {}).get("preferred", {})), "Chapter 11 continuation context."))
    chapter_12 = contexts.get("chapter_12_gaps", {})
    entries.append(_entry("chapter_12_gap", _pattern_direction(chapter_12.get("classified_gaps", {}).get("preferred", {})), None, _pattern_label(chapter_12.get("classified_gaps", {}).get("preferred", {})), "Chapter 12 gap context."))
    chapter_13 = contexts.get("chapter_13_support_resistance", {})
    entries.append(_entry("chapter_13_support_resistance", _chapter_13_direction(chapter_13), None, chapter_13.get("preferred", {}).get("status"), "Chapter 13 support/resistance context."))
    chapter_14 = contexts.get("chapter_14_trendlines", {})
    entries.append(_entry("chapter_14_trendlines", _pattern_direction(chapter_14.get("trendlines", {}).get("preferred", {})), None, _pattern_label(chapter_14.get("trendlines", {}).get("preferred", {})), "Chapter 14 trendline context."))
    chapter_15 = contexts.get("chapter_15_major_trendlines", {}).get("stock_major_trend", {})
    entries.append(_entry("chapter_15_major_trendline", _pattern_direction(chapter_15.get("major_trendline", {})), chapter_15.get("major_trendline", {}).get("authority_score"), _pattern_label(chapter_15.get("major_trendline", {})), "Chapter 15 major trendline context."))
    chapter_16 = contexts.get("chapter_16_market_context", {})
    entries.append(_entry("chapter_16_donchian", _donchian_direction(chapter_16.get("donchian_context", {}).get("overall_state")), None, chapter_16.get("donchian_context", {}).get("overall_state"), "Chapter 16 report-only Donchian context."))
    return entries


def _method_conflict_score(evidence_matrix: dict[str, Any]) -> dict[str, Any]:
    entries = [item for item in evidence_matrix.get("entries", []) if item.get("direction") in {"bullish", "bearish"}]
    bullish = [item for item in entries if item.get("direction") == "bullish"]
    bearish = [item for item in entries if item.get("direction") == "bearish"]
    total = len(entries)
    score = min(len(bullish), len(bearish)) / total if total else 0.0
    level = "High" if score >= 0.35 else "Medium" if score >= 0.20 else "Low"
    return {
        "score": _finite_or_none(score),
        "level": level,
        "bullish_evidence_count": int(len(bullish)),
        "bearish_evidence_count": int(len(bearish)),
        "neutral_evidence_count": int(len(evidence_matrix.get("entries", [])) - total),
        "conflicting_evidence": [item["name"] for item in bullish if bearish] + [item["name"] for item in bearish if bullish],
        "interpretation": _conflict_interpretation(level),
    }


def _volume_context_summary(prices: pd.DataFrame, target_column: str) -> dict[str, Any]:
    if "volume" not in prices.columns:
        return {
            "status": "Unavailable",
            "reason": "volume column was not supplied",
            "chapter_17_note": "Volume is contextual and cannot be used as a standalone signal.",
        }
    close = pd.to_numeric(prices[target_column], errors="coerce")
    volume = pd.to_numeric(prices["volume"], errors="coerce")
    if volume.dropna().empty:
        return {"status": "Unavailable", "reason": "volume column has no numeric values"}
    volume_sma_20 = volume.rolling(20).mean()
    volume_sma_60 = volume.rolling(60).mean()
    volume_std_20 = volume.rolling(20).std()
    latest_volume = volume.iloc[-1]
    latest_to_sma = latest_volume / volume_sma_20.iloc[-1] - 1 if volume_sma_20.iloc[-1] else np.nan
    latest_z = (latest_volume - volume_sma_20.iloc[-1]) / volume_std_20.iloc[-1] if volume_std_20.iloc[-1] else np.nan
    return_20 = close.iloc[-1] / close.iloc[-21] - 1 if len(close) > 21 and close.iloc[-21] else np.nan
    volume_trend = (
        "Rising" if pd.notna(volume_sma_20.iloc[-1]) and pd.notna(volume_sma_60.iloc[-1]) and volume_sma_20.iloc[-1] > volume_sma_60.iloc[-1] * 1.10
        else "Falling" if pd.notna(volume_sma_20.iloc[-1]) and pd.notna(volume_sma_60.iloc[-1]) and volume_sma_20.iloc[-1] < volume_sma_60.iloc[-1] * 0.90
        else "Stable"
    )
    stage = _price_stage(close)
    interpretation = "Volume is normal relative to the recent 20-day baseline."
    if pd.notna(latest_z) and latest_z >= 2.0 and pd.notna(return_20):
        interpretation = "Volume is unusually high; read as breakout participation if price is advancing, or exhaustion/distribution risk if price is extended."
    elif pd.notna(latest_z) and latest_z <= -1.5:
        interpretation = "Volume is unusually quiet; pattern confirmation should be discounted."
    return {
        "status": "Measured",
        "latest_volume": _finite_or_none(latest_volume),
        "volume_to_sma_20": _finite_or_none(latest_to_sma),
        "volume_z_20": _finite_or_none(latest_z),
        "volume_trend_20_vs_60": volume_trend,
        "price_return_20d": _finite_or_none(return_20),
        "price_stage": stage,
        "interpretation": interpretation,
        "chapter_17_note": "Volume follows trend and is relative; it should confirm price action, not replace it.",
    }


def _portfolio_risk_context(
    features: pd.DataFrame,
    security_metadata: dict[str, Any],
    data_manifest: dict[str, Any],
) -> dict[str, Any]:
    latest = features.iloc[-1] if not features.empty else pd.Series(dtype=float)
    beta_columns = [column for column in features.columns if str(column).startswith("relative_") and str(column).endswith("_beta_63d")]
    beta_estimates = {
        str(column): _finite_or_none(latest.get(column))
        for column in beta_columns
        if _finite_or_none(latest.get(column)) is not None
    }
    portfolio_weight = _finite_or_none(security_metadata.get("portfolio_weight"))
    universe = data_manifest.get("universe", {}) if isinstance(data_manifest, dict) else {}
    missing_inputs = []
    if portfolio_weight is None:
        missing_inputs.append("position weight")
    if not beta_estimates:
        missing_inputs.append("benchmark beta or relative-strength context")
    missing_inputs.extend(["cost basis", "portfolio covariance matrix", "tax constraints"])
    return {
        "status": "Measured" if portfolio_weight is not None or beta_estimates else "Limited",
        "portfolio_weight": portfolio_weight,
        "beta_estimates": beta_estimates,
        "universe_context_available": bool(universe),
        "concentration_risk_state": _concentration_state(portfolio_weight),
        "missing_inputs_for_full_portfolio_decision": missing_inputs,
        "chapter_17_note": "Single-security forecasts should be interpreted inside portfolio weight, beta, correlation, and drawdown constraints.",
    }


def _mark_to_market_context(prices: pd.DataFrame, target_column: str) -> dict[str, Any]:
    close = pd.to_numeric(prices[target_column], errors="coerce").dropna()
    if close.empty:
        return {"status": "Unavailable", "reason": "no numeric close prices"}
    latest = close.iloc[-1]
    start = close.iloc[0]
    running_high = close.cummax()
    drawdown = close / running_high - 1
    return {
        "status": "Measured",
        "latest_price": _finite_or_none(latest),
        "run_start_price": _finite_or_none(start),
        "price_return_since_run_start": _finite_or_none(latest / start - 1 if start else np.nan),
        "drawdown_from_observed_high": _finite_or_none(drawdown.iloc[-1]),
        "max_observed_drawdown": _finite_or_none(drawdown.min()),
        "cost_basis_available": False,
        "chapter_17_note": "This is mark-to-market from available price history, not actual account P&L unless cost basis is supplied.",
    }


def _method_performance_ledger(backtests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for horizon, metrics in sorted(backtests.items(), key=lambda item: int(item[0])):
        rows.append(
            {
                "horizon_days": _safe_int(horizon),
                "rows": _safe_int(metrics.get("rows")),
                "cumulative_return": _finite_or_none(metrics.get("cumulative_return")),
                "benchmark_cumulative_return": _finite_or_none(metrics.get("benchmark_cumulative_return")),
                "sharpe_ratio": _finite_or_none(metrics.get("sharpe_ratio")),
                "max_drawdown": _finite_or_none(metrics.get("max_drawdown")),
                "hit_rate": _finite_or_none(metrics.get("hit_rate")),
                "trades": _safe_int(metrics.get("trades")),
                "turnover": _finite_or_none(metrics.get("turnover")),
            }
        )
    weak = [
        row["horizon_days"]
        for row in rows
        if (row.get("sharpe_ratio") is not None and float(row["sharpe_ratio"]) < 0)
        or (row.get("hit_rate") is not None and float(row["hit_rate"]) < 0.50)
    ]
    return {
        "status": "Measured" if rows else "Unavailable",
        "validation_signal_backtests": rows,
        "weak_horizons": weak,
        "chapter_17_note": "Method performance should be evaluated over many regimes; one run is evidence, not proof.",
    }


def _hedging_context(
    final_action: str,
    risk_level: str,
    portfolio_context: dict[str, Any],
    method_conflict: dict[str, Any],
    technical_contexts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    beta_estimates = portfolio_context.get("beta_estimates", {})
    chapter_15_confirmation = technical_contexts.get("chapter_15_major_trendlines", {}).get("broad_market_confirmation", {})
    hedge_review = bool(
        risk_level == "High"
        or method_conflict.get("level") == "High"
        or chapter_15_confirmation.get("status") == "Divergent"
        or final_action == "Sell"
    )
    return {
        "status": "ContextOnly",
        "hedge_review_flag": hedge_review,
        "beta_context_available": bool(beta_estimates),
        "broad_market_confirmation_status": chapter_15_confirmation.get("status"),
        "derivatives_data_available": False,
        "chapter_17_note": (
            "Options and futures may hedge or lever exposure, but this engine has no options-chain, futures-contract, margin, tax, or suitability model."
        ),
    }


def _data_and_method_limitations(
    data_quality_report: dict[str, Any],
    technical_history_quality: dict[str, Any],
    market_feature_comparison: dict[str, Any],
) -> dict[str, Any]:
    return {
        "data_quality_status": data_quality_report.get("status"),
        "data_quality_warnings": list(data_quality_report.get("warnings", []))[:8],
        "technical_history_sufficient": technical_history_quality.get("sufficient_for_classical_technical_analysis"),
        "technical_history_warnings": list(technical_history_quality.get("warnings", []))[:8],
        "market_only_vs_enriched_status": market_feature_comparison.get("status"),
        "chapter_17_note": "The report exposes limitations so the downstream reviewer can avoid false precision.",
    }


def _technical_method_direction_text(value: Any) -> str:
    return str(value or "").lower()


def _forecast_direction(preferred: dict[str, Any]) -> str:
    direction = str(preferred.get("expected_direction", ""))
    if direction == "Upward":
        return "bullish"
    if direction == "Downward":
        return "bearish"
    return "neutral"


def _action_direction(action: str) -> str:
    if action == "Buy":
        return "bullish"
    if action == "Sell":
        return "bearish"
    return "neutral"


def _state_direction(state: Any) -> str:
    text = _technical_method_direction_text(state)
    if text in {"bullish", "uptrend", "trendingup"}:
        return "bullish"
    if text in {"bearish", "downtrend", "trendingdown"}:
        return "bearish"
    return "neutral"


def _long_short_direction(state: Any) -> str:
    text = _technical_method_direction_text(state)
    if text == "long":
        return "bullish"
    if text == "short":
        return "bearish"
    return _state_direction(state)


def _pattern_direction(pattern: dict[str, Any]) -> str:
    text = " ".join(str(pattern.get(key, "")) for key in ("pattern", "status", "direction", "kind", "expected_breakout_direction")).lower()
    if "bearish" in text or "downside" in text or "breakdown" in text or "top" in text and "bottom" not in text:
        return "bearish"
    if "bullish" in text or "upside" in text or "breakout" in text or "bottom" in text:
        return "bullish"
    if "uptrend" in text and "break" not in text:
        return "bullish"
    if "downtrend" in text and "break" not in text:
        return "bearish"
    return "neutral"


def _chapter_13_direction(context: dict[str, Any]) -> str:
    preferred = context.get("preferred", {})
    text = " ".join(str(preferred.get(key, "")) for key in ("status", "direction", "role", "role_reversal")).lower()
    if "supportfailure" in text or "support failure" in text:
        return "bearish"
    if "resistancebreakout" in text or "resistance breakout" in text:
        return "bullish"
    return "neutral"


def _donchian_direction(state: Any) -> str:
    if state == "LongBreakout":
        return "bullish"
    if state == "ShortBreakout":
        return "bearish"
    return "neutral"


def _entry(name: str, direction: str, strength: Any, label: Any, explanation: str) -> dict[str, Any]:
    return {
        "name": name,
        "direction": direction,
        "strength": _finite_or_none(strength),
        "label": label,
        "explanation": explanation,
    }


def _pattern_label(pattern: dict[str, Any]) -> str:
    if not pattern:
        return "Unavailable"
    return f"{pattern.get('pattern', 'Unknown')} {pattern.get('status', 'Unknown')}".strip()


def _price_stage(close: pd.Series) -> str:
    if close.empty or len(close) < 20:
        return "Unknown"
    latest = close.iloc[-1]
    high_252 = close.rolling(min(252, len(close))).max().iloc[-1]
    low_252 = close.rolling(min(252, len(close))).min().iloc[-1]
    if high_252 and latest >= high_252 * 0.95:
        return "NearObservedHigh"
    if latest <= low_252 * 1.05:
        return "NearObservedLow"
    return "MiddleRange"


def _concentration_state(portfolio_weight: float | None) -> str:
    if portfolio_weight is None:
        return "Unknown"
    if portfolio_weight >= 0.25:
        return "High"
    if portfolio_weight >= 0.10:
        return "Medium"
    return "Low"


def _market_feature_degradation(market_feature_comparison: dict[str, Any]) -> float:
    rows = market_feature_comparison.get("horizons", [])
    if not isinstance(rows, list):
        return 0.0
    degraded = 0
    checked = 0
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "compared":
            continue
        checked += 1
        full = row.get("full_metric")
        market = row.get("market_only_metric")
        try:
            if full is not None and market is not None and float(full) > float(market):
                degraded += 1
        except Exception:
            continue
    return degraded / checked if checked else 0.0


def _add_component(components: list[dict[str, Any]], name: str, weight: float, reason: str) -> None:
    components.append({"name": name, "weight": float(weight), "reason": reason})


def _fragility_interpretation(level: str) -> str:
    if level == "High":
        return "Decision is fragile; downstream review should focus on conflicts, weak edge, and risk controls."
    if level == "Medium":
        return "Decision has usable evidence but needs context-aware review."
    return "Decision is comparatively stable for the available evidence."


def _conflict_interpretation(level: str) -> str:
    if level == "High":
        return "Technical methods are materially split; avoid treating one signal as decisive."
    if level == "Medium":
        return "Some technical evidence conflicts with the preferred forecast direction."
    return "Technical methods are not materially split."


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _finite_or_none(value: Any) -> float | None:
    try:
        output = float(value)
    except Exception:
        return None
    return output if np.isfinite(output) else None
