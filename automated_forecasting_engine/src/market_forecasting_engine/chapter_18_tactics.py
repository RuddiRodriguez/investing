from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_forecasting_engine.openai_models import DEFAULT_OPENAI_MODEL, DEFAULT_REASONING_EFFORT
from market_forecasting_engine.openai_responses import call_response


VALID_ACTIONS = {"Buy", "Hold", "Sell"}

TACTICAL_PROFILES: dict[str, dict[str, Any]] = {
    "short_term": {
        "preferred_horizon_days": 1,
        "holding_horizon": "1-5 trading days",
        "max_loss_pct": 0.035,
        "minimum_reward_to_risk": 1.25,
        "minimum_confidence": 0.56,
        "description": "Tighter tactical profile for short swings and quicker invalidation.",
    },
    "intermediate": {
        "preferred_horizon_days": 5,
        "holding_horizon": "5-30 trading days",
        "max_loss_pct": 0.060,
        "minimum_reward_to_risk": 1.50,
        "minimum_confidence": 0.55,
        "description": "Default Magee-style profile for intermediate market moves.",
    },
    "long_term": {
        "preferred_horizon_days": 30,
        "holding_horizon": "30-180 trading days",
        "max_loss_pct": 0.120,
        "minimum_reward_to_risk": 1.75,
        "minimum_confidence": 0.54,
        "description": "Wider profile for long-term holdings that still rejects unmanaged losses.",
    },
}


def analyze_chapter_18_tactical_problem(
    prices: pd.DataFrame,
    forecasts: list[dict[str, Any]],
    part_i_action: str,
    raw_action: str,
    risk_level: str,
    decision_diagnostics: dict[str, Any],
    technical_contexts: dict[str, dict[str, Any]],
    latest_features: pd.Series | dict[str, Any] | None = None,
    chapter_17_llm_packet: dict[str, Any] | None = None,
    tactical_profile: str = "intermediate",
    enable_llm_review: bool = False,
    llm_provider: str = "openai",
    llm_model: str | None = None,
    llm_temperature: float = 0.0,
    llm_reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    llm_timeout_seconds: int = 30,
    llm_env_file: str | None = None,
    llm_review_override: dict[str, Any] | None = None,
    target_column: str = "close",
) -> dict[str, Any]:
    """Build the Chapter 18 tactical plan and optional governed LLM review."""

    target = target_column.lower()
    clean = prices.copy()
    clean.columns = [str(column).lower() for column in clean.columns]
    close = pd.to_numeric(clean[target], errors="coerce").dropna()
    current_price = float(close.iloc[-1]) if not close.empty else 0.0
    profile = _profile_settings(tactical_profile)
    preferred = _preferred_forecast(forecasts, int(profile["preferred_horizon_days"]))
    latest = _latest_feature_dict(latest_features)

    stop_plan = _select_stop_plan(
        action=part_i_action,
        current_price=current_price,
        technical_contexts=technical_contexts,
        latest_features=latest,
        profile=profile,
    )
    target_plan = _select_target_plan(
        action=part_i_action,
        current_price=current_price,
        forecasts=forecasts,
        preferred_forecast=preferred,
        technical_contexts=technical_contexts,
        stop_plan=stop_plan,
        profile=profile,
    )
    mark_to_market = _mark_to_market(prices=clean, target_column=target, current_price=current_price)
    rule_gate = _chapter_18_rule_gate(
        action=part_i_action,
        risk_level=risk_level,
        preferred_forecast=preferred,
        stop_plan=stop_plan,
        target_plan=target_plan,
        decision_diagnostics=decision_diagnostics,
        mark_to_market=mark_to_market,
        profile=profile,
    )
    rule_based_action = "Hold" if rule_gate["hard_blockers"] else part_i_action
    trade_plan = _trade_plan(
        part_i_action=part_i_action,
        rule_based_action=rule_based_action,
        raw_action=raw_action,
        current_price=current_price,
        preferred_forecast=preferred,
        stop_plan=stop_plan,
        target_plan=target_plan,
        rule_gate=rule_gate,
        profile=profile,
    )
    llm_packet = _llm_review_packet(
        part_i_action=part_i_action,
        rule_based_action=rule_based_action,
        raw_action=raw_action,
        risk_level=risk_level,
        preferred_forecast=preferred,
        trade_plan=trade_plan,
        rule_gate=rule_gate,
        mark_to_market=mark_to_market,
        chapter_17_llm_packet=chapter_17_llm_packet or {},
    )
    llm_review = _llm_review_override_or_execute(
        override=llm_review_override,
        enabled=enable_llm_review,
        provider=llm_provider,
        model=llm_model,
        temperature=llm_temperature,
        reasoning_effort=llm_reasoning_effort,
        timeout_seconds=llm_timeout_seconds,
        env_file=llm_env_file,
        packet=llm_packet,
    )
    safety_gate = _apply_llm_safety_gate(
        rule_based_action=rule_based_action,
        llm_review=llm_review,
        rule_gate=rule_gate,
    )
    final_action = safety_gate["final_action"]

    return {
        "principle": (
            "Chapter 18 separates strategy from tactics: no commitment should be made without an entry policy, "
            "invalidation level, stop plan, target context, and mark-to-market discipline."
        ),
        "state": "Measured",
        "part": "Part II - Trading Tactics",
        "chapter": 18,
        "tactical_profile": profile,
        "raw_model_action": raw_action,
        "part_i_action": part_i_action,
        "rule_based_action": rule_based_action,
        "final_action": final_action,
        "trade_plan": trade_plan,
        "rule_gate": rule_gate,
        "llm_review": llm_review,
        "llm_safety_gate": safety_gate,
        "llm_review_packet": llm_packet,
        "mark_to_market": mark_to_market,
        "decision_policy": {
            "mode": "rule_first_optional_llm_review",
            "influences_final_action": True,
            "llm_can_upgrade_hold_to_directional": False,
            "llm_can_flip_direction": False,
            "llm_can_downgrade_to_hold": True,
            "reason": "Rules own eligibility and hard risk limits; the LLM can review context and only pass through the safety gate.",
        },
        "technical_method_card": chapter_18_tactical_problem_method_card(target_column=target),
    }


def chapter_18_tactical_problem_method_card(target_column: str = "close") -> dict[str, Any]:
    return {
        "name": "edwards_magee_chapter_18_tactical_problem",
        "version": "chapter_18_tactical_plan_v1",
        "target_column": target_column.lower(),
        "decision_policy": "rule_first_optional_llm_review_with_safety_gate",
        "implemented_controls": [
            "tactical_profile",
            "entry_policy",
            "stop_plan",
            "target_plan",
            "reward_to_risk_gate",
            "mark_to_market_discipline",
            "rule_based_candidate_action",
            "optional_llm_review",
            "post_llm_hard_rule_safety_gate",
        ],
        "chapter_18_alignment": [
            "strategy_is_not_enough_without_tactics",
            "define_stop_and_objective_before_commitment",
            "mark_losses_to_market",
            "do_not_confuse_company_quality_with_stock_action",
            "cut_weak_or_unmanaged_positions",
            "separate_trader_and_long_term_investor_profiles",
        ],
    }


def _profile_settings(name: str) -> dict[str, Any]:
    normalized = str(name or "intermediate").strip().lower().replace("-", "_")
    profile = dict(TACTICAL_PROFILES.get(normalized, TACTICAL_PROFILES["intermediate"]))
    profile["name"] = normalized if normalized in TACTICAL_PROFILES else "intermediate"
    return profile


def _preferred_forecast(forecasts: list[dict[str, Any]], preferred_horizon: int) -> dict[str, Any]:
    if not forecasts:
        return {}
    exact = next((item for item in forecasts if int(item.get("horizon_days", 0)) == preferred_horizon), None)
    if exact is not None:
        return dict(exact)
    return dict(min(forecasts, key=lambda item: abs(int(item.get("horizon_days", 0)) - preferred_horizon)))


def _select_stop_plan(
    action: str,
    current_price: float,
    technical_contexts: dict[str, dict[str, Any]],
    latest_features: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    if action not in {"Buy", "Sell"} or current_price <= 0:
        return {
            "status": "NotApplicable",
            "level": None,
            "distance_pct": None,
            "source": "no_new_commitment",
            "policy": "No stop is selected while the candidate action is Hold.",
        }

    max_loss = float(profile["max_loss_pct"])
    candidates: list[dict[str, Any]] = []
    magee = technical_contexts.get("magee_basing_points", {}).get("preferred", {})
    chapter_13 = technical_contexts.get("chapter_13_support_resistance", {})
    support = chapter_13.get("support_zones", {}).get("nearest", {})
    resistance = chapter_13.get("resistance_zones", {}).get("nearest", {})

    if action == "Buy":
        _add_stop_candidate(candidates, magee.get("active_basing_stop"), "magee_basing_stop", current_price, action)
        _add_stop_candidate(candidates, support.get("lower") or support.get("center"), "chapter_13_support_zone", current_price, action)
        _add_stop_candidate(candidates, latest_features.get("structure_pivot_support"), "confirmed_pivot_support", current_price, action)
        _add_stop_candidate(candidates, current_price * (1.0 - max_loss), "profile_max_loss_fallback", current_price, action)
        selected = max(candidates, key=lambda item: float(item["level"])) if candidates else None
    else:
        _add_stop_candidate(candidates, resistance.get("upper") or resistance.get("center"), "chapter_13_resistance_zone", current_price, action)
        _add_stop_candidate(candidates, latest_features.get("structure_pivot_resistance"), "confirmed_pivot_resistance", current_price, action)
        _add_stop_candidate(candidates, current_price * (1.0 + max_loss), "profile_max_loss_fallback", current_price, action)
        selected = min(candidates, key=lambda item: float(item["level"])) if candidates else None

    if not selected:
        return {
            "status": "Missing",
            "level": None,
            "distance_pct": None,
            "source": "none",
            "policy": "No valid stop candidate was found.",
        }
    return {
        "status": "Selected",
        "level": _finite_or_none(selected["level"]),
        "distance_pct": _finite_or_none(selected["distance_pct"]),
        "source": selected["source"],
        "candidate_count": len(candidates),
        "candidates": candidates[:8],
        "policy": "Closest valid stop in the adverse direction, with profile fallback when chart levels are too far away.",
    }


def _add_stop_candidate(
    candidates: list[dict[str, Any]],
    level: Any,
    source: str,
    current_price: float,
    action: str,
) -> None:
    value = _finite_or_none(level)
    if value is None or current_price <= 0:
        return
    if action == "Buy" and value >= current_price:
        return
    if action == "Sell" and value <= current_price:
        return
    distance = abs(current_price - value) / current_price
    candidates.append(
        {
            "level": value,
            "distance_pct": _finite_or_none(distance),
            "source": source,
        }
    )


def _select_target_plan(
    action: str,
    current_price: float,
    forecasts: list[dict[str, Any]],
    preferred_forecast: dict[str, Any],
    technical_contexts: dict[str, dict[str, Any]],
    stop_plan: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    if action not in {"Buy", "Sell"} or current_price <= 0:
        return {
            "status": "NotApplicable",
            "level": None,
            "distance_pct": None,
            "reward_to_risk": None,
            "source": "no_new_commitment",
            "policy": "No target is selected while the candidate action is Hold.",
        }

    candidates: list[dict[str, Any]] = []
    _add_target_candidate(candidates, preferred_forecast.get("predicted_price"), "preferred_forecast_price", current_price, action)
    for forecast in forecasts:
        _add_target_candidate(
            candidates,
            forecast.get("predicted_price"),
            f"forecast_{forecast.get('horizon_days')}d_price",
            current_price,
            action,
        )
    chapter_13 = technical_contexts.get("chapter_13_support_resistance", {})
    if action == "Buy":
        resistance = chapter_13.get("resistance_zones", {}).get("nearest", {})
        _add_target_candidate(candidates, resistance.get("center") or resistance.get("lower"), "chapter_13_nearest_resistance", current_price, action)
    else:
        support = chapter_13.get("support_zones", {}).get("nearest", {})
        _add_target_candidate(candidates, support.get("center") or support.get("upper"), "chapter_13_nearest_support", current_price, action)
    for candidate in _objective_candidates(technical_contexts):
        _add_target_candidate(candidates, candidate["level"], candidate["source"], current_price, action)

    stop_distance = _finite_or_none(stop_plan.get("distance_pct"))
    if stop_distance is not None and stop_distance > 0:
        rr = float(profile["minimum_reward_to_risk"])
        fallback = current_price * (1.0 + stop_distance * rr) if action == "Buy" else current_price * (1.0 - stop_distance * rr)
        _add_target_candidate(candidates, fallback, "profile_reward_to_risk_fallback", current_price, action)

    selected = min(candidates, key=lambda item: abs(float(item["level"]) - current_price)) if candidates else None
    reward_to_risk = None
    if selected is not None and stop_distance is not None and stop_distance > 0:
        reward_to_risk = float(selected["distance_pct"]) / stop_distance

    if not selected:
        return {
            "status": "Missing",
            "level": None,
            "distance_pct": None,
            "reward_to_risk": None,
            "source": "none",
            "policy": "No valid target candidate was found.",
        }
    return {
        "status": "Selected",
        "level": _finite_or_none(selected["level"]),
        "distance_pct": _finite_or_none(selected["distance_pct"]),
        "reward_to_risk": _finite_or_none(reward_to_risk),
        "source": selected["source"],
        "candidate_count": len(candidates),
        "candidates": candidates[:12],
        "policy": "Nearest valid objective in the forecast direction, so reward/risk is judged conservatively.",
    }


def _add_target_candidate(
    candidates: list[dict[str, Any]],
    level: Any,
    source: str,
    current_price: float,
    action: str,
) -> None:
    value = _finite_or_none(level)
    if value is None or current_price <= 0:
        return
    if action == "Buy" and value <= current_price:
        return
    if action == "Sell" and value >= current_price:
        return
    distance = abs(value - current_price) / current_price
    candidates.append(
        {
            "level": value,
            "distance_pct": _finite_or_none(distance),
            "source": source,
        }
    )


def _objective_candidates(payload: Any, path: str = "") -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            source = f"{path}.{key}" if path else str(key)
            if key in {"measured_objective", "objective", "price_objective"}:
                level = _finite_or_none(value)
                if level is not None:
                    candidates.append({"level": level, "source": source})
            elif isinstance(value, (dict, list)):
                candidates.extend(_objective_candidates(value, source))
    elif isinstance(payload, list):
        for idx, item in enumerate(payload[:20]):
            candidates.extend(_objective_candidates(item, f"{path}[{idx}]"))
    return candidates[:40]


def _chapter_18_rule_gate(
    action: str,
    risk_level: str,
    preferred_forecast: dict[str, Any],
    stop_plan: dict[str, Any],
    target_plan: dict[str, Any],
    decision_diagnostics: dict[str, Any],
    mark_to_market: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    hard_blockers: list[str] = []
    warnings: list[str] = []
    supporting: list[str] = []

    if action not in VALID_ACTIONS:
        hard_blockers.append(f"Invalid action `{action}` cannot be converted into a tactical plan.")
    if action == "Hold":
        supporting.append("Part I governance already selected Hold, so Chapter 18 avoids a new commitment.")
    if action in {"Buy", "Sell"}:
        confidence = _finite_or_none(preferred_forecast.get("directional_confidence")) or 0.0
        if risk_level == "High":
            hard_blockers.append("Overall risk is High; Chapter 18 blocks new directional commitments.")
        if confidence < float(profile["minimum_confidence"]):
            hard_blockers.append("Preferred forecast confidence is below the tactical profile gate.")
        if stop_plan.get("status") != "Selected":
            hard_blockers.append("No tactical stop is available before entry.")
        if target_plan.get("status") != "Selected":
            hard_blockers.append("No tactical target/objective is available before entry.")
        stop_distance = _finite_or_none(stop_plan.get("distance_pct"))
        if stop_distance is not None and stop_distance > float(profile["max_loss_pct"]) * 1.20:
            hard_blockers.append("Stop distance exceeds the tactical profile loss budget.")
        reward_to_risk = _finite_or_none(target_plan.get("reward_to_risk"))
        if reward_to_risk is not None and reward_to_risk < float(profile["minimum_reward_to_risk"]):
            hard_blockers.append("Conservative reward/risk is below the tactical profile gate.")
        if decision_diagnostics.get("hold_reason"):
            warnings.append(f"Part I hold reason remains relevant: {decision_diagnostics['hold_reason']}.")
    drawdown = _finite_or_none(mark_to_market.get("drawdown_from_252d_high_pct"))
    if drawdown is not None and drawdown <= -0.30:
        warnings.append("Current price is more than 30% below the 252-session high; avoid assuming it is cheap.")
    if not hard_blockers and action in {"Buy", "Sell"}:
        supporting.append("Chapter 18 has a stop, objective, and profile-compatible tactical plan.")

    return {
        "status": "Blocked" if hard_blockers else "Pass",
        "hard_blockers": hard_blockers,
        "warnings": warnings,
        "supporting_reasons": supporting,
        "minimum_reward_to_risk": float(profile["minimum_reward_to_risk"]),
        "maximum_loss_pct": float(profile["max_loss_pct"]),
        "minimum_confidence": float(profile["minimum_confidence"]),
    }


def _trade_plan(
    part_i_action: str,
    rule_based_action: str,
    raw_action: str,
    current_price: float,
    preferred_forecast: dict[str, Any],
    stop_plan: dict[str, Any],
    target_plan: dict[str, Any],
    rule_gate: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    entry_policy = "No new commitment while action is Hold."
    if rule_based_action == "Buy":
        entry_policy = "Long entry is allowed only while price remains above the selected stop/invalidation level."
    elif rule_based_action == "Sell":
        entry_policy = "Short or risk-reduction entry is allowed only while price remains below the selected stop/invalidation level."
    return {
        "candidate_action": part_i_action,
        "rule_based_action": rule_based_action,
        "raw_model_action": raw_action,
        "entry_policy": entry_policy,
        "expected_holding_horizon": profile["holding_horizon"],
        "current_price": _finite_or_none(current_price),
        "preferred_horizon_days": _safe_int(preferred_forecast.get("horizon_days")),
        "preferred_expected_direction": preferred_forecast.get("expected_direction"),
        "preferred_expected_return": _finite_or_none(preferred_forecast.get("expected_return")),
        "preferred_directional_confidence": _finite_or_none(preferred_forecast.get("directional_confidence")),
        "stop_plan": stop_plan,
        "target_plan": target_plan,
        "invalidation_reason": _invalidation_reason(rule_based_action, stop_plan),
        "max_loss_pct": _finite_or_none(stop_plan.get("distance_pct")),
        "reward_to_risk": _finite_or_none(target_plan.get("reward_to_risk")),
        "rule_status": rule_gate["status"],
        "rule_blockers": list(rule_gate.get("hard_blockers", [])),
        "rule_warnings": list(rule_gate.get("warnings", [])),
        "rule_supporting_reasons": list(rule_gate.get("supporting_reasons", [])),
    }


def _invalidation_reason(action: str, stop_plan: dict[str, Any]) -> str:
    source = stop_plan.get("source", "selected stop")
    if action == "Buy":
        return f"Close or decisive trade below {source} invalidates the long tactical plan."
    if action == "Sell":
        return f"Close or decisive trade above {source} invalidates the short/risk-reduction tactical plan."
    return "No new commitment is active."


def _mark_to_market(prices: pd.DataFrame, target_column: str, current_price: float) -> dict[str, Any]:
    close = pd.to_numeric(prices[target_column], errors="coerce").dropna()
    if close.empty or current_price <= 0:
        return {"status": "InsufficientData"}
    prior = float(close.iloc[-2]) if len(close) > 1 else np.nan
    high_252 = float(close.tail(252).max())
    low_252 = float(close.tail(252).min())
    return {
        "status": "Measured",
        "current_price": _finite_or_none(current_price),
        "previous_close": _finite_or_none(prior),
        "one_day_return_pct": _finite_or_none((current_price - prior) / prior if prior else np.nan),
        "twenty_day_return_pct": _finite_or_none(close.pct_change(20).iloc[-1] if len(close) > 20 else np.nan),
        "drawdown_from_252d_high_pct": _finite_or_none((current_price - high_252) / high_252 if high_252 else np.nan),
        "distance_from_252d_low_pct": _finite_or_none((current_price - low_252) / current_price if current_price else np.nan),
        "position_cost_basis": None,
        "unrealized_gain_loss_pct": None,
        "missing_position_context": [
            "portfolio shares and cost basis are not supplied in a single-ticker forecast run",
            "tax basis and account liquidity needs are outside this engine run",
        ],
    }


def _llm_review_packet(
    part_i_action: str,
    rule_based_action: str,
    raw_action: str,
    risk_level: str,
    preferred_forecast: dict[str, Any],
    trade_plan: dict[str, Any],
    rule_gate: dict[str, Any],
    mark_to_market: dict[str, Any],
    chapter_17_llm_packet: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": "Review a rule-based tactical trade plan. Return JSON only.",
        "allowed_actions": ["Buy", "Hold", "Sell"],
        "safety_policy": {
            "rules_own_hard_blockers": True,
            "llm_may_downgrade_directional_to_hold": True,
            "llm_may_upgrade_hold_to_directional": False,
            "llm_may_flip_buy_sell": False,
        },
        "raw_model_action": raw_action,
        "part_i_action": part_i_action,
        "rule_based_action": rule_based_action,
        "risk_level": risk_level,
        "preferred_forecast": {
            "horizon_days": _safe_int(preferred_forecast.get("horizon_days")),
            "expected_direction": preferred_forecast.get("expected_direction"),
            "expected_return": _finite_or_none(preferred_forecast.get("expected_return")),
            "directional_confidence": _finite_or_none(preferred_forecast.get("directional_confidence")),
            "predicted_price": _finite_or_none(preferred_forecast.get("predicted_price")),
        },
        "trade_plan": trade_plan,
        "rule_gate": rule_gate,
        "mark_to_market": mark_to_market,
        "chapter_17_context": chapter_17_llm_packet,
        "required_json_schema": {
            "recommended_action": "Buy|Hold|Sell",
            "confidence": "number from 0 to 1",
            "rationale": "short explanation",
            "risk_notes": ["short risk note"],
            "rule_consistency": "consistent|downgrade|invalid_override_request",
        },
    }


def _llm_review_override_or_execute(
    override: dict[str, Any] | None,
    enabled: bool,
    provider: str,
    model: str | None,
    temperature: float,
    reasoning_effort: str,
    timeout_seconds: int,
    env_file: str | None,
    packet: dict[str, Any],
) -> dict[str, Any]:
    if override is not None:
        review = dict(override)
        review.setdefault("status", "executed")
        review.setdefault("provider", "test_override")
        return review
    if not enabled:
        return {
            "status": "disabled",
            "provider": provider,
            "model": model,
            "recommended_action": None,
            "reason": "LLM review was not enabled for this run.",
        }
    return _execute_llm_review(
        packet=packet,
        provider=provider,
        model=model,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        timeout_seconds=timeout_seconds,
        env_file=env_file,
    )


def _execute_llm_review(
    packet: dict[str, Any],
    provider: str,
    model: str | None,
    temperature: float,
    reasoning_effort: str,
    timeout_seconds: int,
    env_file: str | None,
) -> dict[str, Any]:
    from market_forecasting_engine.llm_trader.run import openai_client_for_provider, resolve_llm_model, resolve_llm_provider

    selected_provider = resolve_llm_provider(provider)
    selected_model = resolve_llm_model(model or _read_provider_model(env_file, selected_provider), provider=selected_provider)
    missing_key = _missing_provider_key(env_file, selected_provider)
    if missing_key:
        return {
            "status": "skipped",
            "provider": selected_provider,
            "model": selected_model,
            "recommended_action": None,
            "reason": missing_key,
        }
    try:
        _ensure_provider_env_from_file(env_file, selected_provider)
        client = openai_client_for_provider(selected_provider, timeout=float(timeout_seconds))
        _, _, parsed = call_response(
            client=client,
            provider=selected_provider,
            model=selected_model,
            system_message=(
                "You are a governed trading decision reviewer. You do not predict prices. "
                "You review the structured tactical plan and return structured JSON only. "
                "Never recommend an action that violates the supplied safety policy."
            ),
            user_message="{{ item.packet }}",
            json_schema=_openai_tactical_review_schema(),
            reasoning_effort=reasoning_effort,
            item={"packet": json.dumps(packet, sort_keys=True)},
            usage_context={"purpose": "chapter_18_tactical_review", "ticker": packet.get("ticker"), "provider": selected_provider},
        )
        if not isinstance(parsed, dict):
            raise ValueError("LLM response JSON must be an object.")
        parsed.setdefault("recommended_action", parsed.get("final_action"))
        return {
            "status": "executed",
            "provider": selected_provider,
            "model": selected_model,
            "recommended_action": parsed.get("recommended_action"),
            "confidence": _finite_or_none(parsed.get("confidence")),
            "rationale": parsed.get("rationale"),
            "risk_notes": list(parsed.get("risk_notes", []))[:8] if isinstance(parsed.get("risk_notes"), list) else [],
            "rule_consistency": parsed.get("rule_consistency"),
            "raw_json": parsed,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "provider": selected_provider,
            "model": selected_model,
            "recommended_action": None,
            "error_type": type(exc).__name__,
            "reason": str(exc)[:500],
        }


def _read_provider_model(env_file: str | None, provider: str) -> str | None:
    if provider == "openai":
        return _read_env_value(env_file, "OPENAI_MODEL")
    if provider == "huggingface":
        return _read_env_value(env_file, "HUGGINGFACE_MODEL") or _read_env_value(env_file, "HF_MODEL")
    if provider == "bedrock":
        return _read_env_value(env_file, "BEDROCK_OPENAI_MODEL") or _read_env_value(env_file, "BEDROCK_MODEL")
    if provider == "llm_studio":
        return _read_env_value(env_file, "LLM_STUDIO_MODEL")
    return None


def _missing_provider_key(env_file: str | None, provider: str) -> str | None:
    if provider == "openai" and not (os.environ.get("OPENAI_API_KEY") or _read_env_value(env_file, "OPENAI_API_KEY")):
        return "OPENAI_API_KEY is not available in the environment or configured .env file."
    if provider == "huggingface" and not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY") or _read_env_value(env_file, "HF_TOKEN") or _read_env_value(env_file, "HUGGINGFACE_API_KEY")):
        return "HF_TOKEN or HUGGINGFACE_API_KEY is not available in the environment or configured .env file."
    return None


def _ensure_provider_env_from_file(env_file: str | None, provider: str) -> None:
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        value = _read_env_value(env_file, "OPENAI_API_KEY")
        if value:
            os.environ["OPENAI_API_KEY"] = value
    if provider == "huggingface":
        if not os.environ.get("HF_TOKEN"):
            value = _read_env_value(env_file, "HF_TOKEN")
            if value:
                os.environ["HF_TOKEN"] = value
        if not os.environ.get("HUGGINGFACE_API_KEY"):
            value = _read_env_value(env_file, "HUGGINGFACE_API_KEY")
            if value:
                os.environ["HUGGINGFACE_API_KEY"] = value


def _openai_tactical_review_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "tactical_review",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "recommended_action": {"type": "string", "enum": ["Buy", "Hold", "Sell"]},
                "final_action": {"type": "string", "enum": ["Buy", "Hold", "Sell"]},
                "confidence": {"type": "number"},
                "rationale": {"type": "string"},
                "risk_notes": {"type": "array", "items": {"type": "string"}},
                "rule_consistency": {"type": "string"},
            },
            "required": [
                "recommended_action",
                "final_action",
                "confidence",
                "rationale",
                "risk_notes",
                "rule_consistency",
            ],
        },
    }


def _apply_llm_safety_gate(
    rule_based_action: str,
    llm_review: dict[str, Any],
    rule_gate: dict[str, Any],
) -> dict[str, Any]:
    if llm_review.get("status") != "executed":
        return {
            "status": "not_applicable",
            "final_action": rule_based_action,
            "accepted_llm_action": False,
            "reason": "No executed LLM recommendation was available.",
        }
    recommended = str(llm_review.get("recommended_action") or "").strip().title()
    if recommended not in VALID_ACTIONS:
        return {
            "status": "rejected",
            "final_action": rule_based_action,
            "accepted_llm_action": False,
            "reason": "LLM recommendation was missing or not one of Buy/Hold/Sell.",
            "recommended_action": recommended or None,
        }
    if recommended == rule_based_action:
        return {
            "status": "accepted",
            "final_action": rule_based_action,
            "accepted_llm_action": True,
            "reason": "LLM recommendation matched the rule-based tactical action.",
            "recommended_action": recommended,
        }
    if rule_gate.get("hard_blockers") and recommended in {"Buy", "Sell"}:
        return {
            "status": "rejected",
            "final_action": rule_based_action,
            "accepted_llm_action": False,
            "reason": "LLM cannot bypass Chapter 18 hard blockers.",
            "recommended_action": recommended,
        }
    if rule_based_action in {"Buy", "Sell"} and recommended == "Hold":
        return {
            "status": "accepted_downgrade",
            "final_action": "Hold",
            "accepted_llm_action": True,
            "reason": "LLM downgraded a directional rule-based action to Hold, which is allowed.",
            "recommended_action": recommended,
        }
    if rule_based_action == "Hold" and recommended in {"Buy", "Sell"}:
        return {
            "status": "rejected",
            "final_action": "Hold",
            "accepted_llm_action": False,
            "reason": "LLM cannot upgrade a rule-based Hold into a directional action.",
            "recommended_action": recommended,
        }
    return {
        "status": "rejected",
        "final_action": rule_based_action,
        "accepted_llm_action": False,
        "reason": "LLM cannot flip Buy to Sell or Sell to Buy.",
        "recommended_action": recommended,
    }


def _read_env_value(env_file: str | None, key: str) -> str | None:
    if not env_file:
        default_path = Path.cwd() / ".env"
        if not default_path.exists():
            return None
        path = default_path
    else:
        path = Path(env_file)
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#") or "=" not in clean:
            continue
        name, value = clean.split("=", 1)
        if name.strip() == key:
            return value.strip().strip('"').strip("'")
    return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _latest_feature_dict(latest_features: pd.Series | dict[str, Any] | None) -> dict[str, Any]:
    if latest_features is None:
        return {}
    if isinstance(latest_features, pd.Series):
        return latest_features.to_dict()
    return dict(latest_features)
