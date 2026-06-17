from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from market_forecasting_engine.llm_trader.prompts import long_term_source_synthesis
from market_forecasting_engine.llm_trader.responses_api import call_response, response_payload
from market_forecasting_engine.long_term_sources import (
    compact_long_term_context_for_llm,
    compact_long_term_context_for_source_synthesis,
)


def build_long_term_source_synthesis_payload(
    *,
    report: dict[str, Any],
    model: str,
    reasoning_effort: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    compact_context = _synthesis_report_long_term_context(report)
    item = {
        "today": datetime.now(UTC).date().isoformat(),
        "ticker": str(report.get("ticker") or "").upper(),
        "long_term_context_json": json.dumps(compact_context, indent=2, sort_keys=True, default=str),
    }
    payload = response_payload(
        model=model,
        system_message=long_term_source_synthesis.system_message,
        user_message=long_term_source_synthesis.user_message,
        json_schema=long_term_source_synthesis.json_schema,
        reasoning_effort=reasoning_effort,
        item=item,
        use_web_search=False,
        search_context_size="low",
    )
    return payload, item


def run_long_term_source_synthesis(
    *,
    report: dict[str, Any],
    llm_provider: str | None,
    llm_model: str | None,
    reasoning_effort: str,
    llm_env_file: str | None,
    timeout_seconds: float,
    dry_run: bool = False,
) -> dict[str, Any]:
    from market_forecasting_engine.llm_trader.run import load_env, openai_client_for_provider, resolve_llm_model, resolve_llm_provider

    compact_context = _synthesis_report_long_term_context(report)
    if not compact_context:
        return {
            "status": "skipped",
            "reason": "No long-term source context is available for synthesis.",
            "source_context_present": False,
        }
    provider = resolve_llm_provider(llm_provider)
    model = resolve_llm_model(llm_model, provider=provider)
    payload, item = build_long_term_source_synthesis_payload(
        report=report,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    base_result = {
        "status": "dry_run" if dry_run else "pending",
        "provider": provider,
        "model": model,
        "source_context_present": True,
        "llm_evidence_manifest": compact_context.get("llm_evidence_manifest", {}),
        "llm_prompt_payload": payload,
    }
    if dry_run:
        return {
            **base_result,
            "reason": "Long-term source synthesis payload was built but no model call was made.",
        }
    try:
        load_env(llm_env_file)
        client = openai_client_for_provider(provider, timeout=float(timeout_seconds))
        payload, raw_response, synthesis = call_response(
            client=client,
            provider=provider,
            model=model,
            system_message=long_term_source_synthesis.system_message,
            user_message=long_term_source_synthesis.user_message,
            json_schema=long_term_source_synthesis.json_schema,
            reasoning_effort=reasoning_effort,
            item=item,
            use_web_search=False,
            search_context_size="low",
            usage_context={
                "purpose": "long_term_source_synthesis_for_autonomous_ceo",
                "ticker": str(report.get("ticker") or "").upper(),
                "provider": provider,
            },
        )
    except Exception as exc:
        return {
            **base_result,
            "status": "error",
            "reason": str(exc),
            "policy": (
                "Source synthesis failed, so the autonomous CEO must still receive the full compact "
                "provider evidence and evidence manifest without omission."
            ),
        }
    return {
        **base_result,
        "status": "executed",
        "synthesis": synthesis,
        "llm_raw_response": raw_response,
        "policy": (
            "This is a preprocessing evidence synthesis. It does not make the final trading decision; "
            "the autonomous CEO receives this synthesis plus the original compact provider evidence."
        ),
    }


def attach_long_term_source_synthesis(report: dict[str, Any], synthesis_result: dict[str, Any]) -> None:
    if not synthesis_result or synthesis_result.get("status") == "skipped":
        return
    report.setdefault("decision_view", {})["long_term_source_synthesis"] = synthesis_result
    report.setdefault("final_decision_reasoning", {})["long_term_source_synthesis_status"] = synthesis_result.get("status")
    long_term_context = report.get("decision_view", {}).get("long_term_context")
    if isinstance(long_term_context, dict):
        long_term_context["llm_source_synthesis"] = {
            "status": synthesis_result.get("status"),
            "provider": synthesis_result.get("provider"),
            "model": synthesis_result.get("model"),
            "synthesis": synthesis_result.get("synthesis"),
            "reason": synthesis_result.get("reason"),
            "llm_evidence_manifest": synthesis_result.get("llm_evidence_manifest", {}),
        }
        consolidated = long_term_context.get("consolidated")
        if isinstance(consolidated, dict):
            consolidated["llm_source_synthesis"] = long_term_context["llm_source_synthesis"]
    technical_view = report.get("technical_view", {})
    if isinstance(technical_view, dict):
        source_context = technical_view.get("long_term_source_context")
        if isinstance(source_context, dict):
            source_context["llm_source_synthesis"] = {
                "status": synthesis_result.get("status"),
                "provider": synthesis_result.get("provider"),
                "model": synthesis_result.get("model"),
                "synthesis": synthesis_result.get("synthesis"),
                "reason": synthesis_result.get("reason"),
                "llm_evidence_manifest": synthesis_result.get("llm_evidence_manifest", {}),
            }


def _compact_report_long_term_context(report: dict[str, Any]) -> dict[str, Any]:
    decision = report.get("decision_view", {}) if isinstance(report.get("decision_view"), dict) else {}
    long_term_context = decision.get("long_term_context", {})
    if not isinstance(long_term_context, dict) or not long_term_context:
        technical = report.get("technical_view", {}) if isinstance(report.get("technical_view"), dict) else {}
        long_term_context = technical.get("long_term_source_context", {})
    compact = compact_long_term_context_for_llm(long_term_context if isinstance(long_term_context, dict) else {})
    return compact


def _synthesis_report_long_term_context(report: dict[str, Any]) -> dict[str, Any]:
    decision = report.get("decision_view", {}) if isinstance(report.get("decision_view"), dict) else {}
    long_term_context = decision.get("long_term_context", {})
    if not isinstance(long_term_context, dict) or not long_term_context:
        technical = report.get("technical_view", {}) if isinstance(report.get("technical_view"), dict) else {}
        long_term_context = technical.get("long_term_source_context", {})
    return compact_long_term_context_for_source_synthesis(long_term_context if isinstance(long_term_context, dict) else {})
