from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import Any

from market_forecasting_engine.llm_trader.responses_api import call_response, response_payload
from market_forecasting_engine.llm_trader.run import load_env, openai_client_for_provider, resolve_llm_model, resolve_llm_provider
from market_forecasting_engine.openai_models import DEFAULT_OPENAI_MODEL, DEFAULT_REASONING_EFFORT


PLANNER_SYSTEM_MESSAGE = """
# Role: Autonomous Portfolio Trader Planner

You are the planning brain of a fully autonomous Alpaca paper-trading agent. You are not a report writer. You manage attention, portfolio state, scan frequency, forecast frequency, order maintenance, and risk posture.

The ticker-level CEO will later make the final Buy/Hold/Sell decision after a full forecast. Your job is to decide what the trader should do this cycle before expensive work is run.

## Responsibilities

- Decide whether this is first-run portfolio construction or ongoing portfolio management.
- Review portfolio state, positions, open orders, cash, buying power, diversity, concentration, market clock, watch plans, and past memory.
- Use market intelligence, macro/political/regulatory/news context, and recent events to decide urgency.
- Decide whether to scan for new tickers.
- Decide which existing positions or watch-plan tickers need refreshed forecasts.
- Decide whether any open orders look stale and should be reviewed or cancelled.
- Decide next wake-up cadence, bounded by config.
- Prefer building a profitable, diversified paper portfolio over producing reports.

## Rules

- Do not directly submit orders.
- Do not invent broker positions.
- Be explicit when you want the execution layer to scan, forecast, cancel/review orders, or only monitor.
- If portfolio is empty, prioritize portfolio construction.
- If portfolio is concentrated, prefer diversified candidates and reduce adding to crowded exposure.
- If market is closed, avoid unnecessary forecast churn unless news/risk is urgent.
- If a price is near a stop, buy-lower, breakout, or sell/trim level, increase urgency.
- Return one JSON object matching the schema.
""".strip()


PLANNER_USER_MESSAGE = """
Today:
{{ item.today }}

Config:
{{ item.config_json }}

Portfolio state:
{{ item.portfolio_state_json }}

Market intelligence:
{{ item.market_intelligence_json }}

Trader memory:
{{ item.memory_json }}

Task:
Create the next cycle plan for the autonomous paper trader.
""".strip()


PLANNER_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "virtual_trader_portfolio_plan",
    "description": "Cycle plan for the autonomous virtual portfolio trader.",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "cycle_mode": {
                "type": "string",
                "enum": [
                    "bootstrap_portfolio",
                    "manage_existing_portfolio",
                    "risk_reduction",
                    "monitor_only",
                    "order_maintenance",
                ],
            },
            "portfolio_assessment": {"type": "string"},
            "diversity_assessment": {"type": "string"},
            "market_context_assessment": {"type": "string"},
            "risk_posture": {
                "type": "string",
                "enum": ["risk_on", "neutral", "cautious", "risk_off"],
            },
            "should_scan_new_candidates": {"type": "boolean"},
            "max_candidates_to_forecast": {"type": "integer"},
            "forecast_tickers": {"type": "array", "items": {"type": "string"}},
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task_type": {
                            "type": "string",
                            "enum": [
                                "scan_new_candidates",
                                "refresh_forecast",
                                "review_open_order",
                                "cancel_stale_order",
                                "monitor_watch_plan",
                                "reduce_risk",
                                "do_nothing",
                            ],
                        },
                        "ticker": {"type": "string"},
                        "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["task_type", "ticker", "priority", "reason"],
                    "additionalProperties": False,
                },
            },
            "order_management_notes": {"type": "array", "items": {"type": "string"}},
            "risk_budget_adjustment": {"type": "string"},
            "next_wakeup_seconds": {"type": "integer"},
            "next_wakeup_reason": {"type": "string"},
            "plan_rationale": {"type": "string"},
        },
        "required": [
            "cycle_mode",
            "portfolio_assessment",
            "diversity_assessment",
            "market_context_assessment",
            "risk_posture",
            "should_scan_new_candidates",
            "max_candidates_to_forecast",
            "forecast_tickers",
            "tasks",
            "order_management_notes",
            "risk_budget_adjustment",
            "next_wakeup_seconds",
            "next_wakeup_reason",
            "plan_rationale",
        ],
        "additionalProperties": False,
    },
}


def run_virtual_trader_planner(
    *,
    portfolio_state: dict[str, Any],
    market_intelligence: dict[str, Any],
    memory_context: dict[str, Any],
    config: dict[str, Any],
    llm_env_file: str | None,
    llm_model: str | None,
    llm_reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    timeout_seconds: float = 90.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    provider = resolve_llm_provider("openai")
    model = resolve_llm_model(llm_model or DEFAULT_OPENAI_MODEL, provider=provider)
    item = {
        "today": datetime.now(UTC).date().isoformat(),
        "config_json": json.dumps(config, indent=2, sort_keys=True, default=str),
        "portfolio_state_json": json.dumps(portfolio_state, indent=2, sort_keys=True, default=str),
        "market_intelligence_json": json.dumps(market_intelligence, indent=2, sort_keys=True, default=str)[:12000],
        "memory_json": json.dumps(memory_context, indent=2, sort_keys=True, default=str)[:12000],
    }
    payload = response_payload(
        model=model,
        system_message=PLANNER_SYSTEM_MESSAGE,
        user_message=PLANNER_USER_MESSAGE,
        json_schema=PLANNER_JSON_SCHEMA,
        reasoning_effort=llm_reasoning_effort,
        item=item,
        use_web_search=True,
        search_context_size=str(config.get("market_intelligence_search_context_size") or "low"),
        require_web_search=False,
    )
    if dry_run:
        return {
            "status": "dry_run",
            "provider": provider,
            "model": model,
            "plan": fallback_virtual_trader_plan(portfolio_state=portfolio_state, memory_context=memory_context, config=config),
            "llm_prompt_payload": payload,
            "reason": "Planner payload built; deterministic fallback plan used because dry_run=True.",
        }
    try:
        load_env(llm_env_file)
        client = openai_client_for_provider(provider, timeout=float(timeout_seconds))
        payload, raw_response, plan = call_response(
            client=client,
            provider=provider,
            model=model,
            system_message=PLANNER_SYSTEM_MESSAGE,
            user_message=PLANNER_USER_MESSAGE,
            json_schema=PLANNER_JSON_SCHEMA,
            reasoning_effort=llm_reasoning_effort,
            item=item,
            use_web_search=True,
            search_context_size=str(config.get("market_intelligence_search_context_size") or "low"),
            usage_context={
                "purpose": "virtual_trader_portfolio_cycle_planner",
                "portfolio_state": portfolio_state.get("state"),
                "provider": provider,
            },
        )
        return {
            "status": "executed",
            "provider": provider,
            "model": model,
            "plan": normalize_virtual_trader_plan(plan, config=config),
            "llm_prompt_payload": payload,
            "llm_raw_response": raw_response,
        }
    except Exception as exc:
        return {
            "status": "fallback",
            "provider": provider,
            "model": model,
            "plan": fallback_virtual_trader_plan(portfolio_state=portfolio_state, memory_context=memory_context, config=config),
            "llm_prompt_payload": payload,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def normalize_virtual_trader_plan(plan: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
    clean = dict(plan or {})
    min_seconds = int(config.get("min_loop_interval_seconds") or 900)
    max_seconds = int(config.get("max_loop_interval_seconds") or 21_600)
    wakeup = _as_int(clean.get("next_wakeup_seconds"), int(config.get("loop_interval_seconds") or 14_400))
    clean["next_wakeup_seconds"] = max(min_seconds, min(max_seconds, wakeup))
    max_candidates = max(1, _as_int(clean.get("max_candidates_to_forecast"), int(config.get("max_managed_candidates") or 5)))
    clean["max_candidates_to_forecast"] = min(max_candidates, int(config.get("max_managed_candidates") or max_candidates))
    clean["forecast_tickers"] = [str(item).upper() for item in clean.get("forecast_tickers", []) if str(item).strip()]
    return clean


def fallback_virtual_trader_plan(
    *,
    portfolio_state: dict[str, Any],
    memory_context: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    positions = portfolio_state.get("positions", []) if isinstance(portfolio_state.get("positions"), list) else []
    open_orders = int(portfolio_state.get("open_orders_count") or 0)
    diversity = portfolio_state.get("diversity", {}) if isinstance(portfolio_state.get("diversity"), dict) else {}
    near_trigger = any(row.get("near_trigger") for row in portfolio_state.get("trigger_proximity", []) or [])
    should_scan = not positions or diversity.get("status") in {"empty", "needs_diversification"}
    forecast_tickers = [str(row.get("symbol")).upper() for row in positions if row.get("symbol")]
    watch_plans = memory_context.get("active_watch_plans", {}) if isinstance(memory_context.get("active_watch_plans"), dict) else {}
    forecast_tickers.extend(str(ticker).upper() for ticker in watch_plans)
    tasks = []
    if should_scan:
        tasks.append({"task_type": "scan_new_candidates", "ticker": "", "priority": "high", "reason": "Portfolio needs construction or diversification."})
    for ticker in list(dict.fromkeys(forecast_tickers))[: int(config.get("max_managed_candidates") or 5)]:
        tasks.append({"task_type": "refresh_forecast", "ticker": ticker, "priority": "high" if near_trigger else "medium", "reason": "Existing position or watch plan requires periodic review."})
    if open_orders:
        tasks.append({"task_type": "review_open_order", "ticker": "", "priority": "high", "reason": "Open paper orders must be checked for staleness and duplication."})
    if not tasks:
        tasks.append({"task_type": "do_nothing", "ticker": "", "priority": "low", "reason": "No positions, orders, or watch plans found beyond normal scan."})
    return normalize_virtual_trader_plan(
        {
            "cycle_mode": "bootstrap_portfolio" if not positions else "manage_existing_portfolio",
            "portfolio_assessment": portfolio_state.get("state", "unknown"),
            "diversity_assessment": diversity.get("status", "unknown"),
            "market_context_assessment": "Fallback deterministic planner used; rely on collected market intelligence and ticker CEO decisions downstream.",
            "risk_posture": "neutral" if not near_trigger else "cautious",
            "should_scan_new_candidates": should_scan,
            "max_candidates_to_forecast": int(config.get("max_managed_candidates") or 5),
            "forecast_tickers": list(dict.fromkeys(forecast_tickers)),
            "tasks": tasks,
            "order_management_notes": ["Review open orders before submitting duplicates."],
            "risk_budget_adjustment": "normal" if diversity.get("status") != "needs_diversification" else "prefer_diversifying_candidates",
            "next_wakeup_seconds": int(portfolio_state.get("next_check_seconds") or config.get("loop_interval_seconds") or 14_400),
            "next_wakeup_reason": portfolio_state.get("cadence_reason", "fallback_schedule"),
            "plan_rationale": "Deterministic fallback plan generated from portfolio state, diversity, watch plans, and open orders.",
        },
        config=config,
    )


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default

