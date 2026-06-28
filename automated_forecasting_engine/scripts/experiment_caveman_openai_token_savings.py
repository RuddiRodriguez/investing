#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from market_forecasting_engine.llm_trader.responses_api import render_template, response_payload
from market_forecasting_engine.openai_models import DEFAULT_OPENAI_MODEL, DEFAULT_REASONING_EFFORT
from market_forecasting_engine.pure_llm_virtual_trader_agent import (
    MARKET_INTELLIGENCE_JSON_SCHEMA,
    MARKET_INTELLIGENCE_SYSTEM_MESSAGE,
    MARKET_INTELLIGENCE_USER_MESSAGE,
    PLANNER_JSON_SCHEMA,
    PLANNER_SYSTEM_MESSAGE,
    PLANNER_USER_MESSAGE,
    SCOUT_JSON_SCHEMA,
    SCOUT_SYSTEM_MESSAGE,
    SCOUT_USER_MESSAGE,
)


CAVEMAN_SYSTEM_SUFFIX = """

# Output Compression Mode

Use terse technical language inside every free-text JSON string.
No filler, caveats, or prose outside the schema.
Prefer compact fragments over paragraphs.
Preserve all required fields, tickers, numbers, dates, URLs, and risk facts exactly.
Do not omit safety-critical blockers.
Use short arrays unless the schema or task requires more.
Return exactly one JSON object matching the schema.
""".strip()


@dataclass(frozen=True)
class ExperimentCase:
    name: str
    system_message: str
    user_message: str
    json_schema: dict[str, Any]
    item: dict[str, Any]
    use_web_search: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline OpenAI prompts vs Caveman-style terse prompts.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL)
    parser.add_argument("--reasoning-effort", default=os.getenv("OPENAI_REASONING_EFFORT") or DEFAULT_REASONING_EFFORT)
    parser.add_argument("--env-file", default=None, help="Optional .env file containing OPENAI_API_KEY.")
    parser.add_argument("--run-api", action="store_true", help="Call OpenAI and record exact usage. Default only estimates prompt size.")
    parser.add_argument("--output", default="automated_forecasting_engine/runs/caveman_openai_token_experiment/latest.json")
    parser.add_argument("--search-context-size", default="low")
    args = parser.parse_args()

    if args.env_file:
        load_env_file(args.env_file)

    cases = build_cases()
    rows: list[dict[str, Any]] = []
    for case in cases:
        baseline = build_variant(case=case, model=args.model, reasoning_effort=args.reasoning_effort, mode="baseline", search_context_size=args.search_context_size)
        verbosity_low = build_variant(case=case, model=args.model, reasoning_effort=args.reasoning_effort, mode="verbosity_low", search_context_size=args.search_context_size)
        caveman = build_variant(case=case, model=args.model, reasoning_effort=args.reasoning_effort, mode="caveman", search_context_size=args.search_context_size)
        row: dict[str, Any] = {
            "case": case.name,
            "dry_estimate": {
                "baseline": estimate_payload(baseline),
                "verbosity_low": estimate_payload(verbosity_low),
                "caveman": estimate_payload(caveman),
            },
        }
        row["dry_estimate"]["verbosity_low_savings"] = savings(row["dry_estimate"]["baseline"], row["dry_estimate"]["verbosity_low"])
        row["dry_estimate"]["caveman_savings"] = savings(row["dry_estimate"]["baseline"], row["dry_estimate"]["caveman"])
        if args.run_api:
            row["actual"] = {
                "baseline": call_openai(payload=baseline),
                "verbosity_low": call_openai(payload=verbosity_low),
                "caveman": call_openai(payload=caveman),
            }
            row["actual"]["verbosity_low_savings"] = actual_savings(row["actual"]["baseline"], row["actual"]["verbosity_low"])
            row["actual"]["caveman_savings"] = actual_savings(row["actual"]["baseline"], row["actual"]["caveman"])
        rows.append(row)
        print_summary_row(row)

    output = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "run_api": bool(args.run_api),
        "notes": [
            "Dry estimates use character counts because tiktoken is not installed in this venv.",
            "Actual OpenAI usage, when --run-api is set, is authoritative.",
            "Caveman-style mode is intentionally isolated to this experiment.",
        ],
        "cases": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output_path}")


def build_cases() -> list[ExperimentCase]:
    broker_state = {
        "account": {"equity": "10000.00", "cash": "4500.00", "buying_power": "9000.00", "status": "ACTIVE"},
        "positions": [{"symbol": "AAPL", "qty": "4", "market_value": "792.80", "unrealized_plpc": "0.011"}],
        "open_orders": [],
        "market_clock": {"is_open": False, "next_open": "2026-06-22T13:30:00Z"},
    }
    memory = {
        "recent_cycles": [
            {"tickers": ["AAPL", "TSLA"], "summary": "Held due market closed; watched lower limit entries."},
            {"tickers": ["MSFT"], "summary": "Rejected new risk until better cash deployment evidence."},
        ],
        "watchlist": ["AAPL", "TSLA", "MSFT"],
    }
    market_intelligence = {
        "market_regime": "cautious",
        "summary": "Indexes near highs, rates uncertain, mega-cap earnings sensitivity elevated.",
        "macro_risks": ["rate path uncertainty", "stretched mega-cap valuations"],
        "sector_themes": ["AI capex focus", "quality large-cap preference"],
        "events_to_watch": ["Fed speakers", "large-cap earnings revisions"],
        "portfolio_implications": ["Prefer liquid names; avoid oversized single-name risk."],
    }
    config = {
        "risk_profile": "aggressive",
        "trader_profile": "aggressive",
        "max_candidates": 2,
        "dry_run": False,
    }
    scout = {
        "scout_summary": "Review existing AAPL and one high-liquidity growth candidate.",
        "candidates": [
            {"ticker": "AAPL", "company": "Apple Inc.", "priority": "high", "candidate_reason": "Existing position and liquid large cap.", "main_catalysts": ["earnings revisions"], "main_risks": ["valuation"]},
            {"ticker": "TSLA", "company": "Tesla Inc.", "priority": "medium", "candidate_reason": "High volatility candidate, needs strict sizing.", "main_catalysts": ["delivery updates"], "main_risks": ["margin pressure"]},
        ],
        "rejected_themes": ["illiquid small caps"],
    }
    return [
        ExperimentCase(
            name="market_intelligence",
            system_message=MARKET_INTELLIGENCE_SYSTEM_MESSAGE,
            user_message=MARKET_INTELLIGENCE_USER_MESSAGE,
            json_schema=MARKET_INTELLIGENCE_JSON_SCHEMA,
            item={
                "today": "2026-06-20",
                "broker_state_json": json.dumps(broker_state, indent=2, sort_keys=True),
                "memory_json": json.dumps(memory, indent=2, sort_keys=True),
            },
        ),
        ExperimentCase(
            name="scout",
            system_message=SCOUT_SYSTEM_MESSAGE,
            user_message=SCOUT_USER_MESSAGE,
            json_schema=SCOUT_JSON_SCHEMA,
            item={
                "today": "2026-06-20",
                "market_intelligence_json": json.dumps(market_intelligence, indent=2, sort_keys=True),
                "broker_state_json": json.dumps(broker_state, indent=2, sort_keys=True),
                "memory_json": json.dumps(memory, indent=2, sort_keys=True),
                "max_candidates": 2,
            },
        ),
        ExperimentCase(
            name="planner",
            system_message=PLANNER_SYSTEM_MESSAGE,
            user_message=PLANNER_USER_MESSAGE,
            json_schema=PLANNER_JSON_SCHEMA,
            item={
                "today": "2026-06-20",
                "config_json": json.dumps(config, indent=2, sort_keys=True),
                "broker_state_json": json.dumps(
                    {
                        **broker_state,
                        "market_intelligence": market_intelligence,
                        "llm_scout": scout,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                "memory_json": json.dumps(memory, indent=2, sort_keys=True),
            },
        ),
    ]


def build_variant(*, case: ExperimentCase, model: str, reasoning_effort: str, mode: str, search_context_size: str) -> dict[str, Any]:
    system_message = case.system_message
    user_message = case.user_message
    if mode == "caveman":
        system_message = f"{system_message}\n\n{CAVEMAN_SYSTEM_SUFFIX}"
    payload = response_payload(
        model=model,
        system_message=system_message,
        user_message=user_message,
        json_schema=case.json_schema,
        reasoning_effort=reasoning_effort,
        item=case.item,
        use_web_search=case.use_web_search,
        search_context_size=search_context_size,
    )
    if mode in {"verbosity_low", "caveman"}:
        payload.setdefault("text", {})["verbosity"] = "low"
    return payload


def estimate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    input_text = json.dumps(payload.get("input", []), separators=(",", ":"), sort_keys=True)
    schema_text = json.dumps(payload.get("text", {}).get("format", {}), separators=(",", ":"), sort_keys=True)
    total_chars = len(input_text) + len(schema_text)
    return {
        "input_chars": len(input_text),
        "schema_chars": len(schema_text),
        "estimated_prompt_tokens": round(total_chars / 4),
        "verbosity": payload.get("text", {}).get("verbosity"),
    }


def savings(base: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    before = base["estimated_prompt_tokens"]
    after = variant["estimated_prompt_tokens"]
    return {"estimated_prompt_tokens_delta": after - before, "estimated_prompt_tokens_pct": pct(before - after, before)}


def actual_savings(base: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    output_before = base.get("output_tokens") or 0
    output_after = variant.get("output_tokens") or 0
    total_before = base.get("total_tokens") or 0
    total_after = variant.get("total_tokens") or 0
    return {
        "output_tokens_delta": output_after - output_before,
        "output_tokens_saved_pct": pct(output_before - output_after, output_before),
        "total_tokens_delta": total_after - total_before,
        "total_tokens_saved_pct": pct(total_before - total_after, total_before),
    }


def call_openai(*, payload: dict[str, Any]) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for --run-api.")
    from openai import OpenAI

    started = time.perf_counter()
    response = OpenAI().responses.create(**payload)
    data = response.model_dump(mode="json")
    usage = data.get("usage") or {}
    output_text = extract_output_text(data)
    return {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": ((usage.get("output_tokens_details") or {}).get("reasoning_tokens")),
        "output_chars": len(output_text),
        "output_preview": output_text[:500],
    }


def extract_output_text(response_data: dict[str, Any]) -> str:
    texts: list[str] = []
    for item in response_data.get("output") or []:
        for part in item.get("content") or []:
            if part.get("type") in {"output_text", "text"}:
                texts.append(str(part.get("text") or ""))
    return "\n".join(texts)


def pct(numerator: float, denominator: float) -> float | None:
    if not denominator:
        return None
    return round(100 * numerator / denominator, 2)


def print_summary_row(row: dict[str, Any]) -> None:
    dry = row["dry_estimate"]
    print(
        f"{row['case']}: dry prompt est baseline={dry['baseline']['estimated_prompt_tokens']} "
        f"low={dry['verbosity_low']['estimated_prompt_tokens']} "
        f"caveman={dry['caveman']['estimated_prompt_tokens']} "
        f"low_delta={dry['verbosity_low_savings']['estimated_prompt_tokens_delta']} "
        f"caveman_delta={dry['caveman_savings']['estimated_prompt_tokens_delta']}"
    )
    if "actual" in row:
        actual = row["actual"]
        print(
            f"  actual output baseline={actual['baseline'].get('output_tokens')} "
            f"low={actual['verbosity_low'].get('output_tokens')} "
            f"caveman={actual['caveman'].get('output_tokens')} "
            f"low_saved={actual['verbosity_low_savings'].get('output_tokens_saved_pct')}% "
            f"caveman_saved={actual['caveman_savings'].get('output_tokens_saved_pct')}%"
        )


def load_env_file(path: str) -> None:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


if __name__ == "__main__":
    main()
