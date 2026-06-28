from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_forecasting_engine.data import normalize_price_frame
from market_forecasting_engine.data_providers import DataRequest, load_prices_with_provider
from market_forecasting_engine.llm_model_catalog import DEFAULT_FULL_LLM_OPTIONS_MODEL, DEFAULT_FULL_LLM_OPTIONS_PROVIDER
from market_forecasting_engine.llm_trader.profiles import trader_profiles
from market_forecasting_engine.llm_trader.prompts import autonomous_trader
from market_forecasting_engine.llm_trader.responses_api import call_response
from market_forecasting_engine.llm_handler import normalize_provider_name, resolve_llm_client_profile
from market_forecasting_engine.openai_models import ModelName


FORECAST_SYSTEM_MESSAGE = """You are a pure LLM stock forecaster.

Python is only the data loader and recorder. No deterministic/statistical forecast engine has produced a forecast before you.
You receive raw/descriptive market data and must form your own 1 day, 1 week, and 1 month stock forecast.
Do not claim that a model, backtest, ARIMA, boosting model, or deterministic engine produced the numbers.
Return exactly one JSON object matching the schema.
"""


def load_env(path: str | None) -> None:
    env_path = Path(path or ".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            clean = line.strip()
            if clean and not clean.startswith("#") and "=" in clean:
                name, value = clean.split("=", 1)
                os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def resolve_llm_provider(provider: str | None) -> str:
    return normalize_provider_name(provider or os.environ.get("LLM_PROVIDER") or "openai")


def resolve_llm_model(model: str | None, *, provider: str | None = None) -> str:
    return resolve_llm_client_profile(provider=resolve_llm_provider(provider), model=model).model


def openai_client_for_provider(provider: str, *, timeout: float):
    if resolve_llm_provider(provider) != "openai":
        return None
    from openai import OpenAI

    return OpenAI(timeout=timeout)

FORECAST_USER_MESSAGE = """Forecast the stock from this raw market packet.

Market packet:
{{ item.market_packet_json }}
"""


FORECAST_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "pure_llm_stock_forecast",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ticker": {"type": "string"},
            "company": {"type": "string"},
            "method": {"type": "string"},
            "current_price": {"type": "number"},
            "forecasts": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "horizon": {"type": "string", "enum": ["1_day", "1_week", "1_month"]},
                        "trading_days": {"type": "integer", "enum": [1, 5, 21]},
                        "direction": {"type": "string", "enum": ["up", "down", "flat"]},
                        "predicted_price": {"type": "number"},
                        "expected_return_pct": {"type": "number"},
                        "confidence": {"type": "number"},
                        "bear_case_price": {"type": "number"},
                        "bull_case_price": {"type": "number"},
                        "reasoning": {"type": "string"},
                        "key_invalidations": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "horizon",
                        "trading_days",
                        "direction",
                        "predicted_price",
                        "expected_return_pct",
                        "confidence",
                        "bear_case_price",
                        "bull_case_price",
                        "reasoning",
                        "key_invalidations",
                    ],
                },
            },
            "overall_view": {"type": "string"},
            "main_risks": {"type": "array", "items": {"type": "string"}},
            "data_limitations": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "ticker",
            "company",
            "method",
            "current_price",
            "forecasts",
            "overall_view",
            "main_risks",
            "data_limitations",
        ],
    },
}


FORECAST_QUALITY_SYSTEM_MESSAGE = """You are the forecast-quality review step for a pure LLM stock forecast.

No deterministic backtest, walk-forward validation, or statistical model validation was run. Do not claim otherwise.
Review the pure LLM forecast, raw market packet, and external LLM evidence. Produce a quality assessment that the CEO can use as a substitute for validated model stats in this LLM-only path.
Return exactly one JSON object matching the schema.
"""

FORECAST_QUALITY_USER_MESSAGE = """Review this pure LLM forecast.

Market packet:
{{ item.market_packet_json }}

Pure LLM forecast:
{{ item.forecast_json }}

External LLM evidence:
{{ item.external_evidence_json }}
"""


FORECAST_QUALITY_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "pure_llm_forecast_quality_review",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {"type": "string", "enum": ["executed", "limited", "failed"]},
            "validation_type": {"type": "string"},
            "quality_summary": {"type": "string"},
            "forecast_consistency_read": {"type": "string"},
            "evidence_alignment_read": {"type": "string"},
            "confidence_adjustment": {"type": "string", "enum": ["raise", "keep", "lower"]},
            "usable_for_ceo": {"type": "boolean"},
            "limitations": {"type": "array", "items": {"type": "string"}},
            "red_flags": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "status",
            "validation_type",
            "quality_summary",
            "forecast_consistency_read",
            "evidence_alignment_read",
            "confidence_adjustment",
            "usable_for_ceo",
            "limitations",
            "red_flags",
        ],
    },
}


MARKET_STRUCTURE_SYSTEM_MESSAGE = """You are a market-structure and liquidity-read step before CEO advice.

Read the recent OHLC tape using this framework:
- Swing lows can hold liquidity below them.
- Equal or repeated lows usually hold more downside liquidity.
- A wick below a swing low can be a liquidity sweep.
- If bullish structure remains valid, price should recover and respect the swept zone.
- If price weakly bounces/retests then closes below the low, treat it as a bearish break of structure.
- After a bearish break, watch consolidation, fair value gaps, former support turning resistance, and uncollected downside liquidity.
- If those align, price may retrace into former support/fair value gap, then rotate lower to sweep remaining liquidity.

Use this as timing/context only. Do not override news, fundamentals, strategy knowledge, portfolio risk, or execution gates.
Return exactly one JSON object matching the schema.
"""


MARKET_STRUCTURE_USER_MESSAGE = """Create a market-structure/liquidity read for CEO advice.

Market packet:
{{ item.market_packet_json }}

Pure LLM forecast:
{{ item.forecast_json }}

Forecast quality:
{{ item.forecast_quality_json }}

Strategy knowledge context:
{{ item.strategy_knowledge_json }}
"""


MARKET_STRUCTURE_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "pure_llm_market_structure_liquidity_read",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {"type": "string", "enum": ["executed", "limited", "failed"]},
            "structure_bias": {"type": "string", "enum": ["bullish", "bearish", "mixed", "unclear"]},
            "swing_low_read": {"type": "string"},
            "liquidity_pools": {"type": "array", "items": {"type": "string"}},
            "sweep_or_break_read": {"type": "string"},
            "support_resistance_flip_read": {"type": "string"},
            "fair_value_gap_read": {"type": "string"},
            "expected_path": {"type": "string"},
            "ceo_implication": {"type": "string"},
            "entry_timing_notes": {"type": "array", "items": {"type": "string"}},
            "invalidation_notes": {"type": "array", "items": {"type": "string"}},
            "limitations": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "status",
            "structure_bias",
            "swing_low_read",
            "liquidity_pools",
            "sweep_or_break_read",
            "support_resistance_flip_read",
            "fair_value_gap_read",
            "expected_path",
            "ceo_implication",
            "entry_timing_notes",
            "invalidation_notes",
            "limitations",
        ],
    },
}


def main() -> None:
    args = build_parser().parse_args()
    load_env(args.llm_env_file)
    progress(args, "START", "pure LLM stock forecast started", ticker=args.ticker.upper(), provider=args.llm_provider, model=args.llm_model)
    record = run_pure_llm_stock_forecast(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_safe_name(args.ticker)}_pure_llm_stock_forecast.json"
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    progress(args, "OUTPUT", "artifact written", output=str(output_path))
    print(json.dumps({"output": str(output_path), "forecast": record["forecast"], "advice": record["advice"]}, indent=2, sort_keys=True, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a pure LLM stock forecast and isolated CEO advice path.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--company", default=None)
    parser.add_argument("--provider", default="yahoo")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--target-column", default="close")
    parser.add_argument("--bars", type=int, default=20, help="Recent bars to pass to the LLM.")
    parser.add_argument("--output-dir", default="automated_forecasting_engine/runs/pure_llm_stock_forecaster")
    parser.add_argument("--llm-provider", choices=("openai", "huggingface", "bedrock", "llm_studio"), default=DEFAULT_FULL_LLM_OPTIONS_PROVIDER)
    parser.add_argument("--llm-model", default=DEFAULT_FULL_LLM_OPTIONS_MODEL)
    parser.add_argument("--fallback-llm-provider", choices=("openai", "huggingface", "bedrock", "llm_studio"), default="openai")
    parser.add_argument("--fallback-llm-model", default=ModelName.GPT_5_4_MINI_2026_03_17.value)
    parser.add_argument(
        "--ceo-llm-provider",
        choices=("openai", "huggingface", "bedrock", "llm_studio"),
        default="openai",
        help="Provider for the CEO advice step. Defaults to OpenAI so the forecast can stay local while CEO uses structured output.",
    )
    parser.add_argument(
        "--ceo-llm-model",
        default=ModelName.GPT_5_4_2026_03_05.value,
        help="Model for the CEO advice step. Defaults to the dated OpenAI GPT-5.4 API model ID.",
    )
    parser.add_argument("--trader-profile", choices=("aggressive", "medium", "conservative"), default="medium")
    parser.add_argument("--trader-name", default="pure_llm_stock_ceo")
    parser.add_argument("--holding-status", choices=("not_owned", "owned"), default="not_owned")
    parser.add_argument("--entry-price", type=float, default=None)
    parser.add_argument("--quantity", type=float, default=None)
    parser.add_argument("--position-value", type=float, default=None)
    parser.add_argument("--account-equity", type=float, default=None)
    parser.add_argument("--portfolio-notes", default="")
    parser.add_argument("--external-evidence-json", default=None)
    parser.add_argument(
        "--skip-llm-evidence",
        action="store_true",
        help="Do not auto-build LLM fresh news/fundamentals/source synthesis when external evidence is not supplied.",
    )
    parser.add_argument("--disable-strategy-knowledge", action="store_true", help="Skip durable strategy knowledge retrieval.")
    parser.add_argument("--strategy-knowledge-corpus-dir", default="automated_forecasting_engine/strategy_knowledge/corpus")
    parser.add_argument("--strategy-knowledge-index", default="automated_forecasting_engine/strategy_knowledge/indexes/strategy_knowledge_text_only.faiss")
    parser.add_argument("--strategy-knowledge-max-chunks", type=int, default=8)
    parser.add_argument("--strategy-knowledge-rebuild-index", action="store_true")
    parser.add_argument("--llm-timeout", type=float, default=900.0)
    parser.add_argument("--llm-env-file", default=None)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--search-context-size", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build packets only; do not call the LLM.")
    return parser


def run_pure_llm_stock_forecast(args: argparse.Namespace) -> dict[str, Any]:
    progress(args, "DATA", "loading stock price data", ticker=args.ticker.upper(), provider=args.provider, interval=args.interval)
    prices, metadata = load_stock_prices(args)
    progress(args, "DATA", "price data loaded", rows=len(prices), last_price=round(float(prices[args.target_column].iloc[-1]), 4))
    progress(args, "PACKET", "building raw/descriptive market packet", bars=int(args.bars))
    external_evidence = load_or_build_external_evidence(args, prices=prices, metadata=metadata)
    prompt_evidence = compact_external_evidence_for_prompt(external_evidence)
    packet = build_market_packet(
        prices,
        ticker=args.ticker,
        company=args.company or args.ticker.upper(),
        metadata=metadata,
        bar_count=int(args.bars),
    )
    if prompt_evidence:
        packet["external_llm_evidence"] = prompt_evidence
    provider = resolve_llm_provider(args.llm_provider)
    model = resolve_llm_model(args.llm_model, provider=provider)
    ceo_provider = resolve_llm_provider(args.ceo_llm_provider)
    ceo_model = resolve_llm_model(args.ceo_llm_model, provider=ceo_provider)
    base_record: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "path": "pure_llm_stock_forecaster",
        "provider": provider,
        "model": model,
        "ceo_provider": ceo_provider,
        "ceo_model": ceo_model,
        "policy": {
            "mode": "pure_llm_stock_forecast_and_ceo_advice",
            "deterministic_forecast_engine_used": False,
            "broker_order_submission": False,
            "advice_is_not_an_order": True,
            "default_execution_gate": "blocked_until_explicit_user_execution_path_exists",
        },
        "market_packet": packet,
    }
    if args.dry_run:
        progress(args, "DRY_RUN", "skipping LLM calls")
        return {
            **base_record,
            "forecast": {"status": "dry_run"},
            "forecast_prompt_payload": None,
            "forecast_raw_response": None,
            "advice": {"status": "dry_run"},
            "advice_prompt_payload": None,
            "advice_raw_response": None,
        }
    forecast_provider, forecast_model, forecast_payload, forecast_raw_response, forecast = call_response_with_fallback(
        args=args,
        purpose="pure_llm_stock_forecast",
        provider=provider,
        model=model,
        system_message=FORECAST_SYSTEM_MESSAGE,
        user_message=FORECAST_USER_MESSAGE,
        json_schema=FORECAST_JSON_SCHEMA,
        item={"market_packet_json": json.dumps(packet, separators=(",", ":"), default=str)},
        use_web_search=False,
    )
    forecast = normalize_forecast(forecast, packet)
    progress(args, "LLM_FORECAST", "pure LLM forecast received", provider=forecast_provider, model=forecast_model)
    progress(args, "QUALITY", "reviewing pure LLM forecast quality", provider=forecast_provider, model=forecast_model)
    forecast_quality = run_forecast_quality_review(args=args, provider=forecast_provider, model=forecast_model, packet=packet, forecast=forecast, external_evidence=prompt_evidence)
    progress(args, "QUALITY", "forecast quality review ready", status=forecast_quality.get("status"))
    technical_packet = build_ceo_technical_packet(packet=packet, forecast=forecast, forecast_quality=forecast_quality)
    strategy_context = build_strategy_context_for_pure_path(args=args, technical_packet=technical_packet)
    external_evidence["strategy_knowledge_context"] = strategy_context
    packet.setdefault("external_llm_evidence", {})["strategy_knowledge_context"] = compact_prompt_value(strategy_context, depth=0)
    progress(args, "STRUCTURE", "reading market structure/liquidity before CEO advice", provider=forecast_provider, model=forecast_model)
    market_structure_read = run_market_structure_liquidity_read(
        args=args,
        provider=forecast_provider,
        model=forecast_model,
        packet=packet,
        forecast=forecast,
        forecast_quality=forecast_quality,
        strategy_context=strategy_context,
    )
    external_evidence["market_structure_liquidity_read"] = market_structure_read
    packet.setdefault("external_llm_evidence", {})["market_structure_liquidity_read"] = compact_prompt_value(market_structure_read, depth=0)
    prompt_evidence = compact_external_evidence_for_prompt(external_evidence)
    progress(args, "STRUCTURE", "market structure/liquidity read ready", status=market_structure_read.get("status"), bias=market_structure_read.get("structure_bias"))
    ceo_handoff_packet = build_compact_ceo_handoff_packet(packet=packet, forecast=forecast, forecast_quality=forecast_quality)
    portfolio_context = build_portfolio_context(args)
    progress(
        args,
        "CEO_ADVICE",
        "calling original CEO prompt with compact handoff packet",
        prompt="autonomous_trader",
        provider=ceo_provider,
        model=ceo_model,
    )
    advice_payload, advice_raw_response, advice = call_ceo_advice(
        args=args,
        provider=ceo_provider,
        model=ceo_model,
        technical_packet=ceo_handoff_packet,
        portfolio_context=portfolio_context,
    )
    progress(args, "CEO_ADVICE", "CEO advice received", decision=advice.get("decision"))
    return {
        **base_record,
        "forecast": forecast,
        "external_llm_evidence": external_evidence,
        "prompt_external_llm_evidence": prompt_evidence,
        "forecast_quality_review": forecast_quality,
        "strategy_knowledge_context": strategy_context,
        "market_structure_liquidity_read": market_structure_read,
        "forecast_provider": forecast_provider,
        "forecast_model": forecast_model,
        "forecast_prompt_payload": forecast_payload,
        "forecast_raw_response": forecast_raw_response,
        "advice": advice,
        "advice_prompt_source": "market_forecasting_engine.llm_trader.prompts.autonomous_trader",
        "ceo_technical_packet": technical_packet,
        "ceo_handoff_packet": ceo_handoff_packet,
        "portfolio_context": portfolio_context,
        "execution_gate": {
            "execution_allowed": False,
            "reason": "Pure LLM stock forecaster produces advice only. No stock broker execution path is configured here.",
            "hard_blocks": ["no_stock_broker_execution_path_configured"],
        },
        "advice_prompt_payload": advice_payload,
        "advice_raw_response": advice_raw_response,
    }


def call_response_with_fallback(
    *,
    args: argparse.Namespace,
    purpose: str,
    provider: str,
    model: str,
    system_message: str,
    user_message: str,
    json_schema: dict[str, Any],
    item: dict[str, Any],
    use_web_search: bool,
) -> tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    attempts = [(resolve_llm_provider(provider), resolve_llm_model(model, provider=provider), "primary")]
    fallback_provider = resolve_llm_provider(getattr(args, "fallback_llm_provider", None) or "openai")
    fallback_model = resolve_llm_model(getattr(args, "fallback_llm_model", None) or ModelName.GPT_5_4_MINI_2026_03_17.value, provider=fallback_provider)
    if (fallback_provider, fallback_model) != (attempts[0][0], attempts[0][1]):
        attempts.append((fallback_provider, fallback_model, "fallback"))
    errors: list[str] = []
    for attempt_provider, attempt_model, route in attempts:
        try:
            progress(args, "LLM_CALL", "calling LLM", purpose=purpose, route=route, provider=attempt_provider, model=attempt_model)
            payload, raw_response, parsed = call_response(
                client=openai_client_for_provider(attempt_provider, timeout=float(args.llm_timeout)),
                provider=attempt_provider,
                model=attempt_model,
                system_message=system_message,
                user_message=user_message,
                json_schema=json_schema,
                item=item,
                use_web_search=bool(use_web_search and attempt_provider in {"openai", "llm_studio"}),
                search_context_size=args.search_context_size,
                reasoning_effort=args.reasoning_effort,
                usage_context={"purpose": purpose, "ticker": args.ticker.upper(), "provider": attempt_provider, "route": route},
            )
            if route == "fallback":
                parsed = {**parsed, "_fallback_from": attempts[0][0], "_fallback_reason": "; ".join(errors[-2:])}
            return attempt_provider, attempt_model, payload, raw_response, parsed
        except Exception as exc:
            message = f"{attempt_provider}/{attempt_model}: {type(exc).__name__}: {exc}"
            errors.append(message)
            progress(args, "LLM_CALL", "LLM call failed", purpose=purpose, route=route, provider=attempt_provider, model=attempt_model, error=f"{type(exc).__name__}: {exc}")
    raise RuntimeError("All LLM routes failed for " + purpose + ": " + " | ".join(errors))


def load_or_build_external_evidence(args: argparse.Namespace, *, prices: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    if args.external_evidence_json:
        return load_external_evidence(args.external_evidence_json)
    if args.skip_llm_evidence:
        return {}
    return build_default_llm_external_evidence(args, prices=prices, metadata=metadata)


def build_default_llm_external_evidence(args: argparse.Namespace, *, prices: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    from market_forecasting_engine.pure_llm_virtual_trader_agent import (
        PureLLMVirtualTraderConfig,
        handoff_enrichment,
        handoff_fresh_news,
        handoff_fundamentals,
        handoff_long_term_synthesis,
        handoff_market_intelligence,
        run_llm_enrichment,
        run_llm_fresh_news,
        run_llm_fundamentals,
        run_llm_long_term_source_synthesis,
        run_llm_market_intelligence,
    )

    config = PureLLMVirtualTraderConfig(
        project_dir=Path.cwd(),
        env_file=args.llm_env_file,
        provider=args.provider,
        start=args.start,
        interval=args.interval,
        bars=int(args.bars),
        trader_profile=args.trader_profile,
        planner_provider=args.llm_provider,
        planner_model=args.llm_model,
        fallback_provider=args.fallback_llm_provider,
        fallback_model=args.fallback_llm_model,
        forecast_llm_provider=args.llm_provider,
        forecast_llm_model=args.llm_model,
        ceo_llm_provider=args.ceo_llm_provider,
        ceo_llm_model=args.ceo_llm_model,
        reasoning_effort=args.reasoning_effort,
        search_context_size=args.search_context_size,
        llm_timeout_seconds=int(args.llm_timeout),
        compact_llm_handoffs=True,
        progress=not bool(args.no_progress),
    )
    ticker = args.ticker.upper()
    company = args.company or ticker
    last_price = float(prices[args.target_column].iloc[-1])
    candidate = {
        "ticker": ticker,
        "company": company,
        "priority": "high",
        "candidate_reason": "User-requested single-ticker pure LLM forecast.",
        "main_catalysts": [],
        "main_risks": [],
        "last_price": last_price,
        "data_provider": metadata.get("provider"),
    }
    broker_state = {
        "status": "standalone_single_ticker_forecast",
        "account": {
            "status": "not_connected",
            "equity": args.account_equity,
            "cash": None,
            "buying_power": None,
            "trading_blocked": None,
        },
        "clock": None,
        "positions": [],
        "open_orders": [],
    }
    memory = {
        "mode": "standalone_single_ticker_forecast",
        "ticker": ticker,
        "note": "No agent memory used; evidence created for this one forecast run.",
    }

    progress(args, "EVIDENCE", "building default LLM external evidence", ticker=ticker)
    market_intelligence = run_llm_market_intelligence(config=config, broker_state=broker_state, memory=memory)
    market_handoff = handoff_market_intelligence(market_intelligence, config=config)
    enrichment = run_llm_enrichment(config=config, candidate=candidate, market_intelligence=market_handoff)
    enrichment_handoff = handoff_enrichment(enrichment, config=config)
    fresh_news = run_llm_fresh_news(config=config, candidate=candidate, market_intelligence=market_handoff)
    fresh_news_handoff = handoff_fresh_news(fresh_news, config=config)
    fundamentals = run_llm_fundamentals(config=config, candidate=candidate, fresh_news=fresh_news_handoff)
    fundamentals_handoff = handoff_fundamentals(fundamentals, config=config)
    long_term_synthesis = run_llm_long_term_source_synthesis(
        config=config,
        candidate=candidate,
        market_intelligence=market_handoff,
        enrichment=enrichment_handoff,
        fresh_news=fresh_news_handoff,
        fundamentals=fundamentals_handoff,
    )
    progress(args, "EVIDENCE", "default LLM external evidence ready", ticker=ticker)
    return {
        "ticker": ticker,
        "candidate": candidate,
        "market_intelligence_llm": market_intelligence,
        "generic_enrichment_llm": enrichment,
        "fresh_news_llm": fresh_news,
        "fundamentals_llm": fundamentals,
        "long_term_source_synthesis_llm": long_term_synthesis,
        "policy": {
            "single_ticker_default_evidence": True,
            "regular_virtual_trader_enrichment_used": False,
            "regular_long_term_source_collectors_used": False,
            "all_sections_created_by_llm": True,
        },
    }


def compact_external_evidence_for_prompt(evidence: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(evidence, dict) or not evidence:
        return {}
    keep = {
        "ticker",
        "candidate",
        "market_intelligence_llm",
        "generic_enrichment_llm",
        "fresh_news_llm",
        "fundamentals_llm",
        "long_term_source_synthesis_llm",
        "strategy_knowledge_context",
        "market_structure_liquidity_read",
        "policy",
    }
    compact = {key: compact_prompt_value(value, depth=0) for key, value in evidence.items() if key in keep}
    compact["prompt_compaction_policy"] = {
        "raw_llm_payloads_removed": True,
        "raw_llm_responses_removed": True,
        "max_string_chars": 1200,
        "max_list_items": 8,
    }
    return compact


def compact_prompt_value(value: Any, *, depth: int) -> Any:
    if depth > 4:
        return summarize_scalar(value, max_chars=300)
    if isinstance(value, dict):
        output = {}
        for key, child in value.items():
            if key in {"llm_prompt_payload", "llm_raw_response", "forecast_prompt_payload", "forecast_raw_response"}:
                continue
            output[key] = compact_prompt_value(child, depth=depth + 1)
        return output
    if isinstance(value, list):
        return [compact_prompt_value(item, depth=depth + 1) for item in value[:8]]
    return summarize_scalar(value, max_chars=1200)


def summarize_scalar(value: Any, *, max_chars: int) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "...[truncated]"
    return value


def load_stock_prices(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = load_prices_with_provider(
        str(args.provider).lower(),
        DataRequest(ticker=args.ticker, start=args.start, end=args.end, interval=args.interval, target_column=args.target_column),
        store=None,
        use_cache=True,
        refresh_cache=True,
    )
    prices = normalize_price_frame(result.frame, target_column=args.target_column).dropna(subset=[args.target_column])
    if prices.empty:
        raise RuntimeError(f"No usable price rows loaded for {args.ticker}.")
    return prices, result.metadata


def build_market_packet(
    prices: pd.DataFrame,
    *,
    ticker: str,
    company: str,
    metadata: dict[str, Any],
    bar_count: int,
) -> dict[str, Any]:
    target = "close"
    latest_price = float(prices[target].iloc[-1])
    recent = prices.tail(max(5, min(int(bar_count), 60)))
    return {
        "ticker": ticker.upper(),
        "company": company,
        "as_of_utc": datetime.now(UTC).isoformat(),
        "last_bar_date": pd.Timestamp(prices.index[-1]).date().isoformat(),
        "last_price": round(latest_price, 4),
        "data_provider_metadata": compact_metadata(metadata),
        "requested_forecast_horizons": [
            {"horizon": "1_day", "trading_days": 1},
            {"horizon": "1_week", "trading_days": 5},
            {"horizon": "1_month", "trading_days": 21},
        ],
        "descriptive_context_not_forecast": {
            "return_1d_pct": trailing_return_pct(prices[target], 1),
            "return_5d_pct": trailing_return_pct(prices[target], 5),
            "return_21d_pct": trailing_return_pct(prices[target], 21),
            "return_63d_pct": trailing_return_pct(prices[target], 63),
            "close_20d_high": round(float(prices[target].tail(20).max()), 4),
            "close_20d_low": round(float(prices[target].tail(20).min()), 4),
            "close_60d_high": round(float(prices[target].tail(60).max()), 4),
            "close_60d_low": round(float(prices[target].tail(60).min()), 4),
        },
        "recent_daily_bars_columns": ["date", "open", "high", "low", "close", "volume"],
        "recent_daily_bars": [
            [
                pd.Timestamp(index).date().isoformat(),
                rounded_float(row.get("open", row[target])),
                rounded_float(row.get("high", row[target])),
                rounded_float(row.get("low", row[target])),
                rounded_float(row[target]),
                None if pd.isna(row.get("volume")) else int(float(row.get("volume"))),
            ]
            for index, row in recent.iterrows()
        ],
    }


def normalize_forecast(forecast: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    forecast = dict(forecast)
    forecast["ticker"] = str(forecast.get("ticker") or packet["ticker"]).upper()
    forecast["company"] = str(forecast.get("company") or packet["company"])
    forecast["current_price"] = float(forecast.get("current_price") or packet["last_price"])
    by_horizon = {str(item.get("horizon")): item for item in forecast.get("forecasts", []) if isinstance(item, dict)}
    normalized = []
    for horizon, trading_days in [("1_day", 1), ("1_week", 5), ("1_month", 21)]:
        item = dict(by_horizon.get(horizon) or {})
        item["horizon"] = horizon
        item["trading_days"] = trading_days
        normalized.append(item)
    forecast["forecasts"] = normalized
    return forecast


def run_forecast_quality_review(
    *,
    args: argparse.Namespace,
    provider: str,
    model: str,
    packet: dict[str, Any],
    forecast: dict[str, Any],
    external_evidence: dict[str, Any],
) -> dict[str, Any]:
    try:
        actual_provider, actual_model, payload, raw_response, parsed = call_response_with_fallback(
            args=args,
            purpose="pure_llm_forecast_quality_review",
            provider=provider,
            model=model,
            system_message=FORECAST_QUALITY_SYSTEM_MESSAGE,
            user_message=FORECAST_QUALITY_USER_MESSAGE,
            json_schema=FORECAST_QUALITY_JSON_SCHEMA,
            item={
                "market_packet_json": json.dumps(packet, separators=(",", ":"), default=str),
                "forecast_json": json.dumps(forecast, separators=(",", ":"), default=str),
                "external_evidence_json": json.dumps(external_evidence, separators=(",", ":"), default=str),
            },
            use_web_search=False,
        )
        return {**parsed, "provider": actual_provider, "model": actual_model, "llm_prompt_payload": payload, "llm_raw_response": raw_response}
    except Exception as exc:
        return {
            "status": "failed",
            "provider": provider,
            "model": model,
            "validation_type": "llm_forecast_quality_review_failed_not_walk_forward_validation",
            "quality_summary": f"Forecast-quality LLM review failed: {type(exc).__name__}: {exc}",
            "forecast_consistency_read": "",
            "evidence_alignment_read": "",
            "confidence_adjustment": "lower",
            "usable_for_ceo": True,
            "limitations": ["No deterministic or walk-forward model validation is available in this pure LLM path."],
            "red_flags": ["Forecast quality review failed before CEO advice."],
        }


def build_strategy_context_for_pure_path(*, args: argparse.Namespace, technical_packet: dict[str, Any]) -> dict[str, Any]:
    if getattr(args, "disable_strategy_knowledge", False):
        return {"status": "skipped", "reason": "disabled by --disable-strategy-knowledge"}
    try:
        from market_forecasting_engine.strategy_knowledge import StrategyKnowledgeRequest, build_strategy_knowledge_context

        progress(args, "STRATEGY", "retrieving durable strategy knowledge for CEO context")
        strategy_report = {
            **technical_packet,
            "decision_view": {
                **(technical_packet.get("decision_view", {}) if isinstance(technical_packet.get("decision_view"), dict) else {}),
                "market_structure_focus": {
                    "concepts": [
                        "swing lows",
                        "liquidity below equal lows",
                        "liquidity sweep",
                        "break of structure",
                        "fair value gap",
                        "former support turns resistance",
                        "downside liquidity target",
                    ]
                },
            },
        }
        context = build_strategy_knowledge_context(
            strategy_report,
            StrategyKnowledgeRequest(
                ticker=args.ticker,
                corpus_dir=args.strategy_knowledge_corpus_dir,
                index_path=args.strategy_knowledge_index,
                llm_env_file=args.llm_env_file,
                max_chunks=int(args.strategy_knowledge_max_chunks),
                rebuild_index=bool(args.strategy_knowledge_rebuild_index),
                timeout_seconds=int(args.llm_timeout),
                include_pdfs=False,
            ),
        )
        progress(args, "STRATEGY", "strategy knowledge context ready", status=context.get("status"))
        return context
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "decision_policy": {
                "feeds_ceo_llm": True,
                "overrides_model_validation": False,
                "overrides_risk_gates": False,
            },
        }


def run_market_structure_liquidity_read(
    *,
    args: argparse.Namespace,
    provider: str,
    model: str,
    packet: dict[str, Any],
    forecast: dict[str, Any],
    forecast_quality: dict[str, Any],
    strategy_context: dict[str, Any],
) -> dict[str, Any]:
    try:
        actual_provider, actual_model, payload, raw_response, parsed = call_response_with_fallback(
            args=args,
            purpose="pure_llm_market_structure_liquidity_read",
            provider=provider,
            model=model,
            system_message=MARKET_STRUCTURE_SYSTEM_MESSAGE,
            user_message=MARKET_STRUCTURE_USER_MESSAGE,
            json_schema=MARKET_STRUCTURE_JSON_SCHEMA,
            item={
                "market_packet_json": json.dumps(packet, separators=(",", ":"), default=str),
                "forecast_json": json.dumps(forecast, separators=(",", ":"), default=str),
                "forecast_quality_json": json.dumps(forecast_quality, separators=(",", ":"), default=str),
                "strategy_knowledge_json": json.dumps(strategy_context, separators=(",", ":"), default=str),
            },
            use_web_search=False,
        )
        return {**parsed, "provider": actual_provider, "model": actual_model, "llm_prompt_payload": payload, "llm_raw_response": raw_response}
    except Exception as exc:
        return {
            "status": "failed",
            "provider": provider,
            "model": model,
            "structure_bias": "unclear",
            "swing_low_read": "",
            "liquidity_pools": [],
            "sweep_or_break_read": f"Market-structure LLM read failed: {type(exc).__name__}: {exc}",
            "support_resistance_flip_read": "",
            "fair_value_gap_read": "",
            "expected_path": "",
            "ceo_implication": "Do not rely on the market-structure/liquidity read for this run.",
            "entry_timing_notes": [],
            "invalidation_notes": [],
            "limitations": ["Market-structure/liquidity read failed."],
        }


def build_ceo_technical_packet(*, packet: dict[str, Any], forecast: dict[str, Any], forecast_quality: dict[str, Any] | None = None) -> dict[str, Any]:
    """Adapt the pure-LLM forecast into the existing CEO prompt packet shape."""
    compact_bars = [
        {"date": item[0], "close": item[4]}
        for item in (packet.get("recent_daily_bars") or [])[-8:]
        if isinstance(item, list) and len(item) >= 5
    ]
    return {
        "ticker": packet["ticker"],
        "company": packet["company"],
        "current_price": packet["last_price"],
        "source_path": "pure_llm_stock_forecaster",
        "forecast_policy": {
            "deterministic_forecast_engine_used": False,
            "forecast_source": "pure_llm_forecast_step",
            "raw_market_packet_only": True,
        },
        "forecasts": forecast.get("forecasts", []),
        "suggested_action": "Hold",
        "risk_level": "Medium",
        "risk_warning": "Pure LLM forecast is not walk-forward validated; use advice as governed context only.",
        "technical_view": {
            "pure_llm_forecast": forecast,
            "forecast_quality_review": forecast_quality or {},
            "external_llm_evidence": packet.get("external_llm_evidence", {}),
            "descriptive_context_not_forecast": packet.get("descriptive_context_not_forecast", {}),
            "recent_close_tape": compact_bars,
        },
        "decision_view": {
            "final_governed_action": "Hold",
            "pure_llm_stock_path": True,
            "long_term_context": (packet.get("external_llm_evidence", {}) or {}).get("long_term_source_synthesis_llm", {}),
            "fresh_news": (packet.get("external_llm_evidence", {}) or {}).get("fresh_news_llm", {}),
            "fundamentals": (packet.get("external_llm_evidence", {}) or {}).get("fundamentals_llm", {}),
            "strategy_knowledge_context": (packet.get("external_llm_evidence", {}) or {}).get("strategy_knowledge_context", {}),
            "market_structure_liquidity_read": (packet.get("external_llm_evidence", {}) or {}).get("market_structure_liquidity_read", {}),
            "forecast_quality_review": forecast_quality or {},
            "execution_policy": {
                "broker_order_submission": False,
                "advice_is_not_an_order": True,
            },
        },
        "operations_view": {
            "validation": {
                "status": "not_applicable",
                "reason": "No deterministic forecast validation is run in the pure LLM stock path.",
            }
        },
        "portfolio_view": {},
    }


CEO_HANDOFF_EXCLUDED_KEYS = {
    "advice_prompt_payload",
    "advice_raw_response",
    "chain_of_thought",
    "forecast_prompt_payload",
    "forecast_raw_response",
    "initial_analysis",
    "llm_prompt_payload",
    "llm_raw_response",
    "pure_llm_artifact",
    "raw_response",
    "reasoning",
    "response_data",
    "scratchpad",
    "thoughts",
}

CEO_HANDOFF_FILLER_PATTERNS = [
    (re.compile(r"\bI\s+(?:need|want|will|would like|am going)\s+to\s+", re.IGNORECASE), ""),
    (re.compile(r"\bwe\s+(?:need|want|will|would like|are going)\s+to\s+", re.IGNORECASE), ""),
    (re.compile(r"\bit\s+is\s+(?:important|worth)\s+(?:to\s+)?(?:note|mention)\s+that\s+", re.IGNORECASE), ""),
    (re.compile(r"\b(?:please\s+)?note\s+that\s+", re.IGNORECASE), ""),
    (re.compile(r"\bit\s+(?:appears|seems|looks)\s+that\s+", re.IGNORECASE), ""),
    (re.compile(r"\bthis\s+(?:suggests|indicates|means)\s+that\s+", re.IGNORECASE), ""),
    (re.compile(r"\bbased\s+on\s+(?:the\s+)?(?:available\s+)?(?:information|data|context|evidence),?\s*", re.IGNORECASE), ""),
    (re.compile(r"\bgiven\s+(?:the\s+)?(?:available\s+)?(?:information|data|context|evidence),?\s*", re.IGNORECASE), ""),
    (re.compile(r"\bin\s+(?:my\s+)?(?:view|opinion),?\s*", re.IGNORECASE), ""),
    (re.compile(r"\b(?:a|an|the)\s+", re.ASCII), ""),
]


def build_compact_ceo_handoff_packet(*, packet: dict[str, Any], forecast: dict[str, Any], forecast_quality: dict[str, Any] | None = None) -> dict[str, Any]:
    """Single-copy facts packet for the original CEO prompt."""
    external = packet.get("external_llm_evidence", {}) if isinstance(packet.get("external_llm_evidence"), dict) else {}
    descriptive = packet.get("descriptive_context_not_forecast", {}) if isinstance(packet.get("descriptive_context_not_forecast"), dict) else {}
    compact_bars = [
        {"date": item[0], "close": item[4]}
        for item in (packet.get("recent_daily_bars") or [])[-6:]
        if isinstance(item, list) and len(item) >= 5
    ]
    return ceo_compact_dict(
        {
            "ticker": packet.get("ticker"),
            "company": packet.get("company"),
            "current_price": packet.get("last_price"),
            "source_path": "pure_llm_stock_forecaster",
            "forecast_policy": {
                "deterministic_forecast_engine_used": False,
                "forecast_source": "llm",
                "advice_is_not_order": True,
            },
            "forecast_facts": {
                "forecasts": forecast.get("forecasts", []),
                "overall_view": forecast.get("overall_view"),
                "main_risks": forecast.get("main_risks"),
                "data_limitations": forecast.get("data_limitations"),
            },
            "quality_facts": ceo_compact_dict(forecast_quality or {}),
            "market_facts": {
                "descriptive_context_not_forecast": descriptive,
                "recent_close_tape": compact_bars,
            },
            "evidence_facts": {
                "candidate": external.get("candidate", {}),
                "market_intelligence": external.get("market_intelligence_llm", {}),
                "fresh_news": external.get("fresh_news_llm", {}),
                "fundamentals": external.get("fundamentals_llm", {}),
                "long_term_context": external.get("long_term_source_synthesis_llm", {}),
                "strategy_knowledge": external.get("strategy_knowledge_context", {}),
                "market_structure_liquidity": external.get("market_structure_liquidity_read", {}),
            },
            "risk_warning": "Pure LLM forecast not walk-forward validated; use advice as governed context only.",
            "suggested_action": "Hold",
            "risk_level": "Medium",
        }
    )


def ceo_compact_dict(value: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, row in value.items():
        if key in CEO_HANDOFF_EXCLUDED_KEYS or row is None:
            continue
        output[key] = ceo_compact_value(row)
    return output


def ceo_compact_value(value: Any) -> Any:
    if isinstance(value, str):
        return ceo_compact_text(value)
    if isinstance(value, list):
        return [ceo_compact_value(row) for row in value]
    if isinstance(value, dict):
        return ceo_compact_dict(value)
    return value


def ceo_compact_text(value: str) -> str:
    text = " ".join(str(value or "").split())
    for pattern, replacement in CEO_HANDOFF_FILLER_PATTERNS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ;,")


def load_external_evidence(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    evidence_path = Path(path).expanduser()
    if not evidence_path.exists():
        return {
            "status": "missing",
            "path": str(evidence_path),
            "warning": "External evidence path was supplied but the file was not found.",
        }
    try:
        loaded = json.loads(evidence_path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {"status": "invalid", "path": str(evidence_path)}
    except Exception as exc:
        return {"status": "unreadable", "path": str(evidence_path), "error": f"{type(exc).__name__}: {exc}"}


def build_portfolio_context(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "holding_status": args.holding_status,
        "entry_price": args.entry_price,
        "quantity": args.quantity,
        "position_value": args.position_value,
        "account_equity": args.account_equity,
        "notes": args.portfolio_notes,
    }


def call_ceo_advice(
    *,
    args: argparse.Namespace,
    provider: str,
    model: str,
    technical_packet: dict[str, Any],
    portfolio_context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    item = {
        "today": datetime.now(UTC).date().isoformat(),
        "ticker": args.ticker.upper(),
        "trader_name": args.trader_name,
        "trader_profile_json": json.dumps(trader_profiles[args.trader_profile], separators=(",", ":"), default=str),
        "portfolio_context_json": json.dumps(portfolio_context, separators=(",", ":"), default=str),
        "technical_packet_json": json.dumps(technical_packet, separators=(",", ":"), default=str),
    }
    actual_provider, actual_model, payload, raw_response, parsed = call_response_with_fallback(
        args=args,
        purpose="pure_llm_stock_ceo_original",
        provider=provider,
        model=model,
        system_message=autonomous_trader.system_message,
        user_message=autonomous_trader.user_message,
        json_schema=autonomous_trader.json_schema,
        item=item,
        use_web_search=False,
    )
    parsed = {**parsed, "provider": actual_provider, "model": actual_model}
    return payload, raw_response, parsed


def trailing_return_pct(series: pd.Series, periods: int) -> float | None:
    if len(series) <= periods:
        return None
    previous = float(series.iloc[-periods - 1])
    if previous == 0:
        return None
    return round(((float(series.iloc[-1]) / previous) - 1.0) * 100.0, 4)


def rounded_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), 4)


def compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": metadata.get("provider"),
        "request": metadata.get("request"),
        "rows": metadata.get("rows"),
        "source": metadata.get("source"),
    }


def _safe_name(value: str) -> str:
    return str(value).upper().replace("/", "_").replace(" ", "_")


def progress(args: argparse.Namespace, stage: str, message: str, **fields: Any) -> None:
    if getattr(args, "no_progress", False):
        return
    suffix = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    line = f"[pure-llm-stock] {datetime.now().strftime('%H:%M:%S')} {stage} | {message}"
    if suffix:
        line = f"{line} | {suffix}"
    print(line, flush=True)


if __name__ == "__main__":
    main()
