from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from market_forecasting_engine.alpaca_broker import AlpacaPaperBroker
from market_forecasting_engine.llm_model_catalog import DEFAULT_FULL_LLM_OPTIONS_MODEL, DEFAULT_FULL_LLM_OPTIONS_PROVIDER
from market_forecasting_engine.llm_trader.responses_api import call_response
from market_forecasting_engine.llm_trader.run import openai_client_for_provider, resolve_llm_model, resolve_llm_provider
from market_forecasting_engine.openai_models import ModelName


PLANNER_SYSTEM_MESSAGE = """
# Role: Pure LLM Alpaca Paper Trader Planner

You manage one isolated Alpaca paper account. You may use web search and your own reasoning to choose a small set of stock tickers to review.

This path is intentionally isolated from the regular virtual trader. Do not assume access to its scout, enrichment board, candidate files, forecast engine, or memory.

Rules:
- Use only the supplied Alpaca account/positions/orders, the isolated memory, and your web/search reasoning.
- Select tickers for pure LLM forecasting and CEO advice.
- Prefer existing positions and open-order symbols before new ideas.
- If the account is small, choose at most one or two actionable candidates.
- Do not place orders. The local safety layer handles paper-order eligibility.
- Return exactly one JSON object matching the schema.
""".strip()


PLANNER_USER_MESSAGE = """
Today:
{{ item.today }}

Configuration:
{{ item.config_json }}

Alpaca paper account state:
{{ item.broker_state_json }}

Isolated pure-LLM memory:
{{ item.memory_json }}

Task:
Choose the tickers this isolated pure-LLM paper trader should forecast this cycle and explain the plan.
""".strip()


PLANNER_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "pure_llm_virtual_trader_plan",
    "description": "Ticker selection and cadence plan for the isolated pure LLM Alpaca paper trader.",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "cycle_mode": {
                "type": "string",
                "enum": ["bootstrap_portfolio", "manage_positions", "monitor_orders", "risk_reduction", "monitor_only"],
            },
            "account_assessment": {"type": "string"},
            "market_assessment": {"type": "string"},
            "risk_posture": {"type": "string", "enum": ["risk_on", "neutral", "cautious", "risk_off"]},
            "forecast_tickers": {"type": "array", "items": {"type": "string"}},
            "max_candidates_to_forecast": {"type": "integer"},
            "ticker_rationales": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["ticker", "priority", "reason"],
                    "additionalProperties": False,
                },
            },
            "order_management_notes": {"type": "array", "items": {"type": "string"}},
            "next_wakeup_seconds": {"type": "integer"},
            "plan_rationale": {"type": "string"},
        },
        "required": [
            "cycle_mode",
            "account_assessment",
            "market_assessment",
            "risk_posture",
            "forecast_tickers",
            "max_candidates_to_forecast",
            "ticker_rationales",
            "order_management_notes",
            "next_wakeup_seconds",
            "plan_rationale",
        ],
        "additionalProperties": False,
    },
}


MARKET_INTELLIGENCE_SYSTEM_MESSAGE = """
# Role: Pure LLM Market Intelligence Analyst

You are the market-intelligence step for an isolated Alpaca paper-trading agent.
Use web search when available. Produce only current market, macro, political/regulatory, earnings, sector, and risk context relevant to deciding what stocks to inspect this cycle.
Do not place orders and do not make final ticker Buy/Hold/Sell decisions.
Return exactly one JSON object matching the schema.
""".strip()


MARKET_INTELLIGENCE_USER_MESSAGE = """
Today:
{{ item.today }}

Alpaca account/positions/orders:
{{ item.broker_state_json }}

Isolated memory:
{{ item.memory_json }}

Task:
Summarize current market conditions and risks for an autonomous paper stock trader.
""".strip()


MARKET_INTELLIGENCE_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "pure_llm_market_intelligence",
    "description": "LLM market intelligence for the isolated pure LLM trader.",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "market_regime": {"type": "string", "enum": ["risk_on", "neutral", "cautious", "risk_off"]},
            "summary": {"type": "string"},
            "macro_risks": {"type": "array", "items": {"type": "string"}},
            "sector_themes": {"type": "array", "items": {"type": "string"}},
            "events_to_watch": {"type": "array", "items": {"type": "string"}},
            "portfolio_implications": {"type": "array", "items": {"type": "string"}},
            "source_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["market_regime", "summary", "macro_risks", "sector_themes", "events_to_watch", "portfolio_implications", "source_notes"],
        "additionalProperties": False,
    },
}


SCOUT_SYSTEM_MESSAGE = """
# Role: Pure LLM Stock Scout

You are the scout step for an isolated pure-LLM Alpaca paper-trading agent.
Use web search and your market judgment to identify candidate stocks. This replaces the regular virtual trader's deterministic scout.

Rules:
- Prefer liquid U.S.-listed stocks suitable for Alpaca paper trading.
- Include existing positions/open-order symbols when they need review.
- For a very small account, choose fewer tickers and avoid high-priced names unless fractional trading is plausible.
- Do not make final Buy/Hold/Sell decisions; select candidates for deeper LLM enrichment and forecast.
- Return exactly one JSON object matching the schema.
""".strip()


SCOUT_USER_MESSAGE = """
Today:
{{ item.today }}

Market intelligence:
{{ item.market_intelligence_json }}

Alpaca account/positions/orders:
{{ item.broker_state_json }}

Isolated memory:
{{ item.memory_json }}

Max candidates:
{{ item.max_candidates }}

Task:
Select a small set of stock tickers for this cycle.
""".strip()


SCOUT_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "pure_llm_stock_scout",
    "description": "LLM-only candidate discovery for the isolated pure LLM trader.",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "scout_summary": {"type": "string"},
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "company": {"type": "string"},
                        "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                        "candidate_reason": {"type": "string"},
                        "main_catalysts": {"type": "array", "items": {"type": "string"}},
                        "main_risks": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["ticker", "company", "priority", "candidate_reason", "main_catalysts", "main_risks"],
                    "additionalProperties": False,
                },
            },
            "rejected_themes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["scout_summary", "candidates", "rejected_themes"],
        "additionalProperties": False,
    },
}


ENRICHMENT_SYSTEM_MESSAGE = """
# Role: Pure LLM Ticker Enrichment Analyst

You are the enrichment step for one stock ticker in an isolated pure-LLM paper trader.
Use web search where available. Build concise evidence for the later pure LLM forecast and CEO advice.
Do not use the regular virtual trader enrichment board or provider snapshots.
Return exactly one JSON object matching the schema.
""".strip()


ENRICHMENT_USER_MESSAGE = """
Today:
{{ item.today }}

Ticker:
{{ item.ticker }}

Candidate:
{{ item.candidate_json }}

Market intelligence:
{{ item.market_intelligence_json }}

Task:
Create LLM-based ticker enrichment: news, catalysts, fundamentals, sentiment, risks, and source notes.
""".strip()


ENRICHMENT_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "pure_llm_ticker_enrichment",
    "description": "LLM-only ticker enrichment for the isolated pure LLM trader.",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "company": {"type": "string"},
            "news_read": {"type": "string"},
            "fundamental_read": {"type": "string"},
            "sentiment_read": {"type": "string"},
            "bullish_evidence": {"type": "array", "items": {"type": "string"}},
            "bearish_evidence": {"type": "array", "items": {"type": "string"}},
            "key_risks": {"type": "array", "items": {"type": "string"}},
            "source_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "ticker",
            "company",
            "news_read",
            "fundamental_read",
            "sentiment_read",
            "bullish_evidence",
            "bearish_evidence",
            "key_risks",
            "source_notes",
        ],
        "additionalProperties": False,
    },
}


FRESH_NEWS_SYSTEM_MESSAGE = """
# Role: Pure LLM Fresh News Analyst

You are the fresh-news step for an isolated pure-LLM Alpaca paper-trading agent.
Use web search when available. If web search is unavailable, clearly state the freshness limitation.
Focus on recent company, sector, market, regulatory, earnings, analyst, and sentiment items that could affect the next 1 day, 1 week, and 1 month.
Do not make final Buy/Hold/Sell decisions.
Return exactly one JSON object matching the schema.
""".strip()


FRESH_NEWS_USER_MESSAGE = """
Today:
{{ item.today }}

Ticker:
{{ item.ticker }}

Candidate:
{{ item.candidate_json }}

Market intelligence:
{{ item.market_intelligence_json }}

Task:
Build the fresh-news evidence block for this ticker.
""".strip()


FRESH_NEWS_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "pure_llm_fresh_news",
    "description": "LLM-only fresh news evidence for the isolated pure LLM trader.",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "status": {"type": "string", "enum": ["executed", "limited", "failed"]},
            "freshness_read": {"type": "string"},
            "news_summary": {"type": "string"},
            "bullish_news": {"type": "array", "items": {"type": "string"}},
            "bearish_news": {"type": "array", "items": {"type": "string"}},
            "events_to_watch": {"type": "array", "items": {"type": "string"}},
            "source_notes": {"type": "array", "items": {"type": "string"}},
            "limitations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["ticker", "status", "freshness_read", "news_summary", "bullish_news", "bearish_news", "events_to_watch", "source_notes", "limitations"],
        "additionalProperties": False,
    },
}


FUNDAMENTALS_SYSTEM_MESSAGE = """
# Role: Pure LLM Fundamentals Analyst

You are the fundamentals step for an isolated pure-LLM Alpaca paper-trading agent.
Use web search when available. If exact provider fundamentals are unavailable, reason from available public company and market information and state limitations.
Do not call or rely on the regular agent's fundamentals providers, long-term source collectors, snapshots, or enrichment board.
Do not make final Buy/Hold/Sell decisions.
Return exactly one JSON object matching the schema.
""".strip()


FUNDAMENTALS_USER_MESSAGE = """
Today:
{{ item.today }}

Ticker:
{{ item.ticker }}

Candidate:
{{ item.candidate_json }}

Fresh news:
{{ item.fresh_news_json }}

Task:
Build the LLM-only fundamentals evidence block for this ticker. Explicitly check ROIC / return on invested capital when available, and say when it is unavailable or unclear.
""".strip()


FUNDAMENTALS_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "pure_llm_fundamentals",
    "description": "LLM-only fundamentals evidence for the isolated pure LLM trader.",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "status": {"type": "string", "enum": ["executed", "limited", "failed"]},
            "business_quality_read": {"type": "string"},
            "valuation_read": {"type": "string"},
            "balance_sheet_read": {"type": "string"},
            "growth_profitability_read": {"type": "string"},
            "analyst_and_estimate_read": {"type": "string"},
            "fundamental_bull_points": {"type": "array", "items": {"type": "string"}},
            "fundamental_bear_points": {"type": "array", "items": {"type": "string"}},
            "source_notes": {"type": "array", "items": {"type": "string"}},
            "limitations": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "ticker",
            "status",
            "business_quality_read",
            "valuation_read",
            "balance_sheet_read",
            "growth_profitability_read",
            "analyst_and_estimate_read",
            "fundamental_bull_points",
            "fundamental_bear_points",
            "source_notes",
            "limitations",
        ],
        "additionalProperties": False,
    },
}


LONG_TERM_SYNTHESIS_SYSTEM_MESSAGE = """
# Role: Pure LLM Long-Term Source Synthesis Analyst

You are the long-term source synthesis step for an isolated pure-LLM Alpaca paper-trading agent.
Mirror the role of the normal agent's long-term source synthesis, but do not use its provider snapshots, source collectors, enrichment board, memory, or strategy corpus.
Synthesize the candidate, market intelligence, fresh news, fundamentals, and enrichment into a long-term evidence read.
Do not make final Buy/Hold/Sell decisions.
Return exactly one JSON object matching the schema.
""".strip()


LONG_TERM_SYNTHESIS_USER_MESSAGE = """
Today:
{{ item.today }}

Ticker:
{{ item.ticker }}

Candidate:
{{ item.candidate_json }}

Market intelligence:
{{ item.market_intelligence_json }}

Generic enrichment:
{{ item.enrichment_json }}

Fresh news:
{{ item.fresh_news_json }}

Fundamentals:
{{ item.fundamentals_json }}

Task:
Create the long-term source synthesis evidence block for CEO context.
""".strip()


LONG_TERM_SYNTHESIS_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "pure_llm_long_term_source_synthesis",
    "description": "LLM-only long-term source synthesis for the isolated pure LLM trader.",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "status": {"type": "string", "enum": ["executed", "limited", "failed"]},
            "synthesis": {"type": "string"},
            "long_term_bull_case": {"type": "array", "items": {"type": "string"}},
            "long_term_bear_case": {"type": "array", "items": {"type": "string"}},
            "evidence_gaps": {"type": "array", "items": {"type": "string"}},
            "ceo_relevance": {"type": "string"},
            "source_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["ticker", "status", "synthesis", "long_term_bull_case", "long_term_bear_case", "evidence_gaps", "ceo_relevance", "source_notes"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class PureLLMVirtualTraderConfig:
    project_dir: str | Path = "/Users/ruddigarcia/Projects/invest"
    output_root: str | Path = "automated_forecasting_engine/runs/pure_llm_virtual_trader_agent"
    memory_path: str | Path = "automated_forecasting_engine/runs/pure_llm_virtual_trader/memory.json"
    env_file: str | Path | None = None
    once: bool = False
    dry_run: bool = False
    loop_interval_seconds: int = 14_400
    min_loop_interval_seconds: int = 900
    max_loop_interval_seconds: int = 21_600
    max_candidates: int = 2
    provider: str = "yahoo"
    start: str = "2024-01-01"
    interval: str = "1d"
    bars: int = 20
    risk_profile: str = "aggressive"
    trader_profile: str = "aggressive"
    planner_provider: str = DEFAULT_FULL_LLM_OPTIONS_PROVIDER
    planner_model: str = DEFAULT_FULL_LLM_OPTIONS_MODEL
    fallback_provider: str = "openai"
    fallback_model: str = ModelName.GPT_5_4_MINI_2026_03_17.value
    forecast_llm_provider: str = DEFAULT_FULL_LLM_OPTIONS_PROVIDER
    forecast_llm_model: str = DEFAULT_FULL_LLM_OPTIONS_MODEL
    ceo_llm_provider: str = "openai"
    ceo_llm_model: str = ModelName.GPT_5_4_2026_03_05.value
    reasoning_effort: str = "none"
    search_context_size: str = "low"
    llm_timeout_seconds: int = 900
    max_notional_per_trade: float = 25.0
    max_position_pct_equity: float = 0.25
    allow_market_closed_orders: bool = False
    allow_repeated_symbol_orders: bool = False
    compact_llm_handoffs: bool = True
    progress: bool = True


def main() -> None:
    args = build_parser().parse_args()
    config = PureLLMVirtualTraderConfig(
        project_dir=args.project_dir,
        output_root=args.output_root,
        memory_path=args.memory_path,
        env_file=args.env_file,
        once=args.once,
        dry_run=args.dry_run,
        loop_interval_seconds=args.loop_interval_seconds,
        min_loop_interval_seconds=args.min_loop_interval_seconds,
        max_loop_interval_seconds=args.max_loop_interval_seconds,
        max_candidates=args.max_candidates,
        provider=args.provider,
        start=args.start,
        interval=args.interval,
        bars=args.bars,
        risk_profile=args.risk_profile,
        trader_profile=args.trader_profile,
        planner_provider=args.planner_provider,
        planner_model=args.planner_model,
        fallback_provider=args.fallback_provider,
        fallback_model=args.fallback_model,
        forecast_llm_provider=args.llm_provider,
        forecast_llm_model=args.llm_model,
        ceo_llm_provider=args.ceo_llm_provider,
        ceo_llm_model=args.ceo_llm_model,
        reasoning_effort=args.reasoning_effort,
        search_context_size=args.search_context_size,
        llm_timeout_seconds=args.llm_timeout_seconds,
        max_notional_per_trade=args.max_notional_per_trade,
        max_position_pct_equity=args.max_position_pct_equity,
        allow_market_closed_orders=args.allow_market_closed_orders,
        allow_repeated_symbol_orders=args.allow_repeated_symbol_orders,
        compact_llm_handoffs=args.compact_llm_handoffs,
        progress=not args.no_progress,
    )
    run_agent(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the isolated pure-LLM Alpaca paper trader.")
    parser.add_argument("--project-dir", default="/Users/ruddigarcia/Projects/invest")
    parser.add_argument("--output-root", default="automated_forecasting_engine/runs/pure_llm_virtual_trader_agent")
    parser.add_argument("--memory-path", default="automated_forecasting_engine/runs/pure_llm_virtual_trader/memory.json")
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--loop-interval-seconds", type=int, default=14_400)
    parser.add_argument("--min-loop-interval-seconds", type=int, default=900)
    parser.add_argument("--max-loop-interval-seconds", type=int, default=21_600)
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--provider", default="yahoo")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--bars", type=int, default=20)
    parser.add_argument("--risk-profile", choices=("conservative", "medium", "aggressive"), default="aggressive")
    parser.add_argument("--trader-profile", choices=("conservative", "medium", "aggressive"), default="aggressive")
    parser.add_argument("--planner-provider", choices=("openai", "huggingface", "bedrock", "llm_studio"), default=DEFAULT_FULL_LLM_OPTIONS_PROVIDER)
    parser.add_argument("--planner-model", default=DEFAULT_FULL_LLM_OPTIONS_MODEL)
    parser.add_argument("--fallback-provider", choices=("openai", "huggingface", "bedrock", "llm_studio"), default="openai")
    parser.add_argument("--fallback-model", default=ModelName.GPT_5_4_MINI_2026_03_17.value)
    parser.add_argument("--llm-provider", choices=("openai", "huggingface", "bedrock", "llm_studio"), default=DEFAULT_FULL_LLM_OPTIONS_PROVIDER)
    parser.add_argument("--llm-model", default=DEFAULT_FULL_LLM_OPTIONS_MODEL)
    parser.add_argument("--ceo-llm-provider", choices=("openai", "huggingface", "bedrock", "llm_studio"), default="openai")
    parser.add_argument("--ceo-llm-model", default=ModelName.GPT_5_4_2026_03_05.value)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--search-context-size", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--llm-timeout-seconds", type=int, default=900)
    parser.add_argument("--max-notional-per-trade", type=float, default=25.0)
    parser.add_argument("--max-position-pct-equity", type=float, default=0.25)
    parser.add_argument("--allow-market-closed-orders", action="store_true")
    parser.add_argument("--allow-repeated-symbol-orders", action="store_true")
    parser.add_argument("--compact-llm-handoffs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def run_agent(config: PureLLMVirtualTraderConfig) -> None:
    project_dir = Path(config.project_dir).expanduser()
    os.chdir(project_dir)
    load_env_override(config.env_file)
    cycle = 0
    progress(config, f"isolated pure LLM agent started output={config.output_root} memory={config.memory_path}")
    while True:
        cycle += 1
        result = run_cycle(config, cycle=cycle)
        progress(
            config,
            f"cycle {cycle} complete tickers={','.join(result.get('forecast_tickers', [])) or 'none'} "
            f"orders={len(result.get('order_plans', []))} next={result.get('next_wakeup_seconds')}s",
        )
        if config.once:
            break
        time.sleep(max(30, int(result.get("next_wakeup_seconds") or config.loop_interval_seconds)))


def run_cycle(config: PureLLMVirtualTraderConfig, *, cycle: int) -> dict[str, Any]:
    output_root = Path(config.output_root).expanduser()
    cycle_dir = output_root / datetime.now(UTC).strftime("cycle_%Y%m%d_%H%M%S")
    cycle_dir.mkdir(parents=True, exist_ok=True)
    progress(config, f"cycle {cycle} started audit_dir={cycle_dir}")
    progress(config, "loading isolated memory")
    memory = load_memory(config.memory_path)
    progress(config, "loading Alpaca synced paper account state")
    broker_state = load_broker_state()
    progress(
        config,
        "broker state loaded "
        f"status={broker_state.get('status')} positions={len(broker_state.get('positions', []) or [])} "
        f"open_orders={len(broker_state.get('open_orders', []) or [])}",
    )
    progress(config, "LLM market intelligence started")
    market_intelligence = run_llm_market_intelligence(config=config, broker_state=broker_state, memory=memory)
    market_intelligence_handoff = handoff_market_intelligence(market_intelligence, config=config)
    progress(
        config,
        "LLM market intelligence finished "
        f"status={market_intelligence.get('status')} provider={market_intelligence.get('provider')} model={market_intelligence.get('model')}",
    )
    progress(config, "LLM portfolio planner started")
    plan = run_llm_planner(
        config=config,
        broker_state=broker_state,
        memory=memory,
        market_intelligence=market_intelligence_handoff,
        scout={},
        enrichments=[],
        evidence_packets=[],
    )
    progress(
        config,
        "LLM portfolio planner finished "
        f"status={plan.get('status')} mode={plan.get('cycle_mode')} risk={plan.get('risk_posture')} "
        f"provider={plan.get('provider')} model={plan.get('model')}",
    )
    progress(config, "LLM scout started")
    scout = run_llm_scout(config=config, broker_state=broker_state, memory=memory, market_intelligence=market_intelligence_handoff)
    scout_handoff = handoff_scout(scout, config=config)
    candidates = normalize_candidates(scout.get("candidates", []), broker_state=broker_state, max_candidates=config.max_candidates)
    progress(
        config,
        "LLM scout finished "
        f"status={scout.get('status')} candidates={','.join(row.get('ticker', '') for row in candidates) or 'none'}",
    )
    enrichments = []
    evidence_packets = []
    handoff_packets = []
    for candidate in candidates:
        progress(config, f"LLM enrichment started ticker={candidate.get('ticker')}")
        candidate_handoff = handoff_candidate(candidate, config=config)
        enrichment = run_llm_enrichment(config=config, candidate=candidate_handoff, market_intelligence=market_intelligence_handoff)
        enrichment_handoff = handoff_enrichment(enrichment, config=config)
        enrichments.append(enrichment)
        progress(
            config,
            "LLM enrichment finished "
            f"ticker={candidate.get('ticker')} status={enrichments[-1].get('status')} "
            f"provider={enrichments[-1].get('provider')} model={enrichments[-1].get('model')}",
        )
        progress(config, f"LLM fresh news started ticker={candidate.get('ticker')}")
        fresh_news = run_llm_fresh_news(config=config, candidate=candidate_handoff, market_intelligence=market_intelligence_handoff)
        fresh_news_handoff = handoff_fresh_news(fresh_news, config=config)
        progress(config, f"LLM fresh news finished ticker={candidate.get('ticker')} status={fresh_news.get('status')} provider={fresh_news.get('provider')} model={fresh_news.get('model')}")
        progress(config, f"LLM fundamentals started ticker={candidate.get('ticker')}")
        fundamentals = run_llm_fundamentals(config=config, candidate=candidate_handoff, fresh_news=fresh_news_handoff)
        fundamentals_handoff = handoff_fundamentals(fundamentals, config=config)
        progress(config, f"LLM fundamentals finished ticker={candidate.get('ticker')} status={fundamentals.get('status')} provider={fundamentals.get('provider')} model={fundamentals.get('model')}")
        progress(config, f"LLM long-term source synthesis started ticker={candidate.get('ticker')}")
        long_term_synthesis = run_llm_long_term_source_synthesis(
            config=config,
            candidate=candidate_handoff,
            market_intelligence=market_intelligence_handoff,
            enrichment=enrichment_handoff,
            fresh_news=fresh_news_handoff,
            fundamentals=fundamentals_handoff,
        )
        long_term_synthesis_handoff = handoff_long_term_synthesis(long_term_synthesis, config=config)
        progress(
            config,
            f"LLM long-term source synthesis finished ticker={candidate.get('ticker')} "
            f"status={long_term_synthesis.get('status')} provider={long_term_synthesis.get('provider')} model={long_term_synthesis.get('model')}",
        )
        full_packet = {
            "ticker": str(candidate.get("ticker") or "").upper(),
            "candidate": candidate,
            "generic_enrichment_llm": enrichment,
            "fresh_news_llm": fresh_news,
            "fundamentals_llm": fundamentals,
            "long_term_source_synthesis_llm": long_term_synthesis,
            "policy": {
                "regular_virtual_trader_enrichment_used": False,
                "regular_long_term_source_collectors_used": False,
                "all_sections_created_by_llm": True,
                "compact_handoff_used_for_downstream_llms": bool(config.compact_llm_handoffs),
            },
        }
        handoff_packet = handoff_evidence_packet(
            {
                "ticker": str(candidate.get("ticker") or "").upper(),
                "candidate": candidate_handoff,
                "generic_enrichment_llm": enrichment_handoff,
                "fresh_news_llm": fresh_news_handoff,
                "fundamentals_llm": fundamentals_handoff,
                "long_term_source_synthesis_llm": long_term_synthesis_handoff,
                "policy": {
                    "regular_virtual_trader_enrichment_used": False,
                    "regular_long_term_source_collectors_used": False,
                    "all_sections_created_by_llm": True,
                    "compact_handoff_used_for_downstream_llms": bool(config.compact_llm_handoffs),
                },
            },
            config=config,
        )
        evidence_packets.append(full_packet)
        handoff_packets.append(handoff_packet)
    tickers = normalize_tickers(
        plan.get("forecast_tickers", []),
        broker_state=broker_state,
        candidates=candidates,
        max_candidates=config.max_candidates,
    )
    progress(config, f"forecast queue ready tickers={','.join(tickers) or 'none'}")
    forecast_rows = []
    order_plans = []
    for index, ticker in enumerate(tickers, start=1):
        progress(config, f"pure LLM forecast {index}/{len(tickers)} {ticker}")
        evidence_packet = evidence_for_ticker(handoff_packets if config.compact_llm_handoffs else evidence_packets, ticker)
        forecast_row = run_pure_llm_forecast(ticker=ticker, config=config, cycle_dir=cycle_dir, broker_state=broker_state, evidence_packet=evidence_packet)
        forecast_rows.append(forecast_row)
        progress(config, f"pure LLM forecast finished ticker={ticker} returncode={forecast_row.get('returncode')}")
        order_plan = build_order_plan(report=forecast_row.get("report", {}), broker_state=broker_state, config=config)
        progress(
            config,
            "order plan built "
            f"ticker={ticker} action={order_plan.get('action')} intent={order_plan.get('intent')} "
            f"allowed={order_plan.get('execution_allowed')} blocks={','.join(order_plan.get('execution_blocks', [])) or 'none'}",
        )
        order_plan["order_result"] = maybe_submit_order(order_plan, config=config)
        progress(
            config,
            "order submit step finished "
            f"ticker={ticker} submitted={order_plan['order_result'].get('submitted')} "
            f"reason={order_plan['order_result'].get('reason', 'submitted')}",
        )
        order_plans.append(order_plan)
    progress(config, "updating isolated memory")
    memory = update_memory(memory, cycle=cycle, plan=plan, broker_state=broker_state, forecast_rows=forecast_rows, order_plans=order_plans)
    save_memory(config.memory_path, memory)
    result = {
        "run_type": "isolated_pure_llm_virtual_trader_agent",
        "cycle": cycle,
        "cycle_dir": str(cycle_dir),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "config": safe_config(config),
        "broker_state": broker_state_for_report(broker_state),
        "market_intelligence": market_intelligence,
        "market_intelligence_handoff": market_intelligence_handoff,
        "scout": scout,
        "scout_handoff": scout_handoff,
        "enrichments": enrichments,
        "evidence_packets": evidence_packets,
        "handoff_packets": handoff_packets,
        "plan": plan,
        "forecast_tickers": tickers,
        "forecasts": forecast_rows,
        "order_plans": order_plans,
        "next_wakeup_seconds": bounded_wakeup(plan.get("next_wakeup_seconds"), config),
        "isolation_policy": {
            "uses_regular_virtual_trader_memory": False,
            "uses_regular_virtual_trader_scout": False,
            "uses_regular_virtual_trader_enrichment": False,
            "uses_deterministic_forecast_engine": False,
            "llm_steps": [
                "market_intelligence",
                "scout",
                "enrichment",
                "fresh_news",
                "fundamentals",
                "long_term_source_synthesis",
                "portfolio_planner",
                "ticker_forecast",
                "forecast_quality_review",
                "ceo_advice",
            ],
            "deterministic_safety_steps": ["alpaca_account_read", "market_clock_block", "position_sizing", "limit_order_payload", "audit_files"],
            "compact_llm_handoffs": bool(config.compact_llm_handoffs),
        },
    }
    progress(config, "writing cycle audit files")
    write_json(cycle_dir / "agent_cycle.json", result)
    write_json(output_root / "latest_cycle.json", {"cycle_dir": str(cycle_dir), "agent_cycle": str(cycle_dir / "agent_cycle.json")})
    progress(config, f"cycle {cycle} audit written file={cycle_dir / 'agent_cycle.json'}")
    return result


def run_llm_market_intelligence(
    *,
    config: PureLLMVirtualTraderConfig,
    broker_state: dict[str, Any],
    memory: dict[str, Any],
) -> dict[str, Any]:
    item = {
        "today": datetime.now(UTC).date().isoformat(),
        "broker_state_json": json.dumps(broker_state_for_prompt(broker_state), indent=2, sort_keys=True, default=str),
        "memory_json": json.dumps(memory, indent=2, sort_keys=True, default=str)[:12000],
    }
    try:
        provider, model, payload, raw_response, parsed = call_llm_step(
            config=config,
            purpose="pure_llm_virtual_trader_market_intelligence",
            system_message=MARKET_INTELLIGENCE_SYSTEM_MESSAGE,
            user_message=MARKET_INTELLIGENCE_USER_MESSAGE,
            json_schema=MARKET_INTELLIGENCE_JSON_SCHEMA,
            item=item,
            prefer_web_search=True,
        )
        return {**parsed, "status": "executed", "provider": provider, "model": model, "llm_prompt_payload": payload, "llm_raw_response": raw_response}
    except Exception as exc:
        provider, model = primary_llm_route(config)
        return {
            "status": "fallback",
            "provider": provider,
            "model": model,
            "market_regime": "cautious",
            "summary": f"LLM market intelligence failed: {type(exc).__name__}: {exc}",
            "macro_risks": [],
            "sector_themes": [],
            "events_to_watch": [],
            "portfolio_implications": ["Fallback limits later steps to broker-visible symbols."],
            "source_notes": [],
        }


def run_llm_scout(
    *,
    config: PureLLMVirtualTraderConfig,
    broker_state: dict[str, Any],
    memory: dict[str, Any],
    market_intelligence: dict[str, Any],
) -> dict[str, Any]:
    item = {
        "today": datetime.now(UTC).date().isoformat(),
        "market_intelligence_json": handoff_json(market_intelligence, config=config, max_chars=12000),
        "broker_state_json": json.dumps(broker_state_for_prompt(broker_state), indent=2, sort_keys=True, default=str),
        "memory_json": json.dumps(memory, indent=2, sort_keys=True, default=str)[:12000],
        "max_candidates": int(config.max_candidates),
    }
    try:
        provider, model, payload, raw_response, parsed = call_llm_step(
            config=config,
            purpose="pure_llm_virtual_trader_scout",
            system_message=SCOUT_SYSTEM_MESSAGE,
            user_message=SCOUT_USER_MESSAGE,
            json_schema=SCOUT_JSON_SCHEMA,
            item=item,
            prefer_web_search=True,
        )
        return {**parsed, "status": "executed", "provider": provider, "model": model, "llm_prompt_payload": payload, "llm_raw_response": raw_response}
    except Exception as exc:
        provider, model = primary_llm_route(config)
        tickers = normalize_tickers([], broker_state=broker_state, candidates=[], max_candidates=config.max_candidates)
        return {
            "status": "fallback",
            "provider": provider,
            "model": model,
            "scout_summary": f"LLM scout failed: {type(exc).__name__}: {exc}",
            "candidates": [
                {"ticker": ticker, "company": ticker, "priority": "high", "candidate_reason": "Broker-visible symbol fallback.", "main_catalysts": [], "main_risks": []}
                for ticker in tickers
            ],
            "rejected_themes": [],
        }


def run_llm_enrichment(
    *,
    config: PureLLMVirtualTraderConfig,
    candidate: dict[str, Any],
    market_intelligence: dict[str, Any],
) -> dict[str, Any]:
    ticker = str(candidate.get("ticker") or "").upper()
    item = {
        "today": datetime.now(UTC).date().isoformat(),
        "ticker": ticker,
        "candidate_json": handoff_json(candidate, config=config, max_chars=4000),
        "market_intelligence_json": handoff_json(market_intelligence, config=config, max_chars=12000),
    }
    try:
        provider, model, payload, raw_response, parsed = call_llm_step(
            config=config,
            purpose="pure_llm_virtual_trader_enrichment",
            system_message=ENRICHMENT_SYSTEM_MESSAGE,
            user_message=ENRICHMENT_USER_MESSAGE,
            json_schema=ENRICHMENT_JSON_SCHEMA,
            item=item,
            prefer_web_search=True,
            extra_usage_context={"ticker": ticker},
        )
        return {**parsed, "status": "executed", "provider": provider, "model": model, "llm_prompt_payload": payload, "llm_raw_response": raw_response}
    except Exception as exc:
        provider, model = primary_llm_route(config)
        return {
            "status": "fallback",
            "provider": provider,
            "model": model,
            "ticker": ticker,
            "company": str(candidate.get("company") or ticker),
            "news_read": f"LLM enrichment failed: {type(exc).__name__}: {exc}",
            "fundamental_read": "",
            "sentiment_read": "",
            "bullish_evidence": [],
            "bearish_evidence": [],
            "key_risks": [],
            "source_notes": [],
        }


def run_llm_fresh_news(
    *,
    config: PureLLMVirtualTraderConfig,
    candidate: dict[str, Any],
    market_intelligence: dict[str, Any],
) -> dict[str, Any]:
    ticker = str(candidate.get("ticker") or "").upper()
    item = {
        "today": datetime.now(UTC).date().isoformat(),
        "ticker": ticker,
        "candidate_json": handoff_json(candidate, config=config, max_chars=4000),
        "market_intelligence_json": handoff_json(market_intelligence, config=config, max_chars=12000),
    }
    try:
        provider, model, payload, raw_response, parsed = call_llm_step(
            config=config,
            purpose="pure_llm_virtual_trader_fresh_news",
            system_message=FRESH_NEWS_SYSTEM_MESSAGE,
            user_message=FRESH_NEWS_USER_MESSAGE,
            json_schema=FRESH_NEWS_JSON_SCHEMA,
            item=item,
            prefer_web_search=True,
            extra_usage_context={"ticker": ticker},
        )
        return {**parsed, "provider": provider, "model": model, "llm_prompt_payload": payload, "llm_raw_response": raw_response}
    except Exception as exc:
        provider, model = primary_llm_route(config)
        return {
            "ticker": ticker,
            "status": "failed",
            "provider": provider,
            "model": model,
            "freshness_read": "Fresh-news LLM step failed.",
            "news_summary": f"LLM fresh news failed: {type(exc).__name__}: {exc}",
            "bullish_news": [],
            "bearish_news": [],
            "events_to_watch": [],
            "source_notes": [],
            "limitations": ["No fresh-news evidence is available for this cycle."],
        }


def run_llm_fundamentals(
    *,
    config: PureLLMVirtualTraderConfig,
    candidate: dict[str, Any],
    fresh_news: dict[str, Any],
) -> dict[str, Any]:
    ticker = str(candidate.get("ticker") or "").upper()
    item = {
        "today": datetime.now(UTC).date().isoformat(),
        "ticker": ticker,
        "candidate_json": handoff_json(candidate, config=config, max_chars=4000),
        "fresh_news_json": handoff_json(fresh_news, config=config, max_chars=12000),
    }
    try:
        provider, model, payload, raw_response, parsed = call_llm_step(
            config=config,
            purpose="pure_llm_virtual_trader_fundamentals",
            system_message=FUNDAMENTALS_SYSTEM_MESSAGE,
            user_message=FUNDAMENTALS_USER_MESSAGE,
            json_schema=FUNDAMENTALS_JSON_SCHEMA,
            item=item,
            prefer_web_search=True,
            extra_usage_context={"ticker": ticker},
        )
        return {**parsed, "provider": provider, "model": model, "llm_prompt_payload": payload, "llm_raw_response": raw_response}
    except Exception as exc:
        provider, model = primary_llm_route(config)
        return {
            "ticker": ticker,
            "status": "failed",
            "provider": provider,
            "model": model,
            "business_quality_read": "",
            "valuation_read": "",
            "balance_sheet_read": "",
            "growth_profitability_read": "",
            "analyst_and_estimate_read": "",
            "fundamental_bull_points": [],
            "fundamental_bear_points": [],
            "source_notes": [],
            "limitations": [f"LLM fundamentals failed: {type(exc).__name__}: {exc}"],
        }


def run_llm_long_term_source_synthesis(
    *,
    config: PureLLMVirtualTraderConfig,
    candidate: dict[str, Any],
    market_intelligence: dict[str, Any],
    enrichment: dict[str, Any],
    fresh_news: dict[str, Any],
    fundamentals: dict[str, Any],
) -> dict[str, Any]:
    ticker = str(candidate.get("ticker") or "").upper()
    item = {
        "today": datetime.now(UTC).date().isoformat(),
        "ticker": ticker,
        "candidate_json": handoff_json(candidate, config=config, max_chars=4000),
        "market_intelligence_json": handoff_json(market_intelligence, config=config, max_chars=12000),
        "enrichment_json": handoff_json(enrichment, config=config, max_chars=12000),
        "fresh_news_json": handoff_json(fresh_news, config=config, max_chars=12000),
        "fundamentals_json": handoff_json(fundamentals, config=config, max_chars=12000),
    }
    try:
        provider, model, payload, raw_response, parsed = call_llm_step(
            config=config,
            purpose="pure_llm_virtual_trader_long_term_source_synthesis",
            system_message=LONG_TERM_SYNTHESIS_SYSTEM_MESSAGE,
            user_message=LONG_TERM_SYNTHESIS_USER_MESSAGE,
            json_schema=LONG_TERM_SYNTHESIS_JSON_SCHEMA,
            item=item,
            prefer_web_search=False,
            extra_usage_context={"ticker": ticker},
        )
        return {**parsed, "provider": provider, "model": model, "llm_prompt_payload": payload, "llm_raw_response": raw_response}
    except Exception as exc:
        provider, model = primary_llm_route(config)
        return {
            "ticker": ticker,
            "status": "failed",
            "provider": provider,
            "model": model,
            "synthesis": f"LLM long-term source synthesis failed: {type(exc).__name__}: {exc}",
            "long_term_bull_case": [],
            "long_term_bear_case": [],
            "evidence_gaps": ["Long-term source synthesis failed."],
            "ceo_relevance": "Treat long-term source evidence as unavailable.",
            "source_notes": [],
        }


def run_llm_planner(
    *,
    config: PureLLMVirtualTraderConfig,
    broker_state: dict[str, Any],
    memory: dict[str, Any],
    market_intelligence: dict[str, Any],
    scout: dict[str, Any],
    enrichments: list[dict[str, Any]],
    evidence_packets: list[dict[str, Any]],
) -> dict[str, Any]:
    item = {
        "today": datetime.now(UTC).date().isoformat(),
        "config_json": json.dumps(safe_config(config), indent=2, sort_keys=True, default=str),
        "broker_state_json": json.dumps(
            {
                **broker_state_for_prompt(broker_state),
                "market_intelligence": handoff_market_intelligence(market_intelligence, config=config),
                "llm_scout": handoff_scout(scout, config=config),
                "llm_enrichments": [handoff_enrichment(row, config=config) for row in enrichments],
                "llm_evidence_packets": [handoff_evidence_packet(row, config=config) for row in evidence_packets],
            },
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )[:24000] if config.compact_llm_handoffs else json.dumps(
            {
                **broker_state_for_prompt(broker_state),
                "market_intelligence": market_intelligence,
                "llm_scout": scout,
                "llm_enrichments": enrichments,
                "llm_evidence_packets": evidence_packets,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )[:24000],
        "memory_json": json.dumps(memory, indent=2, sort_keys=True, default=str)[:12000],
    }
    try:
        provider, model, payload, raw_response, plan = call_llm_step(
            config=config,
            purpose="pure_llm_virtual_trader_planner",
            system_message=PLANNER_SYSTEM_MESSAGE,
            user_message=PLANNER_USER_MESSAGE,
            json_schema=PLANNER_JSON_SCHEMA,
            item=item,
            prefer_web_search=False,
        )
        plan = normalize_plan(plan, config=config)
        plan["provider"] = provider
        plan["model"] = model
        plan["llm_prompt_payload"] = payload
        plan["llm_raw_response"] = raw_response
        return plan
    except Exception as exc:
        provider, model = primary_llm_route(config)
        fallback = fallback_plan(broker_state=broker_state, config=config)
        fallback["status"] = "fallback"
        fallback["reason"] = f"{type(exc).__name__}: {exc}"
        fallback["provider"] = provider
        fallback["model"] = model
        return fallback


def call_llm_step(
    *,
    config: PureLLMVirtualTraderConfig,
    purpose: str,
    system_message: str,
    user_message: str,
    json_schema: dict[str, Any],
    item: dict[str, Any],
    prefer_web_search: bool,
    extra_usage_context: dict[str, Any] | None = None,
) -> tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    attempts = [
        (resolve_llm_provider(config.planner_provider), config.planner_model, "primary_local_priority"),
    ]
    fallback_provider = resolve_llm_provider(config.fallback_provider)
    if fallback_provider != attempts[0][0] or config.fallback_model != attempts[0][1]:
        attempts.append((fallback_provider, config.fallback_model, "fallback"))
    errors = []
    for provider, raw_model, route in attempts:
        model = resolve_llm_model(raw_model, provider=provider)
        use_web_search = bool(prefer_web_search and provider in {"openai", "llm_studio"})
        try:
            progress(
                config,
                f"LLM step started purpose={purpose} route={route} provider={provider} model={model} "
                f"web_search={use_web_search}",
            )
            payload, raw_response, parsed = call_response(
                client=openai_client_for_provider(provider, timeout=float(config.llm_timeout_seconds)),
                provider=provider,
                model=model,
                system_message=system_message,
                user_message=user_message,
                json_schema=json_schema,
                item=item,
                use_web_search=use_web_search,
                search_context_size=config.search_context_size,
                reasoning_effort=config.reasoning_effort,
                usage_context={"purpose": purpose, "provider": provider, "route": route, **(extra_usage_context or {})},
            )
            if route != "primary_local_priority":
                parsed = {**parsed, "_fallback_from": attempts[0][0], "_fallback_reason": "; ".join(errors[-2:])}
            progress(config, f"LLM step finished purpose={purpose} route={route} provider={provider} model={model}")
            return provider, model, payload, raw_response, parsed
        except Exception as exc:
            progress(config, f"LLM step failed purpose={purpose} route={route} provider={provider} model={model} error={type(exc).__name__}: {exc}")
            errors.append(f"{provider}/{model}: {type(exc).__name__}: {exc}")
    raise RuntimeError("All LLM routes failed: " + " | ".join(errors))


HANDOFF_EXCLUDED_KEYS = {
    "chain_of_thought",
    "final_reasoning",
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

HANDOFF_FILLER_PATTERNS = [
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


def handoff_market_intelligence(value: dict[str, Any], *, config: PureLLMVirtualTraderConfig) -> dict[str, Any]:
    if not config.compact_llm_handoffs:
        return value
    return compact_dict(
        {
            "status": value.get("status"),
            "market_regime": value.get("market_regime"),
            "summary": value.get("summary"),
            "macro_risks": value.get("macro_risks"),
            "sector_themes": value.get("sector_themes"),
            "events_to_watch": value.get("events_to_watch"),
            "portfolio_implications": value.get("portfolio_implications"),
            "source_notes": value.get("source_notes"),
        },
    )


def handoff_scout(value: dict[str, Any], *, config: PureLLMVirtualTraderConfig) -> dict[str, Any]:
    if not config.compact_llm_handoffs:
        return value
    candidates = []
    for row in value.get("candidates") or []:
        if isinstance(row, dict):
            candidates.append(
                compact_dict(
                    {
                        "ticker": row.get("ticker"),
                        "company": row.get("company"),
                        "priority": row.get("priority"),
                        "candidate_reason": row.get("candidate_reason"),
                        "main_catalysts": row.get("main_catalysts"),
                        "main_risks": row.get("main_risks"),
                    },
                )
            )
    return compact_dict(
        {
            "status": value.get("status"),
            "scout_summary": value.get("scout_summary"),
            "candidates": candidates,
            "rejected_themes": value.get("rejected_themes"),
        },
    )


def handoff_candidate(value: dict[str, Any], *, config: PureLLMVirtualTraderConfig) -> dict[str, Any]:
    if not config.compact_llm_handoffs:
        return value
    return compact_dict(
        {
            "ticker": value.get("ticker"),
            "company": value.get("company"),
            "priority": value.get("priority"),
            "candidate_reason": value.get("candidate_reason"),
            "main_catalysts": value.get("main_catalysts"),
            "main_risks": value.get("main_risks"),
        },
    )


def handoff_enrichment(value: dict[str, Any], *, config: PureLLMVirtualTraderConfig) -> dict[str, Any]:
    if not config.compact_llm_handoffs:
        return value
    return compact_dict(
        {
            "status": value.get("status"),
            "ticker": value.get("ticker"),
            "company": value.get("company"),
            "news_read": value.get("news_read"),
            "fundamental_read": value.get("fundamental_read"),
            "sentiment_read": value.get("sentiment_read"),
            "bullish_evidence": value.get("bullish_evidence"),
            "bearish_evidence": value.get("bearish_evidence"),
            "key_risks": value.get("key_risks"),
            "source_notes": value.get("source_notes"),
        },
    )


def handoff_fresh_news(value: dict[str, Any], *, config: PureLLMVirtualTraderConfig) -> dict[str, Any]:
    if not config.compact_llm_handoffs:
        return value
    return compact_dict(
        {
            "ticker": value.get("ticker"),
            "status": value.get("status"),
            "freshness_read": value.get("freshness_read"),
            "news_summary": value.get("news_summary"),
            "bullish_news": value.get("bullish_news"),
            "bearish_news": value.get("bearish_news"),
            "events_to_watch": value.get("events_to_watch"),
            "source_notes": value.get("source_notes"),
            "limitations": value.get("limitations"),
        },
    )


def handoff_fundamentals(value: dict[str, Any], *, config: PureLLMVirtualTraderConfig) -> dict[str, Any]:
    if not config.compact_llm_handoffs:
        return value
    return compact_dict(
        {
            "ticker": value.get("ticker"),
            "status": value.get("status"),
            "business_quality_read": value.get("business_quality_read"),
            "valuation_read": value.get("valuation_read"),
            "balance_sheet_read": value.get("balance_sheet_read"),
            "growth_profitability_read": value.get("growth_profitability_read"),
            "analyst_and_estimate_read": value.get("analyst_and_estimate_read"),
            "fundamental_bull_points": value.get("fundamental_bull_points"),
            "fundamental_bear_points": value.get("fundamental_bear_points"),
            "source_notes": value.get("source_notes"),
            "limitations": value.get("limitations"),
        },
    )


def handoff_long_term_synthesis(value: dict[str, Any], *, config: PureLLMVirtualTraderConfig) -> dict[str, Any]:
    if not config.compact_llm_handoffs:
        return value
    return compact_dict(
        {
            "ticker": value.get("ticker"),
            "status": value.get("status"),
            "synthesis": value.get("synthesis"),
            "long_term_bull_case": value.get("long_term_bull_case"),
            "long_term_bear_case": value.get("long_term_bear_case"),
            "evidence_gaps": value.get("evidence_gaps"),
            "ceo_relevance": value.get("ceo_relevance"),
            "source_notes": value.get("source_notes"),
        },
    )


def handoff_evidence_packet(value: dict[str, Any], *, config: PureLLMVirtualTraderConfig) -> dict[str, Any]:
    if not config.compact_llm_handoffs:
        return value
    return compact_dict(value)


def handoff_json(value: Any, *, config: PureLLMVirtualTraderConfig, max_chars: int) -> str:
    if config.compact_llm_handoffs:
        return compact_json(value, max_chars=max_chars)
    return json.dumps(value, indent=2, sort_keys=True, default=str)[:max_chars]


def compact_json(value: Any, *, max_chars: int) -> str:
    text = json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 24)] + "...<handoff_truncated>"


def compact_dict(value: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, row in value.items():
        if key in HANDOFF_EXCLUDED_KEYS or row is None:
            continue
        output[key] = compact_value(row)
    return output


def compact_value(value: Any) -> Any:
    if isinstance(value, str):
        return compact_text(value)
    if isinstance(value, list):
        return [compact_value(row) for row in value]
    if isinstance(value, dict):
        return compact_dict(value)
    return value


def compact_text(value: str) -> str:
    text = " ".join(str(value or "").split())
    for pattern, replacement in HANDOFF_FILLER_PATTERNS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ;,")


def primary_llm_route(config: PureLLMVirtualTraderConfig) -> tuple[str, str]:
    provider = resolve_llm_provider(config.planner_provider)
    return provider, resolve_llm_model(config.planner_model, provider=provider)


def run_pure_llm_forecast(
    *,
    ticker: str,
    config: PureLLMVirtualTraderConfig,
    cycle_dir: Path,
    broker_state: dict[str, Any],
    evidence_packet: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ticker_dir = cycle_dir / "forecasts" / safe_symbol(ticker)
    ticker_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = ticker_dir / "llm_evidence_packet.json"
    write_json(evidence_path, evidence_packet or {"ticker": ticker, "status": "missing"})
    position = position_for_symbol(broker_state.get("positions", []), ticker)
    account = broker_state.get("account") if isinstance(broker_state.get("account"), dict) else {}
    command = [
        sys.executable,
        "-m",
        "market_forecasting_engine.pure_llm_stock_forecaster",
        "--ticker",
        ticker,
        "--company",
        ticker,
        "--provider",
        config.provider,
        "--start",
        config.start,
        "--interval",
        config.interval,
        "--bars",
        str(config.bars),
        "--output-dir",
        str(ticker_dir),
        "--llm-provider",
        config.forecast_llm_provider,
        "--llm-model",
        config.forecast_llm_model,
        "--fallback-llm-provider",
        config.fallback_provider,
        "--fallback-llm-model",
        config.fallback_model,
        "--ceo-llm-provider",
        config.ceo_llm_provider,
        "--ceo-llm-model",
        config.ceo_llm_model,
        "--trader-profile",
        config.trader_profile,
        "--holding-status",
        "owned" if position else "not_owned",
        "--account-equity",
        str(to_float(account.get("equity") or account.get("portfolio_value")) or 0.0),
        "--portfolio-notes",
        "Isolated pure LLM virtual trader path. Evidence packet includes LLM fresh news, LLM fundamentals, "
        "LLM long-term source synthesis, and no regular virtual trader scout, enrichment, memory, provider snapshots, "
        "or deterministic forecast engine.",
        "--external-evidence-json",
        str(evidence_path),
        "--llm-timeout",
        str(config.llm_timeout_seconds),
        "--reasoning-effort",
        config.reasoning_effort,
        "--search-context-size",
        config.search_context_size,
        "--llm-env-file",
        str(config.env_file),
    ]
    if position:
        append_position_args(command, position)
    started = datetime.now(UTC)
    result = subprocess.run(command, cwd=Path.cwd(), text=True, capture_output=True, check=False)
    write_json(ticker_dir / "forecast_command.json", {"command": command})
    (ticker_dir / "forecast_stdout.txt").write_text(result.stdout or "", encoding="utf-8")
    (ticker_dir / "forecast_stderr.txt").write_text(result.stderr or "", encoding="utf-8")
    pure_path = ticker_dir / f"{pure_safe_name(ticker)}_pure_llm_stock_forecast.json"
    pure_report = read_json(pure_path) if pure_path.exists() else {}
    report = adapt_pure_report(pure_report, ticker=ticker)
    if report:
        write_json(ticker_dir / "forecast_report.json", report)
    return {
        "ticker": ticker,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "returncode": result.returncode,
        "forecast_output_dir": str(ticker_dir),
        "pure_llm_report_path": str(pure_path) if pure_path.exists() else None,
        "report": report,
        "error": None if result.returncode == 0 else (result.stderr or result.stdout)[-3000:],
    }


def adapt_pure_report(pure_report: dict[str, Any], *, ticker: str) -> dict[str, Any]:
    if not pure_report:
        return {}
    advice = pure_report.get("advice") if isinstance(pure_report.get("advice"), dict) else {}
    forecast = pure_report.get("forecast") if isinstance(pure_report.get("forecast"), dict) else {}
    return {
        "ticker": str(advice.get("ticker") or forecast.get("ticker") or ticker).upper(),
        "current_price": forecast.get("current_price"),
        "decision": advice.get("decision") or "Hold",
        "llm_final_decision": advice,
        "final_advice": advice.get("final_advice") if isinstance(advice.get("final_advice"), dict) else {},
        "pure_llm_forecast": forecast,
        "forecast_quality_review": pure_report.get("forecast_quality_review") if isinstance(pure_report.get("forecast_quality_review"), dict) else {},
        "external_llm_evidence": pure_report.get("external_llm_evidence") if isinstance(pure_report.get("external_llm_evidence"), dict) else {},
        "pure_llm_artifact": pure_report,
    }


def evidence_for_ticker(evidence_packets: list[dict[str, Any]], ticker: str) -> dict[str, Any]:
    normalized = str(ticker or "").upper()
    for packet in evidence_packets:
        if isinstance(packet, dict) and str(packet.get("ticker") or "").upper() == normalized:
            return packet
    return {"ticker": normalized, "status": "missing", "reason": "No LLM evidence packet matched ticker."}


def build_order_plan(*, report: dict[str, Any], broker_state: dict[str, Any], config: PureLLMVirtualTraderConfig) -> dict[str, Any]:
    ticker = str(report.get("ticker") or "").upper()
    decision = report.get("llm_final_decision") if isinstance(report.get("llm_final_decision"), dict) else {}
    advice = dict(report.get("final_advice") if isinstance(report.get("final_advice"), dict) else {})
    action = str(decision.get("decision") or report.get("decision") or "Hold").lower()
    current_price = to_float(report.get("current_price"))
    account = broker_state.get("account") if isinstance(broker_state.get("account"), dict) else {}
    equity = to_float(account.get("equity") or account.get("portfolio_value")) or 0.0
    buying_power = to_float(account.get("buying_power")) or 0.0
    position = position_for_symbol(broker_state.get("positions", []), ticker)
    open_orders = [row for row in broker_state.get("open_orders", []) if str(row.get("symbol") or "").upper() == ticker]
    clock = broker_state.get("clock") if isinstance(broker_state.get("clock"), dict) else {}
    blocks: list[str] = []
    warnings = ["isolated_pure_llm_path"]
    if broker_state.get("status") != "ok":
        blocks.append(f"broker_unavailable: {broker_state.get('error')}")
    if clock and not bool(clock.get("is_open")) and not config.allow_market_closed_orders:
        blocks.append("alpaca_market_closed")
    if open_orders and not config.allow_repeated_symbol_orders:
        blocks.append("existing_open_order_for_symbol")
    order_payload = None
    intent = "no_trade"
    reason = "ceo_action_not_executable"
    if action == "buy":
        intent = "buy_now"
        limit_price = to_float(advice.get("buy_now_price")) or current_price
        notional = trade_notional(equity=equity, buying_power=buying_power, config=config)
        qty = round(notional / limit_price, 6) if limit_price and notional > 0 else 0.0
        if notional <= 0 or qty <= 0:
            blocks.append("insufficient_buying_power_or_equity")
        if limit_price is None:
            blocks.append("missing_buy_limit_reference")
        if not blocks and limit_price is not None:
            order_payload = limit_order_payload(ticker=ticker, side="buy", qty=qty, limit_price=limit_price, prefix="pllmbuy")
            reason = "ceo_buy_limit_order"
    elif action == "sell":
        intent = "sell_or_trim"
        qty = position_qty(position)
        limit_price = to_float(advice.get("sell_or_trim_price")) or current_price
        if qty <= 0:
            blocks.append("no_owned_position_to_sell")
        if limit_price is None:
            blocks.append("missing_sell_limit_reference")
        if not blocks and limit_price is not None:
            order_payload = limit_order_payload(ticker=ticker, side="sell", qty=qty, limit_price=limit_price, prefix="pllmsell")
            reason = "ceo_sell_limit_order"
    else:
        buy_lower = to_float(advice.get("buy_lower_price")) or to_float(advice.get("buy_lower_zone_high"))
        if buy_lower is not None and current_price is not None and buy_lower < current_price:
            intent = "buy_lower_limit_watch"
            notional = trade_notional(equity=equity, buying_power=buying_power, config=config)
            qty = round(notional / buy_lower, 6) if notional > 0 else 0.0
            if qty > 0 and not blocks:
                order_payload = limit_order_payload(ticker=ticker, side="buy", qty=qty, limit_price=buy_lower, prefix="pllmdip")
                reason = "ceo_hold_buy_lower_limit"
            elif notional <= 0:
                blocks.append("insufficient_buying_power_or_equity")
        else:
            reason = "ceo_hold_watch_only"
    return {
        "ticker": ticker,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "action": action,
        "intent": intent,
        "reason": reason,
        "current_price": current_price,
        "final_advice": advice,
        "execution_allowed": bool(order_payload and not blocks),
        "execution_blocks": blocks,
        "warnings": warnings,
        "order_payload": order_payload,
        "dry_run": config.dry_run,
        "policy": {
            "no_market_orders_for_entries": True,
            "isolated_from_regular_virtual_trader": True,
        },
    }


def maybe_submit_order(order_plan: dict[str, Any], *, config: PureLLMVirtualTraderConfig) -> dict[str, Any]:
    payload = order_plan.get("order_payload")
    if not payload:
        return {"submitted": False, "reason": "no_order_payload"}
    if not order_plan.get("execution_allowed"):
        return {"submitted": False, "reason": "execution_blocked", "blocks": order_plan.get("execution_blocks", [])}
    if config.dry_run:
        return {"submitted": False, "reason": "dry_run", "paper_order_payload": payload}
    try:
        broker = AlpacaPaperBroker()
        response = broker.submit_order(
            symbol=payload["symbol"],
            side=payload["side"],
            order_type="limit",
            qty=payload.get("qty"),
            limit_price=payload.get("limit_price"),
            time_in_force="day",
            client_order_id=payload.get("client_order_id"),
        )
        return {"submitted": True, "broker_response": response, "paper_order_payload": payload}
    except Exception as exc:
        return {"submitted": False, "reason": "broker_submit_failed", "error": f"{type(exc).__name__}: {exc}", "paper_order_payload": payload}


def load_broker_state() -> dict[str, Any]:
    try:
        broker = AlpacaPaperBroker()
        return {
            "status": "ok",
            "account": broker.account(),
            "clock": broker.clock(),
            "positions": broker.positions(),
            "open_orders": broker.orders(status="open", limit=100),
        }
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "positions": [], "open_orders": []}


def normalize_candidates(values: Any, *, broker_state: dict[str, Any], max_candidates: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if isinstance(values, list):
        for value in values:
            if isinstance(value, dict):
                ticker = str(value.get("ticker") or value.get("symbol") or "").upper().strip()
                if ticker:
                    candidates.append(
                        {
                            "ticker": ticker,
                            "company": str(value.get("company") or ticker),
                            "priority": str(value.get("priority") or "medium"),
                            "candidate_reason": str(value.get("candidate_reason") or value.get("reason") or "LLM scout candidate."),
                            "main_catalysts": list(value.get("main_catalysts", [])) if isinstance(value.get("main_catalysts"), list) else [],
                            "main_risks": list(value.get("main_risks", [])) if isinstance(value.get("main_risks"), list) else [],
                        }
                    )
            else:
                ticker = str(value).upper().strip()
                if ticker:
                    candidates.append(
                        {
                            "ticker": ticker,
                            "company": ticker,
                            "priority": "medium",
                            "candidate_reason": "LLM scout ticker candidate.",
                            "main_catalysts": [],
                            "main_risks": [],
                        }
                    )
    broker_tickers = normalize_tickers([], broker_state=broker_state, candidates=[], max_candidates=max_candidates)
    for ticker in broker_tickers:
        if ticker and all(row.get("ticker") != ticker for row in candidates):
            candidates.insert(
                0,
                {
                    "ticker": ticker,
                    "company": ticker,
                    "priority": "high",
                    "candidate_reason": "Existing Alpaca position or open order needs LLM review.",
                    "main_catalysts": [],
                    "main_risks": [],
                },
            )
    clean: list[dict[str, Any]] = []
    for candidate in candidates:
        ticker = str(candidate.get("ticker") or "").upper().strip()
        if ticker and all(row.get("ticker") != ticker for row in clean):
            clean.append({**candidate, "ticker": ticker})
    return clean[: max(1, int(max_candidates))]


def normalize_tickers(values: Any, *, broker_state: dict[str, Any], max_candidates: int, candidates: list[dict[str, Any]] | None = None) -> list[str]:
    tickers = [str(row.get("symbol") or "").upper() for row in broker_state.get("positions", []) if isinstance(row, dict) and row.get("symbol")]
    tickers.extend(str(row.get("symbol") or "").upper() for row in broker_state.get("open_orders", []) if isinstance(row, dict) and row.get("symbol"))
    if isinstance(candidates, list):
        tickers.extend(str(row.get("ticker") or row.get("symbol") or "").upper().strip() for row in candidates if isinstance(row, dict))
    tickers.extend(str(value).upper().strip() for value in values if str(value).strip())
    clean = []
    for ticker in tickers:
        if ticker and ticker not in clean:
            clean.append(ticker)
    return clean[: max(1, int(max_candidates))]


def normalize_plan(plan: dict[str, Any], *, config: PureLLMVirtualTraderConfig) -> dict[str, Any]:
    clean = dict(plan or {})
    clean["forecast_tickers"] = [str(item).upper().strip() for item in clean.get("forecast_tickers", []) if str(item).strip()]
    clean["max_candidates_to_forecast"] = min(max(1, int(clean.get("max_candidates_to_forecast") or config.max_candidates)), config.max_candidates)
    clean["next_wakeup_seconds"] = bounded_wakeup(clean.get("next_wakeup_seconds"), config)
    clean.setdefault("status", "executed")
    return clean


def fallback_plan(*, broker_state: dict[str, Any], config: PureLLMVirtualTraderConfig) -> dict[str, Any]:
    tickers = normalize_tickers([], broker_state=broker_state, max_candidates=config.max_candidates)
    return normalize_plan(
        {
            "cycle_mode": "manage_positions" if tickers else "monitor_only",
            "account_assessment": "Fallback plan used because LLM planner failed.",
            "market_assessment": "No LLM market assessment available.",
            "risk_posture": "cautious",
            "forecast_tickers": tickers,
            "max_candidates_to_forecast": config.max_candidates,
            "ticker_rationales": [{"ticker": ticker, "priority": "high", "reason": "Existing broker position or order."} for ticker in tickers],
            "order_management_notes": ["Fallback does not discover new tickers."],
            "next_wakeup_seconds": config.loop_interval_seconds,
            "plan_rationale": "Fallback limits work to broker-visible symbols only.",
        },
        config=config,
    )


def trade_notional(*, equity: float, buying_power: float, config: PureLLMVirtualTraderConfig) -> float:
    risk_multiplier = {"conservative": 0.05, "medium": 0.10, "aggressive": 0.25}.get(config.risk_profile, 0.10)
    return max(0.0, min(float(config.max_notional_per_trade), equity * float(config.max_position_pct_equity), equity * risk_multiplier, buying_power * 0.95))


def limit_order_payload(*, ticker: str, side: str, qty: float, limit_price: float, prefix: str) -> dict[str, Any]:
    return {
        "symbol": ticker,
        "side": side,
        "order_type": "limit",
        "qty": round(float(qty), 6),
        "limit_price": round(float(limit_price), 2),
        "time_in_force": "day",
        "client_order_id": f"{prefix}_{safe_symbol(ticker)[:12]}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"[:48],
    }


def append_position_args(command: list[str], position: dict[str, Any]) -> None:
    for flag, key in [("--entry-price", "avg_entry_price"), ("--quantity", "qty"), ("--position-value", "market_value")]:
        parsed = to_float(position.get(key))
        if parsed is not None:
            command.extend([flag, str(parsed)])


def update_memory(
    memory: dict[str, Any],
    *,
    cycle: int,
    plan: dict[str, Any],
    broker_state: dict[str, Any],
    forecast_rows: list[dict[str, Any]],
    order_plans: list[dict[str, Any]],
) -> dict[str, Any]:
    output = dict(memory or {})
    output["updated_at_utc"] = datetime.now(UTC).isoformat()
    output["last_broker_snapshot"] = broker_state_for_report(broker_state)
    cycles = list(output.get("cycles", [])) if isinstance(output.get("cycles"), list) else []
    cycles.append(
        {
            "cycle": cycle,
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "plan_summary": {key: plan.get(key) for key in ("cycle_mode", "risk_posture", "forecast_tickers", "next_wakeup_seconds")},
            "decisions": [
                {
                    "ticker": row.get("ticker"),
                    "returncode": row.get("returncode"),
                    "decision": (row.get("report") or {}).get("decision"),
                    "order": plan_row,
                }
                for row, plan_row in zip(forecast_rows, order_plans, strict=False)
            ],
        }
    )
    output["cycles"] = cycles[-25:]
    return output


def load_memory(path: str | Path) -> dict[str, Any]:
    memory_path = Path(path).expanduser()
    if not memory_path.exists():
        return {"created_at_utc": datetime.now(UTC).isoformat(), "isolation": "pure_llm_virtual_trader"}
    try:
        loaded = json.loads(memory_path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {"created_at_utc": datetime.now(UTC).isoformat(), "isolation": "pure_llm_virtual_trader", "prior_memory_unreadable": True}


def save_memory(path: str | Path, memory: dict[str, Any]) -> None:
    write_json(Path(path).expanduser(), memory)


def broker_state_for_prompt(broker_state: dict[str, Any]) -> dict[str, Any]:
    account = broker_state.get("account") if isinstance(broker_state.get("account"), dict) else {}
    return {
        "status": broker_state.get("status"),
        "account": {
            "status": account.get("status"),
            "equity": account.get("equity"),
            "cash": account.get("cash"),
            "buying_power": account.get("buying_power"),
            "trading_blocked": account.get("trading_blocked"),
        },
        "clock": broker_state.get("clock"),
        "positions": broker_state.get("positions", []),
        "open_orders": broker_state.get("open_orders", []),
    }


def broker_state_for_report(broker_state: dict[str, Any]) -> dict[str, Any]:
    account = broker_state.get("account") if isinstance(broker_state.get("account"), dict) else {}
    return {
        "status": broker_state.get("status"),
        "error": broker_state.get("error"),
        "account": {
            "status": account.get("status"),
            "id_suffix": str(account.get("id", ""))[-8:],
            "equity": account.get("equity"),
            "cash": account.get("cash"),
            "buying_power": account.get("buying_power"),
            "trading_blocked": account.get("trading_blocked"),
        },
        "clock": broker_state.get("clock"),
        "positions_count": len(broker_state.get("positions", []) or []),
        "open_orders_count": len(broker_state.get("open_orders", []) or []),
    }


def safe_config(config: PureLLMVirtualTraderConfig) -> dict[str, Any]:
    data = asdict(config)
    data["env_file"] = str(data.get("env_file")) if data.get("env_file") else None
    return data


def bounded_wakeup(value: Any, config: PureLLMVirtualTraderConfig) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(config.loop_interval_seconds)
    return max(int(config.min_loop_interval_seconds), min(int(config.max_loop_interval_seconds), parsed))


def load_env_override(path: str | Path | None) -> None:
    if not path:
        return
    env_path = Path(path).expanduser()
    if not env_path.exists():
        raise FileNotFoundError(f"Env file not found: {env_path}")
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def position_for_symbol(positions: Any, ticker: str) -> dict[str, Any] | None:
    if not isinstance(positions, list):
        return None
    for position in positions:
        if isinstance(position, dict) and str(position.get("symbol") or "").upper() == ticker.upper():
            return position
    return None


def position_qty(position: dict[str, Any] | None) -> float:
    return max(0.0, to_float((position or {}).get("qty")) or 0.0)


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(str(value).replace(",", ""))
    except Exception:
        return None
    return parsed if parsed == parsed else None


def read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return str(path)


def safe_symbol(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_") or "unknown"


def pure_safe_name(value: str) -> str:
    return str(value).upper().replace("/", "_").replace(" ", "_")


def progress(config: PureLLMVirtualTraderConfig, message: str) -> None:
    if config.progress:
        print(f"[pure-llm-virtual-agent] {datetime.now().strftime('%H:%M:%S')} {message}", flush=True)


if __name__ == "__main__":
    main()
