from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from market_forecasting_engine.llm_trader.responses_api import call_response
from market_forecasting_engine.llm_trader.run import openai_client_for_provider, resolve_llm_model, resolve_llm_provider
from market_forecasting_engine.llm_options_trader.common import (
    LLMOptionsRuntimeConfig,
    append_jsonl,
    build_market_packet,
    cancel_order_payload,
    execute_order_payload,
    testnet_broker,
    write_json,
)
from market_forecasting_engine.llm_options_trader.memory import append_memory_event, compact_decision_event, load_recent_memory
from market_forecasting_engine.llm_options_trader.profiles import trader_profile
from market_forecasting_engine.llm_options_trader.prompts import EXIT_JSON_SCHEMA, EXIT_SYSTEM_MESSAGE, EXIT_USER_MESSAGE


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    broker = testnet_broker()
    config = runtime_config(args)
    provider = resolve_llm_provider(args.llm_provider)
    client = openai_client_for_provider(provider, timeout=float(args.llm_timeout_seconds))
    while True:
        record = run_once(args=args, broker=broker, config=config, provider=provider, client=client)
        report_path = output_dir / f"{config.currency.upper()}_llm_exit_agent_report.json"
        write_json(report_path, record)
        append_jsonl(output_dir / "logs" / f"{config.currency.upper()}_llm_exit_agent.jsonl", record)
        print(json.dumps({"report": str(report_path), "action": record["llm_decision"].get("action"), "order_result": record["order_result"]}, indent=2, default=str), flush=True)
        if args.once:
            break
        time.sleep(max(1, int(args.check_interval_seconds)))


def run_once(*, args: argparse.Namespace, broker, config: LLMOptionsRuntimeConfig, provider: str, client) -> dict:
    now = datetime.now(UTC)
    profile = trader_profile(args.trader_profile)
    packet = build_market_packet(broker=broker, config=config, process="exit_agent", now=now)
    packet["trader_profile"] = profile
    packet["trader_memory"] = load_recent_memory(Path(args.output_dir), config.currency, limit=int(args.memory_events))
    _, raw_response, decision = call_response(
        client=client,
        provider=provider,
        model=resolve_llm_model(args.llm_model, provider=provider),
        system_message=EXIT_SYSTEM_MESSAGE,
        user_message=EXIT_USER_MESSAGE,
        json_schema=EXIT_JSON_SCHEMA,
        item={"market_packet_json": json.dumps(packet, indent=2, default=str)},
        use_web_search=bool(args.use_web_search) and provider == "openai",
        search_context_size=args.search_context_size,
        reasoning_effort=args.reasoning_effort,
        usage_context={"process": "llm_options_exit_agent", "currency": config.currency, "instrument_currency": config.instrument_currency, "provider": provider},
    )
    action = str(decision.get("action") or "hold")
    if action == "submit_order":
        order_result = execute_order_payload(
            broker=broker,
            config=config,
            decision=decision,
            require_reduce_only=True,
            execute=bool(args.execute_testnet_orders),
            label=f"llm-exit-{config.currency.upper()}-{now.strftime('%Y%m%d%H%M%S')}",
        )
    elif action == "cancel_order":
        order_result = cancel_order_payload(broker=broker, decision=decision, execute=bool(args.execute_testnet_orders))
    else:
        order_result = {"submitted": False, "reason": "llm_decision_hold"}
    record = {
        "checked_at_utc": now.isoformat(),
        "account_mode": "testnet",
        "agent": "llm_options_exit_agent",
        "trader_profile": profile,
        "python_role": "api_interface_only",
        "market_packet": packet,
        "llm_decision": decision,
        "llm_raw_response": raw_response,
        "execute_testnet_orders": bool(args.execute_testnet_orders),
        "order_result": order_result,
    }
    append_memory_event(Path(args.output_dir), config.currency, compact_decision_event(process="exit_agent", record=record))
    return record


def runtime_config(args: argparse.Namespace) -> LLMOptionsRuntimeConfig:
    return LLMOptionsRuntimeConfig(
        currency=args.currency,
        instrument_currency=args.instrument_currency,
        data_provider=args.data_provider,
        data_interval=args.data_interval,
        lookback_days=int(args.lookback_days),
        max_price_rows=int(args.max_price_rows),
        forecast_hours=tuple(float(value.strip()) for value in args.forecast_hours.split(",") if value.strip()),
        option_chain_limit=int(args.option_chain_limit),
        min_dte=int(args.min_dte),
        max_dte=int(args.max_dte),
        max_order_amount=float(args.max_order_amount),
        max_order_price=float(args.max_order_price),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM-authoritative Deribit testnet option exit agent.")
    parser.add_argument("--currency", default="ETH", help="Deribit base currency, for example ETH, BTC, or another supported option currency.")
    parser.add_argument("--instrument-currency", default="USDC", help="Deribit instrument/account currency, for example USDC, ETH, or BTC.")
    parser.add_argument("--data-provider", default="alpaca")
    parser.add_argument("--data-interval", default="1m")
    parser.add_argument("--lookback-days", type=int, default=20)
    parser.add_argument("--max-price-rows", type=int, default=3500)
    parser.add_argument("--forecast-hours", default="0.25,0.5,1")
    parser.add_argument("--option-chain-limit", type=int, default=100)
    parser.add_argument("--min-dte", type=int, default=1)
    parser.add_argument("--max-dte", type=int, default=14)
    parser.add_argument("--max-order-amount", type=float, default=10.0)
    parser.add_argument("--max-order-price", type=float, default=5000.0)
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--trader-profile", default="gambit")
    parser.add_argument("--memory-events", type=int, default=40)
    parser.add_argument("--llm-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--use-web-search", action="store_true")
    parser.add_argument("--search-context-size", default="low")
    parser.add_argument("--check-interval-seconds", type=int, default=30)
    parser.add_argument("--output-dir", default="automated_forecasting_engine/runs/llm_options_trader_testnet")
    parser.add_argument("--execute-testnet-orders", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser


if __name__ == "__main__":
    main()
