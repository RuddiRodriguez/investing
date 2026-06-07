from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from market_forecasting_engine.alpaca_broker import AlpacaPaperBroker, load_env_file
from market_forecasting_engine.llm_trader.responses_api import call_response
from market_forecasting_engine.llm_trader.run import load_env, openai_client_for_provider, resolve_llm_model, resolve_llm_provider
from market_forecasting_engine.llm_options_trader.alpaca_common import (
    AlpacaLLMOptionsRuntimeConfig,
    alpaca_asset_class,
    apply_forecast_validation_to_alpaca_transition,
    build_alpaca_market_packet,
    compact_alpaca_market_packet,
    validate_alpaca_order_payload,
)
from market_forecasting_engine.llm_options_trader.alpaca_prompts import (
    ALPACA_COMBINED_JSON_SCHEMA,
    ALPACA_COMBINED_SYSTEM_MESSAGE,
    ALPACA_COMBINED_USER_MESSAGE,
    ALPACA_EXIT_JSON_SCHEMA,
    ALPACA_EXIT_SYSTEM_MESSAGE,
    ALPACA_EXIT_USER_MESSAGE,
    PROFIT_POLICY_JSON_SCHEMA,
    PROFIT_POLICY_SYSTEM_MESSAGE,
    PROFIT_POLICY_USER_MESSAGE,
)
from market_forecasting_engine.llm_options_trader.alpaca_shadow_ledger import (
    load_and_update_alpaca_shadow_state,
    record_simulated_alpaca_cancels,
    record_simulated_alpaca_order,
)
from market_forecasting_engine.llm_options_trader.chronos_forecast import build_chronos_forecast
from market_forecasting_engine.llm_options_trader.common import append_jsonl, write_json
from market_forecasting_engine.llm_options_trader.forecast_ledger import update_forecast_ledger
from market_forecasting_engine.llm_options_trader.memory import (
    append_memory_event,
    compact_decision_event,
    load_recent_memory,
    load_strategy_memory,
    update_strategy_memory_from_record,
)
from market_forecasting_engine.llm_options_trader.profiles import strategy_mode_profile, trader_profile


def main() -> None:
    args = _load_config_overrides(build_parser().parse_args())
    load_env(".env")
    output_dir = Path(args.output_dir)
    broker = _alpaca_broker_for_mode(args.account_mode)
    config = runtime_config(args)
    provider = resolve_llm_provider(args.llm_provider)
    client = openai_client_for_provider(provider, timeout=float(args.llm_timeout_seconds))
    while True:
        record = run_once(args=args, broker=broker, config=config, provider=provider, client=client)
        report_path = output_dir / f"{config.ticker.upper()}_alpaca_llm_shadow_report.json"
        write_json(report_path, record)
        append_jsonl(output_dir / "logs" / f"{config.ticker.upper()}_alpaca_llm_shadow_agent.jsonl", record)
        print(
            json.dumps(
                {
                    "report": str(report_path),
                    "venue": "alpaca",
                    "ticker": config.ticker.upper(),
                    "action": record["llm_decision"].get("action"),
                    "intent": record["llm_decision"].get("intent"),
                    "order_result": record["order_result"],
                    "shadow_pnl": (record.get("market_packet") or {}).get("shadow_simulation", {}).get("total_pnl"),
                },
                indent=2,
                default=str,
            ),
            flush=True,
        )
        if args.once:
            break
        time.sleep(max(1, int(args.check_interval_seconds)))


def run_once(*, args: argparse.Namespace, broker: AlpacaPaperBroker, config: AlpacaLLMOptionsRuntimeConfig, provider: str, client) -> dict[str, Any]:
    now = datetime.now(UTC)
    profile = trader_profile(args.trader_profile)
    full_packet = build_alpaca_market_packet(broker=broker, config=config, process="alpaca_shadow_agent", now=now)
    full_packet["account_mode"] = args.account_mode
    full_packet["execution_mode"] = "live_shadow_simulation"
    full_packet["shadow_simulation"] = load_and_update_alpaca_shadow_state(output_dir=Path(args.output_dir), ticker=config.ticker, broker=broker)
    full_packet["trader_profile"] = profile
    full_packet["entry_mandate"] = _entry_mandate(args.entry_bias)
    full_packet["strategy_mode"] = strategy_mode_profile(args.strategy_mode)
    full_packet["trader_memory"] = load_recent_memory(Path(args.output_dir), config.ticker, limit=int(args.memory_events))
    full_packet["strategy_memory"] = load_strategy_memory(Path(args.output_dir), config.ticker, max_lessons=int(args.strategy_memory_lessons))
    full_packet["external_forecasts"] = {
        "chronos": build_chronos_forecast(
            price_bars=full_packet.get("recent_price_bars") or [],
            forecast_hours=config.forecast_hours,
            data_interval=config.data_interval,
            model_name=args.chronos_model,
            context_rows=int(args.chronos_context_rows),
            num_samples=int(args.chronos_num_samples),
            refresh_seconds=int(args.chronos_refresh_seconds),
            enabled=bool(args.enable_chronos_forecast),
        )
    }
    full_packet["forecast_validation"] = update_forecast_ledger(
        output_dir=Path(args.output_dir),
        currency=config.ticker,
        forecast=full_packet["external_forecasts"]["chronos"],
        price_bars=full_packet.get("recent_price_bars") or [],
    )
    full_packet["regime_transition_warning"] = apply_forecast_validation_to_alpaca_transition(
        full_packet.get("regime_transition_warning") or {},
        full_packet["forecast_validation"],
    )
    packet = compact_alpaca_market_packet(
        full_packet,
        max_contracts_per_side=int(args.max_contracts_per_side),
        max_price_bars=int(args.max_prompt_price_bars),
        max_trades=int(args.max_prompt_trades),
    )
    profit_policy_raw_response = None
    profit_policy_decision = None
    if args.enable_profit_policy_llm:
        _, profit_policy_raw_response, profit_policy_decision = call_response(
            client=client,
            provider=provider,
            model=resolve_llm_model(args.profit_policy_llm_model or args.llm_model, provider=provider),
            system_message=PROFIT_POLICY_SYSTEM_MESSAGE,
            user_message=PROFIT_POLICY_USER_MESSAGE,
            json_schema=PROFIT_POLICY_JSON_SCHEMA,
            item={"market_packet_json": json.dumps(packet, indent=2, default=str)},
            use_web_search=False,
            search_context_size=args.search_context_size,
            reasoning_effort=args.reasoning_effort,
            usage_context=_usage_context(args=args, config=config, provider=provider, process="alpaca_llm_options_profit_policy"),
        )
        full_packet["adaptive_profit_policy"] = profit_policy_decision
        packet["adaptive_profit_policy"] = profit_policy_decision
    shadow = packet.get("shadow_simulation") if isinstance(packet.get("shadow_simulation"), dict) else {}
    has_shadow_exposure = bool(shadow.get("open_order_count") or shadow.get("position_count"))
    if has_shadow_exposure:
        exit_packet = _packet_with_shadow_exposure(packet)
        _, raw_response, decision = call_response(
            client=client,
            provider=provider,
            model=resolve_llm_model(args.llm_model, provider=provider),
            system_message=ALPACA_EXIT_SYSTEM_MESSAGE,
            user_message=ALPACA_EXIT_USER_MESSAGE,
            json_schema=ALPACA_EXIT_JSON_SCHEMA,
            item={"market_packet_json": json.dumps(exit_packet, indent=2, default=str)},
            use_web_search=bool(args.use_web_search) and provider == "openai",
            search_context_size=args.search_context_size,
            reasoning_effort=args.reasoning_effort,
            usage_context=_usage_context(args=args, config=config, provider=provider, process="alpaca_llm_options_exit_agent"),
        )
        order_result = _apply_decision(args=args, broker=broker, config=config, decision=decision, now=now, require_exit=True, market_policy=packet.get("market_policy") or {})
        entry_decision = {"action": "hold", "intent": "hold", "confidence": 1.0, "reason": "entry_skipped_while_shadow_exposure_exists", "risks": [], "order_id": None, "order_ids": [], "order": None}
        entry_raw_response = None
        entry_order_result = {"submitted": False, "reason": "entry_skipped_while_shadow_exposure_exists"}
        exit_decision = _normalize_exit_decision(decision)
        exit_raw_response = raw_response
        exit_order_result = order_result
        llm_decision = exit_decision
    else:
        _, raw_response, decision = call_response(
            client=client,
            provider=provider,
            model=resolve_llm_model(args.llm_model, provider=provider),
            system_message=ALPACA_COMBINED_SYSTEM_MESSAGE,
            user_message=ALPACA_COMBINED_USER_MESSAGE,
            json_schema=ALPACA_COMBINED_JSON_SCHEMA,
            item={"market_packet_json": json.dumps(packet, indent=2, default=str)},
            use_web_search=bool(args.use_web_search) and provider == "openai",
            search_context_size=args.search_context_size,
            reasoning_effort=args.reasoning_effort,
            usage_context=_usage_context(args=args, config=config, provider=provider, process="alpaca_llm_options_entry_agent"),
        )
        order_result = _apply_decision(args=args, broker=broker, config=config, decision=decision, now=now, require_exit=False, market_policy=packet.get("market_policy") or {})
        entry_decision = decision
        entry_raw_response = raw_response
        entry_order_result = order_result
        exit_decision = {"action": "hold", "confidence": 1.0, "reason": "no_shadow_position_or_order_to_manage", "risks": [], "order_id": None, "order_ids": [], "order": None}
        exit_raw_response = None
        exit_order_result = {"submitted": False, "reason": "no_shadow_position_or_order_to_manage"}
        llm_decision = decision
    record = {
        "checked_at_utc": now.isoformat(),
        "venue": "alpaca",
        "account_mode": args.account_mode,
        "execution_mode": "live_shadow_simulation",
        "agent": "alpaca_llm_options_shadow_agent",
        "llm_provider": provider,
        "llm_model": resolve_llm_model(args.llm_model, provider=provider),
        "trader_profile": profile,
        "python_role": "api_interface_only",
        "market_packet": packet,
        "dashboard_price_bars": full_packet.get("recent_price_bars") or [],
        "llm_decision": llm_decision,
        "llm_raw_response": raw_response,
        "profit_policy_decision": profit_policy_decision,
        "profit_policy_raw_response": profit_policy_raw_response,
        "entry_decision": entry_decision,
        "entry_raw_response": entry_raw_response,
        "entry_order_result": entry_order_result,
        "exit_decision": exit_decision,
        "exit_raw_response": exit_raw_response,
        "exit_order_result": exit_order_result,
        "simulation_only": True,
        "order_result": order_result,
        "prompt_controls": {
            "max_contracts_per_side": int(args.max_contracts_per_side),
            "max_prompt_price_bars": int(args.max_prompt_price_bars),
            "max_prompt_trades": int(args.max_prompt_trades),
            "enable_chronos_forecast": bool(args.enable_chronos_forecast),
            "chronos_model": args.chronos_model,
            "chronos_refresh_seconds": int(args.chronos_refresh_seconds),
            "enable_profit_policy_llm": bool(args.enable_profit_policy_llm),
            "profit_policy_llm_model": resolve_llm_model(args.profit_policy_llm_model or args.llm_model, provider=provider),
            "entry_bias": args.entry_bias,
            "strategy_mode": args.strategy_mode,
        },
    }
    append_memory_event(Path(args.output_dir), config.ticker, compact_decision_event(process="alpaca_shadow_agent", record=record))
    record["strategy_memory_after_update"] = update_strategy_memory_from_record(
        Path(args.output_dir),
        config.ticker,
        record,
        max_lessons=int(args.strategy_memory_max_lessons),
    )
    return record


def _apply_decision(
    *,
    args: argparse.Namespace,
    broker: AlpacaPaperBroker,
    config: AlpacaLLMOptionsRuntimeConfig,
    decision: dict[str, Any],
    now: datetime,
    require_exit: bool,
    market_policy: dict[str, Any],
) -> dict[str, Any]:
    action = str(decision.get("action") or "hold")
    if action == "submit_order":
        if _entry_blocked_by_market(packet_market_policy=market_policy, config=config, decision=decision):
            return {"submitted": False, "reason": "market_closed_for_non_crypto_shadow_entry"}
        order = decision.get("order") if isinstance(decision.get("order"), dict) else {}
        validation = validate_alpaca_order_payload(order, config=config, require_exit=require_exit)
        if validation.get("blocks"):
            return {"submitted": False, "reason": "format_validation_failed", "blocks": validation["blocks"], "order": order}
        simulated = {
            "submitted": False,
            "reason": "simulated_live_order_not_submitted",
            "validated_order": validation["order"],
            "simulation": {
                "account_mode": "paper",
                "would_submit": validation["order"],
                "policy": "Alpaca paper market/account data was used, but order submission is disabled in this shadow experiment.",
            },
        }
        simulated["simulation"].update(
            record_simulated_alpaca_order(
                output_dir=Path(args.output_dir),
                ticker=config.ticker,
                broker=broker,
                validated_order=validation["order"],
                decision=decision,
                checked_at_utc=now.isoformat(),
            )
        )
        return simulated
    if action == "cancel_order":
        result = {
            "submitted": False,
            "reason": "simulated_live_cancel_not_submitted",
            "order_id": decision.get("order_id"),
            "order_ids": _cancel_order_ids(decision),
            "simulation": {
                "account_mode": "paper",
                "policy": "Alpaca paper market/account data was used, but cancellation is disabled in this shadow experiment.",
            },
        }
        result["simulation"].update(
            record_simulated_alpaca_cancels(output_dir=Path(args.output_dir), ticker=config.ticker, order_ids=_cancel_order_ids(decision), checked_at_utc=now.isoformat())
        )
        return result
    return {"submitted": False, "reason": "llm_decision_hold"}


def _packet_with_shadow_exposure(packet: dict[str, Any]) -> dict[str, Any]:
    updated = dict(packet)
    shadow = updated.get("shadow_simulation") if isinstance(updated.get("shadow_simulation"), dict) else {}
    shadow_positions = shadow.get("positions") if isinstance(shadow.get("positions"), list) else []
    shadow_orders = shadow.get("open_orders") if isinstance(shadow.get("open_orders"), list) else []
    if shadow_positions:
        updated["option_positions"] = shadow_positions
    if shadow_orders:
        updated["open_option_orders"] = shadow_orders
    updated["exit_management_scope"] = {
        "mode": "live_shadow_simulation",
        "instruction": "Manage shadow_simulation exposure as the active portfolio. Exit orders must close or protect these positions only.",
        "allowed_submit_order": "sell limit for an existing positive-size shadow position",
        "allowed_cancel_order": "cancel order_id/order_ids from shadow_simulation.open_orders",
        "disallowed": "new buy entries during exit management",
    }
    return updated


def _normalize_exit_decision(decision: dict[str, Any]) -> dict[str, Any]:
    intent = "hold"
    if decision.get("action") == "cancel_order":
        intent = "cancel_stale_order"
    elif decision.get("action") == "submit_order":
        intent = "close_position"
    return {
        "action": decision.get("action") or "hold",
        "intent": intent,
        "confidence": decision.get("confidence"),
        "market_view": decision.get("reason"),
        "reason": decision.get("reason"),
        "risks": decision.get("risks") or [],
        "order_id": decision.get("order_id"),
        "order_ids": decision.get("order_ids") if isinstance(decision.get("order_ids"), list) else [],
        "order": decision.get("order"),
    }


def _entry_mandate(entry_bias: str) -> dict[str, Any]:
    normalized = str(entry_bias or "unrestricted").strip().lower()
    if normalized == "put_only":
        return {
            "mode": "put_only",
            "instruction": "For new entries, evaluate only long put setups from the provided option_chain. Do not open calls. If no put setup has sufficient edge, return hold.",
            "allowed_entry_intents": ["open_put", "hold"],
            "disallowed_entry_intents": ["open_call"],
        }
    if normalized == "call_only":
        return {
            "mode": "call_only",
            "instruction": "For new entries, evaluate only long call setups from the provided option_chain. Do not open puts. If no call setup has sufficient edge, return hold.",
            "allowed_entry_intents": ["open_call", "hold"],
            "disallowed_entry_intents": ["open_put"],
        }
    return {
        "mode": "unrestricted",
        "instruction": "For equity options, choose call, put, or hold. For Alpaca crypto spot, choose open_spot_long or hold. Use the market data and strategy context.",
        "allowed_entry_intents": ["open_call", "open_put", "open_spot_long", "hold"],
        "disallowed_entry_intents": [],
    }


def _cancel_order_ids(decision: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    if isinstance(decision.get("order_ids"), list):
        ids.extend(str(order_id).strip() for order_id in decision["order_ids"] if str(order_id).strip())
    single = str(decision.get("order_id") or "").strip()
    if single:
        ids.append(single)
    return list(dict.fromkeys(ids))


def _usage_context(*, args: argparse.Namespace, config: AlpacaLLMOptionsRuntimeConfig, provider: str, process: str) -> dict[str, Any]:
    return {
        "process": process,
        "venue": "alpaca",
        "ticker": config.ticker,
        "provider": provider,
        "output_dir": str(args.output_dir),
        "account_mode": args.account_mode,
        "simulation_only": True,
    }


def runtime_config(args: argparse.Namespace) -> AlpacaLLMOptionsRuntimeConfig:
    if str(args.data_provider or "").lower() != "alpaca":
        raise ValueError("Alpaca shadow agent requires --data-provider alpaca so data and execution venue match.")
    return AlpacaLLMOptionsRuntimeConfig(
        ticker=args.ticker,
        data_provider=args.data_provider,
        data_interval=args.data_interval,
        data_feed=args.alpaca_data_feed,
        lookback_days=int(args.lookback_days),
        max_price_rows=int(args.max_price_rows),
        forecast_hours=tuple(float(value.strip()) for value in args.forecast_hours.split(",") if value.strip()),
        option_chain_limit=int(args.option_chain_limit),
        min_dte=int(args.min_dte),
        max_dte=int(args.max_dte),
        max_order_qty=int(args.max_order_qty),
        max_order_price=float(args.max_order_price),
        max_order_debit=float(args.max_order_debit),
        max_crypto_notional=float(args.max_crypto_notional),
    )


def _load_config_overrides(args: argparse.Namespace) -> argparse.Namespace:
    config_path = Path(args.config) if args.config else None
    if not config_path:
        return args
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--config must point to a JSON object.")
    for key, value in payload.items():
        attr = key.replace("-", "_")
        if not hasattr(args, attr):
            raise ValueError(f"Unknown Alpaca shadow config key `{key}`.")
        setattr(args, attr, value)
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM-authoritative Alpaca paper-options live shadow trader.")
    parser.add_argument("--config", default=None, help="Path to a JSON config file. CLI args are parsed first, then config values override them.")
    parser.add_argument("--account-mode", choices=("paper", "live"), default="paper", help="Use Alpaca paper or live account data. Orders remain simulation-only.")
    parser.add_argument("--ticker", default="NVDA", help="Alpaca optionable stock ticker, for example NVDA, TSLA, SPY, or QQQ.")
    parser.add_argument("--data-provider", default="alpaca")
    parser.add_argument("--alpaca-data-feed", default="iex")
    parser.add_argument("--data-interval", default="1m")
    parser.add_argument("--lookback-days", type=int, default=20)
    parser.add_argument("--max-price-rows", type=int, default=3500)
    parser.add_argument("--forecast-hours", default="0.25,0.5,0.75,1")
    parser.add_argument("--option-chain-limit", type=int, default=80)
    parser.add_argument("--max-contracts-per-side", type=int, default=8)
    parser.add_argument("--max-prompt-price-bars", type=int, default=45)
    parser.add_argument("--max-prompt-trades", type=int, default=12)
    parser.add_argument("--enable-chronos-forecast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chronos-model", default="amazon/chronos-t5-tiny")
    parser.add_argument("--chronos-context-rows", type=int, default=512)
    parser.add_argument("--chronos-num-samples", type=int, default=40)
    parser.add_argument("--chronos-refresh-seconds", type=int, default=300)
    parser.add_argument("--min-dte", type=int, default=1)
    parser.add_argument("--max-dte", type=int, default=14)
    parser.add_argument("--max-order-qty", type=int, default=1)
    parser.add_argument("--max-order-price", type=float, default=25.0)
    parser.add_argument("--max-order-debit", type=float, default=1500.0)
    parser.add_argument("--max-crypto-notional", type=float, default=100.0)
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--enable-profit-policy-llm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profit-policy-llm-model", default=None)
    parser.add_argument("--entry-bias", choices=("unrestricted", "put_only", "call_only"), default="unrestricted")
    parser.add_argument("--strategy-mode", default="options_directional")
    parser.add_argument("--trader-profile", default="gambit")
    parser.add_argument("--memory-events", type=int, default=12)
    parser.add_argument("--strategy-memory-lessons", type=int, default=12)
    parser.add_argument("--strategy-memory-max-lessons", type=int, default=24)
    parser.add_argument("--llm-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--use-web-search", action="store_true")
    parser.add_argument("--search-context-size", default="low")
    parser.add_argument("--check-interval-seconds", type=int, default=300)
    parser.add_argument("--output-dir", default="automated_forecasting_engine/runs/alpaca_llm_options_shadow")
    parser.add_argument("--once", action="store_true")
    return parser


def _alpaca_broker_for_mode(account_mode: str) -> AlpacaPaperBroker:
    load_env_file()
    if account_mode == "live":
        return AlpacaPaperBroker(
            base_url=os.getenv("ALPACA_TRADING_BASE_URL_LIVE") or os.getenv("ALPACA_API_BASE_URL_LIVE") or "https://api.alpaca.markets",
            key_id=os.getenv("ALPACA_API_KEY_ID_LIVE") or os.getenv("APCA_API_KEY_ID_LIVE"),
            secret_key=os.getenv("ALPACA_API_SECRET_KEY_LIVE") or os.getenv("APCA_API_SECRET_KEY_LIVE"),
        )
    return AlpacaPaperBroker()


def _entry_blocked_by_market(*, packet_market_policy: dict[str, Any] | None, config: AlpacaLLMOptionsRuntimeConfig, decision: dict[str, Any]) -> bool:
    # The hard non-crypto market-hours gate mirrors paper_options_agent's market_closed block.
    # It is intentionally conservative when the caller did not pass a fresh packet policy.
    if str(decision.get("intent") or "").startswith("close") or str((decision.get("order") or {}).get("side") or "").lower() == "sell":
        return False
    if alpaca_asset_class(config.ticker) == "crypto_spot":
        return False
    policy = packet_market_policy or {}
    if "can_open_new_shadow_entries" not in policy:
        return False
    return not bool(policy.get("can_open_new_shadow_entries"))


if __name__ == "__main__":
    main()
