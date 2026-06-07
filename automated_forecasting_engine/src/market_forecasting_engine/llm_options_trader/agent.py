from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from market_forecasting_engine.llm_trader.responses_api import call_response
from market_forecasting_engine.llm_trader.run import load_env, openai_client_for_provider, resolve_llm_model, resolve_llm_provider
from market_forecasting_engine.llm_options_trader.common import (
    LLMOptionsRuntimeConfig,
    apply_forecast_validation_to_transition,
    append_jsonl,
    build_market_packet,
    cancel_order_payload,
    compact_market_packet,
    execute_order_payload,
    live_broker,
    testnet_broker,
    write_json,
)
from market_forecasting_engine.llm_options_trader.chronos_forecast import build_chronos_forecast
from market_forecasting_engine.llm_options_trader.forecast_ledger import update_forecast_ledger
from market_forecasting_engine.llm_options_trader.memory import (
    append_memory_event,
    compact_decision_event,
    load_recent_memory,
    load_strategy_memory,
    update_strategy_memory_from_record,
)
from market_forecasting_engine.llm_options_trader.profiles import strategy_mode_profile, trader_profile
from market_forecasting_engine.llm_options_trader.prompts import (
    COMBINED_JSON_SCHEMA,
    COMBINED_SYSTEM_MESSAGE,
    COMBINED_USER_MESSAGE,
    EXIT_JSON_SCHEMA,
    EXIT_SYSTEM_MESSAGE,
    EXIT_USER_MESSAGE,
    PROFIT_POLICY_JSON_SCHEMA,
    PROFIT_POLICY_SYSTEM_MESSAGE,
    PROFIT_POLICY_USER_MESSAGE,
)
from market_forecasting_engine.llm_options_trader.shadow_ledger import (
    load_and_update_shadow_state,
    record_simulated_cancels,
    record_simulated_order,
)


def main() -> None:
    args = _load_config_overrides(build_parser().parse_args())
    load_env(".env")
    output_dir = Path(args.output_dir)
    broker = live_broker() if args.account_mode == "live" else testnet_broker()
    config = runtime_config(args)
    provider = resolve_llm_provider(args.llm_provider)
    client = openai_client_for_provider(provider, timeout=float(args.llm_timeout_seconds))
    while True:
        record = run_once(args=args, broker=broker, config=config, provider=provider, client=client)
        report_path = output_dir / f"{config.currency.upper()}_llm_agent_report.json"
        write_json(report_path, record)
        append_jsonl(output_dir / "logs" / f"{config.currency.upper()}_llm_agent.jsonl", record)
        print(
            json.dumps(
                {
                    "report": str(report_path),
                    "action": record["llm_decision"].get("action"),
                    "intent": record["llm_decision"].get("intent"),
                    "order_result": record["order_result"],
                },
                indent=2,
                default=str,
            ),
            flush=True,
        )
        if args.once:
            break
        time.sleep(max(1, int(args.check_interval_seconds)))


def run_once(*, args: argparse.Namespace, broker, config: LLMOptionsRuntimeConfig, provider: str, client) -> dict:
    now = datetime.now(UTC)
    profile = trader_profile(args.trader_profile)
    full_packet = build_market_packet(broker=broker, config=config, process="combined_agent", now=now)
    full_packet["account_mode"] = args.account_mode
    full_packet["execution_mode"] = "live_shadow_simulation" if args.simulation_only else args.account_mode
    if args.simulation_only:
        full_packet["shadow_simulation"] = load_and_update_shadow_state(output_dir=Path(args.output_dir), currency=config.currency, broker=broker)
        full_packet["shadow_trading_budget"] = _shadow_trading_budget(args=args, config=config)
    full_packet["trader_profile"] = profile
    full_packet["entry_mandate"] = _entry_mandate(args.entry_bias)
    full_packet["strategy_mode"] = strategy_mode_profile(args.strategy_mode)
    full_packet["trader_memory"] = load_recent_memory(Path(args.output_dir), config.currency, limit=int(args.memory_events))
    full_packet["strategy_memory"] = load_strategy_memory(Path(args.output_dir), config.currency, max_lessons=int(args.strategy_memory_lessons))
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
        currency=config.currency,
        forecast=full_packet["external_forecasts"]["chronos"],
        price_bars=full_packet.get("recent_price_bars") or [],
    )
    full_packet["forecast_error_feedback"] = full_packet["forecast_validation"].get("error_feedback") or {}
    full_packet["regime_transition_warning"] = apply_forecast_validation_to_transition(
        full_packet.get("regime_transition_warning") or {},
        full_packet["forecast_validation"],
    )
    packet = compact_market_packet(
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
            usage_context=_usage_context(args=args, config=config, provider=provider, process="llm_options_profit_policy"),
        )
        full_packet["adaptive_profit_policy"] = profit_policy_decision
        packet["adaptive_profit_policy"] = profit_policy_decision
    shadow = packet.get("shadow_simulation") if isinstance(packet.get("shadow_simulation"), dict) else {}
    has_shadow_exposure = bool(shadow.get("open_order_count") or shadow.get("position_count"))
    entry_decision: dict | None = None
    entry_raw_response: dict | None = None
    entry_order_result: dict | None = None
    exit_decision: dict | None = None
    exit_raw_response: dict | None = None
    exit_order_result: dict | None = None
    if has_shadow_exposure:
        exit_packet = _packet_with_shadow_exposure(packet)
        _, exit_raw_response, exit_decision = call_response(
            client=client,
            provider=provider,
            model=resolve_llm_model(args.llm_model, provider=provider),
            system_message=EXIT_SYSTEM_MESSAGE,
            user_message=EXIT_USER_MESSAGE,
            json_schema=EXIT_JSON_SCHEMA,
            item={"market_packet_json": json.dumps(exit_packet, indent=2, default=str)},
            use_web_search=bool(args.use_web_search) and provider == "openai",
            search_context_size=args.search_context_size,
            reasoning_effort=args.reasoning_effort,
            usage_context=_usage_context(args=args, config=config, provider=provider, process="llm_options_exit_agent"),
        )
        exit_order_result = _apply_decision(
            args=args,
            broker=broker,
            config=config,
            decision=exit_decision,
            now=now,
            label_prefix="llm-exit",
            default_require_reduce_only=True,
        )
        entry_decision = {
            "action": "hold",
            "intent": "hold",
            "confidence": 1.0,
            "market_view": "Existing shadow exposure is being managed before considering new entries.",
            "reason": "entry_skipped_until_next_cycle_after_exit_review",
            "risks": [],
            "order_id": None,
            "order_ids": [],
            "order": None,
        }
        entry_order_result = {"submitted": False, "reason": "entry_skipped_while_shadow_exposure_exists"}
        decision = _normalize_exit_decision(exit_decision)
        raw_response = exit_raw_response
        order_result = exit_order_result
    else:
        _, entry_raw_response, entry_decision = call_response(
            client=client,
            provider=provider,
            model=resolve_llm_model(args.llm_model, provider=provider),
            system_message=COMBINED_SYSTEM_MESSAGE,
            user_message=COMBINED_USER_MESSAGE,
            json_schema=COMBINED_JSON_SCHEMA,
            item={"market_packet_json": json.dumps(packet, indent=2, default=str)},
            use_web_search=bool(args.use_web_search) and provider == "openai",
            search_context_size=args.search_context_size,
            reasoning_effort=args.reasoning_effort,
            usage_context=_usage_context(args=args, config=config, provider=provider, process="llm_options_entry_agent"),
        )
        entry_order_result = _apply_decision(
            args=args,
            broker=broker,
            config=config,
            decision=entry_decision,
            now=now,
            label_prefix="llm-entry",
            default_require_reduce_only=False,
        )
        exit_decision = {
            "action": "hold",
            "confidence": 1.0,
            "reason": "no_shadow_position_or_order_to_manage",
            "risks": [],
            "order_id": None,
            "order_ids": [],
            "order": None,
        }
        exit_order_result = {"submitted": False, "reason": "no_shadow_position_or_order_to_manage"}
        decision = entry_decision
        raw_response = entry_raw_response
        order_result = entry_order_result
    record = {
        "checked_at_utc": now.isoformat(),
        "account_mode": args.account_mode,
        "agent": "llm_options_agent",
        "llm_provider": provider,
        "llm_model": resolve_llm_model(args.llm_model, provider=provider),
        "trader_profile": profile,
        "python_role": "api_interface_only",
        "market_packet": packet,
        "dashboard_price_bars": full_packet.get("recent_price_bars") or [],
        "llm_decision": decision,
        "llm_raw_response": raw_response,
        "profit_policy_decision": profit_policy_decision,
        "profit_policy_raw_response": profit_policy_raw_response,
        "entry_decision": entry_decision,
        "entry_raw_response": entry_raw_response,
        "entry_order_result": entry_order_result,
        "exit_decision": exit_decision,
        "exit_raw_response": exit_raw_response,
        "exit_order_result": exit_order_result,
        "execute_testnet_orders": bool(args.execute_testnet_orders),
        "simulation_only": bool(args.simulation_only),
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
    append_memory_event(Path(args.output_dir), config.currency, compact_decision_event(process="agent", record=record))
    strategy_memory = update_strategy_memory_from_record(Path(args.output_dir), config.currency, record, max_lessons=int(args.strategy_memory_max_lessons))
    record["strategy_memory_after_update"] = strategy_memory
    return record


def _apply_decision(
    *,
    args: argparse.Namespace,
    broker,
    config: LLMOptionsRuntimeConfig,
    decision: dict,
    now: datetime,
    label_prefix: str,
    default_require_reduce_only: bool,
) -> dict:
    action = str(decision.get("action") or "hold")
    intent = str(decision.get("intent") or "hold")
    if action == "submit_order":
        order = decision.get("order") if isinstance(decision.get("order"), dict) else {}
        require_reduce_only = default_require_reduce_only or bool(order.get("reduce_only")) or intent == "close_position"
        if args.simulation_only:
            validation = execute_order_payload(
                broker=broker,
                config=config,
                decision=decision,
                require_reduce_only=require_reduce_only,
                execute=False,
                label=f"{label_prefix}-{config.currency.upper()}-{now.strftime('%Y%m%d%H%M%S')}",
            )
            order_result = {
                **validation,
                "submitted": False,
                "reason": "simulated_live_order_not_submitted",
                "simulation": {
                    "account_mode": args.account_mode,
                    "would_submit": validation.get("validated_order") or order,
                    "policy": "Live Deribit market/account data was used, but order submission is disabled in simulation-only mode.",
                },
            }
            if validation.get("validated_order"):
                order_result["simulation"].update(
                    record_simulated_order(
                        output_dir=Path(args.output_dir),
                        currency=config.currency,
                        broker=broker,
                        validated_order=validation["validated_order"],
                        decision=decision,
                        checked_at_utc=now.isoformat(),
                    )
                )
            return order_result
        if args.account_mode == "live":
            raise RuntimeError("Live real order submission is disabled for llm_options_trader.agent. Use simulation-only for live shadow runs.")
        return execute_order_payload(
            broker=broker,
            config=config,
            decision=decision,
            require_reduce_only=require_reduce_only,
            execute=bool(args.execute_testnet_orders),
            label=f"{label_prefix}-{config.currency.upper()}-{now.strftime('%Y%m%d%H%M%S')}",
        )
    if action == "cancel_order":
        if args.simulation_only:
            order_result = {
                "submitted": False,
                "reason": "simulated_live_cancel_not_submitted",
                "order_id": decision.get("order_id"),
                "order_ids": _cancel_order_ids(decision),
                "simulation": {
                    "account_mode": args.account_mode,
                    "policy": "Live Deribit market/account data was used, but cancel submission is disabled in simulation-only mode.",
                },
            }
            order_result["simulation"].update(
                record_simulated_cancels(output_dir=Path(args.output_dir), currency=config.currency, order_ids=_cancel_order_ids(decision), checked_at_utc=now.isoformat())
            )
            return order_result
        if args.account_mode == "live":
            raise RuntimeError("Live real cancellation is disabled for llm_options_trader.agent. Use simulation-only for live shadow runs.")
        return cancel_order_payload(broker=broker, decision=decision, execute=bool(args.execute_testnet_orders))
    return {"submitted": False, "reason": "llm_decision_hold"}


def _usage_context(*, args: argparse.Namespace, config: LLMOptionsRuntimeConfig, provider: str, process: str) -> dict:
    return {
        "process": process,
        "currency": config.currency,
        "instrument_currency": config.instrument_currency,
        "provider": provider,
        "output_dir": str(args.output_dir),
        "account_mode": args.account_mode,
        "simulation_only": bool(args.simulation_only),
    }


def _cancel_order_ids(decision: dict) -> list[str]:
    ids: list[str] = []
    if isinstance(decision.get("order_ids"), list):
        ids.extend(str(order_id).strip() for order_id in decision["order_ids"] if str(order_id).strip())
    single = str(decision.get("order_id") or "").strip()
    if single:
        ids.append(single)
    deduped: list[str] = []
    seen: set[str] = set()
    for order_id in ids:
        if order_id not in seen:
            deduped.append(order_id)
            seen.add(order_id)
    return deduped


def _normalize_exit_decision(decision: dict) -> dict:
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


def _packet_with_shadow_exposure(packet: dict) -> dict:
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
        "allowed_submit_order": "reduce-only sell limit for an existing positive-size shadow position",
        "allowed_cancel_order": "cancel order_id/order_ids from shadow_simulation.open_orders",
        "disallowed": "new buy entries during exit management",
    }
    return updated


def _entry_mandate(entry_bias: str) -> dict:
    normalized = str(entry_bias or "unrestricted").strip().lower()
    if normalized == "put_only":
        return {
            "mode": "put_only",
            "instruction": "For new entries, evaluate only long put setups from the provided option_chain. Do not open calls. If no put setup has sufficient edge, return hold.",
            "allowed_entry_intents": ["open_put", "hold"],
            "disallowed_entry_intents": ["open_call"],
            "order_policy": "If trading, submit a buy limit order for one executable put instrument from option_chain with reduce_only=false.",
        }
    if normalized == "call_only":
        return {
            "mode": "call_only",
            "instruction": "For new entries, evaluate only long call setups from the provided option_chain. Do not open puts. If no call setup has sufficient edge, return hold.",
            "allowed_entry_intents": ["open_call", "hold"],
            "disallowed_entry_intents": ["open_put"],
            "order_policy": "If trading, submit a buy limit order for one executable call instrument from option_chain with reduce_only=false.",
        }
    return {
        "mode": "unrestricted",
        "instruction": "For new entries, choose call, put, or hold according to the market data and strategy context.",
        "allowed_entry_intents": ["open_call", "open_put", "hold"],
        "disallowed_entry_intents": [],
    }


def _shadow_trading_budget(*, args: argparse.Namespace, config: LLMOptionsRuntimeConfig) -> dict:
    equity = max(0.0, float(args.shadow_account_equity))
    max_entry_debit = max(0.0, float(args.shadow_max_entry_debit))
    max_session_debit = max(max_entry_debit, float(args.shadow_max_session_debit))
    return {
        "mode": "simulation_only_budget",
        "instruction": (
            "For live_shadow_simulation, evaluate affordability and sizing from this simulated budget, not from the live wallet balance. "
            "The live account is included only to show real venue/account context; Python will not submit real orders."
        ),
        "currency": config.instrument_currency,
        "simulated_equity": equity,
        "max_entry_debit": max_entry_debit,
        "max_session_debit": max_session_debit,
        "max_order_amount": config.max_order_amount,
        "max_order_price": config.max_order_price,
        "position_sizing_guidance": (
            "Use the minimum executable amount for exploratory probes unless conviction and liquidity justify more. "
            "Do not reject a trade solely because the real live wallet balance is smaller than this simulated budget."
        ),
    }


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


def _load_config_overrides(args: argparse.Namespace) -> argparse.Namespace:
    config_path = Path(args.config) if getattr(args, "config", None) else None
    if not config_path:
        return args
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--config must point to a JSON object.")
    for key, value in payload.items():
        attr = key.replace("-", "_")
        if not hasattr(args, attr):
            raise ValueError(f"Unknown Deribit LLM options config key `{key}`.")
        setattr(args, attr, value)
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single LLM-authoritative Deribit testnet option trader.")
    parser.add_argument("--config", default=None, help="Path to a JSON config file. CLI args are parsed first, then config values override them.")
    parser.add_argument("--account-mode", choices=("testnet", "live"), default="testnet")
    parser.add_argument("--currency", default="ETH", help="Deribit base currency, for example ETH, BTC, or another supported option currency.")
    parser.add_argument("--instrument-currency", default="USDC", help="Deribit instrument/account currency, for example USDC, ETH, or BTC.")
    parser.add_argument("--data-provider", default="alpaca")
    parser.add_argument("--data-interval", default="1m")
    parser.add_argument("--lookback-days", type=int, default=20)
    parser.add_argument("--max-price-rows", type=int, default=3500)
    parser.add_argument("--forecast-hours", default="0.25,0.5,1")
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
    parser.add_argument("--max-order-amount", type=float, default=10.0)
    parser.add_argument("--max-order-price", type=float, default=5000.0)
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--enable-profit-policy-llm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profit-policy-llm-model", default=None)
    parser.add_argument("--entry-bias", choices=("unrestricted", "put_only", "call_only"), default="unrestricted")
    parser.add_argument("--strategy-mode", default="crypto_options_directional")
    parser.add_argument("--trader-profile", default="gambit")
    parser.add_argument("--shadow-account-equity", type=float, default=0.0, help="Simulated account equity for live-shadow strategy evaluation.")
    parser.add_argument("--shadow-max-entry-debit", type=float, default=0.0, help="Maximum simulated debit for one live-shadow option entry.")
    parser.add_argument("--shadow-max-session-debit", type=float, default=0.0, help="Maximum simulated debit for all live-shadow entries in one session.")
    parser.add_argument("--memory-events", type=int, default=12)
    parser.add_argument("--strategy-memory-lessons", type=int, default=12)
    parser.add_argument("--strategy-memory-max-lessons", type=int, default=24)
    parser.add_argument("--llm-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--use-web-search", action="store_true")
    parser.add_argument("--search-context-size", default="low")
    parser.add_argument("--check-interval-seconds", type=int, default=300)
    parser.add_argument("--output-dir", default="automated_forecasting_engine/runs/llm_options_trader_testnet")
    parser.add_argument("--execute-testnet-orders", action="store_true")
    parser.add_argument("--simulation-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--once", action="store_true")
    return parser


if __name__ == "__main__":
    main()
