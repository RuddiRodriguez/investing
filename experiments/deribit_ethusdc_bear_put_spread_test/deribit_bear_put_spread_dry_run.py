from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
DERIBIT_BROKER_PATH = REPO_ROOT / "automated_forecasting_engine" / "src" / "market_forecasting_engine" / "deribit_broker.py"
spec = importlib.util.spec_from_file_location("deribit_broker_local", DERIBIT_BROKER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load Deribit broker from {DERIBIT_BROKER_PATH}")
deribit_broker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deribit_broker)
DeribitOptionsBroker = deribit_broker.DeribitOptionsBroker


MAX_LIVE_EUR = 5.0
DEFAULT_OUTPUT = "experiments/deribit_ethusdc_bear_put_spread_test/reports/latest_report.json"


def main() -> None:
    args = build_parser().parse_args()
    load_env_files(args)
    broker = DeribitOptionsBroker(account_mode=args.account_mode)
    state_path = Path(args.state_file)
    state = read_state(state_path)
    cycle = 0
    close_first_effects: list[dict[str, Any]] = []
    skip_first_entry = False
    if args.close_existing_first:
        close_first_effects = close_existing_options(args=args, broker=broker)
        skip_first_entry = any(effect.get("action") in {"cancelled_open_order", "submitted_close_long", "submitted_close_short"} for effect in close_first_effects)
    while True:
        cycle += 1
        report = run_notebook_flow(args=args, broker=broker, state=state)
        report["cycle"] = cycle
        report["continuous"] = bool(args.continuous)
        if skip_first_entry and cycle == 1:
            report["side_effects"] = [*close_first_effects, {"action": "entry_skipped_after_close_existing_first"}]
        else:
            report["side_effects"] = roll_rinse_then_place(args=args, broker=broker, report=report, state=state)
        update_timing_state(state=state, report=report)
        write_state(state_path, state)
        report["submit_orders"] = any(str(effect.get("action") or "").startswith("submitted_") for effect in report["side_effects"])
        report["dry_run"] = not report["submit_orders"]
        write_report(report, Path(args.output))
        print(json.dumps(console_summary(report), indent=2, sort_keys=True, default=str), flush=True)
        if not args.continuous:
            break
        if args.max_cycles is not None and cycle >= int(args.max_cycles):
            break
        time.sleep(max(1, int(args.check_interval_seconds)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Notebook-style Deribit ETH bear put spread test.")
    parser.add_argument("--account-mode", choices=("testnet", "live"), default="testnet")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--spot-instrument", default="ETH_USDC")
    parser.add_argument("--option-currency", default="ETH")
    parser.add_argument("--max-eur", type=float, default=MAX_LIVE_EUR)
    parser.add_argument("--eur-usdc-rate", type=float, default=None)
    parser.add_argument("--risk-free-rate", type=float, default=0.05)
    parser.add_argument("--target-profit-percentage", type=float, default=0.4)
    parser.add_argument("--delta-stop-loss", type=float, default=-0.50)
    parser.add_argument("--iv-stop-loss", type=float, default=0.40)
    parser.add_argument("--min-expiration-days", type=int, default=21)
    parser.add_argument("--max-expiration-days", type=int, default=60)
    parser.add_argument("--strike-range", type=float, default=0.06)
    parser.add_argument("--oi-threshold", type=float, default=1.0)
    parser.add_argument("--loose-filters", action="store_true")
    parser.add_argument("--iv-min", type=float, default=None)
    parser.add_argument("--iv-max", type=float, default=None)
    parser.add_argument("--short-delta-min", type=float, default=None)
    parser.add_argument("--short-delta-max", type=float, default=None)
    parser.add_argument("--long-delta-min", type=float, default=None)
    parser.add_argument("--long-delta-max", type=float, default=None)
    parser.add_argument("--vega-min", type=float, default=None)
    parser.add_argument("--vega-max", type=float, default=None)
    parser.add_argument("--execute-testnet-orders", action="store_true")
    parser.add_argument("--execute-live-orders", action="store_true")
    parser.add_argument("--confirm-live-deribit-options-orders", action="store_true")
    parser.add_argument("--i-understand-this-is-real-money", action="store_true")
    parser.add_argument("--close-existing-first", action="store_true")
    parser.add_argument("--rolling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--check-interval-seconds", type=int, default=60)
    parser.add_argument("--chain-refresh-seconds", type=int, default=900)
    parser.add_argument("--trade-cooldown-seconds", type=int, default=1800)
    parser.add_argument("--max-rolls-per-hour", type=int, default=1)
    parser.add_argument("--stop-after-one-complete-cycle", action="store_true")
    parser.add_argument("--state-file", default="experiments/deribit_ethusdc_bear_put_spread_test/reports/state.json")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def run_notebook_flow(*, args: argparse.Namespace, broker: DeribitOptionsBroker, state: dict[str, Any] | None = None) -> dict[str, Any]:
    now = datetime.now(UTC)
    spot_book = safe_call(lambda: broker.order_book(args.spot_instrument, depth=5))
    underlying_price = spot_price(spot_book["value"] if spot_book["ok"] else {})
    buying_power_limit_usdc = buying_power_limit(args)
    cached_scan = cached_candidate_scan(args=args, state=state or {}, now=now)
    if cached_scan is None:
        instruments = cached_instruments(args=args, broker=broker, state=state or {}, now=now)
        raw_options = instruments["value"] if instruments["ok"] else []
        put_options = get_options(
            raw_options,
            option_type="put",
            underlying_price=underlying_price,
            strike_range=float(args.strike_range),
            min_expiration_days=int(args.min_expiration_days),
            max_expiration_days=int(args.max_expiration_days),
            now=now,
        )
        short_put, long_put, blocks = find_options_for_bear_put_spread(
            broker=broker,
            put_options=put_options,
            underlying_price=underlying_price,
            risk_free_rate=float(args.risk_free_rate),
            buying_power_limit_usdc=buying_power_limit_usdc,
            oi_threshold=float(args.oi_threshold),
            criteria=option_criteria(args),
            now=now,
        )
        candidate = build_candidate(short_put, long_put, blocks)
        option_instrument_count = len(raw_options)
        put_candidate_count = len(put_options)
        if state is not None:
            state["candidate_scan_cache"] = {
                "loaded_at": now.isoformat(),
                "option_currency": args.option_currency,
                "option_instrument_count": option_instrument_count,
                "put_candidate_count_after_notebook_filters": put_candidate_count,
                "candidate": candidate,
                "execution_blocks": unique(blocks),
            }
    else:
        option_instrument_count = int(cached_scan["option_instrument_count"])
        put_candidate_count = int(cached_scan["put_candidate_count_after_notebook_filters"])
        candidate = cached_scan["candidate"]
        blocks = list(cached_scan["execution_blocks"])
    return {
        "generated_at": now.isoformat(),
        "mode": "notebook_style_deribit_bear_put_spread_test",
        "source_notebook": "https://github.com/alpacahq/alpaca-py/blob/master/examples/options/options-bear-put-spread.ipynb",
        "platform": "deribit",
        "base_url": broker.base_url,
        "account_mode": args.account_mode,
        "dry_run": True,
        "submit_orders": False,
        "spot_context_instrument": args.spot_instrument,
        "option_currency": args.option_currency,
        "policy": {
            "testnet_has_no_debit_cap": args.account_mode == "testnet",
            "max_live_eur": float(args.max_eur) if args.account_mode == "live" else None,
            "max_debit_usdc": None if buying_power_limit_usdc is None else round(buying_power_limit_usdc, 2),
            "testnet_submit_requires_execute_testnet_orders": True,
            "live_requires_three_flags": True,
            "order_flow": "buy long put first, then sell short put",
            "roll_rinse": {
                "rolling": bool(args.rolling),
                "target_profit_percentage": float(args.target_profit_percentage),
                "delta_stop_loss": float(args.delta_stop_loss),
                "iv_stop_loss": float(args.iv_stop_loss),
            },
            "timing": {
                "chain_refresh_seconds": int(args.chain_refresh_seconds),
                "trade_cooldown_seconds": int(args.trade_cooldown_seconds),
                "max_rolls_per_hour": int(args.max_rolls_per_hour),
                "stop_after_one_complete_cycle": bool(args.stop_after_one_complete_cycle),
            },
            "filter_mode": "loose" if args.loose_filters else "notebook",
            "criteria": option_criteria(args),
        },
        "spot_context": compact_book(spot_book),
        "underlying_price": underlying_price,
        "option_instrument_count": option_instrument_count,
        "put_candidate_count_after_notebook_filters": put_candidate_count,
        "candidate": candidate,
        "execution_blocks": unique(blocks),
    }


def get_options(
    raw_options: list[dict[str, Any]],
    *,
    option_type: str,
    underlying_price: float | None,
    strike_range: float,
    min_expiration_days: int,
    max_expiration_days: int,
    now: datetime,
) -> list[dict[str, Any]]:
    if underlying_price is None:
        return []
    min_strike = underlying_price * (1.0 - strike_range)
    max_strike = underlying_price * (1.0 + strike_range)
    output = []
    for row in raw_options:
        if str(row.get("option_type") or "").lower() != option_type:
            continue
        strike = number(row.get("strike"))
        expiry = expiry_datetime(row)
        if strike is None or expiry is None:
            continue
        remaining_days = (expiry - now).total_seconds() / 86400.0
        if min_strike <= strike <= max_strike and min_expiration_days <= remaining_days <= max_expiration_days:
            output.append(row)
    return output


def cached_instruments(*, args: argparse.Namespace, broker: DeribitOptionsBroker, state: dict[str, Any], now: datetime) -> dict[str, Any]:
    cache = state.get("instrument_cache") if isinstance(state.get("instrument_cache"), dict) else {}
    age_seconds = None
    if cache.get("loaded_at"):
        try:
            age_seconds = (now - datetime.fromisoformat(cache["loaded_at"])).total_seconds()
        except ValueError:
            age_seconds = None
    if (
        cache.get("option_currency") == args.option_currency
        and isinstance(cache.get("rows"), list)
        and age_seconds is not None
        and age_seconds < int(args.chain_refresh_seconds)
    ):
        return {"ok": True, "value": cache["rows"], "cached": True, "age_seconds": age_seconds}
    result = safe_call(lambda: broker.instruments(currency=args.option_currency, kind="option", expired=False))
    if result["ok"]:
        state["instrument_cache"] = {
            "loaded_at": now.isoformat(),
            "option_currency": args.option_currency,
            "rows": result["value"],
        }
    return result


def cached_candidate_scan(*, args: argparse.Namespace, state: dict[str, Any], now: datetime) -> dict[str, Any] | None:
    cache = state.get("candidate_scan_cache") if isinstance(state.get("candidate_scan_cache"), dict) else {}
    loaded_at = parse_iso_time(cache.get("loaded_at"))
    if loaded_at is None:
        return None
    if cache.get("option_currency") != args.option_currency:
        return None
    if (now - loaded_at).total_seconds() >= int(args.chain_refresh_seconds):
        return None
    if not isinstance(cache.get("candidate"), dict):
        return None
    return cache


def find_options_for_bear_put_spread(
    *,
    broker: DeribitOptionsBroker,
    put_options: list[dict[str, Any]],
    underlying_price: float | None,
    risk_free_rate: float,
    buying_power_limit_usdc: float | None,
    oi_threshold: float,
    criteria: dict[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]],
    now: datetime,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    if underlying_price is None:
        return None, None, ["missing_underlying_price"]
    short_by_expiry: dict[int, list[dict[str, Any]]] = {}
    long_by_expiry: dict[int, list[dict[str, Any]]] = {}
    for option_data in put_options:
        candidate = build_option_dict(
            broker=broker,
            option_data=option_data,
            underlying_price=underlying_price,
            risk_free_rate=risk_free_rate,
            now=now,
        )
        if candidate is None:
            continue
        if (candidate.get("open_interest") or 0.0) < oi_threshold:
            continue
        expiry_key = int(candidate["expiration_timestamp"])
        if check_candidate_option_conditions(candidate, criteria["short_put"]):
            short_by_expiry.setdefault(expiry_key, []).append(candidate)
        if check_candidate_option_conditions(candidate, criteria["long_put"]):
            long_by_expiry.setdefault(expiry_key, []).append(candidate)
    for expiry_key in sorted(set(short_by_expiry) & set(long_by_expiry)):
        short_put, long_put = pair_put_candidates(short_by_expiry[expiry_key], long_by_expiry[expiry_key], underlying_price)
        if short_put and long_put:
            ok, block = check_buying_power(short_put, long_put, buying_power_limit_usdc)
            if ok:
                return short_put, long_put, []
            return short_put, long_put, [block]
    return None, None, ["no_valid_bear_put_spread_found_by_notebook_filters"]


def build_option_dict(
    *,
    broker: DeribitOptionsBroker,
    option_data: dict[str, Any],
    underlying_price: float,
    risk_free_rate: float,
    now: datetime,
) -> dict[str, Any] | None:
    name = str(option_data.get("instrument_name") or "")
    if not name:
        return None
    book = broker.order_book(name, depth=5)
    bid = number(book.get("best_bid_price"))
    ask = number(book.get("best_ask_price"))
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return None
    option_price_eth = (bid + ask) / 2.0
    option_price_usdc = option_price_eth * underlying_price
    expiry = expiry_datetime(option_data)
    strike = number(option_data.get("strike"))
    if expiry is None or strike is None:
        return None
    remaining_days = max(0.0, (expiry - now).total_seconds() / 86400.0)
    iv, delta, gamma, theta, vega = calculate_option_metrics(
        option_price=option_price_usdc,
        strike_price=strike,
        expiration=expiry,
        underlying_price=underlying_price,
        risk_free_rate=risk_free_rate,
        option_type="put",
        now=now,
    )
    stats = book.get("stats") if isinstance(book.get("stats"), dict) else {}
    return {
        "instrument_name": name,
        "underlying_symbol": option_data.get("base_currency"),
        "option_type": "put",
        "strike_price": strike,
        "expiration_timestamp": option_data.get("expiration_timestamp"),
        "remaining_days": remaining_days,
        "initial_option_price_eth": option_price_eth,
        "initial_option_price_usdc": option_price_usdc,
        "bid_eth": bid,
        "ask_eth": ask,
        "iv": iv,
        "initial_delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "open_interest": number(stats.get("open_interest")) or 0.0,
        "size": number(option_data.get("min_trade_amount")) or 1.0,
    }


def check_candidate_option_conditions(candidate: dict[str, Any], criteria: tuple[Any, Any, Any, Any]) -> bool:
    expiration_range, iv_range, delta_range, vega_range = criteria
    return (
        expiration_range[0] <= candidate["remaining_days"] <= expiration_range[1]
        and iv_range[0] <= candidate["iv"] <= iv_range[1]
        and delta_range[0] <= candidate["initial_delta"] <= delta_range[1]
        and vega_range[0] <= candidate["vega"] <= vega_range[1]
    )


def option_criteria(args: argparse.Namespace) -> dict[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]]:
    if args.loose_filters:
        default_iv = (0.05, 2.50)
        default_short_delta = (-0.85, -0.02)
        default_long_delta = (-0.95, -0.02)
        default_vega = (0.0, 2.0)
    else:
        default_iv = (0.20, 0.50)
        default_short_delta = (-0.60, -0.10)
        default_long_delta = (-0.70, -0.20)
        default_vega = (0.01, 0.20)
    iv_range = (value_or(args.iv_min, default_iv[0]), value_or(args.iv_max, default_iv[1]))
    vega_range = (value_or(args.vega_min, default_vega[0]), value_or(args.vega_max, default_vega[1]))
    return {
        "short_put": (
            (float(args.min_expiration_days), float(args.max_expiration_days)),
            iv_range,
            (value_or(args.short_delta_min, default_short_delta[0]), value_or(args.short_delta_max, default_short_delta[1])),
            vega_range,
        ),
        "long_put": (
            (float(args.min_expiration_days), float(args.max_expiration_days)),
            iv_range,
            (value_or(args.long_delta_min, default_long_delta[0]), value_or(args.long_delta_max, default_long_delta[1])),
            vega_range,
        ),
    }


def value_or(value: float | None, default: float) -> float:
    return default if value is None else float(value)


def pair_put_candidates(short_puts: list[dict[str, Any]], long_puts: list[dict[str, Any]], underlying_price: float) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for short_put in sorted(short_puts, key=lambda row: row["strike_price"]):
        for long_put in sorted(long_puts, key=lambda row: row["strike_price"]):
            if short_put["strike_price"] <= underlying_price < long_put["strike_price"]:
                return short_put, long_put
    return None, None


def check_buying_power(short_put: dict[str, Any], long_put: dict[str, Any], buying_power_limit_usdc: float | None) -> tuple[bool, str]:
    if buying_power_limit_usdc is None:
        return True, ""
    size = max(float(short_put["size"]), float(long_put["size"]))
    risk = (long_put["initial_option_price_usdc"] - short_put["initial_option_price_usdc"]) * size
    if risk <= buying_power_limit_usdc:
        return True, ""
    return False, "buying_power_limit_exceeded_for_bear_put_spread_risk"


def place_bear_put_spread_order(*, args: argparse.Namespace, broker: DeribitOptionsBroker, report: dict[str, Any], state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if args.account_mode == "testnet" and not args.execute_testnet_orders:
        return [{"action": "dry_run_only", "reason": "missing_execute_testnet_orders"}]
    if args.account_mode == "live" and not args.execute_live_orders:
        return [{"action": "dry_run_only", "reason": "missing_execute_live_orders"}]
    blocks = execution_blocks(args=args, report=report)
    if blocks:
        return [{"action": "submit_blocked", "blocks": blocks}]
    timing_blocks = timing_blocks_for_entry(args=args, state=state or {})
    if timing_blocks:
        return [{"action": "entry_blocked_by_timing", "blocks": timing_blocks}]
    candidate = report["candidate"]
    short_put = candidate["short_put"]
    long_put = candidate["long_put"]
    amount = float(candidate["amount"])
    label = f"notebook-bear-put-{args.account_mode}"
    effects = []
    try:
        long_order = broker.buy_limit(
            instrument_name=long_put["instrument_name"],
            amount=amount,
            price=round(float(long_put["ask_eth"]), 6),
            label=f"{label}-long",
            post_only=False,
            reduce_only=False,
        )
        effects.append({"action": "submitted_long_put", "order": long_order})
        short_order = broker.sell_limit(
            instrument_name=short_put["instrument_name"],
            amount=amount,
            price=round(float(short_put["bid_eth"]), 6),
            label=f"{label}-short",
            post_only=False,
            reduce_only=False,
        )
        effects.append({"action": "submitted_short_put", "order": short_order})
    except Exception as exc:
        effects.append({"action": "live_submit_failed", "error": f"{type(exc).__name__}: {exc}"})
    return effects


def roll_rinse_then_place(*, args: argparse.Namespace, broker: DeribitOptionsBroker, report: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    if args.stop_after_one_complete_cycle and state.get("completed_cycle"):
        return [{"action": "stopped_after_one_complete_cycle"}]
    management = roll_rinse_bear_put_spread(args=args, broker=broker, report=report, state=state)
    if management and management[-1].get("action") == "hold_existing_spread":
        return management
    if management and not args.rolling:
        return management
    return [*management, *place_bear_put_spread_order(args=args, broker=broker, report=report, state=state)]


def roll_rinse_bear_put_spread(*, args: argparse.Namespace, broker: DeribitOptionsBroker, report: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    spread = current_bear_put_spread(args=args, broker=broker, underlying_price=report.get("underlying_price"))
    if spread is None:
        return []
    short_put = spread["short_put"]
    long_put = spread["long_put"]
    current_short_price = float(short_put["initial_option_price_usdc"])
    target_price = current_short_price * float(args.target_profit_percentage)
    current_delta = float(short_put["initial_delta"])
    current_iv = float(short_put["iv"])
    should_close = (
        current_short_price <= target_price
        or abs(current_delta) <= float(args.delta_stop_loss)
        or current_iv >= float(args.iv_stop_loss)
    )
    if not should_close:
        return [
            {
                "action": "hold_existing_spread",
                "reason": "roll_rinse_exit_criteria_not_met",
                "current_short_price_usdc": round(current_short_price, 2),
                "target_price_usdc": round(target_price, 2),
                "delta": current_delta,
                "iv": current_iv,
                "short_put": short_put["instrument_name"],
                "long_put": long_put["instrument_name"],
            }
        ]
    timing_blocks = timing_blocks_for_roll(args=args, state=state)
    if timing_blocks:
        return [
            {
                "action": "hold_existing_spread",
                "reason": "roll_rinse_exit_criteria_met_but_timing_blocked",
                "blocks": timing_blocks,
                "current_short_price_usdc": round(current_short_price, 2),
                "target_price_usdc": round(target_price, 2),
                "delta": current_delta,
                "iv": current_iv,
                "short_put": short_put["instrument_name"],
                "long_put": long_put["instrument_name"],
            }
        ]
    effects = close_spread_legs(args=args, broker=broker, short_put=short_put, long_put=long_put)
    effects.append(
        {
            "action": "rolled" if args.rolling else "rinsed",
            "reason": "roll_rinse_exit_criteria_met",
            "current_short_price_usdc": round(current_short_price, 2),
            "target_price_usdc": round(target_price, 2),
            "delta": current_delta,
            "iv": current_iv,
        }
    )
    return effects


def current_bear_put_spread(*, args: argparse.Namespace, broker: DeribitOptionsBroker, underlying_price: float | None) -> dict[str, Any] | None:
    if underlying_price is None:
        return None
    instruments = {row.get("instrument_name"): row for row in broker.instruments(currency=args.option_currency, kind="option", expired=False)}
    positions = broker.positions(currency=args.option_currency, kind="option")
    put_positions = []
    for position in positions:
        instrument_name = str(position.get("instrument_name") or "")
        instrument = instruments.get(instrument_name) or parse_instrument_name(instrument_name, args.option_currency)
        if str(instrument.get("option_type") or "").lower() != "put":
            continue
        size = number(position.get("size")) or 0.0
        if size == 0.0:
            continue
        option_dict = build_option_dict(
            broker=broker,
            option_data=instrument,
            underlying_price=float(underlying_price),
            risk_free_rate=float(args.risk_free_rate),
            now=datetime.now(UTC),
        )
        if option_dict is None:
            continue
        option_dict["position_size"] = size
        option_dict["average_price_eth"] = number(position.get("average_price"))
        put_positions.append(option_dict)
    shorts = sorted([row for row in put_positions if row["position_size"] < 0], key=lambda row: row["strike_price"])
    longs = sorted([row for row in put_positions if row["position_size"] > 0], key=lambda row: row["strike_price"])
    for short_put in shorts:
        for long_put in longs:
            if short_put["strike_price"] < long_put["strike_price"]:
                return {"short_put": short_put, "long_put": long_put}
    return None


def close_spread_legs(*, args: argparse.Namespace, broker: DeribitOptionsBroker, short_put: dict[str, Any], long_put: dict[str, Any]) -> list[dict[str, Any]]:
    effects = []
    if args.account_mode == "testnet" and not args.execute_testnet_orders:
        return [{"action": "close_spread_dry_run", "reason": "missing_execute_testnet_orders"}]
    if args.account_mode == "live" and not args.execute_live_orders:
        return [{"action": "close_spread_dry_run", "reason": "missing_execute_live_orders"}]
    try:
        short_book = broker.order_book(short_put["instrument_name"], depth=5)
        long_book = broker.order_book(long_put["instrument_name"], depth=5)
        short_amount = abs(float(short_put.get("position_size") or short_put.get("size") or 1.0))
        long_amount = abs(float(long_put.get("position_size") or long_put.get("size") or 1.0))
        effects.append(
            {
                "action": "submitted_close_short",
                "instrument_name": short_put["instrument_name"],
                "amount": short_amount,
                "order": broker.buy_limit(
                    instrument_name=short_put["instrument_name"],
                    amount=short_amount,
                    price=close_buy_price(short_book),
                    label=f"notebook-rinse-{args.account_mode}",
                    post_only=False,
                    reduce_only=True,
                ),
            }
        )
        effects.append(
            {
                "action": "submitted_close_long",
                "instrument_name": long_put["instrument_name"],
                "amount": long_amount,
                "order": broker.sell_limit(
                    instrument_name=long_put["instrument_name"],
                    amount=long_amount,
                    price=close_sell_price(long_book),
                    label=f"notebook-rinse-{args.account_mode}",
                    post_only=False,
                    reduce_only=True,
                ),
            }
        )
    except Exception as exc:
        effects.append({"action": "close_spread_failed", "error": f"{type(exc).__name__}: {exc}"})
    return effects


def close_existing_options(*, args: argparse.Namespace, broker: DeribitOptionsBroker) -> list[dict[str, Any]]:
    effects: list[dict[str, Any]] = []
    if args.account_mode == "testnet" and not args.execute_testnet_orders:
        return [{"action": "close_existing_dry_run", "reason": "missing_execute_testnet_orders"}]
    if args.account_mode == "live" and not args.execute_live_orders:
        return [{"action": "close_existing_dry_run", "reason": "missing_execute_live_orders"}]
    if args.account_mode == "live" and (not args.confirm_live_deribit_options_orders or not args.i_understand_this_is_real_money):
        return [{"action": "close_existing_blocked", "reason": "missing_live_confirmation_flags"}]
    try:
        open_orders = broker.open_orders(currency=args.option_currency, kind="option")
        for order in open_orders:
            order_id = order.get("order_id")
            if not order_id:
                continue
            try:
                effects.append({"action": "cancelled_open_order", "order": broker.cancel_order(str(order_id))})
            except Exception as exc:
                effects.append({"action": "cancel_open_order_failed", "order_id": order_id, "error": f"{type(exc).__name__}: {exc}"})
        positions = broker.positions(currency=args.option_currency, kind="option")
        for position in positions:
            instrument_name = str(position.get("instrument_name") or "")
            size = number(position.get("size")) or 0.0
            if not instrument_name or size == 0.0:
                continue
            book = broker.order_book(instrument_name, depth=5)
            amount = abs(size)
            if size > 0:
                price = close_sell_price(book)
                effects.append(
                    {
                        "action": "submitted_close_long",
                        "instrument_name": instrument_name,
                        "amount": amount,
                        "price": price,
                        "order": broker.sell_limit(
                            instrument_name=instrument_name,
                            amount=amount,
                            price=price,
                            label=f"notebook-close-{args.account_mode}",
                            post_only=False,
                            reduce_only=True,
                        ),
                    }
                )
            else:
                price = close_buy_price(book)
                effects.append(
                    {
                        "action": "submitted_close_short",
                        "instrument_name": instrument_name,
                        "amount": amount,
                        "price": price,
                        "order": broker.buy_limit(
                            instrument_name=instrument_name,
                            amount=amount,
                            price=price,
                            label=f"notebook-close-{args.account_mode}",
                            post_only=False,
                            reduce_only=True,
                        ),
                    }
                )
    except Exception as exc:
        effects.append({"action": "close_existing_failed", "error": f"{type(exc).__name__}: {exc}"})
    if not effects:
        effects.append({"action": "no_existing_options_to_close"})
    return effects


def close_sell_price(book: dict[str, Any]) -> float:
    bid = number(book.get("best_bid_price"))
    mark = number(book.get("mark_price"))
    price = bid if bid and bid > 0 else (mark or 0.0001) * 0.95
    return max(0.0001, round(price, 6))


def close_buy_price(book: dict[str, Any]) -> float:
    ask = number(book.get("best_ask_price"))
    mark = number(book.get("mark_price"))
    price = ask if ask and ask > 0 else (mark or 0.0001) * 1.05
    return max(0.0001, round(price, 6))


def execution_blocks(*, args: argparse.Namespace, report: dict[str, Any]) -> list[str]:
    blocks = []
    if args.account_mode == "testnet" and not args.execute_testnet_orders:
        blocks.append("missing_execute_testnet_orders")
    if args.account_mode == "live":
        if not args.execute_live_orders:
            blocks.append("missing_execute_live_orders")
        if not args.confirm_live_deribit_options_orders:
            blocks.append("missing_confirm_live_deribit_options_orders")
        if not args.i_understand_this_is_real_money:
            blocks.append("missing_real_money_ack")
        if float(args.max_eur) > MAX_LIVE_EUR:
            blocks.append("max_eur_above_5")
    if report["candidate"]["status"] != "planned":
        blocks.append("candidate_not_planned")
    max_debit = report["policy"].get("max_debit_usdc")
    if max_debit is not None and report["candidate"].get("estimated_risk_usdc", 10**9) > max_debit:
        blocks.append("estimated_risk_above_max_eur")
    return unique(blocks)


def build_candidate(short_put: dict[str, Any] | None, long_put: dict[str, Any] | None, blocks: list[str]) -> dict[str, Any]:
    if not short_put or not long_put:
        return {"status": "blocked", "blocks": ["no_valid_bear_put_spread_found_by_notebook_filters"]}
    amount = max(float(short_put["size"]), float(long_put["size"]))
    risk = (long_put["initial_option_price_usdc"] - short_put["initial_option_price_usdc"]) * amount
    candidate_blocks = list(blocks)
    return {
        "status": "planned" if not candidate_blocks else "blocked",
        "strategy": "bear_put_spread",
        "short_put": short_put,
        "long_put": long_put,
        "amount": amount,
        "estimated_risk_usdc": round(risk, 2),
        "blocks": unique(candidate_blocks),
    }


def calculate_option_metrics(
    *,
    option_price: float,
    strike_price: float,
    expiration: datetime,
    underlying_price: float,
    risk_free_rate: float,
    option_type: str,
    now: datetime,
) -> tuple[float, float, float, float, float]:
    years = max((expiration - now).total_seconds() / (365.0 * 86400.0), 1e-6)
    iv = calculate_implied_volatility(option_price, underlying_price, strike_price, years, risk_free_rate, option_type)
    d1 = (math.log(underlying_price / strike_price) + (risk_free_rate + 0.5 * iv**2) * years) / (iv * math.sqrt(years))
    d2 = d1 - iv * math.sqrt(years)
    delta = normal_cdf(d1) if option_type == "call" else -normal_cdf(-d1)
    gamma = normal_pdf(d1) / (underlying_price * iv * math.sqrt(years))
    vega = underlying_price * normal_pdf(d1) * math.sqrt(years) / 100.0
    if option_type == "put":
        theta = (
            -(underlying_price * normal_pdf(d1) * iv) / (2 * math.sqrt(years))
            + risk_free_rate * strike_price * math.exp(-risk_free_rate * years) * normal_cdf(-d2)
        ) / 365.0
    else:
        theta = (
            -(underlying_price * normal_pdf(d1) * iv) / (2 * math.sqrt(years))
            - risk_free_rate * strike_price * math.exp(-risk_free_rate * years) * normal_cdf(d2)
        ) / 365.0
    return iv, delta, gamma, theta, vega


def calculate_implied_volatility(option_price: float, spot: float, strike: float, years: float, rate: float, option_type: str) -> float:
    low, high = 0.01, 5.0
    for _ in range(80):
        mid = (low + high) / 2.0
        price = black_scholes_price(spot, strike, years, rate, mid, option_type)
        if price > option_price:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0


def black_scholes_price(spot: float, strike: float, years: float, rate: float, vol: float, option_type: str) -> float:
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol**2) * years) / (vol * math.sqrt(years))
    d2 = d1 - vol * math.sqrt(years)
    if option_type == "put":
        return strike * math.exp(-rate * years) * normal_cdf(-d2) - spot * normal_cdf(-d1)
    return spot * normal_cdf(d1) - strike * math.exp(-rate * years) * normal_cdf(d2)


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def safe_call(fn: Any) -> dict[str, Any]:
    try:
        return {"ok": True, "value": fn()}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def compact_book(result: dict[str, Any]) -> dict[str, Any]:
    if not result["ok"]:
        return result
    value = result["value"]
    return {
        "ok": True,
        "instrument_name": value.get("instrument_name"),
        "mark_price": value.get("mark_price"),
        "best_bid_price": value.get("best_bid_price"),
        "best_ask_price": value.get("best_ask_price"),
        "index_price": value.get("index_price"),
    }


def spot_price(book: dict[str, Any]) -> float | None:
    return number(book.get("mark_price")) or book_mid(book) or number(book.get("index_price"))


def book_mid(book: dict[str, Any]) -> float | None:
    bid = number(book.get("best_bid_price"))
    ask = number(book.get("best_ask_price"))
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return None
    return (bid + ask) / 2.0


def expiry_datetime(row: dict[str, Any]) -> datetime | None:
    value = number(row.get("expiration_timestamp"))
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1000.0, tz=UTC)


def parse_instrument_name(instrument_name: str, currency: str) -> dict[str, Any]:
    parts = instrument_name.split("-")
    if len(parts) < 4:
        return {"instrument_name": instrument_name, "option_type": None}
    expiry = parse_deribit_expiry(parts[1])
    option_code = parts[3].upper()
    return {
        "instrument_name": instrument_name,
        "base_currency": currency.upper(),
        "option_type": "put" if option_code == "P" else "call" if option_code == "C" else None,
        "strike": number(parts[2]),
        "expiration_timestamp": int(expiry.timestamp() * 1000) if expiry else None,
        "min_trade_amount": 1.0,
    }


def parse_deribit_expiry(value: str) -> datetime | None:
    months = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12,
    }
    value = value.upper()
    try:
        day = int(value[:2]) if value[:2].isdigit() else int(value[:1])
        month_text = value[-5:-2]
        year = 2000 + int(value[-2:])
        return datetime(year, months[month_text], day, tzinfo=UTC)
    except Exception:
        return None


def resolve_eur_usdc_rate(args: argparse.Namespace) -> float:
    if args.eur_usdc_rate is not None:
        return float(args.eur_usdc_rate)
    try:
        request = Request("https://api.frankfurter.app/latest?from=EUR&to=USD", method="GET")
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return float(payload["rates"]["USD"])
    except Exception:
        return 1.08


def buying_power_limit(args: argparse.Namespace) -> float | None:
    if args.account_mode == "testnet":
        return None
    return float(args.max_eur) * resolve_eur_usdc_rate(args)


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    stamped = output_path.parent / f"report_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    stamped.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True, default=str), encoding="utf-8")


def update_timing_state(*, state: dict[str, Any], report: dict[str, Any]) -> None:
    now = report.get("generated_at") or datetime.now(UTC).isoformat()
    actions = [str(effect.get("action") or "") for effect in report.get("side_effects", [])]
    submitted = [action for action in actions if action.startswith("submitted_")]
    if submitted:
        state["last_trade_at"] = now
    if any(action in {"submitted_long_put", "submitted_short_put"} for action in actions):
        state["last_entry_at"] = now
    if any(action in {"rolled", "rinsed"} for action in actions):
        state.setdefault("roll_events", []).append(now)
    if any(action in {"submitted_close_short", "submitted_close_long"} for action in actions) and any(action in {"submitted_long_put", "submitted_short_put"} for action in actions):
        state["completed_cycle"] = True
    cutoff = datetime.now(UTC).timestamp() - 3600.0
    recent = []
    for value in state.get("roll_events", []):
        try:
            if datetime.fromisoformat(value).timestamp() >= cutoff:
                recent.append(value)
        except Exception:
            pass
    state["roll_events"] = recent


def timing_blocks_for_entry(*, args: argparse.Namespace, state: dict[str, Any]) -> list[str]:
    blocks = []
    last_trade = parse_iso_time(state.get("last_trade_at"))
    if last_trade is not None:
        elapsed = (datetime.now(UTC) - last_trade).total_seconds()
        if elapsed < int(args.trade_cooldown_seconds):
            blocks.append("trade_cooldown_active")
    return blocks


def timing_blocks_for_roll(*, args: argparse.Namespace, state: dict[str, Any]) -> list[str]:
    blocks = timing_blocks_for_entry(args=args, state=state)
    cutoff = datetime.now(UTC).timestamp() - 3600.0
    recent_rolls = 0
    for value in state.get("roll_events", []):
        parsed = parse_iso_time(value)
        if parsed is not None and parsed.timestamp() >= cutoff:
            recent_rolls += 1
    if recent_rolls >= int(args.max_rolls_per_hour):
        blocks.append("max_rolls_per_hour_reached")
    return unique(blocks)


def parse_iso_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def load_env_files(args: argparse.Namespace) -> None:
    paths = [
        Path(args.env_file).expanduser(),
        Path.cwd() / ".env",
        REPO_ROOT / ".env",
        Path("/Users/ruddigarcia/Projects/invest/.env"),
    ]
    seen = set()
    for path in paths:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        load_env_file(path)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "account_mode": report["account_mode"],
        "cycle": report.get("cycle"),
        "continuous": report.get("continuous"),
        "dry_run": report["dry_run"],
        "submit_orders": report["submit_orders"],
        "spot_context_instrument": report["spot_context_instrument"],
        "option_currency": report["option_currency"],
        "underlying_price": report["underlying_price"],
        "put_candidate_count_after_notebook_filters": report["put_candidate_count_after_notebook_filters"],
        "candidate_status": report["candidate"]["status"],
        "estimated_risk_usdc": report["candidate"].get("estimated_risk_usdc"),
        "max_live_eur": report["policy"].get("max_live_eur"),
        "testnet_has_no_debit_cap": report["policy"].get("testnet_has_no_debit_cap"),
        "execution_blocks": report["execution_blocks"],
        "side_effects": report.get("side_effects", []),
    }


if __name__ == "__main__":
    main()
