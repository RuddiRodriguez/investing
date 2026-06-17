from __future__ import annotations

import argparse
import copy
import json
import re
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from market_forecasting_engine.alpaca_broker import AlpacaPaperBroker
from market_forecasting_engine.option_ticker_selector import (
    OptionTickerSelectorConfig,
    select_option_ticker,
    write_selection_report,
)
from market_forecasting_engine.paper_options_agent import (
    apply_option_profile_defaults,
    build_parser as build_options_agent_parser,
    clear_trade_pause_state,
    liquidate_and_stop,
    run_once,
    write_record,
    _summary,
)


DEFAULT_CHEAP_LIQUID_UNIVERSE = ("SOFI", "HOOD", "BAC", "F", "AMD", "PLTR")
CONTROLLER_REPORT = "daily_target_options_controller_report.json"
CONTROLLER_LOG = "daily_target_options_controller.jsonl"


def main() -> None:
    args = apply_option_profile_defaults(build_parser().parse_args())
    args.candidate_tickers = ",".join(_parse_tickers(args.candidate_tickers))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.stop_daily_target:
        record = liquidate_universe(args, reason="manual_stop_daily_target")
        report_path = write_controller_record(output_root, record)
        print(json.dumps({"report": str(report_path), "status": record["status"], "flat": record.get("flat")}, indent=2, default=str), flush=True)
        return
    if args.clear_existing_stop_requests:
        clear_report = clear_universe_pause_state(args)
        print(json.dumps(clear_report, indent=2, default=str), flush=True)

    controller_state = read_controller_state(output_root)
    broker = AlpacaPaperBroker()
    while True:
        record = run_controller_cycle(args=args, broker=broker, controller_state=controller_state)
        controller_state = record.get("controller_state") or controller_state
        report_path = write_controller_record(output_root, record)
        print(json.dumps(_controller_summary(record, report_path), indent=2, default=str), flush=True)
        if args.once or record.get("status") in {"daily_profit_target_reached", "daily_loss_limit_reached", "max_total_trades_reached"}:
            break
        time.sleep(max(1, int(args.check_interval_seconds)))


def build_parser() -> argparse.ArgumentParser:
    parser = build_options_agent_parser()
    parser.description = "Run a portfolio-level Alpaca paper options controller with daily target/loss gates and ticker selection."
    parser.set_defaults(
        ticker="AUTO",
        candidate_tickers=",".join(DEFAULT_CHEAP_LIQUID_UNIVERSE),
        output_dir="automated_forecasting_engine/runs/alpaca_daily_target_options_AUTO",
        forecast_hours="0.25,0.5,1",
        max_total_debit=100.0,
        max_contracts=1,
        max_trades_per_day=2,
        max_open_option_positions=1,
        max_open_option_contracts=1,
        max_open_option_orders=1,
        take_profit_position_pl=12.0,
        max_position_unrealized_loss=12.0,
        max_total_unrealized_profit=18.0,
        max_total_unrealized_loss=15.0,
        take_profit_pct=0.12,
        stop_loss_pct=0.12,
        profit_retrace_from_peak_pct=0.18,
        profit_close_limit_offset_pct=0.01,
        entry_cooldown_minutes=5.0,
        loss_cooldown_minutes=10.0,
        one_trade_per_forecast=True,
    )
    parser.add_argument("--output-root", default="automated_forecasting_engine/runs", help="Root folder for controller and per-ticker run folders.")
    parser.add_argument("--candidate-tickers", default=",".join(DEFAULT_CHEAP_LIQUID_UNIVERSE), help="Comma-separated universe to scan.")
    parser.add_argument("--selector-output-dir", default=None, help="Directory for selector reports. Defaults to output-root/daily_target_selector.")
    parser.add_argument("--max-underlying-price", type=float, default=100.0, help="Selector cap on live underlying price.")
    parser.add_argument("--max-active-tickers", type=int, default=1, help="Maximum concurrently managed tickers after selection.")
    parser.add_argument("--selection-refresh-seconds", type=int, default=300, help="Re-run selector when flat and this many seconds have passed.")
    parser.add_argument("--daily-profit-target", type=float, default=45.0, help="Portfolio daily P/L target. Reaching it liquidates and stops.")
    parser.add_argument("--daily-loss-limit", type=float, default=30.0, help="Portfolio daily P/L loss limit. Breaching it liquidates and stops.")
    parser.add_argument("--max-total-trades", type=int, default=4, help="Maximum filled buy entries across selected tickers for the UTC day.")
    parser.add_argument("--min-selector-score", type=float, default=55.0)
    parser.add_argument("--min-abs-forecast-return", type=float, default=0.001)
    parser.add_argument("--min-intraday-rows", type=int, default=120)
    parser.add_argument("--enable-llm-ticker-selection", action="store_true")
    parser.add_argument("--llm-ticker-provider", choices=("openai", "huggingface", "bedrock", "llm_studio"), default="openai")
    parser.add_argument("--llm-ticker-model", default="gpt-4o-mini")
    parser.add_argument("--llm-ticker-use-web", action="store_true")
    parser.add_argument("--stop-daily-target", action="store_true", help="Cancel/close the whole configured universe and write a controller stop report.")
    parser.add_argument("--clear-existing-stop-requests", action=argparse.BooleanOptionalAction, default=True, help="Clear prior per-ticker stop requests before starting a new daily session.")
    return parser


def run_controller_cycle(*, args: argparse.Namespace, broker: AlpacaPaperBroker, controller_state: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(UTC)
    tickers = _parse_tickers(args.candidate_tickers)
    output_root = Path(args.output_root)
    valuation_before = portfolio_valuation(broker=broker, tickers=tickers, trading_date=now.date().isoformat())
    status, stop_reason = _daily_stop_status(args=args, valuation=valuation_before)
    actions: list[dict[str, Any]] = []
    if stop_reason:
        liquidation = liquidate_universe(args, reason=stop_reason)
        actions.append(liquidation)
        valuation_after = portfolio_valuation(broker=broker, tickers=tickers, trading_date=now.date().isoformat())
        return _controller_record(
            args=args,
            checked_at=now,
            status=status,
            valuation_before=valuation_before,
            valuation_after=valuation_after,
            selector=None,
            active_tickers=[],
            agent_records=[],
            actions=actions,
            controller_state={**controller_state, "stopped_at_utc": now.isoformat(), "stop_reason": stop_reason},
        )

    active_from_broker = sorted(set(valuation_before["open_position_underlyings"]) | set(valuation_before["open_order_underlyings"]))
    active_tickers = [ticker for ticker in active_from_broker if ticker in tickers]
    selector = None
    if not active_tickers and int(valuation_before["entries_today"]) < int(args.max_total_trades):
        last_selected_at = _parse_timestamp(controller_state.get("last_selection_at_utc"))
        should_select = last_selected_at is None or (now - last_selected_at).total_seconds() >= int(args.selection_refresh_seconds)
        if should_select:
            selector = run_selector(args=args, broker=broker, now=now)
            selected = [
                str(row.get("ticker")).upper()
                for row in selector.get("eligible_candidates", [])[: max(1, int(args.max_active_tickers))]
                if row.get("ticker")
            ]
            active_tickers = selected
            controller_state["last_selection_at_utc"] = now.isoformat()
            controller_state["last_selected_tickers"] = active_tickers
        else:
            active_tickers = list(controller_state.get("last_selected_tickers") or [])[: max(1, int(args.max_active_tickers))]

    remaining_trade_slots = max(0, int(args.max_total_trades) - int(valuation_before["entries_today"]))
    if remaining_trade_slots <= 0 and not active_tickers:
        liquidation = liquidate_universe(args, reason="max_total_trades_reached")
        actions.append(liquidation)
        valuation_after = portfolio_valuation(broker=broker, tickers=tickers, trading_date=now.date().isoformat())
        return _controller_record(
            args=args,
            checked_at=now,
            status="max_total_trades_reached",
            valuation_before=valuation_before,
            valuation_after=valuation_after,
            selector=selector,
            active_tickers=[],
            agent_records=[],
            actions=actions,
            controller_state={**controller_state, "stopped_at_utc": now.isoformat(), "stop_reason": "max_total_trades_reached"},
        )

    agent_records = []
    for ticker in active_tickers[: max(1, int(args.max_active_tickers))]:
        agent_args = ticker_args(args, ticker)
        record = run_once(agent_args)
        report_path = write_record(Path(agent_args.output_dir), ticker, record)
        agent_records.append({"ticker": ticker, "report": str(report_path), "summary": _summary(record), "record": record})
    valuation_after = portfolio_valuation(broker=broker, tickers=tickers, trading_date=now.date().isoformat())
    final_status, final_stop_reason = _daily_stop_status(args=args, valuation=valuation_after)
    if final_stop_reason:
        liquidation = liquidate_universe(args, reason=final_stop_reason)
        actions.append(liquidation)
        valuation_after = portfolio_valuation(broker=broker, tickers=tickers, trading_date=now.date().isoformat())
    return _controller_record(
        args=args,
        checked_at=now,
        status=final_status,
        valuation_before=valuation_before,
        valuation_after=valuation_after,
        selector=selector,
        active_tickers=active_tickers,
        agent_records=agent_records,
        actions=actions,
        controller_state=controller_state,
    )


def run_selector(*, args: argparse.Namespace, broker: AlpacaPaperBroker, now: datetime) -> dict[str, Any]:
    selection = select_option_ticker(
        broker=broker,
        config=selector_config_from_args(args),
        now=now,
    )
    selector_dir = Path(args.selector_output_dir) if args.selector_output_dir else Path(args.output_root) / "daily_target_selector"
    report_path = write_selection_report(selector_dir, selection)
    selection["report_path"] = str(report_path)
    return selection


def selector_config_from_args(args: argparse.Namespace) -> OptionTickerSelectorConfig:
    return OptionTickerSelectorConfig(
        tickers=tuple(_parse_tickers(args.candidate_tickers)),
        provider=args.provider,
        alpaca_data_feed=str(args.alpaca_data_feed),
        interval=args.interval,
        lookback_days=int(args.lookback_days),
        forecast_hours=tuple(float(value.strip()) for value in str(args.forecast_hours).split(",") if value.strip()),
        risk_profile=args.risk_profile,
        max_training_rows=int(args.max_training_rows),
        min_dte=int(args.min_dte),
        max_dte=int(args.max_dte),
        allow_0dte=bool(args.allow_0dte),
        max_contract_premium=args.max_contract_premium,
        max_total_debit=float(args.max_total_debit),
        risk_budget_pct=args.risk_budget_pct,
        max_position_equity_pct=float(args.max_position_equity_pct),
        max_spread_pct=float(args.max_spread_pct),
        max_contracts=int(args.max_contracts),
        target_delta=float(args.target_delta),
        max_delta_distance=float(args.max_delta_distance),
        require_greeks=bool(args.require_greeks),
        max_theta_edge_ratio=float(args.max_theta_edge_ratio),
        max_theta_premium_pct_per_day=float(args.max_theta_premium_pct_per_day),
        min_open_interest=int(args.min_open_interest),
        enable_market_regime_filter=bool(args.enable_market_regime_filter),
        allow_range_edge_reversal_entry=bool(args.allow_range_edge_reversal_entry),
        market_regime_lookback_rows=int(args.market_regime_lookback_rows),
        market_regime_breakout_buffer_pct=float(args.market_regime_breakout_buffer_pct),
        market_regime_middle_zone_width=float(args.market_regime_middle_zone_width),
        min_trend_strength_pct=float(args.min_trend_strength_pct),
        enable_impulse_entry=bool(args.enable_impulse_entry),
        impulse_lookback_bars=int(args.impulse_lookback_bars),
        min_impulse_move_pct=float(args.min_impulse_move_pct),
        min_impulse_directional_bars=int(args.min_impulse_directional_bars),
        enable_late_entry_filter=bool(args.enable_late_entry_filter),
        max_late_entry_move_pct=float(args.max_late_entry_move_pct),
        max_ema_extension_pct=float(args.max_ema_extension_pct),
        exhaustion_reversal_bars=int(args.exhaustion_reversal_bars),
        limit_price_offset_pct=float(args.limit_price_offset_pct),
        stop_loss_pct=float(args.stop_loss_pct),
        take_profit_pct=float(args.take_profit_pct),
        stop_limit_offset_pct=float(args.stop_limit_offset_pct),
        min_selector_score=float(args.min_selector_score),
        min_abs_forecast_return=float(args.min_abs_forecast_return),
        min_intraday_rows=int(args.min_intraday_rows),
        max_underlying_price=args.max_underlying_price,
        enable_llm_selection=bool(args.enable_llm_ticker_selection),
        llm_provider=str(args.llm_ticker_provider),
        llm_model=str(args.llm_ticker_model),
        llm_use_web=bool(args.llm_ticker_use_web),
    )


def ticker_args(args: argparse.Namespace, ticker: str) -> argparse.Namespace:
    cloned = copy.deepcopy(args)
    cloned.ticker = ticker.upper()
    cloned.output_dir = str(Path(args.output_root) / f"alpaca_daily_target_options_{ticker.upper()}")
    Path(cloned.output_dir).mkdir(parents=True, exist_ok=True)
    return apply_option_profile_defaults(cloned)


def liquidate_universe(args: argparse.Namespace, *, reason: str) -> dict[str, Any]:
    tickers = _parse_tickers(args.candidate_tickers)
    actions = []
    for ticker in tickers:
        agent_args = ticker_args(args, ticker)
        agent_args.execute_paper_orders = bool(args.execute_paper_orders)
        try:
            record = liquidate_and_stop(agent_args)
            report_path = write_record(Path(agent_args.output_dir), ticker, record)
            actions.append({"ticker": ticker, "report": str(report_path), "summary": _summary(record)})
        except Exception as exc:  # noqa: BLE001 - emergency path must audit every failure
            actions.append({"ticker": ticker, "error": str(exc)})
    broker = AlpacaPaperBroker()
    valuation = portfolio_valuation(broker=broker, tickers=tickers, trading_date=datetime.now(UTC).date().isoformat())
    return {
        "status": "liquidation_requested",
        "reason": reason,
        "execute_paper_orders": bool(args.execute_paper_orders),
        "actions": actions,
        "valuation": valuation,
        "flat": not valuation["open_positions"] and not valuation["open_orders"],
    }


def clear_universe_pause_state(args: argparse.Namespace) -> dict[str, Any]:
    actions = []
    for ticker in _parse_tickers(args.candidate_tickers):
        agent_args = ticker_args(args, ticker)
        actions.append({"ticker": ticker, **clear_trade_pause_state(Path(agent_args.output_dir), ticker)})
    return {
        "status": "cleared_existing_daily_target_stop_requests",
        "tickers": _parse_tickers(args.candidate_tickers),
        "actions": actions,
    }


def portfolio_valuation(*, broker: AlpacaPaperBroker, tickers: list[str], trading_date: str) -> dict[str, Any]:
    tickers_set = {ticker.upper() for ticker in tickers}
    positions = [_compact_position(position) for position in broker.positions() if _option_underlying(str(position.get("symbol") or "")) in tickers_set]
    orders = [_compact_order(order) for order in broker.orders(status="open", limit=500) if _option_underlying(str(order.get("symbol") or "")) in tickers_set]
    fills = recent_option_fills(broker=broker, tickers=tickers_set, trading_date=trading_date)
    realized = realized_pnl_from_fills(fills)
    open_pnl = sum(_float(position.get("unrealized_pl")) for position in positions)
    open_exposure = sum(abs(_float(position.get("market_value")) or _float(position.get("cost_basis"))) for position in positions)
    entries_today = sum(1 for fill in fills if str(fill.get("side") or "").lower() == "buy")
    return {
        "trading_date": trading_date,
        "tickers": sorted(tickers_set),
        "realized_pnl_today": round(realized["realized_pnl"], 2),
        "open_pnl": round(open_pnl, 2),
        "total_pnl": round(realized["realized_pnl"] + open_pnl, 2),
        "open_exposure": round(open_exposure, 2),
        "entries_today": entries_today,
        "round_trips_today": int(realized["closed_lots"]),
        "open_lots": realized["open_lots"],
        "open_positions": positions,
        "open_orders": orders,
        "open_position_underlyings": sorted({_option_underlying(str(position.get("symbol") or "")) for position in positions if _option_underlying(str(position.get("symbol") or ""))}),
        "open_order_underlyings": sorted({_option_underlying(str(order.get("symbol") or "")) for order in orders if _option_underlying(str(order.get("symbol") or ""))}),
        "fills_today": fills,
        "by_symbol": realized["by_symbol"],
    }


def recent_option_fills(*, broker: AlpacaPaperBroker, tickers: set[str], trading_date: str) -> list[dict[str, Any]]:
    try:
        payload = broker._request("GET", f"/v2/account/activities/FILL?after={trading_date}T00:00:00Z&direction=asc&page_size=100")
    except Exception:
        payload = broker.orders(status="closed", limit=500)
    rows = payload if isinstance(payload, list) else []
    fills: list[dict[str, Any]] = []
    for row in rows:
        symbol = str(row.get("symbol") or "").upper()
        underlying = _option_underlying(symbol)
        if not underlying or underlying not in tickers:
            continue
        side = str(row.get("side") or row.get("order_side") or "").lower()
        qty = _float(row.get("qty") or row.get("filled_qty"))
        price = _float(row.get("price") or row.get("filled_avg_price"))
        if qty <= 0 or price <= 0:
            continue
        fills.append(
            {
                "transaction_time": row.get("transaction_time") or row.get("filled_at") or row.get("date"),
                "symbol": symbol,
                "underlying": underlying,
                "side": side,
                "qty": qty,
                "price": price,
                "order_id": row.get("order_id") or row.get("id"),
            }
        )
    fills.sort(key=lambda row: str(row.get("transaction_time") or ""))
    return fills


def realized_pnl_from_fills(fills: list[dict[str, Any]]) -> dict[str, Any]:
    lots: dict[str, deque[dict[str, float]]] = defaultdict(deque)
    by_symbol: dict[str, dict[str, Any]] = {}
    realized = 0.0
    closed_lots = 0
    for fill in fills:
        symbol = str(fill["symbol"])
        side = str(fill["side"]).lower()
        qty = float(fill["qty"])
        price = float(fill["price"])
        stats = by_symbol.setdefault(symbol, {"realized_pnl": 0.0, "buy_qty": 0.0, "sell_qty": 0.0, "closed_lots": 0})
        if side == "buy":
            lots[symbol].append({"qty": qty, "price": price})
            stats["buy_qty"] += qty
            continue
        if side != "sell":
            continue
        stats["sell_qty"] += qty
        remaining = qty
        while remaining > 1e-9 and lots[symbol]:
            lot = lots[symbol][0]
            matched = min(remaining, lot["qty"])
            pnl = (price - lot["price"]) * matched * 100.0
            realized += pnl
            stats["realized_pnl"] += pnl
            stats["closed_lots"] += matched
            closed_lots += int(round(matched))
            lot["qty"] -= matched
            remaining -= matched
            if lot["qty"] <= 1e-9:
                lots[symbol].popleft()
    open_lots = {symbol: round(sum(lot["qty"] for lot in symbol_lots), 4) for symbol, symbol_lots in lots.items() if symbol_lots}
    for stats in by_symbol.values():
        stats["realized_pnl"] = round(float(stats["realized_pnl"]), 2)
        stats["closed_lots"] = round(float(stats["closed_lots"]), 4)
    return {"realized_pnl": round(realized, 2), "closed_lots": closed_lots, "open_lots": open_lots, "by_symbol": by_symbol}


def write_controller_record(output_root: Path, record: dict[str, Any]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / CONTROLLER_REPORT
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    log_path = output_root / "logs" / CONTROLLER_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    state_path = output_root / "daily_target_options_controller_state.json"
    state_path.write_text(json.dumps(record.get("controller_state") or {}, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def read_controller_state(output_root: Path) -> dict[str, Any]:
    path = output_root / "daily_target_options_controller_state.json"
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _controller_record(
    *,
    args: argparse.Namespace,
    checked_at: datetime,
    status: str,
    valuation_before: dict[str, Any],
    valuation_after: dict[str, Any],
    selector: dict[str, Any] | None,
    active_tickers: list[str],
    agent_records: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    controller_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checked_at": checked_at.isoformat(),
        "status": status,
        "controller_config": {
            "candidate_tickers": _parse_tickers(args.candidate_tickers),
            "daily_profit_target": float(args.daily_profit_target),
            "daily_loss_limit": float(args.daily_loss_limit),
            "max_total_trades": int(args.max_total_trades),
            "max_active_tickers": int(args.max_active_tickers),
            "max_underlying_price": args.max_underlying_price,
            "max_total_debit": float(args.max_total_debit),
            "max_contracts": int(args.max_contracts),
            "option_strategy_mode": getattr(args, "option_strategy_mode", "directional"),
            "enable_multi_leg": bool(getattr(args, "enable_multi_leg", False)),
            "max_legs": int(getattr(args, "max_legs", 2)),
            "forecast_engine": getattr(args, "forecast_engine", "advanced"),
            "selection_metric": getattr(args, "selection_metric", "mae"),
            "search_level": getattr(args, "search_level", "fast"),
            "execute_paper_orders": bool(args.execute_paper_orders),
        },
        "valuation_before": valuation_before,
        "valuation_after": valuation_after,
        "selector": _selector_summary(selector),
        "active_tickers": active_tickers,
        "agent_records": agent_records,
        "controller_actions": actions,
        "controller_state": controller_state,
    }


def _controller_summary(record: dict[str, Any], report_path: Path) -> dict[str, Any]:
    valuation = record.get("valuation_after") or record.get("valuation_before") or {}
    return {
        "report": str(report_path),
        "status": record.get("status"),
        "active_tickers": record.get("active_tickers"),
        "total_pnl": valuation.get("total_pnl"),
        "realized_pnl_today": valuation.get("realized_pnl_today"),
        "open_pnl": valuation.get("open_pnl"),
        "entries_today": valuation.get("entries_today"),
        "daily_profit_target": (record.get("controller_config") or {}).get("daily_profit_target"),
        "daily_loss_limit": (record.get("controller_config") or {}).get("daily_loss_limit"),
        "selected": (record.get("selector") or {}).get("selected_ticker"),
        "agent_summaries": [row.get("summary") for row in record.get("agent_records", [])],
    }


def _selector_summary(selector: dict[str, Any] | None) -> dict[str, Any] | None:
    if not selector:
        return None
    return {
        "selected_ticker": selector.get("selected_ticker"),
        "blocked_reason": selector.get("blocked_reason"),
        "report_path": selector.get("report_path"),
        "eligible_candidates": [
            {
                "ticker": row.get("ticker"),
                "score": row.get("score"),
                "direction": (row.get("forecast") or {}).get("expected_direction"),
                "expected_return": (row.get("forecast") or {}).get("expected_return"),
                "latest_price": (row.get("price_metrics") or {}).get("latest_price"),
                "contract": ((row.get("option_trade_plan_summary") or {}).get("selected_contract") or {}).get("symbol"),
                "trade_quality": (row.get("option_trade_plan_summary") or {}).get("trade_quality"),
                "reasons": row.get("reasons"),
            }
            for row in (selector.get("eligible_candidates") or [])[:8]
        ],
        "top_rejected": [
            {
                "ticker": row.get("ticker"),
                "score": row.get("score"),
                "latest_price": (row.get("price_metrics") or {}).get("latest_price"),
                "reasons": row.get("reasons"),
            }
            for row in (selector.get("candidates") or [])[:8]
            if not row.get("eligible")
        ],
    }


def _daily_stop_status(*, args: argparse.Namespace, valuation: dict[str, Any]) -> tuple[str, str | None]:
    total_pnl = float(valuation.get("total_pnl") or 0.0)
    entries_today = int(valuation.get("entries_today") or 0)
    if total_pnl >= float(args.daily_profit_target):
        return "daily_profit_target_reached", "daily_profit_target_reached"
    if total_pnl <= -abs(float(args.daily_loss_limit)):
        return "daily_loss_limit_reached", "daily_loss_limit_reached"
    if entries_today >= int(args.max_total_trades) and not valuation.get("open_positions") and not valuation.get("open_orders"):
        return "max_total_trades_reached", "max_total_trades_reached"
    return "running", None


def _compact_position(position: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": str(position.get("symbol") or "").upper(),
        "underlying": _option_underlying(str(position.get("symbol") or "")),
        "qty": position.get("qty"),
        "avg_entry_price": position.get("avg_entry_price"),
        "current_price": position.get("current_price"),
        "market_value": position.get("market_value"),
        "cost_basis": position.get("cost_basis"),
        "unrealized_pl": position.get("unrealized_pl"),
        "unrealized_plpc": position.get("unrealized_plpc"),
    }


def _compact_order(order: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": order.get("id"),
        "symbol": str(order.get("symbol") or "").upper(),
        "underlying": _option_underlying(str(order.get("symbol") or "")),
        "side": order.get("side"),
        "type": order.get("type"),
        "qty": order.get("qty"),
        "limit_price": order.get("limit_price"),
        "stop_price": order.get("stop_price"),
        "status": order.get("status"),
        "submitted_at": order.get("submitted_at"),
    }


def _option_underlying(symbol: str) -> str | None:
    text = str(symbol or "").upper().replace(" ", "")
    match = re.match(r"^([A-Z]{1,6})\d{6}[CP]\d{8}$", text)
    return match.group(1) if match else None


def _parse_tickers(value: Any) -> list[str]:
    tickers = [part.strip().upper() for part in str(value or "").split(",") if part.strip()]
    return list(dict.fromkeys(tickers))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


if __name__ == "__main__":
    main()
