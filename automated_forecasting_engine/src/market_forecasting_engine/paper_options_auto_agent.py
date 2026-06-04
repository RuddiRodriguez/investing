from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from market_forecasting_engine.alpaca_broker import AlpacaPaperBroker
from market_forecasting_engine.option_ticker_selector import (
    DEFAULT_OPTION_TICKER_UNIVERSE,
    OptionTickerSelectorConfig,
    select_option_ticker,
    write_selection_report,
)
from market_forecasting_engine.paper_options_agent import (
    apply_option_profile_defaults,
    build_parser as build_options_agent_parser,
    clear_trade_pause_state,
    liquidate_and_stop,
    read_stop_request,
    run_once,
    write_record,
    write_stop_request,
    _summary,
)


def main() -> None:
    args = apply_option_profile_defaults(build_parser().parse_args())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.liquidate_and_stop:
        record = liquidate_and_stop(args)
        report_path = write_record(output_dir, args.ticker, record)
        print(json.dumps({"report": str(report_path), **_summary(record)}, indent=2, default=str), flush=True)
        return
    if args.stop_agent:
        request = write_stop_request(output_dir, args.ticker, reason="manual_stop_request")
        print(json.dumps({"status": "stop_requested", "ticker": args.ticker.upper(), "request": request}, indent=2, default=str), flush=True)
        return
    if args.clear_stop_request:
        result = clear_trade_pause_state(output_dir, args.ticker)
        print(json.dumps({"status": "trade_pause_cleared", "ticker": args.ticker.upper(), **result}, indent=2, default=str), flush=True)
        return
    selected = select_start_ticker(args)
    if not selected:
        return
    args.ticker = selected
    print(json.dumps({"status": "starting_paper_options_agent", "selected_ticker": args.ticker.upper(), "output_dir": str(output_dir)}, indent=2), flush=True)
    while True:
        stop_request = read_stop_request(output_dir, args.ticker)
        if stop_request and not args.once:
            print(json.dumps({"status": "stopped_by_request", "ticker": args.ticker.upper(), "request": stop_request}, indent=2, default=str), flush=True)
            break
        record = run_once(args)
        report_path = write_record(output_dir, args.ticker, record)
        print(json.dumps({"report": str(report_path), **_summary(record)}, indent=2, default=str), flush=True)
        if args.once:
            break
        time.sleep(max(1, int(args.check_interval_seconds)))


def build_parser() -> argparse.ArgumentParser:
    parser = build_options_agent_parser()
    parser.description = "Select the best same-day options ticker, then run the existing Alpaca paper options agent for that ticker."
    parser.set_defaults(ticker="AUTO")
    parser.add_argument("--candidate-tickers", default=",".join(DEFAULT_OPTION_TICKER_UNIVERSE), help="Comma-separated universe evaluated before starting the agent.")
    parser.add_argument("--selector-output-dir", default=None, help="Directory for ticker-selection reports. Defaults to output-dir/selector.")
    parser.add_argument("--selector-only", action="store_true", help="Evaluate tickers and write the selection report without starting the trading loop.")
    parser.add_argument("--min-selector-score", type=float, default=55.0)
    parser.add_argument("--min-abs-forecast-return", type=float, default=0.001)
    parser.add_argument("--min-intraday-rows", type=int, default=120)
    parser.add_argument("--enable-llm-ticker-selection", action="store_true", help="Let an LLM rank the candidates after hard liquidity/broker gates pass.")
    parser.add_argument("--llm-ticker-model", default="gpt-4o-mini")
    parser.add_argument("--llm-ticker-use-web", action="store_true", help="Allow the LLM ticker ranker to use web search. Hard broker/liquidity gates still cannot be overridden.")
    return parser


def select_start_ticker(args: argparse.Namespace) -> str | None:
    configured = str(args.ticker or "").strip().upper()
    should_select = configured in {"", "AUTO"}
    if not should_select:
        return configured
    broker = AlpacaPaperBroker()
    selection = select_option_ticker(
        broker=broker,
        config=_selector_config_from_args(args),
        now=datetime.now(UTC),
    )
    selector_dir = Path(args.selector_output_dir) if args.selector_output_dir else Path(args.output_dir) / "selector"
    report_path = write_selection_report(selector_dir, selection)
    summary = {
        "status": "ticker_selection_complete",
        "selected_ticker": selection.get("selected_ticker"),
        "report": str(report_path),
        "blocked_reason": selection.get("blocked_reason"),
        "top_candidates": [
            {
                "ticker": row.get("ticker"),
                "eligible": row.get("eligible"),
                "score": row.get("score"),
                "direction": (row.get("forecast") or {}).get("expected_direction"),
                "expected_return": (row.get("forecast") or {}).get("expected_return"),
                "contract": ((row.get("option_trade_plan_summary") or {}).get("selected_contract") or {}).get("symbol"),
                "reasons": row.get("reasons"),
            }
            for row in (selection.get("eligible_candidates") or selection.get("candidates") or [])[:5]
        ],
        "llm_decision": selection.get("llm_decision"),
    }
    print(json.dumps(summary, indent=2, default=str), flush=True)
    selected = selection.get("selected_ticker")
    if args.selector_only:
        return None
    if not selected:
        print(json.dumps({"status": "not_started", "reason": "no_ticker_passed_selector_gates", "report": str(report_path)}, indent=2), flush=True)
        return None
    return str(selected).upper()


def _selector_config_from_args(args: argparse.Namespace) -> OptionTickerSelectorConfig:
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
        limit_price_offset_pct=float(args.limit_price_offset_pct),
        stop_loss_pct=float(args.stop_loss_pct),
        take_profit_pct=float(args.take_profit_pct),
        stop_limit_offset_pct=float(args.stop_limit_offset_pct),
        min_selector_score=float(args.min_selector_score),
        min_abs_forecast_return=float(args.min_abs_forecast_return),
        min_intraday_rows=int(args.min_intraday_rows),
        enable_llm_selection=bool(args.enable_llm_ticker_selection),
        llm_model=str(args.llm_ticker_model),
        llm_use_web=bool(args.llm_ticker_use_web),
    )


def _parse_tickers(value: Any) -> list[str]:
    tickers = [part.strip().upper() for part in str(value or "").split(",") if part.strip()]
    return list(dict.fromkeys(tickers))


if __name__ == "__main__":
    main()
