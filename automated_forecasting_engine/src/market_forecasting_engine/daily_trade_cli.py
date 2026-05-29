from __future__ import annotations

import argparse
import json
from pathlib import Path

from market_forecasting_engine.daily_trade import DailyTradeConfig, build_daily_trade_plan
from market_forecasting_engine.data_providers import DataRequest, load_prices_with_provider


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a same-session intraday trade plan.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, for example AAPL.")
    parser.add_argument("--csv", help="Optional local intraday OHLCV CSV.")
    parser.add_argument("--provider", default=None, help="Data provider: yahoo or csv.")
    parser.add_argument("--start", default=None, help="Provider start datetime/date.")
    parser.add_argument("--end", default=None, help="Provider end datetime/date.")
    parser.add_argument("--interval", default="5m", help="Intraday interval, for example 1m, 5m, or 15m.")
    parser.add_argument("--target-column", default="close", help="Price column used for decisions.")
    parser.add_argument("--opening-range-bars", type=int, default=6, help="Opening range length in bars.")
    parser.add_argument("--minimum-score-to-trade", type=float, default=2.0, help="Absolute signal score needed to trade.")
    parser.add_argument("--risk-reward", type=float, default=1.8, help="Take-profit distance divided by stop distance.")
    parser.add_argument("--stop-atr-multiple", type=float, default=1.2, help="ATR multiple used for stop distance.")
    parser.add_argument("--max-hold-bars", type=int, default=24, help="Maximum same-session hold length in bars.")
    parser.add_argument("--transaction-cost-bps", type=float, default=2.0, help="Estimated transaction cost in bps.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    provider_name = (args.provider or ("csv" if args.csv else "yahoo")).lower()
    if args.csv:
        provider_name = "csv"
    result = load_prices_with_provider(
        provider_name,
        DataRequest(
            ticker=args.ticker,
            start=args.start,
            end=args.end,
            interval=args.interval,
            target_column=args.target_column,
            source_path=args.csv,
        ),
        store=None,
        use_cache=False,
        refresh_cache=True,
    )
    config = DailyTradeConfig(
        ticker=args.ticker,
        interval=args.interval,
        target_column=args.target_column,
        opening_range_bars=args.opening_range_bars,
        minimum_score_to_trade=args.minimum_score_to_trade,
        risk_reward=args.risk_reward,
        stop_atr_multiple=args.stop_atr_multiple,
        max_hold_bars=args.max_hold_bars,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    report = build_daily_trade_plan(result.frame, config)
    report["data_provider"] = result.metadata

    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
