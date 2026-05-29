from __future__ import annotations

import argparse
import json
from pathlib import Path

from market_forecasting_engine.daily_trade import DailyTradeConfig, build_daily_trade_plan
from market_forecasting_engine.data_providers import DataRequest, load_prices_with_provider
from market_forecasting_engine.plots import write_daily_trade_plot_artifacts


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
    parser.add_argument("--output-dir", default=None, help="Optional directory for JSON and daily-trade plot artifacts.")
    parser.add_argument("--no-plot", action="store_true", help="Disable daily-trade chart artifact generation.")
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
    output_dir = _resolve_output_dir(args.output_dir, args.output)
    if output_dir is not None and not args.no_plot:
        report.setdefault("artifacts", {}).update(
            write_daily_trade_plot_artifacts(
                report,
                result.frame,
                output_dir=output_dir,
                target_column=args.target_column,
            )
        )

    text = json.dumps(report, indent=2)
    output_path = Path(args.output) if args.output else (output_dir / "daily_trade_report.json" if output_dir else None)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


def _resolve_output_dir(output_dir: str | None, output: str | None) -> Path | None:
    if output_dir:
        return Path(output_dir)
    if output:
        return Path(output).parent
    return None


if __name__ == "__main__":
    main()
