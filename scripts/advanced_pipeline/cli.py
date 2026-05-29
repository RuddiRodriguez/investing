"""CLI for the standalone live advanced stock pipeline."""

from __future__ import annotations

import argparse
import json

from .config import PipelineConfig
from .pipeline import AdvancedLivePipeline, result_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", required=True, help="Comma-separated symbols, for example AAPL,MSFT,NVDA.")
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--horizons", default="5,20,60")
    parser.add_argument("--primary-horizon", type=int, default=20)
    parser.add_argument("--min-history-days", type=int, default=260)
    parser.add_argument("--train-window-days", type=int, default=756)
    parser.add_argument("--cache-ttl-hours", type=float, default=24.0)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = tuple(ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip())
    horizons = tuple(int(value.strip()) for value in args.horizons.split(",") if value.strip())
    if args.primary_horizon not in horizons:
        horizons = tuple(sorted(set(horizons + (args.primary_horizon,))))

    config = PipelineConfig(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        benchmark=args.benchmark,
        horizons=horizons,
        primary_horizon=args.primary_horizon,
        min_history_days=args.min_history_days,
        train_window_days=args.train_window_days,
        cache_ttl_hours=args.cache_ttl_hours,
        force_refresh=args.force_refresh,
    )
    result = AdvancedLivePipeline(config).run()
    records = result_records(result)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as file_handle:
            json.dump(records, file_handle, indent=2)

    print(f"As of: {result.as_of_date.date()}")
    print(f"Regime: {result.regime}")
    print(f"Cache: {result.diagnostics['cache_dir']}")
    print(f"Decision counts: {result.diagnostics['decision_counts']}")
    print(
        result.decisions.reset_index()[
            [
                "ticker",
                "decision",
                "confidence",
                "expected_excess_return",
                "lower_bound",
                "upper_bound",
                "risk_score",
                "alpha_score",
                "position_size",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
