from __future__ import annotations

import argparse
import json
from pathlib import Path

from market_forecasting_engine.virtual_trader_pipeline import (
    VirtualTraderPipelineConfig,
    run_virtual_trader_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the autonomous virtual trader against the Alpaca paper account.")
    parser.add_argument("--output-dir", required=True, help="Directory for the active virtual trader board and per-ticker forecasts.")
    parser.add_argument("--selected-candidates", default=None, help="Path to selected_candidates.json from the scout/ranker.")
    parser.add_argument("--enrichment-board", default=None, help="Optional existing enrichment_board.json. If omitted, enrichment is run.")
    parser.add_argument("--memory-path", default="automated_forecasting_engine/runs/virtual_trader/memory.json")
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--provider", default="yahoo")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--horizons", default="5,10,20")
    parser.add_argument("--risk-profile", choices=("conservative", "medium", "aggressive"), default="medium")
    parser.add_argument("--trader-profile", choices=("conservative", "medium", "aggressive"), default="medium")
    parser.add_argument("--llm-env-file", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-reasoning-effort", default="none")
    parser.add_argument("--llm-timeout-seconds", type=int, default=120)
    parser.add_argument("--llm-search-context-size", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--enable-bayesian-heavy", action="store_true")
    parser.add_argument("--include-lstm", action="store_true")
    parser.add_argument("--deep-learning-profile", choices=("off", "fast", "research"), default="off")
    parser.add_argument("--tune", choices=("fixed", "optuna"), default="fixed")
    parser.add_argument("--optuna-trials", type=int, default=25)
    parser.add_argument("--validation-workers", type=int, default=0)
    parser.add_argument("--max-notional-per-trade", type=float, default=100.0)
    parser.add_argument("--max-position-pct-equity", type=float, default=0.025)
    parser.add_argument("--max-total-new-exposure-pct-equity", type=float, default=0.08)
    parser.add_argument("--entry-limit-offset-bps", type=float, default=10.0)
    parser.add_argument("--sell-limit-offset-bps", type=float, default=10.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not submit Alpaca paper orders. Default submits eligible Alpaca paper limit orders.",
    )
    parser.add_argument(
        "--disable-buy-lower-paper-orders",
        action="store_true",
        help="Do not place GTC buy-lower paper limit orders for Hold/wait-for-pullback plans.",
    )
    parser.add_argument("--allow-market-closed-orders", action="store_true")
    parser.add_argument("--allow-repeated-symbol-orders", action="store_true")
    parser.add_argument("--disable-alpaca", action="store_true", help="Skip broker state and order submission.")
    parser.add_argument("--source-synthesis-dry-run", action="store_true")
    parser.add_argument("--skip-enrichment", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    board = run_virtual_trader_pipeline(
        VirtualTraderPipelineConfig(
            output_dir=Path(args.output_dir),
            selected_candidates_path=Path(args.selected_candidates) if args.selected_candidates else None,
            enrichment_board_path=Path(args.enrichment_board) if args.enrichment_board else None,
            memory_path=Path(args.memory_path),
            max_candidates=args.max_candidates,
            provider=args.provider,
            start=args.start,
            interval=args.interval,
            horizons=args.horizons,
            risk_profile=args.risk_profile,
            trader_profile=args.trader_profile,
            llm_env_file=args.llm_env_file,
            llm_model=args.llm_model,
            llm_reasoning_effort=args.llm_reasoning_effort,
            llm_timeout_seconds=args.llm_timeout_seconds,
            llm_search_context_size=args.llm_search_context_size,
            enable_bayesian_heavy=args.enable_bayesian_heavy,
            include_lstm=args.include_lstm,
            deep_learning_profile=args.deep_learning_profile,
            tune=args.tune,
            optuna_trials=args.optuna_trials,
            validation_workers=args.validation_workers,
            max_notional_per_trade=args.max_notional_per_trade,
            max_position_pct_equity=args.max_position_pct_equity,
            max_total_new_exposure_pct_equity=args.max_total_new_exposure_pct_equity,
            entry_limit_offset_bps=args.entry_limit_offset_bps,
            sell_limit_offset_bps=args.sell_limit_offset_bps,
            execute_paper_orders=not args.dry_run,
            place_buy_lower_limit_orders=not args.disable_buy_lower_paper_orders,
            allow_market_closed_orders=args.allow_market_closed_orders,
            allow_repeated_symbol_orders=args.allow_repeated_symbol_orders,
            disable_alpaca=args.disable_alpaca,
            source_synthesis_dry_run=args.source_synthesis_dry_run,
            skip_enrichment=args.skip_enrichment,
            progress=not args.no_progress,
        )
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "execution_mode": board.get("policy", {}).get("default_execution"),
                "decisions": board.get("portfolio_selection", {}).get("ranked", []),
                "artifact_paths": board.get("artifact_paths", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

