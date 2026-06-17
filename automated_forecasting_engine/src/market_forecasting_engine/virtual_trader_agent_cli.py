from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from market_forecasting_engine.virtual_trader_agent import (
    VirtualTraderAgentConfig,
    install_launch_agent,
    run_virtual_trader_agent,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fully autonomous virtual trader agent.")
    parser.add_argument("--project-dir", default="/Users/ruddigarcia/Projects/invest")
    parser.add_argument("--output-root", default="automated_forecasting_engine/runs/virtual_trader_agent")
    parser.add_argument("--memory-path", default="automated_forecasting_engine/runs/virtual_trader/memory.json")
    parser.add_argument("--env-file", default="/Users/ruddigarcia/Projects/invest/.env")
    parser.add_argument("--loop-interval-seconds", type=int, default=14_400)
    parser.add_argument("--min-loop-interval-seconds", type=int, default=900)
    parser.add_argument("--max-loop-interval-seconds", type=int, default=21_600)
    parser.add_argument("--market-intelligence-min-refresh-seconds", type=int, default=3_600)
    parser.add_argument("--market-intelligence-max-refresh-seconds", type=int, default=21_600)
    parser.add_argument("--market-intelligence-search-context-size", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--planner-dry-run", action="store_true", help="Build planner payload but use deterministic fallback plan.")
    parser.add_argument("--once", action="store_true", help="Run one full autonomous cycle and exit.")
    parser.add_argument("--max-universe-tickers", type=int, default=350)
    parser.add_argument("--scout-final-candidates", type=int, default=8)
    parser.add_argument("--max-managed-candidates", type=int, default=5)
    parser.add_argument("--analyst-pages", type=int, default=12)
    parser.add_argument("--scout-start", default="2025-01-01")
    parser.add_argument("--forecast-start", default="2020-01-01")
    parser.add_argument("--provider", default="yahoo")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--horizons", default="5,10,20")
    parser.add_argument("--risk-profile", choices=("conservative", "medium", "aggressive"), default="medium")
    parser.add_argument("--trader-profile", choices=("conservative", "medium", "aggressive"), default="medium")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-reasoning-effort", default="none")
    parser.add_argument("--llm-timeout-seconds", type=int, default=120)
    parser.add_argument("--llm-search-context-size", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--max-notional-per-trade", type=float, default=100.0)
    parser.add_argument("--max-position-pct-equity", type=float, default=0.025)
    parser.add_argument("--dry-run", action="store_true", help="Do not submit paper orders. Default submits eligible Alpaca paper orders.")
    parser.add_argument("--allow-market-closed-orders", action="store_true")
    parser.add_argument("--allow-repeated-symbol-orders", action="store_true")
    parser.add_argument("--source-synthesis-dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--install-launch-agent", action="store_true", help="Install macOS LaunchAgent for automatic restart after login/reboot.")
    parser.add_argument("--load-launch-agent", action="store_true", help="After install, run launchctl bootstrap for this user.")
    args = parser.parse_args()

    config = VirtualTraderAgentConfig(
        project_dir=Path(args.project_dir),
        output_root=Path(args.output_root),
        memory_path=Path(args.memory_path),
        env_file=args.env_file,
        loop_interval_seconds=args.loop_interval_seconds,
        min_loop_interval_seconds=args.min_loop_interval_seconds,
        max_loop_interval_seconds=args.max_loop_interval_seconds,
        market_intelligence_min_refresh_seconds=args.market_intelligence_min_refresh_seconds,
        market_intelligence_max_refresh_seconds=args.market_intelligence_max_refresh_seconds,
        market_intelligence_search_context_size=args.market_intelligence_search_context_size,
        planner_dry_run=args.planner_dry_run,
        once=args.once,
        max_universe_tickers=args.max_universe_tickers,
        scout_final_candidates=args.scout_final_candidates,
        max_managed_candidates=args.max_managed_candidates,
        analyst_pages=args.analyst_pages,
        scout_start=args.scout_start,
        forecast_start=args.forecast_start,
        provider=args.provider,
        interval=args.interval,
        horizons=args.horizons,
        risk_profile=args.risk_profile,
        trader_profile=args.trader_profile,
        llm_model=args.llm_model,
        llm_reasoning_effort=args.llm_reasoning_effort,
        llm_timeout_seconds=args.llm_timeout_seconds,
        llm_search_context_size=args.llm_search_context_size,
        max_notional_per_trade=args.max_notional_per_trade,
        max_position_pct_equity=args.max_position_pct_equity,
        execute_paper_orders=not args.dry_run,
        allow_market_closed_orders=args.allow_market_closed_orders,
        allow_repeated_symbol_orders=args.allow_repeated_symbol_orders,
        source_synthesis_dry_run=args.source_synthesis_dry_run,
        progress=not args.no_progress,
    )
    if args.install_launch_agent:
        plist_path = install_launch_agent(config, python_executable=sys.executable)
        loaded = False
        load_result = None
        if args.load_launch_agent:
            result = subprocess.run(
                ["launchctl", "bootstrap", f"gui/{_uid()}", str(plist_path)],
                text=True,
                capture_output=True,
                check=False,
            )
            loaded = result.returncode == 0
            load_result = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        print(json.dumps({"status": "installed", "plist": str(plist_path), "loaded": loaded, "load_result": load_result}, indent=2))
        return
    run_virtual_trader_agent(config)


def _uid() -> int:
    import os

    return os.getuid()


if __name__ == "__main__":
    main()
