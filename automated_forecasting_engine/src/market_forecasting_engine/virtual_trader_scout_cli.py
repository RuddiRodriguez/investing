from __future__ import annotations

import argparse
import json
from pathlib import Path

from market_forecasting_engine.virtual_trader_scout import ScoutConfig, run_virtual_trader_scout


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the independent virtual trader cheap discovery/scout layer.")
    parser.add_argument("--output-dir", required=True, help="Directory where scout artifacts are written.")
    parser.add_argument("--start", default="2025-01-01", help="Start date for cheap daily price scans.")
    parser.add_argument("--end", default=None, help="Optional end date for cheap daily price scans.")
    parser.add_argument("--provider", default="yahoo", help="Price provider for scout market-action features: yahoo, alpaca, polygon, csv.")
    parser.add_argument("--data-dir", default=None, help="Optional data cache root. Defaults inside output-dir.")
    parser.add_argument("--env-file", default=None, help="Optional .env file for FMP/provider API keys.")
    parser.add_argument("--max-universe-tickers", type=int, default=350, help="Maximum dynamically discovered tickers to price-scan.")
    parser.add_argument("--final-candidates", type=int, default=3, help="Number of candidates to pass to the expensive forecast stage.")
    parser.add_argument("--analyst-pages", type=int, default=12, help="Number of visible StockAnalysis analyst pages to scrape.")
    parser.add_argument("--analyst-timeout-seconds", type=float, default=20.0, help="Timeout for StockAnalysis analyst requests.")
    parser.add_argument("--min-price", type=float, default=5.0, help="Minimum latest price for an eligible candidate.")
    parser.add_argument("--min-avg-dollar-volume", type=float, default=20_000_000.0, help="Minimum 20d average dollar volume for eligibility.")
    parser.add_argument("--max-realized-volatility-20d", type=float, default=1.50, help="Maximum annualized 20d realized volatility for eligibility.")
    parser.add_argument("--disable-llm-ranking", action="store_true", help="Disable bounded LLM/web-search overlay for subjective ranking.")
    parser.add_argument("--llm-rank-top-n", type=int, default=12, help="Eligible shortlist size for subjective LLM ranking.")
    parser.add_argument("--llm-provider", default="openai", help="LLM provider for subjective ranking.")
    parser.add_argument("--llm-model", default=None, help="Optional model override for subjective ranking.")
    parser.add_argument("--llm-reasoning-effort", default="none", help="Reasoning effort for reasoning-capable models.")
    parser.add_argument("--llm-timeout-seconds", type=float, default=90.0, help="Timeout for the subjective LLM ranking call.")
    parser.add_argument("--llm-search-context-size", default="low", choices=["low", "medium", "high"], help="Web-search context size for subjective ranking.")
    parser.add_argument("--llm-no-web-search", action="store_true", help="Disable OpenAI web search tool for subjective ranking.")
    parser.add_argument("--llm-require-web-search", action="store_true", help="Require a web-search call during subjective ranking when supported.")
    parser.add_argument("--portfolio-tickers", default="", help="Comma-separated tickers already held; used for diversification scoring.")
    parser.add_argument("--portfolio-sectors", default="", help="Comma-separated sectors already held; used for diversification scoring.")
    parser.add_argument("--disable-stockanalysis-market-pages", action="store_true", help="Disable StockAnalysis gainers/losers/active/biggest-companies discovery.")
    parser.add_argument("--disable-fmp-market-movers", action="store_true", help="Disable FMP gainers/losers/actives discovery.")
    parser.add_argument("--disable-fmp-news", action="store_true", help="Disable FMP recent-news discovery.")
    parser.add_argument("--disable-fmp-earnings", action="store_true", help="Disable FMP earnings-calendar discovery.")
    parser.add_argument("--refresh-data-cache", action="store_true", help="Refresh price data instead of reusing cached scout bars.")
    parser.add_argument("--no-data-cache", action="store_true", help="Disable scout price data cache reads.")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress output.")
    args = parser.parse_args()

    summary = run_virtual_trader_scout(
        ScoutConfig(
            output_dir=Path(args.output_dir),
            start=args.start,
            end=args.end,
            provider=args.provider,
            data_dir=args.data_dir,
            env_file=args.env_file,
            max_universe_tickers=args.max_universe_tickers,
            final_candidates=args.final_candidates,
            analyst_pages=args.analyst_pages,
            analyst_timeout_seconds=args.analyst_timeout_seconds,
            include_stockanalysis_market_pages=not args.disable_stockanalysis_market_pages,
            include_fmp_market_movers=not args.disable_fmp_market_movers,
            include_fmp_news=not args.disable_fmp_news,
            include_fmp_earnings=not args.disable_fmp_earnings,
            min_price=args.min_price,
            min_avg_dollar_volume=args.min_avg_dollar_volume,
            max_realized_volatility_20d=args.max_realized_volatility_20d,
            enable_llm_ranking=not args.disable_llm_ranking,
            llm_rank_top_n=args.llm_rank_top_n,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_reasoning_effort=args.llm_reasoning_effort,
            llm_timeout_seconds=args.llm_timeout_seconds,
            llm_search_context_size=args.llm_search_context_size,
            llm_use_web_search=not args.llm_no_web_search,
            llm_require_web_search=args.llm_require_web_search,
            portfolio_tickers=tuple(value.strip() for value in args.portfolio_tickers.split(",") if value.strip()),
            portfolio_sectors=tuple(value.strip() for value in args.portfolio_sectors.split(",") if value.strip()),
            refresh_data_cache=args.refresh_data_cache,
            use_data_cache=not args.no_data_cache,
            progress=not args.no_progress,
        )
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "selected": [row.get("ticker") for row in summary.get("selected_candidates", [])],
                "counts": summary.get("counts", {}),
                "artifact_paths": summary.get("artifact_paths", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
