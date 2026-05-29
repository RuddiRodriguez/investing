import argparse

from ingestion.db import init_db
from trader.trader_agent import initialize_trader_agent, start_trader_agent
from trading_loop.loop_agent import run_continuous_trading_loop


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the continuous trading loop.")
    parser.add_argument("--trader-name", default="semiconductor_aggressive_001", help="Trader name")
    parser.add_argument("--sector", default="semiconductors", help="Sector to trade")
    parser.add_argument("--profile-type", default="conservative", help="Trader profile type")
    parser.add_argument("--initial-cash", type=float, default=100.0, help="Initial cash amount")
    parser.add_argument("--price-timeframe", choices=["1d", "1h"], default="1d", help="Price bar timeframe")
    parser.add_argument("--max-companies", type=int, default=None, help="Max companies per sector")
    parser.add_argument("--max-cycles", type=int, default=None, help="Max trading cycles (None = unlimited)")
    parser.add_argument("--sleep-seconds", type=float, default=None, help="Seconds to sleep between cycles")
    parser.add_argument("--force-refresh", action="store_true", default=False, help="Bypass cache and force fresh data fetch")
    args = parser.parse_args()

    init_db()

    initialize_trader_agent(
        trader_name=args.trader_name,
        profile_type=args.profile_type,
        initial_cash=args.initial_cash,
    )
    start_trader_agent(trader_name=args.trader_name)

    result = run_continuous_trading_loop(
        trader_name=args.trader_name,
        sector=args.sector,
        max_cycles=args.max_cycles,
        sleep_seconds=args.sleep_seconds,
        max_companies_per_sector=args.max_companies,
        price_timeframe=args.price_timeframe,
        force_refresh=args.force_refresh,
    )
    print(result)


if __name__ == "__main__":
    main()
