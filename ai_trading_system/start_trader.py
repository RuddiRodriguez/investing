import argparse

from ingestion.db import init_db
from trader.trader_agent import start_trader_agent


DEFAULT_TRADER_NAME = "semiconductor_aggressive_001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a trader by name.")
    parser.add_argument(
        "--trader-name",
        "-t",
        default=DEFAULT_TRADER_NAME,
        help=f"Trader name to start (default: {DEFAULT_TRADER_NAME}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_db()
    result = start_trader_agent(
        trader_name=args.trader_name,
    )
    print(result)


if __name__ == "__main__":
    main()
