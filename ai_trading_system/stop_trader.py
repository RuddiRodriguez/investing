import argparse

from ingestion.db import init_db
from trader.trader_agent import stop_trader_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Stop a running trader agent.")
    parser.add_argument("--trader-name", default="semiconductor_aggressive_001", help="Trader name to stop")
    args = parser.parse_args()

    init_db()
    result = stop_trader_agent(
        trader_name=args.trader_name,
    )
    print(result)


if __name__ == "__main__":
    main()
