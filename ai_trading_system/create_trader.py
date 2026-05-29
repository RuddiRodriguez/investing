from ingestion.db import init_db
from trader.trader_agent import initialize_trader_agent


def main() -> None:
    init_db()
    result = initialize_trader_agent(
        trader_name="semiconductor_aggressive_001",
        profile_type="aggressive",
        initial_cash=10000,
    )
    print(result)


if __name__ == "__main__":
    main()
