from ingestion.db import init_db
from trader.trader_agent import get_trader_status, start_trader_agent, stop_trader_agent


def main() -> None:
    init_db()
    trader_name = "semiconductor_aggressive_001"
    print(get_trader_status(trader_name))
    # Uncomment when needed:
    # print(stop_trader_agent(trader_name))
    # print(start_trader_agent(trader_name))


if __name__ == "__main__":
    main()
