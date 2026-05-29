from ingestion.db import get_connection


def show_table(table_name: str, limit: int = 20) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name} LIMIT ?", (limit,))
    rows = cursor.fetchall()
    print(f"\n--- {table_name} ---")
    for row in rows:
        print(dict(row))
    conn.close()


if __name__ == "__main__":
    show_table("ticker_universe")
    show_table("raw_news")
    show_table("news_events")
    show_table("news_relevance")
    show_table("news_sentiment")
    show_table("ticker_sentiment_signals")
    show_table("price_bars")
    show_table("technical_signals")
    show_table("combined_alpha_signals")
    show_table("risk_decisions")
    show_table("portfolio_positions")
    show_table("trade_plans")
    show_table("trader_profiles")
    show_table("portfolio_state")
    show_table("portfolio_holdings")
    show_table("portfolio_valuation_log")
    show_table("trader_run_log")
    show_table("aggregate_portfolio_log")
    show_table("trade_executions")
    show_table("strategy_knowledge")
    show_table("chart_confirmation_signals")
