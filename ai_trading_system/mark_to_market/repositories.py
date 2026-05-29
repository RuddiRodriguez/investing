from ingestion.db import get_connection, utc_now
from mark_to_market.schemas import (
    HoldingForValuation,
    LatestPrice,
    PortfolioValuationSnapshot,
    UpdatedHoldingValuation,
)


def get_open_holdings(trader_name: str) -> list[HoldingForValuation]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            trader_name,
            ticker,
            company_name,
            direction,
            quantity,
            average_entry_price,
            current_price,
            market_value,
            unrealized_pnl,
            unrealized_pnl_pct,
            position_size
        FROM portfolio_holdings
        WHERE trader_name = ?
          AND quantity > 0
        ORDER BY ticker ASC
        """,
        (trader_name,),
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        HoldingForValuation(
            trader_name=row["trader_name"],
            ticker=row["ticker"],
            company_name=row["company_name"],
            direction=row["direction"],
            quantity=row["quantity"],
            average_entry_price=row["average_entry_price"],
            current_price=row["current_price"],
            market_value=row["market_value"],
            unrealized_pnl=row["unrealized_pnl"],
            unrealized_pnl_pct=row["unrealized_pnl_pct"],
            position_size=row["position_size"],
        )
        for row in rows
    ]


def get_latest_price(ticker: str) -> LatestPrice | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            ticker,
            close,
            timestamp
        FROM price_bars
        WHERE ticker = ?
          AND close IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        (ticker,),
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return LatestPrice(
        ticker=row["ticker"],
        price=row["close"],
        timestamp=row["timestamp"],
    )


def update_holding_valuation(
    updated_holding: UpdatedHoldingValuation,
) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE portfolio_holdings
        SET current_price = ?,
            market_value = ?,
            unrealized_pnl = ?,
            unrealized_pnl_pct = ?,
            position_size = ?,
            updated_at = ?
        WHERE trader_name = ?
          AND ticker = ?
          AND direction = ?
        """,
        (
            updated_holding.current_price,
            updated_holding.market_value,
            updated_holding.unrealized_pnl,
            updated_holding.unrealized_pnl_pct,
            updated_holding.position_size,
            utc_now(),
            updated_holding.trader_name,
            updated_holding.ticker,
            updated_holding.direction,
        ),
    )
    conn.commit()
    conn.close()


def get_cash(trader_name: str) -> float:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT cash
        FROM portfolio_state
        WHERE trader_name = ?
        """,
        (trader_name,),
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return 0.0

    return float(row["cash"])


def update_portfolio_state_from_valuation(
    snapshot: PortfolioValuationSnapshot,
) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE portfolio_state
        SET cash = ?,
            invested_value = ?,
            total_portfolio_value = ?,
            open_positions_count = ?,
            last_updated_at = ?
        WHERE trader_name = ?
        """,
        (
            snapshot.cash,
            snapshot.invested_value,
            snapshot.total_portfolio_value,
            snapshot.open_positions_count,
            utc_now(),
            snapshot.trader_name,
        ),
    )
    cursor.execute(
        """
        UPDATE trader_profiles
        SET current_cash = ?,
            updated_at = ?
        WHERE trader_name = ?
        """,
        (
            snapshot.cash,
            utc_now(),
            snapshot.trader_name,
        ),
    )
    conn.commit()
    conn.close()


def save_portfolio_valuation_log(
    snapshot: PortfolioValuationSnapshot,
) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO portfolio_valuation_log (
            trader_name,
            cash,
            invested_value,
            total_portfolio_value,
            unrealized_pnl,
            unrealized_pnl_pct,
            open_positions_count,
            valuation_reason,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            snapshot.trader_name,
            snapshot.cash,
            snapshot.invested_value,
            snapshot.total_portfolio_value,
            snapshot.unrealized_pnl,
            snapshot.unrealized_pnl_pct,
            snapshot.open_positions_count,
            snapshot.valuation_reason,
            utc_now(),
        ),
    )
    conn.commit()
    conn.close()
