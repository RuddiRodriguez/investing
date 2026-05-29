from ingestion.db import get_connection, utc_now
from market_data.schemas import PriceBar


def get_companies_for_sector(sector: str, limit: int = 40) -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT ticker, company_name
        FROM ticker_universe
        WHERE lower(sector) = lower(?)
        ORDER BY rank ASC
        LIMIT ?
        """,
        (sector, limit),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_tickers_for_sector(sector: str, limit: int = 40) -> list[str]:
    companies = get_companies_for_sector(
        sector=sector,
        limit=limit,
    )
    return [company["ticker"] for company in companies]


def save_price_bar(price_bar: PriceBar) -> int | None:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT OR REPLACE INTO price_bars (
                sector,
                ticker,
                timestamp,
                timeframe,
                open,
                high,
                low,
                close,
                adjusted_close,
                volume,
                source,
                fetched_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                price_bar.sector,
                price_bar.ticker,
                price_bar.timestamp,
                price_bar.timeframe,
                price_bar.open,
                price_bar.high,
                price_bar.low,
                price_bar.close,
                price_bar.adjusted_close,
                price_bar.volume,
                price_bar.source,
                utc_now(),
            ),
        )
        price_bar_id = cursor.lastrowid
        conn.commit()
        return price_bar_id
    except Exception:
        return None
    finally:
        conn.close()


def save_price_bars(price_bars: list[PriceBar]) -> int:
    saved = 0
    for price_bar in price_bars:
        result = save_price_bar(price_bar)
        if result is not None:
            saved += 1
    return saved


def price_data_exists(
    ticker: str,
    timeframe: str = "1d",
    min_rows: int = 20,
) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(*) AS row_count
        FROM price_bars
        WHERE ticker = ?
          AND timeframe = ?
        """,
        (ticker, timeframe),
    )
    row = cursor.fetchone()
    conn.close()
    return row["row_count"] >= min_rows
