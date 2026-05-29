from ingestion.db import get_connection, utc_now
from technical.schemas import PriceBarInput, TechnicalSignal


def get_tickers_with_price_data(sector: str, timeframe: str = "1d") -> list[str]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT DISTINCT ticker
        FROM price_bars
        WHERE lower(sector) = lower(?)
                    AND timeframe = ?
        ORDER BY ticker ASC
        """,
                (sector, timeframe),
    )
    rows = cursor.fetchall()
    conn.close()
    return [row["ticker"] for row in rows]


def get_price_bars_for_ticker(
    ticker: str,
    limit: int = 120,
    timeframe: str = "1d",
) -> list[PriceBarInput]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            sector,
            ticker,
            timestamp,
            close,
            adjusted_close,
            volume
        FROM price_bars
        WHERE ticker = ?
                    AND timeframe = ?
          AND close IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT ?
        """,
                (ticker, timeframe, limit),
    )
    rows = cursor.fetchall()
    conn.close()

    reversed_rows = list(reversed(rows))
    return [
        PriceBarInput(
            sector=row["sector"],
            ticker=row["ticker"],
            timestamp=row["timestamp"],
            close=row["close"],
            adjusted_close=row["adjusted_close"],
            volume=row["volume"],
        )
        for row in reversed_rows
    ]


def get_latest_price_date_for_ticker(ticker: str, timeframe: str = "1d") -> str | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT timestamp
        FROM price_bars
        WHERE ticker = ?
                    AND timeframe = ?
          AND close IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 1
        """,
                (ticker, timeframe),
    )
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return row["timestamp"]


def technical_signal_exists(ticker: str, signal_date: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT 1
        FROM technical_signals
        WHERE ticker = ?
          AND signal_date = ?
        LIMIT 1
        """,
        (ticker, signal_date),
    )
    row = cursor.fetchone()
    conn.close()
    return row is not None


def save_technical_signal(signal: TechnicalSignal) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO technical_signals (
            sector,
            ticker,
            signal_date,
            close,
            return_5d,
            return_20d,
            return_60d,
            volatility_20d,
            sma_20,
            sma_50,
            price_vs_sma_20,
            price_vs_sma_50,
            max_drawdown_60d,
            technical_score,
            technical_label,
            confidence,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            signal.sector,
            signal.ticker,
            signal.signal_date,
            signal.close,
            signal.return_5d,
            signal.return_20d,
            signal.return_60d,
            signal.volatility_20d,
            signal.sma_20,
            signal.sma_50,
            signal.price_vs_sma_20,
            signal.price_vs_sma_50,
            signal.max_drawdown_60d,
            signal.technical_score,
            signal.technical_label,
            signal.confidence,
            utc_now(),
        ),
    )
    signal_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return signal_id
