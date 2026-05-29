import json

from chart_confirmation.schemas import ChartConfirmationSignal, PriceBarForChart
from ingestion.db import get_connection, utc_now


def get_tickers_for_chart_confirmation(sector: str) -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            ticker,
            company_name,
            sector
        FROM ticker_universe
        WHERE lower(sector) = lower(?)
        ORDER BY rank ASC
        """,
        (sector,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_price_bars_for_chart_confirmation(
    ticker: str,
    limit: int = 120,
    timeframe: str = "1d",
) -> list[PriceBarForChart]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            sector,
            ticker,
            timestamp,
            open,
            high,
            low,
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
    rows = list(reversed(rows))
    return [
        PriceBarForChart(
            sector=row["sector"],
            ticker=row["ticker"],
            timestamp=row["timestamp"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            adjusted_close=row["adjusted_close"],
            volume=row["volume"],
        )
        for row in rows
    ]


def save_chart_confirmation_signal(
    signal: ChartConfirmationSignal,
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO chart_confirmation_signals (
            sector,
            ticker,
            company_name,
            signal_date,
            current_price,
            open_price,
            high_price,
            low_price,
            close_price,
            latest_volume,
            trend_status,
            trend_reading,
            base_status,
            support_level,
            resistance_level,
            breakout_status,
            breakout_price,
            volume_confirmation,
            volume_ratio,
            entry_quality,
            extension_pct,
            buy_trigger,
            invalid_buy_reason,
            reason_to_wait,
            current_price_stop_7_pct,
            current_price_stop_8_pct,
            breakout_entry_stop_7_pct,
            breakout_entry_stop_8_pct,
            danger_level,
            sell_signal,
            chart_decision,
            chart_score,
            chart_confidence,
            llm_chart_reason,
            chart_flags,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            signal.sector,
            signal.ticker,
            signal.company_name,
            signal.signal_date,
            signal.current_price,
            signal.open_price,
            signal.high_price,
            signal.low_price,
            signal.close_price,
            signal.latest_volume,
            signal.trend_status,
            signal.trend_reading,
            signal.base_status,
            signal.support_level,
            signal.resistance_level,
            signal.breakout_status,
            signal.breakout_price,
            signal.volume_confirmation,
            signal.volume_ratio,
            signal.entry_quality,
            signal.extension_pct,
            signal.buy_trigger,
            signal.invalid_buy_reason,
            signal.reason_to_wait,
            signal.current_price_stop_7_pct,
            signal.current_price_stop_8_pct,
            signal.breakout_entry_stop_7_pct,
            signal.breakout_entry_stop_8_pct,
            signal.danger_level,
            signal.sell_signal,
            signal.chart_decision,
            signal.chart_score,
            signal.chart_confidence,
            signal.llm_chart_reason,
            json.dumps(signal.chart_flags),
            utc_now(),
        ),
    )
    signal_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return signal_id
