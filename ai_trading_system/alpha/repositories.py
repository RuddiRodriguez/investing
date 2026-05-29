from ingestion.db import get_connection, utc_now
from alpha.schemas import AlphaInput, CombinedAlphaSignal


def get_alpha_inputs(sector: str) -> list[AlphaInput]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        WITH latest_sentiment AS (
            SELECT tss1.*
            FROM ticker_sentiment_signals tss1
            JOIN (
                SELECT ticker, lower(sector) AS sector_lower, MAX(created_at) AS max_created_at
                FROM ticker_sentiment_signals
                GROUP BY ticker, lower(sector)
            ) latest
              ON tss1.ticker = latest.ticker
             AND lower(tss1.sector) = latest.sector_lower
             AND tss1.created_at = latest.max_created_at
        )
        SELECT
            tu.sector,
            tu.ticker,
            tu.company_name,
            ts.signal_date,
            ls.signal_score AS sentiment_score,
            ls.signal_label AS sentiment_label,
            ls.confidence AS sentiment_confidence,
            ts.technical_score,
            ts.technical_label,
            ts.confidence AS technical_confidence,
            ccs.chart_decision,
            ccs.chart_score,
            ccs.chart_confidence,
            ccs.trend_reading,
            ccs.breakout_status,
            ccs.volume_confirmation,
            ccs.entry_quality,
            ccs.support_level,
            ccs.resistance_level,
            ccs.buy_trigger,
            ccs.invalid_buy_reason,
            ccs.reason_to_wait,
            ccs.current_price_stop_7_pct,
            ccs.current_price_stop_8_pct,
            ccs.breakout_entry_stop_7_pct,
            ccs.breakout_entry_stop_8_pct,
            ccs.danger_level
        FROM ticker_universe tu
        LEFT JOIN latest_sentiment ls
            ON tu.ticker = ls.ticker
           AND lower(tu.sector) = lower(ls.sector)
        JOIN technical_signals ts
            ON tu.ticker = ts.ticker
           AND lower(tu.sector) = lower(ts.sector)
        LEFT JOIN chart_confirmation_signals ccs
            ON tu.ticker = ccs.ticker
           AND lower(tu.sector) = lower(ccs.sector)
           AND ccs.signal_date = ts.signal_date
        WHERE lower(tu.sector) = lower(?)
        AND ts.signal_date = (
            SELECT MAX(ts2.signal_date)
            FROM technical_signals ts2
            WHERE ts2.ticker = tu.ticker
              AND lower(ts2.sector) = lower(tu.sector)
        )
        ORDER BY tu.rank ASC
        """,
        (sector,),
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        AlphaInput(
            sector=row["sector"],
            ticker=row["ticker"],
            company_name=row["company_name"],
            signal_date=row["signal_date"],
            sentiment_score=row["sentiment_score"],
            sentiment_label=row["sentiment_label"],
            sentiment_confidence=row["sentiment_confidence"],
            technical_score=row["technical_score"],
            technical_label=row["technical_label"],
            technical_confidence=row["technical_confidence"],
            chart_decision=row["chart_decision"],
            chart_score=row["chart_score"],
            chart_confidence=row["chart_confidence"],
            trend_reading=row["trend_reading"],
            breakout_status=row["breakout_status"],
            volume_confirmation=row["volume_confirmation"],
            entry_quality=row["entry_quality"],
            support_level=row["support_level"],
            resistance_level=row["resistance_level"],
            buy_trigger=row["buy_trigger"],
            invalid_buy_reason=row["invalid_buy_reason"],
            reason_to_wait=row["reason_to_wait"],
            current_price_stop_7_pct=row["current_price_stop_7_pct"],
            current_price_stop_8_pct=row["current_price_stop_8_pct"],
            breakout_entry_stop_7_pct=row["breakout_entry_stop_7_pct"],
            breakout_entry_stop_8_pct=row["breakout_entry_stop_8_pct"],
            stop_loss_7_pct=row["current_price_stop_7_pct"],
            stop_loss_8_pct=row["current_price_stop_8_pct"],
            danger_level=row["danger_level"],
        )
        for row in rows
    ]


def get_combined_alpha_signal(ticker: str, signal_date: str) -> dict | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            sector,
            ticker,
            company_name,
            signal_date,
            sentiment_score,
            sentiment_label,
            sentiment_confidence,
            technical_score,
            technical_label,
            technical_confidence,
            chart_decision,
            chart_score,
            chart_confidence,
            trend_reading,
            breakout_status,
            volume_confirmation,
            entry_quality,
            support_level,
            resistance_level,
            buy_trigger,
            invalid_buy_reason,
            reason_to_wait,
            current_price_stop_7_pct,
            current_price_stop_8_pct,
            breakout_entry_stop_7_pct,
            breakout_entry_stop_8_pct,
            stop_loss_7_pct,
            stop_loss_8_pct,
            danger_level,
            alpha_score,
            alpha_label,
            confidence,
            main_driver
        FROM combined_alpha_signals
        WHERE ticker = ?
          AND signal_date = ?
        LIMIT 1
        """,
        (ticker, signal_date),
    )
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def save_combined_alpha_signal(signal: CombinedAlphaSignal) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO combined_alpha_signals (
            sector,
            ticker,
            company_name,
            signal_date,
            sentiment_score,
            sentiment_label,
            sentiment_confidence,
            technical_score,
            technical_label,
            technical_confidence,
            chart_decision,
            chart_score,
            chart_confidence,
            trend_reading,
            breakout_status,
            volume_confirmation,
            entry_quality,
            support_level,
            resistance_level,
            buy_trigger,
            invalid_buy_reason,
            reason_to_wait,
            current_price_stop_7_pct,
            current_price_stop_8_pct,
            breakout_entry_stop_7_pct,
            breakout_entry_stop_8_pct,
            stop_loss_7_pct,
            stop_loss_8_pct,
            danger_level,
            alpha_score,
            alpha_label,
            confidence,
            main_driver,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            signal.sector,
            signal.ticker,
            signal.company_name,
            signal.signal_date,
            signal.sentiment_score,
            signal.sentiment_label,
            signal.sentiment_confidence,
            signal.technical_score,
            signal.technical_label,
            signal.technical_confidence,
            signal.chart_decision,
            signal.chart_score,
            signal.chart_confidence,
            signal.trend_reading,
            signal.breakout_status,
            signal.volume_confirmation,
            signal.entry_quality,
            signal.support_level,
            signal.resistance_level,
            signal.buy_trigger,
            signal.invalid_buy_reason,
            signal.reason_to_wait,
            signal.current_price_stop_7_pct,
            signal.current_price_stop_8_pct,
            signal.breakout_entry_stop_7_pct,
            signal.breakout_entry_stop_8_pct,
            signal.stop_loss_7_pct,
            signal.stop_loss_8_pct,
            signal.danger_level,
            signal.alpha_score,
            signal.alpha_label,
            signal.confidence,
            signal.main_driver,
            utc_now(),
        ),
    )
    signal_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return signal_id
