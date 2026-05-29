import json

from ingestion.db import get_connection, utc_now
from aggregation.schemas import SentimentEventInput, TickerSentimentSignal


def get_sentiment_events_for_aggregation(
    sector: str,
    limit_per_ticker: int = 20,
) -> list[SentimentEventInput]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            ns.news_event_id,
            ns.sector,
            ns.ticker,
            ns.company_name,
            ne.title,
            ne.event_type,
            ne.affected_direction,
            ne.summary AS event_summary,
            ns.sentiment_score,
            ns.sentiment_label,
            ns.magnitude,
            ns.confidence,
            ns.time_horizon,
            ns.main_driver,
            ns.risk_flags
        FROM news_sentiment ns
        JOIN news_events ne
            ON ns.news_event_id = ne.id
        WHERE lower(ns.sector) = lower(?)
        ORDER BY ns.ticker ASC, ns.id DESC
        """,
        (sector,),
    )
    rows = cursor.fetchall()
    conn.close()

    per_ticker_count: dict[str, int] = {}
    events: list[SentimentEventInput] = []

    for row in rows:
        ticker = row["ticker"]
        if per_ticker_count.get(ticker, 0) >= limit_per_ticker:
            continue

        risk_flags_raw = row["risk_flags"]
        try:
            risk_flags = json.loads(risk_flags_raw) if risk_flags_raw else []
        except json.JSONDecodeError:
            risk_flags = []

        events.append(
            SentimentEventInput(
                news_event_id=row["news_event_id"],
                sector=row["sector"],
                ticker=row["ticker"],
                company_name=row["company_name"],
                title=row["title"],
                event_type=row["event_type"],
                affected_direction=row["affected_direction"],
                event_summary=row["event_summary"],
                sentiment_score=row["sentiment_score"],
                sentiment_label=row["sentiment_label"],
                magnitude=row["magnitude"],
                confidence=row["confidence"],
                time_horizon=row["time_horizon"],
                main_driver=row["main_driver"],
                risk_flags=risk_flags,
            )
        )
        per_ticker_count[ticker] = per_ticker_count.get(ticker, 0) + 1

    return events


def save_ticker_sentiment_signal(signal: TickerSentimentSignal) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO ticker_sentiment_signals (
            sector,
            ticker,
            company_name,
            signal_score,
            signal_label,
            confidence,
            event_count,
            positive_count,
            negative_count,
            neutral_count,
            mixed_count,
            unknown_count,
            strongest_positive_event_id,
            strongest_negative_event_id,
            summary,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            signal.sector,
            signal.ticker,
            signal.company_name,
            signal.signal_score,
            signal.signal_label,
            signal.confidence,
            signal.event_count,
            signal.positive_count,
            signal.negative_count,
            signal.neutral_count,
            signal.mixed_count,
            signal.unknown_count,
            signal.strongest_positive_event_id,
            signal.strongest_negative_event_id,
            signal.summary,
            utc_now(),
        ),
    )
    signal_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return signal_id
