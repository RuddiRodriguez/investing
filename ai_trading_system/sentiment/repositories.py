import json

from ingestion.db import get_connection, utc_now
from sentiment.schemas import (
    RelevantNewsEventInput,
    NewsSentimentDecision,
)


def get_relevant_unprocessed_news_events(limit: int = 50) -> list[RelevantNewsEventInput]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            ne.id,
            ne.sector,
            ne.ticker,
            ne.company_name,
            ne.title,
            ne.summary,
            ne.event_type,
            ne.affected_direction,
            ne.relevance_score AS event_relevance_score,
            ne.confidence AS event_confidence,
            nr.relevance_score AS relevance_agent_score,
            nr.impact_horizon,
            nr.affected_scope,
            nr.reason AS relevance_reason
        FROM news_events ne
        JOIN news_relevance nr
            ON ne.id = nr.news_event_id
        LEFT JOIN news_sentiment ns
            ON ne.id = ns.news_event_id
        WHERE nr.is_relevant = 1
          AND ns.id IS NULL
        ORDER BY ne.id ASC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        RelevantNewsEventInput(
            id=row["id"],
            sector=row["sector"],
            ticker=row["ticker"],
            company_name=row["company_name"],
            title=row["title"],
            summary=row["summary"],
            event_type=row["event_type"],
            affected_direction=row["affected_direction"],
            event_relevance_score=row["event_relevance_score"],
            event_confidence=row["event_confidence"],
            relevance_agent_score=row["relevance_agent_score"],
            impact_horizon=row["impact_horizon"],
            affected_scope=row["affected_scope"],
            relevance_reason=row["relevance_reason"],
        )
        for row in rows
    ]


def save_sentiment_decision(
    event: RelevantNewsEventInput,
    decision: NewsSentimentDecision,
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO news_sentiment (
            news_event_id,
            ticker,
            company_name,
            sector,
            sentiment_score,
            sentiment_label,
            magnitude,
            confidence,
            time_horizon,
            main_driver,
            risk_flags,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.id,
            event.ticker,
            event.company_name,
            event.sector,
            decision.sentiment_score,
            decision.sentiment_label,
            decision.magnitude,
            decision.confidence,
            decision.time_horizon,
            decision.main_driver,
            json.dumps(decision.risk_flags),
            utc_now(),
        ),
    )
    sentiment_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return sentiment_id
