from ingestion.db import get_connection, utc_now
from relevance.schemas import NewsEventInput, NewsRelevanceDecision


def get_unprocessed_news_events(limit: int = 50) -> list[NewsEventInput]:
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
            ne.relevance_score,
            ne.confidence
        FROM news_events ne
        LEFT JOIN news_relevance nr
            ON ne.id = nr.news_event_id
        WHERE nr.id IS NULL
        ORDER BY ne.id ASC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        NewsEventInput(
            id=row["id"],
            sector=row["sector"],
            ticker=row["ticker"],
            company_name=row["company_name"],
            title=row["title"],
            summary=row["summary"],
            event_type=row["event_type"],
            affected_direction=row["affected_direction"],
            relevance_score=row["relevance_score"],
            confidence=row["confidence"],
        )
        for row in rows
    ]


def save_relevance_decision(
    event: NewsEventInput,
    decision: NewsRelevanceDecision,
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO news_relevance (
            news_event_id,
            ticker,
            company_name,
            sector,
            is_relevant,
            relevance_score,
            impact_horizon,
            affected_scope,
            reason,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.id,
            event.ticker,
            event.company_name,
            event.sector,
            1 if decision.is_relevant else 0,
            decision.relevance_score,
            decision.impact_horizon,
            decision.affected_scope,
            decision.reason,
            utc_now(),
        ),
    )
    relevance_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return relevance_id
