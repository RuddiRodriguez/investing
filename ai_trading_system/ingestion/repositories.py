from ingestion.db import get_connection, utc_now
from ingestion.schemas import CompanyUniverse, RawNewsItem, NewsEvent


def save_company_universe(universe: CompanyUniverse) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    saved = 0
    for company in universe.companies:
        cursor.execute(
            """
            INSERT OR REPLACE INTO ticker_universe (
                sector,
                rank,
                ticker,
                company_name,
                exchange,
                country,
                industry,
                relevance_reason,
                source_url,
                discovered_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                universe.sector,
                company.rank,
                company.ticker.upper(),
                company.company_name,
                company.exchange,
                company.country,
                company.industry,
                company.relevance_reason,
                company.source_url,
                utc_now(),
            )
        )
        saved += 1
    conn.commit()
    conn.close()
    return saved


def get_companies_by_sector(sector: str, limit: int = 40) -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            sector,
            rank,
            ticker,
            company_name,
            exchange,
            country,
            industry,
            relevance_reason,
            source_url
        FROM ticker_universe
        WHERE lower(sector) = lower(?)
        ORDER BY rank ASC
        LIMIT ?
        """,
        (sector, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def save_raw_news(sector: str, article: RawNewsItem) -> int | None:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO raw_news (
                sector,
                ticker,
                company_name,
                source,
                title,
                url,
                published_at,
                raw_summary,
                fetched_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sector,
                article.ticker.upper(),
                article.company_name,
                article.source,
                article.title,
                article.url,
                article.published_at,
                article.raw_summary,
                utc_now(),
            )
        )
        raw_news_id = cursor.lastrowid
        conn.commit()
        return raw_news_id
    except Exception:
        return None
    finally:
        conn.close()


def save_news_event(
    raw_news_id: int,
    sector: str,
    article: RawNewsItem,
    event: NewsEvent,
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO news_events (
            raw_news_id,
            sector,
            ticker,
            company_name,
            title,
            summary,
            event_type,
            affected_direction,
            relevance_score,
            confidence,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            raw_news_id,
            sector,
            article.ticker.upper(),
            article.company_name,
            article.title,
            event.summary,
            event.event_type,
            event.affected_direction,
            event.relevance_score,
            event.confidence,
            utc_now(),
        )
    )
    news_event_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return news_event_id
