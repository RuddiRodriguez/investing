from datetime import datetime, timezone

from ingestion.cache import build_cache_key, cache_exists, set_cache
from ingestion.llm_search import (
    discover_companies_by_sector,
    search_latest_news_for_universe,
    structure_news_event,
)
from ingestion.repositories import (
    save_company_universe,
    get_companies_by_sector,
    save_raw_news,
    save_news_event,
)


def chunk_list(items: list, chunk_size: int) -> list[list]:
    return [
        items[index:index + chunk_size]
        for index in range(0, len(items), chunk_size)
    ]


def run_data_ingestion(
    sector: str,
    company_limit: int = 40,
    max_articles_per_company: int = 3,
) -> dict:
    universe_cache_key = build_cache_key(
        "universe",
        sector,
        company_limit,
    )

    if cache_exists(universe_cache_key):
        companies_saved = 0
    else:
        universe = discover_companies_by_sector(
            sector=sector,
            limit=company_limit,
        )
        companies_saved = save_company_universe(universe)
        set_cache(
            cache_key=universe_cache_key,
            cache_type="company_universe",
            value=sector,
            ttl_hours=24 * 7,
        )

    companies = get_companies_by_sector(
        sector=sector,
        limit=company_limit,
    )

    raw_saved = 0
    events_saved = 0
    company_batches = chunk_list(companies, chunk_size=5)

    for batch_index, company_batch in enumerate(company_batches):
        batch_tickers = ",".join(
            company.get("ticker", "")
            for company in company_batch
        )
        news_cache_key = build_cache_key(
            "news",
            sector,
            datetime.now(timezone.utc).date().isoformat(),
            batch_index,
            batch_tickers,
            max_articles_per_company,
        )

        if cache_exists(news_cache_key):
            continue

        news_result = search_latest_news_for_universe(
            sector=sector,
            companies=company_batch,
            max_articles_per_company=max_articles_per_company,
        )

        set_cache(
            cache_key=news_cache_key,
            cache_type="news_search",
            value=sector,
            ttl_hours=12,
        )

        for article in news_result.articles:
            raw_news_id = save_raw_news(
                sector=sector,
                article=article,
            )
            if raw_news_id is None:
                continue
            raw_saved += 1
            event = structure_news_event(
                sector=sector,
                ticker=article.ticker,
                company_name=article.company_name,
                title=article.title,
                raw_summary=article.raw_summary,
            )
            save_news_event(
                raw_news_id=raw_news_id,
                sector=sector,
                article=article,
                event=event,
            )
            events_saved += 1
    return {
        "sector": sector,
        "companies_saved": companies_saved,
        "raw_news_saved": raw_saved,
        "news_events_saved": events_saved,
    }
