from market_data.price_fetcher import fetch_price_bars
from market_data.ticker_backup_agent import resolve_ticker_with_backup_agent
from market_data.repositories import (
    get_companies_for_sector,
    price_data_exists,
    save_price_bars,
)


def run_market_data_ingestion_agent(
    sector: str,
    company_limit: int = 40,
    period: str = "6mo",
    timeframe: str = "1d",
    force_refresh: bool = False,
) -> dict:
    companies = get_companies_for_sector(
        sector=sector,
        limit=company_limit,
    )

    total_bars_saved = 0
    tickers_processed = 0
    tickers_skipped = 0
    tickers_corrected = 0
    ticker_corrections = []
    tickers_no_data = []
    tickers_failed = []

    for company in companies:
        ticker = company["ticker"]
        company_name = company["company_name"]
        try:
            if not force_refresh and price_data_exists(ticker=ticker, timeframe=timeframe):
                tickers_skipped += 1
                continue

            price_bars = fetch_price_bars(
                sector=sector,
                ticker=ticker,
                timeframe=timeframe,
                period=period,
            )

            if not price_bars:
                corrected_ticker = resolve_ticker_with_backup_agent(
                    sector=sector,
                    company_name=company_name,
                    bad_ticker=ticker,
                )
                if corrected_ticker is not None:
                    if not force_refresh and price_data_exists(ticker=corrected_ticker, timeframe=timeframe):
                        tickers_skipped += 1
                        tickers_corrected += 1
                        ticker_corrections.append(
                            {
                                "company_name": company_name,
                                "original_ticker": ticker,
                                "corrected_ticker": corrected_ticker,
                                "status": "corrected_and_skipped_cached",
                            }
                        )
                        continue
                    price_bars = fetch_price_bars(
                        sector=sector,
                        ticker=corrected_ticker,
                        timeframe=timeframe,
                        period=period,
                    )
                    if price_bars:
                        tickers_corrected += 1
                        ticker_corrections.append(
                            {
                                "company_name": company_name,
                                "original_ticker": ticker,
                                "corrected_ticker": corrected_ticker,
                                "status": "corrected_and_fetched",
                            }
                        )

            if not price_bars:
                tickers_no_data.append(
                    {
                        "ticker": ticker,
                        "company_name": company_name,
                        "reason": "no_price_data",
                    }
                )
                continue

            saved = save_price_bars(price_bars)
            total_bars_saved += saved
            tickers_processed += 1
        except Exception as error:
            tickers_failed.append(
                {
                    "ticker": ticker,
                    "company_name": company_name,
                    "error": str(error),
                }
            )

    return {
        "sector": sector,
        "tickers_found": len(companies),
        "tickers_processed": tickers_processed,
        "tickers_skipped": tickers_skipped,
        "tickers_corrected": tickers_corrected,
        "timeframe": timeframe,
        "ticker_corrections": ticker_corrections,
        "tickers_no_data": tickers_no_data,
        "price_bars_saved": total_bars_saved,
        "tickers_failed": tickers_failed,
    }
