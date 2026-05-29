from daily_sentiment_agent import classify_daily_stock_sentiment
from news_types import OrchestratorResult
from retrieve_articles_agent import retrieve_articles


def run_news_stock_sentiment_pipeline(
    date_to_analyze: str,
    sector: str,
    max_articles: int,
) -> OrchestratorResult:
    articles_result = retrieve_articles(
        date_to_analyze=date_to_analyze,
        sector=sector,
        max_articles=max_articles,
    )

    if not articles_result.articles:
        return OrchestratorResult(
            articles_result=articles_result,
            daily_sentiment_result=None,
        )

    daily_sentiment_result = classify_daily_stock_sentiment(
        articles_result=articles_result,
        sector=sector,
    )

    return OrchestratorResult(
        articles_result=articles_result,
        daily_sentiment_result=daily_sentiment_result,
    )
    
def main():
    date_to_analyze = "2026-05-13"
    sector = "defense"
    max_articles = 5

    result = run_news_stock_sentiment_pipeline(
        date_to_analyze=date_to_analyze,
        sector=sector,
        max_articles=max_articles,
    )

    print(result.model_dump_json(indent=2))    

if __name__ == "__main__":
    main()