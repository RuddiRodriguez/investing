from ingestion.db import init_db
from ingestion.ingestion_agent import run_data_ingestion
from relevance.relevance_agent import run_news_relevance_agent
from sentiment.sentiment_agent import run_news_sentiment_agent
from aggregation.aggregation_agent import run_signal_aggregation_agent
from chart_confirmation.chart_confirmation_agent import run_chart_confirmation_agent
from market_data.market_data_agent import run_market_data_ingestion_agent
from technical.technical_agent import run_technical_signal_agent
from alpha.alpha_agent import run_combined_alpha_signal_agent
from risk.risk_agent import run_risk_agent
from portfolio.portfolio_agent import run_portfolio_construction_agent
from trade_plan.trade_plan_agent import run_trade_plan_agent
from execution.execution_agent import run_simulated_execution_agent
from mark_to_market.valuation_agent import run_mark_to_market_agent


# Set to True to bypass cache checks and force fresh API/LLM refresh where supported.
FORCE_REFRESH = False
FAST_MODE = False
PRICE_TIMEFRAME = "1d"


def main() -> None:
    init_db()
    sector = "semiconductors"
    company_limit = 3
    max_articles_per_company = 2
    relevance_limit = 20 if FAST_MODE else 100
    sentiment_limit = 20 if FAST_MODE else 100
    aggregation_limit_per_ticker = 8 if FAST_MODE else 20
    use_llm_summary = True
    market_data_company_limit = company_limit
    technical_price_history_limit = 90 if FAST_MODE else 120
    market_data_period = "6mo" if PRICE_TIMEFRAME == "1d" else "1mo"

    ingestion_result = run_data_ingestion(
        sector=sector,
        company_limit=company_limit,
        max_articles_per_company=max_articles_per_company,
    )
    relevance_result = run_news_relevance_agent(
        limit=relevance_limit,
    )
    sentiment_result = run_news_sentiment_agent(
        limit=sentiment_limit,
    )
    aggregation_result = run_signal_aggregation_agent(
        sector=sector,
        limit_per_ticker=aggregation_limit_per_ticker,
        use_llm_summary=use_llm_summary,
        force_refresh=FORCE_REFRESH,
    )
    market_data_result = run_market_data_ingestion_agent(
        sector=sector,
        company_limit=market_data_company_limit,
        period=market_data_period,
        timeframe=PRICE_TIMEFRAME,
        force_refresh=FORCE_REFRESH,
    )
    technical_result = run_technical_signal_agent(
        sector=sector,
        price_history_limit=technical_price_history_limit,
        timeframe=PRICE_TIMEFRAME,
    )
    chart_confirmation_result = run_chart_confirmation_agent(
        sector=sector,
        price_history_limit=technical_price_history_limit,
        timeframe=PRICE_TIMEFRAME,
    )
    alpha_result = run_combined_alpha_signal_agent(
        sector=sector,
        force_refresh=FORCE_REFRESH,
    )
    risk_result = run_risk_agent(
        sector=sector,
    )
    portfolio_result = run_portfolio_construction_agent(
        sector=sector,
        max_position_size=0.10,
    )
    trade_plan_result = run_trade_plan_agent(
        sector=sector,
    )
    execution_result = run_simulated_execution_agent(
        trader_name="semiconductor_aggressive_001",
        sector=sector,
    )
    valuation_result = run_mark_to_market_agent(
        trader_name="semiconductor_aggressive_001",
    )
    corrections = market_data_result.get("ticker_corrections", [])
    if corrections:
        print("Backup ticker correction agent worked for:")
        for correction in corrections:
            print(
                f"- {correction['company_name']}: "
                f"{correction['original_ticker']} -> {correction['corrected_ticker']} "
                f"({correction['status']})"
            )
    else:
        print("Backup ticker correction agent: no corrections needed or no successful corrections.")

    print({
        "ingestion": ingestion_result,
        "relevance": relevance_result,
        "sentiment": sentiment_result,
        "aggregation": aggregation_result,
        "market_data": market_data_result,
        "technical": technical_result,
        "chart_confirmation": chart_confirmation_result,
        "alpha": alpha_result,
        "risk": risk_result,
        "portfolio": portfolio_result,
        "trade_plan": trade_plan_result,
        "execution": execution_result,
        "valuation": valuation_result,
    })


if __name__ == "__main__":
    main()
