import time
from datetime import datetime, timezone

from aggregation.aggregation_agent import run_signal_aggregation_agent
from alpha.alpha_agent import run_combined_alpha_signal_agent
from chart_confirmation.chart_confirmation_agent import run_chart_confirmation_agent
from execution.execution_agent import run_simulated_execution_agent
from ingestion.ingestion_agent import run_data_ingestion
from mark_to_market.valuation_agent import run_mark_to_market_agent
from market_data.market_data_agent import run_market_data_ingestion_agent
from portfolio.portfolio_agent import run_portfolio_construction_agent
from relevance.relevance_agent import run_news_relevance_agent
from risk.risk_agent import run_risk_agent
from sentiment.sentiment_agent import run_news_sentiment_agent
from technical.technical_agent import run_technical_signal_agent
from trade_plan.trade_plan_agent import run_trade_plan_agent
from strategy_knowledge.strategy_knowledge_agent import ingest_strategy_knowledge
from trader.repositories import (
    build_aggregate_portfolio_snapshot,
    get_portfolio_state,
    get_trader_profile,
    save_aggregate_portfolio_snapshot,
    save_trader_log,
)
from trader.schemas import TraderRunLog


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_sleep_seconds_for_profile(profile_type: str) -> int:
    profile_type = profile_type.lower().strip()
    if profile_type == "aggressive":
        return 60 * 15
    if profile_type == "medium":
        return 60 * 60
    if profile_type == "conservative":
        return 60 * 60 * 4
    return 60 * 60


def get_company_limit_for_profile(profile_type: str) -> int:
    if profile_type == "aggressive":
        return 40
    if profile_type == "medium":
        return 30
    if profile_type == "conservative":
        return 20
    return 30


def get_articles_per_company_for_profile(profile_type: str) -> int:
    if profile_type == "aggressive":
        return 3
    if profile_type == "medium":
        return 2
    if profile_type == "conservative":
        return 1
    return 2


def get_max_position_size_for_profile(profile_type: str) -> float:
    if profile_type == "aggressive":
        return 0.18
    if profile_type == "medium":
        return 0.10
    if profile_type == "conservative":
        return 0.05
    return 0.10


def run_one_trading_cycle(
    trader_name: str,
    sector: str,
    max_companies_per_sector: int | None = None,
    price_timeframe: str = "1d",
    force_refresh: bool = False,
) -> dict:
    profile = get_trader_profile(trader_name)
    if profile is None:
        return {
            "status": "error",
            "message": f"Trader '{trader_name}' does not exist.",
        }
    if profile.status != "running":
        return {
            "status": "stopped",
            "message": f"Trader '{trader_name}' is not running.",
            "trader_status": profile.status,
        }

    company_limit = get_company_limit_for_profile(profile.profile_type)
    if max_companies_per_sector is not None:
        company_limit = max(1, int(max_companies_per_sector))
    max_articles_per_company = get_articles_per_company_for_profile(profile.profile_type)
    max_position_size = get_max_position_size_for_profile(profile.profile_type)
    market_data_period = "6mo" if price_timeframe == "1d" else "1mo"

    cycle_started_at = utc_now()
    save_trader_log(
        TraderRunLog(
            trader_name=trader_name,
            event_type="cycle_started",
            message="Trading cycle started.",
            metadata={
                "sector": sector,
                "profile_type": profile.profile_type,
                "company_limit": company_limit,
                "max_articles_per_company": max_articles_per_company,
                "max_position_size": max_position_size,
                "price_timeframe": price_timeframe,
                "cycle_started_at": cycle_started_at,
            },
        )
    )

    ingestion_result = run_data_ingestion(
        sector=sector,
        company_limit=company_limit,
        max_articles_per_company=max_articles_per_company,
    )
    relevance_result = run_news_relevance_agent(
        limit=200,
    )
    sentiment_result = run_news_sentiment_agent(
        limit=200,
    )
    aggregation_result = run_signal_aggregation_agent(
        sector=sector,
        limit_per_ticker=20,
        use_llm_summary=True,
        force_refresh=force_refresh,
    )
    strategy_knowledge_result = ingest_strategy_knowledge(
        strategy_name="oneil_growth_leadership",
        source_name="oneil_growth_leadership_notes",
        source_type="embedded_notes",
        raw_text="Oneil growth leadership strategy knowledge.",
    )
    market_data_result = run_market_data_ingestion_agent(
        sector=sector,
        company_limit=company_limit,
        period=market_data_period,
        timeframe=price_timeframe,
        force_refresh=force_refresh,
    )
    technical_result = run_technical_signal_agent(
        sector=sector,
        price_history_limit=120,
        timeframe=price_timeframe,
    )
    chart_confirmation_result = run_chart_confirmation_agent(
        sector=sector,
        price_history_limit=120,
        timeframe=price_timeframe,
    )
    alpha_result = run_combined_alpha_signal_agent(
        sector=sector,
        force_refresh=force_refresh,
    )
    risk_result = run_risk_agent(
        sector=sector,
    )
    portfolio_result = run_portfolio_construction_agent(
        sector=sector,
        max_position_size=max_position_size,
    )
    trade_plan_result = run_trade_plan_agent(
        sector=sector,
    )
    execution_result = run_simulated_execution_agent(
        trader_name=trader_name,
        sector=sector,
    )
    valuation_result = run_mark_to_market_agent(
        trader_name=trader_name,
    )

    portfolio_state = get_portfolio_state(trader_name)
    aggregate_snapshot = build_aggregate_portfolio_snapshot()
    save_aggregate_portfolio_snapshot(aggregate_snapshot)

    cycle_result = {
        "trader_name": trader_name,
        "sector": sector,
        "profile_type": profile.profile_type,
        "price_timeframe": price_timeframe,
        "cycle_started_at": cycle_started_at,
        "cycle_finished_at": utc_now(),
        "ingestion": ingestion_result,
        "relevance": relevance_result,
        "sentiment": sentiment_result,
        "aggregation": aggregation_result,
        "strategy_knowledge": strategy_knowledge_result,
        "market_data": market_data_result,
        "technical": technical_result,
        "chart_confirmation": chart_confirmation_result,
        "alpha": alpha_result,
        "risk": risk_result,
        "portfolio": portfolio_result,
        "trade_plan": trade_plan_result,
        "execution": execution_result,
        "valuation": valuation_result,
        "portfolio_state": portfolio_state.model_dump() if portfolio_state else None,
    }

    save_trader_log(
        TraderRunLog(
            trader_name=trader_name,
            event_type="cycle_finished",
            message="Trading cycle finished.",
            metadata=cycle_result,
        )
    )

    return cycle_result


def run_continuous_trading_loop(
    trader_name: str,
    sector: str,
    max_cycles: int | None = None,
    sleep_seconds: int | None = None,
    max_companies_per_sector: int | None = None,
    price_timeframe: str = "1d",
    force_refresh: bool = False,
) -> dict:
    profile = get_trader_profile(trader_name)
    if profile is None:
        return {
            "status": "error",
            "message": f"Trader '{trader_name}' does not exist.",
        }
    if profile.status != "running":
        return {
            "status": "not_running",
            "trader_name": trader_name,
            "trader_status": profile.status,
        }

    if sleep_seconds is None:
        sleep_seconds = get_sleep_seconds_for_profile(profile.profile_type)

    save_trader_log(
        TraderRunLog(
            trader_name=trader_name,
            event_type="loop_started",
            message="Continuous trading loop started.",
            metadata={
                "sector": sector,
                "profile_type": profile.profile_type,
                "sleep_seconds": sleep_seconds,
                "max_cycles": max_cycles,
                "max_companies_per_sector": max_companies_per_sector,
                "price_timeframe": price_timeframe,
            },
        )
    )

    cycles_completed = 0
    last_cycle_result = None

    while True:
        profile = get_trader_profile(trader_name)
        if profile is None:
            break

        if profile.status != "running":
            save_trader_log(
                TraderRunLog(
                    trader_name=trader_name,
                    event_type="loop_stopped",
                    message="Continuous trading loop stopped because trader is not running.",
                    metadata={
                        "status": profile.status,
                        "cycles_completed": cycles_completed,
                    },
                )
            )
            return {
                "status": "stopped",
                "trader_name": trader_name,
                "cycles_completed": cycles_completed,
                "last_cycle_result": last_cycle_result,
            }

        last_cycle_result = run_one_trading_cycle(
            trader_name=trader_name,
            sector=sector,
            max_companies_per_sector=max_companies_per_sector,
            price_timeframe=price_timeframe,
            force_refresh=force_refresh,
        )
        cycles_completed += 1

        if max_cycles is not None and cycles_completed >= max_cycles:
            save_trader_log(
                TraderRunLog(
                    trader_name=trader_name,
                    event_type="loop_finished",
                    message="Continuous trading loop finished after max_cycles.",
                    metadata={
                        "cycles_completed": cycles_completed,
                        "max_cycles": max_cycles,
                    },
                )
            )
            return {
                "status": "completed",
                "trader_name": trader_name,
                "cycles_completed": cycles_completed,
                "last_cycle_result": last_cycle_result,
            }

        time.sleep(sleep_seconds)
