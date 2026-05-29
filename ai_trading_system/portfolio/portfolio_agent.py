from portfolio.llm_portfolio import evaluate_llm_portfolio_decision
from portfolio.position_engine import build_suggested_position
from portfolio.repositories import get_portfolio_inputs, save_portfolio_position
from portfolio.schemas import (
    LlmPortfolioDecision,
    PortfolioInput,
    PortfolioPosition,
    SuggestedPosition,
)


def build_portfolio_position(
    portfolio_input: PortfolioInput,
    suggested_position: SuggestedPosition,
    llm_decision: LlmPortfolioDecision,
) -> PortfolioPosition:
    return PortfolioPosition(
        sector=portfolio_input.sector,
        ticker=portfolio_input.ticker,
        company_name=portfolio_input.company_name,
        signal_date=portfolio_input.signal_date,
        alpha_score=portfolio_input.alpha_score,
        alpha_label=portfolio_input.alpha_label,
        alpha_confidence=portfolio_input.alpha_confidence,
        llm_risk_score=portfolio_input.llm_risk_score,
        llm_risk_label=portfolio_input.llm_risk_label,
        position_size_multiplier=portfolio_input.position_size_multiplier,
        suggested_direction=suggested_position.suggested_direction,
        suggested_position_size=suggested_position.suggested_position_size,
        llm_direction=llm_decision.llm_direction,
        llm_portfolio_action=llm_decision.llm_portfolio_action,
        llm_position_size=llm_decision.llm_position_size,
        llm_confidence=llm_decision.llm_confidence,
        buy_probability=llm_decision.buy_probability,
        portfolio_reason=llm_decision.portfolio_reason,
        portfolio_flags=llm_decision.portfolio_flags,
    )


def run_portfolio_construction_agent(
    sector: str,
    max_position_size: float = 0.10,
) -> dict:
    inputs = get_portfolio_inputs(sector=sector)

    processed = 0
    opened = 0
    skipped = 0
    watched = 0
    reduced = 0
    long_count = 0
    short_count = 0
    none_count = 0
    total_position_size = 0.0

    for portfolio_input in inputs:
        suggested_position = build_suggested_position(
            portfolio_input=portfolio_input,
            max_position_size=max_position_size,
        )
        llm_decision = evaluate_llm_portfolio_decision(
            portfolio_input=portfolio_input,
            suggested_position=suggested_position,
            max_position_size=max_position_size,
        )
        final_position = build_portfolio_position(
            portfolio_input=portfolio_input,
            suggested_position=suggested_position,
            llm_decision=llm_decision,
        )
        save_portfolio_position(final_position)
        processed += 1
        total_position_size += final_position.llm_position_size

        if final_position.llm_portfolio_action == "open":
            opened += 1
        elif final_position.llm_portfolio_action == "skip":
            skipped += 1
        elif final_position.llm_portfolio_action == "watch":
            watched += 1
        elif final_position.llm_portfolio_action == "reduce":
            reduced += 1

        if final_position.llm_direction == "long":
            long_count += 1
        elif final_position.llm_direction == "short":
            short_count += 1
        else:
            none_count += 1

    return {
        "sector": sector,
        "processed": processed,
        "opened": opened,
        "skipped": skipped,
        "watched": watched,
        "reduced": reduced,
        "long": long_count,
        "short": short_count,
        "none": none_count,
        "total_position_size": total_position_size,
    }
