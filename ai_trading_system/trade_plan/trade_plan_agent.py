from trade_plan.llm_trade_plan import evaluate_llm_trade_plan
from trade_plan.repositories import get_trade_plan_inputs, save_trade_plan
from trade_plan.schemas import LlmTradePlanDecision, TradePlan, TradePlanInput


def build_trade_plan(
    trade_input: TradePlanInput,
    llm_decision: LlmTradePlanDecision,
) -> TradePlan:
    return TradePlan(
        sector=trade_input.sector,
        ticker=trade_input.ticker,
        company_name=trade_input.company_name,
        signal_date=trade_input.signal_date,
        llm_direction=trade_input.llm_direction,
        llm_portfolio_action=trade_input.llm_portfolio_action,
        llm_position_size=trade_input.llm_position_size,
        buy_probability=trade_input.buy_probability,
        planned_side=llm_decision.planned_side,
        planned_order_type=llm_decision.planned_order_type,
        planned_time_in_force=llm_decision.planned_time_in_force,
        execution_priority=llm_decision.execution_priority,
        max_slippage_pct=llm_decision.max_slippage_pct,
        trade_plan_status=llm_decision.trade_plan_status,
        trade_reason=llm_decision.trade_reason,
        execution_notes=llm_decision.execution_notes,
    )


def run_trade_plan_agent(sector: str) -> dict:
    inputs = get_trade_plan_inputs(sector=sector)

    processed = 0
    planned = 0
    skipped = 0
    buy = 0
    sell_short = 0
    none = 0
    market = 0
    limit_count = 0

    for trade_input in inputs:
        llm_decision = evaluate_llm_trade_plan(trade_input)
        trade_plan = build_trade_plan(
            trade_input=trade_input,
            llm_decision=llm_decision,
        )
        save_trade_plan(trade_plan)
        processed += 1

        if trade_plan.trade_plan_status == "planned":
            planned += 1
        else:
            skipped += 1

        if trade_plan.planned_side == "buy":
            buy += 1
        elif trade_plan.planned_side == "sell_short":
            sell_short += 1
        else:
            none += 1

        if trade_plan.planned_order_type == "market":
            market += 1
        elif trade_plan.planned_order_type == "limit":
            limit_count += 1

    return {
        "sector": sector,
        "processed": processed,
        "planned": planned,
        "skipped": skipped,
        "buy": buy,
        "sell_short": sell_short,
        "none": none,
        "market": market,
        "limit": limit_count,
    }
