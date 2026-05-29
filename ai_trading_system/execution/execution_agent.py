from execution.execution_context import build_execution_context
from execution.llm_execution import evaluate_llm_execution
from execution.repositories import (
    get_latest_price,
    get_profile_for_execution,
    get_state_for_execution,
    get_trade_plans_for_execution,
    save_trade_execution,
    update_portfolio_state_after_execution,
    upsert_holding_after_execution,
)
from execution.schemas import LlmExecutionDecision, TradeExecution
from trader.repositories import (
    build_aggregate_portfolio_snapshot,
    save_aggregate_portfolio_snapshot,
    save_trader_log,
)
from trader.schemas import TraderRunLog


def get_simulated_slippage(profile_type: str, max_slippage_pct: float) -> float:
    if profile_type == "aggressive":
        return min(max_slippage_pct, 0.003)
    if profile_type == "medium":
        return min(max_slippage_pct, 0.002)
    return min(max_slippage_pct, 0.001)


def get_commission(gross_value: float) -> float:
    return max(1.0, gross_value * 0.0005)


def build_non_filled_execution(
    trader_name: str,
    trade_input,
    llm_decision,
    final_status: str,
) -> TradeExecution:
    return TradeExecution(
        trader_name=trader_name,
        trade_plan_id=trade_input.trade_plan_id,
        sector=trade_input.sector,
        ticker=trade_input.ticker,
        company_name=trade_input.company_name,
        signal_date=trade_input.signal_date,
        side=trade_input.planned_side,
        order_type=trade_input.planned_order_type,
        time_in_force=trade_input.planned_time_in_force,
        requested_position_size=trade_input.llm_position_size,
        execution_price=0.0,
        quantity=0.0,
        gross_value=0.0,
        simulated_slippage_pct=0.0,
        commission=0.0,
        llm_execution_status=llm_decision.llm_execution_status,
        llm_fill_ratio=llm_decision.llm_fill_ratio,
        llm_execution_confidence=llm_decision.llm_execution_confidence,
        llm_execution_reason=llm_decision.llm_execution_reason,
        llm_execution_flags=llm_decision.llm_execution_flags,
        execution_status=final_status,
        execution_reason=llm_decision.llm_execution_reason,
    )


def build_filled_execution(
    trader_name: str,
    trade_input,
    context,
    llm_decision,
) -> TradeExecution:
    fill_value = context.max_executable_value * llm_decision.llm_fill_ratio
    slippage = get_simulated_slippage(
        profile_type=context.profile_type,
        max_slippage_pct=trade_input.max_slippage_pct,
    )

    if trade_input.planned_side == "buy":
        execution_price = context.latest_price * (1 + slippage)
    else:
        execution_price = context.latest_price * (1 - slippage)

    commission_estimate = get_commission(fill_value)
    gross_value = max(0.0, fill_value - commission_estimate)
    quantity = gross_value / execution_price if execution_price > 0 else 0.0
    commission = get_commission(gross_value)

    if quantity <= 0:
        return TradeExecution(
            trader_name=trader_name,
            trade_plan_id=trade_input.trade_plan_id,
            sector=trade_input.sector,
            ticker=trade_input.ticker,
            company_name=trade_input.company_name,
            signal_date=trade_input.signal_date,
            side=trade_input.planned_side,
            order_type=trade_input.planned_order_type,
            time_in_force=trade_input.planned_time_in_force,
            requested_position_size=trade_input.llm_position_size,
            execution_price=0.0,
            quantity=0.0,
            gross_value=0.0,
            simulated_slippage_pct=0.0,
            commission=0.0,
            llm_execution_status=llm_decision.llm_execution_status,
            llm_fill_ratio=llm_decision.llm_fill_ratio,
            llm_execution_confidence=llm_decision.llm_execution_confidence,
            llm_execution_reason=llm_decision.llm_execution_reason,
            llm_execution_flags=llm_decision.llm_execution_flags,
            execution_status="rejected",
            execution_reason="Execution quantity calculated as zero.",
        )

    final_status = "filled" if llm_decision.llm_execution_status == "fill" else "partial_filled"

    return TradeExecution(
        trader_name=trader_name,
        trade_plan_id=trade_input.trade_plan_id,
        sector=trade_input.sector,
        ticker=trade_input.ticker,
        company_name=trade_input.company_name,
        signal_date=trade_input.signal_date,
        side=trade_input.planned_side,
        order_type=trade_input.planned_order_type,
        time_in_force=trade_input.planned_time_in_force,
        requested_position_size=trade_input.llm_position_size,
        execution_price=execution_price,
        quantity=quantity,
        gross_value=gross_value,
        simulated_slippage_pct=slippage,
        commission=commission,
        llm_execution_status=llm_decision.llm_execution_status,
        llm_fill_ratio=llm_decision.llm_fill_ratio,
        llm_execution_confidence=llm_decision.llm_execution_confidence,
        llm_execution_reason=llm_decision.llm_execution_reason,
        llm_execution_flags=llm_decision.llm_execution_flags,
        execution_status=final_status,
        execution_reason=llm_decision.llm_execution_reason,
    )


def run_simulated_execution_agent(
    trader_name: str,
    sector: str,
) -> dict:
    profile = get_profile_for_execution(trader_name)
    state = get_state_for_execution(trader_name)
    if profile is None or state is None:
        return {
            "status": "error",
            "message": "Trader profile or portfolio state not found.",
        }

    trade_inputs = get_trade_plans_for_execution(
        sector=sector,
        trader_name=trader_name,
    )

    processed = 0
    filled = 0
    partial_filled = 0
    rejected = 0
    skipped = 0

    for trade_input in trade_inputs:
        latest_price = get_latest_price(trade_input.ticker)

        if latest_price is None:
            llm_decision = LlmExecutionDecision(
                llm_execution_status="reject",
                llm_fill_ratio=0.0,
                llm_execution_confidence=1.0,
                llm_execution_reason="No latest price available.",
                llm_execution_flags=["missing_price"],
            )
            execution = build_non_filled_execution(
                trader_name=trader_name,
                trade_input=trade_input,
                llm_decision=llm_decision,
                final_status="rejected",
            )
        else:
            state = get_state_for_execution(trader_name)
            context = build_execution_context(
                trader_name=trader_name,
                trade_input=trade_input,
                profile=profile,
                state=state,
                latest_price=latest_price,
            )
            llm_decision = evaluate_llm_execution(context)

            if llm_decision.llm_execution_status == "reject":
                execution = build_non_filled_execution(
                    trader_name=trader_name,
                    trade_input=trade_input,
                    llm_decision=llm_decision,
                    final_status="rejected",
                )
            elif llm_decision.llm_execution_status == "skip":
                execution = build_non_filled_execution(
                    trader_name=trader_name,
                    trade_input=trade_input,
                    llm_decision=llm_decision,
                    final_status="skipped",
                )
            else:
                execution = build_filled_execution(
                    trader_name=trader_name,
                    trade_input=trade_input,
                    context=context,
                    llm_decision=llm_decision,
                )

        save_trade_execution(execution)
        processed += 1

        if execution.execution_status in ["filled", "partial_filled"]:
            direction = "long" if execution.side == "buy" else "short"
            upsert_holding_after_execution(
                trader_name=trader_name,
                ticker=execution.ticker,
                company_name=execution.company_name,
                direction=direction,
                quantity=execution.quantity,
                execution_price=execution.execution_price,
                position_size=execution.requested_position_size,
            )
            cash_change = -(execution.gross_value + execution.commission)
            update_portfolio_state_after_execution(
                trader_name=trader_name,
                cash_change=cash_change,
            )
            save_trader_log(
                TraderRunLog(
                    trader_name=trader_name,
                    event_type="trade_filled",
                    message=f"Simulated execution for {execution.ticker}: {execution.execution_status}.",
                    metadata={
                        "ticker": execution.ticker,
                        "side": execution.side,
                        "quantity": execution.quantity,
                        "execution_price": execution.execution_price,
                        "gross_value": execution.gross_value,
                        "llm_execution_status": execution.llm_execution_status,
                        "llm_fill_ratio": execution.llm_fill_ratio,
                    },
                )
            )
            if execution.execution_status == "filled":
                filled += 1
            else:
                partial_filled += 1
        elif execution.execution_status == "rejected":
            save_trader_log(
                TraderRunLog(
                    trader_name=trader_name,
                    event_type="trade_rejected",
                    message=f"Rejected simulated trade for {execution.ticker}.",
                    metadata={
                        "ticker": execution.ticker,
                        "reason": execution.execution_reason,
                        "llm_flags": execution.llm_execution_flags,
                    },
                )
            )
            rejected += 1
        else:
            skipped += 1

    snapshot = build_aggregate_portfolio_snapshot()
    save_aggregate_portfolio_snapshot(snapshot)

    return {
        "trader_name": trader_name,
        "sector": sector,
        "processed": processed,
        "filled": filled,
        "partial_filled": partial_filled,
        "rejected": rejected,
        "skipped": skipped,
    }
