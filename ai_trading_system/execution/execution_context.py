from execution.schemas import ExecutionContext, ExecutionInput, LatestPrice
from trader.schemas import PortfolioState, TraderProfile


def build_execution_context(
    trader_name: str,
    trade_input: ExecutionInput,
    profile: TraderProfile,
    state: PortfolioState,
    latest_price: LatestPrice,
) -> ExecutionContext:
    portfolio_value = state.total_portfolio_value
    cash_reserve_value = portfolio_value * profile.min_cash_reserve
    available_cash_after_reserve = max(0.0, state.cash - cash_reserve_value)

    max_exposure_value = portfolio_value * profile.max_portfolio_exposure
    available_exposure_value = max(0.0, max_exposure_value - state.invested_value)

    requested_value = portfolio_value * trade_input.llm_position_size
    max_executable_value = min(
        requested_value,
        available_cash_after_reserve,
        available_exposure_value,
    )

    return ExecutionContext(
        trader_name=trader_name,
        profile_type=profile.profile_type,
        trader_status=profile.status,
        cash=state.cash,
        invested_value=state.invested_value,
        total_portfolio_value=state.total_portfolio_value,
        open_positions_count=state.open_positions_count,
        max_position_size=profile.max_position_size,
        max_portfolio_exposure=profile.max_portfolio_exposure,
        min_cash_reserve=profile.min_cash_reserve,
        risk_tolerance=profile.risk_tolerance,
        trade_frequency=profile.trade_frequency,
        ticker=trade_input.ticker,
        company_name=trade_input.company_name,
        planned_side=trade_input.planned_side,
        planned_order_type=trade_input.planned_order_type,
        planned_time_in_force=trade_input.planned_time_in_force,
        execution_priority=trade_input.execution_priority,
        requested_position_size=trade_input.llm_position_size,
        requested_value=requested_value,
        latest_price=latest_price.price,
        latest_price_date=latest_price.timestamp,
        available_cash_after_reserve=available_cash_after_reserve,
        available_exposure_value=available_exposure_value,
        max_executable_value=max_executable_value,
    )
