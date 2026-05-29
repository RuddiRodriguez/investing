from mark_to_market.repositories import (
    get_cash,
    get_latest_price,
    get_open_holdings,
    save_portfolio_valuation_log,
    update_holding_valuation,
    update_portfolio_state_from_valuation,
)
from mark_to_market.schemas import (
    HoldingForValuation,
    LatestPrice,
    PortfolioValuationSnapshot,
    UpdatedHoldingValuation,
)
from trader.repositories import (
    build_aggregate_portfolio_snapshot,
    save_aggregate_portfolio_snapshot,
    save_trader_log,
)
from trader.schemas import TraderRunLog


def calculate_holding_valuation(
    holding: HoldingForValuation,
    latest_price: LatestPrice,
    portfolio_value_before_update: float,
) -> UpdatedHoldingValuation:
    current_price = latest_price.price
    market_value = holding.quantity * current_price

    if holding.direction == "long":
        unrealized_pnl = (current_price - holding.average_entry_price) * holding.quantity
    else:
        unrealized_pnl = (holding.average_entry_price - current_price) * holding.quantity

    entry_value = holding.average_entry_price * holding.quantity
    if entry_value == 0:
        unrealized_pnl_pct = 0.0
    else:
        unrealized_pnl_pct = unrealized_pnl / entry_value

    if portfolio_value_before_update == 0:
        position_size = 0.0
    else:
        position_size = market_value / portfolio_value_before_update

    return UpdatedHoldingValuation(
        trader_name=holding.trader_name,
        ticker=holding.ticker,
        company_name=holding.company_name,
        direction=holding.direction,
        quantity=holding.quantity,
        average_entry_price=holding.average_entry_price,
        current_price=current_price,
        market_value=market_value,
        unrealized_pnl=unrealized_pnl,
        unrealized_pnl_pct=unrealized_pnl_pct,
        position_size=position_size,
    )


def run_mark_to_market_agent(
    trader_name: str,
) -> dict:
    holdings = get_open_holdings(trader_name=trader_name)
    cash = get_cash(trader_name=trader_name)

    if not holdings:
        snapshot = PortfolioValuationSnapshot(
            trader_name=trader_name,
            cash=cash,
            invested_value=0.0,
            total_portfolio_value=cash,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            open_positions_count=0,
            valuation_reason="No open holdings. Portfolio value equals cash.",
        )
        update_portfolio_state_from_valuation(snapshot)
        save_portfolio_valuation_log(snapshot)
        save_trader_log(
            TraderRunLog(
                trader_name=trader_name,
                event_type="mark_to_market",
                message="Portfolio marked to market with no open holdings.",
                metadata=snapshot.model_dump(),
            )
        )
        aggregate_snapshot = build_aggregate_portfolio_snapshot()
        save_aggregate_portfolio_snapshot(aggregate_snapshot)
        return {
            "trader_name": trader_name,
            "holdings_updated": 0,
            "missing_prices": [],
            "portfolio_value": cash,
        }

    portfolio_value_before_update = cash + sum(holding.market_value for holding in holdings)
    updated_holdings = []
    missing_prices = []

    for holding in holdings:
        latest_price = get_latest_price(holding.ticker)
        if latest_price is None:
            missing_prices.append(holding.ticker)
            continue

        updated_holding = calculate_holding_valuation(
            holding=holding,
            latest_price=latest_price,
            portfolio_value_before_update=portfolio_value_before_update,
        )
        update_holding_valuation(updated_holding)
        updated_holdings.append(updated_holding)

    invested_value = sum(holding.market_value for holding in updated_holdings)
    unrealized_pnl = sum(holding.unrealized_pnl for holding in updated_holdings)
    entry_value = sum(holding.average_entry_price * holding.quantity for holding in updated_holdings)

    if entry_value == 0:
        unrealized_pnl_pct = 0.0
    else:
        unrealized_pnl_pct = unrealized_pnl / entry_value

    total_portfolio_value = cash + invested_value

    snapshot = PortfolioValuationSnapshot(
        trader_name=trader_name,
        cash=cash,
        invested_value=invested_value,
        total_portfolio_value=total_portfolio_value,
        unrealized_pnl=unrealized_pnl,
        unrealized_pnl_pct=unrealized_pnl_pct,
        open_positions_count=len(updated_holdings),
        valuation_reason="Portfolio holdings updated with latest available close prices.",
    )
    update_portfolio_state_from_valuation(snapshot)
    save_portfolio_valuation_log(snapshot)
    save_trader_log(
        TraderRunLog(
            trader_name=trader_name,
            event_type="mark_to_market",
            message="Portfolio marked to market.",
            metadata={
                "cash": cash,
                "invested_value": invested_value,
                "total_portfolio_value": total_portfolio_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "holdings_updated": len(updated_holdings),
                "missing_prices": missing_prices,
            },
        )
    )

    aggregate_snapshot = build_aggregate_portfolio_snapshot()
    save_aggregate_portfolio_snapshot(aggregate_snapshot)

    return {
        "trader_name": trader_name,
        "holdings_updated": len(updated_holdings),
        "missing_prices": missing_prices,
        "cash": cash,
        "invested_value": invested_value,
        "total_portfolio_value": total_portfolio_value,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pnl_pct": unrealized_pnl_pct,
    }
