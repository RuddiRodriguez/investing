from trader.schemas import TraderProfile


def build_trader_profile(
    trader_name: str,
    profile_type: str,
    initial_cash: float,
) -> TraderProfile:
    profile_type = profile_type.lower().strip()

    if profile_type == "aggressive":
        return TraderProfile(
            trader_name=trader_name,
            profile_type="aggressive",
            initial_cash=initial_cash,
            current_cash=initial_cash,
            max_position_size=0.18,
            max_portfolio_exposure=0.95,
            min_cash_reserve=0.05,
            trade_frequency="high",
            risk_tolerance=0.85,
            status="running",
        )

    if profile_type == "medium":
        return TraderProfile(
            trader_name=trader_name,
            profile_type="medium",
            initial_cash=initial_cash,
            current_cash=initial_cash,
            max_position_size=0.10,
            max_portfolio_exposure=0.75,
            min_cash_reserve=0.15,
            trade_frequency="medium",
            risk_tolerance=0.55,
            status="running",
        )

    if profile_type == "conservative":
        return TraderProfile(
            trader_name=trader_name,
            profile_type="conservative",
            initial_cash=initial_cash,
            current_cash=initial_cash,
            max_position_size=0.05,
            max_portfolio_exposure=0.45,
            min_cash_reserve=0.35,
            trade_frequency="low",
            risk_tolerance=0.25,
            status="running",
        )

    raise ValueError("profile_type must be one of: aggressive, medium, conservative")
