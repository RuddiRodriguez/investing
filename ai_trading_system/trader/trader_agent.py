from trader.profile_factory import build_trader_profile
from trader.repositories import (
    build_aggregate_portfolio_snapshot,
    create_initial_portfolio_state,
    get_portfolio_state,
    get_trader_profile,
    save_aggregate_portfolio_snapshot,
    save_trader_log,
    save_trader_profile,
    set_trader_status,
    trader_exists,
)
from trader.schemas import TraderRunLog


def initialize_trader_agent(
    trader_name: str,
    profile_type: str,
    initial_cash: float,
) -> dict:
    if trader_exists(trader_name):
        profile = get_trader_profile(trader_name)
        portfolio_state = get_portfolio_state(trader_name)
        save_trader_log(
            TraderRunLog(
                trader_name=trader_name,
                event_type="resume",
                message="Trader already exists. Resuming existing trader state.",
                metadata={
                    "profile_type": profile.profile_type if profile else None,
                    "current_cash": portfolio_state.cash if portfolio_state else None,
                },
            )
        )
        return {
            "status": "resumed",
            "trader_name": trader_name,
            "profile": profile.model_dump() if profile else None,
            "portfolio_state": portfolio_state.model_dump() if portfolio_state else None,
        }

    profile = build_trader_profile(
        trader_name=trader_name,
        profile_type=profile_type,
        initial_cash=initial_cash,
    )
    save_trader_profile(profile)
    create_initial_portfolio_state(profile)
    portfolio_state = get_portfolio_state(trader_name)

    save_trader_log(
        TraderRunLog(
            trader_name=trader_name,
            event_type="created",
            message="New trader profile and initial portfolio state created.",
            metadata={
                "profile_type": profile.profile_type,
                "initial_cash": profile.initial_cash,
            },
        )
    )

    snapshot = build_aggregate_portfolio_snapshot()
    save_aggregate_portfolio_snapshot(snapshot)

    return {
        "status": "created",
        "trader_name": trader_name,
        "profile": profile.model_dump(),
        "portfolio_state": portfolio_state.model_dump() if portfolio_state else None,
    }


def start_trader_agent(trader_name: str) -> dict:
    profile = get_trader_profile(trader_name)
    if profile is None:
        return {
            "status": "error",
            "message": f"Trader '{trader_name}' does not exist.",
        }

    set_trader_status(
        trader_name=trader_name,
        status="running",
    )
    save_trader_log(
        TraderRunLog(
            trader_name=trader_name,
            event_type="started",
            message="Trader status changed to running.",
            metadata=None,
        )
    )

    return {
        "status": "running",
        "trader_name": trader_name,
    }


def stop_trader_agent(trader_name: str) -> dict:
    profile = get_trader_profile(trader_name)
    if profile is None:
        return {
            "status": "error",
            "message": f"Trader '{trader_name}' does not exist.",
        }

    set_trader_status(
        trader_name=trader_name,
        status="stopped",
    )
    save_trader_log(
        TraderRunLog(
            trader_name=trader_name,
            event_type="stopped",
            message="Trader status changed to stopped.",
            metadata=None,
        )
    )

    snapshot = build_aggregate_portfolio_snapshot()
    save_aggregate_portfolio_snapshot(snapshot)

    return {
        "status": "stopped",
        "trader_name": trader_name,
    }


def get_trader_status(trader_name: str) -> dict:
    profile = get_trader_profile(trader_name)
    portfolio_state = get_portfolio_state(trader_name)

    if profile is None:
        return {
            "status": "not_found",
            "trader_name": trader_name,
        }

    return {
        "status": profile.status,
        "trader_name": trader_name,
        "profile": profile.model_dump(),
        "portfolio_state": portfolio_state.model_dump() if portfolio_state else None,
    }


def save_current_aggregate_snapshot() -> dict:
    snapshot = build_aggregate_portfolio_snapshot()
    save_aggregate_portfolio_snapshot(snapshot)
    return snapshot.model_dump()
