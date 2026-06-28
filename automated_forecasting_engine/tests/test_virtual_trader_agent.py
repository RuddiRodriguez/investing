from __future__ import annotations

from market_forecasting_engine.virtual_trader_agent import (
    VirtualTraderAgentConfig,
    _plan_should_forecast,
    _skip_forecast_reason_for_open_orders,
    build_portfolio_state,
)
from market_forecasting_engine.virtual_trader_memory import VirtualTraderMemory


def test_plan_scan_implies_forecast_selected_candidates() -> None:
    assert _plan_should_forecast({"should_scan_new_candidates": True, "tasks": []}) is True
    assert _plan_should_forecast({"tasks": [{"task_type": "scan_new_candidates"}]}) is True


def test_open_order_maintenance_skips_forecast_rerun() -> None:
    reason = _skip_forecast_reason_for_open_orders(
        plan={"cycle_mode": "order_maintenance", "forecast_tickers": ["WFC", "TFC"], "should_scan_new_candidates": False},
        broker_state={"open_orders": [{"symbol": "WFC"}]},
        portfolio_state={"state": "manage_pending_portfolio"},
    )

    assert reason == "open_orders_pending_order_maintenance_only"


def test_scan_can_still_run_with_open_orders_when_planner_requests_scan() -> None:
    reason = _skip_forecast_reason_for_open_orders(
        plan={"cycle_mode": "manage_existing_portfolio", "should_scan_new_candidates": True},
        broker_state={"open_orders": [{"symbol": "WFC"}]},
        portfolio_state={"state": "manage_pending_portfolio"},
    )

    assert reason is None


def test_portfolio_state_empty_bootstrap_has_diversity_signal(tmp_path) -> None:
    memory = VirtualTraderMemory.load(tmp_path / "memory.json")
    state = build_portfolio_state(
        memory=memory,
        broker_state={"account": {"equity": "10000", "cash": "10000", "buying_power": "10000"}, "clock": {"is_open": True}, "positions": [], "open_orders": []},
        config=VirtualTraderAgentConfig(output_root=tmp_path, memory_path=tmp_path / "memory.json"),
    )

    assert state["state"] == "bootstrap_empty_portfolio"
    assert state["diversity"]["status"] == "empty"
    assert state["next_check_seconds"] >= 900


def test_portfolio_state_exposes_order_lifecycle_events(tmp_path) -> None:
    memory = VirtualTraderMemory.load(tmp_path / "memory.json")
    memory.record_order(
        {
            "ticker": "WFC",
            "symbol": "WFC",
            "order_payload": {"client_order_id": "vt_dip_wfc_test", "symbol": "WFC"},
            "order_result": {
                "submitted": True,
                "broker_response": {"id": "order-1", "client_order_id": "vt_dip_wfc_test", "symbol": "WFC", "status": "pending_new"},
            },
        }
    )
    memory.broker_snapshot(
        account={"equity": "10000", "cash": "10000", "buying_power": "10000"},
        positions=[],
        orders=[],
        recent_orders=[{"id": "order-1", "client_order_id": "vt_dip_wfc_test", "symbol": "WFC", "status": "expired"}],
    )
    state = build_portfolio_state(
        memory=memory,
        broker_state={
            "account": {"equity": "10000", "cash": "10000", "buying_power": "10000"},
            "clock": {"is_open": False},
            "positions": [],
            "open_orders": [],
            "recent_orders": [{"id": "order-1", "client_order_id": "vt_dip_wfc_test", "symbol": "WFC", "status": "expired"}],
        },
        config=VirtualTraderAgentConfig(output_root=tmp_path, memory_path=tmp_path / "memory.json"),
    )

    assert state["recent_orders_count"] == 1
    assert state["expired_order_events_count"] == 1
    assert state["order_lifecycle_events"][0]["status"] == "expired"
