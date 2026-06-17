from __future__ import annotations

from market_forecasting_engine.virtual_trader_agent import (
    VirtualTraderAgentConfig,
    _plan_should_forecast,
    build_portfolio_state,
)
from market_forecasting_engine.virtual_trader_memory import VirtualTraderMemory


def test_plan_scan_implies_forecast_selected_candidates() -> None:
    assert _plan_should_forecast({"should_scan_new_candidates": True, "tasks": []}) is True
    assert _plan_should_forecast({"tasks": [{"task_type": "scan_new_candidates"}]}) is True


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

