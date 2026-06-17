from __future__ import annotations

from pathlib import Path

from market_forecasting_engine.virtual_trader_memory import VirtualTraderMemory
from market_forecasting_engine.virtual_trader_pipeline import (
    VirtualTraderPipelineConfig,
    build_alpaca_paper_order_plan,
    maybe_submit_alpaca_order,
)


def test_virtual_trader_memory_persists_watch_plan(tmp_path: Path) -> None:
    memory = VirtualTraderMemory.load(tmp_path / "memory.json")
    memory.record_decision(
        {
            "ticker": "AAA",
            "action": "hold",
            "forecast_output_dir": "/tmp/run",
            "final_advice": {
                "headline": "Wait for pullback.",
                "buy_lower_price": 95.0,
                "stop_loss_price": 90.0,
                "take_profit_price": 110.0,
            },
            "change_triggers": ["pullback stabilizes"],
        }
    )
    memory.save()

    loaded = VirtualTraderMemory.load(tmp_path / "memory.json")
    context = loaded.context_for_ticker("AAA")

    assert context["active_watch_plan"]["buy_lower_price"] == 95.0
    assert context["recent_decisions"][0]["action"] == "hold"


def test_order_plan_defaults_to_paper_limit_buy_when_ceo_buys(tmp_path: Path) -> None:
    memory = VirtualTraderMemory.load(tmp_path / "memory.json")
    config = VirtualTraderPipelineConfig(output_dir=tmp_path, memory_path=tmp_path / "memory.json")
    report = {
        "ticker": "AAA",
        "current_price": 100.0,
        "suggested_action": "Buy",
        "llm_final_decision": {
            "decision": "Buy",
            "confidence": 0.8,
            "final_advice": {"buy_now_price": 100.0, "headline": "Buy with limit."},
        },
        "final_advice": {"buy_now_price": 100.0, "headline": "Buy with limit."},
        "decision_view": {"autonomous_execution_gate": {"execution_blocks": [], "warnings": []}},
    }
    broker_state = {
        "status": "ok",
        "account": {"equity": "10000", "buying_power": "5000"},
        "clock": {"is_open": True},
        "positions": [],
        "open_orders": [],
    }

    plan = build_alpaca_paper_order_plan(
        report=report,
        candidate={"ticker": "AAA", "rank": 1, "ranking_score": 0.8},
        broker_state=broker_state,
        memory=memory,
        config=config,
    )

    assert config.execute_paper_orders is True
    assert plan["execution_allowed"] is True
    assert plan["order_payload"]["side"] == "buy"
    assert plan["order_payload"]["order_type"] == "limit"


def test_maybe_submit_order_respects_dry_run(tmp_path: Path) -> None:
    config = VirtualTraderPipelineConfig(output_dir=tmp_path, execute_paper_orders=False)
    plan = {
        "execution_allowed": True,
        "order_payload": {
            "symbol": "AAA",
            "side": "buy",
            "order_type": "limit",
            "notional": 25.0,
            "limit_price": 10.0,
        },
    }

    result = maybe_submit_alpaca_order(plan, config)

    assert result["submitted"] is False
    assert result["reason"] == "dry_run"

