from __future__ import annotations

from pathlib import Path

from market_forecasting_engine.virtual_trader_memory import VirtualTraderMemory
from market_forecasting_engine.virtual_trader_pipeline import (
    VirtualTraderPipelineConfig,
    _pure_llm_report_for_order_planner,
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


def test_memory_reconciles_submitted_day_order_as_expired(tmp_path: Path) -> None:
    memory = VirtualTraderMemory.load(tmp_path / "memory.json")
    memory.record_order(
        {
            "ticker": "WFC",
            "symbol": "WFC",
            "action": "hold",
            "intent": "buy_lower_limit_watch",
            "execution_allowed": True,
            "order_payload": {
                "client_order_id": "vt_dip_wfc_test",
                "symbol": "WFC",
                "side": "buy",
                "order_type": "limit",
                "notional": 100.0,
                "limit_price": 81.0,
                "time_in_force": "day",
            },
            "order_result": {
                "submitted": True,
                "broker_response": {
                    "id": "alpaca-order-1",
                    "client_order_id": "vt_dip_wfc_test",
                    "symbol": "WFC",
                    "side": "buy",
                    "type": "limit",
                    "notional": "100",
                    "limit_price": "81",
                    "time_in_force": "day",
                    "status": "pending_new",
                    "expires_at": "2026-06-18T20:00:00Z",
                },
            },
        }
    )

    events = memory.broker_snapshot(
        account={"equity": "10000"},
        positions=[],
        orders=[],
        recent_orders=[
            {
                "id": "alpaca-order-1",
                "client_order_id": "vt_dip_wfc_test",
                "symbol": "WFC",
                "side": "buy",
                "type": "limit",
                "notional": "100",
                "limit_price": "81",
                "time_in_force": "day",
                "status": "expired",
                "expires_at": "2026-06-18T20:00:00Z",
                "expired_at": "2026-06-18T20:00:00Z",
            }
        ],
    )
    memory.save()

    loaded = VirtualTraderMemory.load(tmp_path / "memory.json")
    portfolio = loaded.portfolio_context()

    assert events[0]["status"] == "expired"
    assert events[0]["reason"] == "broker_reported_order_expired"
    assert portfolio["recent_order_lifecycle_events"][0]["client_order_id"] == "vt_dip_wfc_test"
    assert portfolio["order_lifecycle"]["alpaca-order-1"]["lifecycle_state"] == "terminal"


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


def test_hold_plan_marks_already_crossed_breakout_as_triggered(tmp_path: Path) -> None:
    memory = VirtualTraderMemory.load(tmp_path / "memory.json")
    config = VirtualTraderPipelineConfig(output_dir=tmp_path, memory_path=tmp_path / "memory.json")
    report = {
        "ticker": "GEV",
        "current_price": 1048.86,
        "suggested_action": "Hold",
        "llm_final_decision": {
            "decision": "Hold",
            "confidence": 0.7,
            "final_advice": {
                "headline": "Hold; wait for confirmation.",
                "buy_lower_price": 987.0,
                "buy_lower_zone_low": 974.0,
                "buy_lower_zone_high": 996.0,
                "buy_above_breakout_price": 1024.0,
                "stop_loss_price": 806.59,
                "take_profit_price": 1181.35,
            },
        },
        "final_advice": {
            "headline": "Hold; wait for confirmation.",
            "buy_lower_price": 987.0,
            "buy_lower_zone_low": 974.0,
            "buy_lower_zone_high": 996.0,
            "buy_above_breakout_price": 1024.0,
            "stop_loss_price": 806.59,
            "take_profit_price": 1181.35,
        },
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
        candidate={"ticker": "GEV", "rank": 1, "ranking_score": 0.8},
        broker_state=broker_state,
        memory=memory,
        config=config,
    )

    assert plan["action"] == "hold"
    assert plan["watch_conditions"]["breakout_buy"]["status"] == "trigger_crossed"
    assert plan["reason"] == "hold_after_breakout_crossed_buy_lower_retest_limit_enabled"
    assert "breakout_price_already_crossed_ceo_requires_confirmation_or_retest" in plan["warnings"]
    assert plan["final_advice"]["watch_condition_status"]["breakout_buy"]["status"] == "trigger_crossed"
    assert plan["order_payload"]["limit_price"] == 987.0
    assert plan["order_payload"]["time_in_force"] == "day"


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


def test_pure_llm_report_adapts_to_existing_order_planner_shape(tmp_path: Path) -> None:
    pure_report = {
        "ceo_provider": "openai",
        "ceo_model": "gpt-5.4-2026-03-05",
        "forecast": {"ticker": "AAA", "current_price": 100.0},
        "advice": {
            "decision": "Buy",
            "confidence": 0.82,
            "final_advice": {
                "headline": "Buy with a limit.",
                "buy_now_price": 100.0,
                "stop_loss_price": 95.0,
                "take_profit_price": 112.0,
            },
        },
        "execution_gate": {
            "execution_allowed": False,
            "hard_blocks": ["no_stock_broker_execution_path_configured"],
        },
    }

    report = _pure_llm_report_for_order_planner(pure_report, ticker="AAA")
    memory = VirtualTraderMemory.load(tmp_path / "memory.json")
    config = VirtualTraderPipelineConfig(output_dir=tmp_path, memory_path=tmp_path / "memory.json", forecast_backend="pure_llm")
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

    assert report["suggested_action"] == "Buy"
    assert report["llm_final_decision"]["decision"] == "Buy"
    assert report["final_advice"]["buy_now_price"] == 100.0
    assert report["decision_view"]["autonomous_execution_gate"]["execution_blocks"] == []
    assert plan["execution_allowed"] is True
    assert plan["order_payload"]["side"] == "buy"
