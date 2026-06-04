from __future__ import annotations

import argparse

import pytest

import market_forecasting_engine.paper_trader_agent as paper_trader_agent
from market_forecasting_engine.paper_trader_agent import (
    _alpaca_symbol,
    _derivative_intent,
    build_order_plan,
    decide_order,
    update_cached_prices,
)


def _args(**overrides):
    defaults = {
        "max_notional": 25.0,
        "min_notional": 5.0,
        "allow_short": False,
        "execute_paper_orders": False,
        "allow_multiple_open_orders": False,
        "max_open_orders": 3,
        "entry_order_type": "limit",
        "exit_order_type": "trailing_stop",
        "entry_limit_offset_bps": 8.0,
        "stop_buffer_bps": 8.0,
        "stop_limit_offset_bps": 5.0,
        "trailing_stop_percent": None,
        "enable_protective_exit": False,
        "enable_llm_decision": False,
        "ticker": "ETH-USD",
        "data_interval": "1m",
        "initial_lookback_hours": 24,
        "minimum_cache_rows": 2,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_alpaca_symbol_converts_crypto_dash_symbol() -> None:
    assert _alpaca_symbol("ETH-USD") == "ETH/USD"
    assert _alpaca_symbol("BTC/USD") == "BTC/USD"


def test_update_cached_prices_uses_existing_cache_when_alpaca_fetch_fails(monkeypatch, tmp_path) -> None:
    cache_path = tmp_path / "cache" / "ETH_USD_1m.csv"
    cache_path.parent.mkdir()
    cache_path.write_text(
        "\n".join(
            [
                "timestamp,open,high,low,close,volume",
                "2026-06-02T10:00:00,2000,2001,1999,2000.5,10",
                "2026-06-02T10:01:00,2001,2002,2000,2001.5,11",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_fetch(*args, **kwargs):
        raise RuntimeError("temporary dns failure")

    monkeypatch.setattr(paper_trader_agent, "load_prices_with_provider", fail_fetch)

    frame = update_cached_prices(_args(), tmp_path)

    assert frame["close"].iloc[-1] == 2001.5
    assert frame.attrs["price_cache_status"] == "fallback_cached_after_fetch_error"
    assert "temporary dns failure" in frame.attrs["price_cache_error"]


def test_update_cached_prices_fails_clearly_without_cache_when_alpaca_fetch_fails(monkeypatch, tmp_path) -> None:
    def fail_fetch(*args, **kwargs):
        raise RuntimeError("temporary dns failure")

    monkeypatch.setattr(paper_trader_agent, "load_prices_with_provider", fail_fetch)

    with pytest.raises(RuntimeError, match="Alpaca price fetch failed and cache has only 0 rows"):
        update_cached_prices(_args(), tmp_path)


def test_derivative_intent_maps_edge_to_call_put_or_no_trade() -> None:
    assert _derivative_intent({"expected_return": 0.002, "predicted_price": 101.0}, 0.001)["action"] == "buy_call"
    assert _derivative_intent({"expected_return": -0.002, "predicted_price": 99.0}, 0.001)["action"] == "buy_put"
    assert _derivative_intent({"expected_return": 0.0002, "predicted_price": 100.0}, 0.001)["action"] == "no_trade"


def test_decide_order_requires_execute_flag_but_prepares_buy_call_order() -> None:
    args = _args()
    decision = decide_order(
        args=args,
        profile_name="aggressive",
        profile_budget=0.0075,
        symbol="ETH/USD",
        latest_price=2000.0,
        forecast_record={
            "forecast": {"predicted_price": 2005.0, "forecast_date": "2026-05-31T12:15:00"},
            "spot_plan": {"forecasts": [{"predicted_price": 2005.0}]},
            "derivative_intent": {"action": "buy_call", "reason": "forecast_up_edge", "expected_return": 0.0025},
        },
        broker_state={"account": {"equity": "10000", "cash": "10000"}, "position": None, "open_orders": []},
    )

    assert decision["action"] == "submit_order"
    assert decision["side"] == "buy"
    assert decision["notional"] == 25.0
    assert decision["execute_paper_orders"] is False
    assert decision["order_plan"]["entry_order"]["type"] == "limit"
    assert decision["order_plan"]["entry_order"]["limit_price"] == 2001.6
    assert decision["order_plan"]["entry_order"]["qty"] == 0.012490008
    assert "notional" not in decision["order_plan"]["entry_order"]


def test_market_entry_uses_notional_for_crypto() -> None:
    args = _args(entry_order_type="market")
    decision = decide_order(
        args=args,
        profile_name="aggressive",
        profile_budget=0.0075,
        symbol="ETH/USD",
        latest_price=2000.0,
        forecast_record={
            "forecast": {"predicted_price": 2005.0, "forecast_date": "2026-05-31T12:15:00"},
            "spot_plan": {"forecasts": [{"predicted_price": 2005.0}]},
            "derivative_intent": {"action": "buy_call", "reason": "forecast_up_edge", "expected_return": 0.0025},
        },
        broker_state={"account": {"equity": "10000", "cash": "10000"}, "position": None, "open_orders": []},
    )

    assert decision["order_plan"]["entry_order"]["type"] == "market"
    assert decision["order_plan"]["entry_order"]["notional"] == 25.0
    assert "qty" not in decision["order_plan"]["entry_order"]


def test_llm_forecast_decision_is_final_trade_intent_when_enabled() -> None:
    args = _args(enable_llm_decision=True)
    decision = decide_order(
        args=args,
        profile_name="aggressive",
        profile_budget=0.0075,
        symbol="ETH/USD",
        latest_price=2000.0,
        forecast_record={
            "forecast": {"predicted_price": 2005.0, "forecast_date": "2026-05-31T12:15:00"},
            "spot_plan": {"forecasts": [{"predicted_price": 2005.0}]},
            "derivative_intent": {"action": "buy_call", "reason": "deterministic_buy", "expected_return": 0.0025},
            "llm_derivative_intent": {"action": "no_trade", "reason": "llm_final_decision_hold", "expected_return": 0.0025},
            "llm_trader": {"status": "executed", "decision": {"decision": "Hold", "confidence": 0.8}},
        },
        broker_state={"account": {"equity": "10000", "cash": "10000"}, "position": None, "open_orders": []},
    )

    assert decision["action"] == "hold"
    assert decision["decision_source"] == "llm_forecast_cache"
    assert decision["derivative_intent"]["reason"] == "llm_final_decision_hold"
    assert "no_directional_edge" in decision["reasons"]


def test_llm_buy_decision_allows_order_when_enabled() -> None:
    args = _args(enable_llm_decision=True)
    decision = decide_order(
        args=args,
        profile_name="aggressive",
        profile_budget=0.0075,
        symbol="ETH/USD",
        latest_price=2000.0,
        forecast_record={
            "forecast": {"predicted_price": 1995.0, "forecast_date": "2026-05-31T12:15:00"},
            "spot_plan": {"forecasts": [{"predicted_price": 1995.0}]},
            "derivative_intent": {"action": "buy_put", "reason": "deterministic_sell", "expected_return": -0.0025},
            "llm_derivative_intent": {"action": "buy_call", "reason": "llm_final_decision_buy", "expected_return": -0.0025},
            "llm_trader": {"status": "executed", "decision": {"decision": "Buy", "confidence": 0.8}},
        },
        broker_state={"account": {"equity": "10000", "cash": "10000"}, "position": None, "open_orders": []},
    )

    assert decision["action"] == "submit_order"
    assert decision["side"] == "buy"
    assert decision["derivative_intent"]["reason"] == "llm_final_decision_buy"


def test_decide_order_blocks_put_signal_without_position_or_short_permission() -> None:
    args = _args(execute_paper_orders=True)
    decision = decide_order(
        args=args,
        profile_name="aggressive",
        profile_budget=0.0075,
        symbol="ETH/USD",
        latest_price=2000.0,
        forecast_record={
            "forecast": {"predicted_price": 1995.0, "forecast_date": "2026-05-31T12:15:00"},
            "spot_plan": {"forecasts": [{"predicted_price": 1995.0}]},
            "derivative_intent": {"action": "buy_put", "reason": "forecast_down_edge", "expected_return": -0.0025},
        },
        broker_state={"account": {"equity": "10000", "cash": "10000"}, "position": None, "open_orders": []},
    )

    assert decision["action"] == "hold"
    assert "put_signal_without_short_or_position" in decision["reasons"]


def test_decide_order_allows_configured_multiple_open_orders() -> None:
    args = _args(execute_paper_orders=True, allow_multiple_open_orders=True, max_open_orders=3)
    decision = decide_order(
        args=args,
        profile_name="aggressive",
        profile_budget=0.0075,
        symbol="ETH/USD",
        latest_price=2000.0,
        forecast_record={
            "forecast": {"predicted_price": 2005.0, "forecast_date": "2026-05-31T12:15:00"},
            "spot_plan": {"forecasts": [{"predicted_price": 2005.0}]},
            "derivative_intent": {"action": "buy_call", "reason": "forecast_up_edge", "expected_return": 0.0025},
        },
        broker_state={"account": {"equity": "10000", "cash": "10000"}, "position": None, "open_orders": [{"id": "one"}]},
    )

    assert decision["action"] == "submit_order"
    assert decision["open_order_count"] == 1


def test_decide_order_blocks_when_max_open_orders_reached() -> None:
    args = _args(execute_paper_orders=True, allow_multiple_open_orders=True, max_open_orders=1)
    decision = decide_order(
        args=args,
        profile_name="aggressive",
        profile_budget=0.0075,
        symbol="ETH/USD",
        latest_price=2000.0,
        forecast_record={
            "forecast": {"predicted_price": 2005.0, "forecast_date": "2026-05-31T12:15:00"},
            "spot_plan": {"forecasts": [{"predicted_price": 2005.0}]},
            "derivative_intent": {"action": "buy_call", "reason": "forecast_up_edge", "expected_return": 0.0025},
        },
        broker_state={"account": {"equity": "10000", "cash": "10000"}, "position": None, "open_orders": [{"id": "one"}]},
    )

    assert decision["action"] == "hold"
    assert "max_open_orders_reached" in decision["reasons"]


def test_decide_order_blocks_opposite_open_order_conflict() -> None:
    args = _args(execute_paper_orders=True, allow_multiple_open_orders=True, max_open_orders=3)
    decision = decide_order(
        args=args,
        profile_name="aggressive",
        profile_budget=0.0075,
        symbol="ETH/USD",
        latest_price=2000.0,
        forecast_record={
            "forecast": {"predicted_price": 1995.0, "forecast_date": "2026-05-31T12:15:00"},
            "spot_plan": {"forecasts": [{"predicted_price": 1995.0}]},
            "derivative_intent": {"action": "buy_put", "reason": "forecast_down_edge", "expected_return": -0.0025},
        },
        broker_state={
            "account": {"equity": "10000", "cash": "10000"},
            "position": {"qty": "0.25", "market_value": "500"},
            "open_orders": [{"id": "buy-one", "symbol": "ETHUSD", "side": "buy", "status": "new"}],
        },
    )

    assert decision["action"] == "hold"
    assert "opposite_open_buy_order_exists" in decision["reasons"]


def test_build_order_plan_exits_existing_position_without_duplicate_protective_order() -> None:
    plan = build_order_plan(
        args=_args(enable_protective_exit=True, exit_order_type="trailing_stop"),
        symbol="ETH/USD",
        side="sell",
        latest_price=2000.0,
        notional=25.0,
        forecast_record={
            "forecast": {"predicted_price": 2010.0, "lower_price": 1985.0, "upper_price": 2020.0},
            "spot_plan": {"trade_plan": {"stop": 1990.0, "take_profit": 2018.0}},
        },
        position={"qty": "0.25", "market_value": "500"},
        open_orders=[],
    )

    entry = plan["entry_order"]
    assert entry["side"] == "sell"
    assert entry["type"] == "stop_limit"
    assert entry["requested_type"] == "trailing_stop"
    assert entry["qty"] == 0.25
    assert entry["stop_price"] == 2000.0
    assert entry["limit_price"] == 1999.0
    assert plan["protective_exit_order"] is None


def test_build_order_plan_adds_protective_trailing_stop_for_held_position_on_buy_side() -> None:
    plan = build_order_plan(
        args=_args(enable_protective_exit=True, exit_order_type="trailing_stop"),
        symbol="ETH/USD",
        side="buy",
        latest_price=2000.0,
        notional=25.0,
        forecast_record={
            "forecast": {"predicted_price": 2010.0, "lower_price": 1985.0, "upper_price": 2020.0},
            "spot_plan": {"trade_plan": {"stop": 1990.0, "take_profit": 2018.0}},
        },
        position={"qty": "0.25", "market_value": "500"},
        open_orders=[],
    )

    protective = plan["protective_exit_order"]
    assert protective["submit_now"] is True
    assert protective["type"] == "stop_limit"
    assert protective["requested_type"] == "trailing_stop"
    assert protective["qty"] == 0.25
    assert protective["stop_price"] == 1990.0
    assert protective["limit_price"] == 1989.01


def test_build_order_plan_can_use_stop_limit_from_model_stop() -> None:
    plan = build_order_plan(
        args=_args(enable_protective_exit=True, exit_order_type="stop_limit", stop_limit_offset_bps=10.0),
        symbol="ETH/USD",
        side="sell",
        latest_price=2000.0,
        notional=25.0,
        forecast_record={
            "forecast": {"predicted_price": 2010.0, "lower_price": 1985.0, "upper_price": 2020.0},
            "spot_plan": {"trade_plan": {"stop": 1990.0, "take_profit": 2018.0}},
        },
        position={"qty": "0.25", "market_value": "500"},
        open_orders=[],
    )

    entry = plan["entry_order"]
    assert entry["type"] == "stop_limit"
    assert entry["stop_price"] == 2000.0
    assert entry["limit_price"] == 1998.0
    assert plan["protective_exit_order"] is None
