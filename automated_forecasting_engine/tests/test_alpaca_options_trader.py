from __future__ import annotations

from datetime import UTC, datetime

import pytest
import pandas as pd

from market_forecasting_engine.alpaca_options_trader import (
    OptionExecutionConfig,
    build_option_exit_plan,
    build_real_option_trade_plan,
    choose_option_exit_orders,
    score_option_contracts,
    size_option_position,
    submit_option_order,
)


class FakeBroker:
    def __init__(self) -> None:
        self.submitted = None

    def option_contracts(self, **kwargs):
        return [
            {
                "symbol": "TSLA260605C00450000",
                "name": "TSLA Jun 05 2026 450 Call",
                "status": "active",
                "tradable": True,
                "expiration_date": "2026-06-05",
                "strike_price": "450",
                "type": "call",
                "open_interest": "100",
            },
            {
                "symbol": "TSLA260605C00550000",
                "name": "TSLA Jun 05 2026 550 Call",
                "status": "active",
                "tradable": True,
                "expiration_date": "2026-06-05",
                "strike_price": "550",
                "type": "call",
                "open_interest": "100",
            },
        ]

    def option_snapshots(self, symbols):
        return {
            "TSLA260605C00450000": {
                "latestQuote": {"bp": 9.8, "ap": 10.2},
                "greeks": {"delta": 0.38, "gamma": 0.012, "theta": -0.18, "vega": 0.42, "rho": 0.04},
            },
            "TSLA260605C00550000": {
                "latestQuote": {"bp": 0.1, "ap": 2.0},
                "greeks": {"delta": 0.08, "gamma": 0.004, "theta": -0.08, "vega": 0.18, "rho": 0.01},
            },
        }

    def submit_order(self, **kwargs):
        self.submitted = kwargs
        return {"id": "paper-option-order", **kwargs}

    def submit_multileg_option_order(self, **kwargs):
        self.submitted = kwargs
        return {"id": "paper-mleg-option-order", **kwargs}


class FakeStraddleBroker:
    def __init__(self) -> None:
        self.submitted = None

    def option_contracts(self, **kwargs):
        option_type = kwargs.get("option_type")
        if option_type == "call":
            return [
                {
                    "symbol": "SOFI260605C00015000",
                    "status": "active",
                    "tradable": True,
                    "expiration_date": "2026-06-05",
                    "strike_price": "15",
                    "type": "call",
                    "open_interest": "1000",
                }
            ]
        if option_type == "put":
            return [
                {
                    "symbol": "SOFI260605P00015000",
                    "status": "active",
                    "tradable": True,
                    "expiration_date": "2026-06-05",
                    "strike_price": "15",
                    "type": "put",
                    "open_interest": "1000",
                }
            ]
        return []

    def option_snapshots(self, symbols):
        return {
            "SOFI260605C00015000": {
                "latestQuote": {"bp": 0.42, "ap": 0.46},
                "greeks": {"delta": 0.51, "gamma": 0.05, "theta": -0.01, "vega": 0.02},
            },
            "SOFI260605P00015000": {
                "latestQuote": {"bp": 0.38, "ap": 0.42},
                "greeks": {"delta": -0.49, "gamma": 0.05, "theta": -0.01, "vega": 0.02},
            },
        }

    def submit_multileg_option_order(self, **kwargs):
        self.submitted = kwargs
        return {"id": "paper-mleg-option-order", **kwargs}


class FakeMultiLegStrategyBroker:
    def __init__(self) -> None:
        self.submitted = None

    def option_contracts(self, **kwargs):
        option_type = kwargs.get("option_type")
        if option_type == "call":
            return [
                {"symbol": "SOFI260619C00015000", "status": "active", "tradable": True, "expiration_date": "2026-06-19", "strike_price": "15", "type": "call", "open_interest": "1000"},
                {"symbol": "SOFI260619C00016000", "status": "active", "tradable": True, "expiration_date": "2026-06-19", "strike_price": "16", "type": "call", "open_interest": "1000"},
            ]
        if option_type == "put":
            return [
                {"symbol": "SOFI260619P00014000", "status": "active", "tradable": True, "expiration_date": "2026-06-19", "strike_price": "14", "type": "put", "open_interest": "1000"},
                {"symbol": "SOFI260619P00015000", "status": "active", "tradable": True, "expiration_date": "2026-06-19", "strike_price": "15", "type": "put", "open_interest": "1000"},
            ]
        return []

    def option_snapshots(self, symbols):
        return {
            "SOFI260619C00015000": {"latestQuote": {"bp": 0.70, "ap": 0.76}, "greeks": {"delta": 0.50, "gamma": 0.04, "theta": -0.01, "vega": 0.02}},
            "SOFI260619C00016000": {"latestQuote": {"bp": 0.20, "ap": 0.24}, "greeks": {"delta": 0.25, "gamma": 0.03, "theta": -0.005, "vega": 0.01}},
            "SOFI260619P00014000": {"latestQuote": {"bp": 0.18, "ap": 0.22}, "greeks": {"delta": -0.25, "gamma": 0.03, "theta": -0.005, "vega": 0.01}},
            "SOFI260619P00015000": {"latestQuote": {"bp": 0.68, "ap": 0.74}, "greeks": {"delta": -0.50, "gamma": 0.04, "theta": -0.01, "vega": 0.02}},
        }

    def submit_multileg_option_order(self, **kwargs):
        self.submitted = kwargs
        return {"id": "paper-mleg-option-order", **kwargs}


class FakeCalendarBroker:
    def __init__(self) -> None:
        self.submitted = None

    def option_contracts(self, **kwargs):
        option_type = kwargs.get("option_type")
        gte = str(kwargs.get("expiration_date_gte") or "")
        if option_type != "call":
            return []
        if gte < "2026-06-10":
            return [{"symbol": "SOFI260605C00015000", "status": "active", "tradable": True, "expiration_date": "2026-06-05", "strike_price": "15", "type": "call", "open_interest": "1000"}]
        return [{"symbol": "SOFI260619C00015000", "status": "active", "tradable": True, "expiration_date": "2026-06-19", "strike_price": "15", "type": "call", "open_interest": "1000"}]

    def option_snapshots(self, symbols):
        return {
            "SOFI260605C00015000": {"latestQuote": {"bp": 0.42, "ap": 0.46}, "greeks": {"delta": 0.50, "gamma": 0.05, "theta": -0.02, "vega": 0.02}},
            "SOFI260619C00015000": {"latestQuote": {"bp": 0.84, "ap": 0.90}, "greeks": {"delta": 0.55, "gamma": 0.03, "theta": -0.01, "vega": 0.04}},
        }

    def submit_multileg_option_order(self, **kwargs):
        self.submitted = kwargs
        return {"id": "paper-mleg-option-order", **kwargs}


def test_score_option_contracts_applies_spread_and_premium_gates() -> None:
    broker = FakeBroker()
    contracts = broker.option_contracts()
    snapshots = broker.option_snapshots([row["symbol"] for row in contracts])

    scored = score_option_contracts(
        contracts=contracts,
        snapshots=snapshots,
        underlying_price=440.0,
        forecast={"predicted_price": 465.0, "expected_return": 0.02, "horizon_hours": 2},
        option_type="call",
        config=OptionExecutionConfig(underlying="TSLA", max_contract_premium=1500.0, max_total_debit=1500.0, max_spread_pct=0.2),
        now=datetime(2026, 6, 1, tzinfo=UTC),
    )

    assert scored[0]["symbol"] == "TSLA260605C00450000"
    assert scored[0]["accepted"] is True
    assert scored[0]["greeks"]["gamma"] == 0.012
    assert scored[0]["greeks"]["theta_decay_usd_for_horizon"] == 1.5
    assert scored[0]["trade_quality"]["grade"] in {"fair", "good", "excellent"}
    assert scored[0]["trade_quality"]["breakeven_price"] == 460.2
    assert "forecast_clears_breakeven" in scored[0]["trade_quality"]["interpretation"]
    assert scored[1]["accepted"] is False
    assert "spread_too_wide" in scored[1]["reasons"]


def test_score_option_contracts_blocks_missing_or_unsafe_greeks() -> None:
    contracts = [
        {
            "symbol": "AAPL260605C00200000",
            "status": "active",
            "tradable": True,
            "expiration_date": "2026-06-05",
            "strike_price": "200",
            "open_interest": "100",
        },
        {
            "symbol": "AAPL260605C00205000",
            "status": "active",
            "tradable": True,
            "expiration_date": "2026-06-05",
            "strike_price": "205",
            "open_interest": "100",
        },
    ]
    snapshots = {
        "AAPL260605C00200000": {"latestQuote": {"bp": 4.9, "ap": 5.1}, "greeks": {"delta": 0.42}},
        "AAPL260605C00205000": {
            "latestQuote": {"bp": 4.9, "ap": 5.1},
            "greeks": {"delta": 0.40, "gamma": 0.01, "theta": -100.0, "vega": 0.3},
        },
    }

    scored = score_option_contracts(
        contracts=contracts,
        snapshots=snapshots,
        underlying_price=198.0,
        forecast={"predicted_price": 210.0, "expected_return": 0.02, "horizon_hours": 4},
        option_type="call",
        config=OptionExecutionConfig(underlying="AAPL", max_spread_pct=0.1),
        now=datetime(2026, 6, 1, tzinfo=UTC),
    )

    by_symbol = {row["symbol"]: row for row in scored}
    assert "missing_greeks" in by_symbol["AAPL260605C00200000"]["reasons"]
    assert "theta_decay_too_large_vs_forecast_edge" in by_symbol["AAPL260605C00205000"]["reasons"]


def test_build_real_option_trade_plan_returns_limit_order_with_whole_contract_qty() -> None:
    broker = FakeBroker()

    plan = build_real_option_trade_plan(
        broker=broker,  # type: ignore[arg-type]
        underlying="TSLA",
        underlying_price=440.0,
        forecast={"predicted_price": 465.0, "expected_return": 0.02, "expected_direction": "Upward", "account_equity": 100_000.0, "horizon_hours": 2},
        config=OptionExecutionConfig(underlying="TSLA", max_contract_premium=1500.0, max_total_debit=1500.0, max_spread_pct=0.2),
        now=datetime(2026, 6, 1, tzinfo=UTC),
    )

    assert plan["action"] == "buy_option"
    assert plan["order"]["symbol"] == "TSLA260605C00450000"
    assert plan["order"]["qty"] == 1
    assert plan["order"]["type"] == "limit"
    assert "notional" not in plan["order"]
    assert plan["exit_plan"]["primary_exit"]["type"] == "stop_limit"
    assert plan["exit_plan"]["take_profit"]["type"] == "limit"


def test_build_real_option_trade_plan_blocks_oscillating_range_middle_entry() -> None:
    broker = FakeBroker()
    prices = pd.DataFrame({"close": [100, 101, 99, 100.5, 99.5, 100.2, 99.8, 100.1, 99.9, 100.0] * 12})

    plan = build_real_option_trade_plan(
        broker=broker,  # type: ignore[arg-type]
        underlying="TSLA",
        underlying_price=100.0,
        forecast={"predicted_price": 104.0, "expected_return": 0.02, "expected_direction": "Upward", "account_equity": 100_000.0, "horizon_hours": 0.25},
        config=OptionExecutionConfig(
            underlying="TSLA",
            enable_market_regime_filter=True,
            market_regime_lookback_rows=60,
            min_trend_strength_pct=0.003,
        ),
        prices=prices,
        now=datetime(2026, 6, 1, tzinfo=UTC),
    )

    assert plan["action"] == "hold"
    assert plan["reason"] == "market_regime_blocks_directional_entry"
    assert plan["market_regime"]["regime"] == "range_bound"
    assert plan["market_regime"]["reason"] == "price_in_middle_of_range"


def test_build_real_option_trade_plan_blocks_late_trend_after_reversal() -> None:
    broker = FakeBroker()
    prices = pd.DataFrame({"close": [1660, 1658, 1662, 1659, 1661, 1657, 1660, 1658, 1662, 1660] * 3 + [1655, 1648, 1640, 1632, 1624, 1616, 1608, 1610, 1612]})

    plan = build_real_option_trade_plan(
        broker=broker,  # type: ignore[arg-type]
        underlying="TSLA",
        underlying_price=1612.0,
        forecast={"predicted_price": 1580.0, "expected_return": -0.02, "expected_direction": "Downward", "account_equity": 100_000.0, "horizon_hours": 0.25},
        config=OptionExecutionConfig(
            underlying="TSLA",
            enable_market_regime_filter=True,
            market_regime_lookback_rows=20,
            min_trend_strength_pct=0.001,
            impulse_lookback_bars=8,
            min_impulse_move_pct=0.004,
            min_impulse_directional_bars=5,
            max_late_entry_move_pct=0.01,
            max_ema_extension_pct=0.004,
            exhaustion_reversal_bars=2,
        ),
        prices=prices,
        now=datetime(2026, 6, 1, tzinfo=UTC),
    )

    assert plan["action"] == "hold"
    assert plan["reason"] == "market_regime_blocks_directional_entry"
    assert plan["market_regime"]["regime"] == "late_trend"
    assert plan["market_regime"]["exhaustion"]["late_entry_block"] is True


def test_submit_option_order_rejects_notional_and_market_orders() -> None:
    broker = FakeBroker()

    with pytest.raises(ValueError, match="Unsupported"):
        submit_option_order(broker, {"symbol": "TSLA260605C00450000", "side": "buy", "type": "market", "qty": 1})

    with pytest.raises(ValueError, match="notional"):
        submit_option_order(
            broker,
            {
                "symbol": "TSLA260605C00450000",
                "side": "buy",
                "type": "limit",
                "qty": 1,
                "notional": 100,
                "limit_price": 1.0,
            },
        )


def test_build_real_option_trade_plan_can_prepare_paper_long_straddle() -> None:
    broker = FakeStraddleBroker()

    plan = build_real_option_trade_plan(
        broker=broker,  # type: ignore[arg-type]
        underlying="SOFI",
        underlying_price=15.0,
        forecast={"predicted_price": 15.05, "expected_return": 0.001, "expected_direction": "Upward", "account_equity": 100_000.0, "horizon_hours": 0.25},
        config=OptionExecutionConfig(
            underlying="SOFI",
            min_dte=1,
            max_dte=7,
            max_total_debit=100.0,
            risk_budget_pct=0.001,
            max_position_equity_pct=0.001,
            max_spread_pct=0.15,
            enable_multi_leg=True,
            option_strategy_mode="long_straddle",
        ),
        now=datetime(2026, 6, 4, tzinfo=UTC),
    )

    assert plan["action"] == "buy_option"
    assert plan["strategy"] == "long_straddle"
    assert plan["order"]["order_class"] == "mleg"
    assert plan["order"]["type"] == "limit"
    assert plan["order"]["limit_price"] == 0.87
    assert [leg["symbol"] for leg in plan["order"]["legs"]] == ["SOFI260605C00015000", "SOFI260605P00015000"]
    assert plan["trade_quality"]["upper_breakeven_price"] == 15.87
    assert plan["trade_quality"]["lower_breakeven_price"] == 14.13
    assert plan["trade_quality"]["grade"] in {"poor", "weak", "fair", "good", "excellent"}


def test_submit_option_order_routes_multileg_limit_payload() -> None:
    broker = FakeBroker()
    order = {
        "order_class": "mleg",
        "type": "limit",
        "qty": 2,
        "limit_price": 0.84,
        "time_in_force": "day",
        "legs": [
            {"side": "buy", "position_intent": "buy_to_open", "symbol": "SOFI260605C00015000", "ratio_qty": 1},
            {"side": "buy", "position_intent": "buy_to_open", "symbol": "SOFI260605P00015000", "ratio_qty": 1},
        ],
    }

    result = submit_option_order(broker, order, client_order_id="test-mleg")

    assert result["id"] == "paper-mleg-option-order"
    assert broker.submitted["legs"][0]["symbol"] == "SOFI260605C00015000"
    assert broker.submitted["limit_price"] == 0.84


def test_build_real_option_trade_plan_can_prepare_paper_short_iron_butterfly() -> None:
    broker = FakeMultiLegStrategyBroker()

    plan = build_real_option_trade_plan(
        broker=broker,  # type: ignore[arg-type]
        underlying="SOFI",
        underlying_price=15.0,
        forecast={"predicted_price": 15.02, "expected_return": 0.001, "expected_direction": "Upward", "account_equity": 100_000.0, "horizon_hours": 0.25},
        config=OptionExecutionConfig(
            underlying="SOFI",
            min_dte=1,
            max_dte=20,
            max_total_debit=100.0,
            max_contracts=1,
            max_spread_pct=0.25,
            enable_multi_leg=True,
            enable_short_option_strategies=True,
            max_legs=4,
            option_strategy_mode="short_iron_butterfly",
        ),
        now=datetime(2026, 6, 4, tzinfo=UTC),
    )

    assert plan["action"] == "buy_option"
    assert plan["strategy"] == "short_iron_butterfly"
    assert plan["order"]["order_class"] == "mleg"
    assert plan["order"]["limit_price"] < 0
    assert [leg["position_intent"] for leg in plan["order"]["legs"]] == ["buy_to_open", "sell_to_open", "sell_to_open", "buy_to_open"]
    assert plan["risk"]["max_defined_loss"] > 0
    assert plan["trade_quality"]["strategy"] == "short_iron_butterfly"


def test_build_real_option_trade_plan_can_prepare_paper_call_calendar() -> None:
    broker = FakeCalendarBroker()

    plan = build_real_option_trade_plan(
        broker=broker,  # type: ignore[arg-type]
        underlying="SOFI",
        underlying_price=15.0,
        forecast={"predicted_price": 15.4, "expected_return": 0.01, "expected_direction": "Upward", "account_equity": 100_000.0, "horizon_hours": 1},
        config=OptionExecutionConfig(
            underlying="SOFI",
            min_dte=1,
            max_dte=35,
            max_total_debit=100.0,
            risk_budget_pct=0.001,
            max_position_equity_pct=0.001,
            max_contracts=1,
            max_spread_pct=0.25,
            enable_multi_leg=True,
            enable_short_option_strategies=True,
            max_legs=2,
            option_strategy_mode="long_call_calendar",
            calendar_near_min_dte=1,
            calendar_near_max_dte=7,
            calendar_far_min_dte=8,
            calendar_far_max_dte=35,
        ),
        now=datetime(2026, 6, 4, tzinfo=UTC),
    )

    assert plan["action"] == "buy_option"
    assert plan["strategy"] == "long_call_calendar"
    assert plan["order"]["order_class"] == "mleg"
    assert plan["order"]["limit_price"] > 0
    assert [leg["position_intent"] for leg in plan["order"]["legs"]] == ["sell_to_open", "buy_to_open"]
    assert plan["risk"]["estimated_debit"] > 0
    assert plan["trade_quality"]["strategy"] == "long_call_calendar"


def test_exit_policy_can_select_trailing_stop_when_requested() -> None:
    plan = choose_option_exit_orders(
        entry_limit_price=10.0,
        qty=1,
        config=OptionExecutionConfig(underlying="TSLA", exit_order_policy="trailing_stop", stop_loss_pct=0.25),
    )

    assert plan["primary_exit"]["type"] == "trailing_stop"
    assert plan["primary_exit"]["trail_percent"] == 25.0
    assert plan["stop_loss"]["type"] == "stop_limit"


def test_size_option_position_uses_equity_risk_budget_and_caps() -> None:
    sizing = size_option_position(
        entry_limit_price=3.0,
        account_equity=100_000.0,
        config=OptionExecutionConfig(
            underlying="TSLA",
            max_contracts=5,
            max_contract_premium=None,
            max_total_debit=1000.0,
            risk_budget_pct=0.0075,
            max_position_equity_pct=0.01,
        ),
    )

    assert sizing["qty"] == 2
    assert sizing["premium_per_contract"] == 300.0
    assert sizing["estimated_debit"] == 600.0
    assert sizing["budget"] == 750.0


def test_size_option_position_blocks_contract_above_premium_cap() -> None:
    sizing = size_option_position(
        entry_limit_price=5.0,
        account_equity=100_000.0,
        config=OptionExecutionConfig(underlying="TSLA", max_contract_premium=350.0),
    )

    assert sizing["qty"] == 0
    assert sizing["reason"] == "premium_per_contract_above_cap"


def test_size_option_position_without_premium_cap_can_buy_expensive_contract_within_budget() -> None:
    sizing = size_option_position(
        entry_limit_price=5.0,
        account_equity=100_000.0,
        config=OptionExecutionConfig(
            underlying="TSLA",
            max_contract_premium=None,
            max_total_debit=1000.0,
            risk_budget_pct=0.0075,
            max_position_equity_pct=0.01,
            max_contracts=3,
        ),
    )

    assert sizing["qty"] == 1
    assert sizing["premium_per_contract"] == 500.0
