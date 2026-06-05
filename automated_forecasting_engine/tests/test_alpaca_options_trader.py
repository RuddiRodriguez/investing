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
