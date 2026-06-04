from __future__ import annotations

from datetime import UTC, datetime

import pytest

from market_forecasting_engine.deribit_options_trader import (
    DeribitOptionExecutionConfig,
    build_fibonacci_analysis,
    build_deribit_option_trade_plan,
    score_deribit_option_contracts,
    size_deribit_option_position,
    submit_deribit_limit_order,
)
from market_forecasting_engine.deribit_options_agent import (
    _cancel_open_option_orders,
    _liquidate_option_positions,
    _manage_expiry_risk,
    _manage_position_unrealized_guards,
    _manage_total_unrealized_profit,
    _manage_total_unrealized_loss,
    _positions_for_total_profit_close,
    _positions_for_total_loss_close,
    _protective_close_triggered,
    _waiting_until_next_forecast,
    execution_block_reasons,
    manage_existing_orders_and_positions,
)


class FakeDeribitBroker:
    def __init__(self) -> None:
        self.submitted = None

    def instruments(self, *, currency: str = "ETH", kind: str = "option", expired: bool = False):
        return [
            {
                "instrument_name": "ETH-5JUN26-2100-C",
                "option_type": "call",
                "strike": 2100.0,
                "expiration_timestamp": 1780650000000,
                "is_active": True,
                "min_trade_amount": 0.1,
                "tick_size": 0.0001,
            },
            {
                "instrument_name": "ETH-5JUN26-2600-C",
                "option_type": "call",
                "strike": 2600.0,
                "expiration_timestamp": 1780650000000,
                "is_active": True,
                "min_trade_amount": 0.1,
                "tick_size": 0.0001,
            },
        ]

    def order_book(self, instrument_name: str, *, depth: int = 5):
        if instrument_name.endswith("2100-C"):
            return {
                "bids": [[0.052, 4]],
                "asks": [[0.055, 5]],
                "mark_price": 0.053,
                "greeks": {"delta": 0.45, "gamma": 0.006, "theta": -2.0, "vega": 0.5, "rho": 0.01},
                "stats": {"volume": 10, "volume_usd": 1000},
                "open_interest": 100,
            }
        return {
            "bids": [[0.001, 1]],
            "asks": [[0.02, 1]],
            "mark_price": 0.01,
            "greeks": {"delta": 0.08, "gamma": 0.001, "theta": -8.0, "vega": 0.2, "rho": 0.01},
            "stats": {"volume": 1, "volume_usd": 50},
            "open_interest": 1,
        }

    def buy_limit(self, **kwargs):
        self.submitted = {"side": "buy", **kwargs}
        return {"order": {"order_id": "deribit-order-1", **self.submitted}}

    def sell_limit(self, **kwargs):
        self.submitted = {"side": "sell", **kwargs}
        return {"order": {"order_id": "deribit-order-2", **self.submitted}}


class FakeExitBroker:
    def __init__(self) -> None:
        self.sell_orders = []
        self.cancelled = []

    def order_book(self, instrument_name: str, *, depth: int = 5):
        return {"bids": [[0.031, 2]], "asks": [[0.032, 2]], "mark_price": 0.031}

    def sell_limit(self, **kwargs):
        self.sell_orders.append(kwargs)
        if kwargs.get("reduce_only") is True:
            raise RuntimeError("invalid_reduce_only_order")
        return {"order": {"order_id": "fallback-exit", **kwargs}}

    def cancel_order(self, order_id: str):
        self.cancelled.append(order_id)
        return {"order_id": order_id, "order_state": "cancelled"}


class FakeNoGreeksBroker(FakeDeribitBroker):
    def order_book(self, instrument_name: str, *, depth: int = 5):
        return {
            "bids": [[0.052, 4]],
            "asks": [[0.055, 5]],
            "mark_price": 0.053,
            "stats": {"volume": 10, "volume_usd": 1000},
            "open_interest": 100,
        }


def test_score_deribit_contracts_uses_live_order_book_gates() -> None:
    broker = FakeDeribitBroker()
    scored = score_deribit_option_contracts(
        broker=broker,  # type: ignore[arg-type]
        instruments=broker.instruments(),
        underlying_price_usd=2000.0,
        forecast={"predicted_price": 2200.0, "expected_return": 0.03},
        option_type="call",
        config=DeribitOptionExecutionConfig(currency="ETH", max_spread_pct=0.2),
        now=datetime(2026, 6, 1, tzinfo=UTC),
    )

    assert scored[0]["instrument_name"] == "ETH-5JUN26-2100-C"
    assert scored[0]["accepted"] is True
    assert scored[0]["greeks"]["delta"] == 0.45
    assert "spread_too_wide" in scored[1]["reasons"]


def test_score_deribit_contracts_can_disable_greek_letter_gates() -> None:
    broker = FakeNoGreeksBroker()
    strict = score_deribit_option_contracts(
        broker=broker,  # type: ignore[arg-type]
        instruments=broker.instruments(),
        underlying_price_usd=2000.0,
        forecast={"predicted_price": 2200.0, "expected_return": 0.03},
        option_type="call",
        config=DeribitOptionExecutionConfig(currency="ETH", max_spread_pct=0.2, greeks_mode="required"),
        now=datetime(2026, 6, 1, tzinfo=UTC),
    )
    off = score_deribit_option_contracts(
        broker=broker,  # type: ignore[arg-type]
        instruments=broker.instruments(),
        underlying_price_usd=2000.0,
        forecast={"predicted_price": 2200.0, "expected_return": 0.03},
        option_type="call",
        config=DeribitOptionExecutionConfig(currency="ETH", max_spread_pct=0.2, greeks_mode="off"),
        now=datetime(2026, 6, 1, tzinfo=UTC),
    )

    assert "missing_greeks" in strict[0]["reasons"]
    assert off[0]["accepted"] is True
    assert off[0]["greeks"]["mode"] == "off"


def test_fibonacci_analysis_detects_supportive_bullish_confluence() -> None:
    import pandas as pd

    prices = pd.DataFrame({"close": [100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139, 142, 145, 148, 150, 146, 140, 134, 130.9]})
    fib = build_fibonacci_analysis(
        prices,
        current_price=130.9,
        forecast={"expected_return": 0.03, "predicted_price": 155},
        lookback_rows=30,
        max_distance_pct=0.01,
    )

    assert fib["status"] == "ok"
    assert fib["confirmation"] == "supportive"
    assert fib["nearest_support"] == 130.9
    assert fib["nearest_target_price"] > 130.9


def test_score_deribit_contracts_blocks_entries_too_close_to_forecast_horizon_and_expiry_close_window() -> None:
    broker = FakeDeribitBroker()
    scored = score_deribit_option_contracts(
        broker=broker,  # type: ignore[arg-type]
        instruments=broker.instruments(),
        underlying_price_usd=2000.0,
        forecast={"predicted_price": 2200.0, "expected_return": 0.03, "horizon_hours": 12},
        option_type="call",
        config=DeribitOptionExecutionConfig(
            currency="ETH",
            max_spread_pct=0.2,
            close_before_expiry_hours=12,
            entry_expiry_buffer_hours=4,
            min_hours_to_expiry_for_entry=18,
        ),
        now=datetime(2026, 6, 4, 7, tzinfo=UTC),
    )

    assert scored[0]["accepted"] is False
    assert "expiry_too_close_for_new_entry" in scored[0]["reasons"]
    assert "expiry_too_close_for_entry_horizon" in scored[0]["reasons"]
    assert scored[0]["required_hours_to_expiry_for_entry"] == 28


def test_build_deribit_trade_plan_uses_real_instrument_and_base_price_limit() -> None:
    broker = FakeDeribitBroker()

    plan = build_deribit_option_trade_plan(
        broker=broker,  # type: ignore[arg-type]
        currency="ETH",
        underlying_price_usd=2000.0,
        forecast={"predicted_price": 2200.0, "expected_return": 0.03, "expected_direction": "Upward"},
        account={"equity": 10.0, "balance": 10.0},
        config=DeribitOptionExecutionConfig(currency="ETH", max_spread_pct=0.2, max_contracts=2.0, max_total_debit_usd=500.0),
        now=datetime(2026, 6, 1, tzinfo=UTC),
    )

    assert plan["action"] == "buy_option"
    assert plan["order"]["instrument_name"] == "ETH-5JUN26-2100-C"
    assert plan["order"]["type"] == "limit"
    assert plan["order"]["price"] > 0
    assert plan["sizing"]["amount"] >= 0.1
    assert plan["risk"]["estimated_debit_usd"] > 0


def test_size_deribit_position_respects_minimum_trade_amount() -> None:
    sizing = size_deribit_option_position(
        entry_limit_price_base=0.05,
        underlying_price_usd=2000.0,
        account={"equity": 1.0},
        config=DeribitOptionExecutionConfig(currency="ETH", max_contracts=3.0, risk_budget_pct=0.01, max_total_debit_usd=100.0),
        min_trade_amount=0.1,
    )

    assert sizing["amount"] == 0.4
    assert sizing["estimated_debit_usd"] == 40.0


def test_submit_deribit_limit_order_rejects_non_limit_orders() -> None:
    broker = FakeDeribitBroker()

    with pytest.raises(ValueError, match="limit"):
        submit_deribit_limit_order(broker, {"type": "market", "side": "buy", "instrument_name": "ETH-X", "amount": 1, "price": 1})  # type: ignore[arg-type]

    result = submit_deribit_limit_order(
        broker,  # type: ignore[arg-type]
        {"type": "limit", "side": "buy", "instrument_name": "ETH-5JUN26-2100-C", "amount": 0.1, "price": 0.05},
        label="test",
    )
    assert result["order"]["instrument_name"] == "ETH-5JUN26-2100-C"
    assert broker.submitted["post_only"] is False


def test_manage_positions_uses_position_entry_for_exit_and_falls_back_after_reduce_only_rejection() -> None:
    from argparse import Namespace

    broker = FakeExitBroker()
    actions = manage_existing_orders_and_positions(
        broker=broker,  # type: ignore[arg-type]
        args=Namespace(currency="ETH", execute_paper_orders=True, stop_loss_pct=0.35, take_profit_pct=0.55, abandon_entry_after_seconds=300),
        open_orders=[],
        positions=[{"instrument_name": "ETH-3JUN26-2000-P", "size": 1, "average_price": 0.0185}],
        state={},
        now=datetime(2026, 6, 1, tzinfo=UTC),
    )

    assert actions[0]["action"] == "take_profit_triggered"
    assert actions[1]["action"] == "submitted_exit_without_reduce_only_after_rejection"
    assert broker.sell_orders[0]["price"] == 0.031
    assert broker.sell_orders[0]["reduce_only"] is True
    assert broker.sell_orders[1]["price"] == 0.031
    assert broker.sell_orders[1]["reduce_only"] is False


def test_manage_positions_replaces_stale_take_profit_when_config_changes() -> None:
    from argparse import Namespace

    broker = FakeExitBroker()
    actions = manage_existing_orders_and_positions(
        broker=broker,  # type: ignore[arg-type]
        args=Namespace(currency="ETH", execute_paper_orders=True, stop_loss_pct=0.35, take_profit_pct=0.30, abandon_entry_after_seconds=300),
        open_orders=[
            {
                "order_id": "old-take-profit",
                "instrument_name": "ETH-3JUN26-2000-P",
                "direction": "sell",
                "price": 0.028675,
            }
        ],
        positions=[{"instrument_name": "ETH-3JUN26-2000-P", "size": 1, "average_price": 0.0185}],
        state={},
        now=datetime(2026, 6, 1, tzinfo=UTC),
    )

    assert actions[0]["action"] == "take_profit_triggered"
    assert broker.cancelled == ["old-take-profit"]
    assert broker.sell_orders[1]["price"] == 0.031


def test_expiry_risk_warns_before_close_window() -> None:
    from argparse import Namespace

    broker = FakeExitBroker()
    actions = _manage_expiry_risk(
        broker=broker,  # type: ignore[arg-type]
        args=Namespace(currency="ETH", execute_paper_orders=True, close_before_expiry_hours=12, expiry_warning_hours=24, liquidation_limit_offset_pct=0.05),
        open_orders=[],
        positions=[{"instrument_name": "ETH-2JUN26-2000-C", "size": 1, "mark_price": 0.02}],
        now=datetime(2026, 6, 1, 16, tzinfo=UTC),
    )

    assert actions == [
        {
            "action": "expiry_position_warning",
            "instrument_name": "ETH-2JUN26-2000-C",
            "expiration_utc": "2026-06-02T08:00:00+00:00",
            "hours_to_expiry": 16.0,
            "close_before_expiry_hours": 12.0,
            "expiry_warning_hours": 24.0,
        }
    ]


def test_expiry_risk_closes_before_expiry_with_reduce_only_limit() -> None:
    from argparse import Namespace

    broker = FakeExitBroker()
    actions = _manage_expiry_risk(
        broker=broker,  # type: ignore[arg-type]
        args=Namespace(currency="ETH", execute_paper_orders=True, close_before_expiry_hours=12, expiry_warning_hours=24, liquidation_limit_offset_pct=0.05),
        open_orders=[{"order_id": "old-exit", "instrument_name": "ETH-2JUN26-2000-C", "direction": "sell"}],
        positions=[{"instrument_name": "ETH-2JUN26-2000-C", "size": 1, "mark_price": 0.02}],
        now=datetime(2026, 6, 2, 1, tzinfo=UTC),
    )

    assert actions[0]["action"] == "expiry_position_close_triggered"
    assert actions[1]["action"] == "cancelled_existing_deribit_exit"
    assert actions[2]["action"] == "submitted_exit_without_reduce_only_after_rejection"
    assert broker.cancelled == ["old-exit"]
    assert broker.sell_orders[0]["reduce_only"] is True
    assert broker.sell_orders[1]["reduce_only"] is False


def test_deribit_liquidate_and_stop_cancels_orders_and_closes_positions() -> None:
    from argparse import Namespace

    broker = FakeExitBroker()
    args = Namespace(currency="ETH", execute_paper_orders=True, liquidation_limit_offset_pct=0.05)
    cancel_actions = _cancel_open_option_orders(
        broker=broker,  # type: ignore[arg-type]
        args=args,
        open_orders=[{"order_id": "open-buy", "instrument_name": "ETH-3JUN26-2000-P", "direction": "buy"}],
        cancel_action="cancelled_open_order_for_liquidate_and_stop",
    )
    close_actions = _liquidate_option_positions(
        broker=broker,  # type: ignore[arg-type]
        args=args,
        open_orders=[],
        positions=[{"instrument_name": "ETH-3JUN26-2000-P", "size": 1, "mark_price": 0.02, "floating_profit_loss": 0.001}],
        now=datetime(2026, 6, 1, tzinfo=UTC),
    )

    assert cancel_actions[0]["action"] == "cancelled_open_order_for_liquidate_and_stop"
    assert broker.cancelled == ["open-buy"]
    assert close_actions[0]["action"] == "manual_liquidate_and_stop_triggered"
    assert close_actions[1]["action"] == "manual_liquidation_position_close_triggered"
    assert broker.sell_orders[1]["price"] == 0.019


def test_stop_request_blocks_new_deribit_entries() -> None:
    from argparse import Namespace

    reasons = execution_block_reasons(
        args=Namespace(max_open_option_orders=5, allow_duplicate_contract_order=False),
        trade_plan={"action": "hold", "reason": "currency_stop_requested"},
        open_orders=[],
    )

    assert reasons == ["currency_stop_requested"]


def test_deribit_blocks_new_entries_when_position_limits_are_reached() -> None:
    from argparse import Namespace

    reasons = execution_block_reasons(
        args=Namespace(
            max_open_option_orders=5,
            allow_duplicate_contract_order=False,
            max_open_option_positions=1,
            max_open_option_contracts=2,
            max_open_option_premium_usd=100,
            allow_mixed_option_direction=False,
        ),
        trade_plan={"action": "buy_option", "order": {"instrument_name": "ETH-5JUN26-1900-P"}},
        open_orders=[],
        positions=[{"instrument_name": "ETH-6JUN26-1800-P", "size": 2, "mark_price": 0.04}],
        underlying_price_usd=1800,
    )

    assert "max_open_option_positions_reached" in reasons
    assert "max_open_option_contracts_reached" in reasons
    assert "max_open_option_premium_usd_reached" in reasons


def test_deribit_blocks_mixed_call_put_direction_by_default() -> None:
    from argparse import Namespace

    reasons = execution_block_reasons(
        args=Namespace(
            max_open_option_orders=5,
            allow_duplicate_contract_order=False,
            max_open_option_positions=5,
            max_open_option_contracts=10,
            max_open_option_premium_usd=1000,
            allow_mixed_option_direction=False,
        ),
        trade_plan={"action": "buy_option", "order": {"instrument_name": "ETH-5JUN26-1900-C"}},
        open_orders=[],
        positions=[{"instrument_name": "ETH-6JUN26-1800-P", "size": 1, "mark_price": 0.01}],
        underlying_price_usd=1800,
    )

    assert "mixed_option_direction_blocked" in reasons


def test_manage_positions_closes_put_when_forecast_reverses_upward() -> None:
    from argparse import Namespace

    broker = FakeExitBroker()
    actions = manage_existing_orders_and_positions(
        broker=broker,  # type: ignore[arg-type]
        args=Namespace(
            currency="ETH",
            execute_paper_orders=True,
            stop_loss_pct=0.35,
            take_profit_pct=0.55,
            abandon_entry_after_seconds=300,
            max_total_unrealized_loss_usd=None,
            max_total_unrealized_profit_usd=None,
            max_position_unrealized_loss_usd=None,
            max_position_unrealized_profit_usd=None,
            enable_forecast_reversal_exit=True,
            min_reversal_edge_pct=0.001,
            liquidation_limit_offset_pct=0.05,
            close_before_expiry_hours=12,
            expiry_warning_hours=24,
        ),
        open_orders=[],
        positions=[{"instrument_name": "ETH-12JUN26-1800-P", "size": 1, "average_price": 0.03, "mark_price": 0.031}],
        state={},
        now=datetime(2026, 6, 1, tzinfo=UTC),
        underlying_price_usd=1800.0,
        forecast={"expected_direction": "Upward", "spot": 1800.0, "predicted_price": 1810.0},
    )

    assert actions[0]["action"] == "forecast_reversal_position_close_triggered"
    assert actions[0]["position_option_type"] == "put"
    assert actions[0]["desired_option_type"] == "call"
    assert actions[1]["action"] == "submitted_exit_without_reduce_only_after_rejection"
    assert broker.sell_orders[0]["instrument_name"] == "ETH-12JUN26-1800-P"
    assert broker.sell_orders[0]["reduce_only"] is True


def test_deribit_total_unrealized_loss_closes_positions_in_usd() -> None:
    from argparse import Namespace

    broker = FakeExitBroker()
    actions = _manage_total_unrealized_loss(
        broker=broker,  # type: ignore[arg-type]
        args=Namespace(
            currency="ETH",
            execute_paper_orders=True,
            max_total_unrealized_loss_usd=50,
            total_loss_close_mode="all",
            liquidation_limit_offset_pct=0.05,
        ),
        open_orders=[],
        positions=[
            {"instrument_name": "ETH-3JUN26-2000-P", "size": 1, "mark_price": 0.02, "floating_profit_loss": -0.02},
            {"instrument_name": "ETH-4JUN26-2100-P", "size": 1, "mark_price": 0.03, "floating_profit_loss": -0.01},
        ],
        now=datetime(2026, 6, 1, tzinfo=UTC),
        underlying_price_usd=2000.0,
    )

    assert actions[0]["action"] == "max_total_unrealized_loss_usd_triggered"
    assert actions[0]["total_unrealized_pl_usd"] == -60
    assert len([action for action in actions if action["action"] == "total_loss_position_close_triggered"]) == 2
    assert broker.sell_orders[1]["price"] == 0.019
    assert broker.sell_orders[3]["price"] == 0.0284


def test_deribit_zero_loss_cutoff_closes_when_pl_is_negative() -> None:
    from argparse import Namespace

    broker = FakeExitBroker()
    actions = _manage_total_unrealized_loss(
        broker=broker,  # type: ignore[arg-type]
        args=Namespace(
            currency="ETH",
            execute_paper_orders=True,
            max_total_unrealized_loss_usd=0,
            total_loss_close_mode="all",
            liquidation_limit_offset_pct=0.05,
        ),
        open_orders=[],
        positions=[{"instrument_name": "ETH-3JUN26-2000-C", "size": 1, "mark_price": 0.02, "floating_profit_loss": -0.001}],
        now=datetime(2026, 6, 1, tzinfo=UTC),
        underlying_price_usd=2000.0,
    )

    assert actions[0]["action"] == "max_total_unrealized_loss_usd_triggered"
    assert actions[0]["loss_cutoff_usd"] == -0.0
    assert broker.sell_orders[1]["price"] == 0.019


def test_deribit_negative_total_loss_value_is_treated_as_absolute_cutoff() -> None:
    from argparse import Namespace

    broker = FakeExitBroker()
    actions = _manage_total_unrealized_loss(
        broker=broker,  # type: ignore[arg-type]
        args=Namespace(
            currency="ETH",
            execute_paper_orders=True,
            max_total_unrealized_loss_usd=-10,
            total_loss_close_mode="all",
            liquidation_limit_offset_pct=0.05,
        ),
        open_orders=[],
        positions=[{"instrument_name": "ETH-3JUN26-2000-C", "size": 1, "mark_price": 0.02, "floating_profit_loss": -0.006}],
        now=datetime(2026, 6, 1, tzinfo=UTC),
        underlying_price_usd=2000.0,
    )

    assert actions[0]["action"] == "max_total_unrealized_loss_usd_triggered"
    assert actions[0]["loss_cutoff_usd"] == -10
    assert broker.sell_orders[0]["instrument_name"] == "ETH-3JUN26-2000-C"


def test_deribit_total_loss_losing_only_filters_winners() -> None:
    positions = [
        {"instrument_name": "LOSS", "floating_profit_loss": -0.01},
        {"instrument_name": "WIN", "floating_profit_loss": 0.02},
    ]

    assert [row["instrument_name"] for row in _positions_for_total_loss_close(positions, "losing_only")] == ["LOSS"]
    assert [row["instrument_name"] for row in _positions_for_total_loss_close(positions, "all")] == ["LOSS", "WIN"]


def test_deribit_total_unrealized_profit_closes_positions_in_usd() -> None:
    from argparse import Namespace

    broker = FakeExitBroker()
    actions = _manage_total_unrealized_profit(
        broker=broker,  # type: ignore[arg-type]
        args=Namespace(
            currency="ETH",
            execute_paper_orders=True,
            max_total_unrealized_profit_usd=50,
            total_profit_close_mode="all",
            liquidation_limit_offset_pct=0.05,
        ),
        open_orders=[],
        positions=[
            {"instrument_name": "ETH-3JUN26-2000-C", "size": 1, "mark_price": 0.02, "floating_profit_loss": 0.02},
            {"instrument_name": "ETH-4JUN26-2100-C", "size": 1, "mark_price": 0.03, "floating_profit_loss": 0.01},
        ],
        now=datetime(2026, 6, 1, tzinfo=UTC),
        underlying_price_usd=2000.0,
    )

    assert actions[0]["action"] == "max_total_unrealized_profit_usd_triggered"
    assert actions[0]["total_unrealized_pl_usd"] == 60
    assert len([action for action in actions if action["action"] == "total_profit_position_close_triggered"]) == 2
    assert broker.sell_orders[1]["price"] == 0.019
    assert broker.sell_orders[3]["price"] == 0.0284


def test_deribit_total_profit_winning_only_filters_losers() -> None:
    positions = [
        {"instrument_name": "LOSS", "floating_profit_loss": -0.01},
        {"instrument_name": "WIN", "floating_profit_loss": 0.02},
    ]

    assert [row["instrument_name"] for row in _positions_for_total_profit_close(positions, "winning_only")] == ["WIN"]
    assert [row["instrument_name"] for row in _positions_for_total_profit_close(positions, "all")] == ["LOSS", "WIN"]


def test_deribit_per_position_loss_guard_closes_only_losing_position() -> None:
    from argparse import Namespace

    broker = FakeExitBroker()
    actions = _manage_position_unrealized_guards(
        broker=broker,  # type: ignore[arg-type]
        args=Namespace(
            currency="ETH",
            execute_paper_orders=True,
            max_position_unrealized_loss_usd=5,
            max_position_unrealized_profit_usd=None,
            liquidation_limit_offset_pct=0.05,
        ),
        open_orders=[],
        positions=[
            {"instrument_name": "ETH-3JUN26-2000-P", "size": 1, "mark_price": 0.02, "floating_profit_loss": -0.003},
            {"instrument_name": "ETH-4JUN26-2100-P", "size": 1, "mark_price": 0.03, "floating_profit_loss": 0.001},
        ],
        now=datetime(2026, 6, 1, tzinfo=UTC),
        underlying_price_usd=2000.0,
    )

    assert actions[0]["action"] == "position_loss_position_close_triggered"
    assert actions[0]["position_unrealized_pl_usd"] == -6
    assert broker.sell_orders[0]["instrument_name"] == "ETH-3JUN26-2000-P"
    assert broker.sell_orders[0]["reduce_only"] is True
    assert not any(order["instrument_name"] == "ETH-4JUN26-2100-P" for order in broker.sell_orders)


def test_deribit_negative_per_position_loss_value_is_treated_as_absolute_cutoff() -> None:
    from argparse import Namespace

    broker = FakeExitBroker()
    actions = _manage_position_unrealized_guards(
        broker=broker,  # type: ignore[arg-type]
        args=Namespace(
            currency="ETH",
            execute_paper_orders=True,
            max_position_unrealized_loss_usd=-5,
            max_position_unrealized_profit_usd=None,
            liquidation_limit_offset_pct=0.05,
        ),
        open_orders=[],
        positions=[{"instrument_name": "ETH-3JUN26-2000-P", "size": 1, "mark_price": 0.02, "floating_profit_loss": -0.003}],
        now=datetime(2026, 6, 1, tzinfo=UTC),
        underlying_price_usd=2000.0,
    )

    assert actions[0]["action"] == "position_loss_position_close_triggered"
    assert actions[0]["position_loss_cutoff_usd"] == -5
    assert broker.sell_orders[0]["instrument_name"] == "ETH-3JUN26-2000-P"


def test_deribit_per_position_profit_guard_closes_only_winning_position() -> None:
    from argparse import Namespace

    broker = FakeExitBroker()
    actions = _manage_position_unrealized_guards(
        broker=broker,  # type: ignore[arg-type]
        args=Namespace(
            currency="ETH",
            execute_paper_orders=True,
            max_position_unrealized_loss_usd=None,
            max_position_unrealized_profit_usd=5,
            liquidation_limit_offset_pct=0.05,
        ),
        open_orders=[],
        positions=[
            {"instrument_name": "ETH-3JUN26-2000-P", "size": 1, "mark_price": 0.02, "floating_profit_loss": -0.001},
            {"instrument_name": "ETH-4JUN26-2100-P", "size": 1, "mark_price": 0.03, "floating_profit_loss": 0.004},
        ],
        now=datetime(2026, 6, 1, tzinfo=UTC),
        underlying_price_usd=2000.0,
    )

    assert actions[1]["action"] == "position_profit_position_close_triggered"
    assert actions[1]["position_unrealized_pl_usd"] == 8
    assert broker.sell_orders[0]["instrument_name"] == "ETH-4JUN26-2100-P"
    assert broker.sell_orders[0]["reduce_only"] is True


def test_protective_close_triggers_wait_until_next_forecast() -> None:
    actions = [{"action": "position_profit_position_close_triggered"}]
    state = {"wait_until_next_forecast_after_close": {"forecast_created_at_utc": "same"}}

    assert _protective_close_triggered(actions) is True
    assert _waiting_until_next_forecast(state, {"created_at_utc": "same"}) is True
    assert _waiting_until_next_forecast(state, {"created_at_utc": "new"}) is False
