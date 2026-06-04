from __future__ import annotations

import argparse
from datetime import UTC, datetime

from market_forecasting_engine.paper_options_agent import (
    apply_option_profile_defaults,
    clear_trade_pause_state,
    _current_option_price,
    _cancel_open_option_orders,
    _liquidate_option_positions,
    _manage_configured_stop_loss,
    _manage_expiry_risk,
    _manage_forecast_reversal_exit,
    _manage_position_unrealized_guards,
    _manage_take_profit,
    _manage_total_unrealized_profit,
    _manage_total_unrealized_loss,
    execution_block_reasons,
    option_entry_guard_reasons,
    _positions_for_total_profit_close,
    _positions_for_total_loss_close,
    _profit_close_limit_price,
    _profit_lock_stop_price,
    read_state,
    verify_liquidation_until_flat,
    write_state,
    write_stop_request,
)


class FakeBroker:
    def __init__(self) -> None:
        self.cancelled: list[str] = []
        self.submitted: list[dict] = []

    def cancel_order(self, order_id: str) -> dict:
        self.cancelled.append(order_id)
        return {"id": order_id, "status": "cancelled"}

    def submit_order(self, **kwargs) -> dict:
        self.submitted.append(kwargs)
        return {"id": "take-profit-order", **kwargs}


class FakePollingBroker(FakeBroker):
    def __init__(self, *, positions_by_poll: list[list[dict]], orders_by_poll: list[list[dict]]) -> None:
        super().__init__()
        self.positions_by_poll = positions_by_poll
        self.orders_by_poll = orders_by_poll
        self.position_calls = 0
        self.order_calls = 0

    def positions(self) -> list[dict]:
        index = min(self.position_calls, len(self.positions_by_poll) - 1)
        self.position_calls += 1
        return self.positions_by_poll[index]

    def orders(self, *, status: str = "open", limit: int = 50) -> list[dict]:
        index = min(self.order_calls, len(self.orders_by_poll) - 1)
        self.order_calls += 1
        return self.orders_by_poll[index]


def _args(**overrides) -> argparse.Namespace:
    defaults = {
        "ticker": "TSLA",
        "take_profit_pct": 0.55,
        "profit_lock_trigger_pct": 0.15,
        "profit_lock_ratio": 0.50,
        "take_profit_position_pl": 50.0,
        "profit_retrace_from_peak_pct": 0.35,
        "profit_close_limit_offset_pct": 0.03,
        "liquidation_limit_offset_pct": 0.05,
        "liquidation_retry_limit_offset_pct": 0.15,
        "liquidation_wait_seconds": 0.0,
        "liquidation_poll_seconds": 0.5,
        "stop_limit_offset_pct": 0.08,
        "stop_loss_pct": 0.35,
        "max_total_unrealized_loss": None,
        "total_loss_close_mode": "all",
        "max_total_unrealized_profit": None,
        "total_profit_close_mode": "all",
        "max_position_unrealized_loss": None,
        "max_position_unrealized_profit": None,
        "enable_forecast_reversal_exit": True,
        "min_reversal_edge_pct": 0.001,
        "close_before_expiry_hours": 2.0,
        "expiry_warning_hours": 24.0,
        "max_open_option_orders": 1,
        "max_open_option_positions": 4,
        "max_open_option_contracts": 4,
        "max_open_option_exposure": 2500.0,
        "max_realized_loss_per_day": 300.0,
        "allow_mixed_option_direction": False,
        "execute_paper_orders": True,
        "disable_profit_taking": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_apply_option_profile_defaults_sets_aggressive_today_defaults() -> None:
    args = argparse.Namespace(
        ticker="TSLA",
        risk_profile="aggressive",
        stop_loss_pct=None,
        take_profit_pct=None,
        profit_lock_trigger_pct=None,
        profit_lock_ratio=None,
        take_profit_position_pl=None,
        profit_retrace_from_peak_pct=None,
        max_spread_pct=None,
        max_theta_edge_ratio=None,
        max_theta_premium_pct_per_day=None,
        entry_cooldown_minutes=None,
        loss_cooldown_minutes=None,
        max_trades_per_day=None,
        max_consecutive_losses=None,
        max_open_option_positions=None,
        max_open_option_contracts=None,
        max_open_option_exposure=None,
        max_realized_loss_per_day=None,
        max_position_unrealized_loss=None,
        max_total_unrealized_profit=None,
        max_total_unrealized_loss=None,
        one_trade_per_forecast=None,
    )

    applied = apply_option_profile_defaults(args)

    assert applied.stop_loss_pct == 0.10
    assert applied.take_profit_pct == 0.55
    assert applied.profit_lock_trigger_pct == 0.08
    assert applied.profit_lock_ratio == 0.75
    assert applied.take_profit_position_pl == 50.0
    assert applied.profit_retrace_from_peak_pct == 0.35
    assert applied.max_spread_pct == 0.15
    assert applied.entry_cooldown_minutes == 1
    assert applied.loss_cooldown_minutes == 2
    assert applied.max_trades_per_day == 50
    assert applied.max_consecutive_losses == 10
    assert applied.max_open_option_positions == 1
    assert applied.max_open_option_contracts == 2
    assert applied.max_open_option_exposure == 2500.0
    assert applied.max_realized_loss_per_day == 300.0
    assert applied.max_position_unrealized_loss == 150.0
    assert applied.max_total_unrealized_profit == 150.0
    assert applied.max_total_unrealized_loss == 225.0
    assert applied.one_trade_per_forecast is False


def test_apply_option_profile_defaults_keeps_cli_overrides() -> None:
    args = argparse.Namespace(
        ticker="SPY",
        risk_profile="medium",
        stop_loss_pct=0.12,
        take_profit_pct=None,
        profit_lock_trigger_pct=None,
        profit_lock_ratio=None,
        take_profit_position_pl=None,
        profit_retrace_from_peak_pct=None,
        max_spread_pct=0.2,
        max_theta_edge_ratio=None,
        max_theta_premium_pct_per_day=None,
        entry_cooldown_minutes=5,
        loss_cooldown_minutes=None,
        max_trades_per_day=None,
        max_consecutive_losses=None,
        max_open_option_positions=None,
        max_open_option_contracts=None,
        max_open_option_exposure=None,
        max_realized_loss_per_day=None,
        max_position_unrealized_loss=None,
        max_total_unrealized_profit=250.0,
        max_total_unrealized_loss=100.0,
        one_trade_per_forecast=False,
    )

    applied = apply_option_profile_defaults(args)

    assert applied.stop_loss_pct == 0.12
    assert applied.take_profit_pct == 0.40
    assert applied.max_spread_pct == 0.2
    assert applied.entry_cooldown_minutes == 5
    assert applied.max_total_unrealized_profit == 250.0
    assert applied.max_total_unrealized_loss == 100.0
    assert applied.one_trade_per_forecast is False


def test_manage_take_profit_cancels_stop_and_submits_sell_limit() -> None:
    broker = FakeBroker()

    actions = _manage_take_profit(
        broker=broker,  # type: ignore[arg-type]
        args=_args(take_profit_position_pl=500.0),
        symbol="TSLA260603P00420000",
        qty=2,
        entry_price=3.65,
        position={"current_price": "5.75", "unrealized_pl": "420"},
        sell_orders=[{"id": "existing-stop", "symbol": "TSLA260603P00420000", "side": "sell", "type": "stop_limit"}],
        profit_peak={"peak_unrealized_pl": 420, "peak_current_price": 5.75},
        now=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
    )

    assert broker.cancelled == ["existing-stop"]
    submitted = dict(broker.submitted[0])
    client_order_id = submitted.pop("client_order_id")
    assert client_order_id.startswith("opt-tp-TSLA-SLA260603P00420000-20260601140000-")
    assert broker.submitted == [
        {
            "symbol": "TSLA260603P00420000",
            "side": "sell",
            "order_type": "limit",
            "qty": 2,
            "limit_price": 5.58,
            "time_in_force": "day",
            "client_order_id": client_order_id,
        }
    ]
    assert [action["action"] for action in actions] == [
        "take_profit_triggered",
        "cancelled_existing_exit_for_take_profit",
        "submitted_take_profit_close",
    ]


def test_manage_take_profit_waits_until_target_is_reached() -> None:
    broker = FakeBroker()

    actions = _manage_take_profit(
        broker=broker,  # type: ignore[arg-type]
        args=_args(take_profit_position_pl=500.0),
        symbol="TSLA260603P00420000",
        qty=2,
        entry_price=3.65,
        position={"current_price": "4.45", "unrealized_pl": "160"},
        sell_orders=[{"id": "existing-stop"}],
        profit_peak={"peak_unrealized_pl": 160, "peak_current_price": 4.45},
        now=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
    )

    assert [action["action"] for action in actions] == [
        "profit_lock_triggered",
        "cancelled_existing_exit_for_profit_lock",
        "submitted_profit_lock_stop",
    ]
    assert broker.cancelled == ["existing-stop"]
    assert broker.submitted[0]["order_type"] == "stop_limit"
    assert broker.submitted[0]["stop_price"] == 4.05


def test_manage_take_profit_closes_individual_position_by_dollar_profit() -> None:
    broker = FakeBroker()

    actions = _manage_take_profit(
        broker=broker,  # type: ignore[arg-type]
        args=_args(ticker="NVDA", take_profit_pct=0.55, take_profit_position_pl=50.0),
        symbol="NVDA260603C00225000",
        qty=1,
        entry_price=3.65,
        position={"current_price": "4.20", "unrealized_pl": "55"},
        sell_orders=[{"id": "existing-stop", "symbol": "NVDA260603C00225000", "side": "sell", "type": "stop_limit"}],
        profit_peak={"peak_unrealized_pl": 55, "peak_current_price": 4.2},
        now=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
    )

    assert broker.cancelled == ["existing-stop"]
    submitted = dict(broker.submitted[0])
    client_order_id = submitted.pop("client_order_id")
    assert client_order_id.startswith("opt-tp-NVDA-VDA260603C00225000-20260601140000-")
    assert broker.submitted == [
        {
            "symbol": "NVDA260603C00225000",
            "side": "sell",
            "order_type": "limit",
            "qty": 1,
            "limit_price": 4.07,
            "time_in_force": "day",
            "client_order_id": client_order_id,
        }
    ]
    assert actions[0]["action"] == "take_profit_triggered"
    assert actions[0]["trigger"] == "position_pl"


def test_manage_take_profit_closes_after_profit_retrace_from_peak() -> None:
    broker = FakeBroker()

    actions = _manage_take_profit(
        broker=broker,  # type: ignore[arg-type]
        args=_args(ticker="NVDA", take_profit_pct=0.55, take_profit_position_pl=50.0, profit_retrace_from_peak_pct=0.35),
        symbol="NVDA260603C00225000",
        qty=1,
        entry_price=3.65,
        position={"current_price": "4.10", "unrealized_pl": "45"},
        sell_orders=[{"id": "old-stop", "symbol": "NVDA260603C00225000", "side": "sell", "type": "stop_limit"}],
        profit_peak={"peak_unrealized_pl": 80, "peak_current_price": 4.45},
        now=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
    )

    assert broker.cancelled == ["old-stop"]
    assert broker.submitted[0]["order_type"] == "limit"
    assert broker.submitted[0]["limit_price"] == 3.98
    assert actions[0]["action"] == "take_profit_triggered"
    assert actions[0]["trigger"] == "profit_retrace"
    assert actions[0]["peak_unrealized_pl"] == 80


def test_current_option_price_can_use_market_value_when_mark_missing() -> None:
    assert _current_option_price({"qty": "2", "market_value": "940"}) == 4.7
    assert _profit_close_limit_price(4.7, 0.03) == 4.56
    assert _profit_lock_stop_price(entry_price=3.65, current_price=4.2, lock_ratio=0.5) == 3.92


def test_profit_management_does_nothing_before_lock_or_take_profit() -> None:
    broker = FakeBroker()

    actions = _manage_take_profit(
        broker=broker,  # type: ignore[arg-type]
        args=_args(take_profit_position_pl=500.0),
        symbol="TSLA260603P00420000",
        qty=2,
        entry_price=3.65,
        position={"current_price": "4.00", "unrealized_pl": "70"},
        sell_orders=[{"id": "existing-stop", "stop_price": "2.38"}],
        profit_peak={"peak_unrealized_pl": 70, "peak_current_price": 4.0},
        now=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
    )

    assert actions == []
    assert broker.cancelled == []
    assert broker.submitted == []


def test_configured_stop_update_raises_existing_stop_when_risk_is_tightened() -> None:
    broker = FakeBroker()

    actions = _manage_configured_stop_loss(
        broker=broker,  # type: ignore[arg-type]
        args=_args(stop_loss_pct=0.10),
        symbol="NVDA260603C00225000",
        qty=3,
        entry_price=2.25,
        position={"current_price": "2.20", "unrealized_pl": "-15"},
        sell_orders=[{"id": "old-stop", "stop_price": "1.46", "limit_price": "1.34"}],
        now=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
    )

    assert broker.cancelled == ["old-stop"]
    submitted = dict(broker.submitted[0])
    client_order_id = submitted.pop("client_order_id")
    assert client_order_id.startswith("opt-stopcfg-TSLA-VDA260603C00225000-20260601140000-")
    assert broker.submitted == [
        {
            "symbol": "NVDA260603C00225000",
            "side": "sell",
            "order_type": "stop_limit",
            "qty": 3,
            "stop_price": 2.02,
            "limit_price": 1.86,
            "time_in_force": "day",
            "client_order_id": client_order_id,
        }
    ]
    assert [action["action"] for action in actions] == [
        "configured_stop_update_triggered",
        "cancelled_existing_exit_for_configured_stop",
        "submitted_configured_stop_update",
    ]


def test_configured_stop_closes_when_new_stop_is_already_breached() -> None:
    broker = FakeBroker()

    actions = _manage_configured_stop_loss(
        broker=broker,  # type: ignore[arg-type]
        args=_args(ticker="NVDA", stop_loss_pct=0.10),
        symbol="NVDA260603C00225000",
        qty=3,
        entry_price=2.25,
        position={"current_price": "1.80", "unrealized_pl": "-135"},
        sell_orders=[{"id": "old-stop", "stop_price": "1.46", "limit_price": "1.34"}],
        now=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
    )

    assert broker.cancelled == ["old-stop"]
    assert broker.submitted[0]["order_type"] == "limit"
    assert broker.submitted[0]["limit_price"] == 1.75
    assert [action["action"] for action in actions] == [
        "configured_stop_already_breached",
        "cancelled_existing_exit_for_configured_stop",
        "submitted_configured_stop_risk_close",
    ]


def test_total_unrealized_loss_closes_all_positions_when_cutoff_is_breached() -> None:
    broker = FakeBroker()

    actions = _manage_total_unrealized_loss(
        broker=broker,  # type: ignore[arg-type]
        args=_args(ticker="NVDA", max_total_unrealized_loss=100),
        open_orders=[
            {"id": "stop-1", "symbol": "NVDA260603C00225000", "side": "sell"},
            {"id": "stop-2", "symbol": "NVDA260605C00225000", "side": "sell"},
        ],
        option_positions=[
            {"symbol": "NVDA260603C00225000", "qty": "3", "current_price": "1.80", "unrealized_pl": "-135"},
            {"symbol": "NVDA260605C00225000", "qty": "2", "current_price": "2.87", "unrealized_pl": "18"},
        ],
        now=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
    )

    assert broker.cancelled == ["stop-1", "stop-2"]
    assert broker.submitted == [
        {
            "symbol": "NVDA260603C00225000",
            "side": "sell",
            "order_type": "limit",
            "qty": 3,
            "limit_price": 1.75,
            "time_in_force": "day",
            "client_order_id": "opt-maxloss-NVDA-260603-00225000-20260601140000",
        },
        {
            "symbol": "NVDA260605C00225000",
            "side": "sell",
            "order_type": "limit",
            "qty": 2,
            "limit_price": 2.78,
            "time_in_force": "day",
            "client_order_id": "opt-maxloss-NVDA-260605-00225000-20260601140000",
        },
    ]
    assert [action["action"] for action in actions] == [
        "max_total_unrealized_loss_triggered",
        "total_loss_position_close_triggered",
        "cancelled_existing_exit_for_total_loss",
        "submitted_total_loss_close",
        "total_loss_position_close_triggered",
        "cancelled_existing_exit_for_total_loss",
        "submitted_total_loss_close",
    ]


def test_total_unrealized_loss_waits_above_cutoff() -> None:
    broker = FakeBroker()

    actions = _manage_total_unrealized_loss(
        broker=broker,  # type: ignore[arg-type]
        args=_args(ticker="NVDA", max_total_unrealized_loss=150),
        open_orders=[{"id": "stop-1", "symbol": "NVDA260603C00225000", "side": "sell"}],
        option_positions=[
            {"symbol": "NVDA260603C00225000", "qty": "3", "current_price": "1.80", "unrealized_pl": "-135"},
            {"symbol": "NVDA260605C00225000", "qty": "2", "current_price": "2.87", "unrealized_pl": "18"},
        ],
        now=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
    )

    assert actions == []
    assert broker.cancelled == []
    assert broker.submitted == []


def test_total_unrealized_loss_can_close_only_losing_positions() -> None:
    broker = FakeBroker()

    actions = _manage_total_unrealized_loss(
        broker=broker,  # type: ignore[arg-type]
        args=_args(ticker="NVDA", max_total_unrealized_loss=100, total_loss_close_mode="losing_only"),
        open_orders=[
            {"id": "stop-1", "symbol": "NVDA260603C00225000", "side": "sell"},
            {"id": "stop-2", "symbol": "NVDA260605C00225000", "side": "sell"},
        ],
        option_positions=[
            {"symbol": "NVDA260603C00225000", "qty": "3", "current_price": "1.80", "unrealized_pl": "-135"},
            {"symbol": "NVDA260605C00225000", "qty": "2", "current_price": "2.87", "unrealized_pl": "18"},
        ],
        now=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
    )

    assert broker.cancelled == ["stop-1"]
    assert len(broker.submitted) == 1
    assert broker.submitted[0]["symbol"] == "NVDA260603C00225000"
    assert [action["action"] for action in actions] == [
        "max_total_unrealized_loss_triggered",
        "total_loss_position_close_triggered",
        "cancelled_existing_exit_for_total_loss",
        "submitted_total_loss_close",
    ]


def test_total_unrealized_profit_closes_all_positions_when_target_is_reached() -> None:
    broker = FakeBroker()

    actions = _manage_total_unrealized_profit(
        broker=broker,  # type: ignore[arg-type]
        args=_args(ticker="NVDA", max_total_unrealized_profit=100),
        open_orders=[
            {"id": "stop-1", "symbol": "NVDA260603C00225000", "side": "sell"},
            {"id": "stop-2", "symbol": "NVDA260605C00225000", "side": "sell"},
        ],
        option_positions=[
            {"symbol": "NVDA260603C00225000", "qty": "3", "current_price": "2.80", "unrealized_pl": "135"},
            {"symbol": "NVDA260605C00225000", "qty": "2", "current_price": "2.87", "unrealized_pl": "18"},
        ],
        now=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
    )

    assert broker.cancelled == ["stop-1", "stop-2"]
    assert broker.submitted == [
        {
            "symbol": "NVDA260603C00225000",
            "side": "sell",
            "order_type": "limit",
            "qty": 3,
            "limit_price": 2.72,
            "time_in_force": "day",
            "client_order_id": "opt-maxtp-NVDA-260603-00225000-20260601140000",
        },
        {
            "symbol": "NVDA260605C00225000",
            "side": "sell",
            "order_type": "limit",
            "qty": 2,
            "limit_price": 2.78,
            "time_in_force": "day",
            "client_order_id": "opt-maxtp-NVDA-260605-00225000-20260601140000",
        },
    ]
    assert [action["action"] for action in actions] == [
        "max_total_unrealized_profit_triggered",
        "total_profit_position_close_triggered",
        "cancelled_existing_exit_for_total_profit",
        "submitted_total_profit_close",
        "total_profit_position_close_triggered",
        "cancelled_existing_exit_for_total_profit",
        "submitted_total_profit_close",
    ]


def test_total_unrealized_profit_can_close_only_winning_positions() -> None:
    broker = FakeBroker()

    actions = _manage_total_unrealized_profit(
        broker=broker,  # type: ignore[arg-type]
        args=_args(ticker="NVDA", max_total_unrealized_profit=100, total_profit_close_mode="winning_only"),
        open_orders=[
            {"id": "stop-1", "symbol": "NVDA260603C00225000", "side": "sell"},
            {"id": "stop-2", "symbol": "NVDA260605C00225000", "side": "sell"},
        ],
        option_positions=[
            {"symbol": "NVDA260603C00225000", "qty": "3", "current_price": "2.80", "unrealized_pl": "135"},
            {"symbol": "NVDA260605C00225000", "qty": "2", "current_price": "2.87", "unrealized_pl": "-18"},
        ],
        now=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
    )

    assert broker.cancelled == ["stop-1"]
    assert len(broker.submitted) == 1
    assert broker.submitted[0]["symbol"] == "NVDA260603C00225000"
    assert [action["action"] for action in actions] == [
        "max_total_unrealized_profit_triggered",
        "total_profit_position_close_triggered",
        "cancelled_existing_exit_for_total_profit",
        "submitted_total_profit_close",
    ]


def test_positions_for_total_loss_close_filters_losing_only() -> None:
    positions = [
        {"symbol": "LOSS", "unrealized_pl": "-10"},
        {"symbol": "WIN", "unrealized_pl": "5"},
    ]

    assert [row["symbol"] for row in _positions_for_total_loss_close(positions, "losing_only")] == ["LOSS"]
    assert [row["symbol"] for row in _positions_for_total_loss_close(positions, "all")] == ["LOSS", "WIN"]


def test_positions_for_total_profit_close_filters_winning_only() -> None:
    positions = [
        {"symbol": "LOSS", "unrealized_pl": "-10"},
        {"symbol": "WIN", "unrealized_pl": "5"},
    ]

    assert [row["symbol"] for row in _positions_for_total_profit_close(positions, "winning_only")] == ["WIN"]
    assert [row["symbol"] for row in _positions_for_total_profit_close(positions, "all")] == ["LOSS", "WIN"]


def test_manual_liquidate_and_stop_closes_every_ticker_position() -> None:
    broker = FakeBroker()

    actions = _liquidate_option_positions(
        broker=broker,  # type: ignore[arg-type]
        args=_args(ticker="NVDA"),
        open_orders=[
            {"id": "stop-1", "symbol": "NVDA260603C00225000", "side": "sell"},
            {"id": "stop-2", "symbol": "NVDA260605C00225000", "side": "sell"},
        ],
        option_positions=[
            {"symbol": "NVDA260603C00225000", "qty": "3", "current_price": "1.80", "unrealized_pl": "-135"},
            {"symbol": "NVDA260605C00225000", "qty": "2", "current_price": "2.87", "unrealized_pl": "18"},
        ],
        now=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
    )

    assert broker.cancelled == ["stop-1", "stop-2"]
    first_id = broker.submitted[0].pop("client_order_id")
    second_id = broker.submitted[1].pop("client_order_id")
    assert first_id.startswith("opt-emergency-NVDA-VDA260603C00225000-20260601140000-")
    assert second_id.startswith("opt-emergency-NVDA-VDA260605C00225000-20260601140000-")
    assert first_id != second_id
    assert broker.submitted == [
        {
            "symbol": "NVDA260603C00225000",
            "side": "sell",
            "order_type": "limit",
            "qty": 3,
            "limit_price": 1.71,
            "time_in_force": "day",
        },
        {
            "symbol": "NVDA260605C00225000",
            "side": "sell",
            "order_type": "limit",
            "qty": 2,
            "limit_price": 2.73,
            "time_in_force": "day",
        },
    ]
    assert [action["action"] for action in actions] == [
        "manual_liquidate_and_stop_triggered",
        "manual_liquidation_position_close_triggered",
        "cancelled_existing_exit_for_manual_liquidation",
        "submitted_manual_liquidation_close",
        "manual_liquidation_position_close_triggered",
        "cancelled_existing_exit_for_manual_liquidation",
        "submitted_manual_liquidation_close",
    ]


def test_verify_liquidation_retries_and_reports_flat() -> None:
    broker = FakePollingBroker(
        positions_by_poll=[
            [{"symbol": "NVDA260603C00225000", "qty": "1", "current_price": "2.00", "unrealized_pl": "10"}],
            [],
        ],
        orders_by_poll=[
            [{"id": "exit-1", "symbol": "NVDA260603C00225000", "side": "sell", "type": "limit"}],
            [],
        ],
    )

    result = verify_liquidation_until_flat(
        broker=broker,  # type: ignore[arg-type]
        args=_args(ticker="NVDA", liquidation_wait_seconds=0, liquidation_poll_seconds=0.5),
        started_at=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
    )

    assert result["flat"] is True
    assert [action["action"] for action in result["actions"]] == [
        "liquidation_retry_triggered",
        "cancelled_open_order_for_liquidation_retry",
        "manual_liquidate_and_stop_triggered",
        "manual_liquidation_position_close_triggered",
        "submitted_manual_liquidation_close",
        "liquidation_verified_flat",
    ]
    assert broker.cancelled == ["exit-1"]
    assert broker.submitted[0]["limit_price"] == 1.7


def test_verify_liquidation_reports_not_flat_after_timeout() -> None:
    broker = FakePollingBroker(
        positions_by_poll=[
            [{"symbol": "NVDA260603C00225000", "qty": "1", "current_price": "2.00", "unrealized_pl": "10"}],
            [{"symbol": "NVDA260603C00225000", "qty": "1", "current_price": "2.00", "unrealized_pl": "10"}],
        ],
        orders_by_poll=[
            [{"id": "exit-1", "symbol": "NVDA260603C00225000", "side": "sell", "type": "limit"}],
            [{"id": "exit-2", "symbol": "NVDA260603C00225000", "side": "sell", "type": "limit"}],
        ],
    )

    result = verify_liquidation_until_flat(
        broker=broker,  # type: ignore[arg-type]
        args=_args(ticker="NVDA", liquidation_wait_seconds=0, liquidation_poll_seconds=0.5),
        started_at=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
    )

    assert result["flat"] is False
    assert result["actions"][-1]["action"] == "liquidation_not_flat_after_timeout"
    assert result["actions"][-1]["remaining_position_count"] == 1


def test_verify_liquidation_cancels_remaining_orders_before_reporting_flat() -> None:
    broker = FakePollingBroker(
        positions_by_poll=[[], []],
        orders_by_poll=[
            [{"id": "buy-entry", "symbol": "NVDA260603C00225000", "side": "buy", "type": "limit"}],
            [],
        ],
    )

    result = verify_liquidation_until_flat(
        broker=broker,  # type: ignore[arg-type]
        args=_args(ticker="NVDA", liquidation_wait_seconds=0, liquidation_poll_seconds=0.5),
        started_at=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
    )

    assert result["flat"] is True
    assert [action["action"] for action in result["actions"]] == [
        "liquidation_cancel_remaining_open_orders",
        "cancelled_remaining_open_order_for_liquidation",
        "liquidation_verified_flat",
    ]
    assert broker.cancelled == ["buy-entry"]


def test_cancel_open_option_orders_cancels_buy_and_sell_orders() -> None:
    broker = FakeBroker()

    actions = _cancel_open_option_orders(
        broker=broker,  # type: ignore[arg-type]
        args=_args(ticker="TSLA"),
        open_orders=[
            {"id": "buy-entry", "symbol": "TSLA260605P00412500", "side": "buy", "type": "limit"},
            {"id": "sell-stop", "symbol": "TSLA260605P00417500", "side": "sell", "type": "stop_limit"},
        ],
        cancel_action="cancelled_open_order_for_liquidate_and_stop",
    )

    assert broker.cancelled == ["buy-entry", "sell-stop"]
    assert [action["action"] for action in actions] == [
        "cancelled_open_order_for_liquidate_and_stop",
        "cancelled_open_order_for_liquidate_and_stop",
    ]


def test_execution_blocks_when_position_count_or_exposure_is_full() -> None:
    reasons = execution_block_reasons(
        args=_args(
            max_open_option_orders=3,
            max_open_option_positions=2,
            max_open_option_exposure=1000.0,
            require_market_open=True,
            allow_duplicate_contract_order=False,
        ),
        clock={"is_open": True},
        trade_plan={
            "action": "buy_option",
            "order": {"symbol": "NVDA260603C00225000", "qty": 1, "limit_price": 3.25},
        },
        open_orders=[],
        option_positions=[
            {"symbol": "NVDA260603C00220000", "qty": "1", "market_value": "700"},
            {"symbol": "NVDA260603C00230000", "qty": "1", "market_value": "450"},
        ],
    )

    assert reasons == [
        "max_open_option_positions_reached",
        "max_open_option_exposure_reached",
    ]


def test_execution_blocks_when_contract_cap_or_mixed_direction_reached() -> None:
    reasons = execution_block_reasons(
        args=_args(
            max_open_option_orders=3,
            max_open_option_positions=4,
            max_open_option_contracts=2,
            max_open_option_exposure=5000.0,
            require_market_open=True,
            allow_duplicate_contract_order=False,
            allow_mixed_option_direction=False,
        ),
        clock={"is_open": True},
        trade_plan={
            "action": "buy_option",
            "order": {"symbol": "NVDA260603P00225000", "qty": 1, "limit_price": 3.25},
        },
        open_orders=[],
        option_positions=[
            {"symbol": "NVDA260603C00220000", "qty": "2", "market_value": "700"},
        ],
    )

    assert reasons == [
        "max_open_option_contracts_reached",
        "mixed_option_direction_blocked",
    ]


def test_position_unrealized_loss_guard_closes_single_position() -> None:
    broker = FakeBroker()
    actions = _manage_position_unrealized_guards(
        broker=broker,  # type: ignore[arg-type]
        args=_args(ticker="NVDA", max_position_unrealized_loss=75.0),
        open_orders=[{"id": "old-stop", "symbol": "NVDA260603C00225000", "side": "sell"}],
        option_positions=[
            {
                "symbol": "NVDA260603C00225000",
                "qty": "1",
                "current_price": "2.25",
                "unrealized_pl": "-90",
            }
        ],
        now=datetime(2026, 6, 1, 16, 0, tzinfo=UTC),
    )

    assert [action["action"] for action in actions] == [
        "position_loss_position_close_triggered",
        "cancelled_existing_exit_for_position_loss_position_close_triggered",
        "submitted_position_loss_position_close_triggered_close",
    ]
    assert broker.cancelled == ["old-stop"]
    assert broker.submitted[0]["symbol"] == "NVDA260603C00225000"
    assert broker.submitted[0]["order_type"] == "limit"


def test_forecast_reversal_exit_closes_opposite_option_type() -> None:
    broker = FakeBroker()
    actions = _manage_forecast_reversal_exit(
        broker=broker,  # type: ignore[arg-type]
        args=_args(ticker="TSLA"),
        open_orders=[],
        option_positions=[
            {
                "symbol": "TSLA260605C00425000",
                "qty": "1",
                "current_price": "5.10",
                "unrealized_pl": "55",
            }
        ],
        now=datetime(2026, 6, 1, 16, 0, tzinfo=UTC),
        forecast={"expected_direction": "downward", "spot": 425.0, "predicted_price": 420.0},
        underlying_price=425.0,
    )

    assert actions[0]["action"] == "forecast_reversal_position_close_triggered"
    assert actions[0]["position_option_type"] == "call"
    assert actions[0]["desired_option_type"] == "put"
    assert broker.submitted[0]["symbol"] == "TSLA260605C00425000"


def test_expiry_risk_closes_near_expiry_position() -> None:
    broker = FakeBroker()
    actions = _manage_expiry_risk(
        broker=broker,  # type: ignore[arg-type]
        args=_args(ticker="TSLA", close_before_expiry_hours=2.0, expiry_warning_hours=24.0),
        open_orders=[],
        option_positions=[
            {
                "symbol": "TSLA260605P00425000",
                "qty": "1",
                "current_price": "3.20",
                "unrealized_pl": "-20",
            }
        ],
        now=datetime(2026, 6, 5, 18, 30, tzinfo=UTC),
    )

    assert actions[0]["action"] == "expiry_position_close_triggered"
    assert actions[0]["hours_to_expiry"] == 1.5
    assert broker.submitted[0]["symbol"] == "TSLA260605P00425000"


def test_option_entry_guard_blocks_overtrading() -> None:
    now = datetime(2026, 6, 1, 16, 0, tzinfo=UTC)
    args = _args(
        risk_profile="aggressive",
        entry_cooldown_minutes=15,
        loss_cooldown_minutes=30,
        max_trades_per_day=1,
        max_consecutive_losses=1,
        one_trade_per_forecast=True,
    )
    closed_orders = [
        {
            "symbol": "TSLA260603P00420000",
            "side": "buy",
            "filled_qty": "1",
            "filled_avg_price": "4.00",
            "filled_at": "2026-06-01T15:50:00+00:00",
        },
        {
            "symbol": "TSLA260603P00420000",
            "side": "sell",
            "filled_qty": "1",
            "filled_avg_price": "3.50",
            "filled_at": "2026-06-01T15:55:00+00:00",
        },
    ]
    forecast_bundle = {"created_at_utc": "2026-06-01T15:45:00+00:00", "selected_forecast": {"horizon_hours": 1.0}}

    guard = option_entry_guard_reasons(
        args=args,
        closed_orders=closed_orders,
        state={"traded_forecast_keys": ["2026-06-01T15:45:00+00:00|1.0"]},
        forecast_bundle=forecast_bundle,
        now=now,
    )

    assert guard["reasons"] == [
        "entry_cooldown_active",
        "loss_cooldown_active",
        "max_trades_per_day_reached",
        "max_consecutive_losses_reached",
        "one_trade_per_forecast_used",
    ]


def test_option_entry_guard_blocks_after_daily_realized_loss_limit() -> None:
    now = datetime(2026, 6, 1, 16, 0, tzinfo=UTC)
    args = _args(
        risk_profile="aggressive",
        entry_cooldown_minutes=0,
        loss_cooldown_minutes=0,
        max_trades_per_day=10,
        max_consecutive_losses=10,
        max_realized_loss_per_day=100,
        one_trade_per_forecast=False,
    )
    closed_orders = [
        {
            "symbol": "TSLA260603P00420000",
            "side": "buy",
            "filled_qty": "1",
            "filled_avg_price": "4.00",
            "filled_at": "2026-06-01T15:00:00+00:00",
        },
        {
            "symbol": "TSLA260603P00420000",
            "side": "sell",
            "filled_qty": "1",
            "filled_avg_price": "2.75",
            "filled_at": "2026-06-01T15:30:00+00:00",
        },
    ]

    guard = option_entry_guard_reasons(
        args=args,
        closed_orders=closed_orders,
        state={},
        forecast_bundle={"created_at_utc": "2026-06-01T15:45:00+00:00", "selected_forecast": {"horizon_hours": 1.0}},
        now=now,
    )

    assert guard["metrics"]["realized_pnl_today"] == -125.0
    assert guard["reasons"] == ["max_realized_loss_per_day_reached"]


def test_entry_guard_waits_until_next_forecast_after_forced_close() -> None:
    now = datetime(2026, 6, 1, 16, 0, tzinfo=UTC)
    args = _args(
        risk_profile="aggressive",
        entry_cooldown_minutes=0,
        loss_cooldown_minutes=0,
        max_trades_per_day=10,
        max_consecutive_losses=10,
        one_trade_per_forecast=False,
    )
    forecast_bundle = {"created_at_utc": "2026-06-01T15:45:00+00:00", "selected_forecast": {"horizon_hours": 1.0}}

    guard = option_entry_guard_reasons(
        args=args,
        closed_orders=[],
        state={"wait_until_next_forecast_after_close": {"forecast_created_at_utc": "2026-06-01T15:45:00+00:00"}},
        forecast_bundle=forecast_bundle,
        now=now,
    )

    assert guard["reasons"] == ["waiting_until_next_forecast_after_close"]


def test_clear_trade_pause_state_removes_stop_and_local_cooldowns(tmp_path) -> None:
    write_stop_request(tmp_path, "NVDA", reason="manual_stop_request")
    write_state(
        tmp_path,
        "NVDA",
        {
            "wait_until_next_forecast_after_close": {"forecast_created_at_utc": "2026-06-01T15:45:00+00:00"},
            "traded_forecast_keys": ["2026-06-01T15:45:00+00:00|1.0"],
            "active_trade": {"status": "entry_submitted"},
            "last_forecast": {"created_at_utc": "2026-06-01T15:45:00+00:00"},
        },
    )

    result = clear_trade_pause_state(tmp_path, "NVDA")
    state = read_state(tmp_path, "NVDA")

    assert result["cleared_stop_request"] is True
    assert set(result["cleared_state_keys"]) == {
        "wait_until_next_forecast_after_close",
        "traded_forecast_keys",
        "active_trade",
    }
    assert "manual_trade_resume_at_utc" in state
    assert "wait_until_next_forecast_after_close" not in state
    assert "traded_forecast_keys" not in state
    assert "active_trade" not in state
    assert state["last_forecast"]["created_at_utc"] == "2026-06-01T15:45:00+00:00"


def test_manual_resume_overrides_entry_and_loss_cooldowns_from_prior_fills() -> None:
    now = datetime(2026, 6, 1, 16, 0, tzinfo=UTC)
    args = _args(
        risk_profile="aggressive",
        entry_cooldown_minutes=15,
        loss_cooldown_minutes=30,
        max_trades_per_day=10,
        max_consecutive_losses=10,
        one_trade_per_forecast=False,
    )
    closed_orders = [
        {
            "symbol": "TSLA260603P00420000",
            "side": "buy",
            "filled_qty": "1",
            "filled_avg_price": "4.00",
            "filled_at": "2026-06-01T15:50:00+00:00",
        },
        {
            "symbol": "TSLA260603P00420000",
            "side": "sell",
            "filled_qty": "1",
            "filled_avg_price": "3.50",
            "filled_at": "2026-06-01T15:55:00+00:00",
        },
    ]
    forecast_bundle = {"created_at_utc": "2026-06-01T15:45:00+00:00", "selected_forecast": {"horizon_hours": 1.0}}

    guard = option_entry_guard_reasons(
        args=args,
        closed_orders=closed_orders,
        state={"manual_trade_resume_at_utc": "2026-06-01T15:56:00+00:00"},
        forecast_bundle=forecast_bundle,
        now=now,
    )

    assert "entry_cooldown_active" not in guard["reasons"]
    assert "loss_cooldown_active" not in guard["reasons"]
