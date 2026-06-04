from __future__ import annotations

from argparse import Namespace

import pandas as pd

from market_forecasting_engine.deribit_broker import DeribitLiveSpotBroker
from market_forecasting_engine.live_trading.deribit_spot_agent import (
    analyze_existing_protection,
    attach_rolling_validation,
    book_quote,
    decide_spot_trade,
    execute_plan,
    normalize_crypto_forecast_timestamps,
)


def _args(**overrides):
    defaults = {
        "instrument": "ETH_USDC",
        "risk_profile": "aggressive",
        "min_edge_pct": 0.001,
        "max_spread_pct": 0.01,
        "max_notional_usdc": 25.0,
        "min_order_base_amount": 0.0001,
        "max_base_position": 0.05,
        "take_profit_pct": 0.08,
        "stop_loss_pct": 0.04,
        "enable_pullback_buy": True,
        "pullback_min_reward_risk": 1.25,
        "pullback_min_reversal_probability": 0.45,
        "pullback_max_entry_distance_pct": 0.05,
        "enable_scale_in_pullback": True,
        "spot_average_entry_price": None,
        "allow_scale_in_without_entry_price": False,
        "scale_in_min_discount_from_entry_pct": 0.02,
        "scale_in_max_existing_loss_pct": 0.06,
        "scale_in_max_addition_fraction": 0.35,
        "scale_in_max_notional_usdc": None,
        "respect_existing_protection": True,
        "protected_position_coverage_ratio": 0.95,
        "sell_now_min_forecast_beyond_stop_pct": 0.01,
        "tighten_stop_buffer_pct": 0.003,
        "inventory_scope": "codex_only",
        "managed_base_balance": None,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_live_spot_decision_buys_on_bullish_forecast_with_sizing_and_protection() -> None:
    plan = decide_spot_trade(
        args=_args(),
        base_account={"balance": 0.0},
        quote_account={"available_funds": 30.0},
        quote={"bid": 1749.0, "ask": 1751.0, "mid": 1750.0},
        latest_price=1750.0,
        forecast={"predicted_price": 1760.0, "expected_direction": "Upward"},
        open_orders=[],
    )

    assert plan["action"] == "buy_spot"
    assert plan["entry_order"]["side"] == "buy"
    assert plan["entry_order"]["type"] == "limit"
    assert plan["entry_order"]["amount"] > 0
    assert plan["protection"] == {}
    assert plan["post_fill_protection_plan"]["take_profit"]["type"] == "limit"
    assert plan["post_fill_protection_plan"]["stop_loss"]["type"] == "stop_market"
    assert plan["post_fill_protection_plan"]["stop_loss"]["trigger"] == "index_price"


def test_live_spot_decision_sells_existing_base_on_bearish_forecast() -> None:
    plan = decide_spot_trade(
        args=_args(),
        base_account={"balance": 0.02},
        quote_account={"available_funds": 5.0},
        managed_base_balance=0.02,
        quote={"bid": 1749.0, "ask": 1751.0, "mid": 1750.0},
        latest_price=1750.0,
        forecast={"predicted_price": 1740.0, "expected_direction": "Downward"},
        open_orders=[],
    )

    assert plan["action"] == "sell_spot"
    assert plan["entry_order"] == {"side": "sell", "type": "limit", "amount": 0.02, "price": 1747.25}


def test_manual_existing_base_is_not_sold_by_default() -> None:
    plan = decide_spot_trade(
        args=_args(),
        base_account={"balance": 0.02},
        quote_account={"available_funds": 5.0},
        managed_base_balance=0.0,
        quote={"bid": 1749.0, "ask": 1751.0, "mid": 1750.0},
        latest_price=1750.0,
        forecast={"predicted_price": 1740.0, "expected_direction": "Downward"},
        open_orders=[],
    )

    assert plan["action"] == "hold"
    assert plan["reason"] == "manual_inventory_not_managed_by_default"
    assert plan["inventory_scope"]["manual_base_balance"] == 0.02


def test_existing_protection_makes_bearish_forecast_hold_instead_of_sell() -> None:
    plan = decide_spot_trade(
        args=_args(),
        base_account={"balance": 0.02},
        quote_account={"available_funds": 5.0},
        managed_base_balance=0.02,
        quote={"bid": 1749.0, "ask": 1751.0, "mid": 1750.0},
        latest_price=1750.0,
        forecast={"predicted_price": 1740.0, "expected_direction": "Downward"},
        open_orders=[
            {"instrument_name": "ETH_USDC", "direction": "sell", "order_type": "stop_market", "order_state": "untriggered", "amount": 0.02, "filled_amount": 0.0, "trigger_price": 1700.0},
            {"instrument_name": "ETH_USDC", "direction": "sell", "order_type": "limit", "order_state": "open", "amount": 0.02, "filled_amount": 0.0, "price": 1900.0},
        ],
    )

    assert plan["action"] == "hold_with_existing_protection"
    assert plan["existing_protection"]["stop_coverage_ratio"] == 1.0
    assert plan["protection_decision"]["decision"] == "respect_existing_protection"


def test_material_break_below_existing_stop_still_sells_now() -> None:
    plan = decide_spot_trade(
        args=_args(),
        base_account={"balance": 0.02},
        quote_account={"available_funds": 5.0},
        managed_base_balance=0.02,
        quote={"bid": 1749.0, "ask": 1751.0, "mid": 1750.0},
        latest_price=1750.0,
        forecast={"predicted_price": 1670.0, "expected_direction": "Downward"},
        open_orders=[
            {"instrument_name": "ETH_USDC", "direction": "sell", "order_type": "stop_market", "order_state": "untriggered", "amount": 0.02, "filled_amount": 0.0, "trigger_price": 1700.0},
        ],
    )

    assert plan["action"] == "sell_spot"
    assert plan["reason"] == "bearish_forecast_breaks_materially_below_existing_stop"


def test_incomplete_existing_protection_plans_replacement() -> None:
    plan = decide_spot_trade(
        args=_args(),
        base_account={"balance": 0.02},
        quote_account={"available_funds": 5.0},
        managed_base_balance=0.02,
        quote={"bid": 1749.0, "ask": 1751.0, "mid": 1750.0},
        latest_price=1750.0,
        forecast={"predicted_price": 1740.0, "expected_direction": "Downward"},
        open_orders=[
            {"instrument_name": "ETH_USDC", "direction": "sell", "order_type": "stop_market", "order_state": "untriggered", "amount": 0.005, "filled_amount": 0.0, "trigger_price": 1700.0},
        ],
    )

    assert plan["action"] == "protect_existing_position"
    assert plan["reason"] == "existing_protection_incomplete_replace_or_add"


def test_open_buy_order_does_not_block_bearish_sell_of_existing_base() -> None:
    plan = decide_spot_trade(
        args=_args(),
        base_account={"balance": 0.02},
        quote_account={"available_funds": 5.0},
        managed_base_balance=0.02,
        quote={"bid": 1749.0, "ask": 1751.0, "mid": 1750.0},
        latest_price=1750.0,
        forecast={"predicted_price": 1740.0, "expected_direction": "Downward"},
        open_orders=[{"instrument_name": "ETH_USDC", "direction": "buy", "order_state": "untriggered"}],
    )

    assert plan["action"] == "sell_spot"


def test_existing_eth_can_scale_in_on_strict_lower_pullback_setup() -> None:
    plan = decide_spot_trade(
        args=_args(spot_average_entry_price=1770.0, max_base_position=0.05),
        base_account={"balance": 0.02},
        quote_account={"available_funds": 30.0},
        managed_base_balance=0.02,
        quote={"bid": 1749.0, "ask": 1751.0, "mid": 1750.0},
        latest_price=1750.0,
        forecast={"predicted_price": 1735.0, "expected_direction": "Downward"},
        open_orders=[],
        forecast_bundle={
            "mean_reversion_dip_buy": {
                "best_setup": {
                    "setup": "conditional_dip_buy",
                    "entry_price": 1715.0,
                    "stop_price": 1690.0,
                    "target_price": 1760.0,
                    "reward_risk": 1.8,
                    "reversal_probability": 0.58,
                    "allowed": True,
                }
            }
        },
    )

    assert plan["action"] == "place_scale_in_pullback_buy"
    assert plan["entry_order"] == {"side": "buy", "type": "limit", "amount": 0.006999, "price": 1715.0}
    assert plan["scale_in_pullback"]["average_entry_price"] == 1770.0
    assert plan["post_fill_protection_plan"]["stop_loss"]["trigger_price"] == 1690.0


def test_existing_eth_scale_in_requires_average_entry_price_by_default() -> None:
    plan = decide_spot_trade(
        args=_args(),
        base_account={"balance": 0.02},
        quote_account={"available_funds": 30.0},
        managed_base_balance=0.02,
        quote={"bid": 1749.0, "ask": 1751.0, "mid": 1750.0},
        latest_price=1750.0,
        forecast={"predicted_price": 1735.0, "expected_direction": "Downward"},
        open_orders=[],
        forecast_bundle={
            "mean_reversion_dip_buy": {
                "best_setup": {
                    "entry_price": 1715.0,
                    "stop_price": 1690.0,
                    "target_price": 1760.0,
                    "reward_risk": 1.8,
                    "reversal_probability": 0.58,
                    "allowed": True,
                }
            }
        },
    )

    assert plan["action"] == "sell_spot"
    assert "missing_existing_average_entry_price" in plan["scale_in_pullback"]["blocking_reasons"]


def test_existing_eth_scale_in_blocks_when_loss_is_too_large() -> None:
    plan = decide_spot_trade(
        args=_args(spot_average_entry_price=1900.0),
        base_account={"balance": 0.02},
        quote_account={"available_funds": 30.0},
        managed_base_balance=0.02,
        quote={"bid": 1749.0, "ask": 1751.0, "mid": 1750.0},
        latest_price=1750.0,
        forecast={"predicted_price": 1735.0, "expected_direction": "Downward"},
        open_orders=[],
        forecast_bundle={
            "mean_reversion_dip_buy": {
                "best_setup": {
                    "entry_price": 1715.0,
                    "stop_price": 1690.0,
                    "target_price": 1760.0,
                    "reward_risk": 1.8,
                    "reversal_probability": 0.58,
                    "allowed": True,
                }
            }
        },
    )

    assert plan["action"] == "sell_spot"
    assert "existing_position_loss_too_large_to_average_down" in plan["scale_in_pullback"]["blocking_reasons"]


def test_bearish_forecast_can_place_lower_pullback_buy_without_existing_base() -> None:
    plan = decide_spot_trade(
        args=_args(),
        base_account={"balance": 0.0},
        quote_account={"available_funds": 30.0},
        quote={"bid": 1749.0, "ask": 1751.0, "mid": 1750.0},
        latest_price=1750.0,
        forecast={"predicted_price": 1735.0, "expected_direction": "Downward"},
        open_orders=[],
        forecast_bundle={
            "mean_reversion_dip_buy": {
                "best_setup": {
                    "setup": "conditional_dip_buy",
                    "entry_price": 1715.0,
                    "stop_price": 1690.0,
                    "target_price": 1760.0,
                    "reward_risk": 1.8,
                    "reversal_probability": 0.58,
                    "allowed": True,
                }
            }
        },
    )

    assert plan["action"] == "place_pullback_buy"
    assert plan["entry_order"] == {"side": "buy", "type": "limit", "amount": 0.014577, "price": 1715.0}
    assert plan["protection"] == {}
    assert plan["post_fill_protection_plan"]["stop_loss"]["trigger_price"] == 1690.0
    assert plan["post_fill_protection_plan"]["take_profit"]["price"] == 1760.0


def test_pullback_buy_blocks_with_visible_reasons_when_setup_is_weak() -> None:
    plan = decide_spot_trade(
        args=_args(),
        base_account={"balance": 0.0},
        quote_account={"available_funds": 30.0},
        quote={"bid": 1749.0, "ask": 1751.0, "mid": 1750.0},
        latest_price=1750.0,
        forecast={"predicted_price": 1735.0, "expected_direction": "Downward"},
        open_orders=[],
        forecast_bundle={
            "mean_reversion_dip_buy": {
                "best_setup": {
                    "entry_price": 1600.0,
                    "stop_price": 1590.0,
                    "target_price": 1610.0,
                    "reward_risk": 1.0,
                    "reversal_probability": 0.30,
                    "allowed": False,
                }
            }
        },
    )

    assert plan["action"] == "hold"
    assert plan["reason"] == "pullback_buy_blocked"
    assert "dip_buy_setup_not_allowed" in plan["pullback_buy"]["blocking_reasons"]
    assert "pullback_entry_too_far_below_market" in plan["pullback_buy"]["blocking_reasons"]


def test_book_quote_uses_top_bid_ask_mid() -> None:
    assert book_quote({"bids": [[100, 1]], "asks": [[102, 1]]}) == {"bid": 100.0, "ask": 102.0, "mid": 101.0}


class PayloadBroker(DeribitLiveSpotBroker):
    def __init__(self):
        self.account_mode = "live"
        self.base_url = "https://www.deribit.com/api/v2"
        self.client_id = "id"
        self.client_secret = "secret"
        self._access_token = "token"
        self._access_token_expires_at = 9999999999.0
        self.calls = []

    def private_get(self, method, params=None):
        self.calls.append((method, params))
        return {"ok": True, "method": method, "params": params}


def test_live_spot_broker_submits_stop_market_trigger_payload() -> None:
    broker = PayloadBroker()
    result = broker.submit_spot_order(
        side="sell",
        instrument_name="ETH_USDC",
        amount=0.01,
        order_type="stop_market",
        trigger_price=1700.0,
        trigger="index_price",
        label="test",
    )

    assert result["ok"] is True
    method, params = broker.calls[0]
    assert method == "sell"
    assert params["instrument_name"] == "ETH_USDC"
    assert params["type"] == "stop_market"
    assert params["trigger_price"] == 1700.0
    assert params["trigger"] == "index_price"


def test_execute_plan_does_not_submit_protection_for_unfilled_buy_entry() -> None:
    broker = PayloadBroker()
    plan = {
        "instrument": "ETH_USDC",
        "action": "place_pullback_buy",
        "entry_order": {"side": "buy", "type": "limit", "amount": 0.01, "price": 1700.0},
        "protection": {},
        "post_fill_protection_plan": {
            "stop_loss": {"side": "sell", "type": "stop_market", "amount": 0.01, "trigger_price": 1660.0, "trigger": "index_price"},
            "take_profit": {"side": "sell", "type": "limit", "amount": 0.01, "price": 1760.0},
        },
    }

    results = execute_plan(broker=broker, plan=plan, open_orders=[], label_base="test", replace_protection=True)

    assert [call[0] for call in broker.calls] == ["buy"]
    assert results[-1]["action"] == "post_fill_protection_pending"


def test_rolling_validation_attaches_metrics_per_horizon() -> None:
    index = pd.date_range("2026-06-01", periods=180, freq="min")
    prices = pd.DataFrame({"close": [1700 + i * 0.2 for i in range(180)]}, index=index)
    forecasts = [{"horizon_bars": 15}, {"horizon_bars": 30}]

    attach_rolling_validation(forecasts, prices, max_samples=40)

    assert forecasts[0]["validation_metrics"]["status"] == "measured"
    assert forecasts[0]["validation_metrics"]["sample_count"] > 0
    assert forecasts[0]["validation_metrics"]["price_mae"] is not None
    assert forecasts[1]["validation_metrics"]["directional_accuracy"] >= 0.0


def test_crypto_forecast_timestamps_use_continuous_clock_not_equity_session_gap() -> None:
    plan = {
        "as_of": "2026-06-04T10:59:00",
        "forecasts": [
            {"horizon_hours": 0.25, "forecast_timestamp": "2026-06-05T00:14:00"},
            {"horizon_hours": 1.0, "forecast_timestamp": "2026-06-05T00:59:00"},
        ],
    }

    normalize_crypto_forecast_timestamps(plan)

    assert plan["forecasts"][0]["forecast_timestamp"] == "2026-06-04T11:14:00"
    assert plan["forecasts"][1]["forecast_timestamp"] == "2026-06-04T11:59:00"
    assert plan["forecasts"][0]["timestamp_policy"] == "continuous_crypto_clock"
