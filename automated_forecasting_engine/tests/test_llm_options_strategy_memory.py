from __future__ import annotations

from datetime import UTC, datetime, timedelta

from market_forecasting_engine.llm_options_trader.chronos_forecast import _short_horizon_signal_points
from market_forecasting_engine.llm_options_trader.common import (
    apply_forecast_validation_to_transition,
    compact_market_packet,
    option_tradeability_summary,
    regime_transition_warning,
    short_tape_summary,
    trend_carry_context,
    multi_window_trend_carry_context,
)
from market_forecasting_engine.llm_options_trader.forecast_ledger import update_forecast_ledger
from market_forecasting_engine.llm_options_trader.forecast_ledger import forecast_error_feedback
from market_forecasting_engine.llm_options_trader.memory import load_strategy_memory, update_strategy_memory_from_record


def test_strategy_memory_learns_from_closed_shadow_loss(tmp_path) -> None:
    record = {
        "checked_at_utc": "2026-06-06T21:00:00+00:00",
        "llm_decision": {"action": "hold", "intent": "hold", "reason": "market chopped against prior put"},
        "order_result": {"submitted": False},
        "market_packet": {
            "price_summary": {"latest_close": 1558.0, "return_1h": -0.01, "return_4h": 0.0},
            "technical_observations": {"trend": "chop"},
            "shadow_simulation": {
                "positions": [
                    {
                        "instrument_name": "ETH_USDC-8JUN26-1550-P",
                        "size": 0,
                        "realized_pnl": -0.34,
                        "unrealized_pnl": 0,
                    }
                ],
                "profit_protection_audit": {},
            },
        },
    }

    updated = update_strategy_memory_from_record(tmp_path, "ETH", record)
    loaded = load_strategy_memory(tmp_path, "ETH")

    assert updated["lesson_count"] >= 1
    assert loaded["status"] == "ok"
    assert any(item["id"] == "closed_loss_put" for item in loaded["lessons"])
    assert "Recent simulated put positions closed at a loss" in loaded["summary"]


def test_short_horizon_signal_is_not_flat_when_recent_prices_move() -> None:
    now = datetime(2026, 6, 6, 21, 0, tzinfo=UTC)
    prices = [1550 + ((index % 6) - 3) * 0.8 + index * 0.08 for index in range(120)]
    series = [(now + timedelta(minutes=index), price) for index, price in enumerate(prices)]

    points = _short_horizon_signal_points(series=series, forecast_hours=(0.083333, 0.166667, 0.25, 0.5), interval_minutes=1)
    predicted = [point["predicted_price"] for point in points]

    assert len(points) == 4
    assert max(predicted) - min(predicted) > 0.01
    assert all(point["signal_components"]["local_std_30"] > 0 for point in points)


def test_forecast_ledger_matures_prior_points_against_real_prices(tmp_path) -> None:
    start = datetime(2026, 6, 6, 21, 0, tzinfo=UTC)
    first_bars = [
        {"timestamp": (start + timedelta(minutes=index)).isoformat(), "close": 100.0 + index * 0.1}
        for index in range(10)
    ]
    forecast = {
        "status": "ok",
        "as_of_timestamp": first_bars[-1]["timestamp"],
        "as_of_price": 100.9,
        "preferred_source": "unit_signal",
        "preferred_horizon_points": [
            {
                "horizon_hours": 0.083333,
                "timestamp": (start + timedelta(minutes=14)).isoformat(),
                "predicted_price": 102.0,
                "lower_price": 101.0,
                "upper_price": 103.0,
                "direction": "up",
            }
        ],
    }

    pending = update_forecast_ledger(output_dir=tmp_path, currency="ETH", forecast=forecast, price_bars=first_bars)
    assert pending["pending_count"] == 1
    assert pending["matured_count"] == 0

    later_bars = first_bars + [
        {"timestamp": (start + timedelta(minutes=index)).isoformat(), "close": 101.5}
        for index in range(10, 16)
    ]
    matured = update_forecast_ledger(output_dir=tmp_path, currency="ETH", forecast=forecast, price_bars=later_bars)

    assert matured["matured_count"] == 1
    assert matured["recent_matured"][-1]["actual_price"] == 101.5
    assert matured["recent_matured"][-1]["error"] == -0.5
    assert "0.083333h" in matured["by_horizon"]
    assert matured["error_feedback"]["status"] == "ok"
    assert "instruction" in matured["error_feedback"]


def test_forecast_error_feedback_teaches_llm_to_adjust_biased_forecasts() -> None:
    feedback = forecast_error_feedback(
        [
            {"error": 2.0, "abs_error": 2.0, "direction_correct": False},
            {"error": 1.0, "abs_error": 1.0, "direction_correct": False},
            {"error": 1.5, "abs_error": 1.5, "direction_correct": True},
        ]
    )

    assert feedback["bias"] == "under_predicted_actual_price"
    assert feedback["reliability"] == "poor_directional_reliability"
    assert "under-read actual price" in feedback["instruction"]
    assert "rejecting calls" in feedback["instruction"]


def test_llm_packet_includes_tape_transition_and_tradeability_inputs() -> None:
    start = datetime(2026, 6, 6, 21, 0, tzinfo=UTC)
    rows = []
    price = 100.0
    for index in range(75):
        price += 0.02 if index < 60 else 0.25
        rows.append(
            {
                "timestamp": start + timedelta(minutes=index),
                "open": price - 0.05,
                "high": price + 0.15,
                "low": price - 0.15,
                "close": price,
                "volume": 10 + index,
            }
        )
    import pandas as pd

    prices = pd.DataFrame(rows).set_index("timestamp")
    tape = short_tape_summary(prices)
    transition = regime_transition_warning(prices, forecast_validation={})
    enriched = apply_forecast_validation_to_transition(
        transition,
        {
            "summary": "Matured forecasts under-read price.",
            "recent_matured": [{"error": 2.0}, {"error": 1.5}],
            "by_horizon": {"0.083333h": {"directional_accuracy": 0.25}},
        },
    )
    tradeability = option_tradeability_summary(
        [
            {
                "instrument_name": "ETH_USDC-8JUN26-100-C",
                "option_type": "call",
                "bid": 4.8,
                "ask": 5.2,
                "mid": 5.0,
                "spread_pct": 0.08,
                "best_bid_amount": 1,
                "best_ask_amount": 1,
                "volume": 5,
                "open_interest": 10,
                "dte": 2,
                "greeks": {"delta": 0.45, "theta": -3.0},
            },
            {
                "instrument_name": "ETH_USDC-8JUN26-100-P",
                "option_type": "put",
                "bid": 3.0,
                "ask": 5.0,
                "mid": 4.0,
                "spread_pct": 0.5,
                "best_bid_amount": 0.1,
                "best_ask_amount": 0.1,
                "volume": 0,
                "open_interest": 0,
                "dte": 2,
                "greeks": {"delta": -0.45, "theta": -15.0},
            },
        ]
    )
    packet = compact_market_packet(
        {
            "short_tape_summary": tape,
            "regime_transition_warning": enriched,
            "option_tradeability_summary": tradeability,
            "option_chain": [],
        }
    )

    assert tape["tape"] == "short_term_up"
    assert enriched["state"] == "chop_transition_up"
    assert "recent_forecasts_under_predicted_actual_price" in enriched["reason_codes"]
    assert tradeability["best_call_tradeability"]["grade"] == "good"
    assert packet["short_tape_summary"]["tape"] == "short_term_up"
    assert packet["option_tradeability_summary"]["best_call_tradeability"]["instrument_name"].endswith("-C")


def test_trend_carry_context_detects_smooth_down_pressure() -> None:
    start = datetime(2026, 6, 7, 10, 0, tzinfo=UTC)
    rows = []
    price = 1700.0
    for index in range(70):
        drift = -0.85
        bounce = 1.4 if index % 11 in {0, 1} else 0.0
        price += drift + bounce
        rows.append(
            {
                "timestamp": start + timedelta(minutes=index),
                "open": price + 0.25,
                "high": price + 0.8,
                "low": price - 0.8,
                "close": price,
                "volume": 20 + index,
            }
        )
    import pandas as pd

    prices = pd.DataFrame(rows).set_index("timestamp")
    context = trend_carry_context(prices)

    assert context["state"] == "trend_carry_down"
    assert "net_down_window" in context["reason_codes"]
    assert "smooth_directional_drift" in context["reason_codes"]
    assert "sma9_below_sma21" in context["reason_codes"]


def test_trend_carry_context_detects_smooth_up_pressure() -> None:
    start = datetime(2026, 6, 7, 10, 0, tzinfo=UTC)
    rows = []
    price = 1600.0
    for index in range(70):
        drift = 0.85
        dip = -1.4 if index % 11 in {0, 1} else 0.0
        price += drift + dip
        rows.append(
            {
                "timestamp": start + timedelta(minutes=index),
                "open": price - 0.25,
                "high": price + 0.8,
                "low": price - 0.8,
                "close": price,
                "volume": 20 + index,
            }
        )
    import pandas as pd

    prices = pd.DataFrame(rows).set_index("timestamp")
    context = trend_carry_context(prices)

    assert context["state"] == "trend_carry_up"
    assert "net_up_window" in context["reason_codes"]
    assert "smooth_directional_drift" in context["reason_codes"]
    assert "sma9_above_sma21" in context["reason_codes"]


def test_trend_carry_context_detects_early_down_after_high_rejection() -> None:
    start = datetime(2026, 6, 7, 17, 0, tzinfo=UTC)
    prices = []
    price = 1620.0
    for index in range(20):
        price += 0.8
        prices.append(price)
    for index in range(25):
        bounce = 0.6 if index in {5, 12, 18} else 0.0
        price += -0.55 + bounce
        prices.append(price)
    rows = [
        {
            "timestamp": start + timedelta(minutes=index),
            "open": value + 0.2,
            "high": value + 0.7,
            "low": value - 0.7,
            "close": value,
            "volume": 20 + index,
        }
        for index, value in enumerate(prices)
    ]
    import pandas as pd

    prices_frame = pd.DataFrame(rows).set_index("timestamp")
    context = trend_carry_context(prices_frame)

    assert context["state"] in {"early_trend_carry_down", "trend_carry_down"}
    assert "rejected_recent_high" in context["reason_codes"]
    assert "sma9_below_sma21" in context["reason_codes"]


def test_multi_window_trend_carry_detects_steady_up_with_fluctuations() -> None:
    start = datetime(2026, 6, 7, 15, 30, tzinfo=UTC)
    rows = []
    price = 1580.0
    for index in range(170):
        wave = -0.9 if index % 17 in {0, 1, 2} else 0.0
        price += 0.32 + wave
        rows.append(
            {
                "timestamp": start + timedelta(minutes=index),
                "open": price - 0.15,
                "high": price + 0.7,
                "low": price - 0.7,
                "close": price,
                "volume": 30 + (index % 20),
            }
        )
    import pandas as pd

    prices = pd.DataFrame(rows).set_index("timestamp")
    context = multi_window_trend_carry_context(prices)

    assert context["bias"] in {"single_window_up", "multi_window_up"}
    assert any((item.get("state") in {"early_trend_carry_up", "trend_carry_up"}) for item in context["windows"].values())
