from __future__ import annotations

import numpy as np
import pandas as pd

from market_forecasting_engine.daily_trade import (
    DailyTradeConfig,
    add_forecast_bars,
    add_trading_bars,
    add_trading_minutes,
    build_intraday_feature_frame,
    build_daily_trade_plan,
    infer_bar_interval_minutes,
)
from market_forecasting_engine.daily_trade_cli import (
    _annotate_mean_reversion_dip_buy,
    _hour_label,
    _limit_training_rows,
    _run_daily_llm_decision,
    _validation_gate,
)
from market_forecasting_engine.plots import write_daily_trade_plot_artifacts


def _intraday_prices(rows: int = 78) -> pd.DataFrame:
    index = pd.date_range("2026-05-28 09:30", periods=rows, freq="5min")
    close = 100 + np.linspace(0, 3.0, rows)
    volume = np.linspace(100_000, 180_000, rows)
    return pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.10,
            "low": close - 0.15,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_infer_bar_interval_minutes_uses_intraday_gaps() -> None:
    prices = _intraday_prices()

    assert infer_bar_interval_minutes(prices.index) == 5.0


def test_daily_trade_plan_builds_long_setup_from_intraday_data() -> None:
    prices = _intraday_prices()
    report = build_daily_trade_plan(prices, DailyTradeConfig(ticker="TEST", minimum_score_to_trade=2.0))

    assert report["requires_intraday_data"] is True
    assert report["has_intraday_data"] is True
    assert report["interval_minutes"] == 5.0
    assert report["trade_plan"]["action"] == "long"
    assert report["trade_plan"]["stop"] < report["latest_price"]
    assert report["trade_plan"]["take_profit"] > report["latest_price"]
    assert [item["horizon_hours"] for item in report["forecasts"]] == [1.0, 2.0, 4.0]
    assert report["forecasts"][0]["forecast_timestamp"] > report["as_of"]


def test_forecast_timestamp_skips_closed_market_hours() -> None:
    forecast_time = add_trading_minutes(pd.Timestamp("2026-05-29 15:55"), 60)

    assert forecast_time == pd.Timestamp("2026-06-01 10:25")


def test_forecast_timestamp_uses_observed_provider_session() -> None:
    index = pd.date_range("2026-05-29 13:30", periods=78, freq="5min")

    forecast_time = add_trading_bars(index, pd.Timestamp("2026-05-29 19:55"), 6, 5.0)

    assert forecast_time == pd.Timestamp("2026-06-01 13:55")


def test_forecast_timestamp_extrapolates_live_partial_session() -> None:
    index = pd.date_range("2026-06-04 08:00", periods=375, freq="1min")

    forecast_time = add_trading_bars(index, pd.Timestamp("2026-06-04 14:14"), 15, 1.0)

    assert forecast_time == pd.Timestamp("2026-06-04 14:29")


def test_forecast_timestamp_keeps_short_live_equity_horizon_same_day() -> None:
    index = pd.date_range("2026-06-04 08:00", periods=975, freq="1min")

    forecast_time = add_trading_bars(index, pd.Timestamp("2026-06-04 14:33"), 15, 1.0)

    assert forecast_time == pd.Timestamp("2026-06-04 14:48")


def test_forecast_timestamp_prefers_live_projection_before_next_template_day() -> None:
    index = pd.date_range("2026-06-04 08:00", periods=424, freq="1min")

    forecast_time = add_trading_bars(index, pd.Timestamp("2026-06-04 15:03"), 15, 1.0)

    assert forecast_time == pd.Timestamp("2026-06-04 15:18")


def test_forecast_timestamp_keeps_weekends_for_247_markets() -> None:
    index = pd.date_range("2026-05-29 00:00", periods=3 * 288, freq="5min")

    forecast_time = add_trading_bars(index, pd.Timestamp("2026-05-29 23:55"), 2, 5.0)

    assert forecast_time == pd.Timestamp("2026-05-30 00:05")


def test_continuous_forecast_timestamp_does_not_use_equity_session_projection() -> None:
    index = pd.date_range("2026-06-05 21:30", periods=120, freq="1min")

    forecast_time = add_forecast_bars(
        index,
        pd.Timestamp("2026-06-05 22:59"),
        15,
        1.0,
        calendar="continuous_24_7",
    )

    assert forecast_time == pd.Timestamp("2026-06-05 23:14")


def test_intraday_feature_frame_adds_session_features() -> None:
    prices = _intraday_prices()

    features = build_intraday_feature_frame(prices)

    assert "intraday_close_to_vwap" in features.columns
    assert "intraday_session_progress" in features.columns
    assert features["intraday_session_progress"].iloc[-1] > features["intraday_session_progress"].iloc[0]


def test_daily_trade_plan_warns_when_input_is_daily() -> None:
    index = pd.bdate_range("2026-01-02", periods=30)
    close = 100 + np.arange(30)
    prices = pd.DataFrame({"close": close, "volume": 100_000}, index=index)

    report = build_daily_trade_plan(prices, DailyTradeConfig(ticker="TEST"))

    assert report["has_intraday_data"] is False
    assert "daily/end-of-day" in report["data_warning"]


def test_daily_trade_plot_artifacts_are_written(tmp_path) -> None:
    prices = _intraday_prices()
    report = build_daily_trade_plan(prices, DailyTradeConfig(ticker="TEST", minimum_score_to_trade=2.0))

    artifacts = write_daily_trade_plot_artifacts(report, prices, output_dir=tmp_path)

    assert (tmp_path / "plots" / "daily_trade_TEST.png").exists()
    assert (tmp_path / "plots" / "daily_trade_TEST.html").exists()
    assert artifacts["daily_trade_plot"].endswith("daily_trade_TEST.png")


def test_validation_gate_blocks_weak_hourly_models() -> None:
    gate = _validation_gate(
        {
            "directional_accuracy": 0.25,
            "holdout_directional_accuracy": 0.40,
            "mae": 0.02,
            "holdout_mae": 0.01,
        }
    )

    assert gate["status"] == "weak_validation"
    assert gate["trade_allowed"] is False
    assert "low_walk_forward_directional_accuracy" in gate["reasons"]
    assert _hour_label(2.0) == "2h"


def test_limit_training_rows_uses_most_recent_rows() -> None:
    prices = _intraday_prices(rows=100)

    limited = _limit_training_rows(prices, 30)

    assert len(limited) == 30
    assert limited.index[0] == prices.index[-30]


def test_mean_reversion_dip_buy_view_adds_conditional_lower_buy_setup() -> None:
    prices = _intraday_prices(rows=240)
    current = float(prices["close"].iloc[-1])
    report = {
        "ticker": "ETH-USD",
        "current_price": current,
        "forecasts": [
            {
                "horizon_days": 24,
                "horizon_hours": 2.0,
                "predicted_price": current * 0.98,
                "lower_price": current * 0.97,
                "upper_price": current * 1.01,
                "expected_direction": "Down",
                "trade_allowed": False,
                "selected_model": "test_model",
            }
        ],
        "decision_view": {
            "production_gate": {
                "chart_confirmation": {
                    "support_level": current * 0.975,
                    "resistance_level": current * 1.01,
                }
            }
        },
    }

    _annotate_mean_reversion_dip_buy(report, prices, "close", risk_profile_name="aggressive")

    view = report["decision_view"]["mean_reversion_dip_buy"]
    assert view["best_setup"]["setup"] == "conditional_dip_buy"
    assert view["best_setup"]["entry_price"] < current
    assert view["best_setup"]["target_price"] > view["best_setup"]["entry_price"]
    assert view["best_setup"]["order_template"]["type"] == "limit"


def test_daily_llm_decision_dry_run_builds_prompt_packet() -> None:
    class Args:
        ticker = "ETH-USD"
        risk_profile = "aggressive"
        trader_name = "test_daily_trader"
        holding_status = "not_owned"
        entry_price = None
        quantity = None
        position_value = None
        account_equity = None
        portfolio_notes = ""
        llm_model = "gpt-5.4-mini-2026-03-17"
        llm_reasoning_effort = "none"
        llm_no_web_search = True
        llm_search_context_size = "low"

    report = {
        "ticker": "ETH-USD",
        "as_of_date": "2026-05-31",
        "current_price": 100.0,
        "suggested_action": "Hold",
        "forecasts": [],
        "decision_view": {
            "production_gate": {"allowed_forecast_count": 0},
            "mean_reversion_dip_buy": {"best_setup": {"entry_price": 95.0}},
        },
    }

    result = _run_daily_llm_decision(report, Args(), dry_run=True)

    assert result["status"] == "dry_run"
    assert result["trader_profile"]["name"] == "aggressive"
    assert result["technical_packet"]["decision_governance"]["mean_reversion_dip_buy"]["best_setup"]["entry_price"] == 95.0
    assert result["llm_prompt_payload"]["tools"] == []
