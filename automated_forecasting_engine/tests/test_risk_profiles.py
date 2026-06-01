from __future__ import annotations

import pandas as pd

from market_forecasting_engine.daily_trade_cli import _annotate_production_decision
from market_forecasting_engine.risk_profiles import risk_profile_for_name


def _prices() -> pd.DataFrame:
    index = pd.date_range("2026-05-30 12:00", periods=80, freq="5min")
    close = 100 + pd.Series(range(80), dtype=float).to_numpy() * 0.02
    return pd.DataFrame({"open": close, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": 1000}, index=index)


def _report() -> dict:
    return {
        "ticker": "TEST",
        "forecast_interval": "5m",
        "data_provider": {"provider": "alpaca"},
        "forecasts": [
            {
                "expected_return": 0.0012,
                "directional_confidence": 0.53,
                "validation_metrics": {"mae": 0.002, "holdout_mae": 0.002, "directional_accuracy": 0.55},
                "validation_gate": {"trade_allowed": True, "status": "pass", "reasons": []},
            }
        ],
    }


def test_risk_profile_lookup_rejects_unknown_name() -> None:
    assert risk_profile_for_name("medium").name == "medium"
    try:
        risk_profile_for_name("reckless")
    except ValueError as exc:
        assert "Unknown risk profile" in str(exc)
    else:
        raise AssertionError("Unknown risk profile should fail.")


def test_aggressive_profile_can_allow_edge_that_conservative_blocks() -> None:
    conservative = _report()
    aggressive = _report()

    _annotate_production_decision(conservative, _prices(), "close", risk_profile_name="conservative")
    _annotate_production_decision(aggressive, _prices(), "close", risk_profile_name="aggressive")

    assert conservative["forecasts"][0]["production_gate"]["status"] == "blocked"
    assert "low_directional_confidence" in conservative["forecasts"][0]["production_gate"]["reasons"]
    assert aggressive["forecasts"][0]["production_gate"]["status"] == "pass"
    assert aggressive["decision_view"]["production_gate"]["risk_profile"]["name"] == "aggressive"
