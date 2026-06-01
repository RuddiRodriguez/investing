from __future__ import annotations

import pandas as pd

from market_forecasting_engine.options_decision import OptionsDecisionConfig, build_options_decision


def _prices() -> pd.DataFrame:
    index = pd.date_range("2026-05-30 12:00", periods=120, freq="5min")
    close = 2000 + pd.Series(range(120), dtype=float).to_numpy() * 0.05
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 10}, index=index)


def test_options_decision_builds_synthetic_candidates() -> None:
    prices = _prices()
    report = {
        "ticker": "ETH-USD",
        "as_of_timestamp": prices.index[-1].isoformat(),
        "current_price": float(prices["close"].iloc[-1]),
        "forecasts": [
            {
                "horizon_hours": 1.0,
                "forecast_date": (prices.index[-1] + pd.Timedelta(hours=1)).isoformat(),
                "predicted_price": float(prices["close"].iloc[-1] * 1.01),
                "lower_price": float(prices["close"].iloc[-1] * 0.995),
                "upper_price": float(prices["close"].iloc[-1] * 1.02),
                "expected_direction": "Upward",
                "directional_confidence": 0.58,
                "validation_metrics": {"mae": 0.003, "holdout_mae": 0.004},
            }
        ],
    }

    decision = build_options_decision(
        report,
        prices,
        config=OptionsDecisionConfig(ticker="ETH-USD", risk_profile="aggressive", strike_count=5, min_edge_pct=0.0),
    )

    assert decision["mode"] == "synthetic_chain_for_research"
    assert decision["risk_profile"] == "aggressive"
    assert decision["top_candidates"]
    assert {"candidate", "reject"} & {item["decision"] for item in decision["top_candidates"]}


def test_options_decision_accepts_real_chain_csv(tmp_path) -> None:
    prices = _prices()
    chain = tmp_path / "chain.csv"
    expiry = prices.index[-1] + pd.Timedelta(hours=1)
    chain.write_text(
        "symbol,option_type,strike,expiry,bid,ask,iv\n"
        f"ETH-CALL,call,1995,{expiry.isoformat()},8.0,8.8,0.7\n"
        f"ETH-PUT,put,1995,{expiry.isoformat()},1.2,1.4,0.7\n",
        encoding="utf-8",
    )
    report = {
        "ticker": "ETH-USD",
        "as_of_timestamp": prices.index[-1].isoformat(),
        "current_price": float(prices["close"].iloc[-1]),
        "forecasts": [
            {
                "horizon_hours": 1.0,
                "forecast_date": expiry.isoformat(),
                "predicted_price": float(prices["close"].iloc[-1] * 1.006),
                "expected_direction": "Upward",
                "directional_confidence": 0.6,
                "validation_metrics": {"mae": 0.002, "holdout_mae": 0.003},
            }
        ],
    }

    decision = build_options_decision(
        report,
        prices,
        config=OptionsDecisionConfig(ticker="ETH-USD", chain_csv=str(chain), min_edge_pct=0.0),
    )

    assert decision["mode"] == "real_chain_csv"
    assert decision["top_candidates"][0]["symbol"] in {"ETH-CALL", "ETH-PUT"}
