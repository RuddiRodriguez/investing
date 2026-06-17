from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from market_forecasting_engine.virtual_trader_scout import (
    DiscoveryRecord,
    ScoutConfig,
    rank_scout_candidates,
    run_virtual_trader_scout,
    score_scout_candidates,
)


def _price_frame(start: float, trend: float, volume: float, periods: int = 260) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="B")
    close = pd.Series(start * np.exp(np.linspace(0.0, trend, periods)), index=index)
    high = close * 1.01
    low = close * 0.99
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_score_scout_candidates_selects_liquid_dynamic_candidate(tmp_path: Path) -> None:
    config = ScoutConfig(output_dir=tmp_path, progress=False, enable_llm_ranking=False)
    records = [
        DiscoveryRecord(
            ticker="AAA",
            source="stockanalysis_visible_analyst_rating",
            reason="visible analyst rating",
            metadata={
                "rating": "Buy",
                "analyst": {"rank": 1, "success_rate": 70.0, "average_return": 20.0, "sector": "Technology"},
            },
        ),
        DiscoveryRecord(
            ticker="AAA",
            source="fmp_recent_news",
            reason="recent news activity",
            metadata={"title": "AAA wins contract"},
        ),
        DiscoveryRecord(
            ticker="BBB",
            source="fmp_market_actives",
            reason="market active",
            metadata={},
        ),
    ]
    frames = {
        "AAA": _price_frame(100.0, 0.35, 2_000_000.0),
        "BBB": _price_frame(8.0, -0.15, 100_000.0),
    }

    scored = score_scout_candidates(["AAA", "BBB"], records, frames, config)

    assert scored[0]["ticker"] == "AAA"
    assert scored[0]["eligible"] is True
    assert scored[0]["score"] > scored[1]["score"]
    assert "liquidity_below_minimum" in scored[1]["rejection_reasons"]


def test_rank_scout_candidates_adds_second_stage_components(tmp_path: Path) -> None:
    config = ScoutConfig(output_dir=tmp_path, progress=False, enable_llm_ranking=False, portfolio_tickers=("BBB",), portfolio_sectors=("Technology",))
    rows = [
        {
            "ticker": "AAA",
            "eligible": True,
            "score": 0.70,
            "sector": "Industrials",
            "metrics": {
                "return_21d": 0.05,
                "return_63d": 0.15,
                "distance_to_sma_200_pct": 0.20,
                "range_position_252d": 0.75,
                "atr_pct_20d": 0.025,
                "realized_volatility_20d": 0.30,
            },
            "score_components": {
                "liquidity": 0.8,
                "momentum": 0.75,
                "breakout": 0.7,
                "pullback": 0.4,
                "analyst_activity": 0.5,
                "unusual_volume": 0.4,
                "news_activity": 0.2,
                "earnings_activity": 0.0,
                "market_mover": 0.5,
                "volatility_fit": 0.8,
                "sector_strength": 0.7,
            },
            "risk_penalty": 0.0,
            "latest_records": [],
        }
    ]

    ranked, audit = rank_scout_candidates(rows, config)

    assert audit["llm_ranking"]["status"] == "not_requested"
    assert ranked[0]["ranking_score"] > 0
    assert set(ranked[0]["ranking_components"]) >= {
        "liquidity_score",
        "trend_score",
        "setup_quality_score",
        "valuation_sanity_score",
        "catalyst_news_score",
        "risk_score",
        "portfolio_diversification_score",
    }


def test_run_virtual_trader_scout_writes_artifacts(monkeypatch, tmp_path: Path) -> None:
    records = [
        DiscoveryRecord(
            ticker="AAA",
            source="stockanalysis_visible_analyst_rating",
            reason="visible analyst rating",
            metadata={"rating": "Buy", "analyst": {"rank": 2, "success_rate": 68.0, "average_return": 12.0, "sector": "Industrials"}},
        )
    ]

    def fake_discover(config: ScoutConfig):
        return records, {"sources": {"test": {"status": "ok", "records": 1}}, "errors": []}

    def fake_prices(tickers, config, data_store):
        return {"AAA": _price_frame(50.0, 0.20, 1_000_000.0)}, {"provider": "test", "ok": ["AAA"], "failed": []}

    monkeypatch.setattr("market_forecasting_engine.virtual_trader_scout.discover_candidate_universe", fake_discover)
    monkeypatch.setattr("market_forecasting_engine.virtual_trader_scout.load_scout_price_frames", fake_prices)

    summary = run_virtual_trader_scout(ScoutConfig(output_dir=tmp_path, progress=False, final_candidates=1, enable_llm_ranking=False))

    assert summary["selected_candidates"][0]["ticker"] == "AAA"
    assert "ranking_score" in summary["selected_candidates"][0]
    assert (tmp_path / "scout_summary.json").exists()
    assert (tmp_path / "candidate_scores.csv").exists()
    assert (tmp_path / "cheap_ranking.json").exists()
    assert (tmp_path / "selected_candidates.json").exists()
    assert (tmp_path / "run_summary.md").exists()
