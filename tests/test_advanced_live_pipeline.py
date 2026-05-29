from __future__ import annotations

import pandas as pd
import numpy as np

from scripts.advanced_pipeline.cache import DataCache
from scripts.advanced_pipeline.config import PipelineConfig
from scripts.advanced_pipeline.pipeline import AdvancedLivePipeline, result_records
from scripts.advanced_pipeline.portfolio import BUY, HOLD, SELL, PortfolioOptimizer


def _prices(periods: int = 380) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=periods)
    step = np.arange(periods)
    return pd.DataFrame(
        {
            "AAA": 100 * (1.0015 ** step),
            "BBB": 100 * (1.0005 ** step) * (1 + 0.015 * np.sin(step / 12)),
            "CCC": 100 * (0.9995 ** step),
            "DDD": 100 * (1.0002 ** step) * (1 + 0.010 * np.cos(step / 9)),
        },
        index=index,
    )


def _fundamentals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "revenue_growth": [0.20, 0.08, -0.02, 0.03],
            "earnings_growth": [0.18, 0.06, -0.05, 0.02],
            "gross_margin": [0.65, 0.45, 0.30, 0.38],
            "debt_to_equity": [20.0, 80.0, 150.0, 60.0],
            "forward_pe": [30.0, 22.0, 15.0, 18.0],
            "free_cashflow_yield": [0.03, 0.04, 0.01, 0.02],
            "sector": ["Tech", "Tech", "Industrial", "Health"],
        }
    ).set_index("ticker")


def _news() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "news_sentiment": [0.50, 0.10, -0.30, 0.00],
            "positive_news_intensity": [0.60, 0.20, 0.00, 0.10],
            "negative_news_intensity": [0.00, 0.10, 0.40, 0.10],
        }
    ).set_index("ticker")


def test_data_cache_reuses_same_request(tmp_path) -> None:
    cache = DataCache(tmp_path, ttl_hours=24)
    params = {"symbols": ["AAA", "BBB"], "start": "2024-01-01"}
    key = cache.key_for("prices", params)
    frame = pd.DataFrame({"AAA": [1.0, 2.0], "BBB": [3.0, 4.0]})

    cache.write_frame(key, frame, params)
    loaded = cache.read_frame(key)

    pd.testing.assert_frame_equal(loaded, frame)
    assert key == cache.key_for("prices", {"start": "2024-01-01", "symbols": ["AAA", "BBB"]})


def test_advanced_live_pipeline_runs_from_live_style_frames(tmp_path) -> None:
    config = PipelineConfig(
        tickers=("AAA", "BBB", "CCC", "DDD"),
        benchmark=None,
        horizons=(5, 20),
        primary_horizon=20,
        min_history_days=260,
        min_model_rows=80,
        train_window_days=300,
        cache_dir=tmp_path,
    )
    pipeline = AdvancedLivePipeline(config)

    result = pipeline.run_from_frames(
        prices=_prices(),
        fundamentals=_fundamentals(),
        news=_news(),
        sectors={"AAA": "Tech", "BBB": "Tech", "CCC": "Industrial", "DDD": "Health"},
    )
    records = result_records(result)

    assert result.as_of_date == _prices().index[-1]
    assert result.target_weights.sum() == 1.0
    assert len(result.decisions) == 4
    assert {record["ticker"] for record in records} == {"AAA", "BBB", "CCC", "DDD"}
    assert {
        "decision",
        "confidence",
        "risk_score",
        "position_size",
        "lower_bound",
        "upper_bound",
        "alpha_score",
        "reason_codes",
    }.issubset(result.decisions.columns)
    assert "decision_counts" in result.diagnostics
    assert {"lower_bound", "upper_bound", "alpha_score", "reason_codes"}.issubset(records[0].keys())


def test_advanced_pipeline_action_has_hold_zone_for_weak_negative_signal(tmp_path) -> None:
    config = PipelineConfig(tickers=("AAA",), cache_dir=tmp_path)
    pipeline = AdvancedLivePipeline(config)

    assert pipeline._action((0.04, 0.65, 0.05, 0.30, 0.09, 0.01)) == BUY
    assert pipeline._action((-0.02, 0.52, 0.05, 0.30, 0.03, -0.07)) == HOLD
    assert pipeline._action((-0.05, 0.65, 0.05, 0.30, -0.01, -0.10)) == SELL


def test_portfolio_optimizer_leaves_sector_cap_excess_in_cash(tmp_path) -> None:
    date = pd.Timestamp("2026-05-14")
    index = pd.MultiIndex.from_tuples(
        [(date, "AAA"), (date, "BBB"), (date, "CCC")],
        names=["date", "ticker"],
    )
    decisions = pd.DataFrame(
        {
            "decision": [BUY, BUY, BUY],
            "expected_volatility": [0.05, 0.06, 0.07],
            "risk_adjusted_score": [3.0, 2.0, 1.0],
            "alpha_score": [0.12, 0.08, 0.04],
            "confidence": [0.8, 0.7, 0.65],
        },
        index=index,
    )
    config = PipelineConfig(
        tickers=("AAA", "BBB", "CCC"),
        max_position_size=0.40,
        max_sector_weight=0.55,
        volatility_target=1.0,
        cache_dir=tmp_path,
    )

    weights = PortfolioOptimizer(config).optimize(
        decisions,
        sectors={"AAA": "Tech", "BBB": "Tech", "CCC": "Health"},
    )

    assert weights["AAA"] <= 0.40
    assert weights["BBB"] <= 0.40
    assert weights["AAA"] + weights["BBB"] <= 0.55 + 1e-9
    assert weights.sum() == 1.0
