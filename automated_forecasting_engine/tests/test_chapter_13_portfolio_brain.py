from __future__ import annotations

import json

import numpy as np
import pandas as pd

from market_forecasting_engine.chapter_13_manifold_visualization import build_chapter_13_manifold_embedding
from market_forecasting_engine.chapter_13_eigen_trading import build_eigen_trading_plan
from market_forecasting_engine.chapter_13_unsupervised_risk import analyze_chapter_13_unsupervised_risk
from market_forecasting_engine.portfolio_brain import run_portfolio_brain


def _prices(seed: int, beta: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=180, freq="B")
    market = np.sin(np.linspace(0, 9, len(index))) * 0.004
    noise = rng.normal(0, 0.006, len(index))
    returns = beta * market + noise
    close = 100 * (1 + pd.Series(returns, index=index)).cumprod()
    return pd.DataFrame({"close": close, "open": close, "high": close * 1.01, "low": close * 0.99, "volume": 1000})


def _report():
    return {
        "summary": {"total_current_value": 6000.0, "holding_count": 3},
        "holdings": [
            {"isin": "A", "ticker": "AAA", "name": "Alpha", "current_value": 3000.0, "current_quantity": 10, "broker_avg_cost": 90.0},
            {"isin": "B", "ticker": "BBB", "name": "Beta", "current_value": 2000.0, "current_quantity": 10, "broker_avg_cost": 95.0},
            {"isin": "C", "ticker": "CCC", "name": "Gamma", "current_value": 1000.0, "current_quantity": 10, "broker_avg_cost": 105.0},
        ],
    }


def test_chapter_13_unsupervised_risk_builds_pca_clusters_and_hrp() -> None:
    prices = {"AAA": _prices(1), "BBB": _prices(2, beta=0.8), "CCC": _prices(3, beta=-0.4)}

    result = analyze_chapter_13_unsupervised_risk(prices, current_values={"AAA": 3000, "BBB": 2000, "CCC": 1000})

    assert result["status"] == "available"
    assert result["pca"]["component_count"] >= 1
    assert "pc1" in result["pca"]["eigenportfolios"]
    assert set(result["ticker_contexts"]) == {"AAA", "BBB", "CCC"}
    assert round(sum(result["hrp"]["risk_weights"].values()), 6) == 1.0
    eigen_plan = build_eigen_trading_plan(result, max_component_gross_notional=100.0)
    assert eigen_plan["status"] == "available"
    assert eigen_plan["plans"][0]["component"] == "pc1"
    assert "do not override" in eigen_plan["policy"]


def test_portfolio_brain_writes_contexts_and_dry_run_basket(tmp_path) -> None:
    prices = {"AAA": _prices(1), "BBB": _prices(2, beta=0.8), "CCC": _prices(3, beta=-0.4)}
    log_dir = tmp_path / "watch" / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "AAA_medium_20260603.jsonl").write_text(
        json.dumps({"ticker": "AAA", "action": "BUY", "price": 101.0}) + "\n",
        encoding="utf-8",
    )

    result = run_portfolio_brain(
        report=_report(),
        output_dir=tmp_path / "brain",
        watch_state_dir=tmp_path / "watch",
        profile="medium",
        prices_by_ticker=prices,
        execute_basket_orders=False,
    )

    report = result["report"]
    assert report["execution"]["status"] == "dry_run"
    assert report["chapter_13_eigen_trading"]["status"] == "available"
    assert (tmp_path / "brain" / "portfolio_brain_report.json").exists()
    context = json.loads((tmp_path / "watch" / "portfolio_contexts" / "aaa_medium.json").read_text())
    assert context["portfolio_brain"]["chapter_13_context"]["ticker"] == "AAA"
    assert "basket_order" in context["portfolio_brain"]


def test_chapter_13_manifold_embedding_tsne_is_visualization_only() -> None:
    prices = {"AAA": _prices(1), "BBB": _prices(2), "CCC": _prices(3), "DDD": _prices(4)}

    result = build_chapter_13_manifold_embedding(prices, method="tsne", min_history=60)

    assert result["status"] == "available"
    assert result["method"] == "tsne"
    assert "Visualization-only" in result["policy"]
    assert len(result["coordinates"]) == 4
