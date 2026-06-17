from __future__ import annotations

import json
from pathlib import Path

from market_forecasting_engine.virtual_trader_enrichment import (
    VirtualTraderEnrichmentConfig,
    load_selected_candidates,
    run_virtual_trader_enrichment,
)


def test_load_selected_candidates_orders_by_rank(tmp_path: Path) -> None:
    path = tmp_path / "selected_candidates.json"
    path.write_text(
        json.dumps(
            [
                {"ticker": "BBB", "rank": 2},
                {"ticker": "AAA", "rank": 1},
                {"ticker": "CCC", "rank": 3},
            ]
        ),
        encoding="utf-8",
    )

    rows = load_selected_candidates(path, max_candidates=2)

    assert [row["ticker"] for row in rows] == ["AAA", "BBB"]


def test_run_virtual_trader_enrichment_writes_independent_artifacts(monkeypatch, tmp_path: Path) -> None:
    selected = tmp_path / "selected_candidates.json"
    selected.write_text(
        json.dumps(
            [
                {
                    "ticker": "AAA",
                    "rank": 1,
                    "ranking_score": 0.77,
                    "score": 0.66,
                    "latest_close": 100.0,
                    "ranking_components": {"trend_score": 0.8, "setup_quality_score": 0.7, "risk_score": 0.6},
                }
            ]
        ),
        encoding="utf-8",
    )

    def fake_collect(request):
        return {
            "ticker": request.ticker,
            "status": "ok",
            "provider_summaries": {"stockanalysis": {"status": "ok"}},
            "source_payloads": {},
            "consolidated": {"status": "ok"},
            "artifacts": {},
        }

    def fake_snapshot(context, snapshot_dir, *, ticker=None):
        path = Path(snapshot_dir) / f"{ticker}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        return path

    def fake_synthesis(**kwargs):
        return {
            "status": "executed",
            "model": "test-model",
            "llm_evidence_manifest": {"all_scraped_sections_passed": True},
            "synthesis": {"bullish_evidence": ["test"]},
        }

    def fake_strategy(report, request):
        return {
            "status": "executed",
            "retrieved_chunks": [{"source_title": "book"}],
            "synthesis": {"applicable_principles": ["test principle"]},
        }

    monkeypatch.setattr("market_forecasting_engine.virtual_trader_enrichment.collect_long_term_source_context", fake_collect)
    monkeypatch.setattr("market_forecasting_engine.virtual_trader_enrichment.append_long_term_source_snapshot", fake_snapshot)
    monkeypatch.setattr("market_forecasting_engine.virtual_trader_enrichment.run_long_term_source_synthesis", fake_synthesis)
    monkeypatch.setattr("market_forecasting_engine.virtual_trader_enrichment.build_strategy_knowledge_context", fake_strategy)

    board = run_virtual_trader_enrichment(
        VirtualTraderEnrichmentConfig(
            selected_candidates_path=selected,
            output_dir=tmp_path / "enriched",
            providers=("stockanalysis",),
            progress=False,
        )
    )

    assert board["policy"]["forecast_free"] is True
    assert board["policy"]["execution_allowed"] is False
    assert board["policy"]["role"].startswith("Deep evidence packets")
    assert board["candidates"][0]["ticker"] == "AAA"
    assert board["candidates"][0]["source_synthesis"]["status"] == "executed"
    assert (tmp_path / "enriched" / "enrichment_board.json").exists()
    assert (tmp_path / "enriched" / "tickers" / "aaa" / "evidence_packet.json").exists()

