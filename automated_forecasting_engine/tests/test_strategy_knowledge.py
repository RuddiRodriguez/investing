from pathlib import Path

from market_forecasting_engine import strategy_knowledge
from market_forecasting_engine.strategy_knowledge import (
    StrategyKnowledgeRequest,
    build_strategy_knowledge_context,
    retrieve_strategy_chunks,
)


def test_strategy_knowledge_builds_faiss_index_and_retrieves_chunks(tmp_path, monkeypatch) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "can_slim_note.md").write_text(
        "# CAN SLIM Note\n\n"
        "Use strong earnings, proper bases, breakout pivots, pullbacks to support, and 7% to 8% stop discipline.",
        encoding="utf-8",
    )
    index_path = tmp_path / "indexes" / "strategy.faiss"

    def fake_embed_texts(texts, request, *, purpose):
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    1.0 if "breakout" in lower or "support" in lower else 0.0,
                    1.0 if "earnings" in lower else 0.0,
                    1.0 if "stop" in lower else 0.0,
                ]
            )
        return vectors, {"status": "fake", "purpose": purpose, "vector_count": len(vectors)}

    monkeypatch.setattr(strategy_knowledge, "_embed_texts", fake_embed_texts)
    request = StrategyKnowledgeRequest(
        ticker="AAPL",
        corpus_dir=corpus_dir,
        index_path=index_path,
        max_chunks=3,
        rebuild_index=True,
    )

    context = build_strategy_knowledge_context(
        {
            "ticker": "AAPL",
            "suggested_action": "Hold",
            "risk_level": "Medium",
            "current_price": 100.0,
            "forecasts": [{"horizon_days": 5, "expected_direction": "Upward", "directional_confidence": 0.61}],
            "technical_view": {"trend_state": {"state": "Bullish"}},
            "decision_view": {"chapter_18_tactical_problem": {"trade_plan": {"entry_policy": "wait for pullback"}}},
        },
        request,
    )

    assert context["status"] == "executed"
    assert context["index_status"]["backend"] == "faiss"
    assert index_path.exists()
    assert Path(str(index_path).replace(".faiss", ".chunks.jsonl")).exists()
    assert context["retrieved_chunks"]
    assert context["synthesis"]["entry_rules"]
    assert context["decision_policy"]["feeds_ceo_llm"] is True


def test_strategy_knowledge_falls_back_to_lexical_without_embeddings(tmp_path, monkeypatch) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "risk.txt").write_text(
        "A pullback to support is acceptable only if stop risk and position sizing are controlled.",
        encoding="utf-8",
    )
    index_path = tmp_path / "strategy.faiss"

    monkeypatch.setattr(
        strategy_knowledge,
        "_embed_texts",
        lambda texts, request, *, purpose: ([], {"status": "skipped", "purpose": purpose}),
    )
    request = StrategyKnowledgeRequest(ticker="MSFT", corpus_dir=corpus_dir, index_path=index_path, max_chunks=2)
    context = build_strategy_knowledge_context(
        {"ticker": "MSFT", "suggested_action": "Hold", "decision_view": {}, "technical_view": {}},
        request,
    )

    assert context["index_status"]["status"] == "lexical_only"
    assert context["retrieved_chunks"]
    assert retrieve_strategy_chunks("support stop risk", request)
