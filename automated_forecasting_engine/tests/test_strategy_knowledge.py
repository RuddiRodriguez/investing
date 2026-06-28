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


def test_strategy_knowledge_retrieves_liquidity_sweep_structure_note(tmp_path, monkeypatch) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "liquidity_sweep.md").write_text(
        "Swing lows hold liquidity. Equal lows hold more liquidity. A wick below a low can be a liquidity sweep. "
        "If price retests former support as resistance and then closes below structure, treat it as bearish break of structure. "
        "Fair value gap plus support-to-resistance flip can point to downside liquidity sweep.",
        encoding="utf-8",
    )
    index_path = tmp_path / "strategy.faiss"
    monkeypatch.setattr(
        strategy_knowledge,
        "_embed_texts",
        lambda texts, request, *, purpose: ([], {"status": "skipped", "purpose": purpose}),
    )
    request = StrategyKnowledgeRequest(ticker="TEST", corpus_dir=corpus_dir, index_path=index_path, max_chunks=2)

    context = build_strategy_knowledge_context(
        {
            "ticker": "TEST",
            "suggested_action": "Hold",
            "decision_view": {
                "market_structure_focus": {
                    "concepts": ["swing lows", "liquidity sweep", "break of structure", "fair value gap", "support turns resistance"]
                }
            },
            "technical_view": {},
        },
        request,
    )

    text = " ".join(chunk["text_excerpt"] for chunk in context["retrieved_chunks"])
    assert context["status"] == "executed"
    assert "liquidity sweep" in text
    assert "break of structure" in text


def test_strategy_knowledge_pins_stock_execution_guidance_for_stock_ceo(tmp_path, monkeypatch) -> None:
    corpus_dir = tmp_path / "corpus"
    guidance_dir = corpus_dir / "long_term_stock_guidance"
    guidance_dir.mkdir(parents=True)
    (guidance_dir / "stock_long_term_execution_guidance.md").write_text(
        "# Long-Term Stock And ETF Execution Guidance\n\n"
        "Check ROIC or return on invested capital when available. "
        "Do not use market orders. Prefer limit order. Do not chase a huge daily jump. "
        "Do not buy more just because it is cheaper. One growth buy per week max. "
        "One ETF buy per month. If a growth stock falls 7% to 8%, review. "
        "Bad news can justify sell or reduce. Consider partial profit when momentum weakens.",
        encoding="utf-8",
    )
    (corpus_dir / "other.md").write_text("Breakout and support note.", encoding="utf-8")
    index_path = tmp_path / "strategy.faiss"
    monkeypatch.setattr(
        strategy_knowledge,
        "_embed_texts",
        lambda texts, request, *, purpose: ([], {"status": "skipped", "purpose": purpose}),
    )

    context = build_strategy_knowledge_context(
        {"ticker": "AAPL", "suggested_action": "Buy", "decision_view": {}, "technical_view": {}},
        StrategyKnowledgeRequest(ticker="AAPL", corpus_dir=corpus_dir, index_path=index_path, max_chunks=1),
    )

    first_chunk = context["retrieved_chunks"][0]
    assert first_chunk["source_path"].endswith(strategy_knowledge.PINNED_STOCK_GUIDANCE_FILENAME)
    assert first_chunk["pin_reason"] == "stock_long_term_execution_guidance"
    assert "ROIC" in " ".join(context["synthesis"]["applicable_principles"])
    assert "limit orders" in " ".join(context["synthesis"]["entry_rules"])
    assert "7% to 8%" in " ".join(context["synthesis"]["risk_and_sizing_rules"])


def test_strategy_knowledge_does_not_pin_stock_guidance_for_crypto_pair(tmp_path, monkeypatch) -> None:
    corpus_dir = tmp_path / "corpus"
    guidance_dir = corpus_dir / "long_term_stock_guidance"
    guidance_dir.mkdir(parents=True)
    (guidance_dir / "stock_long_term_execution_guidance.md").write_text(
        "# Long-Term Stock And ETF Execution Guidance\n\nUse limit order for stock buys.",
        encoding="utf-8",
    )
    index_path = tmp_path / "strategy.faiss"
    monkeypatch.setattr(
        strategy_knowledge,
        "_embed_texts",
        lambda texts, request, *, purpose: ([], {"status": "skipped", "purpose": purpose}),
    )

    context = build_strategy_knowledge_context(
        {"ticker": "ETH-USDC", "suggested_action": "Hold", "decision_view": {}, "technical_view": {}},
        StrategyKnowledgeRequest(ticker="ETH-USDC", corpus_dir=corpus_dir, index_path=index_path, max_chunks=1),
    )

    assert all(not str(chunk.get("source_path") or "").endswith(strategy_knowledge.PINNED_STOCK_GUIDANCE_FILENAME) for chunk in context["retrieved_chunks"])
