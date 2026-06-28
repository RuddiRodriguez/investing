from __future__ import annotations

import pandas as pd

from market_forecasting_engine.llm_model_catalog import DEFAULT_FULL_LLM_OPTIONS_MODEL, DEFAULT_FULL_LLM_OPTIONS_PROVIDER
from market_forecasting_engine.llm_trader.prompts import autonomous_trader
from market_forecasting_engine.openai_models import ModelName
import market_forecasting_engine.pure_llm_stock_forecaster as pure_llm_stock_forecaster
from market_forecasting_engine.pure_llm_stock_forecaster import (
    build_compact_ceo_handoff_packet,
    build_ceo_technical_packet,
    build_market_packet,
    build_parser,
    build_portfolio_context,
    load_or_build_external_evidence,
    call_ceo_advice,
    call_response_with_fallback,
    normalize_forecast,
)


def _prices() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=70, freq="B")
    close = [100 + i * 0.25 for i in range(70)]
    return pd.DataFrame(
        {
            "open": [value - 0.2 for value in close],
            "high": [value + 0.5 for value in close],
            "low": [value - 0.7 for value in close],
            "close": close,
            "volume": [1000 + i for i in range(70)],
        },
        index=index,
    )


def test_parser_defaults_to_full_llm_local_model() -> None:
    args = build_parser().parse_args(["--ticker", "J"])

    assert args.llm_provider == DEFAULT_FULL_LLM_OPTIONS_PROVIDER
    assert args.llm_model == DEFAULT_FULL_LLM_OPTIONS_MODEL
    assert args.fallback_llm_provider == "openai"
    assert args.fallback_llm_model == ModelName.GPT_5_4_MINI_2026_03_17.value
    assert args.ceo_llm_provider == "openai"
    assert args.ceo_llm_model == ModelName.GPT_5_4_2026_03_05.value
    assert args.trader_profile == "medium"
    assert args.skip_llm_evidence is False
    assert args.disable_strategy_knowledge is False
    assert args.strategy_knowledge_max_chunks == 8


def test_market_packet_is_raw_descriptive_not_engine_forecast() -> None:
    packet = build_market_packet(_prices(), ticker="J", company="Jacobs Solutions Inc.", metadata={"provider": "unit"}, bar_count=12)

    assert packet["ticker"] == "J"
    assert packet["requested_forecast_horizons"] == [
        {"horizon": "1_day", "trading_days": 1},
        {"horizon": "1_week", "trading_days": 5},
        {"horizon": "1_month", "trading_days": 21},
    ]
    assert "descriptive_context_not_forecast" in packet
    assert len(packet["recent_daily_bars"]) == 12


def test_normalize_forecast_enforces_required_horizons() -> None:
    packet = build_market_packet(_prices(), ticker="J", company="Jacobs Solutions Inc.", metadata={}, bar_count=5)
    forecast = normalize_forecast(
        {
            "forecasts": [
                {"horizon": "1_week", "trading_days": 99, "predicted_price": 101},
                {"horizon": "1_month", "trading_days": 99, "predicted_price": 102},
                {"horizon": "1_day", "trading_days": 99, "predicted_price": 100},
            ]
        },
        packet,
    )

    assert [(item["horizon"], item["trading_days"]) for item in forecast["forecasts"]] == [
        ("1_day", 1),
        ("1_week", 5),
        ("1_month", 21),
    ]


def test_ceo_packet_reuses_ceo_path_without_deterministic_forecast() -> None:
    packet = build_market_packet(_prices(), ticker="J", company="Jacobs Solutions Inc.", metadata={}, bar_count=5)
    forecast = normalize_forecast({"method": "pure_llm", "forecasts": []}, packet)

    ceo_packet = build_ceo_technical_packet(packet=packet, forecast=forecast)

    assert ceo_packet["source_path"] == "pure_llm_stock_forecaster"
    assert ceo_packet["forecast_policy"]["deterministic_forecast_engine_used"] is False
    assert ceo_packet["decision_view"]["execution_policy"]["broker_order_submission"] is False


def test_compact_ceo_handoff_packet_removes_duplicate_views_and_reasoning() -> None:
    packet = build_market_packet(_prices(), ticker="J", company="Jacobs Solutions Inc.", metadata={}, bar_count=8)
    packet["external_llm_evidence"] = {
        "candidate": {"ticker": "J", "candidate_reason": "Based on the available evidence, the stock has a fresh contract catalyst."},
        "market_intelligence_llm": {
            "status": "executed",
            "summary": "Given the available context, market breadth is stable.",
        },
        "fresh_news_llm": {
            "status": "executed",
            "news_summary": "It is important to note that the company won an infrastructure award.",
            "reasoning": "internal news reasoning",
            "llm_raw_response": {"large": "raw"},
        },
        "fundamentals_llm": {
            "status": "executed",
            "business_quality_read": "The company has a resilient backlog.",
            "scratchpad": "internal fundamental reasoning",
        },
        "long_term_source_synthesis_llm": {
            "status": "executed",
            "synthesis": "In my view, the position has a long-term infrastructure theme.",
        },
        "strategy_knowledge_context": {
            "status": "executed",
            "synthesis": {"entry_rules": ["Former support turning resistance requires patience."]},
        },
        "market_structure_liquidity_read": {
            "status": "executed",
            "structure_bias": "bearish",
            "ceo_implication": "Avoid buying into retest resistance after downside liquidity sweep.",
        },
    }
    forecast = normalize_forecast(
        {
            "method": "pure_llm",
            "forecasts": [
                {
                    "horizon": "1_day",
                    "trading_days": 1,
                    "direction": "flat",
                    "predicted_price": 117,
                    "expected_return_pct": 0,
                    "confidence": 0.5,
                    "bear_case_price": 115,
                    "bull_case_price": 119,
                    "reasoning": "I need to explain this forecast internally.",
                    "key_invalidations": ["the market breaks support"],
                },
                {
                    "horizon": "1_week",
                    "trading_days": 5,
                    "direction": "up",
                    "predicted_price": 120,
                    "expected_return_pct": 2,
                    "confidence": 0.6,
                    "bear_case_price": 114,
                    "bull_case_price": 123,
                    "reasoning": "based on the data, price can rise",
                    "key_invalidations": [],
                },
                {
                    "horizon": "1_month",
                    "trading_days": 21,
                    "direction": "up",
                    "predicted_price": 125,
                    "expected_return_pct": 5,
                    "confidence": 0.55,
                    "bear_case_price": 110,
                    "bull_case_price": 130,
                    "reasoning": "it appears that trend is improving",
                    "key_invalidations": [],
                },
            ],
            "overall_view": "The stock is a cautious hold.",
            "main_risks": ["the stock reverses"],
            "data_limitations": [],
        },
        packet,
    )
    quality = {"status": "executed", "reasoning": "private quality reasoning", "quality_summary": "The forecast is usable."}

    handoff = build_compact_ceo_handoff_packet(packet=packet, forecast=forecast, forecast_quality=quality)
    handoff_text = pure_llm_stock_forecaster.json.dumps(handoff, sort_keys=True, default=str)

    assert "technical_view" not in handoff
    assert "decision_view" not in handoff
    assert "pure_llm_forecast" not in handoff_text
    assert "llm_raw_response" not in handoff_text
    assert "internal news reasoning" not in handoff_text
    assert "private quality reasoning" not in handoff_text
    assert "stock has fresh contract catalyst" in handoff_text
    assert "market breadth is stable" in handoff_text
    assert "company won infrastructure award" in handoff_text
    assert "Former support turning resistance requires patience" in handoff_text
    assert "Avoid buying into retest resistance" in handoff_text
    assert "J" in handoff_text


def test_default_external_evidence_runs_for_single_ticker(monkeypatch) -> None:
    args = build_parser().parse_args(["--ticker", "J", "--company", "Jacobs Solutions Inc.", "--no-progress"])
    calls: list[str] = []

    monkeypatch.setattr(
        pure_llm_stock_forecaster,
        "load_external_evidence",
        lambda path: {"from_file": path},
    )

    def fake_build(args, *, prices, metadata):
        calls.append(args.ticker)
        return {
            "ticker": args.ticker,
            "fresh_news_llm": {"status": "executed"},
            "fundamentals_llm": {"status": "executed"},
            "long_term_source_synthesis_llm": {"status": "executed"},
        }

    monkeypatch.setattr(pure_llm_stock_forecaster, "build_default_llm_external_evidence", fake_build)

    evidence = load_or_build_external_evidence(args, prices=_prices(), metadata={"provider": "unit"})

    assert evidence["fresh_news_llm"]["status"] == "executed"
    assert calls == ["J"]


def test_default_external_evidence_can_be_skipped(monkeypatch) -> None:
    args = build_parser().parse_args(["--ticker", "J", "--skip-llm-evidence", "--no-progress"])
    monkeypatch.setattr(
        pure_llm_stock_forecaster,
        "build_default_llm_external_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not build")),
    )

    assert load_or_build_external_evidence(args, prices=_prices(), metadata={}) == {}


def test_portfolio_context_matches_ceo_prompt_inputs() -> None:
    args = build_parser().parse_args(
        [
            "--ticker",
            "J",
            "--holding-status",
            "owned",
            "--entry-price",
            "100",
            "--quantity",
            "3",
            "--position-value",
            "360",
            "--account-equity",
            "10000",
        ]
    )

    context = build_portfolio_context(args)

    assert context["holding_status"] == "owned"
    assert context["entry_price"] == 100
    assert context["quantity"] == 3
    assert context["position_value"] == 360
    assert context["account_equity"] == 10000


def test_ceo_advice_uses_original_ceo_prompt_and_schema(monkeypatch) -> None:
    args = build_parser().parse_args(["--ticker", "J", "--no-progress"])
    calls: list[dict[str, object]] = []
    fake_client = object()

    def fake_call_response(**kwargs):
        calls.append(kwargs)
        return kwargs, {"id": "ceo"}, {"decision": "Hold"}

    monkeypatch.setattr(pure_llm_stock_forecaster, "call_response", fake_call_response)
    monkeypatch.setattr(pure_llm_stock_forecaster, "openai_client_for_provider", lambda provider, *, timeout: fake_client)

    payload, raw_response, advice = call_ceo_advice(
        args=args,
        provider="openai",
        model=ModelName.GPT_5_4_2026_03_05.value,
        technical_packet={"ticker": "J"},
        portfolio_context={},
    )

    assert payload == calls[0]
    assert raw_response == {"id": "ceo"}
    assert advice["decision"] == "Hold"
    assert len(calls) == 1
    assert calls[0]["system_message"] == autonomous_trader.system_message
    assert calls[0]["user_message"] == autonomous_trader.user_message
    assert calls[0]["json_schema"] is autonomous_trader.json_schema
    assert calls[0]["client"] is fake_client
    assert calls[0]["model"] == ModelName.GPT_5_4_2026_03_05.value
    assert calls[0]["usage_context"]["purpose"] == "pure_llm_stock_ceo_original"


def test_call_response_with_fallback_retries_after_primary_failure(monkeypatch) -> None:
    args = build_parser().parse_args(
        [
            "--ticker",
            "J",
            "--llm-provider",
            "llm_studio",
            "--llm-model",
            "local-model",
            "--fallback-llm-provider",
            "openai",
            "--fallback-llm-model",
            ModelName.GPT_5_4_MINI_2026_03_17.value,
            "--no-progress",
        ]
    )
    calls = []

    def fake_call_response(**kwargs):
        calls.append(kwargs)
        if kwargs["provider"] == "llm_studio":
            raise RuntimeError("context too large")
        return kwargs, {"id": "fallback"}, {"decision": "Hold"}

    monkeypatch.setattr(pure_llm_stock_forecaster, "call_response", fake_call_response)
    monkeypatch.setattr(pure_llm_stock_forecaster, "openai_client_for_provider", lambda provider, *, timeout: None)

    provider, model, payload, raw_response, parsed = call_response_with_fallback(
        args=args,
        purpose="unit",
        provider="llm_studio",
        model="local-model",
        system_message="system",
        user_message="user",
        json_schema={"type": "json_schema", "name": "x", "schema": {"type": "object"}},
        item={},
        use_web_search=False,
    )

    assert provider == "openai"
    assert model == ModelName.GPT_5_4_MINI_2026_03_17.value
    assert raw_response == {"id": "fallback"}
    assert parsed["decision"] == "Hold"
    assert parsed["_fallback_from"] == "llm_studio"
    assert [call["provider"] for call in calls] == ["llm_studio", "openai"]
