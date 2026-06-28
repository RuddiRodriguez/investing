from __future__ import annotations

from market_forecasting_engine.llm_model_catalog import DEFAULT_FULL_LLM_OPTIONS_MODEL, DEFAULT_FULL_LLM_OPTIONS_PROVIDER, LocalModel
from market_forecasting_engine.openai_models import ModelName
from market_forecasting_engine.pure_llm_virtual_trader_agent import (
    PureLLMVirtualTraderConfig,
    build_order_plan,
    build_parser,
    handoff_evidence_packet,
    handoff_fresh_news,
    evidence_for_ticker,
    limit_order_payload,
    normalize_candidates,
    normalize_tickers,
    run_pure_llm_forecast,
    trade_notional,
)


def test_parser_defaults_to_local_priority_with_openai_fallback() -> None:
    args = build_parser().parse_args(["--env-file", "/tmp/paper.env"])

    assert args.planner_provider == DEFAULT_FULL_LLM_OPTIONS_PROVIDER
    assert args.planner_model == DEFAULT_FULL_LLM_OPTIONS_MODEL
    assert args.planner_model == LocalModel.QWEN3_VL_8B_INSTRUCT_MLX
    assert args.fallback_provider == "openai"
    assert args.fallback_model == ModelName.GPT_5_4_MINI_2026_03_17.value
    assert args.risk_profile == "aggressive"
    assert args.trader_profile == "aggressive"
    assert args.compact_llm_handoffs is True


def test_parser_can_disable_compact_llm_handoffs() -> None:
    args = build_parser().parse_args(["--env-file", "/tmp/paper.env", "--no-compact-llm-handoffs"])

    assert args.compact_llm_handoffs is False


def test_normalize_candidates_prioritizes_broker_visible_symbols() -> None:
    broker_state = {
        "positions": [{"symbol": "WFC"}],
        "open_orders": [{"symbol": "J"}],
    }

    candidates = normalize_candidates(
        [{"ticker": "J", "company": "Jacobs", "priority": "medium", "candidate_reason": "LLM scout", "main_catalysts": ["x"], "main_risks": []}],
        broker_state=broker_state,
        max_candidates=3,
    )

    assert [row["ticker"] for row in candidates] == ["WFC", "J"]
    assert candidates[0]["candidate_reason"] == "Existing Alpaca position or open order needs LLM review."


def test_normalize_tickers_merges_candidates_after_positions() -> None:
    broker_state = {"positions": [{"symbol": "WFC"}], "open_orders": []}

    tickers = normalize_tickers(["J"], broker_state=broker_state, candidates=[{"ticker": "BAC"}], max_candidates=3)

    assert tickers == ["WFC", "BAC", "J"]


def test_aggressive_trade_notional_respects_equity_buying_power_and_caps() -> None:
    config = PureLLMVirtualTraderConfig(risk_profile="aggressive", max_notional_per_trade=25, max_position_pct_equity=0.25)

    assert trade_notional(equity=40.34, buying_power=40.34, config=config) == 10.085


def test_limit_order_payload_uses_limit_order_not_market() -> None:
    payload = limit_order_payload(ticker="J", side="buy", qty=0.1, limit_price=120.977, prefix="test")

    assert payload["symbol"] == "J"
    assert payload["side"] == "buy"
    assert payload["order_type"] == "limit"
    assert payload["limit_price"] == 120.98
    assert payload["time_in_force"] == "day"


def test_build_order_plan_blocks_closed_market_by_default() -> None:
    config = PureLLMVirtualTraderConfig(dry_run=True)
    broker_state = {
        "status": "ok",
        "account": {"equity": "40.34", "buying_power": "40.34"},
        "clock": {"is_open": False},
        "positions": [],
        "open_orders": [],
    }
    report = {
        "ticker": "J",
        "current_price": 120.97,
        "llm_final_decision": {"decision": "Buy"},
        "final_advice": {"buy_now_price": 120.5},
    }

    plan = build_order_plan(report=report, broker_state=broker_state, config=config)

    assert plan["action"] == "buy"
    assert plan["execution_allowed"] is False
    assert "alpaca_market_closed" in plan["execution_blocks"]
    assert plan["order_payload"] is None


def test_evidence_for_ticker_selects_matching_packet() -> None:
    packet = evidence_for_ticker([{"ticker": "J", "fresh_news_llm": {"status": "executed"}}], "j")

    assert packet["ticker"] == "J"
    assert packet["fresh_news_llm"]["status"] == "executed"


def test_compact_handoff_strips_raw_llm_baggage_without_dropping_evidence() -> None:
    config = PureLLMVirtualTraderConfig(compact_llm_handoffs=True)
    handoff = handoff_fresh_news(
        {
            "ticker": "J",
            "status": "executed",
            "provider": "openai",
            "model": "gpt-test",
            "freshness_read": "fresh\n\ncontext   with spacing",
            "news_summary": "summary   keeps all facts",
            "bullish_news": [f"bull item {index}" for index in range(6)],
            "bearish_news": [],
            "events_to_watch": [],
            "source_notes": ["source   note"],
            "limitations": [],
            "llm_prompt_payload": {"large": "prompt"},
            "llm_raw_response": {"large": "raw"},
        },
        config=config,
    )

    assert "llm_prompt_payload" not in handoff
    assert "llm_raw_response" not in handoff
    assert "provider" not in handoff
    assert "model" not in handoff
    assert handoff["freshness_read"] == "fresh context with spacing"
    assert handoff["source_notes"] == ["source note"]
    assert handoff["bullish_news"] == [f"bull item {index}" for index in range(6)]


def test_compact_handoff_removes_agent_filler_without_removing_facts() -> None:
    config = PureLLMVirtualTraderConfig(compact_llm_handoffs=True)
    handoff = handoff_fresh_news(
        {
            "ticker": "J",
            "status": "executed",
            "freshness_read": "I need to review the company news. Based on the available evidence, the stock has a fresh contract momentum.",
            "news_summary": "It is important to note that J reported a $250M backlog update; this suggests that the market may re-rate the position as an opportunity.",
            "bullish_news": ["Please note that the company won an infrastructure award."],
            "bearish_news": ["In my view, the stock still faces margin risk."],
            "reasoning": "I think through the trade in several internal steps.",
            "scratchpad": "private intermediate reasoning",
            "events_to_watch": [],
            "source_notes": [],
            "limitations": [],
        },
        config=config,
    )

    assert "reasoning" not in handoff
    assert "scratchpad" not in handoff
    assert handoff["freshness_read"] == "review company news. stock has fresh contract momentum."
    assert handoff["news_summary"] == "J reported $250M backlog update; market may re-rate position as opportunity."
    assert handoff["bullish_news"] == ["company won infrastructure award."]
    assert handoff["bearish_news"] == ["stock still faces margin risk."]


def test_no_compact_handoff_preserves_full_packet() -> None:
    config = PureLLMVirtualTraderConfig(compact_llm_handoffs=False)
    packet = {"ticker": "J", "llm_raw_response": {"kept": True}}

    assert handoff_evidence_packet(packet, config=config) is packet


def test_forecast_subprocess_receives_external_evidence(monkeypatch, tmp_path) -> None:
    calls = []

    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append(command)
        return FakeResult()

    monkeypatch.setattr("market_forecasting_engine.pure_llm_virtual_trader_agent.subprocess.run", fake_run)
    config = PureLLMVirtualTraderConfig(
        output_root=tmp_path / "out",
        memory_path=tmp_path / "memory.json",
        env_file=tmp_path / "paper.env",
        progress=False,
    )

    row = run_pure_llm_forecast(
        ticker="J",
        config=config,
        cycle_dir=tmp_path / "cycle",
        broker_state={"account": {"equity": "40.34"}, "positions": []},
        evidence_packet={"ticker": "J", "fresh_news_llm": {"status": "executed"}},
    )

    evidence_flag_index = calls[0].index("--external-evidence-json")
    evidence_path = calls[0][evidence_flag_index + 1]
    assert row["returncode"] == 0
    assert evidence_path.endswith("llm_evidence_packet.json")
    assert "--external-evidence-json" in calls[0]
