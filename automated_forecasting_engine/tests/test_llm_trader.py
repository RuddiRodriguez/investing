from market_forecasting_engine.llm_trader.profiles import trader_profiles
from market_forecasting_engine.llm_trader.responses_api import response_payload
from market_forecasting_engine.llm_trader.run import build_currency_context, build_technical_packet
from market_forecasting_engine.llm_trader.prompts import autonomous_trader
from market_forecasting_engine.llm_trader.prompts import nontechnical_summary


def test_autonomous_trader_payload_uses_responses_api_shape_and_web_search() -> None:
    payload = response_payload(
        model="gpt-5.4-mini-2026-03-17",
        system_message=autonomous_trader.system_message,
        user_message=autonomous_trader.user_message,
        json_schema=autonomous_trader.json_schema,
        reasoning_effort="low",
        item={
            "today": "2026-05-28",
            "ticker": "AAPL",
            "trader_name": "test_trader",
            "trader_profile_json": "{}",
            "portfolio_context_json": "{}",
            "technical_packet_json": "{}",
        },
        use_web_search=True,
        search_context_size="medium",
    )

    assert payload["model"] == "gpt-5.4-mini-2026-03-17"
    assert payload["tool_choice"] == "auto"
    assert payload["text"]["format"]["name"] == "autonomous_trader_decision"
    assert payload["reasoning"]["effort"] == "low"
    assert payload["input"][0]["role"] == "developer"
    assert payload["input"][1]["role"] == "user"
    assert payload["tools"][0]["type"] == "web_search"
    assert "AAPL" in payload["input"][1]["content"][0]["text"]


def test_nontechnical_summary_payload_uses_second_prompt_without_web_search() -> None:
    payload = response_payload(
        model="gpt-5.4-mini-2026-03-17",
        system_message=nontechnical_summary.system_message,
        user_message=nontechnical_summary.user_message,
        json_schema=nontechnical_summary.json_schema,
        reasoning_effort="low",
        item={
            "ticker": "ASML",
            "trader_name": "autonomous_trader_1",
            "trader_profile_json": '{"name":"medium"}',
            "portfolio_context_json": "{}",
            "currency_context_json": '{"status":"available","usd_to_eur":0.92}',
            "trader_decision_json": '{"decision":"Hold","confidence":0.86}',
            "technical_packet_json": '{"risk_level":"High"}',
        },
        use_web_search=False,
        search_context_size="low",
    )

    assert payload["text"]["format"]["name"] == "nontechnical_trader_summary"
    assert payload["tools"] == []
    assert payload["reasoning"]["effort"] == "low"
    assert "ASML" in payload["input"][1]["content"][0]["text"]
    assert "usd_to_eur" in payload["input"][1]["content"][0]["text"]
    assert "non-technical person" in payload["input"][0]["content"][0]["text"]
    important_price_schema = nontechnical_summary.json_schema["schema"]["properties"]["important_prices"]["items"]
    assert "price_usd" in important_price_schema["properties"]
    assert "price_eur" in important_price_schema["properties"]
    assert "display" in important_price_schema["properties"]
    assert "decision_triggers" in nontechnical_summary.json_schema["schema"]["properties"]
    assert "decision_triggers" in nontechnical_summary.json_schema["schema"]["required"]
    trigger_schema = nontechnical_summary.json_schema["schema"]["properties"]["decision_triggers"]["items"]
    assert trigger_schema["properties"]["decision_goal"]["enum"] == [
        "consider_buy",
        "consider_sell",
        "consider_hold",
        "reduce_risk",
        "take_profit",
        "keep_waiting",
        "manual_review",
    ]
    assert "if_not_owned_action" in trigger_schema["required"]
    assert "if_owned_action" in trigger_schema["required"]


def test_currency_context_accepts_manual_usd_to_eur_rate() -> None:
    class Args:
        usd_eur_rate = 0.92
        dry_run = False

    context = build_currency_context(Args())

    assert context["status"] == "available"
    assert context["usd_to_eur"] == 0.92
    assert context["source"] == "manual_cli_override"


def test_trader_profiles_cover_required_risk_styles() -> None:
    assert set(trader_profiles) == {"aggressive", "medium", "conservative"}
    assert trader_profiles["aggressive"]["risk_budget"] == "higher"
    assert trader_profiles["conservative"]["risk_budget"] == "lower"


def test_build_technical_packet_extracts_decision_and_risk_controls() -> None:
    report = {
        "ticker": "AAPL",
        "as_of_date": "2026-05-28",
        "current_price": 100.0,
        "suggested_action": "Hold",
        "part_i_suggested_action": "Hold",
        "risk_level": "Medium",
        "risk_warning": "Medium risk.",
        "forecasts": [
            {
                "horizon_days": 5,
                "selected_model": "recent_mean_return",
                "expected_direction": "Upward",
                "expected_return": 0.02,
                "directional_confidence": 0.60,
                "predicted_price": 102.0,
                "lower_price": 97.0,
                "upper_price": 106.0,
                "validation_metrics": {"mae": 0.01},
            }
        ],
        "technical_view": {
            "trend_state": {"state": "Bullish"},
            "chapter_13_support_resistance": {
                "support_zones": {"nearest": {"center": 95.0}},
                "resistance_zones": {"nearest": {"center": 110.0}},
            },
        },
        "decision_view": {
            "chapter_18_tactical_problem": {
                "final_action": "Hold",
                "rule_based_action": "Hold",
                "rule_gate": {"status": "Pass"},
                "trade_plan": {"entry_policy": "No new commitment."},
            }
        },
        "operations_view": {
            "chapter_19_validation": {
                "status": "pass",
                "action_gate": {"validated_action": "Hold"},
            }
        },
        "selection_view": {
            "chapter_20_ticker_suitability": {"profile_fit": {"primary_profile": "intermediate_trader"}},
            "chapter_21_chart_selection": {"chart_selection": {"chart_book_bucket": "active_review"}},
        },
        "trade_risk_view": {
            "chapter_23_30_trade_risk_plan": {
                "commitment": {"commitment_type": "active_review_no_new_commitment"},
                "execution_summary": {"initial_stop": None},
            }
        },
        "portfolio_view": {
            "chapter_31_42_portfolio_capital_risk": {
                "portfolio_capital_gate": {"allocation_status": "not_applicable"},
                "capital_summary": {"allocation_status": "not_applicable"},
            }
        },
        "discipline_view": {
            "chapter_39_43_discipline_governance": {
                "status": "consistent",
                "discipline_gate": {"plan_adherence": "consistent"},
            }
        },
        "backtests": {"5": {"sharpe_ratio": 0.4}},
        "selection_metric": "mae",
        "data_version": "abc",
        "model_version": "0.1.0",
    }

    packet = build_technical_packet(report)

    assert packet["ticker"] == "AAPL"
    assert packet["decision_governance"]["chapter_19_status"] == "pass"
    assert packet["decision_governance"]["trade_risk_commitment"]["commitment_type"] == "active_review_no_new_commitment"
    assert packet["decision_governance"]["portfolio_capital_gate"]["allocation_status"] == "not_applicable"
    assert packet["trend"]["chapter_13_support"]["center"] == 95.0
