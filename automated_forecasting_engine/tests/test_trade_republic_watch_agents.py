from pathlib import Path
import json

from market_forecasting_engine.trade_republic_watch_agents import (
    build_agent_plans,
    calendar_for_ticker,
    live_price_provider_for_ticker,
)


def _report():
    return {
        "summary": {
            "holding_count": 3,
            "total_current_value": 108.45,
            "total_unrealized_pl": -5.26,
            "total_unrealized_pl_pct": -4.62,
        },
        "holdings": [
            {
                "isin": "IE00BYZK4552",
                "ticker": "2B76.DE",
                "alpaca_ticker": "",
                "name": "Automation & Robotics USD (Acc)",
                "current_quantity": 1.046876,
                "current_price": 17.888,
                "current_value": 18.73,
                "broker_avg_cost": 18.1512,
                "unrealized_pl": -0.272,
                "unrealized_pl_pct": -1.43,
            },
            {
                "isin": "US67066G1040",
                "ticker": "NVD.DE",
                "alpaca_ticker": "NVDA",
                "name": "NVIDIA",
                "current_quantity": 0.059796,
                "current_price": 195.04,
                "current_value": 11.66,
                "broker_avg_cost": 195.8304,
                "unrealized_pl": -0.049,
                "unrealized_pl_pct": -0.42,
            },
            {
                "isin": "US01609W1027",
                "ticker": "",
                "alpaca_ticker": "BABA",
                "name": "Alibaba ADR",
                "current_quantity": 0.1,
                "current_value": 12.0,
                "broker_avg_cost": 100.0,
            },
        ],
    }


def test_trade_republic_watch_agent_plans_prefer_portfolio_tickers_and_include_context(tmp_path) -> None:
    plans = build_agent_plans(
        report=_report(),
        project_dir=tmp_path,
        state_dir="state",
        profile="medium",
        refresh_after_hours="12",
        interval_seconds="3600",
        stagger_seconds="300",
        llm_env_file=str(tmp_path / ".env"),
        quiet_unchanged=True,
    )

    assert [plan["ticker"] for plan in plans] == ["2B76.DE", "NVD.DE", "BABA"]
    assert plans[0]["calendar"] == "XETR"
    assert plans[0]["live_price_provider"] == "yahoo"
    assert plans[1]["environment"]["TICKER"] == "NVD.DE"
    assert plans[1]["environment"]["HOLDING_STATUS"] == "owned"
    assert plans[1]["environment"]["ENTRY_PRICE"] == "195.8304"
    assert plans[1]["environment"]["QUANTITY"] == "0.059796"
    assert plans[1]["environment"]["ACCOUNT_EQUITY"] == "108.45"
    assert plans[0]["environment"]["STARTUP_DELAY_SECONDS"] == "0"
    assert plans[1]["environment"]["STARTUP_DELAY_SECONDS"] == "300"
    assert plans[2]["environment"]["STARTUP_DELAY_SECONDS"] == "600"
    assert plans[1]["environment"]["PORTFOLIO_CONTEXT_FILE"].endswith("nvd.de_medium.json")
    assert plans[1]["context"]["broker"] == "trade_republic"
    assert plans[1]["context"]["name"] == "NVIDIA"
    assert plans[1]["context"]["position"]["holding_status"] == "owned"
    assert plans[1]["context"]["position"]["avg_cost"] == 195.8304
    assert isinstance(plans[1]["plist_path"], Path)
    assert plans[2]["live_price_provider"] == "auto"


def test_trade_republic_watch_agent_calendar_and_provider_mapping() -> None:
    assert calendar_for_ticker("ASML.AS") == "XAMS"
    assert calendar_for_ticker("NVD.DE") == "XETR"
    assert calendar_for_ticker("BDT.TO") == "XTSE"
    assert calendar_for_ticker("ETH-USD") == "CRYPTO"
    assert calendar_for_ticker("KEYS") == "XNYS"
    assert live_price_provider_for_ticker("NVD.DE") == "yahoo"
    assert live_price_provider_for_ticker("BDT.TO") == "yahoo"
    assert live_price_provider_for_ticker("KEYS") == "auto"


def test_trade_republic_watch_agent_plan_preserves_existing_portfolio_brain_context(tmp_path) -> None:
    context_dir = tmp_path / "state" / "portfolio_contexts"
    context_dir.mkdir(parents=True)
    (context_dir / "nvd.de_medium.json").write_text(
        json.dumps({"portfolio_brain": {"coordinated_action": "HOLD", "risk_budget": {"hrp_weight": 0.2}}}),
        encoding="utf-8",
    )

    plans = build_agent_plans(
        report=_report(),
        project_dir=tmp_path,
        state_dir="state",
        profile="medium",
        refresh_after_hours="12",
        interval_seconds="3600",
        stagger_seconds="300",
        llm_env_file=str(tmp_path / ".env"),
        quiet_unchanged=True,
    )

    nvd = plans[1]["context"]
    assert nvd["source"] == "trade_republic_investment_report_with_portfolio_brain"
    assert nvd["portfolio_brain"]["risk_budget"]["hrp_weight"] == 0.2
