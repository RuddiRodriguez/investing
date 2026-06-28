from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

from market_forecasting_engine.unified_portfolio_dashboard import ask_portfolio_assistant
from market_forecasting_engine.unified_portfolio import (
    attach_holding_classifications,
    build_portfolio_rebalancing,
    build_snaptrade_portfolio_summaries,
    build_trade_republic_portfolio_summary,
    build_portfolio_benchmark,
    build_portfolio_correlation,
    build_portfolio_trade_alpha,
    build_unified_portfolio_state,
    build_weekly_cleanup_review,
    create_price_alert,
    evaluate_price_alerts,
    goal_action_for_holding,
    heuristic_classification,
    next_weekly_cleanup_date,
    normalize_trade_republic_dividends,
    normalize_snaptrade_snapshot,
    normalize_trade_republic_report,
)


def test_normalize_trade_republic_report() -> None:
    rows = normalize_trade_republic_report(
        {
            "summary": {"report_timestamp": "2026-06-17T12:00:00+02:00"},
            "holdings": [
                {
                    "name": "NVIDIA",
                    "isin": "US67066G1040",
                    "ticker": "NVD.DE",
                    "alpaca_ticker": "NVDA",
                    "current_quantity": 0.2,
                    "current_price": 180.0,
                    "current_value": 36.0,
                    "open_cost_basis": 30.0,
                    "unrealized_pl": 6.0,
                    "unrealized_pl_pct": 20.0,
                    "historical_price_status": "matched",
                }
            ],
        }
    )

    assert rows[0]["broker"] == "trade_republic"
    assert rows[0]["account_key"] == "trade_republic:trade_republic"
    assert rows[0]["ticker"] == "NVDA"
    assert rows[0]["current_value"] == 36.0


def test_trade_republic_summary_separates_open_cost_basis_from_account_invested() -> None:
    report = {
        "summary": {
            "total_current_value": 144.38,
            "total_open_cost_basis": 138.8,
            "total_unrealized_pl": 5.58,
            "total_historical_buy_cash": 153.71,
            "total_historical_sell_cash": 9.0,
        },
        "holdings": [
            {
                "historical_sell_cash": 9.0,
                "transaction_sell_shares": 0.01,
                "weighted_paid_price": 100.0,
            }
        ],
    }

    result = build_trade_republic_portfolio_summary(report, holdings=[], dividends=[])

    assert result["total_worth"] == 144.38
    assert result["holdings_cost_basis"] == 138.8
    assert result["account_invested_capital"] == 144.71
    assert result["unrealized_pl"] == 5.58
    assert result["realized_pl"] == 8.0
    assert result["capital_gain"] == 13.58


def test_normalize_trade_republic_dividends() -> None:
    rows = normalize_trade_republic_dividends(
        {
            "dividends": [
                {
                    "date": "2026-06-10",
                    "timestamp": "2026-06-10T08:29:55",
                    "name": "Exxon Mobil",
                    "isin": "US30231G1022",
                    "ticker": "XOM",
                    "shares": 0.04,
                    "after_tax_amount": 0.02,
                    "tax_amount": 0.01,
                    "gross_amount": 0.03,
                    "currency": "EUR",
                }
            ]
        }
    )

    assert rows[0]["account_key"] == "trade_republic:trade_republic"
    assert rows[0]["after_tax_amount"] == 0.02


def test_normalize_snaptrade_snapshot_handles_account_positions() -> None:
    rows = normalize_snaptrade_snapshot(
        {
            "accounts": [
                {
                    "account_id": "acct-1",
                    "account": {
                        "id": "acct-1",
                        "name": "BUX",
                        "brokerage": {"name": "BUX"},
                        "currency": {"code": "EUR"},
                    },
                    "positions": {
                        "results": [
                            {
                                "units": 2,
                                "price": 25.0,
                                "average_purchase_price": 20.0,
                                "instrument": {"symbol": "ABC", "description": "ABC Corp", "kind": "stock"},
                                "currency": {"code": "EUR"},
                            }
                        ]
                    },
                }
            ]
        }
    )

    assert rows == [
        {
            "broker": "snaptrade",
            "source": "snaptrade_snapshot",
            "account_id": "acct-1",
            "account_name": "BUX",
            "account_key": "snaptrade:acct-1",
            "institution": "BUX",
            "name": "ABC Corp",
            "ticker": "ABC",
            "broker_symbol": "ABC",
            "isin": None,
            "asset_type": "stock",
            "quantity": 2.0,
            "current_price": 25.0,
            "current_value": 50.0,
            "cost_basis": 40.0,
            "unrealized_pl": 10.0,
            "unrealized_pl_pct": 25.0,
            "currency": "EUR",
            "instrument_currency": "EUR",
            "price_currency_note": "SnapTrade position price/value/cost basis are treated as account currency.",
            "status": "reported",
            "first_buy_date": None,
            "buy_date_source": "missing",
            "raw": {
                "units": 2,
                "price": 25.0,
                "average_purchase_price": 20.0,
                "instrument": {"symbol": "ABC", "description": "ABC Corp", "kind": "stock"},
                "currency": {"code": "EUR"},
            },
        }
    ]


def test_snaptrade_summary_separates_cash_cost_basis_and_realized_growth() -> None:
    snapshot = {
        "accounts": [
            {
                "account_id": "acct",
                "account": {"id": "acct", "name": "BUX", "brokerage": {"name": "BUX"}},
                "balances": {"total": {"amount": 61.0, "currency": "EUR"}},
                "activities": [],
            }
        ]
    }
    holdings = [
        {
            "account_key": "snaptrade:acct",
            "current_value": 35.84,
            "cost_basis": 40.36,
            "unrealized_pl": -4.52,
            "currency": "EUR",
        }
    ]

    result = build_snaptrade_portfolio_summaries(snapshot, holdings)[0]

    assert result["equities"] == 35.84
    assert result["cash"] == 25.16
    assert result["total_worth"] == 61.0
    assert result["holdings_cost_basis"] == 40.36
    assert result["account_invested_capital"] == 65.52
    assert result["unrealized_pl"] == -4.52


def test_build_unified_portfolio_state_merges_sources(tmp_path: Path) -> None:
    trade_report = tmp_path / "tr.json"
    trade_report.write_text(
        json.dumps(
            {
                "summary": {"report_timestamp": "2026-06-17T12:00:00+02:00"},
                "holdings": [
                    {
                        "name": "Trade Holding",
                        "ticker": "TRD",
                        "current_quantity": 1,
                        "current_price": 10,
                        "current_value": 10,
                        "open_cost_basis": 8,
                        "unrealized_pl": 2,
                    }
                ],
                "dividends": [
                    {
                        "date": "2026-06-10",
                        "timestamp": "2026-06-10T08:29:55",
                        "name": "Trade Dividend",
                        "after_tax_amount": 0.2,
                        "tax_amount": 0.1,
                        "gross_amount": 0.3,
                        "currency": "EUR",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    snap = tmp_path / "snap.json"
    snap.write_text(
        json.dumps(
            {
                "fetched_at": "2026-06-17T12:01:00+02:00",
                "accounts": [
                    {
                        "account_id": "acct",
                        "account": {"id": "acct", "name": "Alpaca Paper", "brokerage": {"name": "Alpaca"}},
                        "positions": [{"symbol": "SPY", "quantity": 1, "price": 500, "average_purchase_price": 490}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    scenario_history = tmp_path / "scenario_history.json"
    scenario_history.write_text(
        json.dumps(
            {
                "fetched_at": "2026-06-17T12:02:00+02:00",
                "source": "test_cache",
                "scenarios": {
                    "financial_crisis_2008": {
                        "prices": {
                            "TRD": {"status": "ok", "return_pct": -20.0},
                            "SPY": {"status": "ok", "return_pct": -30.0},
                        }
                    },
                    "covid_2020": {
                        "prices": {
                            "TRD": {"status": "ok", "return_pct": -10.0},
                            "SPY": {"status": "ok", "return_pct": -15.0},
                        }
                    },
                    "rate_hike_2022": {
                        "prices": {
                            "TRD": {"status": "ok", "return_pct": -5.0},
                            "SPY": {"status": "ok", "return_pct": -12.0},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    state = build_unified_portfolio_state(trade_republic_report=trade_report, snaptrade_snapshot=snap, scenario_history_path=scenario_history)

    assert state["summary"]["holding_count"] == 2
    assert state["summary"]["total_current_value"] == 510.0
    assert state["summary"]["total_cost_basis"] == 498.0
    assert {row["broker"] for row in state["holdings"]} == {"trade_republic", "snaptrade"}
    assert state["dividends"][0]["after_tax_amount"] == 0.2
    assert state["combined_portfolio"]["dividend_after_tax"] == 0.2
    monte_carlo = state["analytics"]["monte_carlo"]["combined:live"]
    assert monte_carlo["path_count"] == 1000
    assert monte_carlo["horizon_days"] == 252
    assert monte_carlo["starting_value"] == state["combined_portfolio"]["total_worth"]
    assert {"p5", "p25", "median", "p75", "p95"}.issubset(monte_carlo["percentiles"])
    assert len(monte_carlo["sample_paths"]) > 0
    risk = state["analytics"]["risk"]["combined:live"]
    assert risk["portfolio_value"] == state["combined_portfolio"]["total_worth"]
    assert {"var_95_pct", "var_99_pct", "volatility_pct", "expected_return_pct", "sharpe_ratio", "max_drawdown_pct"}.issubset(risk)
    frontier = state["analytics"]["frontier"]["combined:live"]
    assert frontier["portfolio_value"] == state["combined_portfolio"]["total_worth"]
    assert frontier["asset_count"] >= 2
    assert frontier["current_portfolio"] is not None
    assert frontier["frontier_points"]
    assert frontier["max_sharpe"] is not None
    scenarios = state["analytics"]["scenarios"]["combined:live"]["scenarios"]
    assert len(scenarios) == 3
    assert scenarios[0]["status"] == "ready"
    assert scenarios[0]["coverage_pct"] == 100.0
    assert scenarios[0]["estimated_return_pct"] < 0
    goal = state["goal_plan"]
    assert goal["name"] == "100 EUR Net Gain Plan"
    assert goal["read_only"] is True
    assert goal["execution_enabled"] is False
    assert goal["target_gain"] == 100.0
    assert goal["paper_excluded"] is True
    assert goal["remaining_gain"] == 88.0
    assert goal["drawdown_budget"] == 24.6
    assert goal["stop_adding_risk_at_gain"] == -27.42
    assert "No options recommendation" in " ".join(goal["guardrails"])
    assert any(row.get("goal_action", {}).get("read_only") for row in state["holdings"])
    assert state["source_errors"] == []


def test_goal_action_labels_core_opportunity_cleanup_and_review() -> None:
    core = goal_action_for_holding({"ticker": "VWCE.DE", "name": "Core ETF", "current_value": 50, "cost_basis": 50}, total_worth=250)
    opportunity = goal_action_for_holding({"ticker": "NVDA", "name": "NVIDIA", "current_value": 20, "cost_basis": 20}, total_worth=250)
    cleanup = goal_action_for_holding({"ticker": "TMC", "name": "Tiny", "current_value": 1.5, "cost_basis": 2.0}, total_worth=250)
    review = goal_action_for_holding({"ticker": "ABC", "name": "Loss", "current_value": 90, "cost_basis": 100}, total_worth=250)

    assert core["action"] == "hold_core"
    assert opportunity["action"] == "watch_opportunity"
    assert cleanup["action"] == "cleanup_watch"
    assert review["action"] == "review_loss"
    assert all(item["read_only"] is True for item in [core, opportunity, cleanup, review])


def test_weekly_cleanup_review_uses_next_tuesday_and_specific_candidates() -> None:
    assert next_weekly_cleanup_date("2026-06-23") == "2026-06-30"
    review = build_weekly_cleanup_review(
        {"key": "combined:live", "total_worth": 250.0},
        [
            {"ticker": "BBAI", "name": "BigBear.ai", "current_value": 1.4, "cost_basis": 2.0, "unrealized_pl_pct": -30.0},
            {"ticker": "AIR", "name": "Airbus", "current_value": 1.8, "cost_basis": 1.7, "unrealized_pl_pct": 5.0},
            {"ticker": "VWCE.DE", "name": "Core ETF", "current_value": 44.0, "cost_basis": 42.0, "unrealized_pl_pct": 4.0},
        ],
        generated_at="2026-06-23T11:00:00+00:00",
    )

    assert review["review_date"] == "2026-06-30"
    assert review["automation_id"] == "weekly-portfolio-cleanup-review"
    assert review["cleanup_candidates"][0]["symbol"] == "BBAI"
    assert "never add" in review["cleanup_candidates"][0]["plain_action"]
    assert review["not_urgent"][0]["symbol"] == "AIR"
    assert review["read_only"] is True
    assert review["execution_enabled"] is False


def test_build_portfolio_benchmark_aligns_returns() -> None:
    portfolio = {
        "key": "portfolio:test",
        "name": "Test Portfolio",
        "currency": "EUR",
        "total_worth": 120.0,
    }
    history_rows = [
        {"date": "2026-06-17", "timestamp": "2026-06-17", "portfolio_key": "portfolio:test", "total_worth": 100.0},
        {"date": "2026-06-18", "timestamp": "2026-06-18", "portfolio_key": "portfolio:test", "total_worth": 110.0},
        {"date": "2026-06-19", "timestamp": "2026-06-19", "portfolio_key": "portfolio:test", "total_worth": 120.0},
    ]
    benchmark_history = {
        "fetched_at": "2026-06-19T10:00:00+00:00",
        "source": "test_cache",
        "symbol": "VOO",
        "name": "Vanguard 500 Index Fund",
        "prices": [
            {"date": "2026-06-17", "close": 500.0},
            {"date": "2026-06-18", "close": 505.0},
            {"date": "2026-06-19", "close": 510.0},
        ],
    }

    result = build_portfolio_benchmark(portfolio, history_rows, benchmark_history)

    assert result["status"] == "limited_history"
    assert result["aligned_point_count"] == 3
    assert result["metrics"]["total_return_pct"] == 20.0
    assert result["metrics"]["benchmark_total_return_pct"] == 2.0
    assert result["metrics"]["excess_return_pct"] == 18.0
    assert {"alpha_pct", "beta", "correlation", "tracking_error_pct", "information_ratio"}.issubset(result["metrics"])
    assert result["series"][-1]["portfolio_return_pct"] == 20.0
    assert result["series"][-1]["benchmark_return_pct"] == 2.0


def test_classification_cache_attaches_metadata() -> None:
    holdings = [{"ticker": "ASML", "name": "ASML Holding", "asset_type": "stock", "currency": "EUR"}]
    cache = {
        "classifications": {
            "ticker:ASML": {
                "asset_class": "Stock",
                "sector": "Semiconductors",
                "geography": "Europe",
                "confidence": "high",
                "source": "test",
            }
        }
    }

    attach_holding_classifications(holdings, cache)

    assert holdings[0]["classification"]["sector"] == "Semiconductors"
    assert holdings[0]["classification"]["geography"] == "Europe"
    assert holdings[0]["classification"]["source"] == "test"


def test_heuristic_classification_uses_fixed_enums() -> None:
    result = heuristic_classification({"ticker": "DFNC.DE", "name": "iShares Europe Defence UCITS ETF", "asset_type": "etf", "currency": "EUR"})

    assert result["asset_class"] == "ETF"
    assert result["sector"] == "Defense & Aerospace"
    assert result["geography"] in {"Europe", "Germany"}


def test_build_portfolio_correlation_from_cached_prices() -> None:
    portfolio = {"key": "acct:1", "name": "Test", "currency": "EUR", "total_worth": 300.0}
    holdings = [
        {"account_key": "acct:1", "ticker": "AAA", "name": "Asset A", "current_value": 100.0},
        {"account_key": "acct:1", "ticker": "BBB", "name": "Asset B", "current_value": 100.0},
        {"account_key": "acct:1", "ticker": "CCC", "name": "Asset C", "current_value": 100.0},
    ]
    price_history = {
        "source": "test_cache",
        "fetched_at": "2026-06-19T10:00:00+00:00",
        "prices": {
            "AAA": [
                {"date": "2026-06-15", "close": 100.0},
                {"date": "2026-06-16", "close": 101.0},
                {"date": "2026-06-17", "close": 102.0},
                {"date": "2026-06-18", "close": 103.0},
            ],
            "BBB": [
                {"date": "2026-06-15", "close": 200.0},
                {"date": "2026-06-16", "close": 202.0},
                {"date": "2026-06-17", "close": 204.0},
                {"date": "2026-06-18", "close": 206.0},
            ],
            "CCC": [
                {"date": "2026-06-15", "close": 100.0},
                {"date": "2026-06-16", "close": 99.0},
                {"date": "2026-06-17", "close": 98.0},
                {"date": "2026-06-18", "close": 97.0},
            ],
        },
    }

    result = build_portfolio_correlation(portfolio, holdings, price_history, min_periods=3)

    assert result["status"] == "limited_history"
    assert result["used_asset_count"] == 3
    assert result["symbols"][0]["symbol"] == "AAA"
    assert result["matrix"][0][0] == 1.0
    assert len(result["matrix"]) == 3
    assert result["pairs"]


def test_build_portfolio_trade_alpha_compares_holding_to_benchmark() -> None:
    portfolio = {"key": "acct:1", "name": "Test", "currency": "EUR"}
    holdings = [
        {
            "account_key": "acct:1",
            "ticker": "AAA",
            "name": "Asset A",
            "cost_basis": 100.0,
            "current_value": 120.0,
            "first_buy_date": "2026-06-17",
            "buy_date_source": "test",
        }
    ]
    benchmark_history = {
        "symbol": "VOO",
        "name": "Vanguard 500 Index Fund",
        "source": "test_cache",
        "prices": [
            {"date": "2026-06-17", "close": 100.0},
            {"date": "2026-06-18", "close": 105.0},
        ],
    }

    result = build_portfolio_trade_alpha(portfolio, holdings, benchmark_history)

    assert result["status"] == "ready"
    assert result["your_return_value"] == 20.0
    assert result["benchmark_return_value"] == 5.0
    assert result["alpha_value"] == 15.0
    assert result["beat_count"] == 1
    assert result["rows"][0]["alpha_pct"] == 15.0


def test_price_alert_triggers_once_and_records_event(tmp_path: Path) -> None:
    alerts_path = tmp_path / "alerts.json"
    events_path = tmp_path / "events.jsonl"
    create_price_alert(
        alerts_path,
        portfolio_key="acct:1",
        symbol="AAA",
        name="Asset A",
        target_price=90.0,
        direction="below",
        basis_price=100.0,
        threshold_pct=10.0,
        currency="EUR",
    )
    portfolios = [{"key": "acct:1", "name": "Test", "kind": "individual", "currency": "EUR"}]
    holdings = [{"account_key": "acct:1", "ticker": "AAA", "name": "Asset A", "current_price": 89.0, "current_value": 89.0, "quantity": 1, "currency": "EUR"}]

    first = evaluate_price_alerts(alerts_path, events_path, portfolios, holdings)
    second = evaluate_price_alerts(alerts_path, events_path, portfolios, holdings)

    assert first["summary"]["triggered"] == 1
    assert second["summary"]["triggered"] == 1
    assert len(events_path.read_text(encoding="utf-8").splitlines()) == 1


def test_build_portfolio_rebalancing_calculates_drift() -> None:
    portfolio = {"key": "acct:1", "name": "Test", "currency": "EUR", "total_worth": 100.0}
    holdings = [
        {"ticker": "AAA", "name": "Asset A", "current_value": 70.0, "current_price": 10.0, "currency": "EUR"},
        {"ticker": "BBB", "name": "Asset B", "current_value": 30.0, "current_price": 5.0, "currency": "EUR"},
    ]
    targets = {"rows": [{"symbol": "AAA", "target_pct": 50.0}, {"symbol": "BBB", "target_pct": 50.0}]}

    result = build_portfolio_rebalancing(portfolio, holdings, targets)

    assert result["target_total_pct"] == 100.0
    assert result["rows"][0]["symbol"] == "AAA"
    assert result["rows"][0]["suggested_action"] == "Sell"
    assert result["rows"][0]["drift_value"] == -20.0
    assert result["rows"][1]["suggested_action"] == "Buy"


def test_portfolio_assistant_uses_common_llm_handler_with_selected_context() -> None:
    state = {
        "portfolio_summaries": [
            {
                "key": "acct:1",
                "name": "Test Portfolio",
                "kind": "individual",
                "currency": "EUR",
                "total_worth": 100.0,
                "capital_gain": 10.0,
            }
        ],
        "holdings": [
            {
                "account_key": "acct:1",
                "ticker": "AAA",
                "name": "Asset A",
                "quantity": 2,
                "current_price": 50.0,
                "current_value": 100.0,
                "cost_basis": 90.0,
                "unrealized_pl": 10.0,
                "unrealized_pl_pct": 11.11,
                "currency": "EUR",
            }
        ],
        "rebalancing": {"acct:1": {"rows": []}},
        "portfolio_history": [],
    }

    class FakeHandler:
        def __init__(self) -> None:
            self.request = None

        def predict(self, request):
            self.request = request
            return SimpleNamespace(
                provider="llm_studio",
                model=request.model,
                parsed={
                    "title": "Portfolio summary",
                    "answer": "The selected portfolio has one holding.",
                    "key_points": ["Asset A is the only holding."],
                    "numbers_used": ["Total worth: 100 EUR"],
                    "risks_or_limitations": ["Only local snapshot data was used."],
                    "follow_up_questions": ["Do you want concentration risk?"],
                },
            )

    handler = FakeHandler()
    result = ask_portfolio_assistant(
        state=state,
        portfolio_key="acct:1",
        question="summarize",
        provider="llm_studio",
        model="local-test",
        env_file=None,
        timeout_seconds=1,
        handler=handler,
    )

    assert result["title"] == "Portfolio summary"
    assert result["provider"] == "llm_studio"
    assert handler.request.provider == "llm_studio"
    assert handler.request.model == "local-test"
    payload_text = json.dumps(handler.request.payload)
    assert "Test Portfolio" in payload_text
    assert "Asset A" in payload_text
    assert "Use only this provided local dashboard state" in payload_text
