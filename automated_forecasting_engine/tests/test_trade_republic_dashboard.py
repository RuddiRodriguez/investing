from __future__ import annotations

import json
from pathlib import Path

from datetime import UTC, datetime

from market_forecasting_engine.trade_republic_dashboard import (
    build_dashboard_state,
    load_latest_forecast_policy_context,
    write_dashboard_snapshot_outputs,
)


def test_build_dashboard_state_summarizes_report(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "summary": {
                    "report_timestamp": "2026-06-02T12:00:00+00:00",
                    "holding_count": 1,
                    "total_open_cost_basis": 100.0,
                    "total_current_value": 112.0,
                    "total_unrealized_pl": 12.0,
                    "total_unrealized_pl_pct": 12.0,
                    "total_historical_buy_cash": 100.0,
                    "total_historical_sell_cash": 0.0,
                    "ticker_resolution_count": 2,
                },
                "holdings": [
                    {
                        "name": "NVIDIA",
                        "isin": "US67066G1040",
                        "ticker": "NVD.DE",
                        "alpaca_ticker": "NVDA",
                        "ticker_resolution_source": "manual_map",
                        "current_quantity": 0.1,
                        "current_price": 195.04,
                        "current_value": 19.5,
                        "open_cost_basis": 18.0,
                        "unrealized_pl": 1.5,
                        "unrealized_pl_pct": 8.3333,
                        "weighted_paid_price": 180.0,
                        "historical_price_status": "matched",
                        "alpaca_weighted_price_at_buy_time": 181.0,
                        "alpaca_paid_vs_market_at_buy_time_pct": -0.55,
                        "alpaca_status": "matched",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    state = build_dashboard_state(report)

    assert state["summary"]["total_current_value"] == 112.0
    assert state["summary"]["total_unrealized_pl_pct"] == 12.0
    assert state["holdings"][0]["name"] == "NVIDIA"
    assert state["holdings"][0]["alpaca_status"] == "matched"
    assert state["holdings"][0]["allocation_pct"] == 17.41
    assert state["holdings"][0]["yahoo_status"] == "matched"


def test_build_dashboard_state_filters_sold_portfolio_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    override_path = tmp_path / "automated_forecasting_engine/runs/watch_agent_state"
    override_path.mkdir(parents=True)
    (override_path / "portfolio_overrides.json").write_text(
        json.dumps({"sold": {"1YD.DE": {"status": "sold", "reason": "user sold position"}}}),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "summary": {
                    "holding_count": 2,
                    "total_open_cost_basis": 118.0,
                    "total_current_value": 132.0,
                    "total_unrealized_pl": 14.0,
                    "total_unrealized_pl_pct": 11.8644,
                },
                "holdings": [
                    {
                        "name": "Broadcom",
                        "ticker": "1YD.DE",
                        "current_value": 20.0,
                        "open_cost_basis": 30.0,
                        "unrealized_pl": -10.0,
                    },
                    {
                        "name": "NVIDIA",
                        "ticker": "NVD.DE",
                        "current_value": 112.0,
                        "open_cost_basis": 88.0,
                        "unrealized_pl": 24.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    state = build_dashboard_state(report)

    assert [row["ticker"] for row in state["holdings"]] == ["NVD.DE"]
    assert state["summary"]["holding_count"] == 1
    assert state["summary"]["total_current_value"] == 112.0
    assert state["summary"]["manual_position_overrides_applied"] == ["1YD.DE"]


def test_build_dashboard_state_reports_missing_file(tmp_path: Path) -> None:
    state = build_dashboard_state(tmp_path / "missing.json")

    assert state["report_error"].startswith("report_not_found")
    assert state["holdings"] == []


def test_load_latest_forecast_policy_context_filters_same_day_logs(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    same_day = log_dir / "NVDA_medium_20260602.jsonl"
    same_day.write_text(
        "\n".join(
            [
                json.dumps({"ticker": "NVDA", "action": "HOLD", "llm_decision": "Hold", "reason": "first", "checked_at": "2026-06-02T09:00:00+00:00"}),
                json.dumps({"ticker": "NVDA", "action": "SELL", "llm_decision": "Trim", "reason": "latest", "checked_at": "2026-06-02T15:00:00+00:00"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    other_day = log_dir / "ASML_medium_20260603.jsonl"
    other_day.write_text(json.dumps({"ticker": "ASML", "action": "HOLD"}) + "\n", encoding="utf-8")

    rows = load_latest_forecast_policy_context(log_dir=log_dir, now=datetime(2026, 6, 2, tzinfo=UTC))

    assert rows == [
        {
            "ticker": "NVDA",
            "action": "SELL",
            "policy": "Trim",
            "llm_decision": "Trim",
            "reason": "latest",
            "checked_at": "2026-06-02T15:00:00+00:00",
            "market_status": None,
            "log_file": str(same_day),
        }
    ]


def test_write_dashboard_snapshot_outputs_writes_all_artifacts(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "summary": {
                    "report_timestamp": "2026-06-02T12:00:00+00:00",
                    "holding_count": 1,
                    "total_open_cost_basis": 100.0,
                    "total_current_value": 112.0,
                    "total_unrealized_pl": 12.0,
                    "total_unrealized_pl_pct": 12.0,
                    "ticker_resolution_count": 0,
                },
                "holdings": [
                    {
                        "name": "NVIDIA",
                        "isin": "US67066G1040",
                        "ticker": "NVD.DE",
                        "ticker_resolution_source": "manual_map",
                        "current_quantity": 0.1,
                        "current_price": 195.04,
                        "current_value": 19.5,
                        "open_cost_basis": 18.0,
                        "unrealized_pl": 1.5,
                        "unrealized_pl_pct": 8.3333,
                        "weighted_paid_price": 180.0,
                        "historical_price_status": "matched",
                        "alpaca_status": "matched",
                    }
                ],
                "ticker_resolution": [],
            }
        ),
        encoding="utf-8",
    )
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "NVD.DE_medium_20260602.jsonl").write_text(
        json.dumps(
            {
                "ticker": "NVD.DE",
                "action": "HOLD",
                "llm_decision": "Hold",
                "reason": "owned_no_sell_trigger",
                "checked_at": "2026-06-02T16:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = write_dashboard_snapshot_outputs(
        report_path=report,
        output_dir=tmp_path / "out",
        watch_log_dir=log_dir,
        now=datetime(2026, 6, 10, tzinfo=UTC),
    )

    state_path = Path(result["dashboard_state_path"])
    html_path = Path(result["dashboard_snapshot_path"])
    summary_path = Path(result["dashboard_summary_path"])
    assert state_path.exists()
    assert html_path.exists()
    assert summary_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["forecast_policy_context"] == []
    html = html_path.read_text(encoding="utf-8")
    assert "Trade Republic End-of-Day Dashboard" in html
    assert "const state =" in html
    assert "NVD.DE" in html
    summary = summary_path.read_text(encoding="utf-8")
    assert "## Latest Forecast / Policy Context" not in summary
    assert "Read-only snapshot: no orders submitted" in summary
