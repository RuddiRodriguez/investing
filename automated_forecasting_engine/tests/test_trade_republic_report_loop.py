from __future__ import annotations

import argparse
import json
from pathlib import Path

from market_forecasting_engine import trade_republic_report_loop as loop


def test_run_loop_once_writes_report_atomically(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "investment_report_latest.json"
    args = argparse.Namespace(
        portfolio=tmp_path / "portfolio.csv",
        transactions=tmp_path / "account_transactions.csv",
        isin_map=tmp_path / "isin_map.csv",
        output=output,
        interval_seconds=60,
        fetch_yahoo=True,
        fetch_alpaca=True,
        refresh_portfolio=False,
        refresh_movements_every=0,
        pytr_timeout_seconds=120,
        once=True,
    )

    monkeypatch.setattr(
        loop,
        "build_report",
        lambda **_: {
            "summary": {
                "report_timestamp": "2026-06-02T18:00:00+02:00",
                "total_current_value": 120.0,
                "total_unrealized_pl": 5.0,
            },
            "holdings": [],
        },
    )

    loop.run_loop(args)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["total_current_value"] == 120.0
    assert not output.with_suffix(".json.tmp").exists()


def test_run_loop_once_preserves_existing_report_on_failure(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "investment_report_latest.json"
    output.write_text('{"summary":{"total_current_value":99}}', encoding="utf-8")
    args = argparse.Namespace(
        portfolio=tmp_path / "portfolio.csv",
        transactions=tmp_path / "account_transactions.csv",
        isin_map=tmp_path / "isin_map.csv",
        output=output,
        interval_seconds=60,
        fetch_yahoo=True,
        fetch_alpaca=True,
        refresh_portfolio=False,
        refresh_movements_every=0,
        pytr_timeout_seconds=120,
        once=True,
    )

    def fail_report(**_: object) -> dict[str, object]:
        raise RuntimeError("market data unavailable")

    monkeypatch.setattr(loop, "build_report", fail_report)

    loop.run_loop(args)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["total_current_value"] == 99


def test_run_loop_reconstructs_positions_when_portfolio_export_fails(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "investment_report_latest.json"
    args = argparse.Namespace(
        portfolio=tmp_path / "portfolio.csv",
        transactions=tmp_path / "account_transactions.csv",
        isin_map=tmp_path / "isin_map.csv",
        output=output,
        interval_seconds=60,
        fetch_yahoo=True,
        fetch_alpaca=True,
        refresh_portfolio=True,
        refresh_movements_every=1,
        pytr_timeout_seconds=120,
        once=True,
    )
    captured: dict[str, object] = {}

    def fail_portfolio(*_: object, **__: object) -> None:
        raise RuntimeError("BAD_SUBSCRIPTION_TYPE")

    def fake_report(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "summary": {
                "report_timestamp": "2026-06-02T18:00:00+02:00",
                "total_current_value": 120.0,
                "total_unrealized_pl": 5.0,
            },
            "holdings": [],
        }

    monkeypatch.setattr(loop, "refresh_portfolio", fail_portfolio)
    monkeypatch.setattr(loop, "refresh_movements", lambda *_: None)
    monkeypatch.setattr(loop, "build_report", fake_report)

    loop.run_loop(args)

    assert captured["reconstruct_positions_from_transactions"] is True
