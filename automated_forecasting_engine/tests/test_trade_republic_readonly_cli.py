from __future__ import annotations

from market_forecasting_engine.trade_republic_readonly_cli import build_parser, build_pytr_command


def test_portfolio_command_is_read_only_pytr_command() -> None:
    args = build_parser().parse_args(
        [
            "--allow-login",
            "--phone-no",
            "+491234567",
            "--pin",
            "1234",
            "portfolio",
            "--output",
            "portfolio.csv",
        ]
    )

    command = build_pytr_command(args)

    assert "portfolio" in command
    assert "limit_order" not in command
    assert "set_price_alarms" not in command
    assert command[-2:] == ["--output", "portfolio.csv"]


def test_movements_command_uses_export_transactions_only() -> None:
    args = build_parser().parse_args(
        [
            "--allow-login",
            "movements",
            "--output-dir",
            "exports",
            "--format",
            "json",
            "--last-days",
            "90",
            "--sort",
        ]
    )

    command = build_pytr_command(args)

    assert "export_transactions" in command
    assert "dl_docs" not in command
    assert "--last_days" in command
    assert "90" in command
    assert "--sort" in command


def test_documents_command_uses_document_download_export() -> None:
    args = build_parser().parse_args(["--allow-login", "documents", "exports/docs", "--flat"])

    command = build_pytr_command(args)

    assert "dl_docs" in command
    assert "exports/docs" in command
    assert "--flat" in command
