from __future__ import annotations

from market_forecasting_engine.trading_router_agent import RouterStatus, RouterTool, extract_tickers, plan_from_question


def test_router_asks_forecast_style_for_ambiguous_stock_advice() -> None:
    plan = plan_from_question("give me advice for AAPL and MSFT")

    assert plan.status == RouterStatus.NEEDS_CLARIFICATION
    assert plan.tickers == ["AAPL", "MSFT"]
    assert "pure LLM" in plan.questions[0]


def test_router_builds_pure_llm_stock_forecast_commands() -> None:
    plan = plan_from_question("pure LLM forecast for AAPL MSFT")

    assert plan.status == RouterStatus.READY
    assert plan.tool == RouterTool.PURE_LLM_STOCK_FORECAST
    assert len(plan.commands) == 2
    assert "market_forecasting_engine.pure_llm_stock_forecaster" in plan.commands[0]
    assert "--ticker" in plan.commands[0]
    assert " && " in plan.command_text[0]
    assert "'&&'" not in plan.command_text[0]


def test_router_builds_classical_stock_forecast_command() -> None:
    plan = plan_from_question("classical model forecast for WM")

    assert plan.status == RouterStatus.READY
    assert plan.tool == RouterTool.CLASSICAL_STOCK_FORECAST
    assert "market_forecasting_engine.cli" in plan.commands[0]
    assert "WM" in plan.commands[0]


def test_router_universe_scan_defaults_to_dry_run_once() -> None:
    plan = plan_from_question("choose one stock from the universe for advice")

    assert plan.status == RouterStatus.READY
    assert plan.tool == RouterTool.UNIVERSE_STOCK_AGENT
    assert "--once" in plan.commands[0]
    assert "--dry-run" in plan.commands[0]
    assert "--max-managed-candidates" in plan.commands[0]


def test_router_expired_alpaca_orders_is_dry_run_by_default() -> None:
    plan = plan_from_question("check expired Alpaca orders for ASML and VTI")

    assert plan.status == RouterStatus.READY
    assert plan.tool == RouterTool.EXPIRED_ALPACA_ORDER_CHECK
    assert "--only-if-expired" in plan.commands[0]
    assert "ASML,VTI" in plan.commands[0]
    assert "--execute-live-orders" not in plan.commands[0]


def test_router_options_asks_for_risk_detail() -> None:
    plan = plan_from_question("TSLA options trade")

    assert plan.status == RouterStatus.NEEDS_CLARIFICATION
    assert plan.tool == RouterTool.ALPACA_PAPER_OPTIONS
    assert "max debit" in plan.questions[0]


def test_router_deribit_spot() -> None:
    plan = plan_from_question("check Deribit ETH/USDC spot plan")

    assert plan.status == RouterStatus.READY
    assert plan.tool == RouterTool.DERIBIT_ETH_SPOT
    assert "market_forecasting_engine.live_trading.deribit_eth_usdc_daily_agent" in plan.commands[0]


def test_extract_tickers_filters_common_terms() -> None:
    assert extract_tickers("pure LLM forecast for AAPL and ROIC check") == ["AAPL"]
