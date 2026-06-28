from __future__ import annotations

import argparse
import json
import re
import shlex
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


DEFAULT_PROJECT_DIR = Path("/Users/ruddigarcia/Projects/invest")
DEFAULT_PYTHON = "./venv/bin/python"


class RouterTool(str, Enum):
    PURE_LLM_STOCK_FORECAST = "pure_llm_stock_forecast"
    CLASSICAL_STOCK_FORECAST = "classical_stock_forecast"
    UNIVERSE_STOCK_AGENT = "universe_stock_agent"
    EXPIRED_ALPACA_ORDER_CHECK = "expired_alpaca_order_check"
    ALPACA_PAPER_OPTIONS = "alpaca_paper_options"
    DERIBIT_ETH_SPOT = "deribit_eth_spot"
    DERIBIT_ETH_OPTIONS = "deribit_eth_options"
    LLM_OPTIONS_LIVE_SHADOW = "llm_options_live_shadow"


class RouterStatus(str, Enum):
    READY = "ready"
    NEEDS_CLARIFICATION = "needs_clarification"
    UNSUPPORTED = "unsupported"


class RouterPlan(BaseModel):
    status: RouterStatus
    reason: str
    tool: RouterTool | None = None
    questions: list[str] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    commands: list[list[str]] = Field(default_factory=list)
    command_text: list[str] = Field(default_factory=list)
    report_hints: list[str] = Field(default_factory=list)
    safety: dict[str, Any] = Field(default_factory=dict)


class ToolContract(BaseModel):
    tool: RouterTool
    purpose: str
    required: list[str]
    safe_defaults: dict[str, Any]
    must_ask_if_missing: list[str]
    forbidden_without_explicit_request: list[str]


TOOL_CONTRACTS: dict[RouterTool, ToolContract] = {
    RouterTool.PURE_LLM_STOCK_FORECAST: ToolContract(
        tool=RouterTool.PURE_LLM_STOCK_FORECAST,
        purpose="One-stock or multi-stock pure LLM forecast with stock CEO advice.",
        required=["ticker"],
        safe_defaults={"orders": "none", "execution": "advice_only"},
        must_ask_if_missing=[],
        forbidden_without_explicit_request=["live_order", "market_order"],
    ),
    RouterTool.CLASSICAL_STOCK_FORECAST: ToolContract(
        tool=RouterTool.CLASSICAL_STOCK_FORECAST,
        purpose="One-stock or multi-stock classical forecast with model validation and stock CEO advice.",
        required=["ticker"],
        safe_defaults={"orders": "none", "execution": "advice_only"},
        must_ask_if_missing=[],
        forbidden_without_explicit_request=["live_order", "market_order"],
    ),
    RouterTool.UNIVERSE_STOCK_AGENT: ToolContract(
        tool=RouterTool.UNIVERSE_STOCK_AGENT,
        purpose="Scan broad stock universe, choose candidates, forecast, and build Alpaca paper plans.",
        required=[],
        safe_defaults={"once": True, "dry_run": True, "max_managed_candidates": 1},
        must_ask_if_missing=["paper_order_or_advice_only"],
        forbidden_without_explicit_request=["paper_order_submission", "live_order"],
    ),
    RouterTool.EXPIRED_ALPACA_ORDER_CHECK: ToolContract(
        tool=RouterTool.EXPIRED_ALPACA_ORDER_CHECK,
        purpose="Check expired Alpaca live advice orders and optionally build replacement limit plans.",
        required=["ticker_or_default_watchlist"],
        safe_defaults={"execution": "dry_run", "only_if_expired": True},
        must_ask_if_missing=[],
        forbidden_without_explicit_request=["live_order"],
    ),
    RouterTool.ALPACA_PAPER_OPTIONS: ToolContract(
        tool=RouterTool.ALPACA_PAPER_OPTIONS,
        purpose="Run Alpaca paper options decision for one underlying.",
        required=["ticker"],
        safe_defaults={"execution": "dry_run"},
        must_ask_if_missing=["max_debit_or_risk_profile"],
        forbidden_without_explicit_request=["paper_order_submission", "live_order"],
    ),
    RouterTool.DERIBIT_ETH_SPOT: ToolContract(
        tool=RouterTool.DERIBIT_ETH_SPOT,
        purpose="Run Deribit ETH/USDC spot daily agent check.",
        required=[],
        safe_defaults={"execution": "dry_run"},
        must_ask_if_missing=[],
        forbidden_without_explicit_request=["live_order"],
    ),
    RouterTool.DERIBIT_ETH_OPTIONS: ToolContract(
        tool=RouterTool.DERIBIT_ETH_OPTIONS,
        purpose="Run Deribit ETH options decision.",
        required=[],
        safe_defaults={"execution": "dry_run"},
        must_ask_if_missing=["max_debit_or_risk_profile"],
        forbidden_without_explicit_request=["live_order"],
    ),
    RouterTool.LLM_OPTIONS_LIVE_SHADOW: ToolContract(
        tool=RouterTool.LLM_OPTIONS_LIVE_SHADOW,
        purpose="Run LLM options live-shadow simulator.",
        required=[],
        safe_defaults={"execution": "shadow_only"},
        must_ask_if_missing=[],
        forbidden_without_explicit_request=["real_order_submission"],
    ),
}


def plan_from_question(question: str, *, project_dir: Path = DEFAULT_PROJECT_DIR, python: str = DEFAULT_PYTHON) -> RouterPlan:
    text = " ".join(str(question or "").split())
    lower = text.lower()
    tickers = extract_tickers(text)
    safety = {
        "default_execution": "advice_or_dry_run_only",
        "live_orders": "blocked_unless_explicitly_requested_and_confirmed",
        "market_orders": "blocked",
    }

    if not text:
        return _needs("Empty question.", ["What do you want to check or run?"], safety=safety)

    if _mentions_options(lower):
        return _plan_options(lower, tickers, project_dir=project_dir, python=python, safety=safety)

    if "expired" in lower and "alpaca" in lower and "order" in lower:
        return _ready(
            RouterTool.EXPIRED_ALPACA_ORDER_CHECK,
            "Expired Alpaca advice-order check requested.",
            [_expired_alpaca_command(project_dir, python, tickers)],
            tickers=tickers,
            safety=safety,
        )

    if "deribit" in lower or "eth/usdc" in lower or "eth-usdc" in lower:
        return _ready(
            RouterTool.DERIBIT_ETH_SPOT,
            "Deribit ETH/USDC spot check requested.",
            [_deribit_spot_command(project_dir, python)],
            tickers=["ETH-USDC"],
            safety=safety,
        )

    if _mentions_universe_scan(lower):
        return _ready(
            RouterTool.UNIVERSE_STOCK_AGENT,
            "Universe stock selection requested.",
            [_universe_stock_agent_command(project_dir, python, dry_run=not _explicit_paper_submit(lower))],
            safety=safety,
        )

    if tickers:
        if "pure" in lower or "llm" in lower:
            return _ready(
                RouterTool.PURE_LLM_STOCK_FORECAST,
                "Pure LLM stock forecast requested.",
                [_pure_llm_command(project_dir, python, ticker) for ticker in tickers],
                tickers=tickers,
                safety=safety,
            )
        if any(token in lower for token in ("classical", "classic", "full forecast", "model", "validation")):
            return _ready(
                RouterTool.CLASSICAL_STOCK_FORECAST,
                "Classical stock forecast requested.",
                [_classical_command(project_dir, python, ticker) for ticker in tickers],
                tickers=tickers,
                safety=safety,
            )
        return _needs(
            "Stock advice request needs forecast style.",
            ["Use pure LLM forecast or classical/model forecast?"],
            tickers=tickers,
            safety=safety,
        )

    return RouterPlan(
        status=RouterStatus.UNSUPPORTED,
        reason="Could not map question to a known local trading agent.",
        questions=["Which workflow: stock forecast, universe scan, expired Alpaca orders, options, or Deribit ETH/USDC?"],
        safety=safety,
    )


def extract_tickers(text: str) -> list[str]:
    tokens = re.findall(r"\b[A-Z]{1,5}(?:[-/][A-Z]{2,5})?\b", text)
    stop = {"CEO", "LLM", "ETF", "ROIC", "USD", "USDC", "ETH"}
    output: list[str] = []
    for token in tokens:
        normalized = token.replace("/", "-").upper()
        if normalized in stop:
            continue
        if normalized not in output:
            output.append(normalized)
    return output


def _plan_options(lower: str, tickers: list[str], *, project_dir: Path, python: str, safety: dict[str, Any]) -> RouterPlan:
    if "shadow" in lower or "simulate" in lower:
        return _ready(
            RouterTool.LLM_OPTIONS_LIVE_SHADOW,
            "Options shadow simulation requested.",
            [_llm_options_live_shadow_command(project_dir, python)],
            safety=safety,
        )
    if "deribit" in lower or "eth" in lower:
        if _needs_risk_detail(lower):
            return _needs(
                "Deribit options request missing risk size.",
                ["What max debit or risk budget should the Deribit options tool use?"],
                tool=RouterTool.DERIBIT_ETH_OPTIONS,
                tickers=tickers,
                safety=safety,
            )
        return _ready(RouterTool.DERIBIT_ETH_OPTIONS, "Deribit ETH options request.", [_deribit_options_command(project_dir, python)], safety=safety)
    if not tickers:
        return _needs("Options request missing underlying ticker.", ["Which ticker should the options agent use?"], safety=safety)
    if _needs_risk_detail(lower):
        return _needs(
            "Alpaca paper options request missing risk size.",
            ["What max debit or risk profile should the paper options tool use?"],
            tool=RouterTool.ALPACA_PAPER_OPTIONS,
            tickers=tickers,
            safety=safety,
        )
    return _ready(
        RouterTool.ALPACA_PAPER_OPTIONS,
        "Alpaca paper options request.",
        [_paper_options_command(project_dir, python, tickers[0])],
        tickers=tickers[:1],
        safety=safety,
    )


def _mentions_options(text: str) -> bool:
    return "option" in text or "dte" in text or "contract" in text or "greeks" in text


def _mentions_universe_scan(text: str) -> bool:
    return any(phrase in text for phrase in ("universe", "choose one stock", "pick one stock", "find one stock", "scan tickers", "scan stocks"))


def _explicit_paper_submit(text: str) -> bool:
    return "submit paper" in text or "execute paper" in text or "paper order" in text


def _needs_risk_detail(text: str) -> bool:
    return not any(word in text for word in ("max debit", "risk budget", "risk-profile", "risk profile", "max-total-debit"))


def _ready(
    tool: RouterTool,
    reason: str,
    commands: list[list[str]],
    *,
    tickers: list[str] | None = None,
    safety: dict[str, Any] | None = None,
) -> RouterPlan:
    return RouterPlan(
        status=RouterStatus.READY,
        tool=tool,
        reason=reason,
        tickers=tickers or [],
        commands=commands,
        command_text=[_shell_text(command) for command in commands],
        report_hints=_report_hints(tool),
        safety=safety or {},
    )


def _needs(
    reason: str,
    questions: list[str],
    *,
    tool: RouterTool | None = None,
    tickers: list[str] | None = None,
    safety: dict[str, Any] | None = None,
) -> RouterPlan:
    return RouterPlan(status=RouterStatus.NEEDS_CLARIFICATION, tool=tool, reason=reason, questions=questions, tickers=tickers or [], safety=safety or {})


def _base(project_dir: Path, python: str, module: str) -> list[str]:
    return ["cd", str(project_dir), "&&", "PYTHONPATH=automated_forecasting_engine/src", python, "-m", module]


def _shell_text(command: list[str]) -> str:
    if "&&" not in command:
        return shlex.join(command)
    split_at = command.index("&&")
    before = shlex.join(command[:split_at])
    after = shlex.join(command[split_at + 1 :])
    return f"{before} && {after}"


def _pure_llm_command(project_dir: Path, python: str, ticker: str) -> list[str]:
    return [
        *_base(project_dir, python, "market_forecasting_engine.pure_llm_stock_forecaster"),
        "--ticker",
        ticker,
        "--company",
        ticker,
        "--provider",
        "yahoo",
        "--interval",
        "1d",
        "--ceo-llm-provider",
        "openai",
        "--trader-profile",
        "medium",
    ]


def _classical_command(project_dir: Path, python: str, ticker: str) -> list[str]:
    return [
        *_base(project_dir, python, "market_forecasting_engine.cli"),
        "--ticker",
        ticker,
        "--provider",
        "yahoo",
        "--start",
        "2020-01-01",
        "--interval",
        "1d",
        "--horizons",
        "5,10,20",
        "--trader-profile",
        "medium",
    ]


def _universe_stock_agent_command(project_dir: Path, python: str, *, dry_run: bool) -> list[str]:
    command = [
        *_base(project_dir, python, "market_forecasting_engine.virtual_trader_agent_cli"),
        "--project-dir",
        str(project_dir),
        "--output-root",
        "automated_forecasting_engine/runs/virtual_trader_agent_router",
        "--memory-path",
        "automated_forecasting_engine/runs/virtual_trader_router/memory.json",
        "--once",
        "--max-managed-candidates",
        "1",
        "--scout-final-candidates",
        "5",
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def _expired_alpaca_command(project_dir: Path, python: str, tickers: list[str]) -> list[str]:
    command = [*_base(project_dir, python, "market_forecasting_engine.live_trading.stocks.advice_order_agent"), "--only-if-expired"]
    if tickers:
        command.extend(["--tickers", ",".join(tickers)])
    return command


def _paper_options_command(project_dir: Path, python: str, ticker: str) -> list[str]:
    return [*_base(project_dir, python, "market_forecasting_engine.paper_options_agent"), "--ticker", ticker, "--risk-profile", "medium", "--entry-order-policy", "limit"]


def _deribit_spot_command(project_dir: Path, python: str) -> list[str]:
    return [
        *_base(project_dir, python, "market_forecasting_engine.live_trading.deribit_eth_usdc_daily_agent"),
        "--project-dir",
        str(project_dir),
        "--output-dir",
        "automated_forecasting_engine/runs/live_deribit_eth_usdc_daily_agent_router",
        "--instrument",
        "ETH_USDC",
        "--ticker",
        "ETH-USDC",
        "--forecast-provider",
        "deribit",
    ]


def _deribit_options_command(project_dir: Path, python: str) -> list[str]:
    return [*_base(project_dir, python, "market_forecasting_engine.deribit_options_agent"), "--account-mode", "live", "--currency", "ETH", "--instrument-currency", "USDC"]


def _llm_options_live_shadow_command(project_dir: Path, python: str) -> list[str]:
    return [*_base(project_dir, python, "market_forecasting_engine.llm_options_trader.agent"), "--account-mode", "live", "--simulation-only", "--currency", "ETH", "--instrument-currency", "USDC"]


def _report_hints(tool: RouterTool) -> list[str]:
    return {
        RouterTool.PURE_LLM_STOCK_FORECAST: ["automated_forecasting_engine/runs/pure_llm_stock_forecaster"],
        RouterTool.CLASSICAL_STOCK_FORECAST: ["forecast_report.json in selected output dir"],
        RouterTool.UNIVERSE_STOCK_AGENT: ["automated_forecasting_engine/runs/virtual_trader_agent_router/latest_cycle.json"],
        RouterTool.EXPIRED_ALPACA_ORDER_CHECK: ["automated_forecasting_engine/runs/live_advice_order_agent/live_advice_order_agent_report.json"],
        RouterTool.ALPACA_PAPER_OPTIONS: ["automated_forecasting_engine/runs/paper_options_agent"],
        RouterTool.DERIBIT_ETH_SPOT: ["automated_forecasting_engine/runs/live_deribit_eth_usdc_daily_agent_router"],
        RouterTool.DERIBIT_ETH_OPTIONS: ["automated_forecasting_engine/runs/deribit_options_agent"],
        RouterTool.LLM_OPTIONS_LIVE_SHADOW: ["automated_forecasting_engine/runs/llm_options_trader_live_shadow"],
    }[tool]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan which local trading agent should answer a natural-language request.")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--project-dir", default=str(DEFAULT_PROJECT_DIR))
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    args = parser.parse_args()
    plan = plan_from_question(" ".join(args.question), project_dir=Path(args.project_dir), python=args.python)
    print(json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
