from __future__ import annotations

import argparse
import json
import os
import plistlib
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REPORT = Path("automated_forecasting_engine/trade_republic_exports/investment_report_latest.json")
DEFAULT_STATE_DIR = "automated_forecasting_engine/runs/watch_agent_state"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install one watch-agent LaunchAgent per Trade Republic holding.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--project-dir", type=Path, default=Path("/Users/ruddigarcia/Projects/invest"))
    parser.add_argument("--state-dir", default=DEFAULT_STATE_DIR)
    parser.add_argument("--profile", choices=("aggressive", "medium", "conservative"), default="medium")
    parser.add_argument("--refresh-after-hours", default="12")
    parser.add_argument("--interval-seconds", default="3600")
    parser.add_argument(
        "--stagger-seconds",
        default="300",
        help="Delay between portfolio watcher starts. Prevents every holding from running a full forecast at once.",
    )
    parser.add_argument("--llm-env-file", default=None)
    parser.add_argument("--quiet-unchanged", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-holdings", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    plans = build_agent_plans(
        report=report,
        project_dir=args.project_dir,
        state_dir=args.state_dir,
        profile=args.profile,
        refresh_after_hours=args.refresh_after_hours,
        interval_seconds=args.interval_seconds,
        stagger_seconds=args.stagger_seconds,
        llm_env_file=args.llm_env_file or str(args.project_dir / ".env"),
        quiet_unchanged=args.quiet_unchanged,
        max_holdings=args.max_holdings,
    )
    print(f"Portfolio holdings with watchable tickers: {len(plans)}")
    for plan in plans:
        print(
            f"- {plan['ticker']} | {plan['name']} | calendar={plan['calendar']} | "
            f"provider={plan['live_price_provider']} | delay={plan['environment'].get('STARTUP_DELAY_SECONDS')}s | label={plan['label']}"
        )
    if args.dry_run:
        return
    install_agent_plans(plans, replace=args.replace)


def build_agent_plans(
    *,
    report: dict[str, Any],
    project_dir: Path,
    state_dir: str,
    profile: str,
    refresh_after_hours: str,
    interval_seconds: str,
    stagger_seconds: str,
    llm_env_file: str,
    quiet_unchanged: bool,
    max_holdings: int | None = None,
) -> list[dict[str, Any]]:
    plans = []
    summary = report.get("summary", {})
    holdings = [holding for holding in report.get("holdings", []) if _watch_ticker(holding)]
    if max_holdings is not None:
        holdings = holdings[: max(0, int(max_holdings))]
    context_dir = project_dir / state_dir / "portfolio_contexts"
    script_path = project_dir / "automated_forecasting_engine/scripts/run_watch_agent_once.sh"
    for index, holding in enumerate(holdings):
        ticker = _watch_ticker(holding)
        label = f"com.marketforecasting.watchagent.portfolio.{_safe_label(ticker)}.{profile}"
        context_path = context_dir / f"{_safe_label(ticker)}_{profile}.json"
        context = _merge_existing_portfolio_brain_context(portfolio_context_for_holding(holding, summary), context_path)
        env = {
            "PROJECT_DIR": str(project_dir),
            "TICKER": ticker,
            "PROFILE": profile,
            "HOLDING_STATUS": "owned",
            "REFRESH_AFTER_HOURS": str(refresh_after_hours),
            "LLM_ENV_FILE": str(llm_env_file),
            "STATE_DIR": state_dir,
            "QUIET_UNCHANGED": "1" if quiet_unchanged else "0",
            "CALENDAR": calendar_for_ticker(ticker),
            "LIVE_PRICE_PROVIDER": live_price_provider_for_ticker(ticker),
            "ENTRY_PRICE": _string_or_empty(holding.get("broker_avg_cost")),
            "QUANTITY": _string_or_empty(holding.get("current_quantity")),
            "POSITION_VALUE": _string_or_empty(holding.get("current_value")),
            "ACCOUNT_EQUITY": _string_or_empty(summary.get("total_current_value")),
            "PORTFOLIO_CONTEXT_FILE": str(context_path),
            "STARTUP_DELAY_SECONDS": str(max(0, int(float(stagger_seconds)) * index)),
            "OPENAI_USAGE_PROCESS_NAME": "watch_agent",
        }
        plans.append(
            {
                "ticker": ticker,
                "name": holding.get("name") or ticker,
                "label": label,
                "plist_path": Path.home() / "Library/LaunchAgents" / f"{label}.plist",
                "script_path": script_path,
                "project_dir": project_dir,
                "state_dir": state_dir,
                "stdout_path": project_dir / state_dir / f"{label}.stdout.log",
                "stderr_path": project_dir / state_dir / f"{label}.stderr.log",
                "start_interval": int(interval_seconds),
                "environment": env,
                "context": context,
                "context_path": context_path,
                "calendar": env["CALENDAR"],
                "live_price_provider": env["LIVE_PRICE_PROVIDER"],
            }
        )
    return plans


def portfolio_context_for_holding(holding: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    ticker = _watch_ticker(holding)
    return {
        "broker": "trade_republic",
        "source": "trade_republic_investment_report",
        "name": holding.get("name"),
        "isin": holding.get("isin"),
        "ticker": ticker,
        "alpaca_ticker": holding.get("alpaca_ticker"),
        "listing": {
            "calendar": calendar_for_ticker(ticker),
            "preferred_region": "europe" if _is_european_yahoo_ticker(ticker) else "us_or_other",
            "price_provider": live_price_provider_for_ticker(ticker),
        },
        "position": {
            "holding_status": "owned",
            "quantity": holding.get("current_quantity"),
            "avg_cost": holding.get("broker_avg_cost"),
            "current_price": holding.get("current_price"),
            "current_value": holding.get("current_value"),
            "open_cost_basis": holding.get("open_cost_basis"),
            "unrealized_pl": holding.get("unrealized_pl"),
            "unrealized_pl_pct": holding.get("unrealized_pl_pct"),
            "historical_buy_cash": holding.get("historical_buy_cash"),
            "historical_sell_cash": holding.get("historical_sell_cash"),
        },
        "portfolio_summary": {
            "total_current_value": summary.get("total_current_value"),
            "total_open_cost_basis": summary.get("total_open_cost_basis"),
            "total_unrealized_pl": summary.get("total_unrealized_pl"),
            "total_unrealized_pl_pct": summary.get("total_unrealized_pl_pct"),
            "holding_count": summary.get("holding_count"),
        },
    }


def install_agent_plans(plans: list[dict[str, Any]], *, replace: bool) -> None:
    for plan in plans:
        install_agent_plan(plan, replace=replace)


def install_agent_plan(plan: dict[str, Any], *, replace: bool) -> None:
    plist_path = Path(plan["plist_path"])
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    Path(plan["context_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(plan["context_path"]).write_text(json.dumps(plan["context"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(plan["stdout_path"]).parent.mkdir(parents=True, exist_ok=True)
    plist = {
        "Label": plan["label"],
        "ProgramArguments": [str(plan["script_path"])],
        "WorkingDirectory": str(plan["project_dir"]),
        "EnvironmentVariables": plan["environment"],
        "RunAtLoad": True,
        "StartInterval": int(plan["start_interval"]),
        "StandardOutPath": str(plan["stdout_path"]),
        "StandardErrorPath": str(plan["stderr_path"]),
    }
    if replace:
        subprocess.run(["launchctl", "bootout", f"gui/{_uid()}", str(plist_path)], check=False, capture_output=True, text=True)
    with plist_path.open("wb") as handle:
        plistlib.dump(plist, handle)
    subprocess.run(["plutil", "-lint", str(plist_path)], check=True)
    subprocess.run(["launchctl", "bootstrap", f"gui/{_uid()}", str(plist_path)], check=True)
    subprocess.run(["launchctl", "kickstart", "-k", f"gui/{_uid()}/{plan['label']}"], check=True)
    print(f"Installed and started {plan['label']} -> {plan['ticker']}")


def calendar_for_ticker(ticker: str) -> str:
    value = str(ticker or "").upper()
    if value.endswith("-USD") or value.endswith("-USDT"):
        return "CRYPTO"
    suffix_map = {
        ".AS": "XAMS",
        ".DE": "XETR",
        ".F": "XFRA",
        ".TO": "XTSE",
        ".MC": "XMAD",
    }
    for suffix, calendar in suffix_map.items():
        if value.endswith(suffix):
            return calendar
    return "XNYS"


def live_price_provider_for_ticker(ticker: str) -> str:
    return "yahoo" if _is_european_yahoo_ticker(ticker) or str(ticker).upper().endswith(".TO") else "auto"


def _watch_ticker(holding: dict[str, Any]) -> str:
    return str(holding.get("ticker") or holding.get("alpaca_ticker") or "").strip().upper()


def _is_european_yahoo_ticker(ticker: str) -> bool:
    value = str(ticker or "").upper()
    return value.endswith((".AS", ".DE", ".F", ".MC"))


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).lower()).strip("._-") or "unknown"


def _string_or_empty(value: Any) -> str:
    return "" if value is None else str(value)


def _merge_existing_portfolio_brain_context(context: dict[str, Any], context_path: Path) -> dict[str, Any]:
    if not context_path.exists():
        return context
    try:
        existing = json.loads(context_path.read_text(encoding="utf-8"))
    except Exception:
        return context
    if isinstance(existing, dict) and isinstance(existing.get("portfolio_brain"), dict):
        merged = dict(context)
        merged["portfolio_brain"] = existing["portfolio_brain"]
        merged["source"] = "trade_republic_investment_report_with_portfolio_brain"
        return merged
    return context


def _uid() -> str:
    return str(os.getuid())


if __name__ == "__main__":
    main()
