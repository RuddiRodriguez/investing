from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from market_forecasting_engine.chapter_13_unsupervised_risk import (
    Chapter13Config,
    analyze_chapter_13_unsupervised_risk,
)
from market_forecasting_engine.chapter_13_eigen_trading import build_eigen_trading_plan
from market_forecasting_engine.data_providers import DataRequest, load_prices_with_provider
from market_forecasting_engine.portfolio_overrides import apply_open_position_overrides, load_portfolio_overrides
from market_forecasting_engine.trade_republic_watch_agents import calendar_for_ticker, live_price_provider_for_ticker


DEFAULT_REPORT = Path("automated_forecasting_engine/trade_republic_exports/investment_report_latest.json")
DEFAULT_STATE_DIR = Path("automated_forecasting_engine/runs/portfolio_brain")
DEFAULT_WATCH_STATE_DIR = Path("automated_forecasting_engine/runs/watch_agent_state")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the portfolio-level brain for Trade Republic holdings.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--watch-state-dir", type=Path, default=DEFAULT_WATCH_STATE_DIR)
    parser.add_argument("--profile", choices=("aggressive", "medium", "conservative"), default="medium")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--max-components", type=int, default=5)
    parser.add_argument("--min-history", type=int, default=120)
    parser.add_argument("--include-ica", action="store_true")
    parser.add_argument("--execute-basket-orders", action="store_true")
    parser.add_argument("--broker", choices=("none", "alpaca"), default="none")
    parser.add_argument("--max-order-notional", type=float, default=250.0)
    parser.add_argument("--max-eigen-component-notional", type=float, default=500.0)
    parser.add_argument("--min-order-notional", type=float, default=10.0)
    parser.add_argument("--write-watch-contexts", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = run_portfolio_brain(
        report=report,
        output_dir=args.output_dir,
        watch_state_dir=args.watch_state_dir,
        profile=args.profile,
        start=args.start,
        end=args.end,
        max_components=args.max_components,
        min_history=args.min_history,
        include_ica=args.include_ica,
        execute_basket_orders=args.execute_basket_orders,
        broker=args.broker,
        max_order_notional=args.max_order_notional,
        max_eigen_component_notional=args.max_eigen_component_notional,
        min_order_notional=args.min_order_notional,
        write_watch_contexts=args.write_watch_contexts,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"Portfolio brain report: {result['artifacts']['report']}")
    print(f"Per-ticker contexts: {result['artifacts'].get('watch_context_dir')}")


def run_portfolio_brain(
    *,
    report: dict[str, Any],
    output_dir: Path = DEFAULT_STATE_DIR,
    watch_state_dir: Path = DEFAULT_WATCH_STATE_DIR,
    profile: str = "medium",
    start: str | None = "2020-01-01",
    end: str | None = None,
    max_components: int = 5,
    min_history: int = 120,
    include_ica: bool = False,
    execute_basket_orders: bool = False,
    broker: str = "none",
    max_order_notional: float = 250.0,
    max_eigen_component_notional: float = 500.0,
    min_order_notional: float = 10.0,
    write_watch_contexts: bool = True,
    prices_by_ticker: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    overrides = load_portfolio_overrides(watch_state_dir / "portfolio_overrides.json")
    report = apply_open_position_overrides(report, overrides)
    holdings = _watchable_holdings(report)
    current_values = {_watch_ticker(item): float(item.get("current_value") or 0.0) for item in holdings}
    prices = prices_by_ticker or download_universe_prices(holdings, start=start, end=end)
    chapter_13 = analyze_chapter_13_unsupervised_risk(
        prices,
        current_values=current_values,
        config=Chapter13Config(start=start, end=end, max_components=max_components, min_history=min_history, include_ica=include_ica),
    )
    latest_decisions = load_latest_watch_decisions(watch_state_dir / "logs", profile=profile)
    coordinated = coordinate_portfolio_decisions(
        holdings=holdings,
        chapter_13=chapter_13,
        latest_decisions=latest_decisions,
        max_order_notional=max_order_notional,
        min_order_notional=min_order_notional,
    )
    eigen_trading = build_eigen_trading_plan(chapter_13, max_component_gross_notional=max_eigen_component_notional)
    execution = execute_basket_plan(coordinated, execute=execute_basket_orders, broker=broker)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    brain_report = {
        "generated_at_utc": generated_at,
        "status": "available" if chapter_13.get("status") == "available" else "partial",
        "source": "portfolio_brain",
        "profile": profile,
        "portfolio_summary": report.get("summary", {}),
        "chapter_13_unsupervised": chapter_13,
        "chapter_13_eigen_trading": eigen_trading,
        "latest_watch_decisions": latest_decisions,
        "coordinated_decisions": coordinated,
        "execution": execution,
    }
    report_path = output_dir / "portfolio_brain_report.json"
    report_path.write_text(json.dumps(brain_report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    context_dir = None
    if write_watch_contexts:
        context_dir = write_watch_contexts_for_agents(
            holdings=holdings,
            report=report,
            chapter_13=chapter_13,
            coordinated=coordinated,
            watch_state_dir=watch_state_dir,
            profile=profile,
            brain_report_path=report_path,
        )
    return {
        "summary": {
            "generated_at_utc": generated_at,
            "status": brain_report["status"],
            "holding_count": len(holdings),
            "analyzed_asset_count": chapter_13.get("universe", {}).get("asset_count", 0),
            "basket_order_count": len(coordinated.get("basket_order_plan", [])),
            "eigen_plan_count": len(eigen_trading.get("plans", [])),
            "execution_status": execution.get("status"),
        },
        "report": brain_report,
        "artifacts": {"report": str(report_path), "watch_context_dir": str(context_dir) if context_dir else None},
    }


def download_universe_prices(holdings: list[dict[str, Any]], *, start: str | None, end: str | None) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for holding in holdings:
        ticker = _watch_ticker(holding)
        if not ticker:
            continue
        provider = "yahoo" if live_price_provider_for_ticker(ticker) == "yahoo" else "yahoo"
        try:
            result = load_prices_with_provider(
                provider,
                DataRequest(ticker=ticker, start=start, end=end, interval="1d", target_column="close"),
                store=None,
                use_cache=False,
                refresh_cache=True,
            )
            output[ticker] = result.frame
        except Exception as exc:
            output[ticker] = pd.DataFrame()
            output[ticker].attrs["fetch_error"] = f"{type(exc).__name__}: {exc}"
    return output


def coordinate_portfolio_decisions(
    *,
    holdings: list[dict[str, Any]],
    chapter_13: dict[str, Any],
    latest_decisions: dict[str, Any],
    max_order_notional: float,
    min_order_notional: float,
) -> dict[str, Any]:
    ticker_contexts = chapter_13.get("ticker_contexts", {}) if chapter_13.get("status") == "available" else {}
    basket: list[dict[str, Any]] = []
    per_ticker: dict[str, Any] = {}
    for holding in holdings:
        ticker = _watch_ticker(holding)
        latest = latest_decisions.get(ticker, {})
        ctx = ticker_contexts.get(ticker, {})
        action = str(latest.get("action") or "HOLD").upper()
        risk_delta = _float(ctx.get("target_minus_current_weight"))
        risk_weight = max(_float(ctx.get("hrp_risk_weight")), 0.0)
        current_value = _float(holding.get("current_value"))
        reason: list[str] = []
        coordinated_action = action
        if action == "BUY" and risk_delta < -0.02:
            coordinated_action = "HOLD"
            reason.append("portfolio_brain_blocks_buy_over_hrp_weight")
        if action == "SELL" and risk_delta > 0.02:
            reason.append("portfolio_brain_sell_reduces_underweight_asset")
        if action == "BUY" and _cluster_has_active_buy(latest_decisions, ticker, ctx):
            coordinated_action = "HOLD"
            reason.append("portfolio_brain_blocks_cluster_duplicate_buy")
        notional = min(float(max_order_notional), max(float(min_order_notional), risk_weight * max(current_value, float(max_order_notional))))
        latest_price = _float(latest.get("price"))
        limit_price = _limit_price(coordinated_action, latest_price)
        order_plan = {
            "ticker": ticker,
            "calendar": calendar_for_ticker(ticker),
            "action": coordinated_action,
            "original_action": action,
            "notional": round(float(notional), 2),
            "latest_price": latest_price or None,
            "limit_price": limit_price,
            "order_type": "limit",
            "submit_eligible": coordinated_action in {"BUY", "SELL"} and limit_price is not None,
            "execution_policy": "dry_run_unless_execute_basket_orders",
            "reason": reason or ["portfolio_brain_no_change"],
        }
        per_ticker[ticker] = {
            "ticker": ticker,
            "latest_watch_decision": latest,
            "chapter_13_context": ctx,
            "coordinated_action": coordinated_action,
            "coordination_reasons": reason,
            "risk_budget": {
                "hrp_weight": ctx.get("hrp_risk_weight"),
                "current_weight": ctx.get("current_portfolio_weight"),
                "target_minus_current": ctx.get("target_minus_current_weight"),
            },
            "basket_order": order_plan,
        }
        if order_plan["submit_eligible"]:
            basket.append(order_plan)
    return {
        "status": "available",
        "policy": "Per-ticker agents remain the signal source; portfolio brain coordinates concentration, HRP risk budget, and basket-level execution plans.",
        "per_ticker": per_ticker,
        "basket_order_plan": basket,
    }


def execute_basket_plan(coordinated: dict[str, Any], *, execute: bool, broker: str) -> dict[str, Any]:
    orders = coordinated.get("basket_order_plan", [])
    if not execute:
        return {"status": "dry_run", "submitted": 0, "orders": orders}
    if broker != "alpaca":
        return {"status": "blocked", "submitted": 0, "reason": "No executable broker selected for basket orders.", "orders": orders}
    from market_forecasting_engine.alpaca_broker import AlpacaPaperBroker

    client = AlpacaPaperBroker()
    submitted = []
    for order in orders:
        if order.get("calendar") not in {"XNYS", "XNAS", "CRYPTO"}:
            submitted.append({"ticker": order["ticker"], "submitted": False, "reason": "broker_not_supported_for_listing"})
            continue
        side = "buy" if order["action"] == "BUY" else "sell"
        if side == "sell":
            submitted.append({"ticker": order["ticker"], "submitted": False, "reason": "sell_requires_verified_broker_position_quantity"})
            continue
        try:
            response = client.submit_order(
                symbol=order["ticker"],
                side=side,
                order_type="limit",
                notional=None,
                qty=round(float(order["notional"]) / max(float(order["limit_price"]), 1e-9), 8),
                limit_price=float(order["limit_price"]),
                time_in_force="day",
            )
            submitted.append({"ticker": order["ticker"], "submitted": True, "response": response})
        except Exception as exc:
            submitted.append({"ticker": order["ticker"], "submitted": False, "error": f"{type(exc).__name__}: {exc}"})
    return {"status": "executed", "submitted": sum(1 for item in submitted if item.get("submitted")), "orders": submitted}


def write_watch_contexts_for_agents(
    *,
    holdings: list[dict[str, Any]],
    report: dict[str, Any],
    chapter_13: dict[str, Any],
    coordinated: dict[str, Any],
    watch_state_dir: Path,
    profile: str,
    brain_report_path: Path,
) -> Path:
    context_dir = watch_state_dir / "portfolio_contexts"
    context_dir.mkdir(parents=True, exist_ok=True)
    summary = report.get("summary", {})
    per_ticker = coordinated.get("per_ticker", {})
    for holding in holdings:
        ticker = _watch_ticker(holding)
        context = {
            "broker": "trade_republic",
            "source": "portfolio_brain",
            "name": holding.get("name"),
            "isin": holding.get("isin"),
            "ticker": ticker,
            "alpaca_ticker": holding.get("alpaca_ticker"),
            "position": {
                "holding_status": "owned",
                "quantity": holding.get("current_quantity"),
                "avg_cost": holding.get("broker_avg_cost"),
                "current_price": holding.get("current_price"),
                "current_value": holding.get("current_value"),
                "unrealized_pl": holding.get("unrealized_pl"),
                "unrealized_pl_pct": holding.get("unrealized_pl_pct"),
            },
            "portfolio_summary": summary,
            "portfolio_brain": {
                "report_path": str(brain_report_path),
                "chapter_13_status": chapter_13.get("status"),
                **(per_ticker.get(ticker, {})),
            },
        }
        (context_dir / f"{_safe_label(ticker)}_{profile}.json").write_text(
            json.dumps(context, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return context_dir


def load_latest_watch_decisions(log_dir: Path, *, profile: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if not log_dir.exists():
        return output
    for path in sorted(log_dir.glob(f"*_{profile}_*.jsonl")):
        ticker = path.name.split(f"_{profile}_", 1)[0].replace("_", "/") if "/" in path.name else path.name.split(f"_{profile}_", 1)[0]
        try:
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                continue
            record = json.loads(lines[-1])
            output[str(record.get("ticker") or ticker).upper()] = record
        except Exception:
            continue
    return output


def _watchable_holdings(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [holding for holding in report.get("holdings", []) if _watch_ticker(holding)]


def _watch_ticker(holding: dict[str, Any]) -> str:
    return str(holding.get("ticker") or holding.get("alpaca_ticker") or "").strip().upper()


def _cluster_has_active_buy(latest_decisions: dict[str, Any], ticker: str, context: dict[str, Any]) -> bool:
    peers = set(context.get("cluster_peers") or [])
    if not peers:
        return False
    return any(str(latest_decisions.get(peer, {}).get("action") or "").upper() == "BUY" for peer in peers if peer != ticker)


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _limit_price(action: str, latest_price: float) -> float | None:
    if latest_price <= 0 or action not in {"BUY", "SELL"}:
        return None
    offset = 0.002
    price = latest_price * (1.0 + offset if action == "BUY" else 1.0 - offset)
    return round(float(price), 2)


def _safe_label(value: str) -> str:
    return str(value).upper().replace("/", "_").replace("-", "_").replace(" ", "_").lower()


if __name__ == "__main__":
    main()
