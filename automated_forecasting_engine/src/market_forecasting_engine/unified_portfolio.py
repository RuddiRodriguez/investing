from __future__ import annotations

import json
import os
import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_TRADE_REPUBLIC_REPORT = Path("trade_republic_exports/investment_report_latest.json")
DEFAULT_TRADE_REPUBLIC_TRANSACTIONS = Path("trade_republic_exports/account_transactions.csv")
DEFAULT_SNAPTRADE_SNAPSHOT = Path("snaptrade_exports/snapshot_latest.json")
DEFAULT_UNIFIED_HISTORY = Path("unified_portfolio_history.jsonl")
DEFAULT_SCENARIO_HISTORY = Path("unified_scenario_history.json")
DEFAULT_BENCHMARK_HISTORY = Path("unified_benchmark_history.json")
DEFAULT_CLASSIFICATION_CACHE = Path("unified_asset_classifications.json")
DEFAULT_ASSET_PRICE_HISTORY = Path("unified_asset_price_history.json")
DEFAULT_PRICE_ALERTS = Path("unified_price_alerts.json")
DEFAULT_ALERT_EVENTS = Path("unified_alert_events.jsonl")
DEFAULT_ALLOCATION_TARGETS = Path("unified_allocation_targets.json")
DEFAULT_WEEKLY_CLEANUP_REVIEW = Path("unified_weekly_cleanup_review.json")
DEFAULT_BENCHMARK_SYMBOL = "VOO"
DEFAULT_BENCHMARK_NAME = "Vanguard 500 Index Fund"
GOAL_NET_GAIN_TARGET = 100.0
GOAL_BASELINE_TOTAL_WORTH = 245.99
GOAL_BASELINE_GAIN = -2.82
GOAL_DRAWDOWN_LIMIT_PCT = 10.0
GOAL_CASH_FLOOR_PCT = 10.0
GOAL_CLEANUP_VALUE_THRESHOLD = 2.0
GOAL_REVIEW_LOSS_PCT = -7.0
GOAL_CORE_SYMBOLS = {"VWCE.DE", "VWCE", "2B76.DE", "2B76", "ASML", "ASML.AS"}
GOAL_OPPORTUNITY_SYMBOLS = {"ASML", "ASML.AS", "TSM", "NVDA", "CRWD", "ETN", "BDT.TO", "BDT"}
ASSET_CLASS_ENUM = ["Cash", "Stock", "ETF", "Fund", "Crypto", "Bond", "Option", "Other"]
SECTOR_ENUM = [
    "Cash",
    "Broad Market ETF",
    "Technology",
    "Semiconductors",
    "Industrials",
    "Defense & Aerospace",
    "Materials",
    "Energy & Mining",
    "Healthcare",
    "Consumer",
    "Financials",
    "Real Estate",
    "Utilities",
    "Communications",
    "Crypto",
    "Other",
]
GEOGRAPHY_ENUM = [
    "Cash",
    "Global",
    "United States",
    "Europe",
    "Germany",
    "France",
    "Netherlands",
    "United Kingdom",
    "Canada",
    "China / Taiwan",
    "Emerging Markets",
    "Other",
]
SCENARIO_DEFINITIONS = [
    {
        "key": "financial_crisis_2008",
        "name": "Financial Crisis 2008",
        "description": "Lehman collapse and global financial meltdown",
        "start": "2008-09-15",
        "end": "2009-03-09",
    },
    {
        "key": "covid_2020",
        "name": "COVID-19 Crash",
        "description": "Fastest bear market in modern history",
        "start": "2020-02-19",
        "end": "2020-03-23",
    },
    {
        "key": "rate_hike_2022",
        "name": "2022 Rate Hike",
        "description": "Fed tightening cycle impact",
        "start": "2022-01-03",
        "end": "2022-10-14",
    },
]


def build_unified_portfolio_state(
    *,
    trade_republic_report: Path | None = DEFAULT_TRADE_REPUBLIC_REPORT,
    trade_republic_transactions: Path | None = DEFAULT_TRADE_REPUBLIC_TRANSACTIONS,
    snaptrade_snapshot: Path | None = DEFAULT_SNAPTRADE_SNAPSHOT,
    portfolio_history_path: Path | None = None,
    scenario_history_path: Path | None = DEFAULT_SCENARIO_HISTORY,
    benchmark_history_path: Path | None = DEFAULT_BENCHMARK_HISTORY,
    classification_cache_path: Path | None = DEFAULT_CLASSIFICATION_CACHE,
    asset_price_history_path: Path | None = DEFAULT_ASSET_PRICE_HISTORY,
    price_alerts_path: Path | None = DEFAULT_PRICE_ALERTS,
    alert_events_path: Path | None = DEFAULT_ALERT_EVENTS,
    allocation_targets_path: Path | None = DEFAULT_ALLOCATION_TARGETS,
    weekly_cleanup_review_path: Path | None = DEFAULT_WEEKLY_CLEANUP_REVIEW,
    persist_history: bool = False,
) -> dict[str, Any]:
    source_errors: list[dict[str, str]] = []
    holdings: list[dict[str, Any]] = []
    dividends: list[dict[str, Any]] = []
    portfolio_summaries: list[dict[str, Any]] = []
    source_timestamps: dict[str, Any] = {}
    snaptrade_payload: dict[str, Any] | None = None

    if trade_republic_report is not None:
        report, error = read_json_file(trade_republic_report)
        if error:
            source_errors.append({"source": "trade_republic", "error": error})
        else:
            source_timestamps["trade_republic"] = (report.get("summary") or {}).get("report_timestamp")
            trade_republic_buy_dates = read_trade_republic_buy_dates(trade_republic_transactions)
            trade_republic_holdings = normalize_trade_republic_report(report, trade_republic_buy_dates=trade_republic_buy_dates)
            trade_republic_dividends = normalize_trade_republic_dividends(report)
            holdings.extend(trade_republic_holdings)
            dividends.extend(trade_republic_dividends)
            portfolio_summaries.append(build_trade_republic_portfolio_summary(report, trade_republic_holdings, trade_republic_dividends))

    if snaptrade_snapshot is not None:
        snapshot, error = read_json_file(snaptrade_snapshot)
        if error:
            source_errors.append({"source": "snaptrade", "error": error})
        else:
            snaptrade_payload = snapshot
            source_timestamps["snaptrade"] = snapshot.get("fetched_at")
            snaptrade_holdings = normalize_snaptrade_snapshot(snapshot)
            snaptrade_dividends = normalize_snaptrade_dividends(snapshot)
            holdings.extend(snaptrade_holdings)
            dividends.extend(snaptrade_dividends)
            portfolio_summaries.extend(build_snaptrade_portfolio_summaries(snapshot, snaptrade_holdings, snaptrade_dividends))

    classification_cache = read_classification_cache(classification_cache_path)
    attach_holding_classifications(holdings, classification_cache)
    holdings.sort(key=lambda row: float(row.get("current_value") or 0), reverse=True)
    dividends.sort(key=lambda row: str(row.get("timestamp") or row.get("date") or ""), reverse=True)
    combined_portfolio = build_combined_portfolio_summary(portfolio_summaries, dividends)
    selected_portfolio = combined_portfolio
    generated_at = datetime.now(UTC).isoformat()
    goal_plan = build_goal_plan(combined_portfolio, portfolio_summaries, holdings, generated_at=generated_at)
    weekly_cleanup_review = read_or_create_weekly_cleanup_review(
        weekly_cleanup_review_path,
        combined_portfolio=combined_portfolio,
        holdings=holdings,
        generated_at=generated_at,
    )
    current_history_rows = current_snapshot_history_rows([combined_portfolio, *portfolio_summaries], generated_at=generated_at)
    api_history_rows = []
    if snaptrade_payload is not None:
        api_history_rows = snaptrade_balance_history_rows(snaptrade_payload, portfolio_summaries)
    if persist_history and portfolio_history_path is not None:
        append_history_rows(portfolio_history_path, current_history_rows)
    stored_history_rows = read_history_rows(portfolio_history_path) if portfolio_history_path is not None else []
    portfolio_history = merge_history_rows([*api_history_rows, *stored_history_rows, *current_history_rows])
    analytics_portfolios = [combined_portfolio, *portfolio_summaries]
    monte_carlo = build_monte_carlo_by_portfolio(analytics_portfolios, portfolio_history)
    risk = build_risk_by_portfolio(analytics_portfolios, portfolio_history)
    frontier = build_frontier_by_portfolio(analytics_portfolios, holdings)
    scenario_history = read_scenario_history(scenario_history_path)
    scenarios = build_scenarios_by_portfolio(analytics_portfolios, holdings, scenario_history)
    benchmark_history = read_benchmark_history(benchmark_history_path)
    benchmark = build_benchmark_by_portfolio(analytics_portfolios, portfolio_history, benchmark_history)
    asset_price_history = read_asset_price_history(asset_price_history_path)
    correlation = build_correlation_by_portfolio(analytics_portfolios, holdings, asset_price_history)
    trade_alpha = build_trade_alpha_by_portfolio(analytics_portfolios, holdings, benchmark_history)
    price_alerts = evaluate_price_alerts(price_alerts_path, alert_events_path, analytics_portfolios, holdings)
    rebalancing = build_rebalancing_by_portfolio(analytics_portfolios, holdings, allocation_targets_path)
    return {
        "generated_at": generated_at,
        "read_only": True,
        "execution_enabled": False,
        "source_paths": {
            "trade_republic_report": str(trade_republic_report) if trade_republic_report else None,
            "trade_republic_transactions": str(trade_republic_transactions) if trade_republic_transactions else None,
            "snaptrade_snapshot": str(snaptrade_snapshot) if snaptrade_snapshot else None,
            "portfolio_history": str(portfolio_history_path) if portfolio_history_path else None,
            "scenario_history": str(scenario_history_path) if scenario_history_path else None,
            "benchmark_history": str(benchmark_history_path) if benchmark_history_path else None,
            "classification_cache": str(classification_cache_path) if classification_cache_path else None,
            "asset_price_history": str(asset_price_history_path) if asset_price_history_path else None,
            "price_alerts": str(price_alerts_path) if price_alerts_path else None,
            "alert_events": str(alert_events_path) if alert_events_path else None,
            "allocation_targets": str(allocation_targets_path) if allocation_targets_path else None,
            "weekly_cleanup_review": str(weekly_cleanup_review_path) if weekly_cleanup_review_path else None,
        },
        "source_timestamps": source_timestamps,
        "source_errors": source_errors,
        "summary": summarize_unified_holdings(holdings),
        "broker_summary": summarize_by(holdings, "broker"),
        "account_summary": summarize_by(holdings, "account_key"),
        "selected_portfolio_key": selected_portfolio["key"],
        "selected_portfolio": selected_portfolio,
        "combined_portfolio": combined_portfolio,
        "goal_plan": goal_plan,
        "weekly_cleanup_review": weekly_cleanup_review,
        "portfolio_summaries": [combined_portfolio, *portfolio_summaries],
        "holdings": holdings,
        "dividends": dividends,
        "portfolio_history": portfolio_history,
        "price_alerts": price_alerts,
        "rebalancing": rebalancing,
        "analytics": {
            "monte_carlo": monte_carlo,
            "risk": risk,
            "frontier": frontier,
            "scenarios": scenarios,
            "benchmark": benchmark,
            "correlation": correlation,
            "trade_alpha": trade_alpha,
        },
    }


def read_price_alerts(path: Path | None = DEFAULT_PRICE_ALERTS) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"version": 1, "alerts": []}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "alerts": []}
    if isinstance(parsed, list):
        return {"version": 1, "alerts": [row for row in parsed if isinstance(row, dict)]}
    if isinstance(parsed, dict):
        alerts = parsed.get("alerts") if isinstance(parsed.get("alerts"), list) else []
        return {"version": int(to_float(parsed.get("version")) or 1), "alerts": [row for row in alerts if isinstance(row, dict)]}
    return {"version": 1, "alerts": []}


def write_price_alerts(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def append_alert_event(path: Path | None, event: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, ensure_ascii=True, default=str) + "\n")


def read_alert_events(path: Path | None = DEFAULT_ALERT_EVENTS, limit: int = 100) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines[-limit:]:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def alert_id() -> str:
    return f"alert_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"


def create_price_alert(
    path: Path,
    *,
    portfolio_key: str,
    symbol: str,
    name: str | None,
    target_price: float,
    direction: str,
    basis_price: float | None = None,
    threshold_pct: float | None = None,
    scope: str = "single",
    currency: str | None = None,
) -> dict[str, Any]:
    payload = read_price_alerts(path)
    now = datetime.now(UTC).isoformat()
    alert = {
        "id": alert_id(),
        "portfolio_key": portfolio_key,
        "symbol": str(symbol or "").upper(),
        "name": name or symbol,
        "target_price": round(float(target_price), 6),
        "direction": "above" if str(direction).lower() == "above" else "below",
        "basis_price": round(float(basis_price), 6) if basis_price is not None else None,
        "threshold_pct": round(float(threshold_pct), 4) if threshold_pct is not None else None,
        "scope": scope,
        "status": "active",
        "currency": currency,
        "created_at": now,
        "updated_at": now,
        "triggered_at": None,
        "last_price": None,
    }
    payload["alerts"].append(alert)
    write_price_alerts(path, payload)
    return alert


def update_price_alert_status(path: Path, alert_id_value: str, status: str) -> dict[str, Any] | None:
    payload = read_price_alerts(path)
    now = datetime.now(UTC).isoformat()
    next_alerts: list[dict[str, Any]] = []
    updated: dict[str, Any] | None = None
    for alert in payload.get("alerts") or []:
        if str(alert.get("id")) != str(alert_id_value):
            next_alerts.append(alert)
            continue
        if status == "deleted":
            updated = {**alert, "status": "deleted", "updated_at": now}
            continue
        alert["status"] = status if status in {"active", "paused", "triggered"} else "paused"
        alert["updated_at"] = now
        if alert["status"] == "active":
            alert["triggered_at"] = None
        updated = alert
        next_alerts.append(alert)
    payload["alerts"] = next_alerts
    write_price_alerts(path, payload)
    return updated


def current_price_index(portfolios: list[dict[str, Any]], holdings: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    included_keys = {str(item.get("key")) for item in portfolios if item.get("include_in_combined") and item.get("kind") != "combined"}
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for holding in holdings:
        symbol = str(holding.get("ticker") or holding.get("broker_symbol") or holding.get("isin") or "").upper()
        if not symbol:
            continue
        account_key = str(holding.get("account_key") or "")
        price = to_float(holding.get("current_price"))
        quantity = abs(to_float(holding.get("quantity")) or 0.0)
        value = to_float(holding.get("current_value"))
        if price is None and value is not None and quantity:
            price = value / quantity
        if price is None:
            continue
        payload = {
            "symbol": symbol,
            "name": holding.get("name") or symbol,
            "price": round(price, 6),
            "currency": holding.get("currency"),
            "portfolio_key": account_key,
        }
        index[(account_key, symbol)] = payload
        if account_key in included_keys:
            index[("combined:live", symbol)] = payload
    return index


def evaluate_price_alerts(
    alerts_path: Path | None,
    events_path: Path | None,
    portfolios: list[dict[str, Any]],
    holdings: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = read_price_alerts(alerts_path)
    price_index = current_price_index(portfolios, holdings)
    now = datetime.now(UTC).isoformat()
    changed = False
    evaluated: list[dict[str, Any]] = []
    for alert in payload.get("alerts") or []:
        row = dict(alert)
        symbol = str(row.get("symbol") or "").upper()
        portfolio_key = str(row.get("portfolio_key") or "combined:live")
        price_payload = price_index.get((portfolio_key, symbol)) or price_index.get(("combined:live", symbol))
        current_price = to_float(price_payload.get("price")) if price_payload else None
        row["last_price"] = current_price
        row["currency"] = row.get("currency") or (price_payload or {}).get("currency")
        distance_pct = None
        target = to_float(row.get("target_price"))
        if current_price is not None and target:
            distance_pct = (target - current_price) / current_price * 100.0
        row["distance_pct"] = round(distance_pct, 4) if distance_pct is not None else None
        should_trigger = False
        if row.get("status") == "active" and current_price is not None and target is not None:
            if row.get("direction") == "above":
                should_trigger = current_price >= target
            else:
                should_trigger = current_price <= target
        if should_trigger:
            row["status"] = "triggered"
            row["triggered_at"] = now
            row["updated_at"] = now
            changed = True
            append_alert_event(
                events_path,
                {
                    "event": "price_alert_triggered",
                    "alert_id": row.get("id"),
                    "portfolio_key": portfolio_key,
                    "symbol": symbol,
                    "direction": row.get("direction"),
                    "target_price": target,
                    "trigger_price": current_price,
                    "currency": row.get("currency"),
                    "triggered_at": now,
                },
            )
        evaluated.append(row)
    if changed and alerts_path is not None:
        write_price_alerts(alerts_path, {"version": payload.get("version", 1), "alerts": evaluated})
    active = [row for row in evaluated if row.get("status") == "active"]
    triggered = [row for row in evaluated if row.get("status") == "triggered"]
    paused = [row for row in evaluated if row.get("status") == "paused"]
    closest = sorted(
        [row for row in active if to_float(row.get("distance_pct")) is not None],
        key=lambda row: abs(float(row.get("distance_pct") or 0.0)),
    )[:5]
    return {
        "summary": {
            "total": len(evaluated),
            "active": len(active),
            "triggered": len(triggered),
            "paused": len(paused),
            "closest": closest[0] if closest else None,
        },
        "alerts": sorted(evaluated, key=lambda row: str(row.get("created_at") or ""), reverse=True),
        "events": list(reversed(read_alert_events(events_path, limit=50))),
    }


def read_allocation_targets(path: Path | None = DEFAULT_ALLOCATION_TARGETS) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"version": 1, "targets": {}}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "targets": {}}
    if isinstance(parsed, dict) and isinstance(parsed.get("targets"), dict):
        return parsed
    return {"version": 1, "targets": {}}


def write_allocation_targets(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def save_allocation_targets(path: Path, portfolio_key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = read_allocation_targets(path)
    clean_rows = []
    for row in rows:
        symbol = str(row.get("symbol") or "").upper()
        target_pct = to_float(row.get("target_pct"))
        if not symbol or target_pct is None:
            continue
        clean_rows.append({"symbol": symbol, "target_pct": round(max(0.0, target_pct), 4)})
    payload.setdefault("targets", {})[portfolio_key] = {"updated_at": datetime.now(UTC).isoformat(), "rows": clean_rows}
    write_allocation_targets(path, payload)
    return payload["targets"][portfolio_key]


def build_rebalancing_by_portfolio(portfolios: list[dict[str, Any]], holdings: list[dict[str, Any]], targets_path: Path | None) -> dict[str, Any]:
    targets_payload = read_allocation_targets(targets_path)
    included_keys = {str(item.get("key")) for item in portfolios if item.get("include_in_combined") and item.get("kind") != "combined"}
    return {
        str(portfolio.get("key")): build_portfolio_rebalancing(
            portfolio,
            holdings_for_portfolio(portfolio, holdings, included_keys),
            targets_payload.get("targets", {}).get(str(portfolio.get("key"))) or {},
        )
        for portfolio in portfolios
        if portfolio.get("key")
    }


def build_portfolio_rebalancing(portfolio: dict[str, Any], holdings: list[dict[str, Any]], target_payload: dict[str, Any]) -> dict[str, Any]:
    currency = str(portfolio.get("currency") or "EUR")
    portfolio_key = str(portfolio.get("key") or "")
    value = to_float(portfolio.get("total_worth")) or sum(to_float(row.get("current_value")) or 0.0 for row in holdings)
    targets = {str(row.get("symbol") or "").upper(): to_float(row.get("target_pct")) or 0.0 for row in target_payload.get("rows") or [] if row.get("symbol")}
    rows = []
    seen: set[str] = set()
    for holding in sorted(holdings, key=lambda row: float(row.get("current_value") or 0), reverse=True):
        symbol = str(holding.get("ticker") or holding.get("broker_symbol") or holding.get("isin") or "").upper()
        if not symbol:
            continue
        seen.add(symbol)
        current_value = to_float(holding.get("current_value")) or 0.0
        current_pct = current_value / value * 100.0 if value else 0.0
        target_pct = targets.get(symbol, current_pct)
        target_value = value * target_pct / 100.0
        drift_value = target_value - current_value
        price = to_float(holding.get("current_price")) or 0.0
        suggested_quantity = drift_value / price if price else None
        rows.append(
            {
                "symbol": symbol,
                "name": holding.get("name") or symbol,
                "current_pct": round(current_pct, 4),
                "target_pct": round(target_pct, 4),
                "current_value": round(current_value, 2),
                "current_price": round(price, 6) if price else None,
                "target_value": round(target_value, 2),
                "drift_pct": round(target_pct - current_pct, 4),
                "drift_value": round(drift_value, 2),
                "suggested_action": "Buy" if drift_value > 0.01 else "Sell" if drift_value < -0.01 else "Hold",
                "suggested_quantity": round(suggested_quantity, 6) if suggested_quantity is not None else None,
                "currency": holding.get("currency") or currency,
            }
        )
    for symbol, target_pct in targets.items():
        if symbol in seen:
            continue
        target_value = value * target_pct / 100.0
        rows.append(
            {
                "symbol": symbol,
                "name": symbol,
                "current_pct": 0.0,
                "target_pct": round(target_pct, 4),
                "current_value": 0.0,
                "current_price": None,
                "target_value": round(target_value, 2),
                "drift_pct": round(target_pct, 4),
                "drift_value": round(target_value, 2),
                "suggested_action": "Buy",
                "suggested_quantity": None,
                "currency": currency,
            }
        )
    return {
        "portfolio_key": portfolio_key,
        "portfolio_name": portfolio.get("name"),
        "currency": currency,
        "portfolio_value": round(value, 2),
        "target_total_pct": round(sum(targets.values()), 4) if targets else None,
        "updated_at": target_payload.get("updated_at"),
        "rows": rows,
        "note": "Suggestions are read-only estimates. They are not broker orders.",
    }


def normalize_trade_republic_report(report: dict[str, Any], *, trade_republic_buy_dates: dict[str, str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    buy_dates = trade_republic_buy_dates or {}
    for item in report.get("holdings") or []:
        if not isinstance(item, dict):
            continue
        ticker = item.get("alpaca_ticker") or item.get("ticker")
        current_value = to_float(item.get("current_value"))
        cost_basis = to_float(item.get("open_cost_basis"))
        unrealized = to_float(item.get("unrealized_pl"))
        rows.append(
            {
                "broker": "trade_republic",
                "source": "trade_republic_report",
                "account_id": "trade_republic",
                "account_name": "Trade Republic",
                "account_key": "trade_republic:trade_republic",
                "institution": "Trade Republic",
                "name": item.get("name") or item.get("isin") or ticker,
                "ticker": ticker,
                "broker_symbol": item.get("ticker"),
                "isin": item.get("isin"),
                "asset_type": "security",
                "quantity": to_float(item.get("current_quantity")),
                "current_price": to_float(item.get("current_price")),
                "current_value": current_value,
                "cost_basis": cost_basis,
                "unrealized_pl": unrealized,
                "unrealized_pl_pct": to_float(item.get("unrealized_pl_pct")),
                "currency": "EUR",
                "status": item.get("historical_price_status") or item.get("alpaca_status") or "reported",
                "first_buy_date": buy_dates.get(str(item.get("isin") or "").upper()),
                "buy_date_source": "trade_republic_transactions_csv" if buy_dates.get(str(item.get("isin") or "").upper()) else "missing",
                "raw": item,
            }
        )
    return rows


def normalize_snaptrade_snapshot(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for account_row in snapshot.get("accounts") or []:
        if not isinstance(account_row, dict):
            continue
        account = account_row.get("account") if isinstance(account_row.get("account"), dict) else {}
        details = account_row.get("details") if isinstance(account_row.get("details"), dict) else {}
        account_id = first_value(account_row, "account_id") or first_value(account, "id", "accountId", "number") or "unknown"
        account_name = first_value(account, "name", "raw_name", "institution_name") or first_value(details, "name", "raw_name") or str(account_id)
        institution = (
            first_value(account, "institution_name", "brokerage.name", "brokerageAuthorization.name", "brokerage_authorization.name")
            or first_value(details, "institution_name", "brokerage.name", "brokerageAuthorization.name")
            or "SnapTrade"
        )
        account_currency = snaptrade_account_currency(account_row, account, details)
        buy_dates = snaptrade_buy_dates_by_symbol(account_row)
        positions = normalize_positions_container(account_row.get("positions"))
        for position in positions:
            instrument = nested_dict(position, "instrument") or {}
            instrument_currency = first_value(instrument, "currency.code", "currency") or first_value(position, "currency.code", "currency")
            symbol = (
                first_value(position, "symbol", "ticker", "universal_symbol.symbol", "universalSymbol.symbol")
                or first_value(instrument, "symbol", "raw_symbol", "ticker", "option_symbol")
                or first_value(position, "symbol.raw_symbol")
            )
            name = first_value(position, "name", "description") or first_value(instrument, "description", "name", "symbol") or symbol
            quantity = first_number(position, "units", "quantity", "open_quantity", "position")
            current_price = first_number(position, "price", "last_price", "market_price", "quote.last_trade_price")
            current_value = first_number(position, "market_value", "marketValue", "value", "notional_value")
            if current_value is None and quantity is not None and current_price is not None:
                current_value = quantity * current_price
            average_purchase_price = first_number(position, "average_purchase_price", "average_price")
            cost_basis_per_unit = first_number(position, "cost_basis")
            cost_basis_total = first_number(position, "total_cost", "book_value")
            cost_basis = cost_basis_total
            if cost_basis is None and cost_basis_per_unit is not None and quantity is not None:
                cost_basis = abs(quantity) * cost_basis_per_unit
            if cost_basis is None and average_purchase_price is not None and quantity is not None:
                cost_basis = abs(quantity) * average_purchase_price
            unrealized = first_number(position, "unrealized_pnl", "unrealized_pl", "gain_loss", "pnl")
            if unrealized is None and current_value is not None and cost_basis is not None:
                unrealized = current_value - cost_basis
            rows.append(
                {
                    "broker": "snaptrade",
                    "source": "snaptrade_snapshot",
                    "account_id": str(account_id),
                    "account_name": str(account_name),
                    "account_key": f"snaptrade:{account_id}",
                    "institution": str(institution),
                    "name": name,
                    "ticker": symbol,
                    "broker_symbol": symbol,
                    "isin": first_value(instrument, "isin"),
                    "asset_type": first_value(instrument, "kind", "type", "security_type") or first_value(position, "type"),
                    "quantity": quantity,
                    "current_price": current_price,
                    "current_value": current_value,
                    "cost_basis": cost_basis,
                    "unrealized_pl": unrealized,
                    "unrealized_pl_pct": pct(unrealized, cost_basis),
                    "currency": account_currency or first_value(position, "currency.code", "currency", "value_currency") or first_value(account, "currency.code", "currency"),
                    "instrument_currency": instrument_currency,
                    "price_currency_note": "SnapTrade position price/value/cost basis are treated as account currency.",
                    "status": "reported",
                    "first_buy_date": buy_dates.get(str(symbol or "").upper()),
                    "buy_date_source": "snaptrade_activities" if buy_dates.get(str(symbol or "").upper()) else "missing",
                    "raw": position,
                }
            )
    return [row for row in rows if row.get("ticker") or row.get("name") or row.get("current_value") is not None]


def normalize_trade_republic_dividends(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in report.get("dividends") or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "broker": "trade_republic",
                "account_id": "trade_republic",
                "account_name": "Trade Republic",
                "account_key": "trade_republic:trade_republic",
                "institution": "Trade Republic",
                "date": str(item.get("date") or "")[:10],
                "timestamp": item.get("timestamp") or item.get("date"),
                "name": item.get("name") or item.get("isin") or item.get("ticker"),
                "ticker": item.get("ticker"),
                "isin": item.get("isin"),
                "shares": to_float(item.get("shares")),
                "after_tax_amount": to_float(item.get("after_tax_amount")) or 0.0,
                "tax_amount": to_float(item.get("tax_amount")) or 0.0,
                "gross_amount": to_float(item.get("gross_amount")) or 0.0,
                "currency": item.get("currency") or "EUR",
                "source": item.get("source") or "trade_republic_report",
            }
        )
    return [row for row in rows if row.get("date") and (row.get("after_tax_amount") or row.get("tax_amount"))]


def build_trade_republic_portfolio_summary(report: dict[str, Any], holdings: list[dict[str, Any]], dividends: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    summary = report.get("summary") or {}
    equity = to_float(summary.get("total_current_value"))
    cost_basis = to_float(summary.get("total_open_cost_basis"))
    unrealized_pl = to_float(summary.get("total_unrealized_pl"))
    historical_buy_cash = to_float(summary.get("total_historical_buy_cash"))
    historical_sell_cash = to_float(summary.get("total_historical_sell_cash")) or 0.0
    dividend_after_tax = sum_numeric(dividends or [], "after_tax_amount")
    if equity is None:
        equity = sum_numeric(holdings, "current_value")
    if cost_basis is None:
        cost_basis = sum_numeric(holdings, "cost_basis")
    if unrealized_pl is None:
        unrealized_pl = equity - cost_basis
    realized_pl = trade_republic_realized_pl(report)
    account_invested_capital = (historical_buy_cash - historical_sell_cash) if historical_buy_cash is not None else cost_basis
    total_growth = unrealized_pl + realized_pl
    return normalize_portfolio_summary(
        {
            "key": "trade_republic:trade_republic",
            "kind": "individual",
            "broker": "trade_republic",
            "name": "Portfolio TradeRepublic",
            "institution": "Trade Republic",
            "account_id": "trade_republic",
            "currency": "EUR",
            "equities": equity,
            "cash": 0.0,
            "initially_invested": account_invested_capital,
            "account_invested_capital": account_invested_capital,
            "holdings_cost_basis": cost_basis,
            "unrealized_pl": unrealized_pl,
            "capital_gain": total_growth,
            "realized_pl": realized_pl,
            "dividend_after_tax": dividend_after_tax,
            "dividend_tax": sum_numeric(dividends or [], "tax_amount"),
            "holding_count": len(holdings),
            "include_in_combined": True,
            "is_paper": False,
            "note": "Cash is not available in the current Trade Republic report export.",
        }
    )


def build_snaptrade_portfolio_summaries(snapshot: dict[str, Any], holdings: list[dict[str, Any]], dividends: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    holdings_by_account: dict[str, list[dict[str, Any]]] = {}
    for holding in holdings:
        holdings_by_account.setdefault(str(holding.get("account_key") or ""), []).append(holding)
    dividends_by_account: dict[str, list[dict[str, Any]]] = {}
    for dividend in dividends or []:
        dividends_by_account.setdefault(str(dividend.get("account_key") or ""), []).append(dividend)
    realized_by_account = snaptrade_realized_pl_by_account(snapshot)
    invested_by_account = snaptrade_invested_capital_by_account(snapshot)
    summaries: list[dict[str, Any]] = []
    for account_row in snapshot.get("accounts") or []:
        if not isinstance(account_row, dict):
            continue
        account = account_row.get("account") if isinstance(account_row.get("account"), dict) else {}
        details = account_row.get("details") if isinstance(account_row.get("details"), dict) else {}
        account_id = first_value(account_row, "account_id") or first_value(account, "id", "accountId", "number") or "unknown"
        account_key = f"snaptrade:{account_id}"
        account_holdings = holdings_by_account.get(account_key, [])
        account_dividends = dividends_by_account.get(account_key, [])
        account_name = first_value(account, "name", "raw_name", "institution_name") or first_value(details, "name", "raw_name") or str(account_id)
        institution = first_value(account, "institution_name", "brokerage.name") or first_value(details, "institution_name", "brokerage.name") or "SnapTrade"
        equities = sum_numeric(account_holdings, "current_value")
        cash, currency = snaptrade_cash_balance(account_row.get("balances"), positions_value=equities)
        cost_basis = sum_numeric(account_holdings, "cost_basis")
        invested_capital = invested_by_account.get(str(account_id), {}).get("total")
        if invested_capital is None or invested_capital == 0:
            invested_capital = cost_basis + cash
        unrealized_pl = sum_numeric(account_holdings, "unrealized_pl")
        realized_pl = realized_by_account.get(str(account_id), {}).get("total", 0.0)
        total_growth = unrealized_pl + (to_float(realized_pl) or 0.0)
        is_paper = bool(first_value(account, "is_paper"))
        summaries.append(
            normalize_portfolio_summary(
                {
                    "key": account_key,
                    "kind": "individual",
                    "broker": "snaptrade",
                    "name": str(account_name),
                    "institution": str(institution),
                    "account_id": str(account_id),
                    "currency": currency or first_value(account, "currency.code", "currency") or "EUR",
                    "equities": equities,
                    "cash": cash,
                    "initially_invested": invested_capital,
                    "account_invested_capital": invested_capital,
                    "holdings_cost_basis": cost_basis,
                    "unrealized_pl": unrealized_pl,
                    "capital_gain": total_growth,
                    "realized_pl": realized_pl,
                    "realized_pl_by_date": realized_by_account.get(str(account_id), {}).get("by_date", {}),
                    "dividend_after_tax": sum_numeric(account_dividends, "after_tax_amount"),
                    "dividend_tax": sum_numeric(account_dividends, "tax_amount"),
                    "holding_count": len(account_holdings),
                    "include_in_combined": not is_paper,
                    "is_paper": is_paper,
                    "note": "Excluded from combined portfolio because this is paper." if is_paper else None,
                }
            )
        )
    return summaries


def snaptrade_account_currency(account_row: dict[str, Any], account: dict[str, Any] | None = None, details: dict[str, Any] | None = None) -> str | None:
    account = account or {}
    details = details or {}
    cash, cash_currency = snaptrade_cash_balance(account_row.get("balances"), positions_value=0.0)
    if cash_currency:
        return cash_currency
    return (
        first_value(details, "balance.total.currency", "currency.code", "currency")
        or first_value(account, "balance.total.currency", "currency.code", "currency")
    )


def build_combined_portfolio_summary(portfolios: list[dict[str, Any]], dividends: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    included = [item for item in portfolios if item.get("include_in_combined")]
    included_keys = {str(item.get("key")) for item in included}
    included_dividends = [row for row in dividends or [] if str(row.get("account_key")) in included_keys]
    currencies = {str(item.get("currency")) for item in included if item.get("currency")}
    equities = sum(to_float(item.get("equities")) or 0.0 for item in included)
    cash = sum(to_float(item.get("cash")) or 0.0 for item in included)
    initially_invested = sum(to_float(item.get("initially_invested")) or 0.0 for item in included)
    account_invested_capital = sum(to_float(item.get("account_invested_capital")) or to_float(item.get("initially_invested")) or 0.0 for item in included)
    holdings_cost_basis = sum(to_float(item.get("holdings_cost_basis")) or to_float(item.get("initially_invested")) or 0.0 for item in included)
    unrealized_pl = sum(to_float(item.get("unrealized_pl")) or 0.0 for item in included)
    capital_gain = sum(to_float(item.get("capital_gain")) or 0.0 for item in included)
    realized_pl = sum(to_float(item.get("realized_pl")) or 0.0 for item in included)
    return normalize_portfolio_summary(
        {
            "key": "combined:live",
            "kind": "combined",
            "broker": "combined",
            "name": "Portfolio Combined Total Worth",
            "institution": "Combined",
            "account_id": "combined_live",
            "currency": next(iter(currencies)) if len(currencies) == 1 else "MIXED",
            "equities": equities,
            "cash": cash,
            "initially_invested": initially_invested,
            "account_invested_capital": account_invested_capital,
            "holdings_cost_basis": holdings_cost_basis,
            "unrealized_pl": unrealized_pl,
            "capital_gain": capital_gain,
            "realized_pl": realized_pl,
            "dividend_after_tax": sum_numeric(included_dividends, "after_tax_amount"),
            "dividend_tax": sum_numeric(included_dividends, "tax_amount"),
            "holding_count": sum(int(item.get("holding_count") or 0) for item in included),
            "include_in_combined": True,
            "is_paper": False,
            "included_portfolio_count": len(included),
            "excluded_portfolio_count": len(portfolios) - len(included),
            "note": "Combined excludes paper portfolios.",
        }
    )


def build_goal_plan(
    combined_portfolio: dict[str, Any],
    portfolio_summaries: list[dict[str, Any]],
    holdings: list[dict[str, Any]],
    *,
    generated_at: str,
) -> dict[str, Any]:
    included_keys = {
        str(item.get("key"))
        for item in portfolio_summaries
        if item.get("include_in_combined") and item.get("kind") != "combined"
    }
    target_gain = GOAL_NET_GAIN_TARGET
    current_gain = to_float(combined_portfolio.get("capital_gain")) or 0.0
    total_worth = to_float(combined_portfolio.get("total_worth")) or 0.0
    cash = to_float(combined_portfolio.get("cash")) or 0.0
    invested = to_float(combined_portfolio.get("initially_invested")) or 0.0
    remaining_gain = max(0.0, target_gain - current_gain)
    baseline_total_worth = GOAL_BASELINE_TOTAL_WORTH if GOAL_BASELINE_TOTAL_WORTH > 0 else total_worth
    baseline_gain = GOAL_BASELINE_GAIN
    drawdown_budget = round(baseline_total_worth * GOAL_DRAWDOWN_LIMIT_PCT / 100.0, 2)
    stop_adding_risk_at_gain = round(baseline_gain - drawdown_budget, 2)
    target_total_worth = round(total_worth + remaining_gain, 2)
    cash_floor_value = round(total_worth * GOAL_CASH_FLOOR_PCT / 100.0, 2)
    cash_pct = pct(cash, total_worth) or 0.0
    return_needed_pct = pct(remaining_gain, total_worth) or 0.0
    status = "on_track" if remaining_gain <= 0 else "stretch"
    if current_gain <= stop_adding_risk_at_gain:
        status = "risk_off"
    elif return_needed_pct >= 35.0:
        status = "high_stretch"

    actions: list[dict[str, Any]] = []
    for holding in holdings:
        if str(holding.get("account_key") or "") not in included_keys:
            continue
        action = goal_action_for_holding(holding, total_worth=total_worth)
        holding["goal_action"] = action
        actions.append(action)

    action_counts: dict[str, int] = {}
    for action in actions:
        key = str(action.get("action") or "no_trade")
        action_counts[key] = action_counts.get(key, 0) + 1

    weekly_review = [
        "Refresh broker data manually before reviewing the goal tracker.",
        "Record combined value, current gain, remaining gain, cash, and top movers.",
        "Only consider adding risk when a holding passes quality, trend, liquidity, concentration, and thesis checks.",
        "If the portfolio reaches the drawdown guardrail, stop adding risk and move to review-only mode.",
    ]
    guardrails = [
        "Read-only plan: no live or paper orders are created by this dashboard.",
        "Deposits do not count as profit toward the 100 EUR gain target.",
        "Do not treat stale, wide, or missing quotes as executable prices.",
        "No options recommendation is implementation-ready without live bid/ask, liquidity, spread, buying power, and explicit execution approval.",
    ]
    plain_english = {
        "where_you_are": f"Portfolio is currently {round(current_gain, 2)} EUR from profit/loss tracking.",
        "target_meaning": f"To reach 100 EUR profit, portfolio still needs {round(remaining_gain, 2)} EUR more gain.",
        "difficulty": f"That needs about {round(return_needed_pct, 2)}% more return from the current portfolio value, so this is a stretch goal.",
        "risk_limit": f"Do not let the attempt cost more than about {drawdown_budget:.2f} EUR from the starting point.",
        "red_line": f"If total gain falls to {stop_adding_risk_at_gain:.2f} EUR, stop adding risk and review only.",
        "cash_rule": f"Keep at least {cash_floor_value:.2f} EUR in cash; current cash cushion is {round(cash - cash_floor_value, 2)} EUR.",
        "weekly_action": "Each week: refresh portfolios, check remaining target, review losers, review big winners, and only watch possible adds.",
        "not_a_trade_signal": "This dashboard does not tell you to buy or sell automatically. It shows what to review.",
    }
    return {
        "version": 1,
        "name": "100 EUR Net Gain Plan",
        "generated_at": generated_at,
        "read_only": True,
        "execution_enabled": False,
        "target_gain": round(target_gain, 2),
        "baseline_total_worth": round(baseline_total_worth, 2),
        "baseline_gain": round(baseline_gain, 2),
        "current_gain": round(current_gain, 2),
        "remaining_gain": round(remaining_gain, 2),
        "return_needed_pct": round(return_needed_pct, 2),
        "target_total_worth": target_total_worth,
        "current_total_worth": round(total_worth, 2),
        "invested_capital": round(invested, 2),
        "cash": round(cash, 2),
        "cash_floor_pct": GOAL_CASH_FLOOR_PCT,
        "cash_floor_value": cash_floor_value,
        "cash_cushion": round(cash - cash_floor_value, 2),
        "cash_pct": round(cash_pct, 2),
        "drawdown_limit_pct": GOAL_DRAWDOWN_LIMIT_PCT,
        "drawdown_budget": drawdown_budget,
        "stop_adding_risk_at_gain": stop_adding_risk_at_gain,
        "cleanup_value_threshold": GOAL_CLEANUP_VALUE_THRESHOLD,
        "review_loss_pct": abs(GOAL_REVIEW_LOSS_PCT),
        "status": status,
        "mode": "read_only_monitoring",
        "source_of_truth": "combined:live",
        "paper_excluded": True,
        "action_counts": action_counts,
        "plain_english": plain_english,
        "weekly_review": weekly_review,
        "guardrails": guardrails,
        "screening_discipline": {
            "name": "CAN SLIM inspired screen",
            "checks": [
                "Earnings and sales growth are improving.",
                "The holding is a market or sector leader, not a weak laggard.",
                "Price and volume show real demand instead of a weak chart.",
                "The broad market trend supports adding risk.",
                "The spread and liquidity are acceptable for the position size.",
            ],
        },
    }


def read_or_create_weekly_cleanup_review(
    path: Path | None,
    *,
    combined_portfolio: dict[str, Any],
    holdings: list[dict[str, Any]],
    generated_at: str,
) -> dict[str, Any]:
    if path is not None and path.exists():
        parsed, error = read_json_file(path)
        if not error and parsed:
            return parsed
    review = build_weekly_cleanup_review(combined_portfolio, holdings, generated_at=generated_at)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(review, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    return review


def build_weekly_cleanup_review(
    combined_portfolio: dict[str, Any],
    holdings: list[dict[str, Any]],
    *,
    generated_at: str,
) -> dict[str, Any]:
    review_date = next_weekly_cleanup_date(generated_at[:10])
    included = [row for row in holdings if row.get("account_key") != "snaptrade:23065fc8-a36d-4dee-9b7d-01f53591def5"]
    tiny = []
    cleanup_candidates = []
    not_urgent = []
    for holding in included:
        value = to_float(holding.get("current_value")) or 0.0
        cost_basis = to_float(holding.get("cost_basis")) or 0.0
        unrealized_pct = to_float(holding.get("unrealized_pl_pct"))
        if unrealized_pct is None and cost_basis:
            unrealized_pct = ((value - cost_basis) / cost_basis) * 100.0
        unrealized_pct = round(unrealized_pct or 0.0, 2)
        if value >= GOAL_CLEANUP_VALUE_THRESHOLD:
            continue
        item = {
            "symbol": str(holding.get("ticker") or holding.get("broker_symbol") or holding.get("isin") or "").upper(),
            "name": holding.get("name"),
            "broker": holding.get("institution") or holding.get("broker"),
            "current_value": round(value, 2),
            "cost_basis": round(cost_basis, 2),
            "unrealized_pl_pct": unrealized_pct,
            "action": "review_on_cleanup_date",
            "plain_action": (
                f"On {review_date}, refresh portfolio. If still below 2 EUR/USD and still worse than -7%, mark for sell/cleanup. "
                "If fee or spread costs more than what you save, leave it alone and never add more."
            ),
        }
        tiny.append(item)
        if unrealized_pct <= GOAL_REVIEW_LOSS_PCT:
            cleanup_candidates.append(item)
        else:
            not_urgent.append(
                {
                    **item,
                    "action": "watch",
                    "plain_action": "Small position. Not urgent. Keep watching. Do not add more money now.",
                }
            )
    cleanup_candidates.sort(key=lambda row: (row.get("unrealized_pl_pct") or 0.0, row.get("current_value") or 0.0))
    not_urgent.sort(key=lambda row: (row.get("unrealized_pl_pct") or 0.0, row.get("current_value") or 0.0))
    return {
        "version": 1,
        "name": "Weekly Portfolio Cleanup Review",
        "automation_id": "weekly-portfolio-cleanup-review",
        "generated_at": generated_at,
        "review_date": review_date,
        "schedule": "Every Tuesday at 09:00 Amsterdam time",
        "read_only": True,
        "execution_enabled": False,
        "portfolio_key": combined_portfolio.get("key") or "combined:live",
        "portfolio_value": round(to_float(combined_portfolio.get("total_worth")) or 0.0, 2),
        "summary": {
            "plain": f"Later means {review_date}, the next planned weekly cleanup review. Until then: do not add money to tiny losing positions.",
            "candidate_count": len(cleanup_candidates),
            "not_urgent_count": len(not_urgent),
            "tiny_position_count": len(tiny),
        },
        "rules": [
            "Refresh portfolio first.",
            "Look only at tiny positions under 2 EUR/USD.",
            "If still under 2 and still losing more than -7%, mark for sell/cleanup.",
            "If transaction fee or spread is bigger than value saved, leave it alone, but never add.",
            "If a tiny position recovers above 2 and improves, keep watching.",
        ],
        "top_message": "No trades. No orders. This is a cleanup checklist for the next weekly review.",
        "actions_now": [
            "Do not add more money to tiny losing positions before the cleanup review.",
            "Use the next review date. Do not choose a random day.",
            "Refresh portfolio before making the cleanup list.",
        ],
        "cleanup_candidates": cleanup_candidates,
        "not_urgent": not_urgent,
    }


def next_weekly_cleanup_date(date_text: str) -> str:
    try:
        current = datetime.fromisoformat(date_text[:10]).date()
    except ValueError:
        current = datetime.now(UTC).date()
    target_weekday = 1  # Tuesday
    days_until = (target_weekday - current.weekday()) % 7
    if days_until == 0:
        days_until = 7
    return (current + timedelta(days=days_until)).isoformat()


def goal_action_for_holding(holding: dict[str, Any], *, total_worth: float) -> dict[str, Any]:
    symbol = str(holding.get("ticker") or holding.get("broker_symbol") or holding.get("isin") or "").upper()
    value = to_float(holding.get("current_value")) or 0.0
    cost_basis = to_float(holding.get("cost_basis")) or 0.0
    unrealized_pct = to_float(holding.get("unrealized_pl_pct"))
    if unrealized_pct is None and cost_basis:
        unrealized_pct = ((value - cost_basis) / cost_basis) * 100.0
    unrealized_pct = unrealized_pct or 0.0
    weight_pct = pct(value, total_worth) or 0.0
    reasons: list[str] = []
    bucket = "watch"
    action = "no_trade"

    if value < GOAL_CLEANUP_VALUE_THRESHOLD:
        bucket = "cleanup"
        action = "cleanup_watch"
        reasons.append("Position is below the cleanup threshold, where spread and broker friction can dominate.")
    elif unrealized_pct <= GOAL_REVIEW_LOSS_PCT:
        bucket = "review"
        action = "review_loss"
        reasons.append("Position is down at least 7%, so review the thesis before adding capital.")
    elif symbol in GOAL_CORE_SYMBOLS:
        bucket = "core"
        action = "hold_core"
        reasons.append("Core holding for diversified or high-quality portfolio exposure.")
    elif symbol in GOAL_OPPORTUNITY_SYMBOLS:
        bucket = "opportunity"
        action = "watch_opportunity"
        reasons.append("Opportunity candidate, but only add after quality, trend, market, spread, and concentration checks pass.")
    else:
        reasons.append("No automatic add rule applies; keep it on watch unless the screen improves.")

    if weight_pct >= 20.0:
        reasons.append("Large portfolio weight; avoid letting one holding dominate the 100 EUR goal.")
    if unrealized_pct > 20.0:
        reasons.append("Strong open gain; review whether partial profit-taking is better than relying on one winner.")

    return {
        "symbol": symbol,
        "name": holding.get("name"),
        "account_key": holding.get("account_key"),
        "bucket": bucket,
        "action": action,
        "current_value": round(value, 2),
        "weight_pct": round(weight_pct, 2),
        "unrealized_pl_pct": round(unrealized_pct, 2),
        "reasons": reasons,
        "read_only": True,
    }


def current_snapshot_history_rows(portfolios: list[dict[str, Any]], *, generated_at: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for portfolio in portfolios:
        rows.append(
            normalize_history_row(
                {
                    "timestamp": generated_at,
                    "date": generated_at[:10],
                    "portfolio_key": portfolio.get("key"),
                    "portfolio_name": portfolio.get("name"),
                    "currency": portfolio.get("currency"),
                    "total_worth": portfolio.get("total_worth"),
                    "equities": portfolio.get("equities"),
                    "cash": portfolio.get("cash"),
                    "initially_invested": portfolio.get("initially_invested"),
                    "account_invested_capital": portfolio.get("account_invested_capital"),
                    "holdings_cost_basis": portfolio.get("holdings_cost_basis"),
                    "unrealized_pl": portfolio.get("unrealized_pl"),
                    "capital_gain": portfolio.get("capital_gain"),
                    "realized_pl": portfolio.get("realized_pl"),
                    "simple_return_pct": portfolio.get("simple_return_pct"),
                    "source": "current_snapshot",
                    "granularity": "snapshot",
                }
            )
        )
    return [row for row in rows if row.get("portfolio_key")]


def snaptrade_balance_history_rows(snapshot: dict[str, Any], portfolios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    portfolio_by_account = {str(item.get("account_id")): item for item in portfolios if item.get("broker") == "snaptrade"}
    realized_by_account = snaptrade_realized_pl_by_account(snapshot)
    invested_by_account = snaptrade_invested_capital_by_account(snapshot)
    rows: list[dict[str, Any]] = []
    for account_row in snapshot.get("accounts") or []:
        if not isinstance(account_row, dict):
            continue
        account_id = str(account_row.get("account_id") or "")
        portfolio = portfolio_by_account.get(account_id)
        if not portfolio:
            continue
        realized_by_date = realized_by_account.get(account_id, {}).get("by_date", {})
        invested_by_date = invested_by_account.get(account_id, {}).get("by_date", {})
        for item in flatten_balance_history(account_row.get("balance_history")):
            timestamp = first_value(item, "timestamp", "time", "date", "trade_date", "created_at")
            total_worth = first_number(
                item,
                "total_value",
                "totalValue",
                "value",
                "balance",
                "account_value",
                "accountValue",
                "net_liquidation_value",
                "netLiquidationValue",
            )
            cash = first_number(item, "cash", "cash_value", "cashValue")
            equities = first_number(item, "market_value", "marketValue", "positions_value", "positionsValue")
            if total_worth is None and cash is not None and equities is not None:
                total_worth = cash + equities
            if total_worth is None:
                continue
            timestamp_text = str(timestamp or snapshot.get("fetched_at") or "")
            date_key = timestamp_text[:10]
            realized_pl = cumulative_value_on_or_before(realized_by_date, date_key)
            invested_capital = cumulative_value_on_or_before(invested_by_date, date_key) if invested_by_date else portfolio.get("initially_invested")
            rows.append(
                normalize_history_row(
                    {
                        "timestamp": timestamp_text,
                        "date": date_key,
                        "portfolio_key": portfolio.get("key"),
                        "portfolio_name": portfolio.get("name"),
                        "currency": first_value(item, "currency.code", "currency") or portfolio.get("currency"),
                        "total_worth": total_worth,
                        "equities": equities,
                        "cash": cash,
                        "initially_invested": invested_capital,
                        "capital_gain": None,
                        "realized_pl": realized_pl,
                        "simple_return_pct": None,
                        "source": "snaptrade_balance_history",
                        "granularity": "daily",
                    }
                )
            )
    rows.extend(combined_history_rows(rows, portfolios))
    return rows


def flatten_balance_history(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and isinstance(value.get("error"), dict):
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("data", "results", "balances", "history", "values", "account_value_history", "accountValueHistory"):
            if isinstance(value.get(key), list):
                return [item for item in value[key] if isinstance(item, dict)]
    return []


def trade_republic_realized_pl(report: dict[str, Any]) -> float:
    realized = 0.0
    for item in report.get("holdings") or []:
        if not isinstance(item, dict):
            continue
        sell_cash = to_float(item.get("historical_sell_cash")) or 0.0
        sell_shares = abs(to_float(item.get("transaction_sell_shares")) or 0.0)
        paid_price = to_float(item.get("weighted_paid_price")) or to_float(item.get("broker_avg_cost"))
        if sell_cash and sell_shares and paid_price is not None:
            realized += sell_cash - sell_shares * paid_price
    return round(realized, 2)


def snaptrade_realized_pl_by_account(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for account_row in snapshot.get("accounts") or []:
        if not isinstance(account_row, dict):
            continue
        account_id = str(account_row.get("account_id") or "")
        if not account_id:
            continue
        activities = flatten_activities(account_row.get("activities"))
        by_date = realized_pl_by_date_from_activities(activities)
        result[account_id] = {"total": sum(by_date.values()), "by_date": cumulative_by_date(by_date)}
    return result


def snaptrade_invested_capital_by_account(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for account_row in snapshot.get("accounts") or []:
        if not isinstance(account_row, dict):
            continue
        account_id = str(account_row.get("account_id") or "")
        if not account_id:
            continue
        by_date = invested_capital_by_date_from_activities(flatten_activities(account_row.get("activities")))
        result[account_id] = {"total": cumulative_value_on_or_before(cumulative_by_date(by_date), "9999-12-31"), "by_date": cumulative_by_date(by_date)}
    return result


def normalize_snaptrade_dividends(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dividend_types = {"DIVIDEND", "DIVIDENDS", "DIV", "CASH_DIVIDEND", "QUALIFIED_DIVIDEND", "ORDINARY_DIVIDEND"}
    for account_row in snapshot.get("accounts") or []:
        if not isinstance(account_row, dict):
            continue
        account = account_row.get("account") if isinstance(account_row.get("account"), dict) else {}
        details = account_row.get("details") if isinstance(account_row.get("details"), dict) else {}
        account_id = first_value(account_row, "account_id") or first_value(account, "id", "accountId", "number") or "unknown"
        account_key = f"snaptrade:{account_id}"
        account_name = first_value(account, "name", "raw_name", "institution_name") or first_value(details, "name", "raw_name") or str(account_id)
        institution = first_value(account, "institution_name", "brokerage.name") or first_value(details, "institution_name", "brokerage.name") or "SnapTrade"
        for activity in flatten_activities(account_row.get("activities")):
            activity_type = str(activity.get("type") or first_value(activity, "activity_type", "category") or "").upper()
            description = str(first_value(activity, "description", "name") or "")
            if activity_type not in dividend_types and "DIVIDEND" not in activity_type and "DIVIDEND" not in description.upper():
                continue
            amount = first_number(activity, "amount", "net_amount", "cash_amount", "value")
            tax = abs(first_number(activity, "withholding_tax", "tax", "taxes") or 0.0)
            if amount is None and tax == 0:
                continue
            symbol = activity_symbol(activity)
            date_key = str(first_value(activity, "trade_date", "settlement_date", "date") or "")[:10]
            rows.append(
                {
                    "broker": "snaptrade",
                    "account_id": str(account_id),
                    "account_name": str(account_name),
                    "account_key": account_key,
                    "institution": str(institution),
                    "date": date_key,
                    "timestamp": first_value(activity, "trade_date", "settlement_date", "date"),
                    "name": description or symbol,
                    "ticker": symbol,
                    "isin": first_value(activity, "symbol.isin", "isin"),
                    "shares": first_number(activity, "units", "quantity"),
                    "after_tax_amount": amount or 0.0,
                    "tax_amount": tax,
                    "gross_amount": (amount or 0.0) + tax,
                    "currency": first_value(activity, "currency.code", "currency") or first_value(account, "currency.code", "currency") or "EUR",
                    "source": "snaptrade_activities",
                }
            )
    return [row for row in rows if row.get("date")]


def invested_capital_by_date_from_activities(activities: list[dict[str, Any]]) -> dict[str, float]:
    by_date: dict[str, float] = {}
    contribution_types = {"CONTRIBUTION", "DEPOSIT", "TRANSFER_IN", "CASH_IN", "ACAT_IN", "DIVIDEND_REINVESTMENT"}
    withdrawal_types = {"WITHDRAWAL", "TRANSFER_OUT", "CASH_OUT", "ACAT_OUT"}
    for activity in activities:
        activity_type = str(activity.get("type") or "").upper()
        amount = to_float(activity.get("amount"))
        date_key = str(first_value(activity, "trade_date", "settlement_date", "date") or "")[:10]
        if not date_key or amount is None:
            continue
        if activity_type in contribution_types:
            by_date[date_key] = by_date.get(date_key, 0.0) + abs(amount)
        elif activity_type in withdrawal_types:
            by_date[date_key] = by_date.get(date_key, 0.0) - abs(amount)
    return {key: round(value, 2) for key, value in by_date.items()}


def flatten_activities(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and isinstance(value.get("error"), dict):
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("data", "results", "activities", "history"):
            if isinstance(value.get(key), list):
                return [item for item in value[key] if isinstance(item, dict)]
    return []


def realized_pl_by_date_from_activities(activities: list[dict[str, Any]]) -> dict[str, float]:
    lots_by_symbol: dict[str, list[dict[str, float]]] = {}
    realized_by_date: dict[str, float] = {}
    sorted_activities = sorted(activities, key=lambda item: str(first_value(item, "trade_date", "settlement_date", "date") or ""))
    for activity in sorted_activities:
        activity_type = str(activity.get("type") or "").upper()
        if activity_type not in {"BUY", "SELL"}:
            continue
        symbol = activity_symbol(activity)
        if not symbol:
            continue
        units = to_float(activity.get("units"))
        price = to_float(activity.get("price"))
        fee = abs(to_float(activity.get("fee")) or 0.0)
        if units is None or price is None or units == 0:
            continue
        quantity = abs(units)
        gross = quantity * price
        date_key = str(first_value(activity, "trade_date", "settlement_date", "date") or "")[:10]
        if activity_type == "BUY" or units > 0:
            cost = gross + fee
            lots_by_symbol.setdefault(symbol, []).append({"quantity": quantity, "unit_cost": cost / quantity})
            continue
        proceeds = gross - fee
        cost_used = consume_lots(lots_by_symbol.setdefault(symbol, []), quantity)
        if cost_used is None:
            continue
        realized_by_date[date_key] = realized_by_date.get(date_key, 0.0) + proceeds - cost_used
    return {key: round(value, 2) for key, value in realized_by_date.items()}


def activity_symbol(activity: dict[str, Any]) -> str | None:
    option_symbol = first_value(activity, "option_symbol.ticker", "option_symbol.symbol")
    if option_symbol:
        return str(option_symbol)
    return str(first_value(activity, "symbol.symbol", "symbol.raw_symbol", "symbol") or "") or None


def consume_lots(lots: list[dict[str, float]], quantity: float) -> float | None:
    remaining = quantity
    cost = 0.0
    while remaining > 1e-9 and lots:
        lot = lots[0]
        lot_quantity = float(lot.get("quantity") or 0.0)
        if lot_quantity <= 1e-9:
            lots.pop(0)
            continue
        used = min(remaining, lot_quantity)
        cost += used * float(lot.get("unit_cost") or 0.0)
        lot["quantity"] = lot_quantity - used
        remaining -= used
        if lot["quantity"] <= 1e-9:
            lots.pop(0)
    if remaining > max(1e-6, quantity * 1e-4):
        return None
    return cost


def cumulative_by_date(values: dict[str, float]) -> dict[str, float]:
    total = 0.0
    cumulative: dict[str, float] = {}
    for date_key in sorted(key for key in values if key):
        total += values[date_key]
        cumulative[date_key] = round(total, 2)
    return cumulative


def cumulative_value_on_or_before(values: dict[str, Any], date_key: str) -> float:
    total = 0.0
    for key in sorted(str(key) for key in values):
        if key <= date_key:
            total = to_float(values.get(key)) or total
        else:
            break
    return round(total, 2)


def combined_history_rows(rows: list[dict[str, Any]], portfolios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    included_keys = {str(item.get("key")) for item in portfolios if item.get("include_in_combined") and item.get("kind") != "combined"}
    if not included_keys:
        return []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("portfolio_key")) in included_keys and row.get("date"):
            grouped.setdefault(str(row["date"]), []).append(row)
    combined: list[dict[str, Any]] = []
    for date_key, date_rows in grouped.items():
        currencies = {str(row.get("currency")) for row in date_rows if row.get("currency")}
        total_worth = sum_numeric(date_rows, "total_worth")
        cash = sum_numeric(date_rows, "cash")
        equities = sum_numeric(date_rows, "equities")
        initially_invested = sum_numeric(date_rows, "initially_invested")
        realized_pl = sum_numeric(date_rows, "realized_pl")
        capital_gain = total_worth - initially_invested if initially_invested else None
        combined.append(
            normalize_history_row(
                {
                    "timestamp": date_key,
                    "date": date_key,
                    "portfolio_key": "combined:live",
                    "portfolio_name": "Portfolio Combined Total Worth",
                    "currency": next(iter(currencies)) if len(currencies) == 1 else "MIXED",
                    "total_worth": total_worth,
                    "equities": equities,
                    "cash": cash,
                    "initially_invested": initially_invested,
                    "capital_gain": capital_gain,
                    "realized_pl": realized_pl,
                    "simple_return_pct": pct(capital_gain, initially_invested),
                    "source": "snaptrade_balance_history_combined",
                    "granularity": "daily",
                }
            )
        )
    return combined


def build_monte_carlo_by_portfolio(
    portfolios: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    *,
    path_count: int = 1000,
    horizon_days: int = 252,
    sample_path_count: int = 80,
) -> dict[str, Any]:
    return {
        str(portfolio.get("key")): build_portfolio_monte_carlo(
            portfolio,
            history_rows,
            path_count=path_count,
            horizon_days=horizon_days,
            sample_path_count=sample_path_count,
        )
        for portfolio in portfolios
        if portfolio.get("key")
    }


def build_risk_by_portfolio(portfolios: list[dict[str, Any]], history_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        str(portfolio.get("key")): build_portfolio_risk(portfolio, history_rows)
        for portfolio in portfolios
        if portfolio.get("key")
    }


def build_portfolio_risk(portfolio: dict[str, Any], history_rows: list[dict[str, Any]]) -> dict[str, Any]:
    key = str(portfolio.get("key") or "")
    currency = str(portfolio.get("currency") or "EUR")
    portfolio_value = to_float(portfolio.get("total_worth")) or 0.0
    points = historical_points_for_portfolio(key, history_rows)
    values = [point["value"] for point in points]
    returns = simple_returns(values)
    source = "portfolio_history"
    if len(returns) >= 20 and np.nanmax(np.abs(returns)) <= 0.20:
        daily_returns = returns
    else:
        annual_return = (to_float(portfolio.get("simple_return_pct")) or 0.0) / 100.0
        daily_mean = annual_return / 252.0
        daily_volatility = max(0.012, abs(annual_return) / np.sqrt(252.0), 0.0005)
        daily_returns = np.array(
            [
                daily_mean - 2.32635 * daily_volatility,
                daily_mean - 1.64485 * daily_volatility,
                daily_mean,
                daily_mean + 1.64485 * daily_volatility,
                daily_mean + 2.32635 * daily_volatility,
            ],
            dtype=float,
        )
        source = "fallback_current_return_proxy_short_or_noisy_history"
    volatility = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
    mean_return = float(np.mean(daily_returns)) if len(daily_returns) else 0.0
    annualized_volatility_pct = volatility * np.sqrt(252.0) * 100.0
    annualized_return_pct = ((1.0 + mean_return) ** 252 - 1.0) * 100.0 if mean_return > -1 else -100.0
    sharpe_ratio = (annualized_return_pct / 100.0) / (annualized_volatility_pct / 100.0) if annualized_volatility_pct else 0.0
    drawdown = max_drawdown(points)
    return {
        "portfolio_key": key,
        "portfolio_name": portfolio.get("name"),
        "currency": currency,
        "portfolio_value": round(portfolio_value, 2),
        "var_95_pct": round(float(np.percentile(daily_returns, 5)) * 100.0, 2) if len(daily_returns) else 0.0,
        "var_99_pct": round(float(np.percentile(daily_returns, 1)) * 100.0, 2) if len(daily_returns) else 0.0,
        "volatility_pct": round(annualized_volatility_pct, 2),
        "expected_return_pct": round(annualized_return_pct, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "max_drawdown_pct": round(drawdown["max_drawdown_pct"], 2),
        "drawdown_peak_date": drawdown.get("peak_date"),
        "drawdown_trough_date": drawdown.get("trough_date"),
        "drawdown_recovered_date": drawdown.get("recovered_date"),
        "source": source,
        "history_point_count": len(points),
    }


def build_frontier_by_portfolio(portfolios: list[dict[str, Any]], holdings: list[dict[str, Any]]) -> dict[str, Any]:
    included_keys = {str(item.get("key")) for item in portfolios if item.get("include_in_combined") and item.get("kind") != "combined"}
    return {
        str(portfolio.get("key")): build_portfolio_frontier(
            portfolio,
            holdings_for_portfolio(portfolio, holdings, included_keys),
        )
        for portfolio in portfolios
        if portfolio.get("key")
    }


def build_scenarios_by_portfolio(
    portfolios: list[dict[str, Any]],
    holdings: list[dict[str, Any]],
    scenario_history: dict[str, Any],
) -> dict[str, Any]:
    included_keys = {str(item.get("key")) for item in portfolios if item.get("include_in_combined") and item.get("kind") != "combined"}
    return {
        str(portfolio.get("key")): build_portfolio_scenarios(
            portfolio,
            holdings_for_portfolio(portfolio, holdings, included_keys),
            scenario_history,
        )
        for portfolio in portfolios
        if portfolio.get("key")
    }


def build_benchmark_by_portfolio(
    portfolios: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    benchmark_history: dict[str, Any],
) -> dict[str, Any]:
    return {
        str(portfolio.get("key")): build_portfolio_benchmark(portfolio, history_rows, benchmark_history)
        for portfolio in portfolios
        if portfolio.get("key")
    }


def build_correlation_by_portfolio(
    portfolios: list[dict[str, Any]],
    holdings: list[dict[str, Any]],
    asset_price_history: dict[str, Any],
) -> dict[str, Any]:
    included_keys = {str(item.get("key")) for item in portfolios if item.get("include_in_combined") and item.get("kind") != "combined"}
    return {
        str(portfolio.get("key")): build_portfolio_correlation(
            portfolio,
            holdings_for_portfolio(portfolio, holdings, included_keys),
            asset_price_history,
        )
        for portfolio in portfolios
        if portfolio.get("key")
    }


def build_trade_alpha_by_portfolio(
    portfolios: list[dict[str, Any]],
    holdings: list[dict[str, Any]],
    benchmark_history: dict[str, Any],
) -> dict[str, Any]:
    included_keys = {str(item.get("key")) for item in portfolios if item.get("include_in_combined") and item.get("kind") != "combined"}
    return {
        str(portfolio.get("key")): build_portfolio_trade_alpha(
            portfolio,
            holdings_for_portfolio(portfolio, holdings, included_keys),
            benchmark_history,
        )
        for portfolio in portfolios
        if portfolio.get("key")
    }


def build_portfolio_trade_alpha(
    portfolio: dict[str, Any],
    holdings: list[dict[str, Any]],
    benchmark_history: dict[str, Any],
) -> dict[str, Any]:
    key = str(portfolio.get("key") or "")
    currency = str(portfolio.get("currency") or "EUR")
    symbol = str(benchmark_history.get("symbol") or DEFAULT_BENCHMARK_SYMBOL)
    name = str(benchmark_history.get("name") or DEFAULT_BENCHMARK_NAME)
    benchmark_rows = sorted(
        [
            {"date": str(row.get("date") or "")[:10], "close": float(to_float(row.get("close")) or 0.0)}
            for row in (benchmark_history.get("prices") or [])
            if isinstance(row, dict) and to_float(row.get("close")) is not None and (to_float(row.get("close")) or 0) > 0
        ],
        key=lambda row: row["date"],
    )
    benchmark_end = benchmark_rows[-1] if benchmark_rows else None
    rows = []
    total_cost_basis = 0.0
    total_current_value = 0.0
    total_your_return = 0.0
    total_benchmark_return = 0.0
    covered_cost_basis = 0.0
    beat_count = 0
    below_count = 0
    for holding in sorted(holdings, key=lambda row: to_float(row.get("current_value")) or 0.0, reverse=True):
        cost_basis = to_float(holding.get("cost_basis")) or 0.0
        current_value = to_float(holding.get("current_value")) or 0.0
        if cost_basis <= 0 and current_value <= 0:
            continue
        buy_date = first_holding_buy_date(holding)
        benchmark_start = benchmark_price_on_or_after(benchmark_rows, buy_date) if buy_date else None
        your_return_value = current_value - cost_basis
        your_return_pct = (your_return_value / cost_basis * 100.0) if cost_basis else 0.0
        benchmark_return_pct = None
        benchmark_return_value = None
        alpha_pct = None
        alpha_value = None
        row_status = "missing buy date" if not buy_date else "missing benchmark"
        if benchmark_start and benchmark_end and benchmark_start["close"] > 0:
            benchmark_return_pct = (benchmark_end["close"] / benchmark_start["close"] - 1.0) * 100.0
            benchmark_return_value = cost_basis * benchmark_return_pct / 100.0
            alpha_pct = your_return_pct - benchmark_return_pct
            alpha_value = your_return_value - benchmark_return_value
            row_status = "beat" if alpha_value >= 0 else "below"
            total_benchmark_return += benchmark_return_value
            covered_cost_basis += cost_basis
            if alpha_value >= 0:
                beat_count += 1
            else:
                below_count += 1
        total_cost_basis += cost_basis
        total_current_value += current_value
        total_your_return += your_return_value
        rows.append(
            {
                "name": holding.get("name") or holding.get("ticker") or holding.get("broker_symbol") or "Holding",
                "ticker": holding.get("ticker") or holding.get("broker_symbol") or "",
                "account_key": holding.get("account_key"),
                "account_name": holding.get("account_name"),
                "cost_basis": round(cost_basis, 2),
                "current_value": round(current_value, 2),
                "your_return_value": round(your_return_value, 2),
                "your_return_pct": round(your_return_pct, 2),
                "benchmark_return_value": round(benchmark_return_value, 2) if benchmark_return_value is not None else None,
                "benchmark_return_pct": round(benchmark_return_pct, 2) if benchmark_return_pct is not None else None,
                "alpha_value": round(alpha_value, 2) if alpha_value is not None else None,
                "alpha_pct": round(alpha_pct, 2) if alpha_pct is not None else None,
                "buy_date": buy_date,
                "buy_date_source": holding.get("buy_date_source") or "missing",
                "status": row_status,
            }
        )
    total_alpha = total_your_return - total_benchmark_return
    total_alpha_pct = (total_alpha / covered_cost_basis * 100.0) if covered_cost_basis else 0.0
    return {
        "portfolio_key": key,
        "portfolio_name": portfolio.get("name"),
        "currency": currency,
        "benchmark_symbol": symbol,
        "benchmark_name": name,
        "source": benchmark_history.get("source") or "benchmark_history_cache",
        "fetched_at": benchmark_history.get("fetched_at"),
        "status": "ready" if covered_cost_basis > 0 else "missing_benchmark_or_buy_dates",
        "cost_basis": round(total_cost_basis, 2),
        "current_value": round(total_current_value, 2),
        "covered_cost_basis": round(covered_cost_basis, 2),
        "your_return_value": round(total_your_return, 2),
        "your_return_pct": round((total_your_return / total_cost_basis * 100.0) if total_cost_basis else 0.0, 2),
        "benchmark_return_value": round(total_benchmark_return, 2),
        "benchmark_return_pct": round((total_benchmark_return / covered_cost_basis * 100.0) if covered_cost_basis else 0.0, 2),
        "alpha_value": round(total_alpha, 2),
        "alpha_pct": round(total_alpha_pct, 2),
        "beat_count": beat_count,
        "below_count": below_count,
        "holding_count": len(rows),
        "benchmark_start_date": benchmark_rows[0]["date"] if benchmark_rows else None,
        "benchmark_end_date": benchmark_end["date"] if benchmark_end else None,
        "rows": rows,
        "note": "Trade Alpha compares each open holding with what the same cost basis would have earned in VOO from that holding's buy date to the latest cached benchmark date.",
    }


def benchmark_price_on_or_after(rows: list[dict[str, Any]], date_value: str | None) -> dict[str, Any] | None:
    if not date_value:
        return None
    date_key = str(date_value)[:10]
    for row in rows:
        if str(row.get("date") or "") >= date_key:
            return row
    return None


def first_holding_buy_date(holding: dict[str, Any]) -> str | None:
    direct = holding.get("first_buy_date") or holding.get("buy_date")
    if direct:
        return str(direct)[:10]
    raw = holding.get("raw") if isinstance(holding.get("raw"), dict) else {}
    for key in ("trade_date", "settlement_date", "date", "time_updated"):
        if raw.get(key):
            return str(raw[key])[:10]
    return None


def build_portfolio_correlation(
    portfolio: dict[str, Any],
    holdings: list[dict[str, Any]],
    asset_price_history: dict[str, Any],
    *,
    max_assets: int = 18,
    min_periods: int = 20,
) -> dict[str, Any]:
    key = str(portfolio.get("key") or "")
    currency = str(portfolio.get("currency") or "EUR")
    candidate_holdings = [
        holding
        for holding in sorted(holdings, key=lambda row: to_float(row.get("current_value")) or 0.0, reverse=True)
        if (to_float(holding.get("current_value")) or 0.0) > 0
    ]
    symbol_holdings: dict[str, dict[str, Any]] = {}
    for holding in candidate_holdings:
        symbol = scenario_symbol(holding)
        if not symbol or symbol.upper() in {"CASH", "USD", "EUR"}:
            continue
        if symbol not in symbol_holdings:
            symbol_holdings[symbol] = holding
    selected_symbols = list(symbol_holdings)[:max_assets]
    prices_by_symbol = asset_price_history.get("prices") if isinstance(asset_price_history.get("prices"), dict) else {}
    returns_by_symbol: dict[str, dict[str, float]] = {}
    for symbol in selected_symbols:
        returns = price_returns_by_date(prices_by_symbol.get(symbol) or [])
        if returns:
            returns_by_symbol[symbol] = returns
    used_symbols = [symbol for symbol in selected_symbols if symbol in returns_by_symbol]
    if len(used_symbols) < 2:
        return {
            "portfolio_key": key,
            "portfolio_name": portfolio.get("name"),
            "currency": currency,
            "source": asset_price_history.get("source") or "missing_price_cache",
            "fetched_at": asset_price_history.get("fetched_at"),
            "status": "insufficient_price_history",
            "asset_count": len(selected_symbols),
            "used_asset_count": len(used_symbols),
            "min_periods": min_periods,
            "symbols": [
                correlation_symbol_payload(symbol, symbol_holdings[symbol])
                for symbol in selected_symbols
            ],
            "matrix": [],
            "pairs": [],
            "note": "Correlation needs at least two non-cash holdings with cached daily price history.",
        }
    common_dates = sorted(set.intersection(*(set(returns_by_symbol[symbol]) for symbol in used_symbols)))
    if len(common_dates) < min_periods:
        return {
            "portfolio_key": key,
            "portfolio_name": portfolio.get("name"),
            "currency": currency,
            "source": asset_price_history.get("source") or "asset_price_history_cache",
            "fetched_at": asset_price_history.get("fetched_at"),
            "status": "limited_overlap",
            "asset_count": len(selected_symbols),
            "used_asset_count": len(used_symbols),
            "overlap_days": len(common_dates),
            "min_periods": min_periods,
            "symbols": [correlation_symbol_payload(symbol, symbol_holdings[symbol]) for symbol in used_symbols],
            "matrix": [],
            "pairs": [],
            "note": f"Only {len(common_dates)} overlapping return days are available; at least {min_periods} are required for a stable matrix.",
        }
    data = np.array(
        [[returns_by_symbol[symbol][date] for date in common_dates] for symbol in used_symbols],
        dtype=float,
    )
    matrix_values = np.corrcoef(data)
    matrix_values = np.nan_to_num(matrix_values, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(matrix_values, 1.0)
    matrix = [[round(float(value), 4) for value in row] for row in matrix_values]
    pairs = []
    for i, symbol_a in enumerate(used_symbols):
        for j, symbol_b in enumerate(used_symbols):
            if j <= i:
                continue
            value = float(matrix_values[i, j])
            pairs.append(
                {
                    "asset_a": symbol_a,
                    "asset_b": symbol_b,
                    "name_a": symbol_holdings[symbol_a].get("name") or symbol_a,
                    "name_b": symbol_holdings[symbol_b].get("name") or symbol_b,
                    "correlation": round(value, 4),
                    "relationship": correlation_relationship(value),
                }
            )
    pairs.sort(key=lambda row: abs(float(row["correlation"])), reverse=True)
    return {
        "portfolio_key": key,
        "portfolio_name": portfolio.get("name"),
        "currency": currency,
        "source": asset_price_history.get("source") or "asset_price_history_cache",
        "fetched_at": asset_price_history.get("fetched_at"),
        "status": "ready" if len(common_dates) >= 60 else "limited_history",
        "asset_count": len(selected_symbols),
        "used_asset_count": len(used_symbols),
        "overlap_days": len(common_dates),
        "start_date": common_dates[0],
        "end_date": common_dates[-1],
        "min_periods": min_periods,
        "symbols": [correlation_symbol_payload(symbol, symbol_holdings[symbol]) for symbol in used_symbols],
        "matrix": matrix,
        "pairs": pairs,
        "note": "Correlation uses daily close-to-close returns from cached Yahoo Finance prices. Cash is excluded because it has no market price series.",
    }


def correlation_symbol_payload(symbol: str, holding: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "name": holding.get("name") or symbol,
        "value": round(to_float(holding.get("current_value")) or 0.0, 2),
        "currency": holding.get("currency"),
        "asset_class": (holding.get("classification") or {}).get("asset_class") or holding.get("asset_type"),
        "sector": (holding.get("classification") or {}).get("sector"),
        "geography": (holding.get("classification") or {}).get("geography"),
    }


def price_returns_by_date(rows: list[dict[str, Any]]) -> dict[str, float]:
    closes = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        date = str(row.get("date") or "")[:10]
        close = to_float(row.get("close"))
        if date and close is not None and close > 0:
            closes.append((date, float(close)))
    closes.sort(key=lambda item: item[0])
    returns: dict[str, float] = {}
    previous: float | None = None
    for date, close in closes:
        if previous is not None and previous > 0:
            returns[date] = close / previous - 1.0
        previous = close
    return returns


def correlation_relationship(value: float) -> str:
    if value >= 0.7:
        return "high positive"
    if value >= 0.3:
        return "moderate positive"
    if value <= -0.7:
        return "high negative"
    if value <= -0.3:
        return "moderate negative"
    return "low or no relationship"


def build_portfolio_benchmark(
    portfolio: dict[str, Any],
    history_rows: list[dict[str, Any]],
    benchmark_history: dict[str, Any],
) -> dict[str, Any]:
    key = str(portfolio.get("key") or "")
    currency = str(portfolio.get("currency") or "EUR")
    symbol = str(benchmark_history.get("symbol") or DEFAULT_BENCHMARK_SYMBOL)
    name = str(benchmark_history.get("name") or DEFAULT_BENCHMARK_NAME)
    benchmark_prices = benchmark_price_by_date(benchmark_history)
    portfolio_points = historical_points_for_portfolio(key, history_rows)
    aligned = []
    for point in portfolio_points:
        price = benchmark_prices.get(str(point.get("date")))
        value = to_float(point.get("value"))
        if value is None or price is None or value <= 0 or price <= 0:
            continue
        aligned.append({"date": str(point["date"]), "portfolio_value": float(value), "benchmark_close": float(price)})
    if len(aligned) < 2:
        return {
            "portfolio_key": key,
            "portfolio_name": portfolio.get("name"),
            "currency": currency,
            "benchmark_symbol": symbol,
            "benchmark_name": name,
            "source": benchmark_history.get("source") or "missing_benchmark_cache",
            "fetched_at": benchmark_history.get("fetched_at"),
            "status": "insufficient_history",
            "history_point_count": len(portfolio_points),
            "aligned_point_count": len(aligned),
            "series": [],
            "note": "Benchmark comparison needs at least two dates where both the portfolio and benchmark have values.",
        }
    start_portfolio = aligned[0]["portfolio_value"]
    start_benchmark = aligned[0]["benchmark_close"]
    series = []
    for row in aligned:
        portfolio_return_pct = (row["portfolio_value"] / start_portfolio - 1.0) * 100.0 if start_portfolio else 0.0
        benchmark_return_pct = (row["benchmark_close"] / start_benchmark - 1.0) * 100.0 if start_benchmark else 0.0
        series.append(
            {
                "date": row["date"],
                "portfolio_value": round(row["portfolio_value"], 2),
                "benchmark_close": round(row["benchmark_close"], 4),
                "portfolio_return_pct": round(portfolio_return_pct, 4),
                "benchmark_return_pct": round(benchmark_return_pct, 4),
            }
        )
    portfolio_values = [row["portfolio_value"] for row in aligned]
    benchmark_values = [row["benchmark_close"] for row in aligned]
    portfolio_returns = simple_returns(portfolio_values)
    benchmark_returns = simple_returns(benchmark_values)
    common_length = min(len(portfolio_returns), len(benchmark_returns))
    portfolio_returns = portfolio_returns[-common_length:] if common_length else np.array([], dtype=float)
    benchmark_returns = benchmark_returns[-common_length:] if common_length else np.array([], dtype=float)
    portfolio_stats = return_stats(portfolio_returns, portfolio_values[0], portfolio_values[-1])
    benchmark_stats = return_stats(benchmark_returns, benchmark_values[0], benchmark_values[-1])
    advanced = benchmark_advanced_stats(portfolio_returns, benchmark_returns)
    excess_return_pct = portfolio_stats["total_return_pct"] - benchmark_stats["total_return_pct"]
    return {
        "portfolio_key": key,
        "portfolio_name": portfolio.get("name"),
        "currency": currency,
        "benchmark_symbol": symbol,
        "benchmark_name": name,
        "source": benchmark_history.get("source") or "benchmark_history_cache",
        "fetched_at": benchmark_history.get("fetched_at"),
        "status": "ready" if len(aligned) >= 20 else "limited_history",
        "history_point_count": len(portfolio_points),
        "aligned_point_count": len(aligned),
        "start_date": aligned[0]["date"],
        "end_date": aligned[-1]["date"],
        "metrics": {
            "total_return_pct": portfolio_stats["total_return_pct"],
            "annual_return_pct": portfolio_stats["annual_return_pct"],
            "volatility_pct": portfolio_stats["volatility_pct"],
            "sharpe_ratio": portfolio_stats["sharpe_ratio"],
            "sortino_ratio": portfolio_stats["sortino_ratio"],
            "benchmark_total_return_pct": benchmark_stats["total_return_pct"],
            "benchmark_annual_return_pct": benchmark_stats["annual_return_pct"],
            "benchmark_volatility_pct": benchmark_stats["volatility_pct"],
            "benchmark_sharpe_ratio": benchmark_stats["sharpe_ratio"],
            "benchmark_sortino_ratio": benchmark_stats["sortino_ratio"],
            "excess_return_pct": round(excess_return_pct, 2),
            **advanced,
        },
        "series": series,
        "note": "Benchmark metrics compare daily portfolio history against cached Yahoo Finance adjusted close prices for the benchmark.",
    }


def benchmark_price_by_date(benchmark_history: dict[str, Any]) -> dict[str, float]:
    rows = benchmark_history.get("prices") if isinstance(benchmark_history, dict) else None
    output: dict[str, float] = {}
    if not isinstance(rows, list):
        return output
    for row in rows:
        if not isinstance(row, dict):
            continue
        date_key = str(row.get("date") or "")[:10]
        close = to_float(row.get("close"))
        if date_key and close is not None and close > 0:
            output[date_key] = float(close)
    return output


def return_stats(returns: np.ndarray, start_value: float, end_value: float) -> dict[str, float]:
    total_return = end_value / start_value - 1.0 if start_value else 0.0
    periods = max(1, len(returns))
    annual_return = (1.0 + total_return) ** (252.0 / periods) - 1.0 if total_return > -1 else -1.0
    volatility = float(np.std(returns, ddof=1)) * np.sqrt(252.0) if len(returns) > 1 else 0.0
    downside = returns[returns < 0]
    downside_volatility = float(np.std(downside, ddof=1)) * np.sqrt(252.0) if len(downside) > 1 else 0.0
    return {
        "total_return_pct": round(total_return * 100.0, 2),
        "annual_return_pct": round(annual_return * 100.0, 2),
        "volatility_pct": round(volatility * 100.0, 2),
        "sharpe_ratio": round(annual_return / volatility, 2) if volatility else 0.0,
        "sortino_ratio": round(annual_return / downside_volatility, 2) if downside_volatility else 0.0,
    }


def benchmark_advanced_stats(portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> dict[str, float]:
    if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
        return {
            "alpha_pct": 0.0,
            "beta": 0.0,
            "correlation": 0.0,
            "r_squared_pct": 0.0,
            "tracking_error_pct": 0.0,
            "information_ratio": 0.0,
            "treynor_ratio": 0.0,
        }
    benchmark_variance = float(np.var(benchmark_returns, ddof=1))
    covariance = float(np.cov(portfolio_returns, benchmark_returns, ddof=1)[0, 1])
    beta = covariance / benchmark_variance if benchmark_variance else 0.0
    correlation = float(np.corrcoef(portfolio_returns, benchmark_returns)[0, 1])
    correlation = correlation if np.isfinite(correlation) else 0.0
    portfolio_annual = float(np.mean(portfolio_returns)) * 252.0
    benchmark_annual = float(np.mean(benchmark_returns)) * 252.0
    alpha = portfolio_annual - beta * benchmark_annual
    active_returns = portfolio_returns - benchmark_returns
    tracking_error = float(np.std(active_returns, ddof=1)) * np.sqrt(252.0)
    information_ratio = (portfolio_annual - benchmark_annual) / tracking_error if tracking_error else 0.0
    treynor = portfolio_annual / beta if beta else 0.0
    return {
        "alpha_pct": round(alpha * 100.0, 2),
        "beta": round(beta, 2),
        "correlation": round(correlation, 2),
        "r_squared_pct": round(correlation * correlation * 100.0, 2),
        "tracking_error_pct": round(tracking_error * 100.0, 2),
        "information_ratio": round(information_ratio, 2),
        "treynor_ratio": round(treynor, 2),
    }


def build_portfolio_scenarios(portfolio: dict[str, Any], holdings: list[dict[str, Any]], scenario_history: dict[str, Any]) -> dict[str, Any]:
    key = str(portfolio.get("key") or "")
    currency = str(portfolio.get("currency") or "EUR")
    portfolio_value = to_float(portfolio.get("total_worth")) or sum(to_float(holding.get("current_value")) or 0.0 for holding in holdings)
    scenario_rows = []
    for scenario in SCENARIO_DEFINITIONS:
        price_rows = ((scenario_history.get("scenarios") or {}).get(scenario["key"]) or {}).get("prices") or {}
        weighted_return = 0.0
        covered_value = 0.0
        contributors = []
        for holding in holdings:
            value = to_float(holding.get("current_value")) or 0.0
            if value <= 0:
                continue
            symbol = scenario_symbol(holding)
            price_row = price_rows.get(symbol) if symbol else None
            if not isinstance(price_row, dict) or to_float(price_row.get("return_pct")) is None:
                continue
            return_pct = float(to_float(price_row.get("return_pct")) or 0.0)
            weight = value / portfolio_value if portfolio_value else 0.0
            weighted_return += weight * return_pct
            covered_value += value
            contributors.append(
                {
                    "name": holding.get("name") or symbol,
                    "ticker": symbol,
                    "value": round(value, 2),
                    "event_return_pct": round(return_pct, 2),
                    "portfolio_impact": round(value * return_pct / 100.0, 2),
                }
            )
        coverage_pct = pct(covered_value, portfolio_value) or 0.0
        scenario_rows.append(
            {
                **scenario,
                "coverage_pct": round(coverage_pct, 2),
                "estimated_return_pct": round(weighted_return, 2),
                "estimated_value_change": round(portfolio_value * weighted_return / 100.0, 2),
                "estimated_end_value": round(portfolio_value * (1.0 + weighted_return / 100.0), 2),
                "available_asset_count": len(contributors),
                "holding_count": len([holding for holding in holdings if (to_float(holding.get("current_value")) or 0.0) > 0]),
                "status": "ready" if coverage_pct >= 50 else "partial" if contributors else "missing",
                "top_contributors": sorted(contributors, key=lambda row: abs(float(row["portfolio_impact"])), reverse=True)[:5],
            }
        )
    return {
        "portfolio_key": key,
        "portfolio_name": portfolio.get("name"),
        "currency": currency,
        "portfolio_value": round(portfolio_value, 2),
        "source": scenario_history.get("source") or "scenario_history_cache",
        "fetched_at": scenario_history.get("fetched_at"),
        "scenarios": scenario_rows,
    }


def scenario_symbol(holding: dict[str, Any]) -> str:
    return str(holding.get("ticker") or holding.get("broker_symbol") or "").strip()


def read_trade_republic_buy_dates(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter=";"))
    except OSError:
        return {}
    dates: dict[str, str] = {}
    for row in rows:
        if str(row.get("Type") or "").strip().lower() != "buy":
            continue
        isin = str(row.get("ISIN") or "").strip().upper()
        date = str(row.get("Date") or "")[:10]
        if not isin or not date:
            continue
        if isin not in dates or date < dates[isin]:
            dates[isin] = date
    return dates


def snaptrade_buy_dates_by_symbol(account_row: dict[str, Any]) -> dict[str, str]:
    activities = account_row.get("activities") if isinstance(account_row.get("activities"), dict) else {}
    rows = activities.get("data") if isinstance(activities.get("data"), list) else []
    dates: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("type") or "").strip().upper() != "BUY":
            continue
        symbol = snaptrade_activity_symbol(row)
        date = str(row.get("trade_date") or row.get("settlement_date") or "")[:10]
        if not symbol or not date:
            continue
        if symbol not in dates or date < dates[symbol]:
            dates[symbol] = date
    return dates


def snaptrade_activity_symbol(row: dict[str, Any]) -> str:
    symbol = row.get("symbol") if isinstance(row.get("symbol"), dict) else {}
    option_symbol = row.get("option_symbol") if isinstance(row.get("option_symbol"), dict) else {}
    raw = (
        first_value(symbol, "symbol", "raw_symbol")
        or first_value(option_symbol, "underlying_symbol.symbol", "underlying_symbol.raw_symbol", "ticker")
        or row.get("symbol")
    )
    return str(raw or "").strip().upper()


def read_scenario_history(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"source": "missing_cache", "scenarios": {}}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"source": "unreadable_cache", "error": str(exc), "scenarios": {}}
    return parsed if isinstance(parsed, dict) else {"source": "invalid_cache", "scenarios": {}}


def read_benchmark_history(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "source": "missing_cache",
            "symbol": DEFAULT_BENCHMARK_SYMBOL,
            "name": DEFAULT_BENCHMARK_NAME,
            "prices": [],
        }
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "source": "unreadable_cache",
            "error": str(exc),
            "symbol": DEFAULT_BENCHMARK_SYMBOL,
            "name": DEFAULT_BENCHMARK_NAME,
            "prices": [],
        }
    if not isinstance(parsed, dict):
        return {
            "source": "invalid_cache",
            "symbol": DEFAULT_BENCHMARK_SYMBOL,
            "name": DEFAULT_BENCHMARK_NAME,
            "prices": [],
        }
    parsed.setdefault("symbol", DEFAULT_BENCHMARK_SYMBOL)
    parsed.setdefault("name", DEFAULT_BENCHMARK_NAME)
    parsed.setdefault("prices", [])
    return parsed


def read_asset_price_history(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"source": "missing_cache", "prices": {}}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"source": "unreadable_cache", "error": str(exc), "prices": {}}
    if not isinstance(parsed, dict):
        return {"source": "invalid_cache", "prices": {}}
    parsed.setdefault("prices", {})
    return parsed


def read_classification_cache(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"source": "missing_cache", "classifications": {}}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"source": "unreadable_cache", "error": str(exc), "classifications": {}}
    if not isinstance(parsed, dict):
        return {"source": "invalid_cache", "classifications": {}}
    parsed.setdefault("classifications", {})
    return parsed


def attach_holding_classifications(holdings: list[dict[str, Any]], cache: dict[str, Any]) -> None:
    classifications = cache.get("classifications") if isinstance(cache, dict) else {}
    classifications = classifications if isinstance(classifications, dict) else {}
    for holding in holdings:
        key = classification_key(holding)
        cached = classifications.get(key)
        if isinstance(cached, dict) and valid_classification(cached):
            holding["classification"] = normalized_classification(cached, source=str(cached.get("source") or "cache"))
        else:
            holding["classification"] = heuristic_classification(holding)


def refresh_asset_classification_cache(
    holdings: list[dict[str, Any]],
    path: Path = DEFAULT_CLASSIFICATION_CACHE,
    *,
    model: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    cache = read_classification_cache(path)
    existing = cache.get("classifications") if isinstance(cache.get("classifications"), dict) else {}
    llm_available = bool(os.environ.get("OPENAI_API_KEY"))
    candidates = []
    seen: set[str] = set()
    for holding in holdings:
        key = classification_key(holding)
        if not key or key in seen:
            continue
        seen.add(key)
        cached = existing.get(key)
        if not force and isinstance(cached, dict) and valid_classification(cached):
            continue
        heuristic = heuristic_classification(holding)
        if not llm_available and not force and heuristic.get("confidence") == "high" and heuristic.get("sector") not in {"Other"}:
            existing[key] = heuristic
            continue
        candidates.append(holding)
    llm_rows: dict[str, dict[str, Any]] = {}
    if candidates:
        llm_rows = classify_holdings_with_llm(candidates, model=model)
    for holding in candidates:
        key = classification_key(holding)
        classification = llm_rows.get(key) or heuristic_classification(holding)
        existing[key] = normalized_classification(classification, source=str(classification.get("source") or "llm_or_heuristic"))
    payload = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "source": "openai_llm_with_heuristic_fallback",
        "asset_class_enum": ASSET_CLASS_ENUM,
        "sector_enum": SECTOR_ENUM,
        "geography_enum": GEOGRAPHY_ENUM,
        "classifications": existing,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def classify_holdings_with_llm(holdings: list[dict[str, Any]], *, model: str | None = None) -> dict[str, dict[str, Any]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {}
    try:
        from openai import OpenAI
    except ImportError:
        return {}
    rows = [
        {
            "key": classification_key(holding),
            "name": holding.get("name"),
            "ticker": holding.get("ticker") or holding.get("broker_symbol"),
            "isin": holding.get("isin"),
            "asset_type": holding.get("asset_type"),
            "currency": holding.get("currency"),
            "exchange": first_value(holding.get("raw") or {}, "instrument.exchange", "exchange"),
        }
        for holding in holdings
    ]
    prompt = {
        "task": "Classify each portfolio holding into fixed broad enums for diversification analysis.",
        "rules": [
            "Return JSON only with key 'classifications', an array with one object per input key.",
            "Every classification object must include key, asset_class, sector, geography, confidence, and reason.",
            "Use exactly one value from each enum.",
            "Use broad, stable categories; do not invent new categories.",
            "For geography, classify by issuer domicile or ETF target region, not by where the company sells products.",
            "For broad global ETFs, use sector 'Broad Market ETF' and geography 'Global' unless the name clearly states a region.",
            "For defense/aerospace companies or defense ETFs, prefer sector 'Defense & Aerospace'.",
            "For mining, metals, rare earths, uranium, antimony, or commodity producers, prefer sector 'Energy & Mining' or 'Materials'.",
            "For ambiguous assets, choose 'Other' with low confidence.",
        ],
        "asset_class_enum": ASSET_CLASS_ENUM,
        "sector_enum": SECTOR_ENUM,
        "geography_enum": GEOGRAPHY_ENUM,
        "holdings": rows,
    }
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model or os.environ.get("OPENAI_CLASSIFICATION_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You classify financial assets for a portfolio dashboard. Follow the provided enums exactly. Every object must include key, asset_class, sector, geography, confidence, and reason.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
        ],
    )
    content = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {}
    output: dict[str, dict[str, Any]] = {}
    for item in parsed.get("classifications") or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "")
        classification = normalized_classification({**item, "source": "openai_llm"}, source="openai_llm")
        if key and valid_classification(classification):
            output[key] = classification
    return output


def classification_key(holding: dict[str, Any]) -> str:
    ticker = str(holding.get("ticker") or holding.get("broker_symbol") or "").strip().upper()
    isin = str(holding.get("isin") or "").strip().upper()
    name = str(holding.get("name") or "").strip().lower()
    if ticker:
        return f"ticker:{ticker}"
    if isin:
        return f"isin:{isin}"
    return f"name:{name}" if name else ""


def valid_classification(value: dict[str, Any]) -> bool:
    return (
        value.get("asset_class") in ASSET_CLASS_ENUM
        and value.get("sector") in SECTOR_ENUM
        and value.get("geography") in GEOGRAPHY_ENUM
    )


def normalized_classification(value: dict[str, Any], *, source: str) -> dict[str, Any]:
    asset_class = value.get("asset_class") if value.get("asset_class") in ASSET_CLASS_ENUM else "Other"
    sector = value.get("sector") if value.get("sector") in SECTOR_ENUM else "Other"
    geography = value.get("geography") if value.get("geography") in GEOGRAPHY_ENUM else "Other"
    confidence = str(value.get("confidence") or "medium").lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    return {
        "asset_class": asset_class,
        "sector": sector,
        "geography": geography,
        "confidence": confidence,
        "source": source,
        "reason": str(value.get("reason") or "")[:240],
        "classified_at": value.get("classified_at") or datetime.now(UTC).isoformat(),
    }


def heuristic_classification(holding: dict[str, Any]) -> dict[str, Any]:
    name = str(holding.get("name") or "")
    ticker = str(holding.get("ticker") or holding.get("broker_symbol") or "")
    text = f"{name} {ticker}".lower()
    asset_type = str(holding.get("asset_type") or "").lower()
    asset_class = "ETF" if "etf" in asset_type or "ucits etf" in text or "all-world" in text or "vanguard ftse" in text or "ishares" in text or "vaneck" in text else "Stock"
    if "crypto" in asset_type or ticker.upper() in {"BTC", "ETH"}:
        asset_class = "Crypto"
    if "cash" in asset_type:
        return normalized_classification(
            {"asset_class": "Cash", "sector": "Cash", "geography": "Cash", "confidence": "high", "source": "heuristic", "reason": "Cash balance."},
            source="heuristic",
        )
    sector = "Other"
    confidence = "medium"
    if any(term in text for term in ["all-world", "world", "vanguard ftse", "msci world"]):
        sector = "Broad Market ETF"
        confidence = "high"
    elif any(term in text for term in ["defence", "defense", "airbus", "lockheed", "raytheon", "rtx", "rheinmetall", "leidos", "heico"]):
        sector = "Defense & Aerospace"
        confidence = "high"
    elif any(term in text for term in ["asml", "semiconductor", "nvidia", "tsmc", "taiwan semiconductor"]):
        sector = "Semiconductors"
        confidence = "high"
    elif any(term in text for term in ["rare earth", "metals", "antimony", "niocorp", "energy fuels", "uranium", "tmc", "idaho strategic"]):
        sector = "Energy & Mining"
        confidence = "high"
    elif any(term in text for term in ["materials", "resources"]):
        sector = "Materials"
    elif any(term in text for term in ["automation", "keysight", "crowdstrike", "bigbear"]):
        sector = "Technology"
    elif any(term in text for term in ["3m", "industrial"]):
        sector = "Industrials"
    elif any(term in text for term in ["lilly", "healthcare", "pharma"]):
        sector = "Healthcare"
    geography = heuristic_geography(holding, text)
    return normalized_classification(
        {
            "asset_class": asset_class,
            "sector": sector,
            "geography": geography,
            "confidence": confidence,
            "source": "heuristic",
            "reason": "Rule-based classification from ticker, name, asset type, exchange, and currency.",
        },
        source="heuristic",
    )


def heuristic_geography(holding: dict[str, Any], text: str) -> str:
    if any(term in text for term in ["all-world", "world", "global"]):
        return "Global"
    raw = holding.get("raw") if isinstance(holding.get("raw"), dict) else {}
    exchange = str(first_value(raw, "instrument.exchange", "exchange") or "").upper()
    symbol = str(holding.get("ticker") or holding.get("broker_symbol") or "").upper()
    isin = str(holding.get("isin") or "").upper()
    currency = str(holding.get("currency") or "").upper()
    if exchange in {"XNAS", "XNYS", "XASE", "ARC"} or currency == "USD":
        return "United States"
    if any(term in text for term in ["nvidia", "crowdstrike", "keysight", "3m company"]):
        return "United States"
    if exchange == "XETR" or symbol.endswith(".DE") or isin.startswith("DE"):
        return "Germany"
    if exchange == "XPAR" or isin.startswith("FR"):
        return "France"
    if exchange == "XAMS" or isin.startswith("NL"):
        return "Netherlands"
    if exchange == "XLON" or isin.startswith("GB"):
        return "United Kingdom"
    if symbol.endswith(".TO") or isin.startswith("CA"):
        return "Canada"
    if "taiwan" in text or symbol in {"TSM", "TSMC"}:
        return "China / Taiwan"
    if currency == "EUR":
        return "Europe"
    return "Other"


def refresh_benchmark_history_cache(
    *,
    symbol: str = DEFAULT_BENCHMARK_SYMBOL,
    name: str = DEFAULT_BENCHMARK_NAME,
    path: Path = DEFAULT_BENCHMARK_HISTORY,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required to refresh benchmark historical data.") from exc
    end_date = end or (datetime.now(UTC) + timedelta(days=1)).date().isoformat()
    start_date = start or (datetime.now(UTC) - timedelta(days=730)).date().isoformat()
    try:
        frame = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:  # pragma: no cover - network/provider dependent
        raise RuntimeError(f"failed to fetch benchmark history for {symbol}: {exc}") from exc
    prices = benchmark_price_rows(frame)
    payload = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "source": "yahoo_finance_yfinance",
        "symbol": symbol,
        "name": name,
        "start": start_date,
        "end": end_date,
        "prices": prices,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def benchmark_price_rows(frame: Any) -> list[dict[str, Any]]:
    if frame is None or getattr(frame, "empty", True):
        return []
    close = frame["Close"] if "Close" in frame else None
    if close is None:
        return []
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        return []
    rows = []
    for index, value in close.items():
        parsed = to_float(value)
        if parsed is None or parsed <= 0:
            continue
        rows.append({"date": str(index.date()), "close": round(parsed, 6)})
    return rows


def refresh_asset_price_history_cache(
    holdings: list[dict[str, Any]],
    path: Path = DEFAULT_ASSET_PRICE_HISTORY,
    *,
    lookback_days: int = 420,
) -> dict[str, Any]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required to refresh asset historical prices.") from exc
    existing = read_asset_price_history(path)
    existing_prices = existing.get("prices") if isinstance(existing.get("prices"), dict) else {}
    symbols = sorted(
        {
            scenario_symbol(holding)
            for holding in holdings
            if scenario_symbol(holding)
            and (to_float(holding.get("current_value")) or 0.0) > 0
            and scenario_symbol(holding).upper() not in {"CASH", "USD", "EUR"}
        }
    )
    end_date = (datetime.now(UTC) + timedelta(days=1)).date().isoformat()
    start_date = (datetime.now(UTC) - timedelta(days=max(30, int(lookback_days)))).date().isoformat()
    prices: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, str] = {}
    for symbol in symbols:
        try:
            frame = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception as exc:  # pragma: no cover - network/provider dependent
            errors[symbol] = str(exc)
            cached = existing_prices.get(symbol)
            if isinstance(cached, list):
                prices[symbol] = cached
            continue
        rows = benchmark_price_rows(frame)
        if rows:
            prices[symbol] = rows
        else:
            cached = existing_prices.get(symbol)
            if isinstance(cached, list):
                prices[symbol] = cached
            errors[symbol] = "no_price_rows"
    payload = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "source": "yahoo_finance_yfinance",
        "start": start_date,
        "end": end_date,
        "lookback_days": int(lookback_days),
        "prices": prices,
        "errors": errors,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def refresh_scenario_history_cache(holdings: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required to refresh scenario historical data.") from exc
    symbols = sorted({scenario_symbol(holding) for holding in holdings if scenario_symbol(holding)})
    payload: dict[str, Any] = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "source": "yahoo_finance_yfinance",
        "scenarios": {scenario["key"]: {**scenario, "prices": {}} for scenario in SCENARIO_DEFINITIONS},
    }
    for symbol in symbols:
        for scenario in SCENARIO_DEFINITIONS:
            start = str(scenario["start"])
            end = str(scenario["end"])
            try:
                frame = yf.download(
                    symbol,
                    start=start,
                    end=(datetime.fromisoformat(end) + timedelta(days=5)).date().isoformat(),
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
            except Exception as exc:  # pragma: no cover - network/provider dependent
                payload["scenarios"][scenario["key"]]["prices"][symbol] = {"status": "fetch_failed", "error": str(exc)}
                continue
            price_row = scenario_price_row(frame, start=start, end=end)
            payload["scenarios"][scenario["key"]]["prices"][symbol] = price_row
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def scenario_price_row(frame: Any, *, start: str, end: str) -> dict[str, Any]:
    if frame is None or getattr(frame, "empty", True):
        return {"status": "no_data"}
    close = frame["Close"] if "Close" in frame else None
    if close is None:
        return {"status": "no_close"}
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        return {"status": "no_close"}
    window = close.loc[(close.index >= start) & (close.index <= end)]
    if window.empty:
        return {"status": "no_window_data"}
    start_close = float(window.iloc[0])
    end_close = float(window.iloc[-1])
    return {
        "status": "ok",
        "start_date": str(window.index[0].date()),
        "end_date": str(window.index[-1].date()),
        "start_close": round(start_close, 6),
        "end_close": round(end_close, 6),
        "return_pct": round((end_close / start_close - 1.0) * 100.0, 2) if start_close else None,
        "rows": int(len(window)),
    }


def holdings_for_portfolio(portfolio: dict[str, Any], holdings: list[dict[str, Any]], included_keys: set[str]) -> list[dict[str, Any]]:
    key = str(portfolio.get("key") or "")
    if portfolio.get("kind") == "combined":
        return [holding for holding in holdings if str(holding.get("account_key")) in included_keys]
    return [holding for holding in holdings if str(holding.get("account_key")) == key]


def build_portfolio_frontier(portfolio: dict[str, Any], holdings: list[dict[str, Any]]) -> dict[str, Any]:
    key = str(portfolio.get("key") or "")
    currency = str(portfolio.get("currency") or "EUR")
    assets = frontier_assets(holdings)
    portfolio_value = to_float(portfolio.get("total_worth")) or sum(asset["current_value"] for asset in assets)
    if len(assets) < 2 or portfolio_value <= 0:
        return {
            "portfolio_key": key,
            "portfolio_name": portfolio.get("name"),
            "currency": currency,
            "portfolio_value": round(portfolio_value, 2),
            "source": "current_holding_return_proxy_insufficient_assets",
            "asset_count": len(assets),
            "asset_points": assets,
            "frontier_points": [],
            "current_portfolio": None,
            "max_sharpe": None,
            "note": "Efficient frontier requires at least two priced assets in the selected portfolio.",
        }
    expected = np.array([asset["mean_return_pct"] / 100.0 for asset in assets], dtype=float)
    volatilities = np.array([max(asset["volatility_pct"] / 100.0, 0.001) for asset in assets], dtype=float)
    covariance = proxy_covariance(volatilities)
    current_weights = np.array([asset["current_value"] / portfolio_value for asset in assets], dtype=float)
    current_weights = current_weights / current_weights.sum()
    current_point = portfolio_frontier_point(current_weights, expected, covariance)
    rng_seed = int(np.frombuffer(key.encode("utf-8"), dtype=np.uint8).sum()) + len(assets) * 997
    rng = np.random.default_rng(rng_seed)
    weights = rng.dirichlet(np.ones(len(assets)), size=5000)
    portfolio_returns = weights @ expected
    portfolio_volatility = np.sqrt(np.einsum("ij,jk,ik->i", weights, covariance, weights))
    sharpe = np.divide(portfolio_returns, portfolio_volatility, out=np.zeros_like(portfolio_returns), where=portfolio_volatility > 0)
    candidates = [
        {
            "weights": weights[index],
            "mean_return_pct": float(portfolio_returns[index] * 100.0),
            "volatility_pct": float(portfolio_volatility[index] * 100.0),
            "sharpe_ratio": float(sharpe[index]),
        }
        for index in range(len(weights))
    ]
    frontier_points = efficient_frontier_points(candidates)
    best_index = int(np.argmax(sharpe)) if len(sharpe) else 0
    max_sharpe = frontier_point_payload(
        {
            "weights": weights[best_index],
            "mean_return_pct": float(portfolio_returns[best_index] * 100.0),
            "volatility_pct": float(portfolio_volatility[best_index] * 100.0),
            "sharpe_ratio": float(sharpe[best_index]),
        },
        assets,
    )
    return {
        "portfolio_key": key,
        "portfolio_name": portfolio.get("name"),
        "currency": currency,
        "portfolio_value": round(portfolio_value, 2),
        "source": "current_holding_return_proxy_assumed_correlation",
        "asset_count": len(assets),
        "asset_points": assets,
        "frontier_points": [frontier_point_payload(point, assets) for point in frontier_points],
        "current_portfolio": {
            "mean_return_pct": round(current_point["mean_return_pct"], 2),
            "volatility_pct": round(current_point["volatility_pct"], 2),
            "sharpe_ratio": round(current_point["sharpe_ratio"], 2),
            "weights": [
                {"ticker": asset["ticker"], "name": asset["name"], "weight_pct": round(float(weight) * 100.0, 2)}
                for asset, weight in zip(assets, current_weights, strict=False)
                if weight > 0.001
            ],
        },
        "max_sharpe": max_sharpe,
        "note": "Proxy frontier: expected returns use current holding return and covariance uses holding volatility with an assumed 0.35 cross-asset correlation until per-asset historical price series are wired.",
    }


def frontier_assets(holdings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for holding in holdings:
        current_value = to_float(holding.get("current_value")) or 0.0
        if current_value <= 0:
            continue
        mean_return_pct = to_float(holding.get("unrealized_pl_pct")) or 0.0
        volatility_pct = max(2.0, abs(mean_return_pct) * 1.35)
        assets.append(
            {
                "name": str(holding.get("name") or holding.get("ticker") or holding.get("broker_symbol") or "Asset"),
                "ticker": str(holding.get("ticker") or holding.get("broker_symbol") or holding.get("isin") or ""),
                "current_value": round(current_value, 2),
                "mean_return_pct": round(mean_return_pct, 2),
                "volatility_pct": round(volatility_pct, 2),
                "sharpe_ratio": round(mean_return_pct / volatility_pct if volatility_pct else 0.0, 2),
            }
        )
    return assets


def proxy_covariance(volatilities: np.ndarray, correlation: float = 0.35) -> np.ndarray:
    corr = np.full((len(volatilities), len(volatilities)), correlation, dtype=float)
    np.fill_diagonal(corr, 1.0)
    return np.outer(volatilities, volatilities) * corr


def portfolio_frontier_point(weights: np.ndarray, expected: np.ndarray, covariance: np.ndarray) -> dict[str, float]:
    mean_return = float(weights @ expected)
    volatility = float(np.sqrt(weights @ covariance @ weights))
    return {
        "mean_return_pct": mean_return * 100.0,
        "volatility_pct": volatility * 100.0,
        "sharpe_ratio": mean_return / volatility if volatility else 0.0,
    }


def efficient_frontier_points(candidates: list[dict[str, Any]], max_points: int = 30) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda point: (point["volatility_pct"], -point["mean_return_pct"]))
    frontier: list[dict[str, Any]] = []
    best_return = -1e9
    for point in ordered:
        if point["mean_return_pct"] > best_return:
            frontier.append(point)
            best_return = point["mean_return_pct"]
    if len(frontier) <= max_points:
        return frontier
    indexes = np.linspace(0, len(frontier) - 1, max_points).round().astype(int)
    return [frontier[int(index)] for index in indexes]


def frontier_point_payload(point: dict[str, Any], assets: list[dict[str, Any]]) -> dict[str, Any]:
    weights = point.get("weights")
    weight_rows = []
    if isinstance(weights, np.ndarray):
        weight_rows = [
            {"ticker": asset["ticker"], "name": asset["name"], "weight_pct": round(float(weight) * 100.0, 2)}
            for asset, weight in zip(assets, weights, strict=False)
            if weight > 0.05
        ]
    return {
        "mean_return_pct": round(float(point["mean_return_pct"]), 2),
        "volatility_pct": round(float(point["volatility_pct"]), 2),
        "sharpe_ratio": round(float(point["sharpe_ratio"]), 2),
        "top_weights": sorted(weight_rows, key=lambda row: row["weight_pct"], reverse=True)[:5],
    }


def build_portfolio_monte_carlo(
    portfolio: dict[str, Any],
    history_rows: list[dict[str, Any]],
    *,
    path_count: int = 1000,
    horizon_days: int = 252,
    sample_path_count: int = 80,
) -> dict[str, Any]:
    key = str(portfolio.get("key") or "")
    starting_value = to_float(portfolio.get("total_worth")) or 0.0
    currency = str(portfolio.get("currency") or "EUR")
    values = historical_values_for_portfolio(key, history_rows)
    returns = simple_returns(values)
    source = "portfolio_history"
    if len(returns) >= 20 and np.nanmax(np.abs(returns)) <= 0.20:
        daily_mean = float(np.mean(returns))
        daily_volatility = float(np.std(returns, ddof=1))
    else:
        annual_return = (to_float(portfolio.get("simple_return_pct")) or 0.0) / 100.0
        daily_mean = annual_return / 252.0
        daily_volatility = max(0.012, abs(annual_return) / np.sqrt(252.0), 0.0005)
        source = "fallback_current_return_proxy_short_or_noisy_history"
    daily_volatility = max(daily_volatility, 0.0001)
    seed = int(np.frombuffer(key.encode("utf-8"), dtype=np.uint8).sum()) + int(round(starting_value * 100))
    rng = np.random.default_rng(seed)
    shocks = rng.normal(loc=daily_mean, scale=daily_volatility, size=(path_count, horizon_days))
    paths = starting_value * np.cumprod(1.0 + shocks, axis=1)
    paths = np.maximum(paths, 0.0)
    final_values = paths[:, -1] if starting_value > 0 else np.zeros(path_count)
    percentiles = {
        "p5": float(np.percentile(final_values, 5)),
        "p25": float(np.percentile(final_values, 25)),
        "median": float(np.percentile(final_values, 50)),
        "p75": float(np.percentile(final_values, 75)),
        "p95": float(np.percentile(final_values, 95)),
    }
    sample_step = max(1, path_count // sample_path_count)
    sampled = paths[::sample_step][:sample_path_count]
    return {
        "portfolio_key": key,
        "portfolio_name": portfolio.get("name"),
        "currency": currency,
        "starting_value": round(starting_value, 2),
        "path_count": path_count,
        "horizon_days": horizon_days,
        "daily_mean_return": round(daily_mean, 8),
        "annualized_mean_return_pct": round(((1.0 + daily_mean) ** 252 - 1.0) * 100.0, 2),
        "annualized_volatility_pct": round(daily_volatility * np.sqrt(252.0) * 100.0, 2),
        "source": source,
        "history_point_count": len(values),
        "percentiles": {name: round(value, 2) for name, value in percentiles.items()},
        "sample_paths": [[round(float(value), 2) for value in row] for row in sampled],
    }


def historical_points_for_portfolio(portfolio_key: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if str(row.get("portfolio_key")) == portfolio_key and to_float(row.get("total_worth")) is not None
    ]
    latest_by_date: dict[str, dict[str, Any]] = {}
    for row in selected:
        date_key = str(row.get("date") or str(row.get("timestamp") or "")[:10])
        previous = latest_by_date.get(date_key)
        if previous is None or str(row.get("timestamp") or row.get("date") or "") >= str(previous.get("timestamp") or previous.get("date") or ""):
            latest_by_date[date_key] = row
    points: list[dict[str, Any]] = []
    for date_key, row in sorted(latest_by_date.items()):
        value = to_float(row.get("total_worth"))
        if value is not None and value > 0:
            points.append({"date": date_key, "value": float(value)})
    return points


def historical_values_for_portfolio(portfolio_key: str, rows: list[dict[str, Any]]) -> list[float]:
    return [point["value"] for point in historical_points_for_portfolio(portfolio_key, rows)]


def max_drawdown(points: list[dict[str, Any]]) -> dict[str, Any]:
    if not points:
        return {"max_drawdown_pct": 0.0, "peak_date": None, "trough_date": None, "recovered_date": None}
    peak_value = points[0]["value"]
    peak_date = points[0]["date"]
    worst_drawdown = 0.0
    worst_peak_date = peak_date
    trough_date = peak_date
    recovered_date = None
    recovery_target = None
    for point in points:
        value = point["value"]
        if value >= peak_value:
            peak_value = value
            peak_date = point["date"]
        drawdown_pct = (value / peak_value - 1.0) * 100.0 if peak_value else 0.0
        if drawdown_pct < worst_drawdown:
            worst_drawdown = drawdown_pct
            worst_peak_date = peak_date
            trough_date = point["date"]
            recovery_target = peak_value
            recovered_date = None
        if recovery_target is not None and recovered_date is None and point["date"] > trough_date and value >= recovery_target:
            recovered_date = point["date"]
    return {
        "max_drawdown_pct": worst_drawdown,
        "peak_date": worst_peak_date,
        "trough_date": trough_date,
        "recovered_date": recovered_date,
    }


def simple_returns(values: list[float]) -> np.ndarray:
    if len(values) < 2:
        return np.array([], dtype=float)
    previous = np.array(values[:-1], dtype=float)
    current = np.array(values[1:], dtype=float)
    valid = previous > 0
    returns = np.zeros_like(current, dtype=float)
    returns[valid] = current[valid] / previous[valid] - 1.0
    return returns[np.isfinite(returns)]


def append_history_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = {history_row_key(row) for row in read_history_rows(path)}
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            key = history_row_key(row)
            if key in existing_keys:
                continue
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True, default=str) + "\n")
            existing_keys.add(key)


def read_history_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            normalized = normalize_history_row(parsed)
            if normalized.get("portfolio_key") and normalized.get("timestamp"):
                rows.append(normalized)
    return rows


def merge_history_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    priority = {"snaptrade_balance_history": 0, "snaptrade_balance_history_combined": 0, "current_snapshot": 1}
    for row in rows:
        normalized = normalize_history_row(row)
        if not normalized.get("portfolio_key") or not normalized.get("timestamp"):
            continue
        key = history_row_key(normalized)
        previous = merged.get(key)
        if previous is None or priority.get(str(normalized.get("source")), 5) >= priority.get(str(previous.get("source")), 5):
            merged[key] = normalized
    return sorted(merged.values(), key=lambda row: (str(row.get("portfolio_key")), str(row.get("timestamp"))))


def history_row_key(row: dict[str, Any]) -> tuple[str, str]:
    timestamp = str(row.get("timestamp") or row.get("date") or "")
    if row.get("granularity") == "daily" and row.get("date"):
        timestamp = str(row.get("date"))
    return str(row.get("portfolio_key") or ""), timestamp


def normalize_history_row(row: dict[str, Any]) -> dict[str, Any]:
    total_worth = to_float(row.get("total_worth"))
    equities = to_float(row.get("equities"))
    cash = to_float(row.get("cash"))
    initially_invested = to_float(row.get("initially_invested"))
    account_invested_capital = to_float(row.get("account_invested_capital"))
    holdings_cost_basis = to_float(row.get("holdings_cost_basis"))
    unrealized_pl = to_float(row.get("unrealized_pl"))
    capital_gain = to_float(row.get("capital_gain"))
    realized_pl = to_float(row.get("realized_pl")) or 0.0
    if capital_gain is None and total_worth is not None and initially_invested is not None:
        capital_gain = total_worth - initially_invested
    timestamp = str(row.get("timestamp") or row.get("date") or "")
    return {
        "timestamp": timestamp,
        "date": str(row.get("date") or timestamp[:10]),
        "portfolio_key": row.get("portfolio_key"),
        "portfolio_name": row.get("portfolio_name"),
        "currency": row.get("currency") or "EUR",
        "total_worth": round(total_worth, 2) if total_worth is not None else None,
        "equities": round(equities, 2) if equities is not None else None,
        "cash": round(cash, 2) if cash is not None else None,
        "initially_invested": round(initially_invested, 2) if initially_invested is not None else None,
        "account_invested_capital": round(account_invested_capital, 2) if account_invested_capital is not None else None,
        "holdings_cost_basis": round(holdings_cost_basis, 2) if holdings_cost_basis is not None else None,
        "unrealized_pl": round(unrealized_pl, 2) if unrealized_pl is not None else None,
        "capital_gain": round(capital_gain, 2) if capital_gain is not None else None,
        "realized_pl": round(realized_pl, 2),
        "simple_return_pct": to_float(row.get("simple_return_pct")) if row.get("simple_return_pct") is not None else pct(capital_gain, initially_invested),
        "source": row.get("source") or "unknown",
        "granularity": row.get("granularity") or "snapshot",
    }


def normalize_portfolio_summary(item: dict[str, Any]) -> dict[str, Any]:
    equities = to_float(item.get("equities")) or 0.0
    cash = to_float(item.get("cash")) or 0.0
    initially_invested = to_float(item.get("initially_invested")) or 0.0
    account_invested_capital = to_float(item.get("account_invested_capital"))
    holdings_cost_basis = to_float(item.get("holdings_cost_basis"))
    unrealized_pl = to_float(item.get("unrealized_pl"))
    capital_gain = to_float(item.get("capital_gain"))
    if capital_gain is None:
        capital_gain = equities - initially_invested
    total_worth = equities + cash
    dividend_after_tax = to_float(item.get("dividend_after_tax")) or 0.0
    dividend_tax = to_float(item.get("dividend_tax")) or 0.0
    return {
        **item,
        "equities": round(equities, 2),
        "cash": round(cash, 2),
        "total_worth": round(total_worth, 2),
        "initially_invested": round(initially_invested, 2),
        "account_invested_capital": round(account_invested_capital, 2) if account_invested_capital is not None else round(initially_invested, 2),
        "holdings_cost_basis": round(holdings_cost_basis, 2) if holdings_cost_basis is not None else round(initially_invested, 2),
        "unrealized_pl": round(unrealized_pl, 2) if unrealized_pl is not None else round(capital_gain, 2),
        "capital_gain": round(capital_gain, 2),
        "realized_pl": round(to_float(item.get("realized_pl")) or 0.0, 2),
        "realized_pl_by_date": item.get("realized_pl_by_date") or {},
        "dividend_after_tax": round(dividend_after_tax, 2),
        "dividend_tax": round(dividend_tax, 2),
        "dividend_gross": round(dividend_after_tax + dividend_tax, 2),
        "simple_return_pct": pct(capital_gain, initially_invested) or 0.0,
        "dividend_yield_pct": to_float(item.get("dividend_yield_pct")) or pct(dividend_after_tax, total_worth) or 0.0,
        "dividend_yield_on_cost_pct": pct(dividend_after_tax, initially_invested) or 0.0,
        "today_gain": to_float(item.get("today_gain")) or 0.0,
        "today_gain_pct": to_float(item.get("today_gain_pct")) or 0.0,
    }


def snaptrade_cash_balance(value: Any, *, positions_value: float = 0.0) -> tuple[float, str | None]:
    rows: list[Any]
    if isinstance(value, dict):
        direct_cash = first_number(value, "cash.amount", "cash", "total_cash.amount", "total_cash")
        if direct_cash is not None:
            direct_currency = first_value(value, "cash.currency", "cash.currency.code", "currency", "currency.code")
            return direct_cash, str(direct_currency) if direct_currency else None
        direct_total = first_number(value, "total.amount", "total")
        direct_currency = first_value(value, "cash.currency", "cash.currency.code", "total.currency", "total.currency.code", "currency", "currency.code")
        if direct_total is not None:
            return max(0.0, direct_total - positions_value), str(direct_currency) if direct_currency else None
    if isinstance(value, dict) and isinstance(value.get("balances"), list):
        rows = value["balances"]
    elif isinstance(value, list):
        rows = value
    else:
        rows = []
    total = 0.0
    currencies: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        total += first_number(row, "cash.amount", "cash", "total_cash.amount", "total_cash", "amount") or 0.0
        currency = first_value(row, "cash.currency", "cash.currency.code", "currency.code", "currency")
        if currency:
            currencies.add(str(currency))
    if len(currencies) == 1:
        return total, next(iter(currencies))
    return total, "MIXED" if currencies else None


def normalize_positions_container(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and isinstance(value.get("error"), dict):
        return []
    if isinstance(value, dict):
        for key in ("positions", "results", "data", "holdings"):
            if isinstance(value.get(key), list):
                return [item for item in value[key] if isinstance(item, dict)]
        if isinstance(value.get("account"), dict) and isinstance(value["account"].get("positions"), list):
            return [item for item in value["account"]["positions"] if isinstance(item, dict)]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def summarize_unified_holdings(holdings: list[dict[str, Any]]) -> dict[str, Any]:
    current_value = sum_numeric(holdings, "current_value")
    cost_basis = sum_numeric(holdings, "cost_basis")
    unrealized_pl = sum_numeric(holdings, "unrealized_pl")
    return {
        "holding_count": len(holdings),
        "total_current_value": round(current_value, 2),
        "total_cost_basis": round(cost_basis, 2),
        "total_unrealized_pl": round(unrealized_pl, 2),
        "total_unrealized_pl_pct": pct(unrealized_pl, cost_basis),
        "broker_count": len({row.get("broker") for row in holdings if row.get("broker")}),
        "account_count": len({row.get("account_key") for row in holdings if row.get("account_key")}),
    }


def summarize_by(holdings: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in holdings:
        grouped.setdefault(str(row.get(key) or "unknown"), []).append(row)
    summary = []
    for group_key, rows in grouped.items():
        current_value = sum_numeric(rows, "current_value")
        cost_basis = sum_numeric(rows, "cost_basis")
        unrealized_pl = sum_numeric(rows, "unrealized_pl")
        first = rows[0] if rows else {}
        summary.append(
            {
                "key": group_key,
                "broker": first.get("broker"),
                "account_name": first.get("account_name"),
                "institution": first.get("institution"),
                "holding_count": len(rows),
                "current_value": round(current_value, 2),
                "cost_basis": round(cost_basis, 2),
                "unrealized_pl": round(unrealized_pl, 2),
                "unrealized_pl_pct": pct(unrealized_pl, cost_basis),
            }
        )
    return sorted(summary, key=lambda row: float(row.get("current_value") or 0), reverse=True)


def read_json_file(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, f"file_not_found: {path}"
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, str(exc)
    return parsed if isinstance(parsed, dict) else {}, None


def sum_numeric(rows: list[dict[str, Any]], key: str) -> float:
    return sum(to_float(row.get(key)) or 0.0 for row in rows)


def pct(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round((numerator / denominator) * 100, 2)


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_number(mapping: dict[str, Any], *paths: str) -> float | None:
    for path in paths:
        value = first_value(mapping, path)
        parsed = to_float(value)
        if parsed is not None:
            return parsed
    return None


def first_value(mapping: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        value = get_path(mapping, path)
        if value not in (None, ""):
            return value
    return None


def get_path(mapping: Any, path: str) -> Any:
    current = mapping
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current


def nested_dict(mapping: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = mapping.get(key)
    return value if isinstance(value, dict) else None
