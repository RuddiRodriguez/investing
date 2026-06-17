from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from market_forecasting_engine.alpaca_broker import AlpacaPaperBroker


DEFAULT_STATE_DIR = Path("automated_forecasting_engine/runs/paper_options_agent")
DEFAULT_REFRESH_SECONDS = 30


def build_dashboard_state(
    *, state_dir: Path, ticker: str, max_history: int = 50, include_live_bars: bool = True
) -> dict[str, Any]:
    report, report_path = read_latest_report(state_dir, ticker)
    state, state_path = read_state(state_dir, ticker)
    history, log_path = read_history(state_dir, ticker, max_history=max_history)
    display_report = _display_report_with_cached_forecast(report, state)
    trade_plan = display_report.get("option_trade_plan") or {}
    selected = trade_plan.get("selected_contract") or {}
    order = trade_plan.get("order") or {}
    exit_plan = trade_plan.get("exit_plan") or {}
    position_pl = summarize_position_pl(report.get("option_positions") or [])
    performance = summarize_performance(history=history, report=report, position_pl=position_pl)
    option_type = _option_type_label(trade_plan.get("option_type"), selected.get("symbol") or order.get("symbol"))
    chart = build_stock_chart_payload(report=display_report, history=history, ticker=ticker, include_live_bars=include_live_bars)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "ticker": ticker.upper(),
        "state_dir": str(state_dir),
        "report_path": str(report_path) if report_path else None,
        "state_path": str(state_path),
        "log_path": str(log_path) if log_path else None,
        "report": report,
        "state": state,
        "summary": {
            "checked_at": report.get("checked_at"),
            "market_is_open": (report.get("market_clock") or {}).get("is_open"),
            "next_open": (report.get("market_clock") or {}).get("next_open"),
            "forecast_cache_status": display_report.get("forecast_cache_status"),
            "forecast_direction": (display_report.get("selected_forecast") or {}).get("expected_direction"),
            "spot": (display_report.get("selected_forecast") or {}).get("spot"),
            "predicted_price": (display_report.get("selected_forecast") or {}).get("predicted_price"),
            "action": trade_plan.get("action") or report.get("command"),
            "strategy": trade_plan.get("strategy") or trade_plan.get("option_type"),
            "strategy_reason": trade_plan.get("strategy_reason") or _strategy_reason_text(trade_plan.get("strategy") or trade_plan.get("option_type"), trade_plan, display_report),
            "option_type": trade_plan.get("option_type"),
            "option_type_label": option_type,
            "contract": selected.get("symbol"),
            "contract_name": selected.get("name"),
            "greeks": selected.get("greeks") or {},
            "spread_pct": selected.get("spread_pct"),
            "open_interest": selected.get("open_interest"),
            "trade_quality": trade_plan.get("trade_quality") or selected.get("trade_quality") or {},
            "entry_type": order.get("type"),
            "entry_limit": order.get("limit_price"),
            "take_profit": (exit_plan.get("take_profit") or {}).get("limit_price"),
            "stop_price": (exit_plan.get("stop_loss") or {}).get("stop_price"),
            "stop_limit": (exit_plan.get("stop_loss") or {}).get("limit_price"),
            "estimated_debit": (trade_plan.get("risk") or {}).get("estimated_debit"),
            "sizing": trade_plan.get("sizing"),
            "execution_blocks": report.get("execution_blocks") or [],
            "order_submitted": (report.get("order_result") or {}).get("submitted"),
            "order_result": report.get("order_result"),
            "position_pl": position_pl,
            "performance": performance,
        },
        "history": history,
        "chart": chart,
    }


def _display_report_with_cached_forecast(report: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    if report.get("forecast_plan") and report.get("selected_forecast"):
        return report
    forecast_bundle = state.get("last_forecast") if isinstance(state.get("last_forecast"), dict) else {}
    forecast_plan = forecast_bundle.get("forecast_plan") if isinstance(forecast_bundle.get("forecast_plan"), dict) else {}
    selected_forecast = forecast_bundle.get("selected_forecast") if isinstance(forecast_bundle.get("selected_forecast"), dict) else {}
    if not forecast_plan and not selected_forecast:
        return report
    display = dict(report)
    display.setdefault("forecast_created_at_utc", forecast_bundle.get("created_at_utc"))
    display.setdefault("forecast_cache_status", "cached_from_state_after_stop_record")
    display["forecast_plan"] = forecast_plan
    display["selected_forecast"] = selected_forecast
    summary = state.get("last_record_summary") if isinstance(state.get("last_record_summary"), dict) else {}
    if summary and not display.get("option_trade_plan"):
        selected_contract_symbol = summary.get("contract")
        display["option_trade_plan"] = {
            "action": summary.get("action"),
            "reason": summary.get("reason"),
            "strategy": summary.get("strategy"),
            "selected_contract": {
                "symbol": selected_contract_symbol,
                "name": summary.get("contract_name"),
                "trade_quality": {
                    "score": summary.get("trade_quality_score"),
                    "grade": summary.get("trade_quality_grade"),
                },
            },
            "order": {
                "symbol": selected_contract_symbol,
                "limit_price": summary.get("limit_price"),
            },
            "risk": {"estimated_debit": summary.get("estimated_debit")},
        }
    return display


def build_stock_chart_payload(
    *,
    report: dict[str, Any],
    history: list[dict[str, Any]],
    ticker: str | None = None,
    include_live_bars: bool = True,
) -> dict[str, Any]:
    broker_points = _recent_stock_bar_points(ticker or str(report.get("ticker") or "")) if include_live_bars else []
    history_points = _history_actual_points(history)
    forecast_plan = report.get("forecast_plan") or {}
    selected_forecast = report.get("selected_forecast") or {}
    as_of = forecast_plan.get("as_of") or report.get("forecast_created_at_utc") or report.get("checked_at")
    as_of_price = _to_float(forecast_plan.get("latest_price")) or _to_float(selected_forecast.get("spot"))
    broker_points_are_stale = _points_are_stale_for_as_of(broker_points, as_of)
    if broker_points and not broker_points_are_stale:
        actual_points = broker_points
        source = "alpaca_stock_bars"
    elif history_points:
        actual_points = history_points
        source = "agent_history_stale_broker_bars" if broker_points_are_stale else "agent_history"
    else:
        actual_points = broker_points
        source = "alpaca_stock_bars_stale" if broker_points_are_stale else "alpaca_stock_bars"
    latest_actual = actual_points[-1] if actual_points else {}
    forecasts = forecast_plan.get("forecasts") or []
    if not forecasts and selected_forecast:
        forecasts = [selected_forecast]
    forecast_points: list[dict[str, Any]] = []
    for forecast in forecasts:
        timestamp = forecast.get("forecast_timestamp") or forecast.get("forecast_date")
        predicted = _to_float(forecast.get("predicted_price"))
        if not timestamp or predicted is None:
            continue
        forecast_points.append(
            {
                "timestamp": str(timestamp),
                "predicted_price": predicted,
                "lower_price": _to_float(forecast.get("lower_price")),
                "upper_price": _to_float(forecast.get("upper_price")),
                "horizon_hours": _to_float(forecast.get("horizon_hours")),
            }
        )
    return {
        "actual_points": actual_points[-10000:],
        "forecast_points": forecast_points,
        "as_of": as_of,
        "as_of_price": as_of_price,
        "latest_actual": latest_actual,
        "source": source,
    }


def _history_actual_points(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actual_points: list[dict[str, Any]] = []
    seen_actual: set[str] = set()
    for row in history:
        timestamp = row.get("checked_at")
        price = _to_float((row.get("selected_forecast") or {}).get("spot"))
        if price is None:
            price = _to_float((row.get("forecast_plan") or {}).get("latest_price"))
        if not timestamp or price is None:
            continue
        key = str(timestamp)
        if key in seen_actual:
            continue
        seen_actual.add(key)
        actual_points.append({"timestamp": key, "price": price})
    return actual_points


def _points_are_stale_for_as_of(points: list[dict[str, Any]], as_of: Any, *, max_gap_hours: float = 24.0) -> bool:
    if not points or not as_of:
        return False
    latest = _parse_timestamp(points[-1].get("timestamp"))
    anchor = _parse_timestamp(as_of)
    if latest is None or anchor is None:
        return False
    return abs((anchor - latest).total_seconds()) > max_gap_hours * 3600.0


def _parse_timestamp(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _recent_stock_bar_points(ticker: str) -> list[dict[str, Any]]:
    symbol = str(ticker or "").upper().strip()
    if not symbol or "/" in symbol or "-" in symbol:
        return []
    end = datetime.now(UTC)
    start = end - pd.Timedelta(days=183)
    broker = AlpacaPaperBroker()
    rows: list[dict[str, Any]] = []
    for feed in (None, "iex"):
        try:
            rows = broker.stock_bars(
                symbol,
                start=start.isoformat().replace("+00:00", "Z"),
                end=end.isoformat().replace("+00:00", "Z"),
                timeframe="5Min",
                feed=feed,
                limit=10000,
            )
            if rows:
                break
        except Exception:
            continue
    points: list[dict[str, Any]] = []
    for row in rows:
        timestamp = row.get("t") or row.get("timestamp")
        close = _to_float(row.get("c") or row.get("close"))
        open_price = _to_float(row.get("o") or row.get("open"))
        high = _to_float(row.get("h") or row.get("high"))
        low = _to_float(row.get("l") or row.get("low"))
        volume = _to_float(row.get("v") or row.get("volume"))
        if timestamp and close is not None:
            points.append(
                {
                    "timestamp": str(timestamp),
                    "price": close,
                    "open": open_price if open_price is not None else close,
                    "high": high if high is not None else close,
                    "low": low if low is not None else close,
                    "close": close,
                    "volume": volume,
                }
            )
    return points


def summarize_position_pl(positions: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_cost = 0.0
    total_value = 0.0
    total_pl = 0.0
    for position in positions:
        symbol = str(position.get("symbol") or "")
        if not symbol:
            continue
        qty = _to_float(position.get("qty"))
        avg_entry = _to_float(position.get("avg_entry_price"))
        current_price = _to_float(position.get("current_price"))
        cost_basis = _to_float(position.get("cost_basis"))
        market_value = _to_float(position.get("market_value"))
        unrealized_pl = _to_float(position.get("unrealized_pl"))
        if cost_basis is None and qty is not None and avg_entry is not None:
            cost_basis = qty * avg_entry * 100
        if market_value is None and qty is not None and current_price is not None:
            market_value = qty * current_price * 100
        if unrealized_pl is None and cost_basis is not None and market_value is not None:
            unrealized_pl = market_value - cost_basis
        total_cost += cost_basis or 0.0
        total_value += market_value or 0.0
        total_pl += unrealized_pl or 0.0
        rows.append(
            {
                "symbol": symbol,
                "option_type": _option_type_label(None, symbol),
                "qty": qty,
                "avg_entry_price": avg_entry,
                "current_price": current_price,
                "cost_basis": cost_basis,
                "market_value": market_value,
                "unrealized_pl": unrealized_pl,
                "unrealized_plpc": _to_float(position.get("unrealized_plpc")),
            }
        )
    status = "flat"
    if total_pl > 0:
        status = "winning"
    elif total_pl < 0:
        status = "losing"
    return {
        "status": status,
        "total_cost": round(total_cost, 2),
        "total_value": round(total_value, 2),
        "total_unrealized_pl": round(total_pl, 2),
        "rows": rows,
    }


def summarize_performance(
    *, history: list[dict[str, Any]], report: dict[str, Any], position_pl: dict[str, Any]
) -> dict[str, Any]:
    metrics = ((report.get("entry_guard") or {}).get("metrics") or {}).copy()
    if not metrics:
        for row in reversed(history):
            metrics = ((row.get("entry_guard") or {}).get("metrics") or {}).copy()
            if metrics:
                break
    realized = _to_float(metrics.get("realized_pnl_today")) or 0.0
    open_pl = _to_float(position_pl.get("total_unrealized_pl")) or 0.0
    open_exposure = _to_float(position_pl.get("total_cost")) or 0.0
    round_trips = int(_to_float(metrics.get("round_trips_today")) or 0)
    entries = int(_to_float(metrics.get("buy_entries_today")) or 0)
    consecutive_losses = int(_to_float(metrics.get("consecutive_losses")) or 0)
    submitted_orders = 0
    blocked_cycles = 0
    hold_cycles = 0
    buy_cycles = 0
    realized_deltas: list[float] = []
    previous_round_trips: int | None = None
    previous_realized: float | None = None
    for row in history:
        if (row.get("order_result") or {}).get("submitted"):
            submitted_orders += 1
        if row.get("execution_blocks"):
            blocked_cycles += 1
        action = str((row.get("option_trade_plan") or {}).get("action") or "")
        if action == "hold":
            hold_cycles += 1
        if action == "buy_option":
            buy_cycles += 1
        row_metrics = (row.get("entry_guard") or {}).get("metrics") or {}
        row_round_trips = int(_to_float(row_metrics.get("round_trips_today")) or 0)
        row_realized = _to_float(row_metrics.get("realized_pnl_today"))
        if row_realized is None:
            continue
        if previous_round_trips is not None and row_round_trips > previous_round_trips and previous_realized is not None:
            realized_deltas.append(round(row_realized - previous_realized, 2))
        previous_round_trips = row_round_trips
        previous_realized = row_realized
    wins = sum(1 for value in realized_deltas if value > 0)
    losses = sum(1 for value in realized_deltas if value < 0)
    scored_round_trips = wins + losses
    win_rate = (wins / scored_round_trips) if scored_round_trips else None
    total = realized + open_pl
    status = "flat"
    if total > 0:
        status = "winning"
    elif total < 0:
        status = "losing"
    return {
        "status": status,
        "realized_pnl_today": round(realized, 2),
        "open_pnl": round(open_pl, 2),
        "total_pnl": round(total, 2),
        "open_exposure": round(open_exposure, 2),
        "entries_today": entries,
        "round_trips_today": round_trips,
        "consecutive_losses": consecutive_losses,
        "win_count_from_log": wins,
        "loss_count_from_log": losses,
        "win_rate_from_log": round(win_rate, 4) if win_rate is not None else None,
        "submitted_orders_in_view": submitted_orders,
        "blocked_cycles_in_view": blocked_cycles,
        "hold_cycles_in_view": hold_cycles,
        "buy_cycles_in_view": buy_cycles,
        "last_filled_order_at": metrics.get("last_filled_order_at"),
        "last_losing_trade_exit_at": metrics.get("last_losing_trade_exit_at"),
    }


def summarize_all_agents(state_dir: Path, tickers: set[str] | None = None, run_prefixes: set[str] | None = None) -> dict[str, Any]:
    allowed = {ticker.upper().strip() for ticker in tickers or set() if ticker.strip()}
    allowed_prefixes = {str(prefix).strip() for prefix in run_prefixes or set() if str(prefix).strip()}
    root = state_dir.parent
    current_date = datetime.now(UTC).date()
    controller_report = _read_json(root / "daily_target_options_controller_report.json")
    controller_active_tickers = {
        str(ticker).upper()
        for ticker in controller_report.get("active_tickers", []) or []
        if str(ticker).strip()
    }
    controller_valuation = controller_report.get("valuation_after") or controller_report.get("valuation_before") or {}
    controller_exposed_tickers = {
        str(ticker).upper()
        for ticker in [
            *(controller_valuation.get("open_position_underlyings") or []),
            *(controller_valuation.get("open_order_underlyings") or []),
        ]
        if str(ticker).strip()
    }
    controller_visible_daily_tickers = controller_active_tickers | controller_exposed_tickers
    candidate_patterns = (
        "alpaca_options_*",
        "alpaca_hybrid_options_paper_*",
        "alpaca_daily_target_options_*",
        "alpaca_chronos_options_*",
    )
    candidates = sorted({path for pattern in candidate_patterns for path in root.glob(pattern)}, key=lambda path: str(path))
    if allowed_prefixes:
        candidates = [path for path in candidates if any(path.name.startswith(prefix) for prefix in allowed_prefixes)]
    if not candidates:
        candidates = [state_dir]
    latest_by_ticker: dict[str, tuple[Path, Path]] = {}
    for candidate in candidates:
        report_paths = sorted(candidate.glob("*_options_agent_report.json"))
        for report_path in report_paths:
            ticker = report_path.name.removesuffix("_options_agent_report.json").upper()
            if allowed and ticker not in allowed:
                continue
            if (
                candidate.name.startswith("alpaca_daily_target_options_")
                and not allowed
                and controller_visible_daily_tickers
                and ticker not in controller_visible_daily_tickers
            ):
                continue
            current = latest_by_ticker.get(ticker)
            if current is None or report_path.stat().st_mtime > current[1].stat().st_mtime:
                latest_by_ticker[ticker] = (candidate, report_path)

    rows: list[dict[str, Any]] = []
    totals = {
        "realized_pnl_today": 0.0,
        "open_pnl": 0.0,
        "total_pnl": 0.0,
        "open_exposure": 0.0,
        "entries_today": 0,
        "round_trips_today": 0,
        "submitted_orders_in_view": 0,
    }
    for ticker, (candidate, report_path) in sorted(latest_by_ticker.items()):
        report = _read_json(report_path)
        history, _ = read_history(candidate, ticker, max_history=200)
        position_pl = summarize_position_pl(report.get("option_positions") or [])
        performance = summarize_performance(history=history, report=report, position_pl=position_pl)
        checked_date = _parse_timestamp(report.get("checked_at"))
        is_stale_closed_report = (
            checked_date is not None
            and checked_date.date() != current_date
            and float(performance.get("open_exposure") or 0.0) <= 0.0
            and float(performance.get("open_pnl") or 0.0) == 0.0
        )
        if is_stale_closed_report:
            continue
        latest_blocks = report.get("execution_blocks") or []
        trade_plan = report.get("option_trade_plan") or {}
        latest_action = trade_plan.get("action") or "-"
        selected_contract = trade_plan.get("selected_contract") or {}
        strategy = trade_plan.get("strategy") or trade_plan.get("option_type") or "-"
        row = {
            "ticker": ticker,
            "state_dir": str(candidate),
            "report_path": str(report_path),
            "checked_at": report.get("checked_at"),
            "action": latest_action,
            "forecast_engine": (trade_plan.get("forecast_engine") or (report.get("forecast_plan") or {}).get("forecast_engine") or {}).get("selected")
            or (report.get("forecast_plan") or {}).get("mode"),
            "forecast_model": ((report.get("selected_forecast") or {}).get("selected_model") or (report.get("selected_forecast") or {}).get("method")),
            "forecast_fallback": ((report.get("forecast_plan") or {}).get("forecast_engine") or {}).get("fallback_from_full_engine"),
            "strategy": strategy,
            "strategy_reason": trade_plan.get("strategy_reason") or _strategy_reason_text(strategy, trade_plan, report),
            "blocks": latest_blocks,
            "forecast_direction": (report.get("selected_forecast") or {}).get("expected_direction"),
            "spot": (report.get("selected_forecast") or {}).get("spot"),
            "contract": selected_contract.get("symbol"),
            "trade_quality": trade_plan.get("trade_quality") or selected_contract.get("trade_quality") or {},
            "performance": performance,
        }
        rows.append(row)
        totals["realized_pnl_today"] += performance["realized_pnl_today"]
        totals["open_pnl"] += performance["open_pnl"]
        totals["total_pnl"] += performance["total_pnl"]
        totals["open_exposure"] += performance["open_exposure"]
        totals["entries_today"] += performance["entries_today"]
        totals["round_trips_today"] += performance["round_trips_today"]
        totals["submitted_orders_in_view"] += performance["submitted_orders_in_view"]
    for key in ("realized_pnl_today", "open_pnl", "total_pnl", "open_exposure"):
        totals[key] = round(totals[key], 2)
    totals["agent_count"] = len(rows)
    totals["status"] = "winning" if totals["total_pnl"] > 0 else "losing" if totals["total_pnl"] < 0 else "flat"
    rows.sort(key=lambda row: row.get("ticker") or "")
    return {"totals": totals, "rows": rows}


def _strategy_reason_text(strategy: Any, trade_plan: dict[str, Any], report: dict[str, Any]) -> str:
    strategy_name = str(strategy or "").lower()
    forecast = report.get("selected_forecast") or {}
    direction = str(forecast.get("expected_direction") or "").lower()
    market_regime = trade_plan.get("market_regime") or {}
    regime = str(market_regime.get("regime") or "").lower()
    if strategy_name == "call":
        return "Directional call: forecast and contract filters currently favor an upward move."
    if strategy_name == "put":
        return "Directional put: forecast and contract filters currently favor a downward move."
    if strategy_name == "long_straddle":
        return "Long straddle: direction is less certain, but the plan expects a large enough move to justify buying both sides."
    if strategy_name == "short_iron_butterfly":
        return "Short iron butterfly: price appears range-bound/choppy, so the plan sells defined-risk premium around the center strike."
    if strategy_name == "long_call_calendar":
        return "Call calendar spread: bullish thesis looks slower, using near/far expiry structure instead of a single call."
    if strategy_name == "long_put_calendar":
        return "Put calendar spread: bearish thesis looks slower, using near/far expiry structure instead of a single put."
    if regime:
        return f"Strategy selected from market regime '{regime}' and {direction or 'flat'} forecast."
    return "Strategy selected from forecast, live option quotes, spread, Greeks, liquidity, and risk budget."


def read_latest_report(state_dir: Path, ticker: str) -> tuple[dict[str, Any], Path | None]:
    path = state_dir / f"{ticker.upper()}_options_agent_report.json"
    if not path.exists():
        return {}, None
    return _read_json(path), path


def read_state(state_dir: Path, ticker: str) -> tuple[dict[str, Any], Path]:
    path = state_dir / "state" / f"{ticker.upper()}_options_agent_state.json"
    if not path.exists():
        return {}, path
    return _read_json(path), path


def read_history(state_dir: Path, ticker: str, *, max_history: int) -> tuple[list[dict[str, Any]], Path | None]:
    paths = sorted((state_dir / "logs").glob(f"{ticker.upper()}_*.jsonl"))
    rows: list[dict[str, Any]] = []
    latest_path = None
    for path in paths:
        latest_path = path
        rows.extend(_read_jsonl(path))
    return rows[-max_history:], latest_path


def dashboard_html(refresh_seconds: int) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Options Agent Dashboard</title>
  <style>
    :root {{
      --bg: #0b1119;
      --panel: #111a26;
      --panel2: #0f1722;
      --text: #e7eef8;
      --muted: #9fb1c8;
      --line: #263548;
      --buy: #2ed48f;
      --hold: #f6b84b;
      --blocked: #ff5c74;
      --accent: #6aa2ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }}
    header {{ display: flex; justify-content: space-between; gap: 16px; padding: 22px 32px; border-bottom: 1px solid var(--line); background: #0c141f; position: sticky; top: 0; z-index: 2; }}
    h1 {{ margin: 0; font-size: 20px; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 18px 22px 30px; }}
    .meta, .small {{ color: var(--muted); font-size: 13px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }}
    .card, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; box-shadow: 0 18px 50px rgba(0,0,0,.16); }}
    .label {{ color: var(--muted); font-size: 12px; margin-bottom: 5px; }}
    .value {{ font-size: 22px; font-weight: 750; overflow-wrap: anywhere; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: 5px 10px; font-size: 12px; font-weight: 750; color: #09111c; background: var(--hold); text-transform: uppercase; }}
    .badge.buy_option {{ background: var(--buy); color:#06130e; }}
    .badge.blocked {{ background: var(--blocked); color:#1d0509; }}
    .win {{ color: var(--buy); }}
    .loss {{ color: var(--blocked); }}
    .grid2 {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(360px, 0.8fr); gap: 14px; }}
    .chart-panel {{ margin-bottom: 14px; overflow-x: auto; }}
    .chart-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 10px; }}
    .legend {{ display: flex; gap: 14px; flex-wrap: wrap; }}
    .key {{ display: inline-flex; gap: 6px; align-items: center; color: var(--muted); font-size: 12px; }}
    .swatch {{ width: 18px; height: 0; border-top: 2px solid currentColor; display: inline-block; }}
    .swatch.actual {{ color: #dbe7f6; }}
    .swatch.forecast {{ color: #ff5c74; border-top-style: dashed; }}
    .chartControls {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:10px 0 12px; }}
    .chartControls button,.chartControls label {{ border:1px solid var(--line); background:var(--panel2); color:var(--text); border-radius:6px; padding:7px 10px; font-size:12px; }}
    .chartControls button.active {{ background:var(--accent); color:#04111f; font-weight:800; }}
    .chartControls label {{ display:flex; gap:6px; align-items:center; cursor:pointer; }}
    svg#stockChart {{ width: 100%; min-width: 920px; height: 520px; display: block; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 6px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 650; }}
    pre {{ margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; color:#d9e4f2; }}
    .blocks {{ color: var(--blocked); font-weight: 650; }}
    @media (max-width: 900px) {{ header {{ flex-direction: column; }} .cards, .grid2 {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Options Agent Dashboard</h1>
      <div class="small" id="source">Loading...</div>
    </div>
    <div class="meta"><div id="updated">Loading...</div><div>Refresh: {refresh_seconds}s</div></div>
  </header>
  <main>
    <section class="cards">
      <div class="card"><div class="label">Decision</div><div class="value"><span id="actionBadge" class="badge">-</span></div><div class="small" id="blocks">-</div></div>
      <div class="card"><div class="label">Forecast</div><div class="value" id="forecast">-</div><div class="small" id="forecastMeta">-</div></div>
      <div class="card"><div class="label">Option Type</div><div class="value" id="optionType">-</div><div class="small">Call profits if stock rises; Put profits if stock falls</div></div>
      <div class="card"><div class="label">Contract</div><div class="value" id="contract">-</div><div class="small" id="contractMeta">-</div></div>
    </section>
    <section class="cards">
      <div class="card"><div class="label">Entry / Risk</div><div class="value" id="entry">-</div><div class="small" id="risk">-</div></div>
      <div class="card"><div class="label">Take Profit</div><div class="value" id="takeProfit">-</div><div class="small">Autonomous sell limit</div></div>
      <div class="card"><div class="label">Stop Loss</div><div class="value" id="stopLoss">-</div><div class="small">Autonomous stop-limit</div></div>
      <div class="card"><div class="label">Market</div><div class="value" id="market">-</div><div class="small" id="nextOpen">-</div></div>
    </section>
    <section class="cards">
      <div class="card"><div class="label">Delta</div><div class="value" id="greekDelta">-</div><div class="small">Directional exposure</div></div>
      <div class="card"><div class="label">Gamma</div><div class="value" id="greekGamma">-</div><div class="small">Delta acceleration</div></div>
      <div class="card"><div class="label">Theta</div><div class="value" id="greekTheta">-</div><div class="small" id="thetaMeta">Time decay review</div></div>
      <div class="card"><div class="label">Vega / Liquidity</div><div class="value" id="greekVega">-</div><div class="small" id="liquidityMeta">-</div></div>
    </section>
    <section class="cards">
      <div class="card"><div class="label">Order Result</div><div class="value" id="orderResult">-</div><div class="small" id="checkedAt">-</div></div>
      <div class="card"><div class="label">Trade Quality</div><div class="value" id="tradeQuality">-</div><div class="small" id="tradeQualityMeta">-</div></div>
    </section>
    <section class="panel chart-panel">
      <div class="chart-head">
        <div><strong>Stock Price And Forecast</strong><div class="small" id="chartMeta">-</div></div>
        <div class="legend">
          <span class="key"><span class="swatch actual"></span>Stock price</span>
          <span class="key"><span class="swatch forecast"></span>Forecast line</span>
        </div>
      </div>
      <div class="chartControls">
        <button type="button" data-range="1h" class="active">1h</button>
        <button type="button" data-range="1d">1d</button>
        <button type="button" data-range="2d">2d</button>
        <button type="button" data-range="1m">1m</button>
        <button type="button" data-range="2m">2m</button>
        <button type="button" data-range="3m">3m</button>
        <button type="button" data-range="6m">6m</button>
        <button type="button" id="forecastZoom">Forecast zoom</button>
        <button type="button" id="zoomIn">Zoom in</button>
        <button type="button" id="zoomOut">Zoom out</button>
        <label><input type="checkbox" id="showLine" checked> line</label>
        <label><input type="checkbox" id="showCandles" checked> candles</label>
        <label><input type="checkbox" id="showSma9" checked> SMA 9</label>
        <label><input type="checkbox" id="showSma21" checked> SMA 21</label>
      </div>
      <svg id="stockChart" role="img" aria-label="Stock price chart with forecast line"></svg>
    </section>
    <section class="panel" style="margin-bottom: 14px;">
      <strong>Open Profit / Loss</strong>
      <div class="cards" style="margin: 12px 0 8px;">
        <div class="card"><div class="label">Plain English</div><div class="value" id="plStatus">-</div><div class="small">Still open, not final profit</div></div>
        <div class="card"><div class="label">Total P/L</div><div class="value" id="plTotal">-</div><div class="small">Unrealized</div></div>
        <div class="card"><div class="label">Cost</div><div class="value" id="plCost">-</div><div class="small">What the open contracts cost</div></div>
        <div class="card"><div class="label">Value Now</div><div class="value" id="plValue">-</div><div class="small">Current broker mark</div></div>
      </div>
      <table>
        <thead><tr><th>Position</th><th>Type</th><th>Qty</th><th>Entry</th><th>Current</th><th>Cost</th><th>Value</th><th>P/L</th></tr></thead>
        <tbody id="plRows"></tbody>
      </table>
    </section>
    <section class="grid2">
      <div class="panel">
        <strong>Recent Decisions</strong>
        <table>
          <thead><tr><th>Checked</th><th>Action</th><th>Type</th><th>Contract</th><th>Entry</th><th>Blocks</th><th>Submitted</th></tr></thead>
          <tbody id="historyRows"></tbody>
        </table>
      </div>
      <div class="panel">
        <strong>Details</strong>
        <pre id="details">Loading...</pre>
      </div>
    </section>
  </main>
  <script>
    const refreshMs = {refresh_seconds * 1000};
    const chartPrefs = {{ range: "1h", zoomFactor: 1, forecastZoom: false, showLine: true, showCandles: true, showSma9: true, showSma21: true }};
    let lastDashboardData = null;
    const fmt = new Intl.NumberFormat(undefined, {{ maximumFractionDigits: 4 }});
    function price(v) {{ return v === null || v === undefined || Number.isNaN(Number(v)) ? "-" : fmt.format(Number(v)); }}
    function money(v) {{
      if (v === null || v === undefined || Number.isNaN(Number(v))) return "-";
      const n = Number(v);
      const sign = n > 0 ? "+" : "";
      return `${{sign}}$${{fmt.format(n)}}`;
    }}
    function time(v) {{ if (!v) return "-"; const d = new Date(v); return Number.isNaN(d.getTime()) ? v : d.toLocaleString(); }}
    function set(id, value) {{ document.getElementById(id).textContent = value; }}
    function summarize(row) {{
      const plan = row.option_trade_plan || {{}};
      const selected = plan.selected_contract || {{}};
      const order = plan.order || {{}};
      return {{
        checked: row.checked_at,
        action: plan.action || "-",
        type: optionTypeLabel(plan.option_type, selected.symbol || order.symbol),
        contract: selected.symbol || "-",
        entry: order.limit_price,
        blocks: row.execution_blocks || [],
        submitted: (row.order_result || {{}}).submitted
      }};
    }}
    function render(data) {{
      lastDashboardData = data;
      const s = data.summary || {{}};
      set("source", `${{data.ticker}} / ${{data.state_dir}}`);
      set("updated", `Updated ${{time(data.generated_at)}}`);
      const badge = document.getElementById("actionBadge");
      const blocked = (s.execution_blocks || []).length > 0;
      badge.textContent = blocked ? "blocked" : (s.action || "-");
      badge.className = `badge ${{blocked ? "blocked" : s.action || ""}}`;
      set("blocks", (s.execution_blocks || []).join(", ") || "no execution blocks");
      set("forecast", `${{s.forecast_direction || "-"}} → ${{price(s.predicted_price)}}`);
      set("forecastMeta", `spot ${{price(s.spot)}} | cache ${{s.forecast_cache_status || "-"}}`);
      set("optionType", s.option_type_label || "-");
      set("contract", s.contract || "-");
      set("contractMeta", `${{s.option_type_label || "-"}} | ${{s.contract_name || "-"}}`);
      set("entry", `${{s.entry_type || "-"}} @ ${{price(s.entry_limit)}}`);
      const sizing = s.sizing || {{}};
      set("risk", `qty ${{sizing.qty ?? "-"}} | debit ${{price(s.estimated_debit)}} | budget ${{price(sizing.budget)}}`);
      const g = s.greeks || {{}};
      set("greekDelta", price(g.delta));
      set("greekGamma", price(g.gamma));
      set("greekTheta", price(g.theta));
      set("thetaMeta", `horizon decay $${{price(g.theta_decay_usd_for_horizon)}} | premium/day ${{price((g.theta_premium_pct_per_day || 0) * 100)}}%`);
      set("greekVega", price(g.vega));
      set("liquidityMeta", `spread ${{price((s.spread_pct || 0) * 100)}}% | open interest ${{price(s.open_interest)}}`);
      const q = s.trade_quality || {{}};
      set("tradeQuality", q.score === undefined ? "-" : `${{price(q.score)}} / 100`);
      set("tradeQualityMeta", q.grade ? `${{q.grade}} | breakeven edge ${{price(q.forecast_breakeven_edge ?? q.expected_move_vs_debit_edge)}} | theta $${{price(q.theta_cost_for_horizon_usd)}}` : "-");
      set("takeProfit", price(s.take_profit));
      set("stopLoss", `${{price(s.stop_price)}} / ${{price(s.stop_limit)}}`);
      set("market", s.market_is_open ? "open" : "closed");
      set("nextOpen", `next open: ${{time(s.next_open)}}`);
      set("orderResult", s.order_submitted ? "submitted" : "not submitted");
      set("checkedAt", time(s.checked_at));
      renderStockChart(data.chart || {{}});
      renderPositionPl(s.position_pl || {{}});
      const rows = document.getElementById("historyRows");
      rows.innerHTML = "";
      (data.history || []).slice().reverse().forEach(row => {{
        const r = summarize(row);
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${{time(r.checked)}}</td><td>${{r.action}}</td><td>${{r.type || "-"}}</td><td>${{r.contract}}</td><td>${{price(r.entry)}}</td><td class="blocks">${{r.blocks.join(", ")}}</td><td>${{r.submitted}}</td>`;
        rows.appendChild(tr);
      }});
      set("details", JSON.stringify({{ summary: s, report_path: data.report_path, state_path: data.state_path, log_path: data.log_path, report: data.report }}, null, 2));
    }}
    function renderStockChart(chart) {{
      const svg = document.getElementById("stockChart");
      const width = Math.max(svg.clientWidth || 1120, 920);
      const height = 520;
      const pad = {{ left: 86, right: 42, top: 34, bottom: 66 }};
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      svg.innerHTML = "";
      const raw = (chart.actual_points || []).map(p => {{
        const close = Number(p.close ?? p.price);
        const hasOhlc = p.open !== undefined || p.high !== undefined || p.low !== undefined || p.close !== undefined;
        return {{
          t: parseTimestampMs(p.timestamp),
          o: Number(p.open ?? close),
          h: Number(p.high ?? close),
          l: Number(p.low ?? close),
          c: close,
          v: Number(p.volume || 0),
          hasOhlc
        }};
      }}).filter(p => Number.isFinite(p.t) && Number.isFinite(p.c));
      const forecasts = (chart.forecast_points || []).map(p => ({{ ...p, x: parseTimestampMs(p.timestamp), y: Number(p.predicted_price), lo: Number(p.lower_price), hi: Number(p.upper_price) }})).filter(p => Number.isFinite(p.x) && Number.isFinite(p.y));
      const latest = raw.length ? raw.at(-1) : null;
      const latestTime = latest ? latest.t : null;
      const maturedCount = forecasts.filter(p => Number.isFinite(latestTime) && p.x <= latestTime).length;
      const pendingCount = forecasts.length - maturedCount;
      const rangeMs = rangeToMs(chartPrefs.range) / Math.max(chartPrefs.zoomFactor, 0.25);
      const asOfTime = chart.as_of ? parseTimestampMs(chart.as_of) : null;
      const asOfPrice = Number(chart.as_of_price);
      const hasForecastContext = forecasts.length > 0 && Number.isFinite(asOfTime) && Number.isFinite(asOfPrice);
      const maxForecastTime = forecasts.length ? Math.max(...forecasts.map(p => p.x)) : latestTime;
      const endTime = chartPrefs.forecastZoom && forecasts.length && Number.isFinite(maxForecastTime) ? maxForecastTime + 10 * 60 * 1000 : latestTime;
      const startTime = chartPrefs.forecastZoom && hasForecastContext ? asOfTime - Math.min(rangeMs, 90 * 60 * 1000) : (Number.isFinite(endTime) ? endTime - rangeMs : null);
      const visibleRaw = raw.filter(p => (!Number.isFinite(startTime) || p.t >= startTime) && (!Number.isFinite(endTime) || p.t <= endTime));
      const fallbackRaw = visibleRaw.length >= 2 ? visibleRaw : raw.slice(-240);
      const candleMinutes = candleBucketMinutes(chartPrefs.range);
      const hasRealOhlc = fallbackRaw.some(p => p.hasOhlc);
      const sparseActual = !hasRealOhlc || String(chart.source || "").startsWith("agent_history");
      const candles = sparseActual ? fallbackRaw.map(p => ({{ t: p.t, o: p.c, h: p.c, l: p.c, c: p.c, v: p.v || 0 }})) : aggregateCandles(fallbackRaw, candleMinutes);
      const closePoints = candles.map(c => ({{ t: c.t, y: c.c }}));
      const longSparseRange = sparseActual && ["1m", "2m", "3m", "6m"].includes(chartPrefs.range);
      const sma9 = sparseActual ? [] : movingAverage(closePoints, 9);
      const sma21 = sparseActual ? [] : movingAverage(closePoints, 21);
      set("chartMeta", `actual ${{sparseActual ? "samples" : "bars"}} ${{raw.length}} | displayed ${{sparseActual ? "samples" : candleMinutes + "m candles"}} ${{candles.length}} | forecast ${{forecasts.length}} | matured ${{maturedCount}} | pending ${{pendingCount}} | range ${{chartPrefs.forecastZoom ? "forecast zoom" : chartPrefs.range}}${{longSparseRange ? " | sparse long range: dots only" : ""}} | source ${{chart.source || "-"}}`);
      const xs = candles.map(p => p.t).concat(forecasts.map(p => p.x)).concat(hasForecastContext ? [asOfTime] : []);
      const ys = candles.flatMap(p => [p.h, p.l]).concat(sma9.map(p => p.y), sma21.map(p => p.y), forecasts.flatMap(p => [p.y, p.lo, p.hi]).filter(Number.isFinite), hasForecastContext ? [asOfPrice] : []);
      if (!xs.length || !ys.length) {{
        const text = el("text", {{ x: 20, y: 32, fill: "#9fb1c8", "font-size": 14 }});
        text.textContent = "No stock price history available yet.";
        svg.appendChild(text);
        return;
      }}
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY0 = Math.min(...ys), maxY0 = Math.max(...ys);
      const yPad = Math.max((maxY0 - minY0) * 0.12, 0.5);
      const minY = minY0 - yPad, maxY = maxY0 + yPad;
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      const x = v => pad.left + ((v - minX) / Math.max(maxX - minX, 1)) * plotW;
      const y = v => pad.top + (1 - ((v - minY) / Math.max(maxY - minY, 1))) * plotH;
      svg.appendChild(el("rect", {{ x: 0, y: 0, width, height, fill: "#111a26" }}));
      for (let i = 0; i <= 4; i++) {{
        const yy = pad.top + i * plotH / 4;
        svg.appendChild(el("line", {{ x1: pad.left, y1: yy, x2: width - pad.right, y2: yy, stroke: "#263548" }}));
        const label = minY + (1 - i / 4) * (maxY - minY);
        const t = el("text", {{ x: pad.left - 10, y: yy + 4, "text-anchor": "end", fill: "#9fb1c8", "font-size": 12 }});
        t.textContent = `$${{price(label)}}`;
        svg.appendChild(t);
      }}
      svg.appendChild(el("line", {{ x1: pad.left, y1: pad.top, x2: pad.left, y2: height - pad.bottom, stroke: "#304157" }}));
      svg.appendChild(el("line", {{ x1: pad.left, y1: height - pad.bottom, x2: width - pad.right, y2: height - pad.bottom, stroke: "#304157" }}));
      const candleWidth = Math.max(2, Math.min(12, plotW / Math.max(candles.length, 1) * .62));
      if (chartPrefs.showCandles && !sparseActual) {{
        candles.forEach(c => {{
          const up = c.c >= c.o;
          const color = up ? "#2ed48f" : "#ff5c74";
          const cx = x(c.t);
          const top = y(Math.max(c.o, c.c));
          const bottom = y(Math.min(c.o, c.c));
          svg.appendChild(el("line", {{ x1: cx, x2: cx, y1: y(c.h), y2: y(c.l), stroke: color, "stroke-width": 1.2 }}));
          svg.appendChild(el("rect", {{ x: cx - candleWidth / 2, y: top, width: candleWidth, height: Math.max(1, bottom - top), fill: color, opacity: .78 }}));
        }});
      }}
      if (sparseActual) {{
        closePoints.forEach(p => {{
          svg.appendChild(el("circle", {{ cx: x(p.t), cy: y(p.y), r: 3.8, fill: "#dbe7f6", opacity: .82 }}));
        }});
      }}
      if (chartPrefs.showLine && closePoints.length > 1 && !longSparseRange) {{
        segmentedLinePaths(closePoints, maxActualLineGapMs(chartPrefs.range, sparseActual)).forEach(segment => {{
          if (segment.length > 1) svg.appendChild(el("path", {{ d: linePath(segment, x, y), fill: "none", stroke: "#dbe7f6", "stroke-width": 2, opacity: .86 }}));
        }});
      }}
      if (chartPrefs.showSma9 && sma9.length) {{
        svg.appendChild(el("path", {{ d: linePath(sma9, x, y), fill: "none", stroke: "#f6b84b", "stroke-width": 2 }}));
        const label = el("text", {{ x: width - pad.right - 70, y: pad.top + 18, fill: "#f6b84b", "font-size": 12 }});
        label.textContent = "SMA 9";
        svg.appendChild(label);
      }}
      if (chartPrefs.showSma21 && sma21.length) {{
        svg.appendChild(el("path", {{ d: linePath(sma21, x, y), fill: "none", stroke: "#2ed48f", "stroke-width": 2 }}));
        const label = el("text", {{ x: width - pad.right - 70, y: pad.top + 36, fill: "#2ed48f", "font-size": 12 }});
        label.textContent = "SMA 21";
        svg.appendChild(label);
      }}
      if (hasForecastContext) {{
        svg.appendChild(el("line", {{ x1: x(asOfTime), x2: x(asOfTime), y1: pad.top, y2: height - pad.bottom, stroke: "#6aa2ff", "stroke-width": 1.5, "stroke-dasharray": "3 4" }}));
      }}
      const band = forecasts.filter(p => Number.isFinite(p.lo) && Number.isFinite(p.hi));
      if (band.length && hasForecastContext) {{
        const upper = [{{t: asOfTime, y: asOfPrice}}].concat(band.map(p => ({{t: p.x, y: p.hi}})));
        const lower = [{{t: asOfTime, y: asOfPrice}}].concat(band.map(p => ({{t: p.x, y: p.lo}}))).reverse();
        svg.appendChild(el("path", {{ d: upper.concat(lower).map((p, i) => `${{i ? "L" : "M"}}${{x(p.t).toFixed(2)}},${{y(p.y).toFixed(2)}}`).join(" ") + "Z", fill: "rgba(255,92,116,0.18)", stroke: "none" }}));
      }}
      const forecastLine = [];
      if (hasForecastContext) forecastLine.push({{ t: asOfTime, y: asOfPrice }});
      forecasts.forEach(p => forecastLine.push({{ t: p.x, y: p.y }}));
      if (forecastLine.length > 1) {{
        svg.appendChild(el("path", {{ d: linePath(forecastLine, x, y), fill: "none", stroke: "#ff5c74", "stroke-width": 2.4, "stroke-dasharray": "7 5" }}));
      }}
      if (hasForecastContext) {{
        svg.appendChild(el("circle", {{ cx: x(asOfTime), cy: y(asOfPrice), r: 5.5, fill: "#6aa2ff" }}));
      }}
      if (latest) {{
        svg.appendChild(el("circle", {{ cx: x(latest.t), cy: y(latest.c), r: 4.5, fill: "#dbe7f6" }}));
        const latestLabel = el("text", {{ x: Math.min(x(latest.t) + 8, width - pad.right - 110), y: y(latest.c) + 4, fill: "#dbe7f6", "font-size": 12 }});
        latestLabel.textContent = `latest $${{price(latest.c)}}`;
        svg.appendChild(latestLabel);
      }}
      forecasts.forEach((p, i) => {{
        const matured = Number.isFinite(latestTime) && p.x <= latestTime;
        const fill = matured ? "#ff5c74" : "#f59e0b";
        svg.appendChild(el("circle", {{ cx: x(p.x), cy: y(p.y), r: 5.5, fill }}));
        const tx = Math.min(width - pad.right - 80, x(p.x) + 10);
        const ty = y(p.y) + (i % 2 === 0 ? -24 : 26);
        const label = el("text", {{ x: tx, y: ty, fill: "#aebbd0", "font-size": 12 }});
        label.textContent = horizonLabel(p.horizon_hours);
        svg.appendChild(label);
        const value = el("text", {{ x: tx, y: ty + 17, fill: "#ffffff", "font-size": 12, "font-weight": 800 }});
        value.textContent = `$${{price(p.y)}}${{matured ? " checked" : ""}}`;
        svg.appendChild(value);
      }});
      const minLabel = el("text", {{ x: pad.left, y: height - 18, fill: "#9fb1c8", "font-size": 12 }});
      minLabel.textContent = new Date(minX).toLocaleString();
      svg.appendChild(minLabel);
      const maxLabel = el("text", {{ x: width - pad.right, y: height - 18, "text-anchor": "end", fill: "#9fb1c8", "font-size": 12 }});
      maxLabel.textContent = new Date(maxX).toLocaleString();
      svg.appendChild(maxLabel);
    }}
    function linePath(points, x, y) {{
      return points.map((p, i) => `${{i ? "L" : "M"}}${{x(p.t).toFixed(2)}},${{y(p.y).toFixed(2)}}`).join(" ");
    }}
    function segmentedLinePaths(points, maxGapMs) {{
      if (!points.length) return [];
      const segments = [[points[0]]];
      for (let i = 1; i < points.length; i++) {{
        const previous = points[i - 1];
        const current = points[i];
        if (Math.abs(current.t - previous.t) > maxGapMs) {{
          segments.push([current]);
        }} else {{
          segments.at(-1).push(current);
        }}
      }}
      return segments;
    }}
    function maxActualLineGapMs(range, sparseActual) {{
      if (!sparseActual) return rangeToMs(range);
      if (range === "1h") return 12 * 60 * 1000;
      if (range === "1d") return 45 * 60 * 1000;
      if (range === "2d") return 90 * 60 * 1000;
      return 3 * 60 * 60 * 1000;
    }}
    function rangeToMs(range) {{
      return ({{ "1h": 1, "1d": 24, "2d": 48, "1m": 24 * 30, "2m": 24 * 60, "3m": 24 * 90, "6m": 24 * 180 }}[range] || 1) * 3600000;
    }}
    function candleBucketMinutes(range) {{
      if (range === "1h") return 5;
      if (range === "1d" || range === "2d") return 15;
      return 60;
    }}
    function aggregateCandles(rows, minutes) {{
      const bucketMs = minutes * 60 * 1000;
      const buckets = new Map();
      for (const r of rows) {{
        const key = Math.floor(r.t / bucketMs) * bucketMs;
        const b = buckets.get(key);
        if (!b) buckets.set(key, {{ t: key, o: r.o, h: r.h, l: r.l, c: r.c, v: r.v || 0 }});
        else {{
          b.h = Math.max(b.h, r.h);
          b.l = Math.min(b.l, r.l);
          b.c = r.c;
          b.v += r.v || 0;
        }}
      }}
      return Array.from(buckets.values()).sort((a, b) => a.t - b.t);
    }}
    function movingAverage(points, window) {{
      const out = [];
      for (let i = window - 1; i < points.length; i++) {{
        const slice = points.slice(i - window + 1, i + 1);
        out.push({{ t: points[i].t, y: slice.reduce((sum, p) => sum + p.y, 0) / window }});
      }}
      return out;
    }}
    function horizonLabel(hours) {{
      const h = Number(hours);
      if (!Number.isFinite(h)) return "-";
      const minutes = Math.round(h * 60);
      if (minutes < 60) return `${{minutes}}m`;
      return `${{Number.isInteger(h) ? h.toFixed(0) : h.toFixed(2)}}h`;
    }}
    function el(name, attrs) {{
      const node = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.entries(attrs || {{}}).forEach(([key, value]) => node.setAttribute(key, value));
      return node;
    }}
    function parseTimestampMs(value) {{
      const raw = String(value || "");
      if (!raw) return NaN;
      const hasZone = /(?:Z|[+-]\\d{{2}}:?\\d{{2}})$/.test(raw);
      return new Date(hasZone ? raw : `${{raw}}Z`).getTime();
    }}
    function renderPositionPl(pl) {{
      const total = Number(pl.total_unrealized_pl || 0);
      const status = pl.status || "flat";
      const statusEl = document.getElementById("plStatus");
      statusEl.textContent = status === "winning" ? "Winning" : status === "losing" ? "Losing" : "Flat";
      statusEl.className = `value ${{status === "winning" ? "win" : status === "losing" ? "loss" : ""}}`;
      const totalEl = document.getElementById("plTotal");
      totalEl.textContent = money(total);
      totalEl.className = `value ${{total > 0 ? "win" : total < 0 ? "loss" : ""}}`;
      set("plCost", `$${{price(pl.total_cost)}}`);
      set("plValue", `$${{price(pl.total_value)}}`);
      const rows = document.getElementById("plRows");
      rows.innerHTML = "";
      (pl.rows || []).forEach(row => {{
        const rowPl = Number(row.unrealized_pl || 0);
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${{row.symbol || "-"}}</td><td>${{row.option_type || optionTypeLabel(null, row.symbol) || "-"}}</td><td>${{price(row.qty)}}</td><td>$${{price(row.avg_entry_price)}}</td><td>$${{price(row.current_price)}}</td><td>$${{price(row.cost_basis)}}</td><td>$${{price(row.market_value)}}</td><td class="${{rowPl > 0 ? "win" : rowPl < 0 ? "loss" : ""}}">${{money(rowPl)}}</td>`;
        rows.appendChild(tr);
      }});
      if (!(pl.rows || []).length) {{
        const tr = document.createElement("tr");
        tr.innerHTML = `<td colspan="8">No open option positions in the latest report.</td>`;
        rows.appendChild(tr);
      }}
    }}
    function setMoneyWithClass(id, value) {{
      const el = document.getElementById(id);
      el.textContent = money(value);
      el.className = `value ${{value > 0 ? "win" : value < 0 ? "loss" : ""}}`;
    }}
    function optionTypeLabel(optionType, symbol) {{
      const raw = String(optionType || "").toLowerCase();
      if (raw === "call") return "CALL";
      if (raw === "put") return "PUT";
      const s = String(symbol || "").toUpperCase();
      if (s.length >= 15) {{
        const code = s.slice(-9, -8);
        if (code === "C") return "CALL";
        if (code === "P") return "PUT";
      }}
      return null;
    }}
    async function refresh() {{
      try {{
        const response = await fetch(`/api/state${{window.location.search}}`, {{ cache: "no-store" }});
        render(await response.json());
      }} catch (error) {{
        set("details", `dashboard_refresh_failed: ${{error}}`);
      }}
    }}
    function rerenderChart() {{
      if (lastDashboardData) renderStockChart(lastDashboardData.chart || {{}});
    }}
    function initChartControls() {{
      document.querySelectorAll("[data-range]").forEach(button => {{
        button.addEventListener("click", () => {{
          chartPrefs.range = button.dataset.range || "1h";
          chartPrefs.forecastZoom = false;
          chartPrefs.zoomFactor = 1;
          document.querySelectorAll("[data-range]").forEach(b => b.classList.toggle("active", b === button));
          document.getElementById("forecastZoom").classList.remove("active");
          rerenderChart();
        }});
      }});
      document.getElementById("forecastZoom").addEventListener("click", event => {{
        chartPrefs.forecastZoom = !chartPrefs.forecastZoom;
        chartPrefs.zoomFactor = 1;
        event.currentTarget.classList.toggle("active", chartPrefs.forecastZoom);
        rerenderChart();
      }});
      document.getElementById("zoomIn").addEventListener("click", () => {{
        chartPrefs.zoomFactor = Math.min(chartPrefs.zoomFactor * 1.5, 12);
        rerenderChart();
      }});
      document.getElementById("zoomOut").addEventListener("click", () => {{
        chartPrefs.zoomFactor = Math.max(chartPrefs.zoomFactor / 1.5, 0.25);
        rerenderChart();
      }});
      [
        ["showLine", "showLine"],
        ["showCandles", "showCandles"],
        ["showSma9", "showSma9"],
        ["showSma21", "showSma21"],
      ].forEach(([id, key]) => {{
        const input = document.getElementById(id);
        input.addEventListener("change", () => {{
          chartPrefs[key] = input.checked;
          rerenderChart();
        }});
      }});
    }}
    initChartControls();
    refresh();
    setInterval(refresh, refreshMs);
  </script>
</body>
</html>"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a generic Alpaca paper-options agent dashboard.")
    parser.add_argument("--ticker", default="TSLA")
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--refresh-seconds", type=int, default=DEFAULT_REFRESH_SECONDS)
    parser.add_argument("--max-history", type=int, default=50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = build_server(args)
    print(f"Serving options dashboard at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


def build_server(args: argparse.Namespace) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(dashboard_html(int(args.refresh_seconds)))
                return
            if parsed.path == "/api/state":
                params = parse_qs(parsed.query)
                ticker = _query_value(params, "ticker", args.ticker)
                state = build_dashboard_state(
                    state_dir=Path(_query_value(params, "state_dir", args.state_dir)),
                    ticker=ticker,
                    max_history=int(_query_value(params, "max_history", str(args.max_history))),
                )
                self._send_json(state)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, format: str, *values: Any) -> None:
            return

        def _send_html(self, payload: str) -> None:
            body = payload.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(_strict_json_value(payload), default=str, allow_nan=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((args.host, int(args.port)), Handler)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    rows.append(parsed)
    except OSError:
        return []
    return rows


def _strict_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, float):
        return None if pd.isna(value) else value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _option_type_label(option_type: Any, symbol: Any) -> str | None:
    raw_type = str(option_type or "").strip().lower()
    if raw_type == "call":
        return "CALL"
    if raw_type == "put":
        return "PUT"
    raw_symbol = str(symbol or "").upper()
    if len(raw_symbol) >= 15:
        option_code = raw_symbol[-9:-8]
        if option_code == "C":
            return "CALL"
        if option_code == "P":
            return "PUT"
    return None


def _query_value(params: dict[str, list[str]], key: str, default: str) -> str:
    values = params.get(key)
    if not values:
        return default
    return values[0] or default


if __name__ == "__main__":
    main()
