from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from market_forecasting_engine.market_history import deribit_currency_to_yahoo_symbol, fetch_yahoo_market_history


DEFAULT_STATE_DIR = Path("automated_forecasting_engine/runs/deribit_options_agent")
DEFAULT_REFRESH_SECONDS = 30


def build_dashboard_state(*, state_dir: Path, currency: str, max_history: int = 100) -> dict[str, Any]:
    report, report_path = read_latest_report(state_dir, currency)
    state, state_path = read_state(state_dir, currency)
    history, log_path = read_history(state_dir, currency, max_history=max_history)
    trade_plan = report.get("option_trade_plan") or {}
    selected = trade_plan.get("selected_contract") or {}
    order = trade_plan.get("order") or {}
    exit_plan = trade_plan.get("exit_plan") or {}
    spot = _to_float((report.get("selected_forecast") or {}).get("spot"))
    position_pl = summarize_position_pl(report.get("option_positions") or [], underlying_price_usd=spot)
    fibonacci = trade_plan.get("fibonacci_analysis") or (report.get("selected_forecast") or {}).get("fibonacci_analysis") or {}
    forecast_chart = build_forecast_chart(history=history, latest_report=report)
    report_age_seconds = _age_seconds(report.get("checked_at"))
    stop_request = _active_stop_request(state.get("stop_request"))
    entry_blocks = report.get("execution_blocks") or []
    management_actions = report.get("management_actions") or []
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "venue": "deribit_testnet",
        "currency": currency.upper(),
        "state_dir": str(state_dir),
        "report_path": str(report_path) if report_path else None,
        "state_path": str(state_path),
        "log_path": str(log_path) if log_path else None,
        "report": report,
        "state": state,
        "summary": {
            "checked_at": report.get("checked_at"),
            "report_age_seconds": report_age_seconds,
            "report_stale": report_age_seconds is None or report_age_seconds > 180,
            "agent_stop_request": stop_request,
            "forecast_cache_status": report.get("forecast_cache_status"),
            "forecast_direction": (report.get("selected_forecast") or {}).get("expected_direction"),
            "spot": (report.get("selected_forecast") or {}).get("spot"),
            "predicted_price": (report.get("selected_forecast") or {}).get("predicted_price"),
            "action": trade_plan.get("action"),
            "reason": trade_plan.get("reason"),
            "option_type": trade_plan.get("option_type"),
            "contract": selected.get("instrument_name"),
            "strike": selected.get("strike"),
            "dte": selected.get("dte"),
            "expiration_utc": selected.get("expiration_utc"),
            "hours_to_expiry": selected.get("hours_to_expiry"),
            "required_hours_to_expiry_for_entry": selected.get("required_hours_to_expiry_for_entry"),
            "bid": selected.get("bid"),
            "ask": selected.get("ask"),
            "spread_pct": selected.get("spread_pct"),
            "greeks": selected.get("greeks") or {},
            "fibonacci": fibonacci,
            "liquidity": selected.get("liquidity") or {},
            "entry_price_base": order.get("price"),
            "amount": order.get("amount"),
            "take_profit_base": (exit_plan.get("take_profit") or {}).get("price"),
            "stop_loss_base": (exit_plan.get("stop_loss") or {}).get("price"),
            "estimated_debit_usd": (trade_plan.get("risk") or {}).get("estimated_debit_usd"),
            "estimated_debit_base": (trade_plan.get("risk") or {}).get("estimated_debit_base"),
            "account": report.get("account") or {},
            "execution_blocks": entry_blocks,
            "entry_blocks": entry_blocks,
            "management_actions": management_actions,
            "position_management_active": bool(management_actions),
            "open_order_count": len(report.get("open_option_orders") or []),
            "position_count": len(report.get("option_positions") or []),
            "order_submitted": (report.get("order_result") or {}).get("submitted"),
            "order_result": report.get("order_result"),
            "position_pl": position_pl,
            "risk_controls": report.get("risk_control_config") or {},
            "feedback": report.get("feedback_context") or {},
        },
        "open_option_orders": report.get("open_option_orders") or [],
        "option_positions": report.get("option_positions") or [],
        "history": history,
        "forecast_chart": forecast_chart,
    }


def build_forecast_chart(*, history: list[dict[str, Any]], latest_report: dict[str, Any]) -> dict[str, Any]:
    actual_by_time: dict[str, dict[str, Any]] = {}
    forecast_points_by_key: dict[tuple[str, str, float | None], dict[str, Any]] = {}
    as_of_points: dict[str, dict[str, Any]] = {}
    rows = [*history]
    if latest_report and (not rows or rows[-1].get("checked_at") != latest_report.get("checked_at")):
        rows.append(latest_report)
    for row in rows:
        checked_at = _iso_or_none(row.get("checked_at"))
        selected = row.get("selected_forecast") or {}
        plan = row.get("forecast_plan") or {}
        spot = _to_float(selected.get("spot"))
        if spot is None:
            spot = _to_float(plan.get("latest_price"))
        if checked_at and spot is not None:
            actual_by_time[checked_at] = {"time": checked_at, "price": spot}
        created_at = _iso_or_none(row.get("forecast_created_at_utc") or (row.get("last_forecast") or {}).get("created_at_utc") or checked_at)
        if created_at and spot is not None:
            as_of_points[created_at] = {"time": created_at, "price": spot}
        created_dt = _parse_time(created_at)
        for forecast in plan.get("forecasts") or []:
            predicted = _to_float(forecast.get("predicted_price"))
            horizon = _to_float(forecast.get("horizon_hours"))
            target_time = _forecast_target_time(created_dt=created_dt, horizon_hours=horizon)
            if target_time is None:
                target_time = _parse_time(forecast.get("forecast_timestamp"))
            if created_at and target_time and predicted is not None:
                target_iso = target_time.isoformat()
                forecast_points_by_key[(created_at, target_iso, horizon)] = {
                    "created_at": created_at,
                    "target_time": target_iso,
                    "predicted_price": predicted,
                    "horizon_hours": horizon,
                    "lower_price": _to_float(forecast.get("lower_price")),
                    "upper_price": _to_float(forecast.get("upper_price")),
                }
    actual = sorted(actual_by_time.values(), key=lambda point: point["time"])[-240:]
    forecast_points = list(forecast_points_by_key.values())
    latest_created = None
    for point in forecast_points:
        latest_created = point["created_at"] if latest_created is None else max(latest_created, point["created_at"])
    latest_forecast = [point for point in forecast_points if point["created_at"] == latest_created] if latest_created else []
    latest_forecast.sort(key=lambda point: (point.get("horizon_hours") is None, point.get("horizon_hours") or 0.0, point["target_time"]))
    as_of = as_of_points.get(latest_created) if latest_created else None
    return {
        "actual": actual,
        "latest_forecast": latest_forecast,
        "as_of": as_of,
        "latest_created_at": latest_created,
        "actual_count": len(actual),
        "forecast_count": len(latest_forecast),
    }


def summarize_position_pl(positions: list[dict[str, Any]], *, underlying_price_usd: float | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_cost_base = 0.0
    total_value_base = 0.0
    total_pl_base = 0.0
    winners = 0
    losers = 0
    for position in positions:
        instrument = str(position.get("instrument_name") or "")
        if not instrument:
            continue
        size = abs(_to_float(position.get("size")) or _to_float(position.get("amount")) or 0.0)
        avg_entry = _to_float(position.get("average_price"))
        mark = _to_float(position.get("mark_price"))
        pl_base = _to_float(position.get("floating_profit_loss"))
        if pl_base is None:
            pl_base = _to_float(position.get("total_profit_loss"))
        cost_base = size * avg_entry if avg_entry is not None else None
        value_base = size * mark if mark is not None else None
        if pl_base is None and cost_base is not None and value_base is not None:
            pl_base = value_base - cost_base
        total_cost_base += cost_base or 0.0
        total_value_base += value_base or 0.0
        total_pl_base += pl_base or 0.0
        if (pl_base or 0.0) > 0:
            winners += 1
        elif (pl_base or 0.0) < 0:
            losers += 1
        rows.append(
            {
                "instrument_name": instrument,
                "size": size,
                "average_price": avg_entry,
                "mark_price": mark,
                "cost_base": cost_base,
                "value_base": value_base,
                "floating_profit_loss_base": pl_base,
                "floating_profit_loss_usd": None if pl_base is None or underlying_price_usd is None else pl_base * underlying_price_usd,
                "expiration_utc": position.get("expiration_utc"),
                "status": "winning" if (pl_base or 0.0) > 0 else "losing" if (pl_base or 0.0) < 0 else "flat",
            }
        )
    status = "flat"
    if total_pl_base > 0:
        status = "winning"
    elif total_pl_base < 0:
        status = "losing"
    return {
        "status": status,
        "winning_positions": winners,
        "losing_positions": losers,
        "flat_positions": max(0, len(rows) - winners - losers),
        "total_cost_base": round(total_cost_base, 8),
        "total_value_base": round(total_value_base, 8),
        "total_unrealized_pl_base": round(total_pl_base, 8),
        "total_cost_usd": None if underlying_price_usd is None else round(total_cost_base * underlying_price_usd, 2),
        "total_value_usd": None if underlying_price_usd is None else round(total_value_base * underlying_price_usd, 2),
        "total_unrealized_pl_usd": None if underlying_price_usd is None else round(total_pl_base * underlying_price_usd, 2),
        "underlying_price_usd": underlying_price_usd,
        "rows": rows,
        "note": "Temporary open P/L view. These are unrealized marks for open Deribit testnet option positions, not final closed-trade results.",
    }


def _age_seconds(value: Any) -> float | None:
    parsed = _parse_time(value)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - parsed.astimezone(UTC)).total_seconds())


def _active_stop_request(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    path = value.get("path")
    if path and not Path(path).exists():
        return None
    return value


def read_latest_report(state_dir: Path, currency: str) -> tuple[dict[str, Any], Path | None]:
    path = state_dir / f"{currency.upper()}_deribit_options_agent_report.json"
    if not path.exists():
        return {}, None
    return _read_json(path), path


def read_state(state_dir: Path, currency: str) -> tuple[dict[str, Any], Path]:
    path = state_dir / "state" / f"{currency.upper()}_deribit_options_agent_state.json"
    if not path.exists():
        return {}, path
    return _read_json(path), path


def read_history(state_dir: Path, currency: str, *, max_history: int) -> tuple[list[dict[str, Any]], Path | None]:
    paths = sorted((state_dir / "logs").glob(f"{currency.upper()}_*.jsonl"))
    rows: list[dict[str, Any]] = []
    latest_path = None
    for path in paths:
        latest_path = path
        rows.extend(_read_jsonl(path))
    return rows[-max_history:], latest_path


def _forecast_target_time(*, created_dt: datetime | None, horizon_hours: float | None) -> datetime | None:
    if created_dt is None or horizon_hours is None:
        return None
    return created_dt + timedelta(hours=float(horizon_hours))


def dashboard_html(refresh_seconds: int) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Deribit Options Agent</title>
  <style>
    :root {{ --bg:#f4f6f8; --panel:#fff; --text:#17202a; --muted:#5d6b7c; --line:#d8dee8; --good:#067647; --warn:#7a4d00; --bad:#b42318; --accent:#1d4ed8; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--text); }}
    header {{ display:flex; justify-content:space-between; gap:16px; padding:18px 22px; background:var(--panel); border-bottom:1px solid var(--line); position:sticky; top:0; z-index:2; }}
    h1 {{ margin:0; font-size:20px; }}
    main {{ max-width:1360px; margin:0 auto; padding:18px 22px 30px; }}
    .small,.meta {{ color:var(--muted); font-size:13px; }}
    .cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:14px; }}
    .card,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
    .label {{ color:var(--muted); font-size:12px; margin-bottom:5px; }}
    .value {{ font-size:22px; font-weight:750; overflow-wrap:anywhere; }}
    .badge {{ display:inline-block; border-radius:999px; padding:5px 10px; font-size:12px; font-weight:750; color:white; background:var(--warn); text-transform:uppercase; }}
    .badge.buy_option {{ background:var(--good); }} .badge.blocked {{ background:var(--bad); }}
    .win {{ color:var(--good); }}
    .loss {{ color:var(--bad); }}
    .grid2 {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(420px,.85fr); gap:14px; margin-bottom:14px; }}
    .chartWrap {{ width:100%; height:460px; }}
    svg {{ display:block; width:100%; height:100%; }}
    .axis {{ stroke:#9aa4b2; stroke-width:1; }}
    .grid {{ stroke:#e5e9f0; stroke-width:1; }}
    .actualLine {{ fill:none; stroke:#233044; stroke-width:2; }}
    .forecastLine {{ fill:none; stroke:#dc2626; stroke-width:2; stroke-dasharray:7 5; }}
    .forecastBand {{ fill:#dc2626; opacity:.10; }}
    .asOf {{ stroke:#1d4ed8; stroke-width:1.5; stroke-dasharray:3 4; }}
    .dotActual {{ fill:#233044; }}
    .dotForecast {{ fill:#dc2626; }}
    .dotAsOf {{ fill:#1d4ed8; }}
    .chartLabel {{ fill:#526071; font-size:12px; }}
    .chartValue {{ fill:#17202a; font-size:12px; font-weight:650; }}
    .callout {{ stroke:#9aa4b2; stroke-width:1; }}
    .chartLegend {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:8px; }}
    .legendItem {{ display:inline-flex; align-items:center; gap:6px; color:var(--muted); font-size:13px; }}
    .legendSwatch {{ width:18px; height:3px; display:inline-block; background:#233044; }}
    .legendSwatch.forecast {{ background:#dc2626; border-top:1px dashed #dc2626; }}
    .legendSwatch.asof {{ background:#1d4ed8; }}
    .rangeControls {{ display:flex; gap:8px; flex-wrap:wrap; margin:10px 0 12px; }}
    .rangeButton {{ border:1px solid var(--line); background:#fff; color:var(--muted); border-radius:7px; padding:7px 10px; font-weight:700; cursor:pointer; }}
    .rangeButton.active {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ padding:8px 6px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
    th {{ color:var(--muted); font-weight:650; }}
    pre {{ margin:0; white-space:pre-wrap; overflow-wrap:anywhere; font-size:12px; max-height:640px; overflow:auto; }}
    .bad {{ color:var(--bad); font-weight:650; }}
    @media (max-width:980px) {{ header {{ flex-direction:column; }} .cards,.grid2 {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <div><h1>Deribit Crypto Options Agent</h1><div class="small" id="source">Loading...</div></div>
    <div class="meta"><div id="updated">Loading...</div><div>Refresh: {refresh_seconds}s</div></div>
  </header>
  <main>
    <section class="cards">
      <div class="card"><div class="label">Decision</div><div class="value"><span id="actionBadge" class="badge">-</span></div><div class="small" id="blocks">-</div></div>
      <div class="card"><div class="label">Forecast</div><div class="value" id="forecast">-</div><div class="small" id="forecastMeta">-</div></div>
      <div class="card"><div class="label">Contract</div><div class="value" id="contract">-</div><div class="small" id="contractMeta">-</div></div>
      <div class="card"><div class="label">Entry</div><div class="value" id="entry">-</div><div class="small" id="risk">-</div></div>
    </section>
    <section class="cards">
      <div class="card"><div class="label">Take Profit</div><div class="value" id="takeProfit">-</div><div class="small">Agent-managed reduce-only limit</div></div>
      <div class="card"><div class="label">Stop Loss</div><div class="value" id="stopLoss">-</div><div class="small">Agent-managed reduce-only limit</div></div>
      <div class="card"><div class="label">Account</div><div class="value" id="account">-</div><div class="small" id="accountMeta">-</div></div>
      <div class="card"><div class="label">Orders / Positions</div><div class="value" id="exposure">-</div><div class="small" id="orderResult">-</div></div>
    </section>
    <section class="panel">
      <strong>Agent Health</strong>
      <div class="cards" style="margin:12px 0 8px;">
        <div class="card"><div class="label">Report Freshness</div><div class="value" id="reportFreshness">-</div><div class="small" id="reportFreshnessMeta">-</div></div>
        <div class="card"><div class="label">Stop Request</div><div class="value" id="stopRequest">-</div><div class="small" id="stopRequestMeta">-</div></div>
        <div class="card"><div class="label">Entry Blocks</div><div class="value" id="entryBlockStatus">-</div><div class="small" id="entryBlockMeta">-</div></div>
        <div class="card"><div class="label">Position Management</div><div class="value" id="positionManagement">-</div><div class="small" id="positionManagementMeta">-</div></div>
      </div>
    </section>
    <section class="panel">
      <strong>Market Price</strong>
      <div class="small" id="marketMeta" style="margin:6px 0 8px;">Loading...</div>
      <div class="rangeControls" id="marketRangeControls">
        <button class="rangeButton" data-range="1d">1D</button>
        <button class="rangeButton" data-range="2d">2D</button>
        <button class="rangeButton" data-range="1m">1M</button>
        <button class="rangeButton" data-range="2m">2M</button>
        <button class="rangeButton" data-range="3m">3M</button>
        <button class="rangeButton active" data-range="6m">6M</button>
      </div>
      <div class="chartWrap" id="marketChart"></div>
    </section>
    <section class="panel">
      <strong>Forecast vs Actual ETH/USD</strong>
      <div class="small" id="chartMeta" style="margin:6px 0 10px;">Loading...</div>
      <div class="chartWrap" id="forecastChart"></div>
      <table style="margin-top:10px;">
        <thead><tr><th>Horizon</th><th>Target Time</th><th>Forecast Price</th><th>Range</th></tr></thead>
        <tbody id="forecastPointRows"></tbody>
      </table>
      <div class="chartLegend">
        <span class="legendItem"><span class="legendSwatch"></span>Actual spot from agent checks</span>
        <span class="legendItem"><span class="legendSwatch forecast"></span>Latest forecast path</span>
        <span class="legendItem"><span class="legendSwatch asof"></span>Forecast generated</span>
      </div>
    </section>
    <section class="panel">
      <strong>Feedback Loop</strong>
      <div class="cards" style="margin:12px 0 8px;">
        <div class="card"><div class="label">Status</div><div class="value" id="feedbackStatus">-</div><div class="small" id="feedbackBlocks">-</div></div>
        <div class="card"><div class="label">Direction Accuracy</div><div class="value" id="feedbackAccuracy">-</div><div class="small" id="feedbackMatured">-</div></div>
        <div class="card"><div class="label">Price Error</div><div class="value" id="feedbackError">-</div><div class="small">Average absolute percent error</div></div>
        <div class="card"><div class="label">Ledger</div><div class="value" id="feedbackLedger">-</div><div class="small" id="feedbackLedgerPath">-</div></div>
      </div>
    </section>
    <section class="panel">
      <strong>Selected Contract Greeks</strong>
      <div class="cards" style="margin:12px 0 8px;">
        <div class="card"><div class="label">Delta</div><div class="value" id="greekDelta">-</div><div class="small">Directional exposure</div></div>
        <div class="card"><div class="label">Gamma</div><div class="value" id="greekGamma">-</div><div class="small">Delta acceleration</div></div>
        <div class="card"><div class="label">Theta</div><div class="value" id="greekTheta">-</div><div class="small" id="thetaMeta">Time decay</div></div>
        <div class="card"><div class="label">Vega / Liquidity</div><div class="value" id="greekVega">-</div><div class="small" id="liquidityMeta">-</div></div>
      </div>
    </section>
    <section class="panel">
      <strong>Fibonacci Analysis</strong>
      <div class="cards" style="margin:12px 0 8px;">
        <div class="card"><div class="label">Confirmation</div><div class="value" id="fibConfirmation">-</div><div class="small" id="fibReason">-</div></div>
        <div class="card"><div class="label">Swing Range</div><div class="value" id="fibSwing">-</div><div class="small" id="fibTrend">-</div></div>
        <div class="card"><div class="label">Nearest Levels</div><div class="value" id="fibLevels">-</div><div class="small">support / resistance</div></div>
        <div class="card"><div class="label">Next Target</div><div class="value" id="fibTarget">-</div><div class="small">nearest Fib level in forecast direction</div></div>
      </div>
    </section>
    <section class="panel">
      <strong>Open Winning / Losing Positions</strong>
      <div class="cards" style="margin:12px 0 8px;">
        <div class="card"><div class="label">Plain English</div><div class="value" id="plStatus">-</div><div class="small" id="plGuardMeta">Open marks, not final results</div></div>
        <div class="card"><div class="label">Total P/L</div><div class="value" id="plTotal">-</div><div class="small" id="plTotalBase">-</div></div>
        <div class="card"><div class="label">Win / Loss Count</div><div class="value" id="plCounts">-</div><div class="small">Positions now</div></div>
        <div class="card"><div class="label">Cost / Value</div><div class="value" id="plCostValue">-</div><div class="small">USD equivalent</div></div>
      </div>
      <table>
        <thead><tr><th>Position</th><th>Size</th><th>Entry</th><th>Mark</th><th>Expires</th><th>Cost</th><th>Value</th><th>P/L</th><th>Status</th></tr></thead>
        <tbody id="plRows"></tbody>
      </table>
    </section>
    <section class="grid2">
      <div class="panel">
        <strong>Recent Decisions</strong>
        <table><thead><tr><th>Checked</th><th>Action</th><th>Contract</th><th>Entry</th><th>Debit</th><th>Blocks</th><th>Submitted</th></tr></thead><tbody id="historyRows"></tbody></table>
      </div>
      <div class="panel"><strong>Details</strong><pre id="details">Loading...</pre></div>
    </section>
    <section class="grid2">
      <div class="panel"><strong>Open Orders</strong><table><thead><tr><th>Instrument</th><th>Side</th><th>Amount</th><th>Price</th><th>State</th></tr></thead><tbody id="ordersRows"></tbody></table></div>
      <div class="panel"><strong>Positions</strong><table><thead><tr><th>Instrument</th><th>Size</th><th>Avg</th><th>Mark</th><th>Expires</th><th>P/L</th></tr></thead><tbody id="positionsRows"></tbody></table></div>
    </section>
  </main>
  <script>
    const refreshMs = {refresh_seconds * 1000};
    let marketRange = "6m";
    let currentCurrency = "ETH";
    const fmt = new Intl.NumberFormat(undefined, {{ maximumFractionDigits: 6 }});
    const usd = new Intl.NumberFormat(undefined, {{ style: "currency", currency: "USD", maximumFractionDigits: 2 }});
    function num(v) {{ return v === null || v === undefined || Number.isNaN(Number(v)) ? "-" : fmt.format(Number(v)); }}
    function money(v) {{ return v === null || v === undefined || Number.isNaN(Number(v)) ? "-" : usd.format(Number(v)); }}
    function time(v) {{ if (!v) return "-"; const d = new Date(v); return Number.isNaN(d.getTime()) ? v : d.toLocaleString(); }}
    function set(id, value) {{ document.getElementById(id).textContent = value; }}
    document.querySelectorAll(".rangeButton").forEach(button => button.addEventListener("click", () => {{
      marketRange = button.dataset.range || "6m";
      document.querySelectorAll(".rangeButton").forEach(item => item.classList.toggle("active", item.dataset.range === marketRange));
      refreshMarket();
    }}));
    function render(data) {{
      const s = data.summary || {{}};
      const rc = s.risk_controls || {{}};
      const blocked = (s.execution_blocks || []).length > 0;
      set("source", `${{data.currency}} options / ${{data.venue}} / ${{data.state_dir}}`);
      currentCurrency = data.currency || currentCurrency;
      set("updated", `Updated ${{time(data.generated_at)}}`);
      const age = s.report_age_seconds;
      set("reportFreshness", s.report_stale ? "STALE" : "Fresh");
      set("reportFreshnessMeta", age === null || age === undefined ? "no checked_at in report" : `last broker check ${{num(age)}} seconds ago at ${{time(s.checked_at)}}`);
      const stop = s.agent_stop_request || null;
      set("stopRequest", stop ? "ACTIVE" : "Clear");
      set("stopRequestMeta", stop ? `${{stop.reason || "stop requested"}} at ${{time(stop.requested_at)}}` : "agent is allowed to continue");
      const badge = document.getElementById("actionBadge");
      badge.textContent = blocked ? "blocked" : (s.action || "-");
      badge.className = `badge ${{blocked ? "blocked" : s.action || ""}}`;
      set("blocks", (s.execution_blocks || []).join(", ") || "no execution blocks");
      set("entryBlockStatus", (s.entry_blocks || []).length ? "New entry blocked" : "Entry free");
      set("entryBlockMeta", (s.entry_blocks || []).join(", ") || "no entry blocks");
      set("positionManagement", s.position_management_active ? "Active" : "No action");
      set("positionManagementMeta", (s.management_actions || []).map(a => a.action).join(", ") || "no position management action in latest check");
      set("forecast", `${{s.forecast_direction || "-"}} → ${{money(s.predicted_price)}}`);
      set("forecastMeta", `spot ${{money(s.spot)}} | cache ${{s.forecast_cache_status || "-"}}`);
      set("contract", s.contract || "-");
      const g = s.greeks || {{}};
      const fib = s.fibonacci || {{}};
      const feedback = s.feedback || {{}};
      const feedbackMetrics = feedback.metrics || {{}};
      const liq = s.liquidity || {{}};
      set("contractMeta", `${{s.option_type || "-"}} | strike ${{money(s.strike)}} | DTE ${{s.dte ?? "-"}} | hours left ${{num(s.hours_to_expiry)}} | expires ${{time(s.expiration_utc)}} | bid/ask ${{num(s.bid)}} / ${{num(s.ask)}}`);
      set("entry", `${{num(s.amount)}} @ ${{num(s.entry_price_base)}}`);
      set("risk", `debit ${{money(s.estimated_debit_usd)}} | spread ${{num((s.spread_pct || 0) * 100)}}% | delta ${{num(g.delta)}} | theta ${{num(g.theta)}}`);
      set("takeProfit", num(s.take_profit_base));
      set("stopLoss", num(s.stop_loss_base));
      set("greekDelta", num(g.delta));
      set("greekGamma", num(g.gamma));
      set("greekTheta", num(g.theta));
      set("thetaMeta", `decay horizon ${{money(g.theta_decay_usd_for_horizon)}} | premium/day ${{num((g.theta_premium_pct_per_day || 0) * 100)}}%`);
      set("greekVega", num(g.vega));
      set("liquidityMeta", `volume ${{num(liq.volume)}} | OI ${{num(liq.open_interest)}}`);
      set("fibConfirmation", fib.enabled === false ? "Off" : (fib.confirmation || fib.status || "-"));
      set("fibReason", fib.reason || "-");
      set("fibSwing", `${{money(fib.swing_low)}} / ${{money(fib.swing_high)}}`);
      set("fibTrend", `${{fib.trend || "-"}} | rows ${{fib.lookback_rows || "-"}}`);
      set("fibLevels", `${{money(fib.nearest_support)}} / ${{money(fib.nearest_resistance)}}`);
      set("fibTarget", money(fib.nearest_target_price));
      set("account", `${{num((s.account || {{}}).equity)}} ${{data.currency}}`);
      set("accountMeta", `available ${{num((s.account || {{}}).available_funds)}} | balance ${{num((s.account || {{}}).balance)}}`);
      set("exposure", `${{s.open_order_count || 0}} / ${{s.position_count || 0}}`);
      set("orderResult", s.order_submitted ? "order submitted" : ((s.entry_blocks || []).length ? "new entry blocked; management still checked" : "no order submitted"));
      set("plGuardMeta", `total win/loss ${{money(rc.max_total_unrealized_profit_usd)}} / ${{money(rc.max_total_unrealized_loss_usd)}} | position win/loss ${{money(rc.max_position_unrealized_profit_usd)}} / ${{money(rc.max_position_unrealized_loss_usd)}}`);
      set("feedbackStatus", feedback.enabled === false ? "Off" : ((feedback.blocks || []).length ? "Blocking" : "Active"));
      set("feedbackBlocks", (feedback.blocks || []).join(", ") || "no feedback blocks");
      set("feedbackAccuracy", feedbackMetrics.direction_accuracy === null || feedbackMetrics.direction_accuracy === undefined ? "-" : `${{num(feedbackMetrics.direction_accuracy * 100)}}%`);
      set("feedbackMatured", `${{feedbackMetrics.matured_horizon_count || 0}} matured horizons | ${{feedbackMetrics.decision_count || 0}} decisions`);
      set("feedbackError", feedbackMetrics.avg_abs_pct_error === null || feedbackMetrics.avg_abs_pct_error === undefined ? "-" : `${{num(feedbackMetrics.avg_abs_pct_error * 100)}}%`);
      set("feedbackLedger", `${{feedbackMetrics.submitted_decision_count || 0}} submitted`);
      set("feedbackLedgerPath", feedback.ledger_path || "-");
      renderForecastChart(data.forecast_chart || {{}});
      renderPositionPl(s.position_pl || {{}}, data.currency);
      renderHistory(data.history || []);
      renderOrders(data.open_option_orders || []);
      renderPositions(data.option_positions || []);
      set("details", JSON.stringify({{ summary:s, management_actions:s.management_actions, report_path:data.report_path, state_path:data.state_path, log_path:data.log_path, report:data.report }}, null, 2));
    }}
    async function refreshMarket() {{
      try {{
        const params = new URLSearchParams(window.location.search);
        params.set("currency", currentCurrency);
        params.set("range", marketRange);
        const response = await fetch(`/api/market?${{params.toString()}}`, {{ cache:"no-store" }});
        renderMarketChart(await response.json());
      }} catch (error) {{
        set("marketMeta", `market_refresh_failed: ${{error}}`);
      }}
    }}
    function renderMarketChart(payload) {{
      const host = document.getElementById("marketChart");
      host.innerHTML = "";
      const points = (payload.points || []).map(p => ({{ time:new Date(p.time), price:Number(p.close) }})).filter(p => !Number.isNaN(p.time.getTime()) && Number.isFinite(p.price));
      set("marketMeta", `${{payload.symbol || currentCurrency}} current ${{money(payload.current_price)}} | ${{(payload.range || marketRange).toUpperCase()}} | ${{payload.interval || "-"}} | ${{payload.point_count || 0}} points | source ${{payload.source || "-"}}`);
      if (!points.length) {{
        host.innerHTML = `<div class="small">No market price history available for this range.</div>`;
        return;
      }}
      const width = Math.max(host.clientWidth || 1000, 820);
      const height = 460;
      const margin = {{ left:72, right:34, top:28, bottom:54 }};
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      const minTime = Math.min(...points.map(p => p.time.getTime()));
      const maxTime = Math.max(...points.map(p => p.time.getTime()));
      const minRaw = Math.min(...points.map(p => p.price));
      const maxRaw = Math.max(...points.map(p => p.price));
      const pad = Math.max((maxRaw - minRaw) * 0.12, Math.abs(maxRaw) * 0.002, 1);
      const minPrice = minRaw - pad;
      const maxPrice = maxRaw + pad;
      const x = d => margin.left + ((d.getTime() - minTime) / Math.max(maxTime - minTime, 1)) * innerW;
      const y = price => margin.top + (1 - ((price - minPrice) / Math.max(maxPrice - minPrice, 1))) * innerH;
      const path = points.map((p, i) => `${{i ? "L" : "M"}}${{x(p.time).toFixed(1)}},${{y(p.price).toFixed(1)}}`).join(" ");
      let svg = `<svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Market price chart">`;
      for (let i = 0; i <= 4; i++) {{
        const yy = margin.top + (innerH / 4) * i;
        const price = maxPrice - ((maxPrice - minPrice) / 4) * i;
        svg += `<line class="grid" x1="${{margin.left}}" y1="${{yy}}" x2="${{width - margin.right}}" y2="${{yy}}"></line>`;
        svg += `<text class="chartLabel" x="8" y="${{yy + 4}}">${{money(price)}}</text>`;
      }}
      svg += `<line class="axis" x1="${{margin.left}}" y1="${{height - margin.bottom}}" x2="${{width - margin.right}}" y2="${{height - margin.bottom}}"></line>`;
      svg += `<line class="axis" x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{height - margin.bottom}}"></line>`;
      svg += `<path class="actualLine" d="${{path}}"></path>`;
      const latest = points[points.length - 1];
      svg += `<circle class="dotActual" cx="${{x(latest.time)}}" cy="${{y(latest.price)}}" r="5"></circle>`;
      svg += `<text class="chartLabel" text-anchor="end" x="${{width - margin.right}}" y="${{height - 12}}">${{latest.time.toLocaleString()}}</text>`;
      svg += `<text class="chartLabel" x="${{margin.left}}" y="${{height - 12}}">${{points[0].time.toLocaleString()}}</text>`;
      svg += `</svg>`;
      host.innerHTML = svg;
    }}
    function renderForecastChart(chart) {{
      const host = document.getElementById("forecastChart");
      host.innerHTML = "";
      const actualAll = (chart.actual || []).map(p => ({{ time: new Date(p.time), price: Number(p.price) }})).filter(p => !Number.isNaN(p.time.getTime()) && Number.isFinite(p.price));
      const forecast = (chart.latest_forecast || []).map(p => ({{
        time: new Date(p.target_time),
        price: Number(p.predicted_price),
        lower: p.lower_price === null || p.lower_price === undefined ? null : Number(p.lower_price),
        upper: p.upper_price === null || p.upper_price === undefined ? null : Number(p.upper_price),
        horizon: p.horizon_hours
      }})).filter(p => !Number.isNaN(p.time.getTime()) && Number.isFinite(p.price));
      const asOf = chart.as_of ? {{ time: new Date(chart.as_of.time), price: Number(chart.as_of.price) }} : null;
      renderForecastPointRows(forecast);
      set("chartMeta", `actual points ${{chart.actual_count || 0}} | forecast points ${{chart.forecast_count || 0}} | generated ${{time(chart.latest_created_at)}}`);
      if (!actualAll.length && !forecast.length) {{
        host.innerHTML = `<div class="small">No forecast or actual price history available yet.</div>`;
        return;
      }}
      const forecastStart = asOf && !Number.isNaN(asOf.time.getTime()) ? asOf.time.getTime() : Math.min(...forecast.map(p => p.time.getTime()), ...actualAll.slice(-1).map(p => p.time.getTime()));
      const forecastEnd = forecast.length ? Math.max(...forecast.map(p => p.time.getTime())) : forecastStart;
      const horizonMs = Math.max(forecastEnd - forecastStart, 60 * 60 * 1000);
      const visibleMin = forecast.length ? forecastStart - Math.max(45 * 60 * 1000, horizonMs * 0.75) : Math.min(...actualAll.map(p => p.time.getTime()));
      const visibleMax = forecast.length ? forecastEnd + Math.max(15 * 60 * 1000, horizonMs * 0.25) : Math.max(...actualAll.map(p => p.time.getTime()));
      let actual = actualAll.filter(p => p.time.getTime() >= visibleMin && p.time.getTime() <= visibleMax);
      if (actual.length < 2) actual = actualAll.slice(-40);
      const allTimes = [...actual.map(p => p.time), ...forecast.map(p => p.time), ...(asOf && !Number.isNaN(asOf.time.getTime()) ? [asOf.time] : [])];
      const allPrices = [
        ...actual.map(p => p.price),
        ...forecast.flatMap(p => [p.price, p.lower, p.upper]).filter(v => v !== null && Number.isFinite(v)),
        ...(asOf && Number.isFinite(asOf.price) ? [asOf.price] : [])
      ];
      const minTime = Math.min(...allTimes.map(d => d.getTime()));
      const maxTime = Math.max(...allTimes.map(d => d.getTime()));
      const minPriceRaw = Math.min(...allPrices);
      const maxPriceRaw = Math.max(...allPrices);
      const pricePad = Math.max((maxPriceRaw - minPriceRaw) * 0.12, Math.abs(maxPriceRaw) * 0.002, 1);
      const minPrice = minPriceRaw - pricePad;
      const maxPrice = maxPriceRaw + pricePad;
      const width = Math.max(host.clientWidth || 1000, 820);
      const height = 460;
      const margin = {{ left:72, right:120, top:28, bottom:54 }};
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      const x = d => margin.left + ((d.getTime() - minTime) / Math.max(maxTime - minTime, 1)) * innerW;
      const y = price => margin.top + (1 - ((price - minPrice) / Math.max(maxPrice - minPrice, 1))) * innerH;
      const pathFor = points => points.map((p, i) => `${{i ? "L" : "M"}}${{x(p.time).toFixed(1)}},${{y(p.price).toFixed(1)}}`).join(" ");
      const pathsForGapAwareActual = points => {{
        if (!points.length) return [];
        const maxGapMs = 8 * 60 * 1000;
        const segments = [];
        let segment = [points[0]];
        for (let i = 1; i < points.length; i++) {{
          const gap = points[i].time.getTime() - points[i - 1].time.getTime();
          if (gap > maxGapMs) {{
            segments.push(segment);
            segment = [points[i]];
          }} else {{
            segment.push(points[i]);
          }}
        }}
        segments.push(segment);
        return segments;
      }};
      const ticks = 4;
      let svg = `<svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Forecast versus actual chart">`;
      for (let i = 0; i <= ticks; i++) {{
        const yy = margin.top + (innerH / ticks) * i;
        const price = maxPrice - ((maxPrice - minPrice) / ticks) * i;
        svg += `<line class="grid" x1="${{margin.left}}" y1="${{yy}}" x2="${{width - margin.right}}" y2="${{yy}}"></line>`;
        svg += `<text class="chartLabel" x="8" y="${{yy + 4}}">${{money(price)}}</text>`;
      }}
      svg += `<line class="axis" x1="${{margin.left}}" y1="${{height - margin.bottom}}" x2="${{width - margin.right}}" y2="${{height - margin.bottom}}"></line>`;
      svg += `<line class="axis" x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{height - margin.bottom}}"></line>`;
      pathsForGapAwareActual(actual).forEach(segment => {{
        if (segment.length > 1) {{
          svg += `<path class="actualLine" d="${{pathFor(segment)}}"></path>`;
        }} else {{
          svg += `<circle class="dotActual" cx="${{x(segment[0].time)}}" cy="${{y(segment[0].price)}}" r="3"></circle>`;
        }}
      }});
      actual.slice(-1).forEach(p => {{
        svg += `<circle class="dotActual" cx="${{x(p.time)}}" cy="${{y(p.price)}}" r="4"></circle>`;
        svg += `<text class="chartLabel" x="${{x(p.time) + 7}}" y="${{y(p.price) + 4}}">latest ${{money(p.price)}}</text>`;
      }});
      if (asOf && !Number.isNaN(asOf.time.getTime()) && Number.isFinite(asOf.price)) {{
        svg += `<line class="asOf" x1="${{x(asOf.time)}}" y1="${{margin.top}}" x2="${{x(asOf.time)}}" y2="${{height - margin.bottom}}"></line>`;
        svg += `<circle class="dotAsOf" cx="${{x(asOf.time)}}" cy="${{y(asOf.price)}}" r="5"></circle>`;
      }}
      if (forecast.length) {{
        const pathPoints = asOf && Number.isFinite(asOf.price) ? [asOf, ...forecast] : forecast;
        const upper = forecast.filter(p => p.upper !== null).map(p => ({{ time:p.time, price:p.upper }}));
        const lower = forecast.filter(p => p.lower !== null).map(p => ({{ time:p.time, price:p.lower }})).reverse();
        if (upper.length && lower.length) {{
          svg += `<path class="forecastBand" d="${{pathFor(upper)}} L ${{pathFor(lower).replace(/^M/, "")}} Z"></path>`;
        }}
        if (pathPoints.length > 1) svg += `<path class="forecastLine" d="${{pathFor(pathPoints)}}"></path>`;
        forecast.forEach((p, index) => {{
          const xx = x(p.time); const yy = y(p.price);
          const labelRight = xx > width - margin.right - 96;
          const labelX = labelRight ? xx - 12 : xx + 12;
          const anchor = labelRight ? "end" : "start";
          const labelY = yy + ((index % 2 === 0) ? -24 : 30);
          const clampedLabelY = Math.max(margin.top + 14, Math.min(height - margin.bottom - 12, labelY));
          svg += `<circle class="dotForecast" cx="${{xx}}" cy="${{yy}}" r="5"></circle>`;
          svg += `<line class="callout" x1="${{xx}}" y1="${{yy}}" x2="${{labelX}}" y2="${{clampedLabelY - 6}}"></line>`;
          svg += `<text class="chartLabel" text-anchor="${{anchor}}" x="${{labelX}}" y="${{clampedLabelY - 4}}">${{formatHorizon(p.horizon)}}</text>`;
          svg += `<text class="chartValue" text-anchor="${{anchor}}" x="${{labelX}}" y="${{clampedLabelY + 12}}">${{money(p.price)}}</text>`;
        }});
      }}
      const startLabel = new Date(minTime).toLocaleString();
      const endLabel = new Date(maxTime).toLocaleString();
      svg += `<text class="chartLabel" x="${{margin.left}}" y="${{height - 12}}">${{startLabel}}</text>`;
      svg += `<text class="chartLabel" text-anchor="end" x="${{width - margin.right}}" y="${{height - 12}}">${{endLabel}}</text>`;
      svg += `</svg>`;
      host.innerHTML = svg;
    }}
    function formatHorizon(hours) {{
      const value = Number(hours);
      if (!Number.isFinite(value)) return "-";
      const minutes = Math.round(value * 60);
      return minutes < 60 ? `${{minutes}}m` : `${{num(value)}}h`;
    }}
    function renderForecastPointRows(forecast) {{
      const rows = document.getElementById("forecastPointRows");
      rows.innerHTML = "";
      forecast.forEach(point => {{
        const tr = document.createElement("tr");
        const range = point.lower !== null && point.upper !== null ? `${{money(point.lower)}} - ${{money(point.upper)}}` : "-";
        tr.innerHTML = `<td>${{formatHorizon(point.horizon)}}</td><td>${{time(point.time)}}</td><td>${{money(point.price)}}</td><td>${{range}}</td>`;
        rows.appendChild(tr);
      }});
      if (!forecast.length) {{
        const tr = document.createElement("tr");
        tr.innerHTML = `<td colspan="4">No active forecast points yet.</td>`;
        rows.appendChild(tr);
      }}
    }}
    function renderHistory(history) {{
      const rows = document.getElementById("historyRows"); rows.innerHTML = "";
      history.slice().reverse().forEach(row => {{
        const plan = row.option_trade_plan || {{}}; const selected = plan.selected_contract || {{}}; const order = plan.order || {{}};
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${{time(row.checked_at)}}</td><td>${{plan.action || "-"}}</td><td>${{selected.instrument_name || "-"}}</td><td>${{num(order.price)}}</td><td>${{money((plan.risk || {{}}).estimated_debit_usd)}}</td><td class="bad">${{(row.execution_blocks || []).join(", ")}}</td><td>${{((row.order_result || {{}}).submitted)}}</td>`;
        rows.appendChild(tr);
      }});
    }}
    function renderOrders(orders) {{
      const rows = document.getElementById("ordersRows"); rows.innerHTML = "";
      orders.forEach(order => {{
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${{order.instrument_name || "-"}}</td><td>${{order.direction || order.side || "-"}}</td><td>${{num(order.amount)}}</td><td>${{num(order.price)}}</td><td>${{order.order_state || "-"}}</td>`;
        rows.appendChild(tr);
      }});
    }}
    function renderPositions(positions) {{
      const rows = document.getElementById("positionsRows"); rows.innerHTML = "";
      positions.forEach(pos => {{
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${{pos.instrument_name || "-"}}</td><td>${{num(pos.size)}}</td><td>${{num(pos.average_price)}}</td><td>${{num(pos.mark_price)}}</td><td>${{time(pos.expiration_utc)}}</td><td>${{num(pos.floating_profit_loss || pos.total_profit_loss)}}</td>`;
        rows.appendChild(tr);
      }});
    }}
    function renderPositionPl(pl, currency) {{
      const totalUsd = Number(pl.total_unrealized_pl_usd || 0);
      const totalBase = Number(pl.total_unrealized_pl_base || 0);
      const status = pl.status || "flat";
      const statusEl = document.getElementById("plStatus");
      statusEl.textContent = status === "winning" ? "Winning" : status === "losing" ? "Losing" : "Flat";
      statusEl.className = `value ${{status === "winning" ? "win" : status === "losing" ? "loss" : ""}}`;
      const totalEl = document.getElementById("plTotal");
      totalEl.textContent = money(totalUsd);
      totalEl.className = `value ${{totalUsd > 0 ? "win" : totalUsd < 0 ? "loss" : ""}}`;
      set("plTotalBase", `${{num(totalBase)}} ${{currency}}`);
      set("plCounts", `${{pl.winning_positions || 0}} / ${{pl.losing_positions || 0}}`);
      set("plCostValue", `${{money(pl.total_cost_usd)}} / ${{money(pl.total_value_usd)}}`);
      const rows = document.getElementById("plRows");
      rows.innerHTML = "";
      (pl.rows || []).forEach(row => {{
        const rowUsd = Number(row.floating_profit_loss_usd || 0);
        const rowBase = Number(row.floating_profit_loss_base || 0);
        const statusClass = row.status === "winning" ? "win" : row.status === "losing" ? "loss" : "";
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${{row.instrument_name || "-"}}</td><td>${{num(row.size)}}</td><td>${{num(row.average_price)}}</td><td>${{num(row.mark_price)}}</td><td>${{time(row.expiration_utc)}}</td><td>${{num(row.cost_base)}} ${{currency}}</td><td>${{num(row.value_base)}} ${{currency}}</td><td class="${{statusClass}}">${{money(rowUsd)}}<br><span class="small">${{num(rowBase)}} ${{currency}}</span></td><td class="${{statusClass}}">${{row.status || "flat"}}</td>`;
        rows.appendChild(tr);
      }});
      if (!(pl.rows || []).length) {{
        const tr = document.createElement("tr");
        tr.innerHTML = `<td colspan="9">No open Deribit option positions in the latest report.</td>`;
        rows.appendChild(tr);
      }}
    }}
    async function refresh() {{
      try {{
        const response = await fetch(`/api/state${{window.location.search}}`, {{ cache: "no-store" }});
        render(await response.json());
        refreshMarket();
      }} catch (error) {{
        set("details", `dashboard_refresh_failed: ${{error}}`);
      }}
    }}
    refresh(); setInterval(refresh, refreshMs);
  </script>
</body>
</html>"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the Deribit crypto-options agent dashboard.")
    parser.add_argument("--currency", choices=("BTC", "ETH"), default="ETH")
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8792)
    parser.add_argument("--refresh-seconds", type=int, default=DEFAULT_REFRESH_SECONDS)
    parser.add_argument("--max-history", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = build_server(args)
    print(f"Serving Deribit options dashboard at http://{args.host}:{args.port}", flush=True)
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
                currency = _query_value(params, "currency", args.currency)
                state = build_dashboard_state(
                    state_dir=Path(_query_value(params, "state_dir", args.state_dir)),
                    currency=currency,
                    max_history=int(_query_value(params, "max_history", str(args.max_history))),
                )
                self._send_json(state)
                return
            if parsed.path == "/api/market":
                params = parse_qs(parsed.query)
                currency = _query_value(params, "currency", args.currency).upper()
                range_key = _query_value(params, "range", "6m")
                self._send_json(fetch_yahoo_market_history(symbol=deribit_currency_to_yahoo_symbol(currency), range_key=range_key))
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


def _query_value(params: dict[str, list[str]], key: str, default: str) -> str:
    values = params.get(key)
    if not values:
        return default
    return values[0] or default


def _to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _iso_or_none(value: Any) -> str | None:
    parsed = _parse_time(value)
    if parsed is not None:
        return parsed.isoformat()
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


if __name__ == "__main__":
    main()
