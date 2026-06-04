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


DEFAULT_STATE_DIR = Path("automated_forecasting_engine/runs/paper_options_agent")
DEFAULT_REFRESH_SECONDS = 30


def build_dashboard_state(*, state_dir: Path, ticker: str, max_history: int = 50) -> dict[str, Any]:
    report, report_path = read_latest_report(state_dir, ticker)
    state, state_path = read_state(state_dir, ticker)
    history, log_path = read_history(state_dir, ticker, max_history=max_history)
    trade_plan = report.get("option_trade_plan") or {}
    selected = trade_plan.get("selected_contract") or {}
    order = trade_plan.get("order") or {}
    exit_plan = trade_plan.get("exit_plan") or {}
    position_pl = summarize_position_pl(report.get("option_positions") or [])
    option_type = _option_type_label(trade_plan.get("option_type"), selected.get("symbol") or order.get("symbol"))
    chart = build_stock_chart_payload(report=report, history=history)
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
            "forecast_cache_status": report.get("forecast_cache_status"),
            "forecast_direction": (report.get("selected_forecast") or {}).get("expected_direction"),
            "spot": (report.get("selected_forecast") or {}).get("spot"),
            "predicted_price": (report.get("selected_forecast") or {}).get("predicted_price"),
            "action": trade_plan.get("action"),
            "option_type": trade_plan.get("option_type"),
            "option_type_label": option_type,
            "contract": selected.get("symbol"),
            "contract_name": selected.get("name"),
            "greeks": selected.get("greeks") or {},
            "spread_pct": selected.get("spread_pct"),
            "open_interest": selected.get("open_interest"),
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
        },
        "history": history,
        "chart": chart,
    }


def build_stock_chart_payload(*, report: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
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
    forecast_plan = report.get("forecast_plan") or {}
    selected_forecast = report.get("selected_forecast") or {}
    as_of = forecast_plan.get("as_of") or report.get("forecast_created_at_utc") or report.get("checked_at")
    as_of_price = _to_float(forecast_plan.get("latest_price")) or _to_float(selected_forecast.get("spot"))
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
        "actual_points": actual_points[-240:],
        "forecast_points": forecast_points,
        "as_of": as_of,
        "as_of_price": as_of_price,
    }


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
      --bg: #f5f7fa;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #637083;
      --line: #d9dee8;
      --buy: #067647;
      --hold: #7c5c00;
      --blocked: #b42318;
      --accent: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }}
    header {{ display: flex; justify-content: space-between; gap: 16px; padding: 18px 22px; border-bottom: 1px solid var(--line); background: var(--panel); position: sticky; top: 0; z-index: 2; }}
    h1 {{ margin: 0; font-size: 20px; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 18px 22px 30px; }}
    .meta, .small {{ color: var(--muted); font-size: 13px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }}
    .card, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .label {{ color: var(--muted); font-size: 12px; margin-bottom: 5px; }}
    .value {{ font-size: 22px; font-weight: 750; overflow-wrap: anywhere; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: 5px 10px; font-size: 12px; font-weight: 750; color: white; background: var(--hold); text-transform: uppercase; }}
    .badge.buy_option {{ background: var(--buy); }}
    .badge.blocked {{ background: var(--blocked); }}
    .win {{ color: var(--buy); }}
    .loss {{ color: var(--blocked); }}
    .grid2 {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(360px, 0.8fr); gap: 14px; }}
    .chart-panel {{ margin-bottom: 14px; overflow-x: auto; }}
    .chart-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 10px; }}
    .legend {{ display: flex; gap: 14px; flex-wrap: wrap; }}
    .key {{ display: inline-flex; gap: 6px; align-items: center; color: var(--muted); font-size: 12px; }}
    .swatch {{ width: 18px; height: 0; border-top: 2px solid currentColor; display: inline-block; }}
    .swatch.actual {{ color: #111827; }}
    .swatch.forecast {{ color: #dc2626; border-top-style: dashed; }}
    svg#stockChart {{ width: 100%; min-width: 820px; height: 360px; display: block; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 6px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 650; }}
    pre {{ margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; }}
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
    </section>
    <section class="panel chart-panel">
      <div class="chart-head">
        <strong>Stock Price And Forecast</strong>
        <div class="legend">
          <span class="key"><span class="swatch actual"></span>Stock price</span>
          <span class="key"><span class="swatch forecast"></span>Forecast line</span>
        </div>
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
      const width = Math.max(svg.clientWidth || 960, 820);
      const height = 360;
      const pad = {{ left: 62, right: 28, top: 20, bottom: 44 }};
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      svg.innerHTML = "";
      const actual = (chart.actual_points || []).map(p => [new Date(p.timestamp).getTime(), Number(p.price)]).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
      const forecasts = (chart.forecast_points || []).map(p => ({{ ...p, x: new Date(p.timestamp).getTime(), y: Number(p.predicted_price) }})).filter(p => Number.isFinite(p.x) && Number.isFinite(p.y));
      const asOfTime = chart.as_of ? new Date(chart.as_of).getTime() : null;
      const asOfPrice = Number(chart.as_of_price);
      const xs = actual.map(p => p[0]).concat(forecasts.map(p => p.x)).concat(Number.isFinite(asOfTime) ? [asOfTime] : []);
      const ys = actual.map(p => p[1]).concat(forecasts.flatMap(p => [p.y, p.lower_price, p.upper_price]).map(Number).filter(Number.isFinite)).concat(Number.isFinite(asOfPrice) ? [asOfPrice] : []);
      if (!xs.length || !ys.length) {{
        const text = el("text", {{ x: 20, y: 32, fill: "#637083", "font-size": 14 }});
        text.textContent = "No stock price history available yet.";
        svg.appendChild(text);
        return;
      }}
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY0 = Math.min(...ys), maxY0 = Math.max(...ys);
      const yPad = Math.max((maxY0 - minY0) * 0.08, 0.5);
      const minY = minY0 - yPad, maxY = maxY0 + yPad;
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      const x = v => pad.left + ((v - minX) / Math.max(maxX - minX, 1)) * plotW;
      const y = v => pad.top + (1 - ((v - minY) / Math.max(maxY - minY, 1))) * plotH;
      svg.appendChild(el("rect", {{ x: 0, y: 0, width, height, fill: "#fff" }}));
      for (let i = 0; i <= 4; i++) {{
        const yy = pad.top + i * plotH / 4;
        svg.appendChild(el("line", {{ x1: pad.left, y1: yy, x2: width - pad.right, y2: yy, stroke: "#e5e7eb" }}));
        const label = minY + (1 - i / 4) * (maxY - minY);
        const t = el("text", {{ x: 8, y: yy + 4, fill: "#637083", "font-size": 12 }});
        t.textContent = price(label);
        svg.appendChild(t);
      }}
      if (actual.length > 1) {{
        svg.appendChild(el("path", {{ d: actual.map((p, i) => `${{i ? "L" : "M"}}${{x(p[0]).toFixed(2)}},${{y(p[1]).toFixed(2)}}`).join(" "), fill: "none", stroke: "#111827", "stroke-width": 2 }}));
      }}
      const forecastLine = [];
      if (Number.isFinite(asOfTime) && Number.isFinite(asOfPrice)) forecastLine.push([asOfTime, asOfPrice]);
      forecasts.forEach(p => forecastLine.push([p.x, p.y]));
      if (forecastLine.length > 1) {{
        svg.appendChild(el("path", {{ d: forecastLine.map((p, i) => `${{i ? "L" : "M"}}${{x(p[0]).toFixed(2)}},${{y(p[1]).toFixed(2)}}`).join(" "), fill: "none", stroke: "#dc2626", "stroke-width": 2, "stroke-dasharray": "7 5" }}));
      }}
      if (Number.isFinite(asOfTime) && Number.isFinite(asOfPrice)) {{
        svg.appendChild(el("circle", {{ cx: x(asOfTime), cy: y(asOfPrice), r: 5, fill: "#2563eb" }}));
      }}
      forecasts.forEach(p => {{
        svg.appendChild(el("circle", {{ cx: x(p.x), cy: y(p.y), r: 5, fill: "#dc2626" }}));
        const t = el("text", {{ x: x(p.x) + 7, y: y(p.y) - 7, fill: "#991b1b", "font-size": 11 }});
        t.textContent = `${{price(p.horizon_hours)}}h ${{price(p.y)}}`;
        svg.appendChild(t);
      }});
      const axis = el("line", {{ x1: pad.left, y1: height - pad.bottom, x2: width - pad.right, y2: height - pad.bottom, stroke: "#9ca3af" }});
      svg.appendChild(axis);
      const minLabel = el("text", {{ x: pad.left, y: height - 14, fill: "#637083", "font-size": 12 }});
      minLabel.textContent = new Date(minX).toLocaleTimeString();
      svg.appendChild(minLabel);
      const maxLabel = el("text", {{ x: width - pad.right - 90, y: height - 14, fill: "#637083", "font-size": 12 }});
      maxLabel.textContent = new Date(maxX).toLocaleTimeString();
      svg.appendChild(maxLabel);
    }}
    function el(name, attrs) {{
      const node = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.entries(attrs || {{}}).forEach(([key, value]) => node.setAttribute(key, value));
      return node;
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
