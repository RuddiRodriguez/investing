from __future__ import annotations

import argparse
import json
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from market_forecasting_engine.market_history import deribit_currency_to_yahoo_symbol, fetch_yahoo_market_history


DEFAULT_REPORT = Path("automated_forecasting_engine/runs/live_trading/deribit_live_account_report.json")
DEFAULT_SPOT_AGENT_REPORT = Path("automated_forecasting_engine/runs/live_deribit_spot_agent/ETH_USDC_spot_agent_report.json")


def dashboard_html(refresh_seconds: int) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Deribit Account Dashboard</title>
  <style>
    :root {{ --bg:#f4f6f9; --panel:#fff; --text:#182230; --muted:#5d6b7c; --line:#d7dee8; --good:#067647; --bad:#b42318; --warn:#8a4b0f; --accent:#2563eb; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--text); }}
    header {{ display:flex; justify-content:space-between; gap:16px; padding:16px 22px; background:var(--panel); border-bottom:1px solid var(--line); position:sticky; top:0; z-index:3; }}
    h1 {{ margin:0; font-size:20px; }}
    main {{ max-width:1600px; margin:0 auto; padding:18px 22px 34px; }}
    .small {{ color:var(--muted); font-size:13px; }}
    .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:14px; }}
    .panel,.card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
    .label {{ color:var(--muted); font-size:12px; margin-bottom:6px; }}
    .value {{ font-size:22px; font-weight:760; overflow-wrap:anywhere; }}
    .badge {{ display:inline-block; border-radius:999px; padding:5px 10px; font-size:12px; font-weight:760; color:white; background:var(--accent); text-transform:uppercase; }}
    .badge.safe {{ background:var(--good); }}
    .badge.warn {{ background:var(--warn); }}
    .tabs {{ display:flex; flex-wrap:wrap; gap:8px; margin:14px 0; }}
    .tab {{ border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:9px 12px; cursor:pointer; font-weight:700; color:var(--muted); }}
    .tab.active {{ color:white; background:var(--accent); border-color:var(--accent); }}
    .section {{ display:none; }}
    .section.active {{ display:block; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ padding:8px 7px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
    th {{ color:var(--muted); font-weight:700; }}
    .pos {{ color:var(--good); font-weight:700; }}
    .neg {{ color:var(--bad); font-weight:700; }}
    .chartWrap {{ width:100%; height:420px; }}
    svg {{ display:block; width:100%; height:100%; }}
    .axis {{ stroke:#9aa4b2; stroke-width:1; }}
    .chartGrid {{ stroke:#e5e9f0; stroke-width:1; }}
    .priceLine {{ fill:none; stroke:#233044; stroke-width:2; }}
    .forecastLine {{ fill:none; stroke:#dc2626; stroke-width:2; stroke-dasharray:7 5; }}
    .band {{ fill:#dc2626; opacity:.12; }}
    .forecastDot {{ fill:#dc2626; }}
    .asofLine {{ stroke:#2563eb; stroke-width:2; stroke-dasharray:3 4; }}
    .priceDot {{ fill:#233044; }}
    .chartLabel {{ fill:#526071; font-size:12px; }}
    .rangeControls {{ display:flex; gap:8px; flex-wrap:wrap; margin:10px 0 12px; }}
    .rangeButton {{ border:1px solid var(--line); background:#fff; color:var(--muted); border-radius:7px; padding:7px 10px; font-weight:700; cursor:pointer; }}
    .rangeButton.active {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
    pre {{ margin:0; white-space:pre-wrap; overflow:auto; max-height:560px; font-size:12px; }}
    @media (max-width:1000px) {{ header {{ flex-direction:column; }} .grid {{ grid-template-columns:1fr 1fr; }} }}
    @media (max-width:680px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <div><h1>Deribit Account Dashboard</h1><div class="small" id="source">Loading...</div></div>
    <div class="small"><div id="updated">Loading...</div><div>Refresh: {refresh_seconds}s</div></div>
  </header>
  <main>
    <section class="grid">
      <div class="card"><div class="label">Safety</div><div class="value"><span class="badge safe">Read Only</span></div><div class="small" id="safety">No order submission</div></div>
      <div class="card"><div class="label">Positions</div><div class="value" id="positions">-</div><div class="small" id="positionSplit">-</div></div>
      <div class="card"><div class="label">Open Orders</div><div class="value" id="openOrders">-</div><div class="small" id="orderSplit">-</div></div>
      <div class="card"><div class="label">Floating P/L</div><div class="value" id="floatingPl">-</div><div class="small" id="totalPl">-</div></div>
    </section>
    <section class="panel" style="margin-bottom:14px;">
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
    <section class="panel" style="margin-bottom:14px;">
      <strong>Sell / Protection Orders</strong>
      <table>
        <thead><tr><th>Order</th><th>Type</th><th>Amount</th><th>Level</th><th>Meaning</th></tr></thead>
        <tbody id="sellProtectionRows"></tbody>
      </table>
    </section>
    <section class="panel" style="margin-bottom:14px;">
      <strong>Live Spot Agent Decision</strong>
      <div class="grid" style="margin-top:12px;">
        <div class="card"><div class="label">Action</div><div class="value" id="spotAction">-</div><div class="small" id="spotReason">-</div></div>
        <div class="card"><div class="label">Market</div><div class="value" id="spotMarket">-</div><div class="small" id="spotInstrument">-</div></div>
        <div class="card"><div class="label">Forecast</div><div class="value" id="spotForecast">-</div><div class="small" id="spotForecastMeta">-</div></div>
        <div class="card"><div class="label">Execution</div><div class="value" id="spotExecution">-</div><div class="small" id="spotUpdated">-</div></div>
      </div>
      <div class="grid" style="margin-top:12px;">
        <div class="card"><div class="label">Planned Entry</div><div class="value" id="spotEntry">-</div><div class="small" id="spotEntryMeta">-</div></div>
        <div class="card"><div class="label">Post-Fill Protection</div><div class="value" id="spotProtection">-</div><div class="small" id="spotProtectionMeta">-</div></div>
        <div class="card"><div class="label">Pullback Gate</div><div class="value" id="spotPullback">-</div><div class="small" id="spotPullbackMeta">-</div></div>
        <div class="card"><div class="label">Blocks</div><div class="value" id="spotBlocks">-</div><div class="small">visible decision blockers</div></div>
      </div>
      <div class="grid" style="margin-top:12px;">
        <div class="card"><div class="label">Existing Stop Coverage</div><div class="value" id="spotStopCoverage">-</div><div class="small" id="spotStopCoverageMeta">-</div></div>
        <div class="card"><div class="label">Existing Target Coverage</div><div class="value" id="spotTargetCoverage">-</div><div class="small" id="spotTargetCoverageMeta">-</div></div>
        <div class="card"><div class="label">Protection Decision</div><div class="value" id="spotProtectionDecision">-</div><div class="small" id="spotProtectionDecisionMeta">-</div></div>
        <div class="card"><div class="label">Protection Policy</div><div class="value" id="spotProtectionPolicy">-</div><div class="small">respects previous protection unless risk worsens</div></div>
      </div>
      <div class="chartWrap" id="spotForecastChart" style="height:360px; margin-top:12px;"></div>
      <table style="margin-top:12px;">
        <thead><tr><th>Horizon</th><th>Prediction</th><th>Band</th><th>Direction</th><th>Model</th><th>Validation / Error</th></tr></thead>
        <tbody id="spotForecastRows"></tbody>
      </table>
    </section>
    <div class="tabs">
      <button class="tab active" data-tab="overview">Overview</button>
      <button class="tab" data-tab="options">Options</button>
      <button class="tab" data-tab="nonOptions">Futures / Spot</button>
      <button class="tab" data-tab="orders">Orders</button>
      <button class="tab" data-tab="trades">Trades</button>
      <button class="tab" data-tab="raw">Raw Report</button>
    </div>
    <section id="overview" class="section active">
      <div class="panel"><strong>Account Summary By Currency</strong><table><thead><tr><th>Currency</th><th>Equity</th><th>Balance</th><th>Available</th><th>Initial Margin</th><th>Maintenance Margin</th><th>Options Value</th><th>Total P/L</th><th>Delta Total</th></tr></thead><tbody id="accounts"></tbody></table></div>
      <div class="panel" style="margin-top:14px;"><strong>Access Issues</strong><table><thead><tr><th>Endpoint</th><th>Currency</th><th>Kind</th><th>Error</th></tr></thead><tbody id="accessIssues"></tbody></table></div>
    </section>
    <section id="options" class="section">
      <div class="panel"><strong>Option Positions</strong><table><thead><tr><th>Contract</th><th>Call/Put</th><th>Expiry</th><th>Strike</th><th>Direction</th><th>Size</th><th>Avg</th><th>Mark</th><th>Floating P/L</th><th>Total P/L</th><th>Delta</th><th>Gamma</th><th>Vega</th></tr></thead><tbody id="optionPositions"></tbody></table></div>
    </section>
    <section id="nonOptions" class="section">
      <div class="panel"><strong>Futures / Spot Positions</strong><table><thead><tr><th>Instrument</th><th>Kind</th><th>Direction</th><th>Size</th><th>Avg</th><th>Mark</th><th>Floating P/L</th><th>Total P/L</th><th>Delta</th></tr></thead><tbody id="nonOptionPositions"></tbody></table></div>
    </section>
    <section id="orders" class="section">
      <div class="panel"><strong>Open Option Orders</strong><table><thead><tr><th>Contract</th><th>Call/Put</th><th>Direction</th><th>Type</th><th>State</th><th>Amount</th><th>Filled</th><th>Limit / Trigger</th><th>Created</th></tr></thead><tbody id="optionOpenOrders"></tbody></table></div>
      <div class="panel" style="margin-top:14px;"><strong>Open Futures / Spot Orders</strong><table><thead><tr><th>Instrument</th><th>Kind</th><th>Direction</th><th>Type</th><th>State</th><th>Amount</th><th>Filled</th><th>Limit / Trigger</th><th>Created</th></tr></thead><tbody id="nonOptionOpenOrders"></tbody></table></div>
      <div class="panel" style="margin-top:14px;"><strong>Recent Option Orders</strong><table><thead><tr><th>Contract</th><th>Call/Put</th><th>Direction</th><th>Type</th><th>State</th><th>Amount</th><th>Filled</th><th>Avg</th><th>Updated</th></tr></thead><tbody id="optionHistory"></tbody></table></div>
    </section>
    <section id="trades" class="section">
      <div class="panel"><strong>Recent Option Trades</strong><table><thead><tr><th>Contract</th><th>Call/Put</th><th>Direction</th><th>Amount</th><th>Price</th><th>Fee</th><th>Fee Currency</th><th>Time</th></tr></thead><tbody id="optionTrades"></tbody></table></div>
      <div class="panel" style="margin-top:14px;"><strong>Recent Futures / Spot Trades</strong><table><thead><tr><th>Instrument</th><th>Kind</th><th>Direction</th><th>Amount</th><th>Price</th><th>Fee</th><th>Fee Currency</th><th>Time</th></tr></thead><tbody id="nonOptionTrades"></tbody></table></div>
    </section>
    <section id="raw" class="section"><div class="panel"><strong>Full JSON Report</strong><pre id="rawJson">Loading...</pre></div></section>
  </main>
  <script>
    const refreshMs = {refresh_seconds * 1000};
    let marketRange = "6m";
    let currentCurrency = "ETH";
    const num = new Intl.NumberFormat(undefined, {{ maximumFractionDigits:8 }});
    const usd = new Intl.NumberFormat(undefined, {{ style:"currency", currency:"USD", maximumFractionDigits:2 }});
    function fmt(v) {{ return v === null || v === undefined || Number.isNaN(Number(v)) ? "-" : num.format(Number(v)); }}
    function money(v) {{ return v === null || v === undefined || Number.isNaN(Number(v)) ? "-" : usd.format(Number(v)); }}
    function time(v) {{ if (!v) return "-"; const d = new Date(v); return Number.isNaN(d.getTime()) ? v : d.toLocaleString(); }}
    function set(id, value) {{ document.getElementById(id).textContent = value; }}
    function plClass(v) {{ return Number(v) > 0 ? "pos" : Number(v) < 0 ? "neg" : ""; }}
    document.querySelectorAll(".rangeButton").forEach(button => button.addEventListener("click", () => {{
      marketRange = button.dataset.range || "6m";
      document.querySelectorAll(".rangeButton").forEach(item => item.classList.toggle("active", item.dataset.range === marketRange));
      refreshMarket();
    }}));
    document.querySelectorAll(".tab").forEach(button => button.addEventListener("click", () => {{
      document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".section").forEach(s => s.classList.remove("active"));
      button.classList.add("active");
      document.getElementById(button.dataset.tab).classList.add("active");
    }}));
    async function refresh() {{
      try {{
        const response = await fetch("/api/report", {{ cache:"no-store" }});
        render(await response.json());
        refreshMarket();
        refreshSpotAgent();
      }} catch (error) {{
        set("rawJson", `dashboard_refresh_failed: ${{error}}`);
      }}
    }}
    async function refreshSpotAgent() {{
      try {{
        const response = await fetch("/api/spot-agent-report", {{ cache:"no-store" }});
        renderSpotAgent(await response.json());
      }} catch (error) {{
        set("spotReason", `spot_agent_refresh_failed: ${{error}}`);
      }}
    }}
    function renderSpotAgent(report) {{
      const decision = report.decision || {{}};
      const market = report.market || {{}};
      const forecast = report.forecast || {{}};
      const safety = report.safety || {{}};
      const entry = decision.entry_order || {{}};
      const protection = decision.post_fill_protection_plan || decision.protection || {{}};
      const pullback = decision.pullback_buy || {{}};
      const scaleIn = decision.scale_in_pullback || {{}};
      const existingProtection = decision.existing_protection || {{}};
      const protectionDecision = decision.protection_decision || {{}};
      set("spotAction", decision.action || "-");
      set("spotReason", decision.reason || "-");
      set("spotMarket", money(market.latest_price));
      set("spotInstrument", report.instrument || "-");
      set("spotForecast", `${{forecast.expected_direction || "-"}} → ${{money(forecast.predicted_price)}}`);
      set("spotForecastMeta", `edge ${{decision.edge_pct === null || decision.edge_pct === undefined ? "-" : (Number(decision.edge_pct) * 100).toFixed(3) + "%"}}`);
      set("spotExecution", safety.execute_live_orders ? "LIVE" : "DRY RUN");
      set("spotUpdated", `updated ${{time(report.checked_at)}}`);
      set("spotEntry", entry.side ? `${{entry.side}} ${{fmt(entry.amount)}} @ ${{money(entry.price)}}` : "-");
      set("spotEntryMeta", entry.type ? `${{entry.type}} order` : "no entry order planned");
      const stop = protection.stop_loss || {{}};
      const take = protection.take_profit || {{}};
      set("spotProtection", stop.trigger_price || take.price ? `stop ${{money(stop.trigger_price)}} / target ${{money(take.price)}}` : "-");
      set("spotProtectionMeta", decision.post_fill_protection_plan ? "submitted only after buy fill is confirmed" : "immediate protection only for existing filled position");
      const setup = pullback.setup || scaleIn.setup || {{}};
      set("spotPullback", setup.entry_price ? `buy lower @ ${{money(setup.entry_price)}}` : "-");
      const scaleMeta = scaleIn.average_entry_price ? `scale-in avg ${{money(scaleIn.average_entry_price)}} | loss ${{(Number(scaleIn.existing_loss_pct || 0) * 100).toFixed(2)}}%` : "";
      set("spotPullbackMeta", setup.reward_risk ? `R/R ${{fmt(setup.reward_risk)}} | reversal ${{(Number(setup.reversal_probability || 0) * 100).toFixed(1)}}% ${{scaleMeta}}` : (pullback.policy || scaleIn.policy || "-"));
      set("spotBlocks", ((pullback.blocking_reasons || scaleIn.blocking_reasons || decision.execution_blocks || [])).join(", ") || "-");
      set("spotStopCoverage", existingProtection.stop_coverage_ratio === undefined ? "-" : `${{(Number(existingProtection.stop_coverage_ratio || 0) * 100).toFixed(1)}}%`);
      set("spotStopCoverageMeta", existingProtection.nearest_stop_price ? `nearest stop ${{money(existingProtection.nearest_stop_price)}}` : "no existing stop detected");
      set("spotTargetCoverage", existingProtection.take_profit_coverage_ratio === undefined ? "-" : `${{(Number(existingProtection.take_profit_coverage_ratio || 0) * 100).toFixed(1)}}%`);
      set("spotTargetCoverageMeta", existingProtection.nearest_take_profit_price ? `nearest target ${{money(existingProtection.nearest_take_profit_price)}}` : "no existing target detected");
      set("spotProtectionDecision", protectionDecision.decision || "-");
      set("spotProtectionDecisionMeta", protectionDecision.predicted_below_stop_pct ? `predicted below stop ${{(Number(protectionDecision.predicted_below_stop_pct) * 100).toFixed(2)}}%` : (protectionDecision.current_stop_gap_pct ? `stop gap ${{(Number(protectionDecision.current_stop_gap_pct) * 100).toFixed(2)}}%` : "-"));
      set("spotProtectionPolicy", protectionDecision.policy ? "active" : "-");
      renderSpotForecastRows(report);
      renderSpotForecastChart(report);
    }}
    function renderSpotForecastRows(report) {{
      const body = document.getElementById("spotForecastRows");
      body.innerHTML = "";
      const bundle = report.forecast_bundle || {{}};
      const plan = bundle.forecast_plan || {{}};
      const forecasts = plan.forecasts || [];
      forecasts.forEach(row => {{
        const lower = row.lower_price;
        const upper = row.upper_price;
        const direction = Number(row.predicted_price) > Number(plan.latest_price) ? "Upward" : Number(row.predicted_price) < Number(plan.latest_price) ? "Downward" : "Flat";
        const metrics = row.validation_metrics || {{}};
        const validation = metrics.price_mae || metrics.return_mae || metrics.directional_accuracy
          ? `price MAE ${{money(metrics.price_mae)}} | return MAE ${{fmt(metrics.return_mae)}} | dir ${{(Number(metrics.directional_accuracy || 0) * 100).toFixed(1)}}% | n=${{metrics.sample_count || 0}}`
          : (metrics.status || "-");
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${{fmt(row.horizon_hours)}}h</td><td>${{money(row.predicted_price)}}</td><td>${{money(lower)}} - ${{money(upper)}}</td><td>${{direction}}</td><td>${{row.selected_model || row.model || row.method || "-"}}</td><td>${{validation}}</td>`;
        body.appendChild(tr);
      }});
      if (!forecasts.length) body.innerHTML = `<tr><td colspan="6">No forecast horizons in the latest spot-agent report.</td></tr>`;
    }}
    function renderSpotForecastChart(report) {{
      const host = document.getElementById("spotForecastChart");
      host.innerHTML = "";
      const bundle = report.forecast_bundle || {{}};
      const plan = bundle.forecast_plan || {{}};
      const tail = (bundle.price_tail || []).map(p => ({{ time:new Date(p.time), price:Number(p.close) }})).filter(p => !Number.isNaN(p.time.getTime()) && Number.isFinite(p.price));
      const forecasts = (plan.forecasts || []).map(row => {{
        const baseTime = new Date(plan.as_of || report.checked_at);
        const horizonMs = Number(row.horizon_hours || 0) * 3600 * 1000;
        return {{
          time: row.forecast_timestamp ? new Date(row.forecast_timestamp) : new Date(baseTime.getTime() + horizonMs),
          price: Number(row.predicted_price),
          lower: Number(row.lower_price),
          upper: Number(row.upper_price),
          horizon: Number(row.horizon_hours || 0)
        }};
      }}).filter(p => !Number.isNaN(p.time.getTime()) && Number.isFinite(p.price));
      if (!tail.length && !forecasts.length) {{
        host.innerHTML = `<div class="small">No actual/forecast points in latest spot-agent report.</div>`;
        return;
      }}
      const width = Math.max(host.clientWidth || 1000, 820);
      const height = 360;
      const margin = {{ left:76, right:42, top:24, bottom:50 }};
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      const allTimes = [...tail.map(p => p.time.getTime()), ...forecasts.map(p => p.time.getTime())];
      const allPrices = [...tail.map(p => p.price), ...forecasts.flatMap(p => [p.price, p.lower, p.upper]).filter(Number.isFinite)];
      const minTime = Math.min(...allTimes);
      const maxTime = Math.max(...allTimes);
      const minRaw = Math.min(...allPrices);
      const maxRaw = Math.max(...allPrices);
      const pad = Math.max((maxRaw - minRaw) * 0.16, Math.abs(maxRaw) * 0.002, 1);
      const minPrice = minRaw - pad;
      const maxPrice = maxRaw + pad;
      const x = d => margin.left + ((d.getTime() - minTime) / Math.max(maxTime - minTime, 1)) * innerW;
      const y = price => margin.top + (1 - ((price - minPrice) / Math.max(maxPrice - minPrice, 1))) * innerH;
      const actualPath = tail.map((p, i) => `${{i ? "L" : "M"}}${{x(p.time).toFixed(1)}},${{y(p.price).toFixed(1)}}`).join(" ");
      const forecastPath = forecasts.map((p, i) => `${{i ? "L" : "M"}}${{x(p.time).toFixed(1)}},${{y(p.price).toFixed(1)}}`).join(" ");
      let svg = `<svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Live spot forecast chart">`;
      for (let i = 0; i <= 4; i++) {{
        const yy = margin.top + (innerH / 4) * i;
        const price = maxPrice - ((maxPrice - minPrice) / 4) * i;
        svg += `<line class="chartGrid" x1="${{margin.left}}" y1="${{yy}}" x2="${{width - margin.right}}" y2="${{yy}}"></line>`;
        svg += `<text class="chartLabel" x="8" y="${{yy + 4}}">${{money(price)}}</text>`;
      }}
      svg += `<line class="axis" x1="${{margin.left}}" y1="${{height - margin.bottom}}" x2="${{width - margin.right}}" y2="${{height - margin.bottom}}"></line>`;
      svg += `<line class="axis" x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{height - margin.bottom}}"></line>`;
      if (actualPath) svg += `<path class="priceLine" d="${{actualPath}}"></path>`;
      if (forecasts.length) {{
        const upper = forecasts.map((p, i) => `${{i ? "L" : "M"}}${{x(p.time).toFixed(1)}},${{y(p.upper).toFixed(1)}}`).join(" ");
        const lower = forecasts.slice().reverse().map((p, i) => `${{i ? "L" : "L"}}${{x(p.time).toFixed(1)}},${{y(p.lower).toFixed(1)}}`).join(" ");
        svg += `<path class="band" d="${{upper}} ${{lower}} Z"></path>`;
        svg += `<path class="forecastLine" d="${{forecastPath}}"></path>`;
        forecasts.forEach(p => {{
          svg += `<circle class="forecastDot" cx="${{x(p.time)}}" cy="${{y(p.price)}}" r="4"></circle>`;
          svg += `<text class="chartLabel" x="${{x(p.time) + 6}}" y="${{y(p.price) - 6}}">${{fmt(p.horizon)}}h</text>`;
        }});
      }}
      const asof = new Date(plan.as_of || report.checked_at);
      if (!Number.isNaN(asof.getTime())) svg += `<line class="asofLine" x1="${{x(asof)}}" y1="${{margin.top}}" x2="${{x(asof)}}" y2="${{height - margin.bottom}}"></line>`;
      svg += `<text class="chartLabel" x="${{margin.left}}" y="${{height - 12}}">${{new Date(minTime).toLocaleString()}}</text>`;
      svg += `<text class="chartLabel" text-anchor="end" x="${{width - margin.right}}" y="${{height - 12}}">${{new Date(maxTime).toLocaleString()}}</text>`;
      svg += `</svg>`;
      host.innerHTML = svg;
    }}
    function render(report) {{
      const overview = report.overview || {{}};
      const options = report.options || {{ summary:{{}}, positions:[], open_orders:[], order_history:[], trades:[] }};
      const nonOptions = report.non_options || {{ summary:{{}}, positions:[], open_orders:[], order_history:[], trades:[] }};
      currentCurrency = (report.currencies || [currentCurrency])[0] || currentCurrency;
      set("source", `${{report.venue || "-"}} | ${{report.endpoint || "-"}} | currencies ${{(report.currencies || []).join(",")}}`);
      set("updated", `Updated ${{time(report.checked_at)}}`);
      set("safety", (report.safety || {{}}).policy || "Read-only");
      set("positions", overview.position_count || 0);
      set("positionSplit", `${{overview.option_position_count || 0}} options | ${{overview.non_option_position_count || 0}} futures/spot`);
      set("openOrders", overview.open_order_count || 0);
      set("orderSplit", `${{overview.option_open_order_count || 0}} options | ${{overview.non_option_open_order_count || 0}} futures/spot`);
      const floating = document.getElementById("floatingPl");
      floating.textContent = fmt(overview.floating_profit_loss);
      floating.className = `value ${{plClass(overview.floating_profit_loss)}}`;
      set("totalPl", `total P/L ${{fmt(overview.total_profit_loss)}}`);
      renderAccounts(report.account_summaries || {{}});
      renderAccessIssues(report.access_issues || []);
      renderPositions("optionPositions", options.positions || [], true);
      renderPositions("nonOptionPositions", nonOptions.positions || [], false);
      renderOrders("optionOpenOrders", options.open_orders || [], true, true);
      renderOrders("nonOptionOpenOrders", nonOptions.open_orders || [], false, true);
      renderSellProtectionRows(nonOptions.open_orders || []);
      renderOrders("optionHistory", options.order_history || [], true, false);
      renderTrades("optionTrades", options.trades || [], true);
      renderTrades("nonOptionTrades", nonOptions.trades || [], false);
      set("rawJson", JSON.stringify(report, null, 2));
    }}
    function renderSellProtectionRows(rows) {{
      const body = document.getElementById("sellProtectionRows");
      body.innerHTML = "";
      const visible = rows.filter(row => row.instrument_name === "ETH_USDC" && ["sell", "buy"].includes(String(row.direction || "").toLowerCase()));
      visible.forEach(row => {{
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${{orderName(row)}}</td><td>${{row.order_type || "-"}}</td><td>${{fmt(row.amount)}} ETH</td><td>${{orderLevel(row)}}</td><td>${{orderMeaning(row)}}</td>`;
        body.appendChild(tr);
      }});
      if (!visible.length) {{
        body.innerHTML = `<tr><td colspan="5">No ETH_USDC sell, stop, or buy limit orders in the latest live report.</td></tr>`;
      }}
    }}
    function orderName(row) {{
      const direction = String(row.direction || "").toLowerCase();
      const type = String(row.order_type || "").toLowerCase();
      if (direction === "sell" && type === "limit") return "Sell limit";
      if (direction === "sell" && type === "stop_market") return "Sell stop";
      if (direction === "buy" && type === "limit") return "Buy limit";
      return `${{row.direction || "-"}} ${{row.order_type || "-"}}`;
    }}
    function orderLevel(row) {{
      if (row.order_type === "stop_market") return `trigger ${{fmt(row.trigger_price)}} USDC`;
      return `${{fmt(row.price)}} USDC`;
    }}
    function orderMeaning(row) {{
      const direction = String(row.direction || "").toLowerCase();
      const type = String(row.order_type || "").toLowerCase();
      if (direction === "sell" && type === "limit") return `Take profit if ETH rises to ${{fmt(row.price)}}`;
      if (direction === "sell" && type === "stop_market") return `Protective stop if ETH falls to ${{fmt(row.trigger_price)}}`;
      if (direction === "buy" && type === "limit") return "Not protection; this is a buy/re-entry style order";
      return "-";
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
      const height = 420;
      const margin = {{ left:72, right:34, top:24, bottom:50 }};
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
        svg += `<line class="chartGrid" x1="${{margin.left}}" y1="${{yy}}" x2="${{width - margin.right}}" y2="${{yy}}"></line>`;
        svg += `<text class="chartLabel" x="8" y="${{yy + 4}}">${{money(price)}}</text>`;
      }}
      svg += `<line class="axis" x1="${{margin.left}}" y1="${{height - margin.bottom}}" x2="${{width - margin.right}}" y2="${{height - margin.bottom}}"></line>`;
      svg += `<line class="axis" x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{height - margin.bottom}}"></line>`;
      svg += `<path class="priceLine" d="${{path}}"></path>`;
      const latest = points[points.length - 1];
      svg += `<circle class="priceDot" cx="${{x(latest.time)}}" cy="${{y(latest.price)}}" r="5"></circle>`;
      svg += `<text class="chartLabel" text-anchor="end" x="${{width - margin.right}}" y="${{height - 12}}">${{latest.time.toLocaleString()}}</text>`;
      svg += `<text class="chartLabel" x="${{margin.left}}" y="${{height - 12}}">${{points[0].time.toLocaleString()}}</text>`;
      svg += `</svg>`;
      host.innerHTML = svg;
    }}
    function renderAccounts(rows) {{
      const body = document.getElementById("accounts"); body.innerHTML = "";
      Object.entries(rows).forEach(([currency, row]) => {{
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${{currency}}</td><td>${{fmt(row.equity)}}</td><td>${{fmt(row.balance)}}</td><td>${{fmt(row.available_funds)}}</td><td>${{fmt(row.initial_margin)}}</td><td>${{fmt(row.maintenance_margin)}}</td><td>${{fmt(row.options_value)}}</td><td class="${{plClass(row.total_pl)}}">${{fmt(row.total_pl)}}</td><td>${{fmt(row.delta_total)}}</td>`;
        body.appendChild(tr);
      }});
      if (!Object.keys(rows).length) body.innerHTML = `<tr><td colspan="9">No account summary in latest report.</td></tr>`;
    }}
    function renderAccessIssues(rows) {{
      const body = document.getElementById("accessIssues"); body.innerHTML = "";
      rows.forEach(row => {{
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${{row.endpoint || "-"}}</td><td>${{row.currency || "-"}}</td><td>${{row.kind || "-"}}</td><td>${{row.error || "-"}}</td>`;
        body.appendChild(tr);
      }});
      if (!rows.length) body.innerHTML = `<tr><td colspan="4">No API access issues in latest report.</td></tr>`;
    }}
    function renderPositions(id, rows, isOption) {{
      const body = document.getElementById(id); body.innerHTML = "";
      rows.forEach(row => {{
        const opt = row.option_details || {{}};
        const tr = document.createElement("tr");
        if (isOption) {{
          tr.innerHTML = `<td>${{row.instrument_name || "-"}}</td><td>${{opt.option_type || "-"}}</td><td>${{opt.expiration || "-"}}</td><td>${{fmt(opt.strike)}}</td><td>${{row.direction || "-"}}</td><td>${{fmt(row.size)}}</td><td>${{fmt(row.average_price)}}</td><td>${{fmt(row.mark_price)}}</td><td class="${{plClass(row.floating_profit_loss)}}">${{fmt(row.floating_profit_loss)}}</td><td class="${{plClass(row.total_profit_loss)}}">${{fmt(row.total_profit_loss)}}</td><td>${{fmt(row.delta)}}</td><td>${{fmt(row.gamma)}}</td><td>${{fmt(row.vega)}}</td>`;
        }} else {{
          tr.innerHTML = `<td>${{row.instrument_name || "-"}}</td><td>${{row.kind || "-"}}</td><td>${{row.direction || "-"}}</td><td>${{fmt(row.size)}}</td><td>${{fmt(row.average_price)}}</td><td>${{fmt(row.mark_price)}}</td><td class="${{plClass(row.floating_profit_loss)}}">${{fmt(row.floating_profit_loss)}}</td><td class="${{plClass(row.total_profit_loss)}}">${{fmt(row.total_profit_loss)}}</td><td>${{fmt(row.delta)}}</td>`;
        }}
        body.appendChild(tr);
      }});
      if (!rows.length) body.innerHTML = `<tr><td colspan="13">No rows in latest report.</td></tr>`;
    }}
    function renderOrders(id, rows, isOption, openOnly) {{
      const body = document.getElementById(id); body.innerHTML = "";
      rows.forEach(row => {{
        const opt = row.option_details || {{}};
        const tr = document.createElement("tr");
        const name = row.instrument_name || "-";
        const typeCell = isOption ? `<td>${{opt.option_type || "-"}}</td>` : `<td>${{row.kind || "-"}}</td>`;
        const price = openOnly ? orderPriceLabel(row) : fmt(row.average_price);
        const stamp = openOnly ? row.creation_timestamp : row.last_update_timestamp;
        tr.innerHTML = `<td>${{name}}</td>${{typeCell}}<td>${{row.direction || "-"}}</td><td>${{row.order_type || "-"}}</td><td>${{row.order_state || "-"}}</td><td>${{fmt(row.amount)}}</td><td>${{fmt(row.filled_amount)}}</td><td>${{price}}</td><td>${{time(stamp)}}</td>`;
        body.appendChild(tr);
      }});
      if (!rows.length) body.innerHTML = `<tr><td colspan="9">No rows in latest report.</td></tr>`;
    }}
    function orderPriceLabel(row) {{
      if (row.order_type === "stop_market") {{
        return `trigger ${{fmt(row.trigger_price)}} ${{row.trigger ? `(${{row.trigger}})` : ""}}`;
      }}
      return fmt(row.price);
    }}
    function renderTrades(id, rows, isOption) {{
      const body = document.getElementById(id); body.innerHTML = "";
      rows.forEach(row => {{
        const opt = row.option_details || {{}};
        const typeCell = isOption ? `<td>${{opt.option_type || "-"}}</td>` : `<td>${{row.kind || "-"}}</td>`;
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${{row.instrument_name || "-"}}</td>${{typeCell}}<td>${{row.direction || "-"}}</td><td>${{fmt(row.amount)}}</td><td>${{fmt(row.price)}}</td><td>${{fmt(row.fee)}}</td><td>${{row.fee_currency || "-"}}</td><td>${{time(row.timestamp)}}</td>`;
        body.appendChild(tr);
      }});
      if (!rows.length) body.innerHTML = `<tr><td colspan="8">No rows in latest report.</td></tr>`;
    }}
    refresh(); setInterval(refresh, refreshMs);
  </script>
</body>
</html>"""


def build_server(*, report_path: Path, spot_agent_report_path: Path, host: str, port: int, refresh_seconds: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(dashboard_html(refresh_seconds))
                return
            if parsed.path == "/api/report":
                self._send_json(_read_report(report_path))
                return
            if parsed.path == "/api/spot-agent-report":
                self._send_json(_read_spot_agent_report(spot_agent_report_path))
                return
            if parsed.path == "/api/market":
                params = parse_qs(parsed.query)
                currency = _query_value(params, "currency", "ETH").upper()
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
            body = json.dumps(_strict_json(payload), default=str, allow_nan=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((host, port), Handler)


def _read_report(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "checked_at": datetime.now().isoformat(),
            "mode": "read_only_deribit_account_report",
            "venue": "missing_report",
            "endpoint": None,
            "safety": {"order_submission_enabled": False, "policy": f"Report not found at {path}."},
            "currencies": [],
            "account_summaries": {},
            "overview": {},
            "options": {"summary": {}, "positions": [], "open_orders": [], "order_history": [], "trades": []},
            "non_options": {"summary": {}, "positions": [], "open_orders": [], "order_history": [], "trades": []},
            "access_issues": [],
        }
    return parsed if isinstance(parsed, dict) else {}


def _read_spot_agent_report(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "checked_at": datetime.now().isoformat(),
            "mode": "live_deribit_spot_agent",
            "instrument": "ETH_USDC",
            "safety": {"execute_live_orders": False},
            "market": {},
            "forecast": {},
            "decision": {"action": "missing_report", "reason": f"Report not found at {path}."},
        }
    return parsed if isinstance(parsed, dict) else {}


def _query_value(params: dict[str, list[str]], key: str, default: str) -> str:
    values = params.get(key)
    return values[0] if values and values[0] else default


def _strict_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strict_json(item) for item in value]
    if isinstance(value, float):
        return None if pd.isna(value) else value
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the read-only Deribit account dashboard.")
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT))
    parser.add_argument("--spot-agent-report-path", default=str(DEFAULT_SPOT_AGENT_REPORT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8794)
    parser.add_argument("--refresh-seconds", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = build_server(report_path=Path(args.report_path), spot_agent_report_path=Path(args.spot_agent_report_path), host=args.host, port=int(args.port), refresh_seconds=int(args.refresh_seconds))
    print(f"Serving Deribit read-only dashboard at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
