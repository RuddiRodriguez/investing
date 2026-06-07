from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from market_forecasting_engine.llm_options_trader.common import LLMOptionsRuntimeConfig, load_underlying_prices, price_summary, recent_price_bars


DEFAULT_STATE_DIR = Path("automated_forecasting_engine/runs/llm_options_trader_testnet")


def build_dashboard_state(*, state_dir: Path, currency: str, max_memory: int = 80) -> dict[str, Any]:
    agent_report, agent_path = _read_report(state_dir, currency, "agent")
    entry_report, entry_path = _read_report(state_dir, currency, "entry")
    exit_report, exit_path = _read_report(state_dir, currency, "exit")
    agent_log = _read_jsonl_any(
        state_dir / "logs" / f"{currency.upper()}_llm_agent.jsonl",
        state_dir / "logs" / f"{currency.upper()}_alpaca_llm_shadow_agent.jsonl",
    )
    entry_log = _read_jsonl(state_dir / "logs" / f"{currency.upper()}_llm_entry_agent.jsonl")
    exit_log = _read_jsonl(state_dir / "logs" / f"{currency.upper()}_llm_exit_agent.jsonl")
    memory = _read_jsonl(state_dir / "memory" / f"{currency.upper()}_llm_trader_memory.jsonl")[-max_memory:]
    strategy_memory = _read_json(state_dir / "memory" / f"{currency.upper()}_strategy_memory.json") or {}
    latest = agent_report or entry_report or exit_report or {}
    latest_packet = latest.get("market_packet") if isinstance(latest.get("market_packet"), dict) else {}
    latest_provider = latest.get("llm_provider") if isinstance(latest, dict) else None
    llm_usage = read_llm_usage(currency=currency, state_dir=state_dir, provider=latest_provider)
    agent_decision = _decision_summary(agent_report)
    entry_decision = _subdecision_summary(agent_report, decision_key="entry_decision", result_key="entry_order_result")
    exit_decision = _subdecision_summary(agent_report, decision_key="exit_decision", result_key="exit_order_result")
    if not entry_decision:
        entry_decision = agent_decision if agent_report else _decision_summary(entry_report)
    if not exit_decision:
        exit_decision = agent_decision if agent_report else _decision_summary(exit_report)
    packet = latest_packet
    fallback_bars = []
    fallback_price_summary: dict[str, Any] = {}
    if not (latest.get("dashboard_price_bars") or packet.get("recent_price_bars")):
        fallback = _fallback_price_payload(currency)
        fallback_bars = fallback.get("recent_price_bars") or []
        fallback_price_summary = fallback.get("price_summary") or {}
    account = packet.get("account") if isinstance(packet.get("account"), dict) else {}
    shadow = packet.get("shadow_simulation") if isinstance(packet.get("shadow_simulation"), dict) else {}
    llm_cost = float(llm_usage.get("total_cost_usd") or 0.0)
    pnl = _pnl_summary(account=account, shadow=shadow, llm_cost_usd=llm_cost)
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "currency": currency.upper(),
        "state_dir": str(state_dir),
        "agent_report_path": str(agent_path) if agent_path else None,
        "entry_report_path": str(entry_path) if entry_path else None,
        "exit_report_path": str(exit_path) if exit_path else None,
        "agent_report": agent_report,
        "entry_report": entry_report,
        "exit_report": exit_report,
        "summary": {
            "venue": packet.get("venue") or "deribit",
            "asset_class": packet.get("asset_class") or ("option" if packet.get("option_chain") else None),
            "account_mode": packet.get("account_mode") or (latest.get("account_mode") if isinstance(latest, dict) else None),
            "execution_mode": packet.get("execution_mode") or ("simulation_only" if latest.get("simulation_only") else None),
            "latest_underlying_price": packet.get("latest_underlying_price"),
            "price_data_provider": packet.get("price_data_provider") or ("deribit" if fallback_bars else None),
            "price_data_interval": packet.get("price_data_interval") or ("1m" if fallback_bars else None),
            "entry_checked_at": entry_report.get("checked_at_utc"),
            "exit_checked_at": exit_report.get("checked_at_utc"),
            "agent_checked_at": agent_report.get("checked_at_utc"),
            "entry_age_seconds": _age_seconds(entry_report.get("checked_at_utc")),
            "exit_age_seconds": _age_seconds(exit_report.get("checked_at_utc")),
            "agent_age_seconds": _age_seconds(agent_report.get("checked_at_utc")),
            "agent_decision": agent_decision,
            "entry_decision": entry_decision,
            "exit_decision": exit_decision,
            "account": account,
            "pnl": pnl,
            "shadow_simulation": shadow,
            "open_order_count": len(packet.get("open_option_orders") or []),
            "position_count": len(packet.get("option_positions") or []),
            "memory_count": len(memory),
            "strategy_lesson_count": len(strategy_memory.get("lessons") or []),
            "trader_profile": packet.get("trader_profile") or {},
            "llm_usage": llm_usage,
            "spot_instrument": packet.get("spot_instrument") or {},
        },
        "account": account,
        "pnl": pnl,
        "shadow_simulation": shadow,
        "open_option_orders": packet.get("open_option_orders") or [],
        "option_positions": packet.get("option_positions") or [],
        "recent_option_trades": packet.get("recent_option_trades") or [],
        "price_summary": packet.get("price_summary") or fallback_price_summary or {},
        "technical_observations": packet.get("technical_observations") or {},
        "short_tape_summary": packet.get("short_tape_summary") or {},
        "regime_transition_warning": packet.get("regime_transition_warning") or {},
        "option_tradeability_summary": packet.get("option_tradeability_summary") or {},
        "adaptive_profit_policy": packet.get("adaptive_profit_policy") or latest.get("profit_policy_decision") or {},
        "strategy_knowledge": packet.get("strategy_knowledge") or {},
        "strategy_memory": packet.get("strategy_memory") or strategy_memory or {},
        "forecast_validation": packet.get("forecast_validation") or {},
        "chronos_forecast": ((packet.get("external_forecasts") or {}).get("chronos") if isinstance(packet.get("external_forecasts"), dict) else {}) or {},
        "recent_price_bars": latest.get("dashboard_price_bars") or packet.get("recent_price_bars") or fallback_bars,
        "option_chain": packet.get("option_chain") or [],
        "spot_instrument": packet.get("spot_instrument") or {},
        "trader_memory": memory,
        "llm_usage": llm_usage,
        "agent_history": agent_log[-80:],
        "entry_history": entry_log[-80:],
        "exit_history": exit_log[-80:],
    }


def _fallback_price_payload(currency: str) -> dict[str, Any]:
    try:
        config = LLMOptionsRuntimeConfig(
            currency=currency.upper(),
            instrument_currency="USDC",
            data_provider="deribit",
            data_interval="1m",
            lookback_days=2,
            max_price_rows=3000,
        )
        prices = load_underlying_prices(config=config, now=datetime.now(UTC))
    except Exception as exc:
        return {"status": "failed", "reason": str(exc), "recent_price_bars": [], "price_summary": {}}
    return {
        "status": "ok",
        "recent_price_bars": recent_price_bars(prices, rows=3000),
        "price_summary": price_summary(prices),
    }


def dashboard_html(refresh_seconds: int) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM Options Trader Protocol</title>
  <style>
    :root {{ --bg:#080b10; --panel:#101720; --panel2:#141e2a; --text:#e9f0f8; --muted:#8ea0b6; --line:#263448; --good:#2ed48f; --bad:#ff5c74; --warn:#f6b84b; --accent:#6aa2ff; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header {{ position:sticky; top:0; z-index:2; padding:16px 22px; background:rgba(16,23,32,.96); border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:18px; }}
    h1 {{ margin:0; font-size:20px; letter-spacing:0; }}
    main {{ max-width:1480px; margin:0 auto; padding:18px 22px 34px; }}
    .small {{ color:var(--muted); font-size:13px; }}
    .grid4 {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:14px; }}
    .grid2 {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:14px; margin-bottom:14px; }}
    .panel,.card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; box-shadow:0 10px 28px rgba(0,0,0,.20); }}
    .card {{ background:var(--panel2); min-height:94px; }}
    .label {{ color:var(--muted); font-size:12px; margin-bottom:6px; }}
    .value {{ font-size:22px; font-weight:760; overflow-wrap:anywhere; }}
    .badge {{ display:inline-block; padding:5px 10px; border-radius:999px; background:var(--warn); color:#111; font-size:12px; font-weight:800; text-transform:uppercase; }}
    .badge.submit_order {{ background:var(--good); color:#04120c; }} .badge.hold {{ background:#354255; color:#dfe8f5; }} .badge.cancel_order {{ background:var(--warn); color:#1b1203; }}
    .bad {{ color:var(--bad); }} .good {{ color:var(--good); }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ text-align:left; vertical-align:top; padding:8px 7px; border-bottom:1px solid var(--line); }}
    th {{ color:var(--muted); font-weight:700; }}
    pre {{ margin:0; white-space:pre-wrap; overflow-wrap:anywhere; max-height:620px; overflow:auto; color:#d9e4f2; font-size:12px; }}
    .chart {{ height:520px; width:100%; }}
    .chartControls {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:10px 0 6px; }}
    .chartControls button,.chartControls label {{ border:1px solid var(--line); background:#0b1119; color:var(--text); border-radius:6px; padding:6px 9px; font-size:12px; }}
    .chartControls button.active {{ background:var(--accent); color:#04111f; font-weight:800; }}
    .chartControls label {{ display:flex; gap:6px; align-items:center; cursor:pointer; }}
    svg {{ width:100%; height:100%; display:block; }}
    .axis,.grid {{ stroke:#304157; stroke-width:1; }} .line {{ fill:none; stroke:#dbe7f6; stroke-width:2; }} .dot {{ fill:var(--accent); }}
    @media (max-width:1000px) {{ header,.grid2 {{ grid-template-columns:1fr; flex-direction:column; }} .grid4 {{ grid-template-columns:1fr 1fr; }} }}
    @media (max-width:620px) {{ .grid4 {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <div><h1>LLM Options Trader Protocol</h1><div class="small" id="subtitle">Loading...</div></div>
    <div class="small"><div id="updated">Loading...</div><div>Refresh: {refresh_seconds}s</div></div>
  </header>
  <main>
    <section class="grid4">
      <div class="card"><div class="label">Total Win/Loss</div><div class="value" id="totalPnl">-</div><div class="small">realized + open option P/L</div></div>
      <div class="card"><div class="label">Realized</div><div class="value" id="realizedPnl">-</div><div class="small">closed trades this session</div></div>
      <div class="card"><div class="label">Open P/L</div><div class="value" id="unrealizedPnl">-</div><div class="small">current open positions</div></div>
      <div class="card"><div class="label">Net After LLM Cost</div><div class="value" id="netPnl">-</div><div class="small" id="netMeta">cost-adjusted test result</div></div>
    </section>
    <section class="grid4">
      <div class="card"><div class="label">Trader</div><div class="value" id="traderName">-</div><div class="small" id="traderRole">-</div></div>
      <div class="card"><div class="label">Underlying</div><div class="value" id="spot">-</div><div class="small" id="priceMeta">-</div></div>
      <div class="card"><div class="label">Entry LLM</div><div class="value"><span id="entryBadge" class="badge">-</span></div><div class="small" id="entryMeta">-</div></div>
      <div class="card"><div class="label">Exit LLM</div><div class="value"><span id="exitBadge" class="badge">-</span></div><div class="small" id="exitMeta">-</div></div>
    </section>
    <section class="grid4">
      <div class="card"><div class="label">Account</div><div class="value" id="account">-</div><div class="small" id="accountMeta">-</div></div>
      <div class="card"><div class="label">Exposure</div><div class="value" id="exposure">-</div><div class="small">open orders / positions</div></div>
      <div class="card"><div class="label">Memory</div><div class="value" id="memoryCount">-</div><div class="small">recent choices supplied to LLM</div></div>
      <div class="card"><div class="label">Lessons</div><div class="value" id="lessonCount">-</div><div class="small">persistent strategy memory</div></div>
      <div class="card"><div class="label">LLM Cost</div><div class="value" id="llmCost">-</div><div class="small" id="llmUsage">-</div></div>
    </section>
    <section class="grid2">
      <div class="panel"><strong>Entry Decision</strong><pre id="entryDecision">Loading...</pre></div>
      <div class="panel"><strong>Exit Decision</strong><pre id="exitDecision">Loading...</pre></div>
    </section>
    <section class="panel">
      <strong id="chartTitle">Forecast vs Actual</strong>
      <div class="chartControls">
        <button type="button" data-hours="1" class="active">1h</button>
        <button type="button" data-hours="24">24h</button>
        <button type="button" data-hours="36">36h</button>
        <button type="button" data-hours="48">48h</button>
        <button type="button" id="forecastZoom">Forecast zoom</button>
        <button type="button" id="zoomIn">Zoom in</button>
        <button type="button" id="zoomOut">Zoom out</button>
        <label><input type="checkbox" id="showLine" checked> line</label>
        <label><input type="checkbox" id="showCandles" checked> 5m candles</label>
        <label><input type="checkbox" id="showSma9" checked> SMA 9</label>
        <label><input type="checkbox" id="showSma21" checked> SMA 21</label>
      </div>
      <div class="small" id="chartMeta">-</div><div class="chart" id="priceChart"></div>
    </section>
    <section class="grid2">
      <div class="panel"><strong>Technical Observations Given to LLM</strong><pre id="technical">Loading...</pre></div>
      <div class="panel"><strong>Adaptive Profit Policy LLM</strong><pre id="profitPolicy">Loading...</pre></div>
    </section>
    <section class="grid2">
      <div class="panel"><strong>Short Tape / Regime Transition</strong><pre id="regimeTransition">Loading...</pre></div>
      <div class="panel"><strong>Option Tradeability Summary</strong><pre id="tradeability">Loading...</pre></div>
    </section>
    <section class="grid2">
      <div class="panel"><strong>Forecast Validation</strong><pre id="forecastValidation">Loading...</pre></div>
      <div class="panel"><strong>Latest Forecast Signal</strong><pre id="forecastSignal">Loading...</pre></div>
    </section>
    <section class="grid2">
      <div class="panel"><strong>Memory Summary</strong><pre id="memorySummary">Loading...</pre></div>
      <div class="panel"><strong>Strategy Lessons</strong><pre id="strategyMemory">Loading...</pre></div>
    </section>
    <section class="grid2">
      <div class="panel"><strong>Shadow Profit Protection Audit</strong><pre id="profitAudit">Loading...</pre></div>
      <div class="panel"><strong>Strategy Knowledge Base</strong><pre id="strategyKnowledge">Loading...</pre></div>
    </section>
    <section class="panel">
      <strong>LLM Usage / Cost</strong>
      <table><thead><tr><th>Process</th><th>Model</th><th>Calls</th><th>Input</th><th>Cached</th><th>Output</th><th>Total</th><th>Cost</th></tr></thead><tbody id="usageRows"></tbody></table>
    </section>
    <section class="panel" id="instrumentSection">
      <strong id="instrumentTitle">Option Chain Snapshot Given to LLM</strong>
      <div class="small" id="instrumentMeta">-</div>
      <table><thead><tr><th>Instrument</th><th>Type</th><th>Strike</th><th>DTE</th><th>Bid</th><th>Ask</th><th>Mark</th><th>Spread</th><th>Delta</th><th>Volume</th><th>OI</th></tr></thead><tbody id="chainRows"></tbody></table>
    </section>
    <section class="grid2">
      <div class="panel"><strong>Open Orders</strong><table><thead><tr><th>Instrument</th><th>Side</th><th>Amount</th><th>Price</th><th>State</th></tr></thead><tbody id="ordersRows"></tbody></table></div>
      <div class="panel"><strong>Positions</strong><table><thead><tr><th>Instrument</th><th>Size</th><th>Avg</th><th>Mark</th><th>P/L</th></tr></thead><tbody id="positionsRows"></tbody></table></div>
    </section>
    <section class="grid2">
      <div class="panel"><strong>Entry History</strong><table><thead><tr><th>Time</th><th>Action</th><th>Order</th><th>Submitted</th></tr></thead><tbody id="entryHistoryRows"></tbody></table></div>
      <div class="panel"><strong>Exit History</strong><table><thead><tr><th>Time</th><th>Action</th><th>Order</th><th>Submitted</th></tr></thead><tbody id="exitHistoryRows"></tbody></table></div>
    </section>
    <section class="panel"><strong>Raw Audit JSON</strong><pre id="raw">Loading...</pre></section>
  </main>
  <script>
	    const refreshMs = {refresh_seconds * 1000};
	    const chartPrefs = {{ hours: 1, forecastZoom: false, showLine: true, showCandles: true, showSma9: true, showSma21: true }};
    const fmt = new Intl.NumberFormat(undefined, {{ maximumFractionDigits: 6 }});
    const money = new Intl.NumberFormat(undefined, {{ style: "currency", currency: "USD", maximumFractionDigits: 2 }});
    function n(v) {{ return v === null || v === undefined || Number.isNaN(Number(v)) ? "-" : fmt.format(Number(v)); }}
    function m(v) {{ return v === null || v === undefined || Number.isNaN(Number(v)) ? "-" : money.format(Number(v)); }}
    function pct(v) {{ return v === null || v === undefined || Number.isNaN(Number(v)) ? "-" : (Number(v)*100).toFixed(2)+"%"; }}
    function time(v) {{ const d = new Date(v); return v && !Number.isNaN(d.getTime()) ? d.toLocaleString() : "-"; }}
    function set(id, text) {{ document.getElementById(id).textContent = text; }}
    function badge(id, action) {{ const el = document.getElementById(id); el.textContent = action || "-"; el.className = "badge " + (action || "hold"); }}
    async function refresh() {{
      const params = new URLSearchParams(window.location.search);
      const currency = params.get("currency") || "ETH";
      const res = await fetch(`/api/state?currency=${{encodeURIComponent(currency)}}`);
      const data = await res.json();
      render(data);
    }}
    function render(data) {{
      const s = data.summary || {{}};
      const p = s.trader_profile || {{}};
      const entry = s.entry_decision || {{}};
      const exit = s.exit_decision || {{}};
      const account = s.account || {{}};
      const pnl = s.pnl || {{}};
	      set("subtitle", `${{data.currency}} / ${{s.venue || "deribit"}} / ${{s.account_mode || "unknown"}} / ${{s.execution_mode || "unknown"}} / ${{data.state_dir}}`);
	      set("chartTitle", `Forecast vs Actual ${{data.currency}}/${{(s.account?.currency || "USDC")}}`);
      set("updated", `Updated ${{time(data.generated_at_utc)}}`);
      setPnl("totalPnl", pnl.total_pnl_usd);
      setPnl("realizedPnl", pnl.realized_pnl_usd);
      setPnl("unrealizedPnl", pnl.unrealized_pnl_usd);
      setPnl("netPnl", pnl.net_after_llm_cost_usd);
      set("netMeta", `${{pnl.source || "pnl"}} | LLM cost ${{m(pnl.llm_cost_usd || 0)}} | settlement ${{pnl.currency || "-"}}`);
      set("traderName", p.name || "-");
      set("traderRole", p.role || "-");
      set("spot", m(s.latest_underlying_price));
	      set("priceMeta", `${{s.price_data_provider || "unknown"}} / ${{s.price_data_interval || "?"}} source bars | displayed as 5m candles`);
      badge("entryBadge", entry.action || "-");
      set("entryMeta", `confidence ${{n(entry.confidence)}} | age ${{n(s.agent_age_seconds ?? s.entry_age_seconds)}}s`);
      badge("exitBadge", exit.action || "-");
      set("exitMeta", `confidence ${{n(exit.confidence)}} | age ${{n(s.agent_age_seconds ?? s.exit_age_seconds)}}s`);
      set("account", `${{n(account.equity)}} ${{account.currency || "USDC"}}`);
      set("accountMeta", `available ${{n(account.available_funds)}} | rpl ${{n(account.options_session_rpl)}} | upl ${{n(account.options_session_upl)}}`);
      set("exposure", `${{s.open_order_count || 0}} / ${{s.position_count || 0}}`);
      set("memoryCount", String(s.memory_count || 0));
      set("lessonCount", String(s.strategy_lesson_count || data.strategy_memory?.lesson_count || 0));
      set("llmCost", m(s.llm_usage?.total_cost_usd || 0));
      set("llmUsage", `${{s.llm_usage?.total_calls || 0}} calls | ${{n(s.llm_usage?.total_tokens || 0)}} tokens`);
      set("entryDecision", JSON.stringify(data.agent_report?.entry_decision || data.entry_report?.llm_decision || data.agent_report?.llm_decision || {{}}, null, 2));
      set("exitDecision", JSON.stringify(data.agent_report?.exit_decision || data.exit_report?.llm_decision || {{}}, null, 2));
      set("technical", JSON.stringify(data.technical_observations || {{}}, null, 2));
      set("profitPolicy", JSON.stringify(data.adaptive_profit_policy || {{}}, null, 2));
      set("regimeTransition", JSON.stringify({{ short_tape_summary:data.short_tape_summary || {{}}, regime_transition_warning:data.regime_transition_warning || {{}} }}, null, 2));
      set("tradeability", JSON.stringify(data.option_tradeability_summary || {{}}, null, 2));
      set("forecastValidation", JSON.stringify(data.forecast_validation || {{}}, null, 2));
      set("forecastSignal", JSON.stringify(data.chronos_forecast?.preferred_horizon_points || data.chronos_forecast?.horizon_points || [], null, 2));
      set("profitAudit", JSON.stringify(data.shadow_simulation?.profit_protection_audit || {{}}, null, 2));
      set("memorySummary", JSON.stringify((data.trader_memory || []).slice(-12), null, 2));
      set("strategyMemory", JSON.stringify(data.strategy_memory || {{}}, null, 2));
      set("strategyKnowledge", JSON.stringify(data.strategy_knowledge || {{}}, null, 2));
      set("raw", JSON.stringify({{ agent_report:data.agent_report, entry_report:data.entry_report, exit_report:data.exit_report }}, null, 2));
	      renderChart(data.recent_price_bars || [], data.chronos_forecast || {{}});
      renderInstrumentTable(data, s);
      table("usageRows", data.llm_usage?.rows || [], row => [row.process, row.model, n(row.calls), n(row.input_tokens), n(row.cached_input_tokens), n(row.output_tokens), n(row.total_tokens), m(row.estimated_cost_usd)]);
      table("ordersRows", data.open_option_orders || [], row => [row.instrument_name, row.direction, n(row.amount), n(row.price), row.order_state]);
      table("positionsRows", data.option_positions || [], row => [row.instrument_name, n(row.size), n(row.average_price), n(row.mark_price), n(row.floating_profit_loss)]);
      const decisionHistory = (data.agent_history && data.agent_history.length) ? data.agent_history : (data.entry_history || []);
      table("entryHistoryRows", decisionHistory, row => [time(row.checked_at_utc), row.llm_decision?.action, orderText(row.llm_decision?.order), row.order_result?.submitted]);
      table("exitHistoryRows", (data.agent_history && data.agent_history.length) ? data.agent_history : (data.exit_history || []), row => [time(row.checked_at_utc), row.llm_decision?.intent || row.llm_decision?.action, orderText(row.llm_decision?.order || row.llm_decision?.order_id), row.order_result?.submitted]);
    }}
    function orderText(order) {{ return order ? (typeof order === "string" ? order : `${{order.side}} ${{order.amount}} ${{order.instrument_name}} @ ${{order.price}}`) : "-"; }}
    function setPnl(id, value) {{
      const el = document.getElementById(id);
      el.textContent = m(value || 0);
      el.className = "value " + (Number(value || 0) >= 0 ? "good" : "bad");
    }}
    function table(id, rows, mapper) {{
      document.getElementById(id).innerHTML = rows.length ? rows.map(row => `<tr>${{mapper(row).map(cell => `<td>${{cell ?? "-"}}</td>`).join("")}}</tr>`).join("") : `<tr><td colspan="12" class="small">No rows</td></tr>`;
    }}
    function renderInstrumentTable(data, summary) {{
      const assetClass = summary.asset_class || data.agent_report?.market_packet?.asset_class || "";
      if (assetClass === "crypto_spot") {{
        const spot = data.spot_instrument || summary.spot_instrument || {{}};
        set("instrumentTitle", "Spot Crypto Instrument Given to LLM");
        set("instrumentMeta", "Alpaca does not provide crypto options here; this agent can only simulate spot buy/sell decisions for this symbol.");
        table("chainRows", [spot], row => [row.symbol || data.currency, row.asset_class || "crypto_spot", "-", "-", "-", "-", "-", "-", "-", "-", "-"]);
        return;
      }}
      set("instrumentTitle", "Option Chain Snapshot Given to LLM");
      set("instrumentMeta", `${{(data.option_chain || []).length}} option rows`);
      table("chainRows", (data.option_chain || []).slice(0, 80), row => [row.instrument_name, row.option_type, n(row.strike), n(row.dte), n(row.bid), n(row.ask), n(row.mark_price), pct(row.spread_pct), n(row.greeks?.delta), n(row.volume), n(row.open_interest)]);
    }}
	    function renderChart(bars, chronos) {{
	      const el = document.getElementById("priceChart");
	      const rawBars = bars.map(b => ({{
	        t:new Date(b.timestamp).getTime(), o:Number(b.open), h:Number(b.high), l:Number(b.low), c:Number(b.close), v:Number(b.volume || 0)
	      }})).filter(p => Number.isFinite(p.t) && Number.isFinite(p.c));
	      const plottedForecastRows = chronos.preferred_horizon_points || chronos.horizon_points || [];
	      const forecast = (chronos.forecast_path || []).map(p => ({{ t:new Date(p.timestamp).getTime(), y:Number(p.median_price), lo:Number(p.lower_price), hi:Number(p.upper_price) }})).filter(p => Number.isFinite(p.t) && Number.isFinite(p.y));
	      const horizonPoints = plottedForecastRows.map(p => ({{ t:new Date(p.timestamp).getTime(), y:Number(p.predicted_price), lo:Number(p.lower_price), hi:Number(p.upper_price), h:Number(p.horizon_hours), source:chronos.preferred_source || "chronos_median" }})).filter(p => Number.isFinite(p.t) && Number.isFinite(p.y));
	      const lastT = rawBars.length ? rawBars.at(-1).t : Date.now();
	      const maxForecastT = Math.max(...forecast.map(p=>p.t), ...horizonPoints.map(p=>p.t), lastT);
	      const forecastWindowHours = Math.max(1, (maxForecastT - lastT) / 3600000);
	      const historyHours = chartPrefs.forecastZoom ? Math.min(1, Math.max(0.25, forecastWindowHours * .35)) : chartPrefs.hours;
	      const visibleRaw = rawBars.filter(p => p.t >= lastT - historyHours * 3600 * 1000);
	      const candles = aggregateCandles(visibleRaw, 5);
	      const points = candles.map(b => ({{ t:b.t, y:b.c }}));
	      const sma9 = movingAverage(points, 9);
	      const sma21 = movingAverage(points, 21);
	      const chronosStatus = chronos.status ? ` | Chronos ${{chronos.status}} ${{chronos.model || ""}} | plotted ${{chronos.preferred_source || "chronos_median"}}${{chronos.chronos_collapsed ? " | chronos median collapsed" : ""}}` : "";
	      const rangeText = chartPrefs.forecastZoom ? `forecast zoom, history ${{historyHours.toFixed(2)}}h` : `range ${{chartPrefs.hours}}h`;
	      set("chartMeta", points.length ? `source bars ${{rawBars.length}} | displayed 5m candles ${{candles.length}} | ${{rangeText}} | forecast points ${{horizonPoints.length || forecast.length}} | generated ${{time(chronos.generated_at_utc)}}${{chronosStatus}}` : `No price bars${{chronosStatus}}`);
	      if (points.length < 2) {{ el.innerHTML = "<div class='small'>No chart data</div>"; return; }}
	      const w = el.clientWidth || 900, h = el.clientHeight || 520, padL = 86, padR = 42, padT = 34, padB = 66;
	      const asOfT = chronos.as_of_timestamp ? new Date(chronos.as_of_timestamp).getTime() : points.at(-1).t;
	      const asOfPrice = Number.isFinite(Number(chronos.as_of_price)) ? Number(chronos.as_of_price) : points.at(-1).y;
	      const chartPoints = points;
	      const dotPoints = horizonPoints.length ? horizonPoints : forecast.filter((_, i) => i % 15 === 14).slice(0, 4);
	      const allX = chartPoints.map(p=>p.t).concat(forecast.map(p=>p.t), dotPoints.map(p=>p.t), [asOfT]);
	      const allY = candles.flatMap(p=>[p.h,p.l]).concat(sma9.map(p=>p.y), sma21.map(p=>p.y), forecast.flatMap(p=>[p.y,p.lo,p.hi]).filter(Number.isFinite), dotPoints.flatMap(p=>[p.y,p.lo,p.hi]).filter(Number.isFinite), [asOfPrice]).filter(Number.isFinite);
      const minX = Math.min(...allX), maxX = Math.max(...allX);
      const minY = Math.min(...allY), maxY = Math.max(...allY);
      const y0 = minY - (maxY-minY || 1)*0.12, y1 = maxY + (maxY-minY || 1)*0.12;
      const x = v => padL + (v-minX)/Math.max(maxX-minX,1)*(w-padL-padR);
      const y = v => padT + (y1-v)/Math.max(y1-y0,1)*(h-padT-padB);
	      function linePath(pts) {{
	        if (!pts.length) return "";
	        let out = `M${{x(pts[0].t).toFixed(1)}} ${{y(pts[0].y).toFixed(1)}}`;
	        for (let i=1; i<pts.length; i++) out += ` L${{x(pts[i].t).toFixed(1)}} ${{y(pts[i].y).toFixed(1)}}`;
	        return out;
	      }}
	      const actualPath = linePath(chartPoints);
	      const sma9Path = linePath(sma9);
	      const sma21Path = linePath(sma21);
      const forecastLinePoints = [{{t:asOfT, y:asOfPrice}}].concat(dotPoints);
      const fd = forecastLinePoints.map((p,i)=>`${{i?'L':'M'}}${{x(p.t).toFixed(1)}} ${{y(p.y).toFixed(1)}}`).join(" ");
      const band = forecast.length ? forecast.map((p,i)=>`${{i?'L':'M'}}${{x(p.t).toFixed(1)}} ${{y(p.hi).toFixed(1)}}`).join(" ") + " " + forecast.slice().reverse().map(p=>`L${{x(p.t).toFixed(1)}} ${{y(p.lo).toFixed(1)}}`).join(" ") + " Z" : "";
      const gridVals = [0,.25,.5,.75,1].map(g => y0 + (1-g)*(y1-y0));
      const grids = gridVals.map(v => `<line class="grid" x1="${{padL}}" x2="${{w-padR}}" y1="${{y(v).toFixed(1)}}" y2="${{y(v).toFixed(1)}}"/><text x="${{padL-10}}" y="${{(y(v)+4).toFixed(1)}}" text-anchor="end" fill="#9fb1c8" font-size="12">${{m(v)}}</text>`).join("");
      const axes = `<line class="axis" x1="${{padL}}" x2="${{padL}}" y1="${{padT}}" y2="${{h-padB}}"/><line class="axis" x1="${{padL}}" x2="${{w-padR}}" y1="${{h-padB}}" y2="${{h-padB}}"/>`;
      const asof = `<line x1="${{x(asOfT)}}" x2="${{x(asOfT)}}" y1="${{padT}}" y2="${{h-padB}}" stroke="#6aa2ff" stroke-width="1.5" stroke-dasharray="3 4"/>`;
	      const bandPath = forecast.length ? `<path d="${{band}}" fill="rgba(255,92,116,.20)" stroke="none"/>` : "";
	      const forecastPath = dotPoints.length ? `<path d="${{fd}}" fill="none" stroke="#ff5c74" stroke-width="2.2" stroke-dasharray="6 6"/>` : "";
	      const candleWidth = Math.max(2, Math.min(12, (w-padL-padR) / Math.max(candles.length, 1) * .62));
	      const candleSvg = chartPrefs.showCandles ? candles.map(c => {{
	        const up = c.c >= c.o;
	        const color = up ? "#2ed48f" : "#ff5c74";
	        const cx = x(c.t);
	        const top = y(Math.max(c.o, c.c));
	        const bottom = y(Math.min(c.o, c.c));
	        const bodyH = Math.max(1, bottom - top);
	        return `<line x1="${{cx.toFixed(1)}}" x2="${{cx.toFixed(1)}}" y1="${{y(c.h).toFixed(1)}}" y2="${{y(c.l).toFixed(1)}}" stroke="${{color}}" stroke-width="1.2"/><rect x="${{(cx-candleWidth/2).toFixed(1)}}" y="${{top.toFixed(1)}}" width="${{candleWidth.toFixed(1)}}" height="${{bodyH.toFixed(1)}}" fill="${{color}}" opacity=".78"/>`;
	      }}).join("") : "";
	      const actualLine = chartPrefs.showLine ? `<path d="${{actualPath}}" fill="none" stroke="#dbe7f6" stroke-width="2.0" opacity=".85"/>` : "";
	      const sma9Svg = chartPrefs.showSma9 && sma9.length ? `<path d="${{sma9Path}}" fill="none" stroke="#f6b84b" stroke-width="2.0"/><text x="${{w-padR-58}}" y="${{padT+18}}" fill="#f6b84b" font-size="12">SMA 9</text>` : "";
	      const sma21Svg = chartPrefs.showSma21 && sma21.length ? `<path d="${{sma21Path}}" fill="none" stroke="#2ed48f" stroke-width="2.0"/><text x="${{w-padR-58}}" y="${{padT+36}}" fill="#2ed48f" font-size="12">SMA 21</text>` : "";
      const dotSvg = dotPoints.map((p,i) => {{
        const label = horizonLabel(p.h);
        const lx = Math.min(w-padR-70, x(p.t)+12);
        const ly = y(p.y) + (i % 2 === 0 ? -24 : 26);
        return `<circle cx="${{x(p.t)}}" cy="${{y(p.y)}}" r="5.5" fill="#ff5c74"/><text x="${{lx}}" y="${{ly}}" fill="#aebbd0" font-size="13">${{label}}</text><text x="${{lx}}" y="${{ly+18}}" fill="#ffffff" font-size="13" font-weight="800">${{m(p.y)}}</text>`;
      }}).join("");
      const latestLabel = `<circle cx="${{x(asOfT)}}" cy="${{y(asOfPrice)}}" r="5.5" fill="#6aa2ff"/><circle cx="${{x(asOfT)+28}}" cy="${{y(asOfPrice)}}" r="4.5" fill="#dbe7f6"/><text x="${{x(asOfT)+38}}" y="${{y(asOfPrice)+4}}" fill="#dbe7f6" font-size="13">latest ${{m(asOfPrice)}}</text>`;
      const xLabels = `<text x="${{padL}}" y="${{h-18}}" fill="#9fb1c8" font-size="12">${{time(minX)}}</text><text x="${{w-padR}}" y="${{h-18}}" text-anchor="end" fill="#9fb1c8" font-size="12">${{time(maxX)}}</text>`;
	      el.innerHTML = `<svg viewBox="0 0 ${{w}} ${{h}}">${{grids}}${{axes}}${{bandPath}}${{candleSvg}}${{actualLine}}${{sma9Svg}}${{sma21Svg}}${{asof}}${{forecastPath}}${{latestLabel}}${{dotSvg}}${{xLabels}}</svg>`;
	    }}
	    function aggregateCandles(rows, minutes) {{
	      const bucketMs = minutes * 60 * 1000;
	      const buckets = new Map();
	      for (const r of rows) {{
	        const key = Math.floor(r.t / bucketMs) * bucketMs;
	        const b = buckets.get(key);
	        const high = Number.isFinite(r.h) ? r.h : r.c;
	        const low = Number.isFinite(r.l) ? r.l : r.c;
	        const open = Number.isFinite(r.o) ? r.o : r.c;
	        if (!b) buckets.set(key, {{ t:key, o:open, h:high, l:low, c:r.c, v:r.v || 0 }});
	        else {{ b.h = Math.max(b.h, high); b.l = Math.min(b.l, low); b.c = r.c; b.v += r.v || 0; }}
	      }}
	      return Array.from(buckets.values()).sort((a,b)=>a.t-b.t);
	    }}
	    function movingAverage(points, window) {{
	      const out = [];
	      for (let i=window-1; i<points.length; i++) {{
	        let sum = 0;
	        for (let j=i-window+1; j<=i; j++) sum += points[j].y;
	        out.push({{ t: points[i].t, y: sum / window }});
	      }}
	      return out;
	    }}
	    function horizonLabel(hours) {{
      if (!Number.isFinite(hours)) return "";
      const minutes = Math.round(hours * 60);
      if (minutes < 60) return `${{minutes}}m`;
      if (minutes % 60 === 0) return `${{minutes/60}}h`;
      return `${{minutes}}m`;
	    }}
	    document.addEventListener("change", (event) => {{
	      if (event.target?.id === "showLine") chartPrefs.showLine = event.target.checked;
	      if (event.target?.id === "showCandles") chartPrefs.showCandles = event.target.checked;
	      if (event.target?.id === "showSma9") chartPrefs.showSma9 = event.target.checked;
	      if (event.target?.id === "showSma21") chartPrefs.showSma21 = event.target.checked;
	      refresh().catch(err => set("subtitle", "dashboard_refresh_failed: " + err));
	    }});
	    document.addEventListener("click", (event) => {{
	      if (event.target?.id === "zoomIn") {{
	        chartPrefs.forecastZoom = false;
	        chartPrefs.hours = Math.max(0.25, chartPrefs.hours / 2);
	        document.querySelectorAll("button[data-hours]").forEach(btn => btn.classList.toggle("active", Number(btn.dataset.hours) === chartPrefs.hours));
	        document.getElementById("forecastZoom")?.classList.remove("active");
	        refresh().catch(err => set("subtitle", "dashboard_refresh_failed: " + err));
	        return;
	      }}
	      if (event.target?.id === "zoomOut") {{
	        chartPrefs.forecastZoom = false;
	        chartPrefs.hours = Math.min(48, chartPrefs.hours * 2);
	        document.querySelectorAll("button[data-hours]").forEach(btn => btn.classList.toggle("active", Number(btn.dataset.hours) === chartPrefs.hours));
	        document.getElementById("forecastZoom")?.classList.remove("active");
	        refresh().catch(err => set("subtitle", "dashboard_refresh_failed: " + err));
	        return;
	      }}
	      if (event.target?.id === "forecastZoom") {{
	        chartPrefs.forecastZoom = !chartPrefs.forecastZoom;
	        event.target.classList.toggle("active", chartPrefs.forecastZoom);
	        document.querySelectorAll("button[data-hours]").forEach(btn => btn.classList.toggle("active", false));
	        refresh().catch(err => set("subtitle", "dashboard_refresh_failed: " + err));
	        return;
	      }}
	      const button = event.target?.closest?.("button[data-hours]");
	      if (!button) return;
	      chartPrefs.forecastZoom = false;
	      chartPrefs.hours = Number(button.dataset.hours || 24);
	      document.querySelectorAll("button[data-hours]").forEach(btn => btn.classList.toggle("active", btn === button));
	      document.getElementById("forecastZoom")?.classList.remove("active");
	      refresh().catch(err => set("subtitle", "dashboard_refresh_failed: " + err));
	    }});
    refresh().catch(err => set("subtitle", "dashboard_refresh_failed: " + err));
    setInterval(() => refresh().catch(err => set("subtitle", "dashboard_refresh_failed: " + err)), refreshMs);
  </script>
</body>
</html>"""


def build_server(args: argparse.Namespace) -> ThreadingHTTPServer:
    state_dir = Path(args.state_dir)
    refresh_seconds = int(args.refresh_seconds)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            currency = str((params.get("currency") or [args.currency])[0]).upper()
            if parsed.path == "/api/state":
                payload = build_dashboard_state(state_dir=state_dir, currency=currency, max_memory=int(args.max_memory))
                self._send_json(payload)
                return
            if parsed.path in {"/", "/index.html"}:
                self._send_html(dashboard_html(refresh_seconds))
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(_json_sanitize(payload), default=_json_default, allow_nan=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((args.host, int(args.port)), Handler)


def main() -> None:
    args = build_parser().parse_args()
    server = build_server(args)
    print(f"Serving LLM options protocol dashboard at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dashboard for LLM-authoritative Deribit options trader.")
    parser.add_argument("--currency", default="ETH")
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8797)
    parser.add_argument("--refresh-seconds", type=int, default=15)
    parser.add_argument("--max-memory", type=int, default=80)
    return parser


def read_llm_usage(*, currency: str, state_dir: Path | None = None, provider: str | None = None) -> dict[str, Any]:
    root = Path("automated_forecasting_engine/runs/openai_usage")
    wanted = {
        "alpaca_llm_options_entry_agent",
        "alpaca_llm_options_exit_agent",
        "alpaca_llm_options_profit_policy",
        "llm_options_agent",
        "llm_options_entry_agent",
        "llm_options_exit_agent",
        "llm_options_profit_policy",
    }
    wanted_state_dir = str(state_dir) if state_dir else None
    wanted_provider = str(provider).strip().lower() if provider else None
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for path in root.rglob("*.jsonl"):
        for row in _read_jsonl(path):
            context = row.get("context") if isinstance(row.get("context"), dict) else {}
            process = str(context.get("process") or (row.get("routing") or {}).get("process") or "")
            if process not in wanted:
                continue
            if str(context.get("currency") or currency).upper() != currency.upper():
                continue
            row_provider = str(row.get("provider") or context.get("provider") or "").strip().lower()
            if wanted_provider and row_provider and row_provider != wanted_provider:
                continue
            row_state_dir = str(context.get("output_dir") or "")
            if wanted_state_dir:
                if row_state_dir:
                    if row_state_dir != wanted_state_dir:
                        continue
                elif "llm_options_trader_live_shadow_hf" in wanted_state_dir:
                    continue
            model = str(row.get("model") or "unknown")
            key = (process, model)
            bucket = groups.setdefault(
                key,
                {
                    "process": process,
                    "model": model,
                    "calls": 0,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                },
            )
            usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
            details = usage.get("input_tokens_details") if isinstance(usage.get("input_tokens_details"), dict) else {}
            bucket["calls"] += 1
            bucket["input_tokens"] += int(_num(usage.get("input_tokens")))
            bucket["cached_input_tokens"] += int(_num(usage.get("cached_input_tokens") or details.get("cached_tokens")))
            bucket["output_tokens"] += int(_num(usage.get("output_tokens")))
            bucket["total_tokens"] += int(_num(usage.get("total_tokens")))
            bucket["estimated_cost_usd"] += _num(row.get("estimated_cost_usd"))
    rows = sorted(groups.values(), key=lambda item: item["estimated_cost_usd"], reverse=True)
    return {
        "rows": [{**row, "estimated_cost_usd": round(float(row["estimated_cost_usd"]), 6)} for row in rows],
        "total_calls": sum(int(row["calls"]) for row in rows),
        "total_tokens": sum(int(row["total_tokens"]) for row in rows),
        "total_cost_usd": round(sum(float(row["estimated_cost_usd"]) for row in rows), 6),
    }


def _pnl_summary(*, account: dict[str, Any], shadow: dict[str, Any], llm_cost_usd: float) -> dict[str, Any]:
    if shadow:
        realized = _num(shadow.get("realized_pnl"))
        unrealized = _num(shadow.get("unrealized_pnl"))
        source = "shadow_simulation"
    else:
        realized = _num(account.get("options_session_rpl"))
        unrealized = _num(account.get("options_session_upl"))
        source = "broker_account"
    total = realized + unrealized
    return {
        "currency": account.get("currency") or "USDC",
        "source": source,
        "realized_pnl_usd": round(realized, 6),
        "unrealized_pnl_usd": round(unrealized, 6),
        "total_pnl_usd": round(total, 6),
        "llm_cost_usd": round(float(llm_cost_usd or 0.0), 6),
        "net_after_llm_cost_usd": round(total - float(llm_cost_usd or 0.0), 6),
    }


def _decision_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    decision = report.get("llm_decision") if isinstance(report.get("llm_decision"), dict) else {}
    order_result = report.get("order_result") if isinstance(report.get("order_result"), dict) else {}
    if not decision and not order_result:
        return {}
    return {
        "action": decision.get("action"),
        "confidence": decision.get("confidence"),
        "reason": decision.get("reason"),
        "order": decision.get("order"),
        "order_id": decision.get("order_id"),
        "submitted": order_result.get("submitted"),
        "result_reason": order_result.get("reason"),
        "blocks": order_result.get("blocks"),
    }


def _subdecision_summary(report: dict[str, Any], *, decision_key: str, result_key: str) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    decision = report.get(decision_key) if isinstance(report.get(decision_key), dict) else {}
    order_result = report.get(result_key) if isinstance(report.get(result_key), dict) else {}
    if not decision and not order_result:
        return {}
    return {
        "action": decision.get("action"),
        "intent": decision.get("intent"),
        "confidence": decision.get("confidence"),
        "reason": decision.get("reason"),
        "order": decision.get("order"),
        "order_id": decision.get("order_id"),
        "submitted": order_result.get("submitted"),
        "result_reason": order_result.get("reason"),
        "blocks": order_result.get("blocks"),
    }


def _read_report(state_dir: Path, currency: str, kind: str) -> tuple[dict[str, Any], Path | None]:
    if kind == "agent":
        paths = [
            state_dir / f"{currency.upper()}_llm_agent_report.json",
            state_dir / f"{currency.upper()}_alpaca_llm_shadow_report.json",
        ]
    else:
        paths = [state_dir / f"{currency.upper()}_llm_{kind}_agent_report.json"]
    path = next((candidate for candidate in paths if candidate.exists()), paths[0])
    if not path.exists():
        return {}, None
    return _read_json(path), path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _read_jsonl_any(*paths: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_jsonl(path))
    rows.sort(key=lambda row: str(row.get("checked_at_utc") or row.get("generated_at_utc") or ""))
    return rows


def _age_seconds(value: Any) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return max(0.0, (datetime.now(UTC) - parsed.astimezone(UTC)).total_seconds())
    except Exception:
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _num(value: Any) -> float:
    try:
        return float(value or 0)
    except Exception:
        return 0.0


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            return None
        return value
    if hasattr(value, "item"):
        return _json_sanitize(value.item())
    return value


if __name__ == "__main__":
    main()
