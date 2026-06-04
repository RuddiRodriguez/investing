from __future__ import annotations

import argparse
import json
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_REPORT = Path("automated_forecasting_engine/runs/live_trading/live_account_report.json")


def dashboard_html(refresh_seconds: int) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Trading Read-Only Dashboard</title>
  <style>
    :root {{ --bg:#f5f7fa; --panel:#fff; --text:#182230; --muted:#5d6b7c; --line:#d7dee8; --good:#067647; --bad:#b42318; --warn:#8a4b0f; --accent:#1d4ed8; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--text); }}
    header {{ display:flex; justify-content:space-between; gap:16px; padding:16px 22px; background:var(--panel); border-bottom:1px solid var(--line); position:sticky; top:0; z-index:3; }}
    h1 {{ margin:0; font-size:20px; }}
    main {{ max-width:1500px; margin:0 auto; padding:18px 22px 34px; }}
    .small {{ color:var(--muted); font-size:13px; }}
    .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:14px; }}
    .panel,.card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
    .label {{ color:var(--muted); font-size:12px; margin-bottom:6px; }}
    .value {{ font-size:22px; font-weight:760; overflow-wrap:anywhere; }}
    .badge {{ display:inline-block; border-radius:999px; padding:5px 10px; font-size:12px; font-weight:760; color:white; background:var(--accent); text-transform:uppercase; }}
    .badge.safe {{ background:var(--good); }}
    .tabs {{ display:flex; gap:8px; margin:14px 0; }}
    .tab {{ border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:9px 12px; cursor:pointer; font-weight:700; color:var(--muted); }}
    .tab.active {{ color:white; background:var(--accent); border-color:var(--accent); }}
    .section {{ display:none; }}
    .section.active {{ display:block; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ padding:8px 7px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
    th {{ color:var(--muted); font-weight:700; }}
    .pos {{ color:var(--good); font-weight:700; }}
    .neg {{ color:var(--bad); font-weight:700; }}
    pre {{ margin:0; white-space:pre-wrap; overflow:auto; max-height:520px; font-size:12px; }}
    @media (max-width:1000px) {{ header {{ flex-direction:column; }} .grid {{ grid-template-columns:1fr 1fr; }} }}
    @media (max-width:680px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <div><h1>Live Trading Read-Only Dashboard</h1><div class="small" id="source">Loading...</div></div>
    <div class="small"><div id="updated">Loading...</div><div>Refresh: {refresh_seconds}s</div></div>
  </header>
  <main>
    <section class="grid">
      <div class="card"><div class="label">Safety</div><div class="value"><span class="badge safe">Read Only</span></div><div class="small" id="safety">No order submission</div></div>
      <div class="card"><div class="label">Equity</div><div class="value" id="equity">-</div><div class="small" id="cash">-</div></div>
      <div class="card"><div class="label">Market Value</div><div class="value" id="marketValue">-</div><div class="small" id="costBasis">-</div></div>
      <div class="card"><div class="label">Unrealized P/L</div><div class="value" id="totalPl">-</div><div class="small" id="counts">-</div></div>
    </section>
    <div class="tabs">
      <button class="tab active" data-tab="overview">Overview</button>
      <button class="tab" data-tab="stocks">Stocks</button>
      <button class="tab" data-tab="options">Options</button>
      <button class="tab" data-tab="raw">Raw Report</button>
    </div>
    <section id="overview" class="section active">
      <div class="grid">
        <div class="card"><div class="label">Stock Positions</div><div class="value" id="stockPositionCount">-</div><div class="small" id="stockPl">-</div></div>
        <div class="card"><div class="label">Option Positions</div><div class="value" id="optionPositionCount">-</div><div class="small" id="optionPl">-</div></div>
        <div class="card"><div class="label">Open Orders</div><div class="value" id="openOrders">-</div><div class="small">stocks / options</div></div>
        <div class="card"><div class="label">Recent Orders</div><div class="value" id="recentOrders">-</div><div class="small">last broker rows</div></div>
      </div>
    </section>
    <section id="stocks" class="section">
      <div class="panel"><strong>Stock Positions</strong><table><thead><tr><th>Symbol</th><th>Qty</th><th>Avg Entry</th><th>Current</th><th>Market Value</th><th>Cost</th><th>P/L</th><th>P/L %</th></tr></thead><tbody id="stockPositions"></tbody></table></div>
      <div class="panel" style="margin-top:14px;"><strong>Stock Open Orders</strong><table><thead><tr><th>Symbol</th><th>Side</th><th>Type</th><th>Qty</th><th>Filled</th><th>Limit</th><th>Stop</th><th>Status</th><th>Submitted</th></tr></thead><tbody id="stockOpenOrders"></tbody></table></div>
      <div class="panel" style="margin-top:14px;"><strong>Stock Order History</strong><table><thead><tr><th>Symbol</th><th>Side</th><th>Type</th><th>Qty</th><th>Filled</th><th>Avg Fill</th><th>Status</th><th>Submitted</th><th>Filled</th></tr></thead><tbody id="stockRecentOrders"></tbody></table></div>
    </section>
    <section id="options" class="section">
      <div class="panel"><strong>Option Positions</strong><table><thead><tr><th>Contract</th><th>Type</th><th>Underlying</th><th>Expiry</th><th>Strike</th><th>Qty</th><th>Avg Entry</th><th>Current</th><th>Market Value</th><th>P/L</th><th>P/L %</th></tr></thead><tbody id="optionPositions"></tbody></table></div>
      <div class="panel" style="margin-top:14px;"><strong>Option Open Orders</strong><table><thead><tr><th>Contract</th><th>Type</th><th>Side</th><th>Order Type</th><th>Qty</th><th>Filled</th><th>Limit</th><th>Stop</th><th>Status</th><th>Submitted</th></tr></thead><tbody id="optionOpenOrders"></tbody></table></div>
      <div class="panel" style="margin-top:14px;"><strong>Option Order History</strong><table><thead><tr><th>Contract</th><th>Type</th><th>Side</th><th>Order Type</th><th>Qty</th><th>Filled</th><th>Avg Fill</th><th>Status</th><th>Submitted</th><th>Filled</th></tr></thead><tbody id="optionRecentOrders"></tbody></table></div>
    </section>
    <section id="raw" class="section"><div class="panel"><strong>Full JSON Report</strong><pre id="rawJson">Loading...</pre></div></section>
  </main>
  <script>
    const refreshMs = {refresh_seconds * 1000};
    const usd = new Intl.NumberFormat(undefined, {{ style:"currency", currency:"USD", maximumFractionDigits:2 }});
    const num = new Intl.NumberFormat(undefined, {{ maximumFractionDigits:6 }});
    function money(v) {{ return v === null || v === undefined || Number.isNaN(Number(v)) ? "-" : usd.format(Number(v)); }}
    function fmt(v) {{ return v === null || v === undefined || Number.isNaN(Number(v)) ? "-" : num.format(Number(v)); }}
    function pct(v) {{ return v === null || v === undefined || Number.isNaN(Number(v)) ? "-" : `${{(Number(v)*100).toFixed(2)}}%`; }}
    function time(v) {{ if (!v) return "-"; const d = new Date(v); return Number.isNaN(d.getTime()) ? v : d.toLocaleString(); }}
    function set(id, value) {{ document.getElementById(id).textContent = value; }}
    function plClass(v) {{ return Number(v) > 0 ? "pos" : Number(v) < 0 ? "neg" : ""; }}
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
      }} catch (error) {{
        set("rawJson", `dashboard_refresh_failed: ${{error}}`);
      }}
    }}
    function render(report) {{
      const account = report.account || {{}};
      const overview = report.overview || {{}};
      const stocks = report.stocks || {{ summary:{{}}, positions:[], open_orders:[], recent_orders:[] }};
      const options = report.options || {{ summary:{{}}, positions:[], open_orders:[], recent_orders:[] }};
      set("source", `${{report.venue || "-"}} / ${{report.mode || "-"}}`);
      set("updated", `Updated ${{time(report.checked_at)}}`);
      set("safety", (report.safety || {{}}).policy || "Read-only");
      set("equity", money(account.equity || account.portfolio_value));
      set("cash", `cash ${{money(account.cash)}} | buying power ${{money(account.buying_power)}}`);
      set("marketValue", money(overview.total_market_value));
      set("costBasis", `cost basis ${{money(overview.total_cost_basis)}}`);
      const totalPl = document.getElementById("totalPl");
      totalPl.textContent = money(overview.total_unrealized_pl);
      totalPl.className = `value ${{plClass(overview.total_unrealized_pl)}}`;
      set("counts", `${{overview.position_count || 0}} positions | ${{overview.open_order_count || 0}} open orders`);
      set("stockPositionCount", stocks.summary.position_count || 0);
      set("stockPl", `P/L ${{money(stocks.summary.unrealized_pl)}} | ${{pct(stocks.summary.unrealized_pl_pct_weighted)}}`);
      set("optionPositionCount", options.summary.position_count || 0);
      set("optionPl", `P/L ${{money(options.summary.unrealized_pl)}} | ${{pct(options.summary.unrealized_pl_pct_weighted)}}`);
      set("openOrders", `${{stocks.summary.open_order_count || 0}} / ${{options.summary.open_order_count || 0}}`);
      set("recentOrders", `${{stocks.summary.recent_order_count || 0}} / ${{options.summary.recent_order_count || 0}}`);
      renderPositions("stockPositions", stocks.positions || [], false);
      renderPositions("optionPositions", options.positions || [], true);
      renderOrders("stockOpenOrders", stocks.open_orders || [], false, true);
      renderOrders("stockRecentOrders", stocks.recent_orders || [], false, false);
      renderOrders("optionOpenOrders", options.open_orders || [], true, true);
      renderOrders("optionRecentOrders", options.recent_orders || [], true, false);
      set("rawJson", JSON.stringify(report, null, 2));
    }}
    function renderPositions(id, rows, isOption) {{
      const body = document.getElementById(id); body.innerHTML = "";
      rows.forEach(row => {{
        const opt = row.option_details || {{}};
        const tr = document.createElement("tr");
        if (isOption) {{
          tr.innerHTML = `<td>${{row.symbol || "-"}}</td><td>${{opt.option_type || "-"}}</td><td>${{opt.underlying || "-"}}</td><td>${{opt.expiration || "-"}}</td><td>${{money(opt.strike)}}</td><td>${{fmt(row.qty)}}</td><td>${{money(row.avg_entry_price)}}</td><td>${{money(row.current_price)}}</td><td>${{money(row.market_value)}}</td><td class="${{plClass(row.unrealized_pl)}}">${{money(row.unrealized_pl)}}</td><td class="${{plClass(row.unrealized_pl)}}">${{pct(row.unrealized_pl_pct)}}</td>`;
        }} else {{
          tr.innerHTML = `<td>${{row.symbol || "-"}}</td><td>${{fmt(row.qty)}}</td><td>${{money(row.avg_entry_price)}}</td><td>${{money(row.current_price)}}</td><td>${{money(row.market_value)}}</td><td>${{money(row.cost_basis)}}</td><td class="${{plClass(row.unrealized_pl)}}">${{money(row.unrealized_pl)}}</td><td class="${{plClass(row.unrealized_pl)}}">${{pct(row.unrealized_pl_pct)}}</td>`;
        }}
        body.appendChild(tr);
      }});
      if (!rows.length) body.innerHTML = `<tr><td colspan="12">No rows in latest report.</td></tr>`;
    }}
    function renderOrders(id, rows, isOption, openOnly) {{
      const body = document.getElementById(id); body.innerHTML = "";
      rows.forEach(row => {{
        const opt = row.option_details || {{}};
        const tr = document.createElement("tr");
        let tail = "";
        if (openOnly) {{
          tail = `<td>${{money(row.stop_price)}}</td><td>${{row.status || "-"}}</td><td>${{time(row.submitted_at)}}</td>`;
        }} else {{
          tail = `<td>${{money(row.filled_avg_price)}}</td><td>${{row.status || "-"}}</td><td>${{time(row.submitted_at)}}</td><td>${{time(row.filled_at)}}</td>`;
        }}
        if (isOption) {{
          tr.innerHTML = `<td>${{row.symbol || "-"}}</td><td>${{opt.option_type || "-"}}</td><td>${{row.side || "-"}}</td><td>${{row.type || "-"}}</td><td>${{fmt(row.qty)}}</td><td>${{fmt(row.filled_qty)}}</td><td>${{money(row.limit_price)}}</td>${{tail}}`;
        }} else {{
          tr.innerHTML = `<td>${{row.symbol || "-"}}</td><td>${{row.side || "-"}}</td><td>${{row.type || "-"}}</td><td>${{fmt(row.qty)}}</td><td>${{fmt(row.filled_qty)}}</td><td>${{money(row.limit_price)}}</td>${{tail}}`;
        }}
        body.appendChild(tr);
      }});
      const colspan = isOption ? 10 : 9;
      if (!rows.length) body.innerHTML = `<tr><td colspan="${{colspan}}">No rows in latest report.</td></tr>`;
    }}
    refresh(); setInterval(refresh, refreshMs);
  </script>
</body>
</html>"""


def build_server(*, report_path: Path, host: str, port: int, refresh_seconds: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/" or self.path.startswith("/?"):
                self._send_html(dashboard_html(refresh_seconds))
                return
            if self.path == "/api/report":
                self._send_json(_read_report(report_path))
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
            "mode": "read_only_live_account_report",
            "venue": "missing_report",
            "safety": {"order_submission_enabled": False, "policy": f"Report not found at {path}."},
            "account": {},
            "overview": {},
            "stocks": {"summary": {}, "positions": [], "open_orders": [], "recent_orders": []},
            "options": {"summary": {}, "positions": [], "open_orders": [], "recent_orders": []},
        }
    return parsed if isinstance(parsed, dict) else {}


def _strict_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strict_json(item) for item in value]
    if isinstance(value, float):
        return None if pd.isna(value) else value
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the read-only live trading dashboard.")
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8793)
    parser.add_argument("--refresh-seconds", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = build_server(report_path=Path(args.report_path), host=args.host, port=int(args.port), refresh_seconds=int(args.refresh_seconds))
    print(f"Serving live trading read-only dashboard at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
