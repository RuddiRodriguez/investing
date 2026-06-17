from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd

from market_forecasting_engine.paper_options_dashboard import _strict_json_value, summarize_all_agents


DEFAULT_RUNS_ROOT = Path("automated_forecasting_engine/runs")
DEFAULT_REFRESH_SECONDS = 30


def build_performance_state(*, runs_root: Path, tickers: set[str] | None = None, run_prefixes: set[str] | None = None) -> dict[str, Any]:
    sentinel_state_dir = runs_root / "_portfolio_performance"
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "runs_root": str(runs_root),
        "tickers": sorted(tickers or []),
        "run_prefixes": sorted(run_prefixes or []),
        "portfolio": summarize_all_agents(sentinel_state_dir, tickers=tickers, run_prefixes=run_prefixes),
    }


def dashboard_html(refresh_seconds: int) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Paper Options Performance</title>
  <style>
    :root {{
      --bg: #0b1119;
      --panel: #111a26;
      --panel2: #0f1722;
      --text: #e7eef8;
      --muted: #9fb1c8;
      --line: #263548;
      --win: #2ed48f;
      --loss: #ff5c74;
      --accent: #6aa2ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }}
    header {{ padding: 24px 34px; border-bottom: 1px solid var(--line); background: #0c141f; position: sticky; top: 0; z-index: 2; }}
    h1 {{ margin: 0; font-size: 24px; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 20px 22px 34px; }}
    .small {{ color: var(--muted); font-size: 13px; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; box-shadow: 0 18px 50px rgba(0,0,0,.18); }}
    .hero {{ border-color: #35506f; background: linear-gradient(180deg, #132033, #101925); margin-bottom: 14px; }}
    .heroHead {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 14px; }}
    .status {{ font-size: 28px; font-weight: 850; }}
    .grid {{ display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 10px; }}
    .metric {{ background: rgba(7,12,19,.38); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }}
    .label {{ color: var(--muted); font-size: 12px; margin-bottom: 5px; }}
    .value {{ font-size: 22px; font-weight: 800; overflow-wrap: anywhere; }}
    .win {{ color: var(--win); }}
    .loss {{ color: var(--loss); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 9px 7px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 700; }}
    .blocks {{ color: var(--loss); font-weight: 650; }}
    .muted {{ color: var(--muted); }}
    @media (max-width: 1100px) {{ .grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}
    @media (max-width: 760px) {{ .grid {{ grid-template-columns: 1fr; }} .heroHead {{ flex-direction: column; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Paper Options Performance</h1>
    <div class="small" id="meta">Loading...</div>
  </header>
  <main>
    <section class="panel hero">
      <div class="heroHead">
        <div>
          <strong>All Opened Paper Options Agents</strong>
          <div class="small" id="plain">Waiting for agent data...</div>
        </div>
        <div class="status" id="status">-</div>
      </div>
      <div class="grid">
        <div class="metric"><div class="label">Total Today</div><div class="value" id="total">-</div><div class="small">realized + open</div></div>
        <div class="metric"><div class="label">Realized</div><div class="value" id="realized">-</div><div class="small">closed trades</div></div>
        <div class="metric"><div class="label">Open P/L</div><div class="value" id="openPl">-</div><div class="small">current positions</div></div>
        <div class="metric"><div class="label">Open Exposure</div><div class="value" id="exposure">-</div><div class="small">money currently at risk</div></div>
        <div class="metric"><div class="label">Closed Trades</div><div class="value" id="trades">-</div><div class="small" id="tradesMeta">-</div></div>
        <div class="metric"><div class="label">Submitted Orders</div><div class="value" id="submitted">-</div><div class="small">latest log window</div></div>
        <div class="metric"><div class="label">Current Strategy</div><div class="value" id="strategy">-</div><div class="small" id="strategyWhy">-</div></div>
      </div>
    </section>
    <section class="panel">
      <strong>Agents</strong>
      <table style="margin-top: 10px;">
        <thead>
          <tr><th>Agent</th><th>Status</th><th>Total</th><th>Realized</th><th>Open P/L</th><th>Exposure</th><th>Entries</th><th>Closed</th><th>Strategy / Why</th><th>Latest Decision</th><th>Block / Reason</th><th>Checked</th></tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </section>
  </main>
  <script>
    const refreshMs = {refresh_seconds * 1000};
    const fmt = new Intl.NumberFormat(undefined, {{ maximumFractionDigits: 4 }});
    function price(v) {{ return v === null || v === undefined || Number.isNaN(Number(v)) ? "-" : fmt.format(Number(v)); }}
    function money(v) {{
      if (v === null || v === undefined || Number.isNaN(Number(v))) return "-";
      const n = Number(v);
      return `${{n > 0 ? "+" : ""}}$${{price(n)}}`;
    }}
    function time(v) {{ if (!v) return "-"; const d = new Date(v); return Number.isNaN(d.getTime()) ? v : d.toLocaleString(); }}
    function set(id, value) {{ document.getElementById(id).textContent = value; }}
    function setMoney(id, value) {{
      const el = document.getElementById(id);
      const n = Number(value || 0);
      el.textContent = money(n);
      el.className = `value ${{n > 0 ? "win" : n < 0 ? "loss" : ""}}`;
    }}
    function render(data) {{
      const portfolio = data.portfolio || {{}};
      const totals = portfolio.totals || {{}};
      const rows = portfolio.rows || [];
      const total = Number(totals.total_pnl || 0);
      set("meta", `${{data.runs_root}} | updated ${{time(data.generated_at)}} | refresh ${{Math.round(refreshMs / 1000)}}s`);
      const status = document.getElementById("status");
      status.textContent = total > 0 ? "Winning" : total < 0 ? "Losing" : "Flat";
      status.className = `status ${{total > 0 ? "win" : total < 0 ? "loss" : ""}}`;
      set("plain", `${{totals.agent_count || rows.length}} agents discovered automatically from run folders.`);
      setMoney("total", total);
      setMoney("realized", totals.realized_pnl_today || 0);
      setMoney("openPl", totals.open_pnl || 0);
      set("exposure", `$${{price(totals.open_exposure || 0)}}`);
      set("trades", `${{totals.round_trips_today || 0}}`);
      set("tradesMeta", `${{totals.entries_today || 0}} entries`);
      set("submitted", `${{totals.submitted_orders_in_view || 0}}`);
      const active = rows.find(row => row.action && row.action !== "-") || rows[0] || {{}};
      set("strategy", active.strategy || "-");
      set("strategyWhy", active.strategy_reason || "-");
      const body = document.getElementById("rows");
      body.innerHTML = "";
      rows.forEach(row => {{
        const perf = row.performance || {{}};
        const rowTotal = Number(perf.total_pnl || 0);
        const tr = document.createElement("tr");
        const quality = row.trade_quality || {{}};
        const qualityText = quality.grade ? `quality: ${{quality.grade}}${{quality.score !== undefined ? " " + quality.score : ""}}` : "";
        const forecastText = row.forecast_engine ? `forecast: ${{row.forecast_engine}}${{row.forecast_model ? " / " + row.forecast_model : ""}}${{row.forecast_fallback ? " / fallback" : ""}}` : "";
        tr.innerHTML = `<td><strong>${{row.ticker || "-"}}</strong><div class="small">${{row.state_dir || ""}}</div></td><td class="${{rowTotal > 0 ? "win" : rowTotal < 0 ? "loss" : ""}}">${{rowTotal > 0 ? "Winning" : rowTotal < 0 ? "Losing" : "Flat"}}</td><td class="${{rowTotal > 0 ? "win" : rowTotal < 0 ? "loss" : ""}}">${{money(rowTotal)}}</td><td>${{money(perf.realized_pnl_today || 0)}}</td><td>${{money(perf.open_pnl || 0)}}</td><td>$${{price(perf.open_exposure || 0)}}</td><td>${{perf.entries_today || 0}}</td><td>${{perf.round_trips_today || 0}}</td><td><strong>${{row.strategy || "-"}}</strong><div class="small">${{row.strategy_reason || "-"}}</div><div class="small">${{forecastText}}</div><div class="small">${{qualityText}}</div></td><td>${{row.action || "-"}} ${{row.forecast_direction ? "(" + row.forecast_direction + ")" : ""}}</td><td class="blocks">${{(row.blocks || []).join(", ") || "no block"}}</td><td class="muted">${{time(row.checked_at)}}</td>`;
        body.appendChild(tr);
      }});
      if (!rows.length) {{
        const tr = document.createElement("tr");
        tr.innerHTML = `<td colspan="12">No paper options agent reports found yet.</td>`;
        body.appendChild(tr);
      }}
    }}
    async function refresh() {{
      try {{
        const response = await fetch("/api/performance", {{ cache: "no-store" }});
        render(await response.json());
      }} catch (error) {{
        set("plain", `dashboard_refresh_failed: ${{error}}`);
      }}
    }}
    refresh();
    setInterval(refresh, refreshMs);
  </script>
</body>
</html>"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve portfolio performance for Alpaca paper options agents.")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8806)
    parser.add_argument("--refresh-seconds", type=int, default=DEFAULT_REFRESH_SECONDS)
    parser.add_argument("--tickers", default="", help="Optional comma-separated ticker filter for the portfolio view.")
    parser.add_argument("--run-prefixes", default="", help="Optional comma-separated run folder prefixes to include.")
    return parser


def build_server(args: argparse.Namespace) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/" or self.path.startswith("/?"):
                self._send_html(dashboard_html(int(args.refresh_seconds)))
                return
            if self.path == "/api/performance":
                tickers = _parse_tickers(str(args.tickers or ""))
                run_prefixes = _parse_prefixes(str(args.run_prefixes or ""))
                self._send_json(build_performance_state(runs_root=Path(args.runs_root), tickers=tickers, run_prefixes=run_prefixes))
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


def _parse_tickers(raw: str) -> set[str]:
    return {item.strip().upper() for item in raw.split(",") if item.strip()}


def _parse_prefixes(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def main() -> None:
    args = build_parser().parse_args()
    pd.set_option("mode.copy_on_write", True)
    server = build_server(args)
    print(f"Serving paper options performance dashboard at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
