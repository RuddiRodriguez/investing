from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


DEFAULT_REPORT_PATH = Path("trade_republic_exports/investment_report_latest.json")
DEFAULT_REFRESH_SECONDS = 60


def main() -> None:
    args = build_parser().parse_args()
    run_server(report_path=args.report, host=args.host, port=args.port, refresh_seconds=args.refresh_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Trade Republic portfolio performance dashboard.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH, help="Path to investment report JSON.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--refresh-seconds", type=int, default=DEFAULT_REFRESH_SECONDS)
    return parser


def build_dashboard_state(report_path: Path) -> dict[str, Any]:
    report, error = read_report(report_path)
    summary = report.get("summary") or {}
    holdings = report.get("holdings") or []
    rows = [summarize_holding(row) for row in holdings if isinstance(row, dict)]
    rows.sort(key=lambda row: float(row.get("current_value") or 0), reverse=True)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_path": str(report_path),
        "report_error": error,
        "report_timestamp": summary.get("report_timestamp"),
        "summary": {
            "holding_count": summary.get("holding_count", len(rows)),
            "total_open_cost_basis": _round(summary.get("total_open_cost_basis")),
            "total_current_value": _round(summary.get("total_current_value")),
            "total_unrealized_pl": _round(summary.get("total_unrealized_pl")),
            "total_unrealized_pl_pct": _round(summary.get("total_unrealized_pl_pct")),
            "total_historical_buy_cash": _round(summary.get("total_historical_buy_cash")),
            "total_historical_sell_cash": _round(summary.get("total_historical_sell_cash")),
            "ticker_resolution_count": summary.get("ticker_resolution_count", 0),
        },
        "holdings": rows,
        "ticker_resolution": report.get("ticker_resolution") or [],
    }


def read_report(report_path: Path) -> tuple[dict[str, Any], str | None]:
    if not report_path.exists():
        return {}, f"report_not_found: {report_path}"
    try:
        parsed = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, str(exc)
    return parsed if isinstance(parsed, dict) else {}, None


def summarize_holding(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": row.get("name"),
        "isin": row.get("isin"),
        "ticker": row.get("ticker"),
        "alpaca_ticker": row.get("alpaca_ticker"),
        "source": row.get("ticker_resolution_source"),
        "quantity": _round(row.get("current_quantity"), 6),
        "current_price": _round(row.get("current_price"), 4),
        "current_value": _round(row.get("current_value")),
        "cost_basis": _round(row.get("open_cost_basis")),
        "unrealized_pl": _round(row.get("unrealized_pl")),
        "unrealized_pl_pct": _round(row.get("unrealized_pl_pct")),
        "paid_price": _round(row.get("weighted_paid_price"), 4),
        "yahoo_buy_close": _round(row.get("weighted_market_price_at_buy"), 4),
        "alpaca_buy_time_price": _round(row.get("alpaca_weighted_price_at_buy_time"), 4),
        "alpaca_diff_pct": _round(row.get("alpaca_paid_vs_market_at_buy_time_pct")),
        "alpaca_status": row.get("alpaca_status"),
    }


def run_server(*, report_path: Path, host: str, port: int, refresh_seconds: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(dashboard_html(refresh_seconds))
                return
            if parsed.path == "/api/state":
                params = parse_qs(parsed.query)
                selected_report = Path(params.get("report", [str(report_path)])[0])
                self._send_json(build_dashboard_state(selected_report))
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=True, default=str).encode("utf-8")
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

    server = ThreadingHTTPServer((host, int(port)), Handler)
    url = f"http://{host}:{port}"
    print(f"Trade Republic dashboard serving {report_path} at {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def dashboard_html(refresh_seconds: int) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trade Republic Performance</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #667085;
      --line: #d9dee8;
      --buy: #067647;
      --sell: #b42318;
      --warn: #9a6700;
      --accent: #2563eb;
      --soft: #eef2f7;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); overflow-x: hidden; }}
    header {{ display: flex; justify-content: space-between; gap: 18px; padding: 18px 22px; border-bottom: 1px solid var(--line); background: var(--panel); position: sticky; top: 0; z-index: 2; }}
    h1 {{ margin: 0; font-size: 20px; line-height: 1.2; }}
    main {{ max-width: 1360px; margin: 0 auto; padding: 18px 22px 32px; }}
    .small, .meta {{ color: var(--muted); font-size: 13px; }}
    .toolbar {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 14px; }}
    .tabs {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    button {{ border: 1px solid var(--line); background: var(--panel); color: var(--text); border-radius: 7px; padding: 8px 10px; font-weight: 650; cursor: pointer; }}
    button.active {{ background: var(--text); color: white; border-color: var(--text); }}
    .cards {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; min-width: 0; }}
    .card, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; max-width: 100%; overflow-x: auto; }}
    .charts {{ display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr) minmax(0, 1fr); gap: 12px; margin-bottom: 14px; min-width: 0; }}
    .chart-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 10px; margin-bottom: 10px; }}
    .chart-title {{ font-weight: 760; font-size: 14px; }}
    canvas {{ display: block; width: 100%; height: 250px; }}
    .label {{ color: var(--muted); font-size: 12px; margin-bottom: 6px; }}
    .value {{ font-size: 22px; font-weight: 760; overflow-wrap: anywhere; }}
    .gain {{ color: var(--buy); }}
    .loss {{ color: var(--sell); }}
    .warn {{ color: var(--warn); }}
    .grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 340px); gap: 14px; align-items: start; min-width: 0; }}
    .grid > *, .cards > * {{ min-width: 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 9px 7px; border-bottom: 1px solid var(--line); text-align: right; vertical-align: middle; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 700; white-space: nowrap; }}
    tr:hover td {{ background: #fafbfc; }}
    .name {{ font-weight: 720; }}
    .sub {{ color: var(--muted); font-size: 12px; margin-top: 3px; }}
    .barwrap {{ width: 120px; height: 8px; background: var(--soft); border-radius: 99px; overflow: hidden; display: inline-block; vertical-align: middle; }}
    .bar {{ height: 100%; background: var(--accent); width: 0%; }}
    .status {{ display: inline-block; min-width: 72px; text-align: center; border: 1px solid var(--line); border-radius: 999px; padding: 4px 8px; color: var(--muted); font-size: 12px; }}
    .status.matched {{ color: var(--buy); border-color: rgba(6,118,71,.35); background: rgba(6,118,71,.06); }}
    .status.no_bars, .status.unsupported_symbol {{ color: var(--warn); border-color: rgba(154,103,0,.35); background: rgba(154,103,0,.06); }}
    pre {{ margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; max-width: 100%; }}
    @media (max-width: 980px) {{ header, .toolbar {{ flex-direction: column; align-items: flex-start; }} main {{ padding: 14px 12px 28px; }} .cards, .charts, .grid {{ grid-template-columns: minmax(0, 1fr); }} canvas {{ height: 230px; }} table {{ font-size: 12px; min-width: 620px; }} th, td {{ padding: 8px 5px; }} .hide-sm {{ display: none; }} }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Trade Republic Performance</h1>
      <div class="small" id="source">Loading report...</div>
    </div>
    <div class="meta"><div id="updated">Loading...</div><div>Auto refresh: {refresh_seconds}s</div></div>
  </header>
  <main>
    <section class="cards">
      <div class="card"><div class="label">Current Value</div><div class="value" id="currentValue">-</div><div class="small">Open holdings</div></div>
      <div class="card"><div class="label">Open Cost</div><div class="value" id="costBasis">-</div><div class="small">Broker cost basis</div></div>
      <div class="card"><div class="label">Unrealized P/L</div><div class="value" id="totalPl">-</div><div class="small" id="totalPlPct">-</div></div>
      <div class="card"><div class="label">Historical Buys</div><div class="value" id="buyCash">-</div><div class="small">For mapped holdings</div></div>
      <div class="card"><div class="label">Holdings</div><div class="value" id="holdingCount">-</div><div class="small" id="tickerResolution">-</div></div>
    </section>
    <section class="charts">
      <div class="panel">
        <div class="chart-head"><div class="chart-title">Allocation By Value</div><div class="small" id="allocationMeta">-</div></div>
        <canvas id="allocationChart"></canvas>
      </div>
      <div class="panel">
        <div class="chart-head"><div class="chart-title">Profit / Loss By Holding</div><div class="small">EUR</div></div>
        <canvas id="plChart"></canvas>
      </div>
      <div class="panel">
        <div class="chart-head"><div class="chart-title">Cost Versus Current</div><div class="small">Total</div></div>
        <canvas id="costValueChart"></canvas>
      </div>
    </section>
    <section class="toolbar">
      <div class="tabs">
        <button id="sortValue" class="active">Value</button>
        <button id="sortPl">P/L</button>
        <button id="sortPlPct">P/L %</button>
        <button id="sortName">Name</button>
      </div>
      <div class="small" id="error"></div>
    </section>
    <section class="grid">
      <div class="panel">
        <table>
          <thead>
            <tr>
              <th>Holding</th>
              <th>Qty</th>
              <th>Value</th>
              <th>Cost</th>
              <th>P/L</th>
              <th>P/L %</th>
              <th class="hide-sm">Paid</th>
              <th class="hide-sm">Alpaca</th>
              <th class="hide-sm">Status</th>
            </tr>
          </thead>
          <tbody id="rows"></tbody>
        </table>
      </div>
      <aside class="panel">
        <strong>Selected Holding</strong>
        <pre id="details">Click a row for details.</pre>
      </aside>
    </section>
  </main>
  <script>
    const refreshMs = {refresh_seconds * 1000};
    let state = null;
    let sortKey = "current_value";
    let sortDir = -1;
    const fmt = new Intl.NumberFormat(undefined, {{ maximumFractionDigits: 2, minimumFractionDigits: 2 }});
    const fmt4 = new Intl.NumberFormat(undefined, {{ maximumFractionDigits: 4, minimumFractionDigits: 2 }});
    function money(v) {{ return v === null || v === undefined || Number.isNaN(Number(v)) ? "-" : fmt.format(Number(v)); }}
    function num(v, d=2) {{ return v === null || v === undefined || Number.isNaN(Number(v)) ? "-" : Number(v).toFixed(d); }}
    function cls(v) {{ return Number(v || 0) > 0 ? "gain" : Number(v || 0) < 0 ? "loss" : ""; }}
    function set(id, value) {{ document.getElementById(id).textContent = value; }}
    function activate(id) {{ for (const b of document.querySelectorAll("button")) b.classList.remove("active"); document.getElementById(id).classList.add("active"); }}
    function color(name) {{ return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }}
    function label(row) {{ return String(row.ticker || row.name || row.isin || "-").slice(0, 16); }}
    function fitCanvas(id) {{
      const canvas = document.getElementById(id);
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return {{ canvas, ctx, width: rect.width, height: rect.height }};
    }}
    function drawEmpty(ctx, width, height) {{
      ctx.fillStyle = color("--muted");
      ctx.font = "13px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      ctx.fillText("No holdings data", 14, height / 2);
    }}
    function drawAllocation(rows) {{
      const {{ ctx, width, height }} = fitCanvas("allocationChart");
      ctx.clearRect(0, 0, width, height);
      const visible = rows.filter(r => Number(r.current_value || 0) > 0).slice(0, 8);
      const total = rows.reduce((sum, r) => sum + Number(r.current_value || 0), 0);
      set("allocationMeta", total ? `${{money(total)}} total` : "-");
      if (!visible.length) return drawEmpty(ctx, width, height);
      const max = Math.max(...visible.map(r => Number(r.current_value || 0)), 1);
      const left = 118;
      const right = 62;
      const rowH = Math.min(24, (height - 18) / visible.length);
      ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      visible.forEach((r, i) => {{
        const y = 12 + i * rowH;
        const value = Number(r.current_value || 0);
        const barW = Math.max(2, (width - left - right) * value / max);
        ctx.fillStyle = color("--muted");
        ctx.fillText(label(r), 0, y + 14);
        ctx.fillStyle = color("--accent");
        ctx.fillRect(left, y + 3, barW, 12);
        ctx.fillStyle = color("--text");
        ctx.fillText(money(value), left + barW + 8, y + 14);
      }});
    }}
    function drawProfitLoss(rows) {{
      const {{ ctx, width, height }} = fitCanvas("plChart");
      ctx.clearRect(0, 0, width, height);
      const visible = rows.filter(r => r.unrealized_pl !== null && r.unrealized_pl !== undefined).slice(0, 8);
      if (!visible.length) return drawEmpty(ctx, width, height);
      const left = 112;
      const right = 54;
      const mid = left + (width - left - right) / 2;
      const maxAbs = Math.max(1, ...visible.map(r => Math.abs(Number(r.unrealized_pl || 0))));
      const rowH = Math.min(24, (height - 18) / visible.length);
      ctx.strokeStyle = color("--line");
      ctx.beginPath();
      ctx.moveTo(mid, 8);
      ctx.lineTo(mid, height - 8);
      ctx.stroke();
      ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      visible.forEach((r, i) => {{
        const y = 12 + i * rowH;
        const value = Number(r.unrealized_pl || 0);
        const barW = Math.max(2, (width - left - right) / 2 * Math.abs(value) / maxAbs);
        ctx.fillStyle = color("--muted");
        ctx.fillText(label(r), 0, y + 14);
        ctx.fillStyle = value >= 0 ? color("--buy") : color("--sell");
        if (value >= 0) ctx.fillRect(mid, y + 3, barW, 12);
        else ctx.fillRect(mid - barW, y + 3, barW, 12);
        ctx.fillStyle = color("--text");
        ctx.fillText(money(value), value >= 0 ? mid + barW + 7 : mid - barW - 48, y + 14);
      }});
    }}
    function drawCostValue(rows) {{
      const {{ ctx, width, height }} = fitCanvas("costValueChart");
      ctx.clearRect(0, 0, width, height);
      const cost = rows.reduce((sum, r) => sum + Number(r.cost_basis || 0), 0);
      const value = rows.reduce((sum, r) => sum + Number(r.current_value || 0), 0);
      const values = [cost, value];
      const names = ["Open cost", "Current"];
      const max = Math.max(...values, 1);
      const base = height - 34;
      const barW = Math.min(74, (width - 68) / 3);
      ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      values.forEach((v, i) => {{
        const x = 38 + i * (barW + 36);
        const h = Math.max(2, (height - 78) * v / max);
        ctx.fillStyle = i === 0 ? color("--muted") : (value >= cost ? color("--buy") : color("--sell"));
        ctx.fillRect(x, base - h, barW, h);
        ctx.fillStyle = color("--text");
        ctx.fillText(money(v), x - 4, base - h - 8);
        ctx.fillStyle = color("--muted");
        ctx.fillText(names[i], x - 4, base + 18);
      }});
      ctx.strokeStyle = color("--line");
      ctx.beginPath();
      ctx.moveTo(24, base);
      ctx.lineTo(width - 18, base);
      ctx.stroke();
    }}
    function renderCharts(rows) {{
      drawAllocation(rows);
      drawProfitLoss(rows);
      drawCostValue(rows);
    }}
    async function load() {{
      const res = await fetch("/api/state", {{ cache: "no-store" }});
      state = await res.json();
      render();
    }}
    function render() {{
      const s = state.summary || {{}};
      set("source", `${{state.report_path || "-"}}`);
      set("updated", `Report: ${{state.report_timestamp || "-"}} | Loaded: ${{state.generated_at || "-"}}`);
      set("currentValue", money(s.total_current_value));
      set("costBasis", money(s.total_open_cost_basis));
      set("totalPl", money(s.total_unrealized_pl));
      document.getElementById("totalPl").className = `value ${{cls(s.total_unrealized_pl)}}`;
      set("totalPlPct", `${{num(s.total_unrealized_pl_pct)}}%`);
      set("buyCash", money(s.total_historical_buy_cash));
      set("holdingCount", String(s.holding_count ?? "-"));
      set("tickerResolution", `Ticker resolutions: ${{s.ticker_resolution_count ?? 0}}`);
      set("error", state.report_error || "");
      const rows = [...(state.holdings || [])].sort((a,b) => {{
        if (sortKey === "name") return sortDir * String(a.name || "").localeCompare(String(b.name || ""));
        return sortDir * (Number(a[sortKey] || 0) - Number(b[sortKey] || 0));
      }});
      const maxAbs = Math.max(1, ...rows.map(r => Math.abs(Number(r.unrealized_pl || 0))));
      renderCharts(rows);
      document.getElementById("rows").innerHTML = rows.map((r, i) => `
        <tr data-index="${{i}}">
          <td><div class="name">${{r.name || r.isin}}</div><div class="sub">${{r.ticker || "-"}} · ${{r.isin || "-"}}</div></td>
          <td>${{num(r.quantity, 6)}}</td>
          <td>${{money(r.current_value)}}</td>
          <td>${{money(r.cost_basis)}}</td>
          <td class="${{cls(r.unrealized_pl)}}">${{money(r.unrealized_pl)}} <span class="barwrap"><span class="bar" style="width:${{Math.min(100, Math.abs(Number(r.unrealized_pl || 0)) / maxAbs * 100)}}%; background:${{Number(r.unrealized_pl || 0) >= 0 ? "var(--buy)" : "var(--sell)"}}"></span></span></td>
          <td class="${{cls(r.unrealized_pl_pct)}}">${{num(r.unrealized_pl_pct)}}%</td>
          <td class="hide-sm">${{money(r.paid_price)}}</td>
          <td class="hide-sm">${{money(r.alpaca_buy_time_price)}}</td>
          <td class="hide-sm"><span class="status ${{r.alpaca_status || ""}}">${{r.alpaca_status || "-"}}</span></td>
        </tr>`).join("");
      [...document.querySelectorAll("tbody tr")].forEach((tr, i) => tr.addEventListener("click", () => set("details", JSON.stringify(rows[i], null, 2))));
      if (rows.length) set("details", JSON.stringify(rows[0], null, 2));
    }}
    document.getElementById("sortValue").onclick = () => {{ sortKey = "current_value"; sortDir = -1; activate("sortValue"); render(); }};
    document.getElementById("sortPl").onclick = () => {{ sortKey = "unrealized_pl"; sortDir = -1; activate("sortPl"); render(); }};
    document.getElementById("sortPlPct").onclick = () => {{ sortKey = "unrealized_pl_pct"; sortDir = -1; activate("sortPlPct"); render(); }};
    document.getElementById("sortName").onclick = () => {{ sortKey = "name"; sortDir = 1; activate("sortName"); render(); }};
    window.addEventListener("resize", () => {{ if (state) render(); }});
    load().catch(err => set("error", String(err)));
    setInterval(() => load().catch(err => set("error", String(err))), refreshMs);
  </script>
</body>
</html>"""


def _round(value: Any, digits: int = 2) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
