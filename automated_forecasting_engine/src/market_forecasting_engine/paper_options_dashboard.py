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
            "contract": selected.get("symbol"),
            "contract_name": selected.get("name"),
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
        },
        "history": history,
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
    .grid2 {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(360px, 0.8fr); gap: 14px; }}
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
      <div class="card"><div class="label">Contract</div><div class="value" id="contract">-</div><div class="small" id="contractMeta">-</div></div>
      <div class="card"><div class="label">Entry / Risk</div><div class="value" id="entry">-</div><div class="small" id="risk">-</div></div>
    </section>
    <section class="cards">
      <div class="card"><div class="label">Take Profit</div><div class="value" id="takeProfit">-</div><div class="small">Autonomous sell limit</div></div>
      <div class="card"><div class="label">Stop Loss</div><div class="value" id="stopLoss">-</div><div class="small">Autonomous stop-limit</div></div>
      <div class="card"><div class="label">Market</div><div class="value" id="market">-</div><div class="small" id="nextOpen">-</div></div>
      <div class="card"><div class="label">Order Result</div><div class="value" id="orderResult">-</div><div class="small" id="checkedAt">-</div></div>
    </section>
    <section class="grid2">
      <div class="panel">
        <strong>Recent Decisions</strong>
        <table>
          <thead><tr><th>Checked</th><th>Action</th><th>Contract</th><th>Entry</th><th>Blocks</th><th>Submitted</th></tr></thead>
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
    function time(v) {{ if (!v) return "-"; const d = new Date(v); return Number.isNaN(d.getTime()) ? v : d.toLocaleString(); }}
    function set(id, value) {{ document.getElementById(id).textContent = value; }}
    function summarize(row) {{
      const plan = row.option_trade_plan || {{}};
      const selected = plan.selected_contract || {{}};
      const order = plan.order || {{}};
      return {{
        checked: row.checked_at,
        action: plan.action || "-",
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
      set("contract", s.contract || "-");
      set("contractMeta", s.contract_name || "-");
      set("entry", `${{s.entry_type || "-"}} @ ${{price(s.entry_limit)}}`);
      const sizing = s.sizing || {{}};
      set("risk", `qty ${{sizing.qty ?? "-"}} | debit ${{price(s.estimated_debit)}} | budget ${{price(sizing.budget)}}`);
      set("takeProfit", price(s.take_profit));
      set("stopLoss", `${{price(s.stop_price)}} / ${{price(s.stop_limit)}}`);
      set("market", s.market_is_open ? "open" : "closed");
      set("nextOpen", `next open: ${{time(s.next_open)}}`);
      set("orderResult", s.order_submitted ? "submitted" : "not submitted");
      set("checkedAt", time(s.checked_at));
      const rows = document.getElementById("historyRows");
      rows.innerHTML = "";
      (data.history || []).slice().reverse().forEach(row => {{
        const r = summarize(row);
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${{time(r.checked)}}</td><td>${{r.action}}</td><td>${{r.contract}}</td><td>${{price(r.entry)}}</td><td class="blocks">${{r.blocks.join(", ")}}</td><td>${{r.submitted}}</td>`;
        rows.appendChild(tr);
      }});
      set("details", JSON.stringify({{ summary: s, report_path: data.report_path, state_path: data.state_path, log_path: data.log_path, report: data.report }}, null, 2));
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


def _query_value(params: dict[str, list[str]], key: str, default: str) -> str:
    values = params.get(key)
    if not values:
        return default
    return values[0] or default


if __name__ == "__main__":
    main()
