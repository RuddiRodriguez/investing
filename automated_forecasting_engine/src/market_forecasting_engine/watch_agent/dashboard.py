from __future__ import annotations

import argparse
import json
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


DEFAULT_LOG_DIR = Path("automated_forecasting_engine/runs/watch_agent_state/logs")
DEFAULT_REFRESH_SECONDS = 10


def read_watch_logs(log_dir: Path, max_history: int = 50) -> dict[str, Any]:
    rows = []
    errors = []
    for path in sorted(log_dir.glob("*.jsonl")):
        try:
            file_rows = _read_jsonl(path)
        except OSError as exc:
            errors.append({"file": str(path), "error": str(exc)})
            continue
        for row in file_rows:
            row["_log_file"] = str(path)
            rows.append(row)
    rows.sort(key=lambda row: str(row.get("checked_at") or ""))
    latest_by_key: dict[str, dict[str, Any]] = {}
    history_by_key: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ticker = str(row.get("ticker") or "UNKNOWN")
        profile = str(row.get("profile") or "default")
        key = f"{ticker}_{profile}"
        latest_by_key[key] = row
        history_by_key.setdefault(key, []).append(row)
    latest = sorted(latest_by_key.values(), key=lambda row: str(row.get("ticker") or ""))
    histories = {
        key: value[-max_history:]
        for key, value in sorted(history_by_key.items(), key=lambda item: item[0])
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "log_dir": str(log_dir),
        "log_files": [str(path) for path in sorted(log_dir.glob("*.jsonl"))],
        "latest": latest,
        "histories": histories,
        "errors": errors,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                rows.append(
                    {
                        "ticker": "UNKNOWN",
                        "profile": "unknown",
                        "checked_at": None,
                        "action": "PARSE_ERROR",
                        "reason": f"{path.name}:{line_number}: {exc}",
                        "_log_file": str(path),
                    }
                )
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def dashboard_html(refresh_seconds: int) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Watch Agent Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #18202a;
      --muted: #687386;
      --line: #d8dde6;
      --buy: #087f5b;
      --sell: #b42318;
      --hold: #775d00;
      --accent: #1f6feb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 22px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    h1 {{ margin: 0; font-size: 20px; font-weight: 650; }}
    .meta {{ color: var(--muted); font-size: 13px; text-align: right; }}
    main {{ padding: 18px 22px 28px; max-width: 1180px; margin: 0 auto; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .card-head {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .ticker {{ font-size: 24px; font-weight: 700; line-height: 1; }}
    .profile {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
    .badge {{
      min-width: 78px;
      text-align: center;
      border-radius: 999px;
      padding: 7px 10px;
      font-weight: 750;
      font-size: 13px;
      color: #fff;
    }}
    .BUY {{ background: var(--buy); }}
    .SELL {{ background: var(--sell); }}
    .HOLD {{ background: var(--hold); }}
    .PARSE_ERROR {{ background: var(--sell); }}
    .price {{ font-size: 34px; font-weight: 760; margin: 8px 0 2px; }}
    .reason {{ color: var(--muted); font-size: 14px; min-height: 20px; }}
    .fields {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 12px;
      margin-top: 16px;
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }}
    .field span {{ display: block; color: var(--muted); font-size: 12px; }}
    .field strong {{ display: block; font-size: 14px; margin-top: 2px; overflow-wrap: anywhere; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 8px 6px;
      white-space: nowrap;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    .history {{ margin-top: 20px; }}
    .empty {{
      background: var(--panel);
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 24px;
      color: var(--muted);
    }}
    @media (max-width: 620px) {{
      header {{ align-items: flex-start; flex-direction: column; }}
      .meta {{ text-align: left; }}
      main {{ padding: 14px; }}
      .fields {{ grid-template-columns: 1fr; }}
      .price {{ font-size: 30px; }}
      table {{ display: block; overflow-x: auto; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Watch Agent Dashboard</h1>
      <div class="meta" id="source"></div>
    </div>
    <div class="meta">
      <div id="updated">Loading...</div>
      <div>Refresh: {refresh_seconds}s</div>
    </div>
  </header>
  <main>
    <section class="grid" id="latest"></section>
    <section class="history card">
      <div class="card-head">
        <div>
          <div class="ticker" style="font-size:18px">Recent Checks</div>
          <div class="profile">Last records from each ticker/profile log</div>
        </div>
      </div>
      <div id="history"></div>
    </section>
  </main>
  <script>
    const refreshMs = {refresh_seconds * 1000};
    const fmt = new Intl.NumberFormat(undefined, {{ maximumFractionDigits: 2 }});
    function money(value) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return fmt.format(Number(value));
    }}
    function time(value) {{
      if (!value) return "-";
      const date = new Date(value);
      return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
    }}
    function cls(action) {{
      return ["BUY", "SELL", "HOLD", "PARSE_ERROR"].includes(action) ? action : "HOLD";
    }}
    function card(row) {{
      return `<article class="card">
        <div class="card-head">
          <div>
            <div class="ticker">${{row.ticker || "UNKNOWN"}}</div>
            <div class="profile">${{row.profile || "default"}} · ${{time(row.checked_at)}}</div>
          </div>
          <div class="badge ${{cls(row.action)}}">${{row.action || "-"}}</div>
        </div>
        <div class="price">${{money(row.price)}}</div>
        <div class="reason">${{row.reason || ""}}</div>
        <div class="fields">
          <div class="field"><span>LLM Decision</span><strong>${{row.llm_decision || "-"}}</strong></div>
          <div class="field"><span>Confidence</span><strong>${{row.llm_confidence ?? "-"}}</strong></div>
          <div class="field"><span>Buy Near</span><strong>${{money(row.buy_near)}}</strong></div>
          <div class="field"><span>Buy Above</span><strong>${{money(row.buy_above)}}</strong></div>
          <div class="field"><span>Sell Near</span><strong>${{money(row.sell_near)}}</strong></div>
          <div class="field"><span>Stop Loss</span><strong>${{money(row.stop_loss)}}</strong></div>
          <div class="field"><span>Take Profit</span><strong>${{money(row.take_profit)}}</strong></div>
          <div class="field"><span>Refreshed</span><strong>${{row.forecast_refreshed_this_run ? "yes" : "no"}}</strong></div>
        </div>
      </article>`;
    }}
    function historyTable(rows) {{
      const all = Object.values(rows.histories || {{}}).flat().sort((a, b) => String(b.checked_at || "").localeCompare(String(a.checked_at || ""))).slice(0, 30);
      if (!all.length) return '<div class="empty">No watch-agent records found.</div>';
      return `<table><thead><tr><th>Time</th><th>Ticker</th><th>Action</th><th>Price</th><th>Reason</th><th>Printed</th></tr></thead><tbody>${{all.map(row => `
        <tr>
          <td>${{time(row.checked_at)}}</td>
          <td>${{row.ticker || "-"}}</td>
          <td>${{row.action || "-"}}</td>
          <td>${{money(row.price)}}</td>
          <td>${{row.reason || "-"}}</td>
          <td>${{row.printed ? "yes" : "no"}}</td>
        </tr>`).join("")}}</tbody></table>`;
    }}
    async function load() {{
      try {{
        const response = await fetch('/api/state', {{ cache: 'no-store' }});
        const data = await response.json();
        document.getElementById('source').textContent = data.log_dir || "";
        document.getElementById('updated').textContent = `Updated ${{time(data.generated_at)}}`;
        const latest = data.latest || [];
        document.getElementById('latest').innerHTML = latest.length ? latest.map(card).join("") : '<div class="empty">No watch-agent records found.</div>';
        document.getElementById('history').innerHTML = historyTable(data);
      }} catch (error) {{
        document.getElementById('updated').textContent = `Load failed: ${{error}}`;
      }}
    }}
    load();
    setInterval(load, refreshMs);
  </script>
</body>
</html>"""


class WatchDashboardHandler(BaseHTTPRequestHandler):
    log_dir: Path = DEFAULT_LOG_DIR
    refresh_seconds: int = DEFAULT_REFRESH_SECONDS

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_text(dashboard_html(self.refresh_seconds), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/state":
            query = parse_qs(parsed.query)
            max_history = int(query.get("max_history", ["50"])[0])
            payload = read_watch_logs(self.log_dir, max_history=max(1, min(max_history, 500)))
            self._send_json(payload)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_text(self, payload: str, content_type: str) -> None:
        encoded = payload.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: dict[str, Any]) -> None:
        self._send_text(json.dumps(payload, sort_keys=True, default=str), "application/json; charset=utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a minimal watch-agent JSONL dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--refresh-seconds", type=int, default=DEFAULT_REFRESH_SECONDS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    handler = type(
        "ConfiguredWatchDashboardHandler",
        (WatchDashboardHandler,),
        {
            "log_dir": Path(args.log_dir),
            "refresh_seconds": max(1, int(args.refresh_seconds)),
        },
    )
    server = ThreadingHTTPServer((args.host, int(args.port)), handler)
    print(f"Watch dashboard: http://{args.host}:{args.port}")
    print(f"Reading logs from: {Path(args.log_dir)}")
    server.serve_forever()


if __name__ == "__main__":
    main()
