from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from market_forecasting_engine.portfolio_overrides import (
    TRADE_REPUBLIC_OVERRIDE_FILE,
    WATCH_STATE_OVERRIDE_FILE,
    apply_open_position_overrides,
    load_portfolio_overrides,
)


DEFAULT_REPORT_PATH = Path("trade_republic_exports/investment_report_latest.json")
DEFAULT_REFRESH_SECONDS = 60
DEFAULT_WATCH_LOG_DIR = Path("automated_forecasting_engine/runs/watch_agent_state/logs")


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
    overrides = load_portfolio_overrides(Path.cwd() / WATCH_STATE_OVERRIDE_FILE, Path.cwd() / TRADE_REPUBLIC_OVERRIDE_FILE)
    report = apply_open_position_overrides(report, overrides)
    summary = report.get("summary") or {}
    holdings = report.get("holdings") or []
    total_current_value = _round(summary.get("total_current_value"))
    rows = [summarize_holding(row, total_current_value=total_current_value) for row in holdings if isinstance(row, dict)]
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
            "manual_position_overrides_applied": summary.get("manual_position_overrides_applied") or [],
            "manual_position_override_note": summary.get("manual_position_override_note"),
        },
        "holdings": rows,
        "ticker_resolution": report.get("ticker_resolution") or [],
        "portfolio_overrides": overrides,
    }


def read_report(report_path: Path) -> tuple[dict[str, Any], str | None]:
    if not report_path.exists():
        return {}, f"report_not_found: {report_path}"
    try:
        parsed = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, str(exc)
    return parsed if isinstance(parsed, dict) else {}, None


def summarize_holding(row: dict[str, Any], *, total_current_value: float | None = None) -> dict[str, Any]:
    current_value = _round(row.get("current_value"))
    return {
        "name": row.get("name"),
        "isin": row.get("isin"),
        "ticker": row.get("ticker"),
        "alpaca_ticker": row.get("alpaca_ticker"),
        "source": row.get("ticker_resolution_source"),
        "quantity": _round(row.get("current_quantity"), 6),
        "current_price": _round(row.get("current_price"), 4),
        "current_value": current_value,
        "cost_basis": _round(row.get("open_cost_basis")),
        "unrealized_pl": _round(row.get("unrealized_pl")),
        "unrealized_pl_pct": _round(row.get("unrealized_pl_pct")),
        "paid_price": _round(row.get("weighted_paid_price"), 4),
        "yahoo_buy_close": _round(row.get("weighted_market_price_at_buy"), 4),
        "yahoo_status": row.get("historical_price_status"),
        "alpaca_buy_time_price": _round(row.get("alpaca_weighted_price_at_buy_time"), 4),
        "alpaca_diff_pct": _round(row.get("alpaca_paid_vs_market_at_buy_time_pct")),
        "alpaca_status": row.get("alpaca_status"),
        "allocation_pct": _allocation_pct(current_value, total_current_value),
    }


def render_static_dashboard_html(state: dict[str, Any]) -> str:
    payload = json.dumps(state, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trade Republic End-of-Day Dashboard</title>
  <style>
    :root {{
      --bg: #f3f2ec;
      --panel: #fffdf7;
      --panel-2: #f7f4ea;
      --text: #18212b;
      --muted: #677381;
      --line: #d7d2c4;
      --gain: #0c7a43;
      --loss: #b83a2f;
      --warn: #9a6700;
      --accent: #0057b8;
      --accent-soft: #d9e8f9;
      --shadow: 0 18px 40px rgba(24, 33, 43, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      background:
        radial-gradient(circle at top left, rgba(0,87,184,0.10), transparent 32%),
        linear-gradient(180deg, #f8f6ef 0%, #ece8dc 100%);
    }}
    .shell {{ max-width: 1440px; margin: 0 auto; padding: 28px 20px 40px; }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,253,247,0.96), rgba(247,244,234,0.92));
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 24px;
      margin-bottom: 18px;
    }}
    .hero-top {{
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: flex-start;
      flex-wrap: wrap;
    }}
    h1 {{ margin: 0 0 8px; font-size: 34px; line-height: 1.05; }}
    .lede {{ margin: 0; color: var(--muted); font-size: 15px; line-height: 1.5; }}
    .meta {{ color: var(--muted); font-size: 13px; text-align: right; line-height: 1.5; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .card, .panel {{
      background: rgba(255,253,247,0.88);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 8px 20px rgba(24, 33, 43, 0.05);
      min-width: 0;
    }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }}
    .value {{ font-size: 28px; font-weight: 700; line-height: 1.1; overflow-wrap: anywhere; }}
    .subvalue {{ color: var(--muted); font-size: 13px; margin-top: 8px; }}
    .gain {{ color: var(--gain); }}
    .loss {{ color: var(--loss); }}
    .warn {{ color: var(--warn); }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.6fr) minmax(300px, 0.8fr);
      gap: 14px;
      align-items: start;
    }}
    .section-title {{ margin: 0 0 12px; font-size: 18px; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 4px 10px;
      border: 1px solid var(--line);
      color: var(--muted);
      background: var(--panel-2);
      font-size: 12px;
      white-space: nowrap;
    }}
    .pill.gain {{ background: rgba(12,122,67,0.08); border-color: rgba(12,122,67,0.25); }}
    .pill.loss {{ background: rgba(184,58,47,0.08); border-color: rgba(184,58,47,0.25); }}
    .pill.warn {{ background: rgba(154,103,0,0.08); border-color: rgba(154,103,0,0.25); }}
    .list {{ display: grid; gap: 10px; }}
    .list-item {{ display: grid; gap: 6px; padding: 12px 0; border-bottom: 1px solid var(--line); }}
    .list-item:last-child {{ border-bottom: 0; padding-bottom: 0; }}
    .row-top {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      flex-wrap: wrap;
    }}
    .name {{ font-weight: 700; font-size: 16px; }}
    .detail {{ color: var(--muted); font-size: 12px; }}
    .allocation {{
      position: relative;
      height: 10px;
      background: var(--accent-soft);
      border-radius: 999px;
      overflow: hidden;
    }}
    .allocation > span {{
      position: absolute;
      inset: 0 auto 0 0;
      background: linear-gradient(90deg, #0057b8, #2f88d6);
      border-radius: 999px;
    }}
    .table-wrap {{ overflow: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 1140px; font-size: 13px; }}
    th, td {{ padding: 11px 10px; border-bottom: 1px solid var(--line); text-align: right; vertical-align: top; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }}
    tr:hover td {{ background: rgba(255,255,255,0.45); }}
    .holding-name {{ font-weight: 700; }}
    .holding-sub {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
    .mono {{ font-family: "SFMono-Regular", Menlo, Monaco, monospace; }}
    .footer-note {{ margin-top: 18px; color: var(--muted); font-size: 13px; }}
    @media (max-width: 1080px) {{
      .cards, .grid {{ grid-template-columns: minmax(0, 1fr); }}
      h1 {{ font-size: 28px; }}
      .meta {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div class="hero-top">
        <div>
          <h1>Trade Republic End-of-Day Dashboard</h1>
          <p class="lede">Read-only portfolio snapshot generated from the local Trade Republic investment report. No broker orders were submitted.</p>
        </div>
        <div class="meta" id="heroMeta"></div>
      </div>
      <div class="cards" id="cards"></div>
    </div>
    <div class="grid">
      <div class="panel">
        <h2 class="section-title">Holdings</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Holding</th>
                <th>Ticker</th>
                <th>ISIN</th>
                <th>Qty</th>
                <th>Current Price</th>
                <th>Current Value</th>
                <th>Allocation</th>
                <th>Cost Basis</th>
                <th>Unrealized P/L</th>
                <th>Unrealized P/L %</th>
                <th>Paid Price</th>
                <th>Yahoo</th>
                <th>Alpaca</th>
              </tr>
            </thead>
            <tbody id="rows"></tbody>
          </table>
        </div>
      </div>
      <div style="display:grid; gap:14px;">
        <div class="panel">
          <h2 class="section-title">Top Holdings By Value</h2>
          <div class="list" id="topHoldings"></div>
        </div>
        <div class="panel">
          <h2 class="section-title">Largest Winners / Losers</h2>
          <div class="list" id="plMoves"></div>
        </div>
        <div class="panel">
          <h2 class="section-title">Ticker / Data Quality</h2>
          <div class="list" id="issues"></div>
        </div>
        <div class="panel" id="forecastPanel" hidden>
          <h2 class="section-title">Latest Forecast / Policy Context</h2>
          <div class="list" id="forecastContext"></div>
        </div>
      </div>
    </div>
    <p class="footer-note">Snapshot file is standalone and embeds the dashboard state directly. Report source stays at <span class="mono" id="reportPath"></span>.</p>
  </div>
  <script>
    const state = {payload};
    const fmtMoney = new Intl.NumberFormat(undefined, {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }});
    function money(v) {{ return v === null || v === undefined || Number.isNaN(Number(v)) ? "-" : fmtMoney.format(Number(v)); }}
    function num(v, d = 2) {{ return v === null || v === undefined || Number.isNaN(Number(v)) ? "-" : Number(v).toFixed(d); }}
    function cls(v) {{ return Number(v || 0) > 0 ? "gain" : Number(v || 0) < 0 ? "loss" : ""; }}
    function text(v) {{ return v === null || v === undefined || v === "" ? "-" : String(v); }}
    function pill(value) {{
      const klass = /matched|ok|open|hold/i.test(String(value || "")) ? "gain" : /partial|warn|unsupported|closed|stale/i.test(String(value || "")) ? "warn" : /error|fail|missing/i.test(String(value || "")) ? "loss" : "";
      return `<span class="pill ${{klass}}">${{text(value)}}</span>`;
    }}
    function renderCards() {{
      const s = state.summary || {{}};
      const plClass = cls(s.total_unrealized_pl);
      const cards = [
        ["Current Value", money(s.total_current_value), `${{text(s.holding_count)}} open holdings`],
        ["Open Cost Basis", money(s.total_open_cost_basis), "Broker open cost basis"],
        ["Unrealized P/L", money(s.total_unrealized_pl), `${{num(s.total_unrealized_pl_pct)}}%`],
        ["Ticker Resolution", String(s.ticker_resolution_count ?? 0), "Resolved tickers in report"],
        ["Report Status", state.report_error ? "Stale / Error" : "Fresh read", state.report_error || "Report parsed successfully"],
      ];
      document.getElementById("cards").innerHTML = cards.map(([label, value, sub], index) => `
        <div class="card">
          <div class="label">${{label}}</div>
          <div class="value ${{index === 2 ? plClass : ""}}">${{value}}</div>
          <div class="subvalue">${{sub}}</div>
        </div>`).join("");
    }}
    function renderMeta() {{
      document.getElementById("heroMeta").innerHTML = `
        <div>Generated: ${{text(state.generated_at)}}</div>
        <div>Report timestamp: ${{text(state.report_timestamp)}}</div>
        <div>Report path: <span class="mono">${{text(state.report_path)}}</span></div>`;
      document.getElementById("reportPath").textContent = text(state.report_path);
    }}
    function renderRows() {{
      const rows = [...(state.holdings || [])];
      document.getElementById("rows").innerHTML = rows.map((row) => `
        <tr>
          <td>
            <div class="holding-name">${{text(row.name)}}</div>
            <div class="holding-sub">${{text(row.source)}}</div>
          </td>
          <td class="mono">${{text(row.ticker)}}</td>
          <td class="mono">${{text(row.isin)}}</td>
          <td>${{num(row.quantity, 6)}}</td>
          <td>${{money(row.current_price)}}</td>
          <td>${{money(row.current_value)}}</td>
          <td>
            <div>${{num(row.allocation_pct)}}%</div>
            <div class="allocation"><span style="width:${{Math.max(0, Math.min(100, Number(row.allocation_pct || 0)))}}%"></span></div>
          </td>
          <td>${{money(row.cost_basis)}}</td>
          <td class="${{cls(row.unrealized_pl)}}">${{money(row.unrealized_pl)}}</td>
          <td class="${{cls(row.unrealized_pl_pct)}}">${{num(row.unrealized_pl_pct)}}%</td>
          <td>${{money(row.paid_price)}}</td>
          <td>${{pill(row.yahoo_status)}}</td>
          <td>${{pill(row.alpaca_status)}}</td>
        </tr>`).join("");
    }}
    function renderTopHoldings() {{
      const rows = [...(state.holdings || [])].slice(0, 5);
      document.getElementById("topHoldings").innerHTML = rows.map((row) => `
        <div class="list-item">
          <div class="row-top"><div class="name">${{text(row.name)}}</div><div>${{money(row.current_value)}}</div></div>
          <div class="detail">${{text(row.ticker)}} · Allocation ${{num(row.allocation_pct)}}% · P/L ${{num(row.unrealized_pl_pct)}}%</div>
          <div class="allocation"><span style="width:${{Math.max(0, Math.min(100, Number(row.allocation_pct || 0)))}}%"></span></div>
        </div>`).join("");
    }}
    function renderPlMoves() {{
      const rows = [...(state.holdings || [])].filter((row) => row.unrealized_pl_pct !== null && row.unrealized_pl_pct !== undefined);
      const winners = [...rows].filter((row) => Number(row.unrealized_pl || 0) > 0).sort((a, b) => Number(b.unrealized_pl_pct || 0) - Number(a.unrealized_pl_pct || 0)).slice(0, 3);
      const losers = [...rows].filter((row) => Number(row.unrealized_pl || 0) < 0).sort((a, b) => Number(a.unrealized_pl_pct || 0) - Number(b.unrealized_pl_pct || 0)).slice(0, 3);
      const combined = [
        ...winners.map((row) => ({{ row, label: "winner" }})),
        ...losers.map((row) => ({{ row, label: "loser" }})),
      ];
      document.getElementById("plMoves").innerHTML = combined.length ? combined.map((entry) => `
        <div class="list-item">
          <div class="row-top"><div class="name">${{text(entry.row.name)}}</div><div>${{pill(entry.label)}}</div></div>
          <div class="row-top"><div class="detail">${{text(entry.row.ticker)}} · ${{num(entry.row.unrealized_pl_pct)}}% · Cost ${{money(entry.row.cost_basis)}} · Value ${{money(entry.row.current_value)}}</div><div class="${{cls(entry.row.unrealized_pl)}}">${{money(entry.row.unrealized_pl)}}</div></div>
        </div>`).join("") : `<div class="detail">No realized winners or losers were available from the current open holdings snapshot.</div>`;
    }}
    function renderIssues() {{
      const reportIssues = (state.report_error ? [{{ label: "report_error", detail: state.report_error }}] : []);
      const resolutionIssues = (state.ticker_resolution || []).map((row) => ({{ label: row.isin || row.ticker || "resolution", detail: row.error || row.reason || JSON.stringify(row) }}));
      const holdingIssues = (state.holdings || []).filter((row) => row.alpaca_status && row.alpaca_status !== "matched" || row.yahoo_status && row.yahoo_status !== "matched").map((row) => ({{
        label: row.ticker || row.name || row.isin || "holding",
        detail: `Yahoo=${{text(row.yahoo_status)}} | Alpaca=${{text(row.alpaca_status)}}`
      }}));
      const issues = [...reportIssues, ...resolutionIssues, ...holdingIssues].slice(0, 12);
      document.getElementById("issues").innerHTML = issues.length ? issues.map((item) => `
        <div class="list-item">
          <div class="row-top"><div class="name">${{text(item.label)}}</div><div>${{pill("attention")}}</div></div>
          <div class="detail">${{text(item.detail)}}</div>
        </div>`).join("") : `<div class="detail">No ticker-resolution or data-quality issues were detected in this snapshot.</div>`;
    }}
    function renderForecastContext() {{
      const rows = state.forecast_policy_context || [];
      if (!rows.length) return;
      document.getElementById("forecastPanel").hidden = false;
      document.getElementById("forecastContext").innerHTML = rows.map((row) => `
        <div class="list-item">
          <div class="row-top"><div class="name">${{text(row.ticker)}}</div><div>${{pill(row.action || row.llm_decision || "unknown")}}</div></div>
          <div class="detail">${{text(row.policy || row.llm_decision)}} · Reason: ${{text(row.reason)}} · Checked: ${{text(row.checked_at)}}</div>
        </div>`).join("");
    }}
    renderMeta();
    renderCards();
    renderRows();
    renderTopHoldings();
    renderPlMoves();
    renderIssues();
    renderForecastContext();
  </script>
</body>
</html>"""


def build_dashboard_summary_markdown(state: dict[str, Any]) -> str:
    summary = state.get("summary") or {}
    holdings = state.get("holdings") or []
    top_holdings = holdings[:5]
    winners = sorted(
        [row for row in holdings if float(row.get("unrealized_pl") or 0) > 0],
        key=lambda row: float(row.get("unrealized_pl_pct") or float("-inf")),
        reverse=True,
    )[:3]
    losers = sorted(
        [row for row in holdings if float(row.get("unrealized_pl") or 0) < 0],
        key=lambda row: float(row.get("unrealized_pl_pct") or float("inf")),
    )[:3]
    issues = _collect_issue_rows(state)
    lines = [
        "# Trade Republic End-of-Day Dashboard",
        "",
        f"- Generated: {state.get('generated_at') or '-'}",
        f"- Report timestamp: {state.get('report_timestamp') or '-'}",
        f"- Report path: `{state.get('report_path') or '-'}`",
        f"- Read-only snapshot: no orders submitted",
        "",
        "## Summary",
        "",
        f"- Total current value: {_money_text(summary.get('total_current_value'))}",
        f"- Total open cost basis: {_money_text(summary.get('total_open_cost_basis'))}",
        f"- Total unrealized P/L: {_signed_money_text(summary.get('total_unrealized_pl'))} ({_pct_text(summary.get('total_unrealized_pl_pct'))})",
        f"- Holding count: {summary.get('holding_count', 0)}",
        f"- Ticker resolution count: {summary.get('ticker_resolution_count', 0)}",
        "",
        "## Top Holdings By Value",
        "",
    ]
    for row in top_holdings:
        lines.append(
            f"- {row.get('name') or row.get('isin')}: {_money_text(row.get('current_value'))} "
            f"({_pct_text(row.get('allocation_pct'))} allocation, {_pct_text(row.get('unrealized_pl_pct'))} P/L)"
        )
    lines.extend(["", "## Biggest Winners", ""])
    if winners:
        for row in winners:
            lines.append(f"- {row.get('name') or row.get('isin')}: {_signed_money_text(row.get('unrealized_pl'))} ({_pct_text(row.get('unrealized_pl_pct'))})")
    else:
        lines.append("- None. No open holding is currently above cost basis.")
    lines.extend(["", "## Biggest Losers", ""])
    if losers:
        for row in losers:
            lines.append(f"- {row.get('name') or row.get('isin')}: {_signed_money_text(row.get('unrealized_pl'))} ({_pct_text(row.get('unrealized_pl_pct'))})")
    else:
        lines.append("- None. No open holding is currently below cost basis.")
    lines.extend(["", "## Ticker / Data-Quality Issues", ""])
    if issues:
        for issue in issues:
            lines.append(f"- {issue['label']}: {issue['detail']}")
    else:
        lines.append("- None detected in report or holding status fields.")
    forecast_rows = state.get("forecast_policy_context") or []
    if forecast_rows:
        lines.extend(["", "## Latest Forecast / Policy Context", ""])
        for row in forecast_rows:
            lines.append(
                f"- {row.get('ticker')}: {row.get('action') or row.get('llm_decision') or '-'} | "
                f"{row.get('policy') or row.get('llm_decision') or '-'} | {row.get('reason') or '-'}"
            )
    return "\n".join(lines) + "\n"


def load_latest_forecast_policy_context(
    *,
    log_dir: Path = DEFAULT_WATCH_LOG_DIR,
    target_day: str | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    resolved_day = target_day or (now or datetime.now().astimezone()).strftime("%Y%m%d")
    if not resolved_day or not log_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob(f"*_{resolved_day}.jsonl")):
        latest = _read_latest_jsonl_record(path)
        if not latest:
            continue
        rows.append(
            {
                "ticker": latest.get("ticker") or _ticker_from_log_path(path),
                "action": latest.get("action"),
                "policy": latest.get("llm_decision"),
                "llm_decision": latest.get("llm_decision"),
                "reason": latest.get("reason"),
                "checked_at": latest.get("checked_at"),
                "market_status": latest.get("market_status"),
                "log_file": str(path),
            }
        )
    rows.sort(key=lambda row: str(row.get("ticker") or ""))
    return rows


def write_dashboard_snapshot_outputs(
    *,
    report_path: Path,
    output_dir: Path,
    watch_log_dir: Path = DEFAULT_WATCH_LOG_DIR,
    now: datetime | None = None,
) -> dict[str, Any]:
    state = build_dashboard_state(report_path)
    state["forecast_policy_context"] = load_latest_forecast_policy_context(log_dir=watch_log_dir, now=now)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "dashboard_state.json"
    html_path = output_dir / "dashboard_snapshot.html"
    summary_path = output_dir / "dashboard_summary.md"
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    html_path.write_text(render_static_dashboard_html(state), encoding="utf-8")
    summary_path.write_text(build_dashboard_summary_markdown(state), encoding="utf-8")
    return {
        "state": state,
        "dashboard_state_path": str(state_path),
        "dashboard_snapshot_path": str(html_path),
        "dashboard_summary_path": str(summary_path),
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


def _allocation_pct(current_value: float | None, total_current_value: float | None) -> float | None:
    if current_value is None or total_current_value in (None, 0):
        return None
    return _round((current_value / total_current_value) * 100)


def _money_text(value: Any) -> str:
    rounded = _round(value)
    return "-" if rounded is None else f"EUR {rounded:,.2f}"


def _signed_money_text(value: Any) -> str:
    rounded = _round(value)
    if rounded is None:
        return "-"
    sign = "+" if rounded > 0 else ""
    return f"{sign}EUR {rounded:,.2f}"


def _pct_text(value: Any) -> str:
    rounded = _round(value)
    if rounded is None:
        return "-"
    sign = "+" if rounded > 0 else ""
    return f"{sign}{rounded:.2f}%"


def _collect_issue_rows(state: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    report_error = state.get("report_error")
    if report_error:
        issues.append({"label": "report_error", "detail": str(report_error)})
    for row in state.get("ticker_resolution") or []:
        if not isinstance(row, dict):
            continue
        label = str(row.get("isin") or row.get("ticker") or "ticker_resolution")
        detail = str(row.get("error") or row.get("reason") or row)
        issues.append({"label": label, "detail": detail})
    for row in state.get("holdings") or []:
        if not isinstance(row, dict):
            continue
        yahoo_status = row.get("yahoo_status")
        alpaca_status = row.get("alpaca_status")
        if yahoo_status and yahoo_status != "matched":
            issues.append({"label": str(row.get("ticker") or row.get("name") or row.get("isin") or "holding"), "detail": f"Yahoo status={yahoo_status}"})
        if alpaca_status and alpaca_status != "matched":
            issues.append({"label": str(row.get("ticker") or row.get("name") or row.get("isin") or "holding"), "detail": f"Alpaca status={alpaca_status}"})
    return issues


def _read_latest_jsonl_record(path: Path) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    latest = parsed
    except OSError:
        return None
    return latest


def _ticker_from_log_path(path: Path) -> str:
    stem = path.stem
    parts = stem.rsplit("_", 2)
    return parts[0] if parts else stem


if __name__ == "__main__":
    main()
