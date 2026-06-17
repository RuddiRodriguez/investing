from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from market_forecasting_engine.data_providers import DataRequest, load_prices_with_provider
from market_forecasting_engine.portfolio_overrides import is_sold_ticker, load_portfolio_overrides


DEFAULT_LOG_DIR = Path("automated_forecasting_engine/runs/watch_agent_state/logs")
DEFAULT_REFRESH_SECONDS = 10
CHART_RANGES = {
    "1M": 31,
    "3M": 93,
    "6M": 186,
    "1Y": 366,
    "5Y": 366 * 5,
}
CHART_CACHE_SECONDS = 15 * 60
_CHART_CACHE: dict[tuple[str, str], tuple[datetime, dict[str, Any]]] = {}


def read_watch_logs(log_dir: Path, max_history: int = 50) -> dict[str, Any]:
    rows = []
    errors = []
    overrides = load_portfolio_overrides(log_dir.parent / "portfolio_overrides.json")
    for path in sorted(log_dir.glob("*.jsonl")):
        try:
            file_rows = _read_jsonl(path)
        except OSError as exc:
            errors.append({"file": str(path), "error": str(exc)})
            continue
        for row in file_rows:
            if is_sold_ticker(str(row.get("ticker") or ""), overrides):
                continue
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
    for placeholder in _portfolio_placeholders(log_dir, overrides=overrides):
        ticker = str(placeholder.get("ticker") or "UNKNOWN")
        profile = str(placeholder.get("profile") or "default")
        key = f"{ticker}_{profile}"
        latest_by_key.setdefault(key, placeholder)
    latest = sorted(latest_by_key.values(), key=lambda row: str(row.get("ticker") or ""))
    histories = {
        key: value[-max_history:]
        for key, value in sorted(history_by_key.items(), key=lambda item: item[0])
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "log_dir": str(log_dir),
        "log_files": [
            str(path)
            for path in sorted(log_dir.glob("*.jsonl"))
            if not is_sold_ticker(_ticker_from_log_filename(path), overrides)
        ],
        "latest": latest,
        "histories": histories,
        "errors": errors,
        "portfolio_overrides": overrides,
    }


def _portfolio_placeholders(log_dir: Path, *, overrides: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    context_dir = log_dir.parent / "portfolio_contexts"
    overrides = overrides or {}
    placeholders = []
    for path in sorted(context_dir.glob("*.json")):
        try:
            context = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(context, dict):
            continue
        position = context.get("position", {}) if isinstance(context.get("position"), dict) else {}
        listing = context.get("listing", {}) if isinstance(context.get("listing"), dict) else {}
        ticker = str(context.get("ticker") or "").upper()
        if not ticker:
            continue
        if is_sold_ticker(ticker, overrides):
            continue
        profile = _profile_from_context_path(path)
        placeholders.append(
            {
                "checked_at": None,
                "ticker": ticker,
                "profile": profile,
                "holding_status": position.get("holding_status") or "owned",
                "price": position.get("current_price"),
                "action": "STARTING",
                "reason": "Waiting for first watch-agent decision log.",
                "portfolio_context": context,
                "portfolio_name": context.get("name"),
                "portfolio_isin": context.get("isin"),
                "portfolio_broker": context.get("broker") or "trade_republic",
                "portfolio_quantity": position.get("quantity"),
                "portfolio_entry_price": position.get("avg_cost"),
                "portfolio_position_value": position.get("current_value"),
                "portfolio_unrealized_pl": position.get("unrealized_pl"),
                "portfolio_unrealized_pl_pct": position.get("unrealized_pl_pct"),
                "market_status": "pending_first_check",
                "asset_class": "crypto" if ticker.endswith(("-USD", "-USDT")) else "stock",
                "price_provider": listing.get("price_provider"),
                "forecast_refreshed_this_run": False,
            }
        )
    return placeholders


def _profile_from_context_path(path: Path) -> str:
    stem = path.stem
    for suffix in ("_aggressive", "_medium", "_conservative"):
        if stem.endswith(suffix):
            return suffix.removeprefix("_")
    return "medium"


def _ticker_from_log_filename(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:-2])
    return path.stem


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


def read_price_chart(ticker: str, range_key: str = "3M") -> dict[str, Any]:
    normalized_range = str(range_key or "3M").upper()
    if normalized_range not in CHART_RANGES:
        normalized_range = "3M"
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return {"ticker": symbol, "range": normalized_range, "rows": [], "error": "missing_ticker"}

    cache_key = (symbol, normalized_range)
    now = datetime.now().astimezone()
    cached = _CHART_CACHE.get(cache_key)
    if cached and (now - cached[0]).total_seconds() < CHART_CACHE_SECONDS:
        return cached[1]

    requested_days = CHART_RANGES[normalized_range]
    start = (now.date() - timedelta(days=max(requested_days * 2, requested_days + 30))).isoformat()
    try:
        result = _load_chart_prices(symbol=symbol, start=start)
        frame = result.frame.copy()
        cutoff = pd.Timestamp(now).tz_convert(None) - timedelta(days=requested_days)
        frame = frame[frame.index >= cutoff].copy()
        if not frame.empty and pd.Timestamp(frame.index.min()).tz_localize(None) > cutoff + timedelta(days=min(10, requested_days // 4)):
            wider_start = (now.date() - timedelta(days=max(requested_days * 4, requested_days + 120))).isoformat()
            wider_result = _load_chart_prices(symbol=symbol, start=wider_start)
            wider_frame = wider_result.frame.copy()
            wider_frame = wider_frame[wider_frame.index >= cutoff].copy()
            if len(wider_frame) > len(frame):
                result = wider_result
                frame = wider_frame
        if frame.empty:
            frame = result.frame.tail(max(5, min(CHART_RANGES[normalized_range], len(result.frame)))).copy()
        rows = []
        for index, row in frame.iterrows():
            close = _number_or_none(row.get("close"))
            if close is None:
                continue
            rows.append(
                {
                    "date": index.isoformat() if hasattr(index, "isoformat") else str(index),
                    "close": close,
                    "volume": _number_or_none(row.get("volume")) or 0.0,
                }
            )
        payload = {
            "ticker": symbol,
            "range": normalized_range,
            "provider": "yahoo",
            "generated_at": now.isoformat(),
            "rows": rows,
            "error": None,
        }
    except Exception as exc:
        payload = {
            "ticker": symbol,
            "range": normalized_range,
            "provider": "yahoo",
            "generated_at": now.isoformat(),
            "rows": [],
            "error": str(exc),
        }
    _CHART_CACHE[cache_key] = (now, payload)
    return payload


def _load_chart_prices(*, symbol: str, start: str):
    return load_prices_with_provider(
        "yahoo",
        DataRequest(
            ticker=symbol,
            start=start,
            interval="1d",
            target_column="close",
            adjustment_policy="auto_adjust",
        ),
    )


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in {float("inf"), float("-inf")} else None


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
      grid-template-columns: 1fr;
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
    .STARTING {{ background: var(--accent); }}
    .PARSE_ERROR {{ background: var(--sell); }}
    .price {{ font-size: 34px; font-weight: 760; margin: 8px 0 2px; }}
    .reason {{ color: var(--muted); font-size: 14px; min-height: 20px; }}
    .card-layout {{
      display: grid;
      grid-template-columns: minmax(330px, 0.95fr) minmax(420px, 1.35fr);
      gap: 14px;
      align-items: stretch;
    }}
    .chart-panel {{
      border-left: 1px solid var(--line);
      padding-left: 14px;
      min-height: 300px;
    }}
    .range-controls {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;
    }}
    .range-buttons {{ display: flex; gap: 6px; flex-wrap: wrap; }}
    .range-button {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      padding: 5px 8px;
      font-size: 12px;
      cursor: pointer;
    }}
    .range-button.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }}
    .chart-title {{ font-size: 13px; color: var(--muted); }}
    .chart-canvas {{
      width: 100%;
      height: 260px;
      display: block;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    .badge-label {{ color: var(--muted); font-size: 11px; text-align: center; margin-bottom: 4px; }}
    .conflict {{
      margin-top: 10px;
      border: 1px solid #f2c94c;
      background: #fff8db;
      color: #5f4700;
      border-radius: 6px;
      padding: 8px 10px;
      font-size: 13px;
    }}
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
      .card-layout {{ grid-template-columns: 1fr; }}
      .chart-panel {{ border-left: none; border-top: 1px solid var(--line); padding-left: 0; padding-top: 12px; }}
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
      return ["BUY", "SELL", "HOLD", "STARTING", "PARSE_ERROR"].includes(action) ? action : "HOLD";
    }}
    function hasDecisionConflict(row) {{
      const llm = String(row.llm_decision || "").toUpperCase();
      const action = String(row.action || "").toUpperCase();
      return (action === "BUY" && llm && llm !== "BUY") || (action === "SELL" && llm && llm !== "SELL");
    }}
    const selectedRanges = new Map();
    function chartKey(row) {{
      return `${{row.ticker || "UNKNOWN"}}_${{row.profile || "default"}}`.replace(/[^a-zA-Z0-9_-]/g, "_");
    }}
    function selectedRangeFor(key) {{
      return selectedRanges.get(key) || "3M";
    }}
    function card(row) {{
      const key = chartKey(row);
      const selectedRange = selectedRangeFor(key);
      return `<article class="card">
        <div class="card-layout">
          <div>
            <div class="card-head">
              <div>
                <div class="ticker">${{row.ticker || "UNKNOWN"}}</div>
                <div class="profile">${{row.profile || "default"}} · ${{time(row.checked_at)}}</div>
                ${{row.portfolio_name ? `<div class="profile">${{row.portfolio_name}}</div>` : ""}}
              </div>
              <div>
                <div class="badge-label">Watch alert</div>
                <div class="badge ${{cls(row.action)}}">${{row.action || "-"}}</div>
              </div>
            </div>
            <div class="price">${{money(row.price)}}</div>
            <div class="reason">${{row.reason || ""}}</div>
            ${{hasDecisionConflict(row) ? `<div class="conflict">Watch alert conflicts with LLM decision (${{row.llm_decision}}). Treat this as a stale/level-trigger warning until the agent writes a fresh check.</div>` : ""}}
            <div class="fields">
              <div class="field"><span>LLM Decision</span><strong>${{row.llm_decision || "-"}}</strong></div>
              <div class="field"><span>Confidence</span><strong>${{row.llm_confidence ?? "-"}}</strong></div>
              <div class="field"><span>Market</span><strong>${{row.market_status || "-"}}</strong></div>
              <div class="field"><span>Asset Class</span><strong>${{row.asset_class || "-"}}</strong></div>
              <div class="field"><span>Price Provider</span><strong>${{row.price_provider || "-"}}</strong></div>
              <div class="field"><span>Latest Bar</span><strong>${{time(row.latest_price_time)}}</strong></div>
              <div class="field"><span>Broker</span><strong>${{row.portfolio_broker || "-"}}</strong></div>
              <div class="field"><span>ISIN</span><strong>${{row.portfolio_isin || "-"}}</strong></div>
              <div class="field"><span>Quantity</span><strong>${{money(row.portfolio_quantity)}}</strong></div>
              <div class="field"><span>Avg Cost</span><strong>${{money(row.portfolio_entry_price)}}</strong></div>
              <div class="field"><span>Position Value</span><strong>${{money(row.portfolio_position_value)}}</strong></div>
              <div class="field"><span>Unrealized P/L</span><strong>${{money(row.portfolio_unrealized_pl)}} (${{money(row.portfolio_unrealized_pl_pct)}}%)</strong></div>
              <div class="field"><span>Buy Near</span><strong>${{money(row.buy_near)}}</strong></div>
              <div class="field"><span>Buy Above</span><strong>${{money(row.buy_above)}}</strong></div>
              <div class="field"><span>Sell Near</span><strong>${{money(row.sell_near)}}</strong></div>
              <div class="field"><span>Stop Loss</span><strong>${{money(row.stop_loss)}}</strong></div>
              <div class="field"><span>Take Profit</span><strong>${{money(row.take_profit)}}</strong></div>
              <div class="field"><span>Refreshed</span><strong>${{row.forecast_refreshed_this_run ? "yes" : "no"}}</strong></div>
            </div>
          </div>
          <div class="chart-panel">
            <div class="range-controls">
              <div class="chart-title">Price / Volume</div>
              <div class="range-buttons" data-chart-buttons="${{key}}">
                ${{["1M", "3M", "6M", "1Y", "5Y"].map(range => `<button class="range-button ${{range === selectedRange ? "active" : ""}}" data-ticker="${{row.ticker || ""}}" data-key="${{key}}" data-range="${{range}}">${{range}}</button>`).join("")}}
              </div>
            </div>
            <canvas class="chart-canvas" id="chart_${{key}}" width="760" height="320" data-ticker="${{row.ticker || ""}}" data-range="${{selectedRange}}"></canvas>
          </div>
        </div>
      </article>`;
    }}
    const chartCache = new Map();
    async function loadChart(canvas, range) {{
      const ticker = canvas.dataset.ticker;
      if (!ticker) return;
      const cacheKey = `${{ticker}}_${{range}}`;
      let payload = chartCache.get(cacheKey);
      if (!payload) {{
        const response = await fetch(`/api/chart?ticker=${{encodeURIComponent(ticker)}}&range=${{encodeURIComponent(range)}}`, {{ cache: 'no-store' }});
        payload = await response.json();
        chartCache.set(cacheKey, payload);
      }}
      drawChart(canvas, payload);
    }}
    function drawChart(canvas, payload) {{
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, w, h);
      const rows = payload.rows || [];
      if (!rows.length) {{
        ctx.fillStyle = '#687386';
        ctx.font = '14px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
        ctx.fillText(payload.error || 'No chart data', 18, 36);
        return;
      }}
      const pad = {{ left: 54, right: 16, top: 24, bottom: 54 }};
      const priceH = Math.round((h - pad.top - pad.bottom) * 0.68);
      const volTop = pad.top + priceH + 20;
      const volH = h - volTop - pad.bottom + 18;
      const closes = rows.map(r => Number(r.close)).filter(Number.isFinite);
      const volumes = rows.map(r => Number(r.volume || 0)).filter(Number.isFinite);
      const minP = Math.min(...closes);
      const maxP = Math.max(...closes);
      const maxV = Math.max(...volumes, 1);
      const x = i => pad.left + (rows.length === 1 ? 0 : i * (w - pad.left - pad.right) / (rows.length - 1));
      const yPrice = v => pad.top + (maxP === minP ? priceH / 2 : (maxP - v) * priceH / (maxP - minP));
      const yVol = v => volTop + volH - (v / maxV) * volH;
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      for (let i = 0; i < 4; i++) {{
        const y = pad.top + i * priceH / 3;
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
      }}
      ctx.fillStyle = '#93c5fd';
      rows.forEach((row, i) => {{
        const barW = Math.max(1, (w - pad.left - pad.right) / Math.max(rows.length, 1) * 0.7);
        const bx = x(i) - barW / 2;
        const by = yVol(Number(row.volume || 0));
        ctx.fillRect(bx, by, barW, volTop + volH - by);
      }});
      ctx.strokeStyle = '#1f6feb';
      ctx.lineWidth = 2;
      ctx.beginPath();
      rows.forEach((row, i) => {{
        const px = x(i);
        const py = yPrice(Number(row.close));
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }});
      ctx.stroke();
      const last = rows[rows.length - 1];
      ctx.fillStyle = '#1f6feb';
      ctx.beginPath(); ctx.arc(x(rows.length - 1), yPrice(Number(last.close)), 3.5, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#18202a';
      ctx.font = '12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
      ctx.fillText(`${{payload.ticker}} ${{payload.range}}`, pad.left, 16);
      ctx.fillStyle = '#687386';
      ctx.fillText(`High ${{money(maxP)}}`, 8, pad.top + 4);
      ctx.fillText(`Low ${{money(minP)}}`, 8, pad.top + priceH);
      ctx.fillText('Volume', 8, volTop + 12);
      const firstDate = new Date(rows[0].date);
      const lastDate = new Date(last.date);
      ctx.fillText(Number.isNaN(firstDate.getTime()) ? rows[0].date : firstDate.toLocaleDateString(), pad.left, h - 14);
      const lastLabel = Number.isNaN(lastDate.getTime()) ? last.date : lastDate.toLocaleDateString();
      const labelWidth = ctx.measureText(lastLabel).width;
      ctx.fillText(lastLabel, w - pad.right - labelWidth, h - 14);
      const priceLabel = money(last.close);
      const priceWidth = ctx.measureText(priceLabel).width;
      ctx.fillStyle = '#1f6feb';
      ctx.fillText(priceLabel, w - pad.right - priceWidth, Math.max(14, yPrice(Number(last.close)) - 6));
    }}
    function activateCharts() {{
      document.querySelectorAll('canvas.chart-canvas').forEach(canvas => loadChart(canvas, canvas.dataset.range || '3M').catch(error => drawChart(canvas, {{ rows: [], error: String(error) }})));
      document.querySelectorAll('.range-button').forEach(button => {{
        button.addEventListener('click', event => {{
          const target = event.currentTarget;
          const key = target.dataset.key;
          const range = target.dataset.range || '3M';
          selectedRanges.set(key, range);
          document.querySelectorAll(`[data-chart-buttons="${{key}}"] .range-button`).forEach(item => item.classList.toggle('active', item === target));
          const canvas = document.getElementById(`chart_${{key}}`);
          if (canvas) {{
            canvas.dataset.range = range;
            loadChart(canvas, range).catch(error => drawChart(canvas, {{ rows: [], error: String(error) }}));
          }}
        }});
      }});
    }}
    function historyTable(rows) {{
      const all = Object.values(rows.histories || {{}}).flat().sort((a, b) => String(b.checked_at || "").localeCompare(String(a.checked_at || ""))).slice(0, 30);
      if (!all.length) return '<div class="empty">No watch-agent records found.</div>';
      return `<table><thead><tr><th>Time</th><th>Ticker</th><th>Name</th><th>Action</th><th>Market</th><th>Price</th><th>P/L</th><th>Reason</th><th>Printed</th></tr></thead><tbody>${{all.map(row => `
        <tr>
          <td>${{time(row.checked_at)}}</td>
          <td>${{row.ticker || "-"}}</td>
          <td>${{row.portfolio_name || "-"}}</td>
          <td>${{row.action || "-"}}</td>
          <td>${{row.market_status || "-"}}</td>
          <td>${{money(row.price)}}</td>
          <td>${{money(row.portfolio_unrealized_pl)}}</td>
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
        activateCharts();
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
        if parsed.path == "/api/chart":
            query = parse_qs(parsed.query)
            ticker = query.get("ticker", [""])[0]
            range_key = query.get("range", ["3M"])[0]
            self._send_json(read_price_chart(ticker, range_key))
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
