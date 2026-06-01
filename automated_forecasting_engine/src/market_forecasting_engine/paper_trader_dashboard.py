from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from market_forecasting_engine.data import normalize_price_frame
from market_forecasting_engine.data_providers import DataRequest, load_prices_with_provider
from market_forecasting_engine.paper_trader_agent import _cache_path, _safe


DEFAULT_STATE_DIR = Path("automated_forecasting_engine/runs/paper_trader_agent")
DEFAULT_REFRESH_SECONDS = 60


def build_dashboard_state(
    *,
    state_dir: Path,
    ticker: str,
    profile: str,
    interval: str,
    provider: str,
    chart_window: str,
    lookback_hours: float,
    max_points: int,
    runs_dir: Path | None = None,
) -> dict[str, Any]:
    state, state_path = read_agent_state(state_dir, ticker, profile)
    latest_log, log_path = read_latest_agent_log(state_dir, ticker, profile)
    actual, data_error = load_actual_series(
        state_dir=state_dir,
        ticker=ticker,
        interval=interval,
        provider=provider,
        chart_window=chart_window,
        lookback_hours=lookback_hours,
    )
    actual = actual.tail(max(10, int(max_points)))
    forecast_record = state.get("last_forecast") or (latest_log.get("forecast") if latest_log else {}) or {}
    discovered_forecasts = discover_ordered_forecasts(
        runs_dir=runs_dir or _default_runs_dir(state_dir),
        ticker=ticker,
        agent_forecast_record=forecast_record,
    )
    forecast_points = build_forecast_points({"forecasts": discovered_forecasts}, actual)
    latest_price = _latest_actual_price(actual)
    decision = latest_log.get("decision") if latest_log else None
    broker_state = latest_log.get("broker_state") if latest_log else None
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "ticker": ticker.upper(),
        "profile": profile,
        "provider": provider,
        "interval": interval,
        "state_path": str(state_path),
        "log_path": str(log_path) if log_path else None,
        "cache_path": str(_cache_path(state_dir, ticker, interval)),
        "data_error": data_error,
        "actual": _frame_to_points(actual),
        "latest_actual": latest_price,
        "forecast": summarize_forecast(forecast_record),
        "forecast_points": forecast_points,
        "forecast_sources": _forecast_sources(discovered_forecasts),
        "decision": summarize_decision(decision),
        "broker_state": broker_state,
        "latest_log": latest_log,
    }


def read_agent_state(state_dir: Path, ticker: str, profile: str) -> tuple[dict[str, Any], Path]:
    path = state_dir / "state" / f"{_safe(ticker)}_{profile}.json"
    if not path.exists():
        return {}, path
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_error": str(exc)}, path
    return parsed if isinstance(parsed, dict) else {}, path


def read_latest_agent_log(state_dir: Path, ticker: str, profile: str) -> tuple[dict[str, Any], Path | None]:
    pattern = f"{_safe(ticker)}_{profile}_*.jsonl"
    paths = sorted((state_dir / "logs").glob(pattern))
    latest: dict[str, Any] = {}
    latest_path: Path | None = None
    for path in paths:
        try:
            rows = _read_jsonl(path)
        except OSError:
            continue
        if rows:
            latest = rows[-1]
            latest_path = path
    return latest, latest_path


def load_actual_series(
    *,
    state_dir: Path,
    ticker: str,
    interval: str,
    provider: str,
    chart_window: str,
    lookback_hours: float,
) -> tuple[pd.DataFrame, str | None]:
    start_timestamp = _chart_start_timestamp(chart_window=chart_window, lookback_hours=lookback_hours)
    start = start_timestamp.isoformat().replace("+00:00", "Z")
    try:
        result = load_prices_with_provider(
            provider,
            DataRequest(ticker=ticker, start=start, interval=interval, target_column="close"),
            store=None,
            use_cache=False,
            refresh_cache=True,
        )
        return normalize_price_frame(result.frame, target_column="close"), None
    except Exception as exc:
        cached = _read_cached_prices(_cache_path(state_dir, ticker, interval))
        if cached.empty:
            return cached, str(exc)
        cutoff = pd.Timestamp(start_timestamp).tz_localize(None)
        return cached[cached.index >= cutoff], f"live_provider_failed_using_cache: {exc}"


def build_forecast_points(forecast_record: dict[str, Any], actual: pd.DataFrame) -> list[dict[str, Any]]:
    points = []
    for forecast in _forecast_rows(forecast_record):
        timestamp = forecast.get("forecast_date") or forecast.get("forecast_timestamp")
        predicted = _float_or_none(forecast.get("predicted_price"))
        if not timestamp or predicted is None:
            continue
        actual_price = actual_at_or_after(actual, timestamp)
        spot = _float_or_none(forecast.get("spot"))
        points.append(
            {
                "horizon_hours": _float_or_none(forecast.get("horizon_hours")),
                "timestamp": _iso_string(timestamp),
                "predicted_price": predicted,
                "lower_price": _float_or_none(forecast.get("lower_price")),
                "upper_price": _float_or_none(forecast.get("upper_price")),
                "expected_direction": forecast.get("expected_direction"),
                "directional_confidence": _float_or_none(forecast.get("directional_confidence")),
                "spot": spot,
                "source": forecast.get("source"),
                "source_path": forecast.get("source_path"),
                "source_as_of": forecast.get("source_as_of"),
                "matured": actual_price is not None,
                "actual_price": actual_price,
                "error": None if actual_price is None else actual_price - predicted,
                "direction_hit": _direction_hit(spot, predicted, actual_price),
            }
        )
    return points


def actual_at_or_after(actual: pd.DataFrame, timestamp: str) -> float | None:
    if actual.empty or "close" not in actual:
        return None
    target = pd.Timestamp(timestamp).tz_localize(None)
    frame = actual.sort_index()
    matches = frame[frame.index >= target]
    if matches.empty:
        return None
    return float(matches["close"].iloc[0])


def summarize_forecast(forecast_record: dict[str, Any]) -> dict[str, Any]:
    forecast = forecast_record.get("forecast") or {}
    llm = forecast_record.get("llm_trader") or {}
    llm_decision = llm.get("decision") or {}
    dip_buy = forecast_record.get("mean_reversion_dip_buy") or {}
    return {
        "created_at_utc": forecast_record.get("created_at_utc"),
        "as_of": (forecast_record.get("spot_plan") or {}).get("as_of"),
        "horizon_minutes": forecast_record.get("horizon_minutes"),
        "forecast_time": forecast.get("forecast_date") or forecast.get("forecast_timestamp"),
        "spot": forecast.get("spot"),
        "predicted_price": forecast.get("predicted_price"),
        "lower_price": forecast.get("lower_price"),
        "upper_price": forecast.get("upper_price"),
        "expected_direction": forecast.get("expected_direction"),
        "directional_confidence": forecast.get("directional_confidence"),
        "llm_status": llm.get("status"),
        "llm_decision": llm_decision.get("decision"),
        "llm_confidence": llm_decision.get("confidence"),
        "entry_plan": llm_decision.get("entry_plan"),
        "dip_buy": dip_buy.get("best_setup"),
    }


def summarize_decision(decision: dict[str, Any] | None) -> dict[str, Any] | None:
    if not decision:
        return None
    llm_decision = decision.get("llm_decision") or {}
    return {
        "action": decision.get("action"),
        "side": decision.get("side"),
        "source": decision.get("decision_source"),
        "reasons": decision.get("reasons") or [],
        "forecast_price": decision.get("forecast_price"),
        "forecast_time": decision.get("forecast_time"),
        "llm_status": decision.get("llm_status"),
        "llm_decision": llm_decision.get("decision"),
        "llm_confidence": llm_decision.get("confidence"),
        "entry_plan": llm_decision.get("entry_plan"),
        "order_plan": decision.get("order_plan"),
    }


def dashboard_html(refresh_seconds: int) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Paper Trader Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fa;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #637083;
      --line: #d9dee8;
      --accent: #2563eb;
      --actual: #243447;
      --forecast: #dc2626;
      --pending: #f59e0b;
      --matured: #16a34a;
      --hold: #7c5c00;
      --sell: #b42318;
      --buy: #067647;
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
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      padding: 18px 22px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 3;
    }}
    h1 {{ margin: 0; font-size: 20px; font-weight: 700; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 18px 22px 30px; }}
    .meta {{ color: var(--muted); font-size: 13px; line-height: 1.45; text-align: right; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }}
    .card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .label {{ color: var(--muted); font-size: 12px; margin-bottom: 5px; }}
    .value {{ font-size: 24px; font-weight: 750; overflow-wrap: anywhere; }}
    .small {{ color: var(--muted); font-size: 12px; margin-top: 4px; overflow-wrap: anywhere; }}
    .status {{
      display: inline-block;
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
      font-weight: 750;
      color: #fff;
      background: var(--hold);
      text-transform: uppercase;
    }}
    .status.buy {{ background: var(--buy); }}
    .status.sell {{ background: var(--sell); }}
    .status.submit_order {{ background: var(--buy); }}
    .status.hold {{ background: var(--hold); }}
    .chart-panel {{ padding: 0; overflow-x: auto; overflow-y: hidden; }}
    .chart-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 14px 14px 8px;
      border-bottom: 1px solid var(--line);
    }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 10px; color: var(--muted); font-size: 12px; }}
    .key {{ display: inline-flex; align-items: center; gap: 5px; }}
    .swatch {{ width: 18px; height: 3px; border-radius: 999px; background: var(--actual); display: inline-block; }}
    .swatch.asof {{ background: #2563eb; height: 8px; width: 8px; }}
    .swatch.forecast {{ background: var(--forecast); border-top: 2px dashed var(--forecast); height: 0; }}
    .swatch.pending {{ background: var(--pending); height: 8px; width: 8px; }}
    .swatch.matured {{ background: var(--matured); height: 8px; width: 8px; }}
    svg {{ display: block; width: 100%; min-width: 1040px; height: 560px; background: #fff; }}
    .grid2 {{ display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(320px, 0.7fr); gap: 14px; margin-top: 14px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 6px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 650; }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 12px;
      color: #273444;
    }}
    .error {{ border-color: #f2c94c; background: #fff8db; color: #614700; }}
    @media (max-width: 900px) {{
      header {{ flex-direction: column; }}
      .meta {{ text-align: left; }}
      .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .grid2 {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 560px) {{
      main {{ padding: 14px; }}
      .cards {{ grid-template-columns: 1fr; }}
      svg {{ height: 460px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Paper Trader Dashboard</h1>
      <div class="small" id="source">Loading agent state...</div>
    </div>
    <div class="meta">
      <div id="updated">Loading...</div>
      <div>Refresh: {refresh_seconds}s</div>
    </div>
  </header>
  <main>
    <section class="cards">
      <div class="card"><div class="label">Latest Price</div><div class="value" id="latestPrice">-</div><div class="small" id="latestTime">-</div></div>
      <div class="card"><div class="label">Forecast</div><div class="value" id="forecastPrice">-</div><div class="small" id="forecastMeta">-</div></div>
      <div class="card"><div class="label">Decision</div><div class="value"><span class="status" id="decisionBadge">-</span></div><div class="small" id="decisionMeta">-</div></div>
      <div class="card"><div class="label">LLM</div><div class="value" id="llmValue">-</div><div class="small" id="llmMeta">-</div></div>
    </section>
    <section id="errorPanel" class="card error" style="display:none"></section>
    <section class="panel chart-panel">
      <div class="chart-head">
        <div>
          <strong>Forecast vs Actual</strong>
          <div class="small">Actual data starts from the previous date window and refreshes with the agent view.</div>
        </div>
        <div class="legend">
          <span class="key"><span class="swatch"></span>Actual</span>
          <span class="key"><span class="swatch asof"></span>As-of</span>
          <span class="key"><span class="swatch forecast"></span>Forecast</span>
          <span class="key"><span class="swatch matured"></span>Matured</span>
          <span class="key"><span class="swatch pending"></span>Pending</span>
        </div>
      </div>
      <svg id="chart" role="img" aria-label="Paper trader forecast chart"></svg>
    </section>
    <section class="grid2">
      <div class="panel">
        <strong>Forecast Points</strong>
        <table>
          <thead><tr><th>Time</th><th>Horizon</th><th>Pred</th><th>Actual</th><th>Error</th><th>Hit</th><th>Source</th></tr></thead>
          <tbody id="forecastRows"></tbody>
        </table>
      </div>
      <div class="panel">
        <strong>Agent Details</strong>
        <pre id="details">Loading...</pre>
      </div>
    </section>
  </main>
  <script>
    const refreshMs = {refresh_seconds * 1000};
    const money = new Intl.NumberFormat(undefined, {{ maximumFractionDigits: 4 }});
    function fmtPrice(value) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return money.format(Number(value));
    }}
    function fmtPct(value) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return `${{(Number(value) * 100).toFixed(1)}}%`;
    }}
    function fmtTime(value) {{
      if (!value) return "-";
      const d = new Date(value);
      if (Number.isNaN(d.getTime())) return value;
      return d.toLocaleString();
    }}
    function setText(id, value) {{
      document.getElementById(id).textContent = value;
    }}
    function updateCards(data) {{
      const latest = data.latest_actual || {{}};
      const forecast = data.forecast || {{}};
      const decision = data.decision || {{}};
      setText("source", `${{data.ticker}} / ${{data.profile}} / ${{data.provider}} / ${{data.interval}}`);
      setText("updated", `Updated ${{fmtTime(data.generated_at)}}`);
      setText("latestPrice", fmtPrice(latest.close));
      setText("latestTime", fmtTime(latest.timestamp));
      setText("forecastPrice", fmtPrice(forecast.predicted_price));
      setText("forecastMeta", `${{forecast.expected_direction || "-"}} for ${{fmtTime(forecast.forecast_time)}}`);
      const badge = document.getElementById("decisionBadge");
      const action = decision.action || forecast.llm_decision || "-";
      badge.textContent = action;
      badge.className = `status ${{String(action).toLowerCase()}}`;
      setText("decisionMeta", `${{decision.side || "-"}} | ${{(decision.reasons || []).join(", ") || "no blocking reason"}}`);
      setText("llmValue", forecast.llm_decision || forecast.llm_status || "-");
      setText("llmMeta", `status=${{forecast.llm_status || "-"}} confidence=${{fmtPct(forecast.llm_confidence)}}`);
      const error = document.getElementById("errorPanel");
      if (data.data_error) {{
        error.style.display = "block";
        error.textContent = data.data_error;
      }} else {{
        error.style.display = "none";
      }}
    }}
    function drawChart(data) {{
      const svg = document.getElementById("chart");
      const width = Math.max(svg.clientWidth || 1040, 1040);
      const height = svg.clientHeight || 560;
      const pad = {{ left: 86, right: 118, top: 54, bottom: 76 }};
      const actual = (data.actual || []).map(p => [new Date(p.timestamp).getTime(), Number(p.close)]).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
      const forecasts = (data.forecast_points || []).map(p => ({{ ...p, x: new Date(p.timestamp).getTime(), y: Number(p.predicted_price) }})).filter(p => Number.isFinite(p.x) && Number.isFinite(p.y));
      const asOfTime = data.forecast && data.forecast.as_of ? new Date(data.forecast.as_of).getTime() : null;
      const asOfPrice = data.forecast && data.forecast.spot ? Number(data.forecast.spot) : null;
      const latest = data.latest_actual && data.latest_actual.timestamp ? [new Date(data.latest_actual.timestamp).getTime(), Number(data.latest_actual.close)] : null;
      const xs = actual.map(p => p[0]).concat(forecasts.map(p => p.x)).concat(Number.isFinite(asOfTime) ? [asOfTime] : []);
      const ys = actual.map(p => p[1]).concat(forecasts.flatMap(p => [p.y, p.lower_price, p.upper_price]).map(Number).filter(Number.isFinite));
      if (Number.isFinite(asOfPrice)) ys.push(asOfPrice);
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      svg.innerHTML = "";
      if (!xs.length || !ys.length) {{
        svg.innerHTML = `<text x="${{width / 2}}" y="${{height / 2}}" text-anchor="middle" fill="#637083">No price data available yet</text>`;
        return;
      }}
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minYRaw = Math.min(...ys), maxYRaw = Math.max(...ys);
      const yPad = Math.max((maxYRaw - minYRaw) * 0.08, maxYRaw * 0.001, 1);
      const minY = minYRaw - yPad, maxY = maxYRaw + yPad;
      const plotWidth = width - pad.left - pad.right;
      const plotHeight = height - pad.top - pad.bottom;
      const x = v => pad.left + ((v - minX) / Math.max(maxX - minX, 1)) * plotWidth;
      const y = v => pad.top + (1 - ((v - minY) / Math.max(maxY - minY, 1))) * plotHeight;
      const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
      const add = node => svg.appendChild(node);
      const make = (name, attrs) => {{
        const node = document.createElementNS("http://www.w3.org/2000/svg", name);
        for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
        return node;
      }};
      const title = make("text", {{ x: width / 2, y: 26, "text-anchor": "middle", fill: "#111827", "font-size": "18", "font-weight": "650" }});
      const latestTitle = latest && Number.isFinite(latest[0]) ? ` to ${{new Date(latest[0]).toISOString().slice(0, 16).replace("T", " ")}} UTC` : "";
      title.textContent = `${{data.ticker.replace("-", "/")}} Forecast vs Actual${{latestTitle}}`;
      add(title);
      for (let i = 0; i <= 5; i++) {{
        const gridY = pad.top + i * plotHeight / 5;
        add(make("line", {{ x1: pad.left, x2: width - pad.right, y1: gridY, y2: gridY, stroke: "#e5e7eb" }}));
        const value = maxY - i * (maxY - minY) / 5;
        const text = make("text", {{ x: pad.left - 12, y: gridY + 4, "text-anchor": "end", fill: "#374151", "font-size": "12" }});
        text.textContent = Number(value).toFixed(2);
        add(text);
      }}
      const tickCount = width < 1180 ? 6 : 8;
      for (let i = 0; i <= tickCount; i++) {{
        const xx = pad.left + i * plotWidth / tickCount;
        add(make("line", {{ x1: xx, x2: xx, y1: pad.top, y2: height - pad.bottom, stroke: "#e5e7eb" }}));
        const value = minX + i * (maxX - minX) / tickCount;
        const d = new Date(value);
        const label = `${{String(d.getUTCMonth() + 1).padStart(2, "0")}}-${{String(d.getUTCDate()).padStart(2, "0")}} ${{String(d.getUTCHours()).padStart(2, "0")}}`;
        const text = make("text", {{ x: xx, y: height - 34, "text-anchor": "middle", fill: "#374151", "font-size": "12", transform: `rotate(-32 ${{xx}} ${{height - 34}})` }});
        text.textContent = label;
        add(text);
      }}
      add(make("line", {{ x1: pad.left, x2: width - pad.right, y1: height - pad.bottom, y2: height - pad.bottom, stroke: "#111827" }}));
      add(make("line", {{ x1: pad.left, x2: pad.left, y1: pad.top, y2: height - pad.bottom, stroke: "#111827" }}));
      const actualPath = actual.map((p, i) => `${{i ? "L" : "M"}}${{x(p[0]).toFixed(2)}},${{y(p[1]).toFixed(2)}}`).join(" ");
      add(make("path", {{ d: actualPath, fill: "none", stroke: "#1f2937", "stroke-width": "2", "stroke-linejoin": "round", "stroke-linecap": "round" }}));
      if (Number.isFinite(asOfTime)) {{
        add(make("line", {{ x1: x(asOfTime), x2: x(asOfTime), y1: pad.top, y2: height - pad.bottom, stroke: "#2563eb", "stroke-width": "1.8", "stroke-dasharray": "2 3" }}));
      }}
      const forecastLine = [];
      if (Number.isFinite(asOfTime) && Number.isFinite(asOfPrice)) forecastLine.push([asOfTime, asOfPrice]);
      forecasts.forEach(p => forecastLine.push([p.x, p.y]));
      if (forecastLine.length > 1) {{
        const d = forecastLine.map((p, i) => `${{i ? "L" : "M"}}${{x(p[0]).toFixed(2)}},${{y(p[1]).toFixed(2)}}`).join(" ");
        add(make("path", {{ d, fill: "none", stroke: "#ef2323", "stroke-width": "2.2", "stroke-dasharray": "7 5", "stroke-linecap": "round", "stroke-linejoin": "round" }}));
      }}
      if (Number.isFinite(asOfTime) && Number.isFinite(asOfPrice)) {{
        add(make("circle", {{ cx: x(asOfTime), cy: y(asOfPrice), r: 7, fill: "#2563eb", stroke: "#ffffff", "stroke-width": "2" }}));
      }}
      forecasts.forEach(p => {{
        if (p.matured) {{
          add(make("circle", {{ cx: x(p.x), cy: y(p.y), r: 6.5, fill: "#e5252a", stroke: "#ffffff", "stroke-width": "2" }}));
        }} else {{
          const size = 8;
          const cx = x(p.x), cy = y(p.y);
          add(make("polygon", {{ points: `${{cx}},${{cy - size}} ${{cx + size}},${{cy}} ${{cx}},${{cy + size}} ${{cx - size}},${{cy}}`, fill: "#f59e0b", stroke: "#ffffff", "stroke-width": "2" }}));
        }}
        const horizon = Number.isFinite(Number(p.horizon_hours)) ? `${{Number(p.horizon_hours)}}h` : "";
        const px = x(p.x);
        const py = y(p.y);
        const anchor = px > width - pad.right - 70 ? "end" : "start";
        const labelX = anchor === "end" ? px - 10 : px + 10;
        const labelY = clamp(py - 12, pad.top + 14, height - pad.bottom - 10);
        const label = make("text", {{ x: labelX, y: labelY, "text-anchor": anchor, fill: "#8a2d2d", "font-size": "12", "font-weight": "650" }});
        label.textContent = `${{horizon}} ${{Math.round(Number(p.predicted_price))}}`;
        add(label);
      }});
      if (latest && Number.isFinite(latest[0]) && Number.isFinite(latest[1])) {{
        const lx = x(latest[0]);
        const ly = y(latest[1]);
        add(make("circle", {{ cx: lx, cy: ly, r: 5.5, fill: "#111827", stroke: "#ffffff", "stroke-width": "1.5" }}));
        const latestAnchor = lx > width - pad.right - 80 ? "end" : "start";
        const latestX = latestAnchor === "end" ? lx - 10 : lx + 10;
        const latestY = clamp(ly + 18, pad.top + 16, height - pad.bottom - 8);
        const label = make("text", {{ x: latestX, y: latestY, "text-anchor": latestAnchor, fill: "#1f2937", "font-size": "12" }});
        label.textContent = `latest ${{fmtPrice(latest[1])}}`;
        add(label);
      }}
      const yAxis = make("text", {{ x: pad.left, y: pad.top - 10, fill: "#111827", "font-size": "13", "text-anchor": "start", "font-weight": "600" }});
      yAxis.textContent = `${{data.ticker.replace("-", "/")}} price`;
      add(yAxis);
      const xAxis = make("text", {{ x: width / 2, y: height - 6, fill: "#111827", "font-size": "13", "text-anchor": "middle" }});
      xAxis.textContent = "Time UTC";
      add(xAxis);
    }}
    function updateTables(data) {{
      const rows = document.getElementById("forecastRows");
      rows.innerHTML = "";
      (data.forecast_points || []).forEach(p => {{
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${{fmtTime(p.timestamp)}}</td><td>${{p.horizon_hours || "-"}}h</td><td>${{fmtPrice(p.predicted_price)}}</td><td>${{fmtPrice(p.actual_price)}}</td><td>${{p.error === null || p.error === undefined ? "-" : Number(p.error).toFixed(4)}}</td><td>${{p.direction_hit === null || p.direction_hit === undefined ? "-" : p.direction_hit}}</td><td>${{p.source || "-"}}</td>`;
        rows.appendChild(tr);
      }});
      const details = {{
        forecast: data.forecast,
        decision: data.decision,
        broker_state: data.broker_state,
        forecast_sources: data.forecast_sources,
        state_path: data.state_path,
        log_path: data.log_path,
        cache_path: data.cache_path
      }};
      setText("details", JSON.stringify(details, null, 2));
    }}
    async function refresh() {{
      try {{
        const response = await fetch(`/api/state${{window.location.search}}`, {{ cache: "no-store" }});
        const data = await response.json();
        updateCards(data);
        drawChart(data);
        updateTables(data);
      }} catch (error) {{
        const panel = document.getElementById("errorPanel");
        panel.style.display = "block";
        panel.textContent = `dashboard_refresh_failed: ${{error}}`;
      }}
    }}
    window.addEventListener("resize", () => refresh());
    refresh();
    setInterval(refresh, refreshMs);
  </script>
</body>
</html>"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a live Alpaca paper-trader dashboard.")
    parser.add_argument("--ticker", default="ETH-USD")
    parser.add_argument("--profile", default="aggressive", choices=("conservative", "medium", "aggressive"))
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--provider", default="alpaca")
    parser.add_argument("--runs-dir", default="automated_forecasting_engine/runs", help="Root to scan for saved multi-horizon forecast reports.")
    parser.add_argument(
        "--chart-window",
        choices=("previous-utc-date", "lookback-hours"),
        default="previous-utc-date",
        help="Use previous UTC midnight to now by default so the chart matches validation plots.",
    )
    parser.add_argument("--lookback-hours", type=float, default=36.0, help="Price chart window. 36h covers the previous date through now for 24/7 crypto.")
    parser.add_argument("--max-points", type=int, default=2500)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--refresh-seconds", type=int, default=DEFAULT_REFRESH_SECONDS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = build_server(args)
    print(f"Serving paper trader dashboard at http://{args.host}:{args.port}", flush=True)
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
                profile = _query_value(params, "profile", args.profile)
                interval = _query_value(params, "interval", args.interval)
                state = build_dashboard_state(
                    state_dir=Path(args.state_dir),
                    ticker=ticker,
                    profile=profile,
                    interval=interval,
                    provider=_query_value(params, "provider", args.provider),
                    chart_window=_query_value(params, "chart_window", args.chart_window),
                    lookback_hours=float(_query_value(params, "lookback_hours", str(args.lookback_hours))),
                    max_points=int(_query_value(params, "max_points", str(args.max_points))),
                    runs_dir=Path(_query_value(params, "runs_dir", args.runs_dir)),
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _read_cached_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    frame = frame.set_index("timestamp")
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    return normalize_price_frame(frame, target_column="close")


def discover_ordered_forecasts(
    *,
    runs_dir: Path,
    ticker: str,
    agent_forecast_record: dict[str, Any],
) -> list[dict[str, Any]]:
    by_horizon: dict[float, dict[str, Any]] = {}
    for row in _forecast_rows(agent_forecast_record):
        normalized = _normalize_forecast_row(
            row,
            source="live_agent",
            source_path=None,
            source_as_of=(agent_forecast_record.get("spot_plan") or {}).get("as_of") or agent_forecast_record.get("created_at_utc"),
            fallback_spot=(agent_forecast_record.get("spot_plan") or {}).get("latest_price"),
        )
        if normalized:
            by_horizon[float(normalized["horizon_hours"])] = normalized
    for path in sorted(runs_dir.rglob("daily_trade_report.json")):
        report = _read_json_file(path)
        if not report or not _same_ticker(report.get("ticker"), ticker):
            continue
        as_of = report.get("as_of_timestamp") or (report.get("daily_trade_view") or {}).get("as_of")
        fallback_spot = report.get("current_price") or (report.get("daily_trade_view") or {}).get("latest_price")
        for row in _report_forecast_rows(report):
            normalized = _normalize_forecast_row(
                row,
                source="saved_forecast",
                source_path=str(path),
                source_as_of=as_of,
                fallback_spot=fallback_spot,
            )
            if not normalized:
                continue
            horizon = float(normalized["horizon_hours"])
            existing = by_horizon.get(horizon)
            if existing is None or _source_sort_key(normalized) >= _source_sort_key(existing):
                by_horizon[horizon] = normalized
    return [by_horizon[key] for key in sorted(by_horizon)]


def _normalize_forecast_row(
    row: dict[str, Any],
    *,
    source: str,
    source_path: str | None,
    source_as_of: Any,
    fallback_spot: Any,
) -> dict[str, Any] | None:
    horizon = _float_or_none(row.get("horizon_hours"))
    if horizon is None and row.get("horizon_minutes") is not None:
        horizon = _float_or_none(row.get("horizon_minutes"))
        horizon = None if horizon is None else horizon / 60.0
    predicted = _float_or_none(row.get("predicted_price"))
    if horizon is None or predicted is None:
        return None
    forecast_time = row.get("forecast_date") or row.get("forecast_timestamp")
    if not forecast_time and source_as_of:
        forecast_time = (pd.Timestamp(source_as_of) + pd.Timedelta(hours=horizon)).isoformat()
    if not forecast_time:
        return None
    output = dict(row)
    output["horizon_hours"] = horizon
    output["forecast_date"] = _iso_string(forecast_time)
    output["predicted_price"] = predicted
    output["spot"] = _float_or_none(row.get("spot")) or _float_or_none(fallback_spot)
    output["source"] = source
    output["source_path"] = source_path
    output["source_as_of"] = _iso_string(source_as_of) if source_as_of else None
    return output


def _report_forecast_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    forecasts = report.get("forecasts") or []
    if not forecasts:
        forecasts = (report.get("daily_trade_view") or {}).get("forecasts") or []
    return [row for row in forecasts if isinstance(row, dict)]


def _forecast_sources(forecasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {}
    for row in forecasts:
        key = (row.get("source"), row.get("source_path"), row.get("source_as_of"))
        seen[key] = {
            "source": row.get("source"),
            "source_path": row.get("source_path"),
            "source_as_of": row.get("source_as_of"),
        }
    return list(seen.values())


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _same_ticker(left: Any, right: str) -> bool:
    if left is None:
        return False
    return _ticker_key(str(left)) == _ticker_key(right)


def _ticker_key(value: str) -> str:
    return value.upper().replace("/", "").replace("-", "").replace("_", "")


def _source_sort_key(row: dict[str, Any]) -> tuple[pd.Timestamp, int]:
    as_of = row.get("source_as_of")
    timestamp = pd.Timestamp.min if not as_of else pd.Timestamp(as_of).tz_localize(None)
    priority = 1 if row.get("source") == "live_agent" else 0
    return timestamp, priority


def _default_runs_dir(state_dir: Path) -> Path:
    marker = "runs"
    for parent in [state_dir, *state_dir.parents]:
        if parent.name == marker:
            return parent
    return Path("automated_forecasting_engine/runs")


def _chart_start_timestamp(*, chart_window: str, lookback_hours: float) -> datetime:
    now = datetime.now(UTC)
    if chart_window == "previous-utc-date":
        today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return today_midnight - timedelta(days=1)
    return now - timedelta(hours=float(lookback_hours))


def _forecast_rows(forecast_record: dict[str, Any]) -> list[dict[str, Any]]:
    if forecast_record.get("forecasts"):
        return [row for row in forecast_record["forecasts"] if isinstance(row, dict)]
    if forecast_record.get("forecast"):
        return [forecast_record["forecast"]]
    spot_plan = forecast_record.get("spot_plan") or {}
    forecasts = spot_plan.get("forecasts") or []
    return [row for row in forecasts if isinstance(row, dict)]


def _frame_to_points(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    output = []
    for timestamp, row in frame.iterrows():
        output.append({"timestamp": pd.Timestamp(timestamp).isoformat(), "close": _float_or_none(row.get("close"))})
    return output


def _latest_actual_price(frame: pd.DataFrame) -> dict[str, Any] | None:
    if frame.empty:
        return None
    row = frame.iloc[-1]
    return {"timestamp": pd.Timestamp(frame.index[-1]).isoformat(), "close": _float_or_none(row.get("close"))}


def _direction_hit(spot: float | None, predicted: float, actual: float | None) -> bool | None:
    if spot is None or actual is None:
        return None
    predicted_direction = 1 if predicted > spot else -1 if predicted < spot else 0
    actual_direction = 1 if actual > spot else -1 if actual < spot else 0
    return predicted_direction == actual_direction


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


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


def _iso_string(value: Any) -> str:
    return pd.Timestamp(value).isoformat()


def _query_value(params: dict[str, list[str]], key: str, default: str) -> str:
    values = params.get(key)
    if not values:
        return default
    return values[0] or default


if __name__ == "__main__":
    main()
