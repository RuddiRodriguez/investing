from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_forecasting_engine.live_trading.deribit_dashboard import (
    DEFAULT_REPORT,
    deribit_currency_to_yahoo_symbol,
    fetch_yahoo_market_history,
)


DEFAULT_AGENT_REPORT = Path("automated_forecasting_engine/runs/live_deribit_eth_usdc_daily_agent/ETH_USDC_daily_agent_report.json")


def main() -> None:
    args = build_parser().parse_args()
    st.set_page_config(page_title="Deribit ETH/USDC Dashboard", layout="wide")
    st.markdown(f"<script>setTimeout(() => window.location.reload(), {max(5, args.refresh_seconds) * 1000});</script>", unsafe_allow_html=True)
    st.title("Deribit ETH/USDC Live Dashboard")
    st.caption(f"Auto-refresh {args.refresh_seconds}s | account report: `{args.report_path}` | agent report: `{args.agent_report_path}`")

    account_report = read_json(Path(args.report_path))
    agent_report = read_json(Path(args.agent_report_path))
    render_header(account_report, agent_report)
    render_market_chart(args, agent_report)
    render_forecast_chart(agent_report, agent_report_path=Path(args.agent_report_path))
    render_orders(account_report, agent_report)
    render_decision(agent_report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Streamlit/Plotly dashboard for the Deribit ETH/USDC daily agent.")
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT))
    parser.add_argument("--agent-report-path", default=str(DEFAULT_AGENT_REPORT))
    parser.add_argument("--refresh-seconds", type=int, default=20)
    return parser


def read_json(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"missing_report": str(path), "checked_at": datetime.now().isoformat()}
    return parsed if isinstance(parsed, dict) else {}


def render_header(account: dict[str, Any], agent: dict[str, Any]) -> None:
    decision = agent.get("decision") or {}
    market = agent.get("market") or {}
    forecast = agent.get("forecast") or {}
    safety = agent.get("safety") or {}
    overview = account.get("overview") or {}

    cols = st.columns(5)
    cols[0].metric("Live price", fmt_money(market.get("latest_price")))
    cols[1].metric("Agent action", str(decision.get("action") or "-"), str(decision.get("reason") or ""))
    cols[2].metric("CEO decision", str((forecast.get("ceo_decision") or {}).get("decision") or "-"), str((forecast.get("ceo_decision") or {}).get("confidence") or ""))
    cols[3].metric("Open orders", overview.get("open_order_count", 0))
    cols[4].metric("Execution", "LIVE" if safety.get("execute_live_orders") else "DRY/R/O")

    st.caption(f"Agent checked: `{agent.get('checked_at_utc') or agent.get('checked_at') or '-'}`")


def render_market_chart(args: argparse.Namespace, agent: dict[str, Any]) -> None:
    st.subheader("Real Price")
    range_key = st.segmented_control("Range", ["1d", "2d", "1m", "2m", "3m", "6m"], default="6m", key="market_range")
    currency = ((agent.get("market") or {}).get("account") or {}).keys()
    symbol = deribit_currency_to_yahoo_symbol("ETH" if "ETH" in currency or not currency else list(currency)[0])
    payload = fetch_yahoo_market_history(symbol=symbol, range_key=str(range_key))
    points = pd.DataFrame(payload.get("points") or [])
    if points.empty:
        st.info("No market history available.")
        return
    points["time"] = pd.to_datetime(points["time"])
    points["close"] = pd.to_numeric(points["close"], errors="coerce")
    points = points.dropna(subset=["time", "close"])
    if points.empty:
        st.info("No valid market history available.")
        return

    first = points.iloc[0]
    latest = points.iloc[-1]
    high = points.loc[points["close"].idxmax()]
    low = points.loc[points["close"].idxmin()]
    change_pct = ((latest["close"] - first["close"]) / first["close"] * 100.0) if first["close"] else None
    summary = st.columns(4)
    summary[0].metric("Start", fmt_money(first["close"]), first["time"].date().isoformat())
    summary[1].metric("Low", fmt_money(low["close"]), low["time"].date().isoformat())
    summary[2].metric("High", fmt_money(high["close"]), high["time"].date().isoformat())
    summary[3].metric("Latest", fmt_money(latest["close"]), "-" if change_pct is None else f"{change_pct:.2f}%")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=points["time"],
            y=points["close"],
            mode="lines",
            name="Actual ETH/USD",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Actual: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[latest["time"]],
            y=[latest["close"]],
            mode="markers+text",
            name="Latest",
            text=[fmt_money(latest["close"])],
            textposition="top right",
            marker={"size": 10},
            hovertemplate="Latest<br>%{x|%Y-%m-%d %H:%M}<br>$%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(height=430, margin={"l": 20, "r": 20, "t": 30, "b": 20}, hovermode="x unified", yaxis_tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)


def render_forecast_chart(agent: dict[str, Any], *, agent_report_path: Path) -> None:
    st.subheader("Forecast Path")
    forecast = agent.get("forecast") or {}
    rows = forecast.get("forecasts") or []
    if not rows:
        st.info("No forecast rows in agent report.")
        return
    as_of = parse_dt(forecast.get("created_at_utc") or agent.get("checked_at_utc") or agent.get("checked_at"))
    frame = pd.DataFrame(normalize_forecast_rows(rows, as_of=as_of))
    matured = pd.DataFrame(load_matured_forecast_points(agent_report_path))
    if as_of is not None and not matured.empty:
        matured["_forecast_checked_at_dt"] = pd.to_datetime(matured["forecast_checked_at"], errors="coerce", utc=True)
        as_of_utc = pd.Timestamp(as_of).tz_convert("UTC") if pd.Timestamp(as_of).tzinfo else pd.Timestamp(as_of).tz_localize("UTC")
        matured = matured[(matured["_forecast_checked_at_dt"] - as_of_utc).abs() <= pd.Timedelta(seconds=1)].drop(columns=["_forecast_checked_at_dt"])
    if not matured.empty:
        current_horizons = {float(value) for value in frame["horizon_hours"].dropna().tolist()}
        matured = matured[matured["horizon_hours"].astype(float).isin(current_horizons)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["target_time"], y=frame["lower_price"], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=frame["target_time"],
            y=frame["upper_price"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(220,38,38,0.16)",
            line={"width": 0},
            name="Forecast band",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["target_time"],
            y=frame["predicted_price"],
            mode="lines+markers+text",
            text=[f"{row.horizon_hours:g}h<br>{fmt_money(row.predicted_price)}" for row in frame.itertuples()],
            textposition=["top left", "bottom center", "top right"][: len(frame)],
            name="Forecast",
            line={"dash": "dash", "color": "#dc2626"},
            marker={"size": 10, "color": "#dc2626"},
            customdata=frame[["horizon_hours", "expected_direction", "selected_model", "lower_price", "upper_price"]],
            hovertemplate="%{customdata[0]}h forecast<br>Target: %{x|%Y-%m-%d %H:%M}<br>Prediction: $%{y:,.2f}<br>Band: $%{customdata[3]:,.2f} - $%{customdata[4]:,.2f}<br>Direction: %{customdata[1]}<br>Model: %{customdata[2]}<extra></extra>",
        )
    )
    if not matured.empty:
        fig.add_trace(
            go.Scatter(
                x=matured["target_time"],
                y=matured["predicted_price"],
                mode="markers",
                name="Matured forecast",
                marker={"symbol": "diamond", "size": 10, "color": "#7c3aed"},
                customdata=matured[["horizon_hours", "actual_price", "error", "direction_hit", "forecast_checked_at"]],
                hovertemplate="Matured %{customdata[0]}h forecast<br>Target: %{x|%Y-%m-%d %H:%M}<br>Predicted: $%{y:,.2f}<br>Actual: $%{customdata[1]:,.2f}<br>Error: $%{customdata[2]:,.2f}<br>Direction hit: %{customdata[3]}<br>Forecast run: %{customdata[4]}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=matured["target_time"],
                y=matured["actual_price"],
                mode="markers+text",
                name="Actual at matured horizon",
                marker={"symbol": "x", "size": 11, "color": "#059669"},
                text=[fmt_money(value) for value in matured["actual_price"]],
                textposition="top center",
                customdata=matured[["horizon_hours", "predicted_price", "error", "direction_hit", "observed_at"]],
                hovertemplate="Actual after %{customdata[0]}h target<br>Observed: %{customdata[4]}<br>Actual: $%{y:,.2f}<br>Predicted: $%{customdata[1]:,.2f}<br>Error: $%{customdata[2]:,.2f}<br>Direction hit: %{customdata[3]}<extra></extra>",
            )
        )
    fig.update_layout(height=420, margin={"l": 20, "r": 20, "t": 30, "b": 20}, hovermode="x unified", yaxis_tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(frame[["horizon_hours", "predicted_price", "lower_price", "upper_price", "expected_direction", "selected_model"]], use_container_width=True)
    if matured.empty:
        st.caption("No matured points for the current forecast yet. A point appears after a target time passes and a later hourly agent check records the actual live price.")
    else:
        st.caption("Matured points compare prior forecast targets against the first later hourly live check.")
        st.dataframe(
            matured[
                [
                    "forecast_checked_at",
                    "horizon_hours",
                    "target_time",
                    "predicted_price",
                    "actual_price",
                    "error",
                    "direction_hit",
                    "observed_at",
                ]
            ],
            use_container_width=True,
        )


def render_orders(account: dict[str, Any], agent: dict[str, Any]) -> None:
    st.subheader("Orders and Protection")
    non_options = account.get("non_options") or {}
    open_orders = pd.DataFrame(non_options.get("open_orders") or [])
    order_results = pd.DataFrame(agent.get("order_results") or [])
    cols = st.columns(2)
    with cols[0]:
        st.caption("Live open non-option orders")
        st.dataframe(open_orders, use_container_width=True)
    with cols[1]:
        st.caption("Latest agent order results")
        st.dataframe(order_results, use_container_width=True)


def render_decision(agent: dict[str, Any]) -> None:
    st.subheader("CEO and Execution Decision")
    cols = st.columns(2)
    with cols[0]:
        st.caption("Final advice")
        st.json((agent.get("forecast") or {}).get("final_advice") or {})
    with cols[1]:
        st.caption("Execution plan")
        st.json(agent.get("decision") or {})


def normalize_forecast_rows(rows: list[dict[str, Any]], *, as_of: datetime | None) -> list[dict[str, Any]]:
    normalized = []
    base = as_of or datetime.now()
    for row in rows:
        bars = float(row.get("horizon_hours") or row.get("horizon_bars") or row.get("horizon_days") or 0.0)
        target = parse_dt(row.get("forecast_timestamp")) or (base + timedelta(hours=bars))
        normalized.append(
            {
                "horizon_hours": bars,
                "target_time": target,
                "predicted_price": to_float(row.get("predicted_price")),
                "lower_price": to_float(row.get("lower_price")),
                "upper_price": to_float(row.get("upper_price")),
                "expected_direction": row.get("expected_direction") or "-",
                "selected_model": row.get("selected_model") or row.get("model") or row.get("method") or "-",
            }
        )
    return normalized


def load_matured_forecast_points(agent_report_path: Path) -> list[dict[str, Any]]:
    reports = load_agent_history(agent_report_path)
    observations = []
    for report in reports:
        checked_at = parse_dt(report.get("checked_at_utc") or report.get("checked_at"))
        price = to_float((report.get("market") or {}).get("latest_price"))
        if checked_at is not None and price is not None:
            observations.append({"checked_at": checked_at, "price": price})
    observations.sort(key=lambda row: row["checked_at"])
    matured: dict[tuple[str, float], dict[str, Any]] = {}
    for report in reports:
        forecast = report.get("forecast") or {}
        rows = forecast.get("forecasts") or []
        forecast_at = parse_dt(forecast.get("created_at_utc") or report.get("checked_at_utc") or report.get("checked_at"))
        if not rows or forecast_at is None:
            continue
        current_price = to_float((report.get("market") or {}).get("latest_price"))
        for row in normalize_forecast_rows(rows, as_of=forecast_at):
            target_time = row.get("target_time")
            predicted = to_float(row.get("predicted_price"))
            horizon = to_float(row.get("horizon_hours"))
            if target_time is None or predicted is None or horizon is None:
                continue
            observation = first_observation_at_or_after(observations, target_time)
            if observation is None:
                continue
            actual = observation["price"]
            direction_hit = direction_match(start_price=current_price, predicted_price=predicted, actual_price=actual)
            key = (forecast_at.isoformat(), float(horizon))
            matured[key] = {
                "forecast_checked_at": forecast_at.isoformat(),
                "horizon_hours": float(horizon),
                "target_time": target_time,
                "predicted_price": predicted,
                "actual_price": actual,
                "error": actual - predicted,
                "absolute_error": abs(actual - predicted),
                "direction_hit": direction_hit,
                "observed_at": observation["checked_at"].isoformat(),
                "selected_model": row.get("selected_model"),
            }
    return sorted(matured.values(), key=lambda row: (row["target_time"], row["horizon_hours"]))


def load_agent_history(agent_report_path: Path) -> list[dict[str, Any]]:
    root = agent_report_path.parent
    candidates = [agent_report_path]
    candidates.extend(sorted((root / "snapshots").glob("*.json")))
    candidates.extend(sorted((root / "logs").glob("*.jsonl")))
    reports: dict[str, dict[str, Any]] = {}
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    report = json.loads(line)
                except json.JSONDecodeError:
                    continue
                add_report(reports, report)
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        add_report(reports, report)
    return sorted(reports.values(), key=lambda row: parse_dt(row.get("checked_at_utc") or row.get("checked_at")) or datetime.min)


def add_report(reports: dict[str, dict[str, Any]], report: Any) -> None:
    if not isinstance(report, dict):
        return
    checked_at = str(report.get("checked_at_utc") or report.get("checked_at") or "")
    if not checked_at:
        return
    reports[checked_at] = report


def first_observation_at_or_after(observations: list[dict[str, Any]], target_time: datetime) -> dict[str, Any] | None:
    for observation in observations:
        if observation["checked_at"] >= target_time:
            return observation
    return None


def direction_match(*, start_price: float | None, predicted_price: float, actual_price: float) -> str:
    if start_price is None:
        return "unknown"
    predicted_direction = 1 if predicted_price > start_price else -1 if predicted_price < start_price else 0
    actual_direction = 1 if actual_price > start_price else -1 if actual_price < start_price else 0
    if predicted_direction == 0 or actual_direction == 0:
        return "flat"
    return "yes" if predicted_direction == actual_direction else "no"


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return pd.Timestamp(value).to_pydatetime()
    except Exception:
        return None


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_money(value: Any) -> str:
    number = to_float(value)
    return "-" if number is None else f"${number:,.2f}"


if __name__ == "__main__":
    main()
