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
MAX_MATURED_OBSERVATION_LAG_HOURS = 2.0


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
    st.subheader("Forecast Paths")
    history_rows = pd.DataFrame(load_forecast_history_points(agent_report_path))
    if history_rows.empty:
        st.info("No forecast rows in agent history.")
        return
    matured = pd.DataFrame(load_matured_forecast_points(agent_report_path))

    fig = go.Figure()
    run_labels = list(history_rows["run_label"].drop_duplicates())
    latest_label = run_labels[-1]
    historical_rows = history_rows[history_rows["run_label"] != latest_label].copy()
    if not historical_rows.empty:
        historical_rows["target_bucket"] = pd.to_datetime(historical_rows["target_time"], errors="coerce").dt.round("h")
        historical_rows["horizon_key"] = pd.to_numeric(historical_rows["horizon_hours"], errors="coerce").round(3)
        historical_summary = (
            historical_rows.dropna(subset=["target_bucket", "horizon_key"])
            .groupby(["horizon_key", "target_bucket"], as_index=False)
            .agg(
                predicted_mean=("predicted_price", "mean"),
                predicted_std=("predicted_price", "std"),
                lower_mean=("lower_price", "mean"),
                upper_mean=("upper_price", "mean"),
                sample_count=("predicted_price", "count"),
            )
            .sort_values(["horizon_key", "target_bucket"])
        )
        historical_summary["predicted_std"] = historical_summary["predicted_std"].fillna(0.0)
        historical_summary["mean_minus_std"] = historical_summary["predicted_mean"] - historical_summary["predicted_std"]
        historical_summary["mean_plus_std"] = historical_summary["predicted_mean"] + historical_summary["predicted_std"]
        for horizon, horizon_frame in historical_summary.groupby("horizon_key", sort=True):
            color = horizon_color(horizon)
            label = horizon_label(horizon)
            fig.add_trace(
                go.Scatter(
                    x=horizon_frame["target_bucket"],
                    y=horizon_frame["mean_minus_std"],
                    mode="lines",
                    name=f"{label} historical -1 std",
                    line={"color": rgba(color, 0.35), "width": 1.2, "dash": "dash"},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=horizon_frame["target_bucket"],
                    y=horizon_frame["mean_plus_std"],
                    mode="lines",
                    name=f"{label} historical +1 std",
                    line={"color": rgba(color, 0.35), "width": 1.2, "dash": "dash"},
                    fill="tonexty",
                    fillcolor=rgba(color, 0.08),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=horizon_frame["target_bucket"],
                    y=horizon_frame["predicted_mean"],
                    mode="lines",
                    name=f"{label} historical average",
                    line={"dash": "dot", "color": color, "width": 2},
                    customdata=horizon_frame[["horizon_key", "sample_count", "predicted_std", "mean_minus_std", "mean_plus_std", "lower_mean", "upper_mean"]],
                    hovertemplate="Historical average by horizon<br>Horizon: %{customdata[0]:g}h<br>Target: %{x|%Y-%m-%d %H:%M}<br>Mean prediction: $%{y:,.2f}<br>-1 std: $%{customdata[3]:,.2f}<br>+1 std: $%{customdata[4]:,.2f}<br>Std dev: $%{customdata[2]:,.2f}<br>Mean forecast band: $%{customdata[5]:,.2f} - $%{customdata[6]:,.2f}<br>Samples: %{customdata[1]}<extra></extra>",
                    showlegend=False,
                )
            )
    for run_label, frame in history_rows.groupby("run_label", sort=False):
        frame = frame.sort_values("target_time")
        is_latest = run_label == latest_label
        if not is_latest:
            continue
        color = "#dc2626" if is_latest else "#94a3b8"
        if is_latest:
            fig.add_trace(go.Scatter(x=frame["target_time"], y=frame["lower_price"], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
            fig.add_trace(
                go.Scatter(
                    x=frame["target_time"],
                    y=frame["upper_price"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(220,38,38,0.12)",
                    line={"width": 0},
                    name=f"{run_label} band",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=frame["target_time"],
                y=frame["predicted_price"],
                mode="lines+markers",
                name=run_label,
                line={"dash": "dash", "color": color, "width": 3 if is_latest else 1.5},
                marker={"size": 10 if is_latest else 7, "color": color, "symbol": "circle"},
                opacity=1.0 if is_latest else 0.42,
                showlegend=False,
                customdata=frame[["horizon_hours", "expected_direction", "selected_model", "lower_price", "upper_price", "forecast_checked_at"]],
                hovertemplate="Run: %{customdata[5]}<br>%{customdata[0]}h forecast<br>Target: %{x|%Y-%m-%d %H:%M}<br>Prediction: $%{y:,.2f}<br>Band: $%{customdata[3]:,.2f} - $%{customdata[4]:,.2f}<br>Direction: %{customdata[1]}<br>Model: %{customdata[2]}<extra></extra>",
            )
        )
    if not matured.empty:
        matured["maturity_status"] = matured.get("maturity_status", "fresh")
        matured["observation_lag_hours"] = pd.to_numeric(matured.get("observation_lag_hours"), errors="coerce")
        graph_matured = matured[matured["maturity_status"].astype(str) == "fresh"].copy()
    else:
        graph_matured = matured
    if not graph_matured.empty:
        matured_colors = [horizon_color(value) for value in graph_matured["horizon_hours"]]
        fig.add_trace(
            go.Scatter(
                x=graph_matured["target_time"],
                y=graph_matured["predicted_price"],
                mode="markers",
                name="Matured forecast",
                marker={"symbol": "diamond", "size": 10, "color": matured_colors, "line": {"color": "#111827", "width": 0.5}},
                showlegend=False,
                customdata=graph_matured[["horizon_hours", "actual_price", "error", "direction_hit", "forecast_checked_at", "run_label", "observation_lag_hours"]],
                hovertemplate="Run: %{customdata[5]}<br>Matured %{customdata[0]}h forecast<br>Target: %{x|%Y-%m-%d %H:%M}<br>Predicted: $%{y:,.2f}<br>Actual: $%{customdata[1]:,.2f}<br>Error: $%{customdata[2]:,.2f}<br>Observed lag: %{customdata[6]:.2f}h<br>Direction hit: %{customdata[3]}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=graph_matured["target_time"],
                y=graph_matured["actual_price"],
                mode="markers",
                name="Actual at matured horizon",
                marker={"symbol": "x", "size": 11, "color": matured_colors, "line": {"width": 2}},
                showlegend=False,
                customdata=graph_matured[["horizon_hours", "predicted_price", "error", "direction_hit", "observed_at", "run_label", "observation_lag_hours"]],
                hovertemplate="Run: %{customdata[5]}<br>Actual after %{customdata[0]}h target<br>Observed: %{customdata[4]}<br>Actual: $%{y:,.2f}<br>Predicted: $%{customdata[1]:,.2f}<br>Error: $%{customdata[2]:,.2f}<br>Observed lag: %{customdata[6]:.2f}h<br>Direction hit: %{customdata[3]}<extra></extra>",
            )
        )
    fig.update_layout(
        height=620,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        hovermode="x unified",
        dragmode="zoom",
        showlegend=False,
        yaxis_tickprefix="$",
        xaxis={
            "type": "date",
            "title": "Target date/time",
            "tickformat": "%m-%d %H:%M",
            "rangeslider": {"visible": True, "thickness": 0.08},
            "rangeselector": {
                "buttons": [
                    {"count": 12, "label": "12h", "step": "hour", "stepmode": "backward"},
                    {"count": 24, "label": "24h", "step": "hour", "stepmode": "backward"},
                    {"count": 48, "label": "48h", "step": "hour", "stepmode": "backward"},
                    {"count": 7, "label": "7d", "step": "day", "stepmode": "backward"},
                    {"step": "all", "label": "All"},
                ]
            },
        },
    )
    st.caption("Historical averages are grouped by horizon, not only by date. Use drag-to-zoom, double-click to reset, the Plotly toolbar, or the range slider under the chart to focus on a specific period.")
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )
    st.caption(f"Showing {history_rows['run_label'].nunique()} forecast runs and {len(history_rows)} forecast horizon points.")
    st.dataframe(
        history_rows[
            [
                "run_label",
                "forecast_checked_at",
                "horizon_hours",
                "target_time",
                "predicted_price",
                "lower_price",
                "upper_price",
                "expected_direction",
                "selected_model",
            ]
        ],
        use_container_width=True,
    )
    if matured.empty:
        st.caption("No matured points yet for the selected forecast runs. A point appears after a target time passes and a later hourly agent check records the actual live price.")
    else:
        stale_count = int((matured["maturity_status"].astype(str) != "fresh").sum()) if "maturity_status" in matured else 0
        if stale_count:
            st.caption(f"Matured graph excludes {stale_count} stale points whose first available live check was too far after the target time. They remain in the table for audit.")
        else:
            st.caption("Matured points compare each forecast target against the first later hourly live check.")
        render_matured_error_chart(graph_matured)
        st.dataframe(
            matured[
                [
                    "run_label",
                    "forecast_checked_at",
                    "horizon_hours",
                    "target_time",
                    "predicted_price",
                    "actual_price",
                    "error",
                    "observation_lag_hours",
                    "maturity_status",
                    "direction_hit",
                    "observed_at",
                ]
            ],
            use_container_width=True,
        )


def render_matured_error_chart(matured: pd.DataFrame) -> None:
    st.subheader("Matured Error: Actual - Prediction")
    frame = matured.copy()
    frame["target_time"] = pd.to_datetime(frame["target_time"], errors="coerce")
    frame["horizon_hours"] = pd.to_numeric(frame["horizon_hours"], errors="coerce")
    frame["error"] = pd.to_numeric(frame["error"], errors="coerce")
    frame["absolute_error"] = pd.to_numeric(frame["absolute_error"], errors="coerce")
    frame = frame.dropna(subset=["target_time", "horizon_hours", "error", "absolute_error"]).sort_values(["horizon_hours", "target_time"])
    if frame.empty:
        st.info("No valid matured forecast errors to plot yet.")
        return
    mae = frame["absolute_error"].mean()
    bias = frame["error"].mean()
    direction_rate = (frame["direction_hit"].astype(str).str.lower() == "yes").mean()
    cols = st.columns(3)
    cols[0].metric("Matured MAE", fmt_money(mae))
    cols[1].metric("Bias", fmt_money(bias))
    cols[2].metric("Direction hit", f"{direction_rate * 100:.1f}%")

    fig = go.Figure()
    for horizon, horizon_frame in frame.groupby("horizon_hours", sort=True):
        fig.add_trace(
            go.Scatter(
                x=horizon_frame["target_time"],
                y=horizon_frame["error"],
                name=f"{horizon_label(horizon)} error",
                mode="lines",
                line={"color": horizon_color(horizon), "width": 2},
                showlegend=True,
                customdata=horizon_frame[["run_label", "horizon_hours", "predicted_price", "actual_price", "direction_hit", "observed_at"]],
                hovertemplate="Run: %{customdata[0]}<br>%{customdata[1]:g}h target: %{x|%Y-%m-%d %H:%M}<br>Error: $%{y:,.2f}<br>Predicted: $%{customdata[2]:,.2f}<br>Actual: $%{customdata[3]:,.2f}<br>Direction hit: %{customdata[4]}<br>Observed: %{customdata[5]}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color="#475569")
    fig.update_layout(
        height=430,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        hovermode="x unified",
        dragmode="zoom",
        showlegend=True,
        legend={"orientation": "h", "y": 1.08, "x": 0, "title": {"text": "Horizon"}},
        yaxis={"title": "Actual - prediction", "tickprefix": "$"},
        xaxis={
            "type": "date",
            "title": "Matured target date/time",
            "tickformat": "%m-%d %H:%M",
            "rangeslider": {"visible": True, "thickness": 0.08},
            "rangeselector": {
                "buttons": [
                    {"count": 24, "label": "24h", "step": "hour", "stepmode": "backward"},
                    {"count": 48, "label": "48h", "step": "hour", "stepmode": "backward"},
                    {"count": 7, "label": "7d", "step": "day", "stepmode": "backward"},
                    {"step": "all", "label": "All"},
                ]
            },
        },
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )
    horizon_metrics = (
        frame.assign(direction_ok=frame["direction_hit"].astype(str).str.lower() == "yes")
        .groupby("horizon_hours", as_index=False)
        .agg(
            points=("error", "count"),
            mae=("absolute_error", "mean"),
            bias=("error", "mean"),
            direction_hit=("direction_ok", "mean"),
        )
    )
    horizon_metrics["horizon"] = horizon_metrics["horizon_hours"].map(horizon_label)
    horizon_metrics["mae"] = horizon_metrics["mae"].map(fmt_money)
    horizon_metrics["bias"] = horizon_metrics["bias"].map(fmt_money)
    horizon_metrics["direction_hit"] = horizon_metrics["direction_hit"].map(lambda value: f"{value * 100:.1f}%")
    st.dataframe(horizon_metrics[["horizon", "points", "mae", "bias", "direction_hit"]], use_container_width=True)


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


def load_forecast_history_points(agent_report_path: Path) -> list[dict[str, Any]]:
    reports = load_agent_history(agent_report_path)
    runs: dict[str, dict[str, Any]] = {}
    for report in reports:
        forecast = report.get("forecast") or {}
        rows = forecast.get("forecasts") or []
        forecast_at = parse_dt(forecast.get("created_at_utc") or report.get("checked_at_utc") or report.get("checked_at"))
        if not rows or forecast_at is None:
            continue
        key = forecast_at.isoformat()
        checked_at = parse_dt(report.get("checked_at_utc") or report.get("checked_at")) or forecast_at
        existing = runs.get(key)
        if existing is None or checked_at > existing["checked_at"]:
            runs[key] = {"forecast_at": forecast_at, "checked_at": checked_at, "rows": rows}

    history: list[dict[str, Any]] = []
    ordered_runs = sorted(runs.values(), key=lambda item: item["forecast_at"])
    for run_index, run in enumerate(ordered_runs, start=1):
        forecast_at = run["forecast_at"]
        run_label = f"run {run_index} | {forecast_at.strftime('%m-%d %H:%M')}"
        for row in normalize_forecast_rows(run["rows"], as_of=forecast_at):
            history.append(
                {
                    **row,
                    "run_index": run_index,
                    "run_label": run_label,
                    "forecast_checked_at": forecast_at.isoformat(),
                }
            )
    return history


def load_matured_forecast_points(agent_report_path: Path) -> list[dict[str, Any]]:
    reports = load_agent_history(agent_report_path)
    forecast_runs = {
        str(row["forecast_checked_at"]): str(row["run_label"])
        for row in load_forecast_history_points(agent_report_path)
        if row.get("forecast_checked_at") and row.get("run_label")
    }
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
            observation_lag_hours = max(0.0, (observation["checked_at"] - target_time).total_seconds() / 3600.0)
            maturity_status = "fresh" if observation_lag_hours <= MAX_MATURED_OBSERVATION_LAG_HOURS else "stale_observation_lag"
            direction_hit = direction_match(start_price=current_price, predicted_price=predicted, actual_price=actual)
            key = (forecast_at.isoformat(), float(horizon))
            matured[key] = {
                "run_label": forecast_runs.get(forecast_at.isoformat(), forecast_at.strftime("%m-%d %H:%M")),
                "forecast_checked_at": forecast_at.isoformat(),
                "horizon_hours": float(horizon),
                "target_time": target_time,
                "predicted_price": predicted,
                "actual_price": actual,
                "error": actual - predicted,
                "absolute_error": abs(actual - predicted),
                "observation_lag_hours": observation_lag_hours,
                "maturity_status": maturity_status,
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


def horizon_label(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return "-"
    return f"{number:g}h"


def horizon_color(value: Any) -> str:
    number = to_float(value)
    palette = ["#2563eb", "#f97316", "#16a34a", "#7c3aed", "#db2777", "#0891b2", "#dc2626"]
    if number is None:
        return "#64748b"
    common = {
        1.0: "#2563eb",
        2.0: "#f97316",
        3.0: "#16a34a",
        6.0: "#7c3aed",
        12.0: "#db2777",
        24.0: "#0891b2",
        48.0: "#dc2626",
    }
    rounded = round(float(number), 3)
    if rounded in common:
        return common[rounded]
    return palette[int(abs(rounded) * 10) % len(palette)]


def rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    if len(value) != 6:
        return f"rgba(100,116,139,{alpha})"
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha})"


if __name__ == "__main__":
    main()
