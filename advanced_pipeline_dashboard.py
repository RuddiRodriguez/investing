"""Standalone Streamlit dashboard for advanced pipeline paper-trading runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.errors import EmptyDataError

from scripts.advanced_pipeline.forecast import (
    fetch_forecast_market_data,
    generate_forecast,
    load_forecast_artifacts,
    resolve_prediction_start_date,
    save_forecast_artifacts,
    train_forecast_artifacts,
)
from scripts.backtrader_pipeline import run_advanced_backtrader_analysis
from scripts.stochastic_pipeline import run_stochastic_analysis, run_stochastic_backtest
from scripts.transformer_pipeline import (
    generate_transformer_analysis,
    load_transformer_artifacts,
    train_transformer_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_ROOT = PROJECT_ROOT / "simulation_runs" / "advanced_pipeline"


st.set_page_config(page_title="Advanced Pipeline Paper Agent", page_icon="📊", layout="wide")


def list_runs(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    return sorted([path for path in run_root.iterdir() if path.is_dir()], reverse=True)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path, date_columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
    for column in date_columns or []:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


@st.cache_data(show_spinner="Loading historical market data...")
def load_cached_forecast_market_data(
    ticker: str,
    prediction_start_date: str,
    refresh_token: int,
) -> pd.DataFrame:
    return fetch_forecast_market_data(
        ticker=ticker,
        prediction_start_date=prediction_start_date,
        force_refresh=refresh_token > 0,
    )


def load_run_bundle(run_dir: Path) -> dict[str, pd.DataFrame | dict]:
    return {
        "metadata": load_json(run_dir / "metadata.json"),
        "summary": load_json(run_dir / "summary.json"),
        "daily": load_csv(run_dir / "daily_summary.csv", ["signal_date", "next_date"]),
        "trades": load_csv(run_dir / "trades.csv", ["signal_date"]),
        "positions": load_csv(run_dir / "positions.csv", ["signal_date", "entry_date"]),
        "decisions": load_csv(run_dir / "decision_snapshots.csv", ["signal_date"]),
    }


def compute_weekly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    weekly = daily.copy()
    weekly["week"] = weekly["signal_date"].dt.to_period("W").dt.start_time
    summary = (
        weekly.groupby("week", as_index=False)
        .agg(
            weekly_pnl=("daily_pnl", "sum"),
            weekly_return=("daily_return", lambda values: (1.0 + values).prod() - 1.0),
            end_portfolio_value=("portfolio_value_end", "last"),
            end_benchmark_value=("benchmark_value_end", "last"),
        )
        .sort_values("week")
    )
    return summary


def compute_drawdown(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    drawdown = daily[["signal_date", "portfolio_value_end"]].copy()
    drawdown["drawdown"] = drawdown["portfolio_value_end"] / drawdown["portfolio_value_end"].cummax() - 1.0
    return drawdown


def latest_positions(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return positions
    latest_date = positions["signal_date"].max()
    latest = positions.loc[positions["signal_date"] == latest_date].copy()
    return latest.sort_values("market_value", ascending=False)


def render_header(bundle: dict[str, pd.DataFrame | dict], run_dir: Path) -> None:
    metadata = bundle["metadata"]
    summary = bundle["summary"]
    daily = bundle["daily"]
    st.title("Advanced Pipeline Paper Agent")
    st.caption("Separate live monitor for the historical paper-trading agent. Signals run on daily bars, with live file refresh every 10 seconds.")
    st.write(f"Run: {run_dir.name}")
    st.caption(
        "Decision frequency: "
        f"{metadata.get('decision_frequency', 'daily_close')} | "
        "Execution: "
        f"{metadata.get('execution_timing', 'n/a')} | "
        "Data polling: every "
        f"{metadata.get('poll_seconds', 'n/a')} seconds when waiting for a new day"
    )
    progress = 0.0
    if metadata.get("simulation_days"):
        progress = min(1.0, float(metadata.get("completed_steps", 0)) / float(metadata.get("simulation_days", 1)))
    st.progress(progress)

    if daily.empty:
        st.info("This run has not produced any daily snapshots yet.")
        return

    latest_value = float(daily["portfolio_value_end"].iloc[-1])
    latest_benchmark = float(daily["benchmark_value_end"].iloc[-1])
    total_return = float(summary.get("total_return", latest_value / daily["portfolio_value_start"].iloc[0] - 1.0))
    benchmark_return = float(summary.get("benchmark_return", latest_benchmark / daily["benchmark_value_start"].iloc[0] - 1.0))
    max_drawdown = float(summary.get("max_drawdown", compute_drawdown(daily)["drawdown"].min()))
    trade_count = int(summary.get("trade_count", 0))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Portfolio value", f"{latest_value:,.2f}")
    col2.metric("Total return", f"{total_return:.2%}")
    col3.metric("Benchmark return", f"{benchmark_return:.2%}")
    col4.metric("Max drawdown", f"{max_drawdown:.2%}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Trade count", f"{trade_count}")
    col6.metric("Win rate", f"{float(summary.get('win_rate', 0.0)):.2%}")
    col7.metric("Completed days", f"{metadata.get('completed_steps', 0)} / {metadata.get('simulation_days', 0)}")
    col8.metric("Latest signal day", str(metadata.get("latest_signal_date", "n/a")))


def render_charts(bundle: dict[str, pd.DataFrame | dict]) -> None:
    daily = bundle["daily"]
    weekly = compute_weekly_summary(daily)
    drawdown = compute_drawdown(daily)
    if daily.empty:
        return

    equity = go.Figure()
    equity.add_trace(go.Scatter(x=daily["signal_date"], y=daily["portfolio_value_end"], mode="lines", name="Portfolio"))
    equity.add_trace(go.Scatter(x=daily["signal_date"], y=daily["benchmark_value_end"], mode="lines", name="Benchmark"))
    equity.update_layout(title="Portfolio vs Benchmark", height=380)

    pnl = px.bar(daily, x="signal_date", y="daily_pnl", title="Daily PnL")
    pnl.update_layout(height=320)

    weekly_chart = px.bar(weekly, x="week", y="weekly_return", title="Weekly Return") if not weekly.empty else go.Figure()
    weekly_chart.update_layout(height=320)

    drawdown_chart = px.area(drawdown, x="signal_date", y="drawdown", title="Drawdown") if not drawdown.empty else go.Figure()
    drawdown_chart.update_layout(height=320)

    top_left, top_right = st.columns(2)
    top_left.plotly_chart(equity, use_container_width=True)
    top_right.plotly_chart(pnl, use_container_width=True)
    bottom_left, bottom_right = st.columns(2)
    bottom_left.plotly_chart(weekly_chart, use_container_width=True)
    bottom_right.plotly_chart(drawdown_chart, use_container_width=True)


def render_tables(bundle: dict[str, pd.DataFrame | dict]) -> None:
    positions = latest_positions(bundle["positions"])
    trades = bundle["trades"]
    decisions = bundle["decisions"]

    if not positions.empty:
        st.subheader("Current Portfolio")
        display = positions[[
            "ticker",
            "decision",
            "price",
            "average_cost",
            "units",
            "market_value",
            "portfolio_weight",
            "unrealized_pnl",
            "entry_date",
            "holding_days",
        ]].copy()
        st.dataframe(display, use_container_width=True)

    if not trades.empty:
        st.subheader("Recent Trades")
        recent_trades = trades.sort_values("signal_date", ascending=False).head(30)
        st.dataframe(recent_trades, use_container_width=True)

    if not decisions.empty:
        st.subheader("Latest Decision Snapshot")
        latest_date = decisions["signal_date"].max()
        latest = decisions.loc[decisions["signal_date"] == latest_date].copy()
        st.dataframe(
            latest[[
                "ticker",
                "decision",
                "confidence",
                "expected_excess_return",
                "lower_bound",
                "upper_bound",
                "risk_score",
                "alpha_score",
                "main_positive_drivers",
                "main_risks",
                "reason_codes",
            ]],
            use_container_width=True,
        )


def render_dashboard(run_dir: Path) -> None:
    bundle = load_run_bundle(run_dir)
    render_header(bundle, run_dir)
    render_charts(bundle)
    render_tables(bundle)


def render_forecast_summary(result: dict, selected_threshold_pct: float) -> None:
    forecast = result["forecast"]
    final_row = forecast.iloc[-1]
    decision_horizon = int(result["metadata"].get("decision_horizon", int(final_row["horizon_day"])))
    decision_row = forecast.loc[forecast["horizon_day"] == decision_horizon].iloc[-1]
    actual_available = forecast["actual_price"].notna().any() if "actual_price" in forecast.columns else False
    st.subheader("Forecast Summary")
    st.metric("Prediction start", result["prediction_start_date"])
    st.metric("Anchor price", f"{result['anchor_price']:.2f}")
    st.metric("Forecast end price", f"{final_row['predicted_price']:.2f}")
    st.metric("Lower / Upper band", f"{final_row['lower_price']:.2f} / {final_row['upper_price']:.2f}")
    st.metric("Primary decision horizon", f"{decision_horizon} day")
    if result["metadata"].get("signal_benchmark_symbol"):
        st.metric("Signal benchmark", str(result["metadata"]["signal_benchmark_symbol"]))
    st.metric(
        f"Prob. excess return >= {selected_threshold_pct:.1f}%",
        f"{decision_row['probability_threshold_hit']:.2%}",
    )
    if "trust_probability" in decision_row.index:
        st.metric("Trust probability", f"{decision_row['trust_probability']:.2%}")
    if "trade_probability" in decision_row.index:
        st.metric("Trade probability", f"{decision_row['trade_probability']:.2%}")
    latest_actual = forecast.loc[forecast["actual_price"].notna()].tail(1)
    if actual_available and not latest_actual.empty:
        actual_row = latest_actual.iloc[0]
        st.metric("Latest actual price", f"{actual_row['actual_price']:.2f}")
        st.metric("Latest residual", f"{actual_row['residual']:.2f}")
    metrics = result["metadata"].get("validation_metrics", {}).get(decision_horizon, {})
    if metrics:
        gate = result["metadata"].get("deployment_gate", {})
        best_mae_baseline = gate.get("best_baseline_mae", np.nan)
        best_accuracy_baseline = gate.get("best_baseline_accuracy", np.nan)
        st.metric("Validation MAE", f"{metrics.get('model_validation_mae_log_return', float('nan')):.4f}")
        st.metric("Dummy validation MAE", f"{metrics.get('dummy_validation_mae_log_return', float('nan')):.4f}")
        st.metric(
            "Relative momentum baseline MAE",
            f"{metrics.get('relative_momentum_validation_mae_log_return', float('nan')):.4f}",
        )
        st.metric("Best baseline MAE", f"{best_mae_baseline:.4f}")
        st.metric(
            "Validation threshold accuracy",
            f"{metrics.get('model_validation_threshold_accuracy', float('nan')):.2%}",
        )
        st.metric(
            "Dummy threshold accuracy",
            f"{metrics.get('dummy_validation_threshold_accuracy', float('nan')):.2%}",
        )
        st.metric("Best baseline accuracy", f"{best_accuracy_baseline:.2%}")
        st.metric(
            "Probability calibration gap",
            f"{metrics.get('validation_probability_calibration_gap', float('nan')):.2%}",
        )
        st.metric(
            "Trusted threshold accuracy",
            f"{metrics.get('trusted_validation_threshold_accuracy', float('nan')):.2%}",
        )
        st.metric("Mean trust probability", f"{metrics.get('mean_trust_probability', float('nan')):.2%}")
        if result["metadata"].get("suggested_action"):
            st.metric("Suggested action", str(result["metadata"]["suggested_action"]))
        st.metric(
            "Interval coverage",
            f"{metrics.get('model_validation_interval_coverage', float('nan')):.2%}",
        )
        st.metric("Validation bias", f"{metrics.get('model_validation_bias_log_return', float('nan')):.4f}")
        residual_drift = result["metadata"].get("recent_residual_drift")
        if residual_drift is not None:
            st.metric("Recent residual drift", f"{residual_drift:.2f}")
        st.metric(
            "Walk-forward folds / embargo",
            f"{int(metrics.get('walk_forward_folds', 0))} / {int(metrics.get('embargo_days', 0))}",
        )
        if gate:
            if gate.get("approved"):
                st.success(f"Deployment gate passed for the primary {decision_horizon}-day horizon.")
            else:
                st.warning(
                    "Low confidence: " + ", ".join(str(reason).replace("_", " ") for reason in gate.get("reasons", []))
                )


def build_horizon_diagnostics_table(result: dict) -> pd.DataFrame:
    metrics = result["metadata"].get("validation_metrics", {})
    if not metrics:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for horizon_key, values in sorted(metrics.items(), key=lambda item: int(item[0])):
        horizon = int(horizon_key)
        rows.append(
            {
                "horizon": horizon,
                "decision_focus": horizon == int(result["metadata"].get("decision_horizon", -1)),
                "model_mae": values.get("model_validation_mae_log_return"),
                "dummy_mae": values.get("dummy_validation_mae_log_return"),
                "best_baseline_mae": np.nanmin(
                    [
                        values.get("dummy_validation_mae_log_return", np.nan),
                        values.get("zero_validation_mae_log_return", np.nan),
                        values.get("last_return_validation_mae_log_return", np.nan),
                        values.get("rolling_mean_validation_mae_log_return", np.nan),
                        values.get("rolling_mean_10d_validation_mae_log_return", np.nan),
                        values.get("benchmark_match_validation_mae_log_return", np.nan),
                        values.get("relative_last_validation_mae_log_return", np.nan),
                        values.get("relative_momentum_validation_mae_log_return", np.nan),
                    ]
                ),
                "model_acc": values.get("model_validation_threshold_accuracy"),
                "dummy_acc": values.get("dummy_validation_threshold_accuracy"),
                "best_baseline_acc": np.nanmax(
                    [
                        values.get("dummy_validation_threshold_accuracy", np.nan),
                        values.get("last_return_validation_threshold_accuracy", np.nan),
                        values.get("rolling_mean_validation_threshold_accuracy", np.nan),
                        values.get("rolling_mean_10d_validation_threshold_accuracy", np.nan),
                    ]
                ),
                "trusted_acc": values.get("trusted_validation_threshold_accuracy"),
                "bias": values.get("model_validation_bias_log_return"),
                "coverage": values.get("model_validation_interval_coverage"),
                "calibration_gap": values.get("validation_probability_calibration_gap"),
                "trust_prob": values.get("mean_trust_probability"),
                "signal_benchmark": values.get("signal_benchmark_symbol"),
            }
        )
    return pd.DataFrame(rows)


def build_actual_price_series(result: dict) -> pd.DataFrame:
    """Join pre-forecast history and known future actuals into one visible series."""

    history = result["history"].copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history[["date", "price"]]

    forecast = result["forecast"].copy()
    forecast["forecast_date"] = pd.to_datetime(forecast["forecast_date"], errors="coerce")
    known_real_prices = forecast.loc[forecast["actual_price"].notna(), ["forecast_date", "actual_price"]].copy()
    known_real_prices = known_real_prices.rename(columns={"forecast_date": "date", "actual_price": "price"})

    actual = pd.concat([history, known_real_prices], ignore_index=True)
    actual = actual.dropna(subset=["date", "price"])
    return actual.sort_values("date").drop_duplicates(subset=["date"], keep="last")


def build_metrics_table(result: dict) -> pd.DataFrame:
    metrics = result["metadata"].get("validation_metrics", {})
    if not metrics:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for horizon_key, values in sorted(metrics.items(), key=lambda item: int(item[0])):
        horizon = int(horizon_key)
        rows.append(
            {
                "horizon_day": horizon,
                "walk_forward_folds": values.get("walk_forward_folds"),
                "tuning_walk_forward_folds": values.get("tuning_walk_forward_folds"),
                "embargo_days": values.get("embargo_days"),
                "train_rows": values.get("train_rows"),
                "validation_rows": values.get("validation_rows"),
                "train_window": f"{values.get('train_start_date', '')} -> {values.get('train_end_date', '')}",
                "validation_window": f"{values.get('validation_start_date', '')} -> {values.get('validation_end_date', '')}",
                "model_train_mae": values.get("model_train_mae_log_return"),
                "dummy_train_mae": values.get("dummy_train_mae_log_return"),
                "model_validation_mae": values.get("model_validation_mae_log_return"),
                "dummy_validation_mae": values.get("dummy_validation_mae_log_return"),
                "zero_validation_mae": values.get("zero_validation_mae_log_return"),
                "last_return_validation_mae": values.get("last_return_validation_mae_log_return"),
                "rolling_mean_validation_mae": values.get("rolling_mean_validation_mae_log_return"),
                "rolling_mean_10d_validation_mae": values.get("rolling_mean_10d_validation_mae_log_return"),
                "benchmark_match_validation_mae": values.get("benchmark_match_validation_mae_log_return"),
                "relative_last_validation_mae": values.get("relative_last_validation_mae_log_return"),
                "relative_momentum_validation_mae": values.get("relative_momentum_validation_mae_log_return"),
                "best_baseline_validation_mae": np.nanmin(
                    [
                        values.get("dummy_validation_mae_log_return", np.nan),
                        values.get("zero_validation_mae_log_return", np.nan),
                        values.get("last_return_validation_mae_log_return", np.nan),
                        values.get("rolling_mean_validation_mae_log_return", np.nan),
                        values.get("rolling_mean_10d_validation_mae_log_return", np.nan),
                        values.get("benchmark_match_validation_mae_log_return", np.nan),
                        values.get("relative_last_validation_mae_log_return", np.nan),
                        values.get("relative_momentum_validation_mae_log_return", np.nan),
                    ]
                ),
                "validation_mae_delta_vs_dummy": values.get("dummy_validation_mae_log_return", np.nan)
                - values.get("model_validation_mae_log_return", np.nan),
                "model_train_rmse": values.get("model_train_rmse_log_return"),
                "dummy_train_rmse": values.get("dummy_train_rmse_log_return"),
                "model_validation_rmse": values.get("model_validation_rmse_log_return"),
                "dummy_validation_rmse": values.get("dummy_validation_rmse_log_return"),
                "model_train_accuracy": values.get("model_train_threshold_accuracy"),
                "dummy_train_accuracy": values.get("dummy_train_threshold_accuracy"),
                "model_validation_accuracy": values.get("model_validation_threshold_accuracy"),
                "dummy_validation_accuracy": values.get("dummy_validation_threshold_accuracy"),
                "last_return_validation_accuracy": values.get("last_return_validation_threshold_accuracy"),
                "rolling_mean_validation_accuracy": values.get("rolling_mean_validation_threshold_accuracy"),
                "rolling_mean_10d_validation_accuracy": values.get("rolling_mean_10d_validation_threshold_accuracy"),
                "best_baseline_validation_accuracy": np.nanmax(
                    [
                        values.get("dummy_validation_threshold_accuracy", np.nan),
                        values.get("last_return_validation_threshold_accuracy", np.nan),
                        values.get("rolling_mean_validation_threshold_accuracy", np.nan),
                        values.get("rolling_mean_10d_validation_threshold_accuracy", np.nan),
                    ]
                ),
                "validation_accuracy_delta_vs_dummy": values.get("model_validation_threshold_accuracy", np.nan)
                - values.get("dummy_validation_threshold_accuracy", np.nan),
                "model_validation_brier": values.get("model_validation_brier"),
                "dummy_validation_brier": values.get("dummy_validation_brier"),
                "trusted_validation_brier": values.get("trusted_validation_brier"),
                "trusted_validation_accuracy": values.get("trusted_validation_threshold_accuracy"),
                "mean_trust_probability": values.get("mean_trust_probability"),
                "validation_event_rate": values.get("validation_event_rate"),
                "validation_predicted_probability_mean": values.get("validation_predicted_probability_mean"),
                "validation_probability_calibration_gap": values.get("validation_probability_calibration_gap"),
                "interval_coverage": values.get("model_validation_interval_coverage"),
                "interval_width": values.get("model_validation_interval_width"),
                "interval_adjustment": values.get("interval_adjustment"),
                "validation_bias": values.get("model_validation_bias_log_return"),
                "validation_tp": values.get("validation_tp"),
                "validation_tn": values.get("validation_tn"),
                "validation_fp": values.get("validation_fp"),
                "validation_fn": values.get("validation_fn"),
            }
        )

    return pd.DataFrame(rows)


def build_forecast_price_chart(result: dict, *, log_y_axis: bool = False) -> go.Figure:
    actual_prices = build_actual_price_series(result)
    forecast = result["forecast"].copy()
    forecast["forecast_date"] = pd.to_datetime(forecast["forecast_date"], errors="coerce")
    cutoff = pd.Timestamp(result["prediction_start_date"])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=actual_prices["date"],
            y=actual_prices["price"],
            mode="lines",
            name="Actual price",
            line={"color": "#111827", "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast["forecast_date"],
            y=forecast["upper_price"],
            mode="lines",
            line={"color": "rgba(59,130,246,0.0)"},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast["forecast_date"],
            y=forecast["lower_price"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(59,130,246,0.18)",
            line={"color": "rgba(59,130,246,0.0)"},
            name="Confidence band",
            hoverinfo="skip",
        )
    )
    error_plus = (forecast["upper_price"] - forecast["predicted_price"]).clip(lower=0.0)
    error_minus = (forecast["predicted_price"] - forecast["lower_price"]).clip(lower=0.0)
    figure.add_trace(
        go.Scatter(
            x=forecast["forecast_date"],
            y=forecast["predicted_price"],
            mode="lines+markers",
            name="Forecast median",
            line={"color": "#2563eb", "width": 3},
            marker={"size": 7},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": error_plus,
                "arrayminus": error_minus,
                "color": "rgba(37,99,235,0.6)",
                "thickness": 1,
            },
        )
    )
    known_real_prices = forecast.loc[forecast["actual_price"].notna()].copy()
    if not known_real_prices.empty:
        figure.add_trace(
            go.Scatter(
                x=known_real_prices["forecast_date"],
                y=known_real_prices["actual_price"],
                mode="markers",
                name="Real price after forecast start",
                marker={"color": "#059669", "size": 8, "symbol": "circle"},
            )
        )
    figure.add_vline(x=cutoff, line_dash="dash", line_color="#6b7280")
    figure.add_annotation(
        x=cutoff,
        y=actual_prices["price"].max(),
        text="Forecast start",
        showarrow=False,
        yshift=14,
        font={"color": "#6b7280"},
    )
    figure.update_layout(
        title="Actual Price and Forecast",
        height=560,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 40, "r": 24, "t": 80, "b": 40},
    )
    figure.update_yaxes(type="log" if log_y_axis else "linear")
    return figure


def build_residual_chart(result: dict) -> go.Figure | None:
    forecast = result["forecast"].copy()
    forecast = forecast.loc[forecast["actual_price"].notna()].copy()
    if forecast.empty:
        return None
    colors = ["#059669" if value >= 0 else "#dc2626" for value in forecast["residual"]]
    figure = go.Figure(
        go.Bar(
            x=forecast["forecast_date"],
            y=forecast["residual"],
            marker_color=colors,
            name="Residual",
        )
    )
    figure.add_hline(y=0.0, line_dash="dash", line_color="#6b7280")
    figure.update_layout(title="Residuals (Actual - Predicted)", height=320)
    return figure


def build_stochastic_price_chart(result: dict, *, log_y_axis: bool = False) -> go.Figure:
    history = result["history"].copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    simulation_start = pd.Timestamp(result["simulation_start_date"])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["price"],
            mode="lines",
            name="History",
            line={"color": "#1f2937", "width": 2},
        )
    )

    for label, color in (("gbm", "#2563eb"), ("jump", "#7c3aed"), ("garch", "#dc2626"), ("egarch", "#059669"), ("regime", "#d97706")):
        model_result = result[label]
        cone = model_result["price_cone"].copy() if label in {"gbm", "jump"} else model_result.price_cone.copy()
        cone = cone.reset_index().rename(columns={cone.index.name or "index": "date"})
        if "index" in cone.columns and "date" not in cone.columns:
            cone = cone.rename(columns={"index": "date"})
        cone["date"] = pd.to_datetime(cone["date"], errors="coerce")
        model_name = model_result["model_name"] if label == "gbm" else model_result.model_name
        fill_color = {
            "gbm": "rgba(37,99,235,0.10)",
            "jump": "rgba(124,58,237,0.10)",
            "garch": "rgba(220,38,38,0.10)",
            "egarch": "rgba(5,150,105,0.10)",
            "regime": "rgba(217,119,6,0.12)",
        }[label]
        figure.add_trace(
            go.Scatter(x=cone["date"], y=cone["p90"], mode="lines", line={"color": color, "width": 0}, hoverinfo="skip", showlegend=False)
        )
        figure.add_trace(
            go.Scatter(
                x=cone["date"],
                y=cone["p10"],
                mode="lines",
                line={"color": color, "width": 0},
                fill="tonexty",
                fillcolor=fill_color,
                hoverinfo="skip",
                name=f"{model_name} 10-90% band",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=cone["date"],
                y=cone["p50"],
                mode="lines",
                name=f"{model_name} median",
                line={"color": color, "width": 2, "dash": "dash" if label in {"gbm", "jump", "regime"} else "solid"},
            )
        )

    actual_future_plot = result.get("actual_future_plot")
    if isinstance(actual_future_plot, pd.DataFrame) and not actual_future_plot.empty:
        future = actual_future_plot.copy()
        future["date"] = pd.to_datetime(future["date"], errors="coerce")
        figure.add_trace(
            go.Scatter(
                x=future["date"],
                y=future["price"],
                mode="lines+markers",
                name="Realized future price",
                line={"color": "#111827", "width": 3},
                marker={"size": 6},
            )
        )

    figure.add_vline(x=simulation_start, line_dash="dash", line_color="#6b7280")
    figure.add_annotation(
        x=simulation_start,
        y=float(history["price"].max()),
        text="Simulation start",
        showarrow=False,
        yshift=14,
        font={"color": "#6b7280"},
    )
    figure.update_layout(title="Stochastic Price Cones", height=520)
    figure.update_yaxes(type="log" if log_y_axis else "linear")
    return figure


def build_stochastic_volatility_chart(result: dict) -> go.Figure:
    figure = go.Figure()
    for model_result, color in ((result["garch"], "#dc2626"), (result["egarch"], "#059669")):
        forecast = model_result.volatility_forecast.copy().reset_index().rename(columns={"index": "date"})
        forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce")
        figure.add_trace(
            go.Scatter(
                x=forecast["date"],
                y=forecast["annualized_volatility"],
                mode="lines+markers",
                name=model_result.model_name,
                line={"color": color, "width": 2},
            )
        )
    figure.update_layout(
        title="Forward Volatility Forecast",
        height=360,
        yaxis_tickformat=".1%",
    )
    return figure


def build_stochastic_actual_future_series(prices: pd.DataFrame, simulation_start: pd.Timestamp, horizon_days: int) -> pd.DataFrame:
    future = prices.loc[prices.index > simulation_start, [prices.columns[0]]].head(horizon_days).copy()
    if future.empty:
        return pd.DataFrame(columns=["date", "price"])
    future = future.reset_index()
    future.columns = ["date", "price"]
    return future


def build_stochastic_actual_post_start_series(prices: pd.DataFrame, simulation_start: pd.Timestamp) -> pd.DataFrame:
    future = prices.loc[prices.index > simulation_start, [prices.columns[0]]].copy()
    if future.empty:
        return pd.DataFrame(columns=["date", "price"])
    future = future.reset_index()
    future.columns = ["date", "price"]
    return future


def build_stochastic_comparison_table(result: dict) -> pd.DataFrame:
    comparison = result["comparison"].copy()
    actual_future = result.get("actual_future")
    if not isinstance(actual_future, pd.DataFrame) or actual_future.empty:
        comparison["actual_terminal_price"] = np.nan
        comparison["terminal_error"] = np.nan
        comparison["terminal_abs_error"] = np.nan
        return comparison

    actual_terminal_price = float(actual_future["price"].iloc[-1])
    comparison["actual_terminal_price"] = actual_terminal_price
    comparison["terminal_error"] = comparison["terminal_median_price"] - actual_terminal_price
    comparison["terminal_abs_error"] = comparison["terminal_error"].abs()
    return comparison


def build_stochastic_backtest_error_chart(backtest: dict) -> go.Figure:
    detail = backtest["detail"].copy()
    detail["simulation_start_date"] = pd.to_datetime(detail["simulation_start_date"], errors="coerce")
    figure = px.line(
        detail,
        x="simulation_start_date",
        y="terminal_abs_error",
        color="model",
        markers=True,
        title="Rolling Backtest Terminal Absolute Error",
    )
    figure.update_layout(height=360)
    return figure


def build_stochastic_regime_probability_chart(result: dict) -> go.Figure:
    regime = result["regime"]
    forecast_probabilities = regime.forecast_state_probabilities.copy().reset_index().rename(columns={"index": "date"})
    figure = go.Figure()
    for column in forecast_probabilities.columns:
        if column == "date":
            continue
        figure.add_trace(
            go.Scatter(
                x=forecast_probabilities["date"],
                y=forecast_probabilities[column],
                mode="lines",
                stackgroup="one",
                name=column.replace("_", " ").title(),
            )
        )
    figure.update_layout(title="Regime Probability Forecast", height=320, yaxis_tickformat=".0%")
    return figure


def build_ta_indicator_frame(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    from ta import add_all_ta_features

    symbol = ticker.upper()
    column_map = {
        "Open": f"OPEN_{symbol}",
        "High": f"HIGH_{symbol}",
        "Low": f"LOW_{symbol}",
        "Close": symbol,
        "Volume": f"VOLUME_{symbol}",
    }
    missing = [name for name, column in column_map.items() if column not in prices.columns]
    if missing:
        raise ValueError(f"Technical analysis for {symbol} requires OHLCV data. Missing: {', '.join(missing)}.")

    frame = pd.DataFrame(
        {name: pd.to_numeric(prices[column], errors="coerce") for name, column in column_map.items()},
        index=pd.to_datetime(prices.index, errors="coerce"),
    )
    frame = frame.dropna().sort_index()
    if frame.empty:
        raise ValueError(f"No valid OHLCV rows are available for {symbol}.")

    ta_frame = add_all_ta_features(
        frame.copy(),
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=False,
    )
    required_indicator_columns = [
        "trend_sma_fast",
        "trend_sma_slow",
        "volatility_bbh",
        "volatility_bbl",
        "volatility_bbp",
        "momentum_rsi",
        "momentum_stoch_rsi_k",
        "momentum_stoch_rsi_d",
        "trend_macd",
        "trend_macd_signal",
        "trend_macd_diff",
        "trend_adx",
        "trend_adx_pos",
        "trend_adx_neg",
        "volatility_atr",
        "volume_cmf",
        "volume_obv",
    ]
    ta_frame = ta_frame.dropna(subset=required_indicator_columns).copy()
    if ta_frame.empty:
        raise ValueError(f"Not enough history is available to compute TA indicators for {symbol}.")
    return ta_frame


def build_ta_signal_table(ta_frame: pd.DataFrame) -> pd.DataFrame:
    latest = ta_frame.iloc[-1]
    rsi_signal = "Overbought" if latest["momentum_rsi"] >= 70.0 else "Oversold" if latest["momentum_rsi"] <= 30.0 else "Neutral"
    macd_signal = "Bullish momentum" if latest["trend_macd_diff"] >= 0.0 else "Bearish momentum"
    trend_signal = "Strong trend" if latest["trend_adx"] >= 25.0 else "Weak trend"
    band_signal = (
        "Above upper band"
        if latest["Close"] >= latest["volatility_bbh"]
        else "Below lower band"
        if latest["Close"] <= latest["volatility_bbl"]
        else "Inside bands"
    )
    volume_signal = "Accumulation" if latest["volume_cmf"] >= 0.0 else "Distribution"
    rows = [
        {"area": "Momentum", "indicator": "RSI", "value": latest["momentum_rsi"], "signal": rsi_signal},
        {"area": "Trend", "indicator": "MACD diff", "value": latest["trend_macd_diff"], "signal": macd_signal},
        {"area": "Trend", "indicator": "ADX", "value": latest["trend_adx"], "signal": trend_signal},
        {"area": "Volatility", "indicator": "ATR", "value": latest["volatility_atr"], "signal": "Range expansion" if latest["volatility_atr"] > ta_frame["volatility_atr"].tail(20).median() else "Range compression"},
        {"area": "Volatility", "indicator": "Bollinger position", "value": latest["volatility_bbp"], "signal": band_signal},
        {"area": "Volume", "indicator": "Chaikin Money Flow", "value": latest["volume_cmf"], "signal": volume_signal},
    ]
    return pd.DataFrame(rows)


def build_ta_price_chart(ta_frame: pd.DataFrame, lookback_days: int, *, log_y_axis: bool = False) -> go.Figure:
    display = ta_frame.tail(lookback_days)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=display.index, y=display["Close"], mode="lines", name="Close", line={"color": "#111827", "width": 3}))
    figure.add_trace(go.Scatter(x=display.index, y=display["trend_sma_fast"], mode="lines", name="SMA fast", line={"color": "#2563eb", "width": 2}))
    figure.add_trace(go.Scatter(x=display.index, y=display["trend_sma_slow"], mode="lines", name="SMA slow", line={"color": "#dc2626", "width": 2}))
    figure.add_trace(go.Scatter(x=display.index, y=display["volatility_bbh"], mode="lines", name="Bollinger high", line={"color": "rgba(5,150,105,0.0)"}, showlegend=False, hoverinfo="skip"))
    figure.add_trace(go.Scatter(x=display.index, y=display["volatility_bbl"], mode="lines", name="Bollinger band", line={"color": "rgba(5,150,105,0.0)"}, fill="tonexty", fillcolor="rgba(5,150,105,0.12)"))
    figure.update_layout(title="Price Structure", height=420)
    figure.update_yaxes(type="log" if log_y_axis else "linear")
    return figure


def build_ta_momentum_chart(ta_frame: pd.DataFrame, lookback_days: int) -> go.Figure:
    display = ta_frame.tail(lookback_days)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=display.index, y=display["momentum_rsi"], mode="lines", name="RSI", line={"color": "#7c3aed", "width": 2}))
    figure.add_trace(go.Scatter(x=display.index, y=display["momentum_stoch_rsi_k"] * 100.0, mode="lines", name="Stoch RSI %K", line={"color": "#0f766e", "width": 2}))
    figure.add_trace(go.Scatter(x=display.index, y=display["momentum_stoch_rsi_d"] * 100.0, mode="lines", name="Stoch RSI %D", line={"color": "#f59e0b", "width": 2}))
    figure.add_hline(y=70.0, line_dash="dash", line_color="#9ca3af")
    figure.add_hline(y=30.0, line_dash="dash", line_color="#9ca3af")
    figure.update_layout(title="Momentum", height=320, yaxis_range=[0, 100])
    return figure


def build_ta_macd_chart(ta_frame: pd.DataFrame, lookback_days: int) -> go.Figure:
    display = ta_frame.tail(lookback_days)
    colors = ["#059669" if value >= 0 else "#dc2626" for value in display["trend_macd_diff"]]
    figure = go.Figure()
    figure.add_trace(go.Bar(x=display.index, y=display["trend_macd_diff"], name="MACD hist", marker_color=colors, opacity=0.5))
    figure.add_trace(go.Scatter(x=display.index, y=display["trend_macd"], mode="lines", name="MACD", line={"color": "#2563eb", "width": 2}))
    figure.add_trace(go.Scatter(x=display.index, y=display["trend_macd_signal"], mode="lines", name="Signal", line={"color": "#111827", "width": 2}))
    figure.add_hline(y=0.0, line_dash="dash", line_color="#9ca3af")
    figure.update_layout(title="MACD", height=320)
    return figure


def build_ta_trend_strength_chart(ta_frame: pd.DataFrame, lookback_days: int) -> go.Figure:
    display = ta_frame.tail(lookback_days)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=display.index, y=display["trend_adx"], mode="lines", name="ADX", line={"color": "#111827", "width": 3}))
    figure.add_trace(go.Scatter(x=display.index, y=display["trend_adx_pos"], mode="lines", name="+DI", line={"color": "#059669", "width": 2}))
    figure.add_trace(go.Scatter(x=display.index, y=display["trend_adx_neg"], mode="lines", name="-DI", line={"color": "#dc2626", "width": 2}))
    figure.add_hline(y=25.0, line_dash="dash", line_color="#9ca3af")
    figure.update_layout(title="Trend Strength", height=320)
    return figure


def build_backtrader_equity_chart(result: dict) -> go.Figure:
    equity_curve = result["full"]["equity_curve"].copy()
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=equity_curve["date"], y=equity_curve["portfolio_value"], mode="lines", name="Strategy", line={"color": "#111827", "width": 3}))
    figure.add_trace(go.Scatter(x=equity_curve["date"], y=equity_curve["benchmark_value"], mode="lines", name="Benchmark", line={"color": "#2563eb", "width": 2}))
    figure.update_layout(title="Strategy Equity vs Benchmark", height=380)
    return figure


def build_backtrader_price_chart(result: dict, *, log_y_axis: bool = False) -> go.Figure:
    equity_curve = result["full"]["equity_curve"].copy()
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=equity_curve["date"], y=equity_curve["close"], mode="lines", name="Close", line={"color": "#111827", "width": 3}))
    figure.add_trace(go.Scatter(x=equity_curve["date"], y=equity_curve["ema_fast"], mode="lines", name="EMA fast", line={"color": "#059669", "width": 2}))
    figure.add_trace(go.Scatter(x=equity_curve["date"], y=equity_curve["ema_slow"], mode="lines", name="EMA slow", line={"color": "#dc2626", "width": 2}))
    entries = equity_curve.loc[equity_curve["signal"] == "enter"]
    exits = equity_curve.loc[equity_curve["signal"] == "exit"]
    if not entries.empty:
        figure.add_trace(go.Scatter(x=entries["date"], y=entries["close"], mode="markers", name="Entries", marker={"color": "#2563eb", "size": 9, "symbol": "triangle-up"}))
    if not exits.empty:
        figure.add_trace(go.Scatter(x=exits["date"], y=exits["close"], mode="markers", name="Exits", marker={"color": "#7c2d12", "size": 9, "symbol": "triangle-down"}))
    figure.update_layout(title="Price and Strategy Signals", height=380)
    figure.update_yaxes(type="log" if log_y_axis else "linear")
    return figure


def build_backtrader_optimization_chart(result: dict) -> go.Figure:
    optimization = result["optimization_results"].copy()
    figure = px.scatter(
        optimization,
        x="max_drawdown_pct",
        y="total_return_pct",
        color="score",
        hover_data=["fast_ema", "slow_ema", "rsi_entry", "adx_min", "trade_count", "sharpe_ratio"],
        title="Optimization Frontier",
    )
    figure.update_layout(height=360)
    return figure


def build_transformer_price_chart(result: dict, *, plot_start_date: pd.Timestamp | None = None, log_y_axis: bool = False) -> go.Figure:
    history = result["history"].copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    if plot_start_date is not None:
        history = history.loc[history["date"] >= plot_start_date].copy()

    def _xaxis_range(*date_series: pd.Series) -> list[str] | None:
        timestamps = []
        for values in date_series:
            converted = pd.to_datetime(values, errors="coerce")
            if not converted.empty:
                timestamps.append(converted.dropna())
        if not timestamps:
            return None
        combined = pd.concat(timestamps, ignore_index=True)
        if combined.empty:
            return None
        return [combined.min().isoformat(), combined.max().isoformat()]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["close"],
            mode="lines",
            name=result["summary"]["ticker"],
            line={"color": "#111827", "width": 3},
        )
    )

    historical_predictions = result["historical_predictions"].copy()
    if not historical_predictions.empty:
        historical_predictions["date"] = pd.to_datetime(historical_predictions["date"], errors="coerce")
        if plot_start_date is not None:
            historical_predictions = historical_predictions.loc[historical_predictions["date"] >= plot_start_date].copy()
        if not historical_predictions.empty:
            figure.add_trace(
                go.Scatter(
                    x=historical_predictions["date"],
                    y=historical_predictions["close"],
                    mode="markers",
                    name="Historical test probabilities",
                    marker={
                        "size": 8,
                        "color": historical_predictions["probability_outperform"],
                        "colorscale": "Viridis",
                        "colorbar": {"title": "P(outperform)"},
                    },
                    customdata=np.stack(
                        [
                            historical_predictions["probability_outperform"],
                            historical_predictions["stock_future_return_pct"],
                            historical_predictions["benchmark_future_return_pct"],
                        ],
                        axis=1,
                    ),
                    hovertemplate="Date=%{x}<br>Close=%{y:.2f}<br>P(outperform)=%{customdata[0]:.2%}<br>Stock fwd=%{customdata[1]:.2f}%<br>Benchmark fwd=%{customdata[2]:.2f}%<extra></extra>",
                )
            )

    actual_future = result["actual_future"].copy()
    if not actual_future.empty:
        actual_future["date"] = pd.to_datetime(actual_future["date"], errors="coerce")
        figure.add_trace(
            go.Scatter(
                x=actual_future["date"],
                y=actual_future["close"],
                mode="lines+markers",
                name="Post-cutoff realized price",
                line={"color": "#2563eb", "width": 2},
                marker={"size": 6},
            )
        )

    figure.add_vline(x=pd.Timestamp(result["summary"]["date"]), line_dash="dash", line_color="#6b7280")
    xaxis_range = _xaxis_range(
        history["date"],
        historical_predictions["date"] if "date" in historical_predictions.columns else pd.Series(dtype="datetime64[ns]"),
        actual_future["date"] if "date" in actual_future.columns else pd.Series(dtype="datetime64[ns]"),
        pd.Series([result["summary"]["date"]]),
    )
    figure.update_layout(
        title="Transformer Price Context",
        height=460,
        xaxis={
            "type": "date",
            "range": xaxis_range,
            "rangeslider": {"visible": True},
            "rangeselector": {
                "buttons": [
                    {"count": 3, "label": "3m", "step": "month", "stepmode": "backward"},
                    {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
                    {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
                    {"step": "all", "label": "All"},
                ]
            },
        },
    )
    figure.update_yaxes(type="log" if log_y_axis else "linear")
    return figure


def build_transformer_probability_chart(result: dict, *, plot_start_date: pd.Timestamp | None = None) -> go.Figure:
    predictions = result["historical_predictions"].copy()
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
    if plot_start_date is not None:
        predictions = predictions.loc[predictions["date"] >= plot_start_date].copy()
    figure = go.Figure()
    if not predictions.empty:
        figure.add_trace(
            go.Scatter(
                x=predictions["date"],
                y=predictions["probability_outperform"],
                mode="lines+markers",
                name="Historical probability",
                line={"color": "#2563eb", "width": 2},
                marker={"size": 5},
            )
        )
    cutoff = pd.Timestamp(result["summary"]["date"])
    figure.add_trace(
        go.Scatter(
            x=[cutoff],
            y=[result["summary"]["probability_of_outperformance"]],
            mode="markers",
            name="Current as-of probability",
            marker={"color": "#dc2626", "size": 12, "symbol": "diamond"},
        )
    )
    figure.add_hline(y=0.5, line_dash="dash", line_color="#6b7280")
    probability_x_range = None
    prediction_dates = predictions["date"].dropna()
    if not prediction_dates.empty:
        probability_x_range = [min(prediction_dates.min(), cutoff).isoformat(), max(prediction_dates.max(), cutoff).isoformat()]
    else:
        probability_x_range = [cutoff.isoformat(), cutoff.isoformat()]
    figure.update_layout(
        title="Probability of Relative Outperformance",
        height=320,
        yaxis_range=[0, 1],
        xaxis={"type": "date", "range": probability_x_range},
    )
    return figure


def render_transformer_tab() -> None:
    st.title("Transformer Lab")
    st.caption(
        "Transformer-based probability model for stock outperformance versus a benchmark, with ticker-specific saved models and date-aware retraining."
    )

    default_tickers = ["NVDA", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "TSLA", "AVGO", "SPY", "QQQ"]
    benchmark_options = ["SPY", "QQQ", "XLK", "SOXX"]
    refresh_token_key = "transformer_market_data_refresh_token"
    if refresh_token_key not in st.session_state:
        st.session_state[refresh_token_key] = 0

    col1, col2, col3, col4, col5 = st.columns([1.2, 1.0, 1.0, 1.0, 0.8])
    with col1:
        ticker = st.selectbox("Ticker", default_tickers, index=0, key="transformer_ticker")
    with col2:
        benchmark = st.selectbox("Benchmark", benchmark_options, index=0, key="transformer_benchmark")
    with col3:
        forecast_horizon_days = st.slider("Forecast horizon days", min_value=1, max_value=20, value=5, key="transformer_horizon")
    with col4:
        sequence_length = st.slider("Sequence length", min_value=16, max_value=96, value=32, step=8, key="transformer_sequence_length")
    with col5:
        refresh_prices = st.button("Refresh prices", key="transformer_refresh_prices")
    if refresh_prices:
        st.session_state[refresh_token_key] += 1

    market_prices = load_cached_forecast_market_data(
        ticker=ticker,
        prediction_start_date=str(pd.Timestamp.today().date()),
        refresh_token=int(st.session_state[refresh_token_key]),
    )
    available_dates = market_prices.index
    default_date = available_dates[-1].date()

    col6, col7, col8, col9 = st.columns([1.2, 1.0, 1.0, 1.0])
    with col6:
        analysis_end_date = st.date_input(
            "Analysis end date",
            value=default_date,
            min_value=available_dates.min().date(),
            max_value=available_dates.max().date(),
            key="transformer_analysis_end_date",
        )
    with col7:
        plot_lookback_days = st.slider("Plot lookback days", min_value=60, max_value=1260, value=252, step=21, key="transformer_plot_lookback")
    with col8:
        retrain_model = st.checkbox("Retrain model", value=False, key="transformer_retrain_model")
    with col9:
        transformer_log_y_axis = st.checkbox("Log price axis", value=False, key="transformer_log_price_axis")

    st.caption(
        "The saved transformer model is stored per ticker and benchmark. If the selected analysis date differs from the saved model cutoff, retrain is required to avoid leakage and mismatch."
    )
    trigger = st.button("Run transformer analysis", type="primary", key="run_transformer_analysis")
    if not trigger:
        st.info("Choose a ticker, benchmark, and date, then run the transformer analysis.")
        return

    resolved_analysis_end = resolve_prediction_start_date(market_prices.index, analysis_end_date)
    if resolved_analysis_end.date() != analysis_end_date:
        st.info(
            f"{analysis_end_date} is not a trading day in the downloaded data. Using {resolved_analysis_end.date()} instead."
        )

    progress_bar = st.progress(0.0)
    status = st.empty()

    def progress_callback(step: int, total: int, message: str) -> None:
        progress_bar.progress(step / max(total, 1))
        status.write(message)

    try:
        if retrain_model:
            artifacts = train_transformer_artifacts(
                ticker=ticker,
                benchmark=benchmark,
                prices=market_prices,
                analysis_end_date=str(resolved_analysis_end.date()),
                forecast_horizon_days=int(forecast_horizon_days),
                sequence_length=int(sequence_length),
                progress_callback=progress_callback,
            )
            progress_bar.progress(1.0)
            status.success(f"Transformer model trained and saved for {ticker} vs {benchmark}.")
        else:
            progress_bar.progress(0.25)
            status.write(f"Loading saved transformer model for {ticker} vs {benchmark}...")
            artifacts = load_transformer_artifacts(ticker, benchmark)
            if artifacts.forecast_horizon_days != int(forecast_horizon_days) or artifacts.sequence_length != int(sequence_length):
                raise ValueError(
                    "Saved transformer model was trained with different horizon or sequence settings. Enable retrain to use the selected configuration."
                )
        result = generate_transformer_analysis(
            ticker=ticker,
            benchmark=benchmark,
            prices=market_prices,
            analysis_end_date=str(resolved_analysis_end.date()),
            artifacts=artifacts,
        )
        progress_bar.progress(1.0)
        status.success(f"Transformer analysis ready for {ticker} vs {benchmark}.")
    except FileNotFoundError as exc:
        status.error(str(exc))
        st.error(str(exc))
        return
    except ImportError:
        st.error("PyTorch is not installed in the project environment. Install dependencies and rerun the dashboard.")
        return
    except Exception as exc:
        status.error(str(exc))
        st.error(str(exc))
        return

    plot_start_date = pd.Timestamp(result["summary"]["date"]) - pd.Timedelta(days=int(plot_lookback_days))
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    metric_col1.metric("As-of date", str(pd.Timestamp(result["summary"]["date"]).date()))
    metric_col2.metric("Latest close", f"{result['summary']['latest_close']:.2f}")
    metric_col3.metric("P(outperform)", f"{result['summary']['probability_of_outperformance']:.2%}")
    metric_col4.metric("Trade view", str(result["summary"]["trade_view"]).replace("_", " ").title())
    metric_col5.metric("Model file", Path(result["summary"]["model_path"]).name)

    top_left, top_right = st.columns([2.0, 1.2])
    top_left.plotly_chart(
        build_transformer_price_chart(result, plot_start_date=plot_start_date, log_y_axis=transformer_log_y_axis),
        use_container_width=True,
    )
    top_right.plotly_chart(
        build_transformer_probability_chart(result, plot_start_date=plot_start_date),
        use_container_width=True,
    )

    actual_future = result["actual_future"].copy()
    if not actual_future.empty:
        st.subheader("Post-cutoff Realized Window")
        display_future = actual_future.copy()
        display_future["stock_return_pct"] = display_future["stock_return_pct"] * 100.0
        display_future["benchmark_return_pct"] = display_future["benchmark_return_pct"] * 100.0
        display_future["excess_return_pct"] = display_future["excess_return_pct"] * 100.0
        st.dataframe(
            display_future.style.format(
                {
                    "close": "{:.2f}",
                    "benchmark_close": "{:.2f}",
                    "stock_return_pct": "{:.2f}%",
                    "benchmark_return_pct": "{:.2f}%",
                    "excess_return_pct": "{:.2f}%",
                }
            ),
            use_container_width=True,
        )
    else:
        st.info("No realized future window exists yet after the selected analysis date.")

    st.subheader("Validation Metrics")
    metrics_rows = []
    for split_name, values in result["validation_metrics"].items():
        metrics_rows.append({"split": split_name, **values})
    st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)

    st.subheader("Training History")
    training_history = result["training_history"].copy()
    if not training_history.empty:
        history_chart = px.line(training_history, x="epoch", y=["train_loss", "valid_loss"], title="Transformer Training Loss")
        history_chart.update_layout(height=320)
        st.plotly_chart(history_chart, use_container_width=True)
        st.dataframe(training_history, use_container_width=True)

    st.subheader("Historical Test Predictions")
    historical_predictions = result["historical_predictions"].copy()
    if not historical_predictions.empty:
        st.dataframe(
            historical_predictions.tail(50).style.format(
                {
                    "close": "{:.2f}",
                    "benchmark_close": "{:.2f}",
                    "probability_outperform": "{:.2%}",
                    "stock_future_return_pct": "{:.2f}%",
                    "benchmark_future_return_pct": "{:.2f}%",
                }
            ),
            use_container_width=True,
        )


def render_backtrader_tab() -> None:
    st.title("Backtrader Lab")
    st.caption(
        "Advanced backtrader2 analysis with parameter search, in-sample optimization, out-of-sample validation, ATR risk sizing, and trade-level analyzers."
    )

    default_tickers = ["NVDA", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "TSLA", "AVGO", "SPY", "QQQ"]
    refresh_token_key = "backtrader_market_data_refresh_token"
    if refresh_token_key not in st.session_state:
        st.session_state[refresh_token_key] = 0

    col1, col2, col3, col4 = st.columns([1.1, 1.1, 1.0, 0.8])
    with col1:
        ticker = st.selectbox("Ticker", default_tickers, index=0, key="backtrader_ticker")
    with col2:
        optimization_mode = st.selectbox("Optimization mode", ["focused", "balanced", "thorough"], index=1, key="backtrader_optimization_mode")
    with col3:
        log_y_axis = st.checkbox("Log price axis", value=False, key="backtrader_log_y_axis")
    with col4:
        refresh_prices = st.button("Refresh prices", key="backtrader_refresh_prices")
    if refresh_prices:
        st.session_state[refresh_token_key] += 1

    col5, col6, col7, col8, col9 = st.columns(5)
    with col5:
        lookback_days = st.slider("Lookback days", min_value=180, max_value=1260, value=504, step=21, key="backtrader_lookback")
    with col6:
        train_ratio = st.slider("Train split", min_value=0.55, max_value=0.85, value=0.70, step=0.05, key="backtrader_train_ratio")
    with col7:
        initial_cash = st.number_input("Initial cash", min_value=10000.0, max_value=1000000.0, value=100000.0, step=5000.0, key="backtrader_initial_cash")
    with col8:
        commission_bps = st.number_input("Commission bps", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key="backtrader_commission_bps")
    with col9:
        slippage_bps = st.number_input("Slippage bps", min_value=0.0, max_value=100.0, value=5.0, step=1.0, key="backtrader_slippage_bps")

    st.markdown(
        """
        **What this tab does**

        1. Builds a multi-factor trend-following strategy with ATR risk sizing and trailing stops.
        2. Optimizes parameters on the training window.
        3. Re-runs the best configuration on out-of-sample history and on the full dataset.
        4. Surfaces analyzers, trades, drawdown, and the optimization frontier.
        """
    )

    trigger = st.button("Run advanced backtrader analysis", type="primary", key="run_backtrader_analysis")
    if not trigger:
        st.info("Choose a stock and run the backtrader analysis.")
        return

    try:
        market_prices = load_cached_forecast_market_data(
            ticker=ticker,
            prediction_start_date=str(pd.Timestamp.today().date()),
            refresh_token=int(st.session_state[refresh_token_key]),
        )
    except Exception as exc:
        st.error(str(exc))
        return

    progress_bar = st.progress(0.0)
    status = st.empty()

    def progress_callback(step: int, total: int, message: str) -> None:
        progress_bar.progress(step / max(total, 1))
        status.write(message)

    try:
        result = run_advanced_backtrader_analysis(
            market_prices,
            ticker=ticker,
            lookback_days=int(lookback_days),
            train_ratio=float(train_ratio),
            optimization_mode=str(optimization_mode),
            initial_cash=float(initial_cash),
            commission_rate=float(commission_bps) / 10000.0,
            slippage_rate=float(slippage_bps) / 10000.0,
            progress_callback=progress_callback,
        )
        progress_bar.progress(1.0)
        status.success(f"Advanced backtrader analysis complete for {ticker}.")
    except ImportError:
        st.error("The `backtrader2` package is not installed in the project environment. Install dependencies and rerun the dashboard.")
        return
    except Exception as exc:
        st.error(str(exc))
        return

    metrics_table = result["metrics_table"].copy()
    full_metrics = metrics_table.loc[metrics_table["sample"] == "Full"].iloc[0]
    test_metrics = metrics_table.loc[metrics_table["sample"] == "Test"].iloc[0]
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5, metric_col6 = st.columns(6)
    metric_col1.metric("Split date", result["split_date"])
    metric_col2.metric("Full return", f"{full_metrics['total_return_pct']:.2f}%")
    metric_col3.metric("Test excess return", f"{test_metrics['excess_return_pct']:.2f}%")
    metric_col4.metric("Full max drawdown", f"{full_metrics['max_drawdown_pct']:.2f}%")
    metric_col5.metric("Test Sharpe", f"{test_metrics['sharpe_ratio']:.2f}" if pd.notna(test_metrics["sharpe_ratio"]) else "n/a")
    metric_col6.metric("Full trades", f"{int(full_metrics['trade_count'])}")

    left, right = st.columns(2)
    left.plotly_chart(build_backtrader_equity_chart(result), use_container_width=True)
    right.plotly_chart(build_backtrader_price_chart(result, log_y_axis=log_y_axis), use_container_width=True)

    st.plotly_chart(build_backtrader_optimization_chart(result), use_container_width=True)

    st.subheader("Best Parameters")
    st.dataframe(pd.DataFrame([result["best_params"]]), use_container_width=True)

    st.subheader("Sample Metrics")
    st.dataframe(
        metrics_table.style.format(
            {
                "final_value": "{:.2f}",
                "total_return_pct": "{:.2f}%",
                "annual_return_pct": "{:.2f}%",
                "benchmark_return_pct": "{:.2f}%",
                "excess_return_pct": "{:.2f}%",
                "max_drawdown_pct": "{:.2f}%",
                "win_rate": "{:.2%}",
                "profit_factor": "{:.2f}",
                "sharpe_ratio": "{:.2f}",
                "sqn": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Optimization Leaderboard")
    st.dataframe(
        result["optimization_results"].head(30).style.format(
            {
                "score": "{:.4f}",
                "total_return_pct": "{:.2f}%",
                "annual_return_pct": "{:.2f}%",
                "benchmark_return_pct": "{:.2f}%",
                "max_drawdown_pct": "{:.2f}%",
                "win_rate": "{:.2%}",
                "profit_factor": "{:.2f}",
                "sharpe_ratio": "{:.2f}",
                "sqn": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Trade Log")
    st.dataframe(result["full"]["trade_log"], use_container_width=True)

    st.subheader("Signal History")
    st.dataframe(result["full"]["signals"].tail(40), use_container_width=True)


def render_ta_tab() -> None:
    st.title("TA Library")
    st.caption("Technical-analysis indicators powered by the `ta` Python library on daily OHLCV data for a selected stock.")

    default_tickers = ["NVDA", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "TSLA", "AVGO", "SPY", "QQQ"]
    refresh_token_key = "ta_market_data_refresh_token"
    if refresh_token_key not in st.session_state:
        st.session_state[refresh_token_key] = 0

    control_col1, control_col2, control_col3, control_col4 = st.columns([1.2, 1, 1, 0.8])
    with control_col1:
        ticker = st.selectbox("Ticker", default_tickers, index=0, key="ta_ticker")
    with control_col2:
        lookback_days = st.slider("Lookback days", min_value=60, max_value=504, value=180, step=10, key="ta_lookback_days")
    with control_col3:
        ta_log_y_axis = st.checkbox("Log price axis", value=False, key="ta_log_y_axis")
    with control_col4:
        refresh_prices = st.button("Refresh prices", key="ta_refresh_prices")
    if refresh_prices:
        st.session_state[refresh_token_key] += 1

    trigger = st.button("Run TA analysis", type="primary", key="run_ta_analysis")
    if not trigger:
        st.info("Choose a stock and run the TA analysis.")
        return

    try:
        market_prices = load_cached_forecast_market_data(
            ticker=ticker,
            prediction_start_date=str(pd.Timestamp.today().date()),
            refresh_token=int(st.session_state[refresh_token_key]),
        )
        ta_frame = build_ta_indicator_frame(market_prices, ticker)
    except ImportError:
        st.error("The `ta` package is not installed in the project environment. Install dependencies and rerun the dashboard.")
        return
    except Exception as exc:
        st.error(str(exc))
        return

    latest = ta_frame.iloc[-1]
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5, metric_col6 = st.columns(6)
    metric_col1.metric("Close", f"{latest['Close']:.2f}")
    metric_col2.metric("RSI", f"{latest['momentum_rsi']:.1f}")
    metric_col3.metric("MACD diff", f"{latest['trend_macd_diff']:.3f}")
    metric_col4.metric("ADX", f"{latest['trend_adx']:.1f}")
    metric_col5.metric("ATR", f"{latest['volatility_atr']:.2f}")
    metric_col6.metric("CMF", f"{latest['volume_cmf']:.3f}")

    top_left, top_right = st.columns(2)
    top_left.plotly_chart(build_ta_price_chart(ta_frame, lookback_days, log_y_axis=ta_log_y_axis), use_container_width=True)
    top_right.plotly_chart(build_ta_momentum_chart(ta_frame, lookback_days), use_container_width=True)

    bottom_left, bottom_right = st.columns(2)
    bottom_left.plotly_chart(build_ta_macd_chart(ta_frame, lookback_days), use_container_width=True)
    bottom_right.plotly_chart(build_ta_trend_strength_chart(ta_frame, lookback_days), use_container_width=True)

    st.subheader("Latest Indicator Readout")
    st.dataframe(build_ta_signal_table(ta_frame).style.format({"value": "{:.4f}"}), use_container_width=True)

    st.subheader("Indicator Table")
    display_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "trend_sma_fast",
        "trend_sma_slow",
        "volatility_bbh",
        "volatility_bbl",
        "momentum_rsi",
        "momentum_stoch_rsi_k",
        "momentum_stoch_rsi_d",
        "trend_macd",
        "trend_macd_signal",
        "trend_macd_diff",
        "trend_adx",
        "trend_adx_pos",
        "trend_adx_neg",
        "volatility_atr",
        "volume_cmf",
        "volume_obv",
    ]
    st.dataframe(ta_frame[display_columns].tail(30), use_container_width=True)


def render_stochastic_tab() -> None:
    st.title("Stochastic Models")
    st.caption(
        "Probabilistic market-dynamics modeling with GBM for price paths and GARCH / EGARCH for volatility clustering and persistence."
    )

    default_tickers = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "META", "AMZN", "TSLA", "AVGO"]
    refresh_token_key = "stochastic_market_data_refresh_token"
    if refresh_token_key not in st.session_state:
        st.session_state[refresh_token_key] = 0

    control_col1, control_col2, control_col3, control_col4, control_col5, control_col6 = st.columns([1.2, 1, 1, 1, 0.8, 1])
    with control_col1:
        ticker = st.selectbox("Ticker", default_tickers, index=0, key="stochastic_ticker")
    with control_col2:
        horizon_days = st.slider("Forecast days", min_value=5, max_value=90, value=30, key="stochastic_horizon_days")
    with control_col3:
        simulation_paths = st.slider("GBM paths", min_value=200, max_value=5000, value=1500, step=100, key="stochastic_paths")
    with control_col4:
        random_seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1, key="stochastic_seed")
    with control_col5:
        refresh_prices = st.button("Refresh prices", key="stochastic_refresh_prices")
    with control_col6:
        stochastic_log_y_axis = st.checkbox("Log price axis", value=False, key="stochastic_log_y_axis")
    if refresh_prices:
        st.session_state[refresh_token_key] += 1

    market_prices = load_cached_forecast_market_data(
        ticker=ticker,
        prediction_start_date=str(pd.Timestamp.today().date()),
        refresh_token=int(st.session_state[refresh_token_key]),
    )
    available_dates = market_prices.index
    default_date = available_dates[-1].date()
    simulation_start_input = st.date_input(
        "Simulation start date",
        value=default_date,
        min_value=available_dates.min().date(),
        max_value=available_dates.max().date(),
        key="stochastic_simulation_start_date",
    )
    resolved_simulation_start = resolve_prediction_start_date(market_prices.index, simulation_start_input)
    if resolved_simulation_start.date() != simulation_start_input:
        st.info(
            f"{simulation_start_input} is not a trading day in the downloaded data. "
            f"Using {resolved_simulation_start.date()} instead."
        )

    st.markdown(
        """
        **What this tab does**

        1. Fits a geometric Brownian motion baseline for forward price cones.
        2. Fits GARCH(1,1) and EGARCH(1,1) volatility processes.
        3. Shows probabilistic price ranges and forward volatility instead of a single point forecast.
        """
    )

    trigger = st.button("Run stochastic models", type="primary", key="run_stochastic_models")
    if not trigger:
        st.info("Choose a ticker and click Run stochastic models.")
        return

    workflow_progress = st.progress(0.0)
    workflow_status = st.empty()

    def progress_callback(step: int, total: int, message: str) -> None:
        workflow_progress.progress(step / max(total, 1))
        workflow_status.write(message)

    try:
        modeling_prices = market_prices.loc[market_prices.index <= resolved_simulation_start, [market_prices.columns[0]]].copy()
        result = run_stochastic_analysis(
            modeling_prices,
            horizon_days=horizon_days,
            num_paths=int(simulation_paths),
            seed=int(random_seed),
            progress_callback=progress_callback,
        )
        result["simulation_start_date"] = str(resolved_simulation_start.date())
        result["actual_future"] = build_stochastic_actual_future_series(market_prices, resolved_simulation_start, horizon_days)
        result["actual_future_plot"] = build_stochastic_actual_post_start_series(market_prices, resolved_simulation_start)
        workflow_progress.progress(1.0)
        workflow_status.success(f"Stochastic model run complete for {ticker} from {resolved_simulation_start.date()}.")
    except Exception as exc:
        workflow_status.error(str(exc))
        st.error(str(exc))
        return

    comparison = build_stochastic_comparison_table(result)
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Last price", f"{result['last_price']:.2f}")
    metric_col2.metric("GBM median terminal price", f"{comparison.loc[comparison['model'] == 'GBM', 'terminal_median_price'].iloc[0]:.2f}")
    metric_col3.metric("GARCH terminal vol", f"{result['garch'].volatility_forecast.iloc[-1]['annualized_volatility']:.2%}")
    metric_col4.metric("Current regime", str(result["regime"].current_regime).replace("_", " ").title())

    actual_future = result.get("actual_future")
    if isinstance(actual_future, pd.DataFrame) and not actual_future.empty:
        actual_col1, actual_col2, actual_col3 = st.columns(3)
        actual_col1.metric("Actual terminal price", f"{float(actual_future['price'].iloc[-1]):.2f}")
        actual_col2.metric("Observed days", f"{len(actual_future)} / {horizon_days}")
        actual_col3.metric(
            "Best terminal abs. error",
            f"{float(comparison['terminal_abs_error'].min()):.2f}",
        )
    else:
        st.info("No realized future prices exist yet after the selected simulation start date, so comparison is forecast-only.")

    left, right = st.columns([2.2, 1.2])
    left.plotly_chart(build_stochastic_price_chart(result, log_y_axis=stochastic_log_y_axis), use_container_width=True)
    right.plotly_chart(build_stochastic_volatility_chart(result), use_container_width=True)

    st.subheader("Model Comparison")
    st.dataframe(
        comparison.style.format(
            {
                "annualized_volatility": "{:.2%}",
                "terminal_median_price": "{:.2f}",
                "terminal_p10_price": "{:.2f}",
                "terminal_p90_price": "{:.2f}",
                "actual_terminal_price": "{:.2f}",
                "terminal_error": "{:.2f}",
                "terminal_abs_error": "{:.2f}",
                "log_likelihood": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

    parameter_rows = [
        {"model": "GBM", **result["gbm"]["parameters"]},
        {"model": result["jump"]["model_name"], **result["jump"]["parameters"]},
        {"model": result["garch"].model_name, **result["garch"].parameters},
        {"model": result["egarch"].model_name, **result["egarch"].parameters},
        {"model": result["regime"].model_name, **result["regime"].parameters},
    ]
    st.subheader("Estimated Parameters")
    st.dataframe(pd.DataFrame(parameter_rows), use_container_width=True)

    st.subheader("Regime Diagnostics")
    regime_col1, regime_col2 = st.columns([1.2, 1.8])
    regime_col1.dataframe(result["regime"].state_summary.reset_index(drop=True), use_container_width=True)
    regime_col2.plotly_chart(build_stochastic_regime_probability_chart(result), use_container_width=True)

    realized_vol = result["log_returns"].rolling(20).std() * np.sqrt(252)
    realized_vol = realized_vol.dropna().rename("realized_20d_vol")
    if not realized_vol.empty:
        figure = go.Figure(
            go.Scatter(x=realized_vol.index, y=realized_vol.values, mode="lines", name="Realized 20d volatility", line={"color": "#1f2937", "width": 2})
        )
        figure.update_layout(title="Recent Realized Volatility", height=320, yaxis_tickformat=".1%")
        st.plotly_chart(figure, use_container_width=True)

    st.subheader("Rolling Historical Backtest")
    backtest_col1, backtest_col2 = st.columns(2)
    with backtest_col1:
        evaluation_windows = st.slider(
            "Evaluation windows",
            min_value=3,
            max_value=60,
            value=30,
            key="stochastic_backtest_windows",
            help="How many historical start dates to evaluate from the latest eligible history.",
        )
    with backtest_col2:
        step_days = st.slider(
            "Backtest step days",
            min_value=1,
            max_value=20,
            value=5,
            key="stochastic_backtest_step_days",
            help="Spacing between evaluated simulation start dates.",
        )

    run_backtest = st.button("Run rolling backtest", key="run_stochastic_backtest")
    if run_backtest:
        backtest_progress = st.progress(0.0)
        backtest_status = st.empty()

        def backtest_progress_callback(step: int, total: int, message: str) -> None:
            backtest_progress.progress(step / max(total, 1))
            backtest_status.write(message)

        try:
            backtest = run_stochastic_backtest(
                market_prices[[market_prices.columns[0]]],
                horizon_days=horizon_days,
                evaluation_windows=int(evaluation_windows),
                step_days=int(step_days),
                num_paths=int(simulation_paths),
                seed=int(random_seed),
                progress_callback=backtest_progress_callback,
            )
            backtest_progress.progress(1.0)
            backtest_status.success("Rolling stochastic backtest complete.")
        except Exception as exc:
            backtest_status.error(str(exc))
            st.error(str(exc))
        else:
            summary = backtest["summary"].copy()
            detail = backtest["detail"].copy()
            st.dataframe(
                summary.style.format(
                    {
                        "median_terminal_error": "{:.2f}",
                        "mean_terminal_abs_error": "{:.2f}",
                        "median_terminal_abs_error": "{:.2f}",
                        "rmse_terminal_error": "{:.2f}",
                        "coverage_10_90": "{:.2%}",
                        "upper_breach_rate": "{:.2%}",
                        "lower_breach_rate": "{:.2%}",
                        "mean_annualized_volatility": "{:.2%}",
                        "mean_realized_annualized_volatility": "{:.2%}",
                        "mean_volatility_forecast_error": "{:.2%}",
                        "median_volatility_forecast_error": "{:.2%}",
                        "mean_volatility_forecast_abs_error": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )
            st.plotly_chart(build_stochastic_backtest_error_chart(backtest), use_container_width=True)
            st.dataframe(
                detail.style.format(
                    {
                        "start_price": "{:.2f}",
                        "actual_terminal_price": "{:.2f}",
                        "terminal_median_price": "{:.2f}",
                        "terminal_error": "{:.2f}",
                        "terminal_abs_error": "{:.2f}",
                        "terminal_p10_price": "{:.2f}",
                        "terminal_p90_price": "{:.2f}",
                        "annualized_volatility": "{:.2%}",
                        "realized_annualized_volatility": "{:.2%}",
                        "volatility_forecast_error": "{:.2%}",
                        "volatility_forecast_abs_error": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )


def render_forecast_tab() -> None:
    st.title("Stock Forecast")
    st.caption(
        "Direct multi-horizon LightGBM forecasting with leak-safe training cutoff, confidence bands, and residual checks."
    )

    default_tickers = [
        "NVDA",
        "AAPL",
        "MSFT",
        "META",
        "GOOGL",
        "AMZN",
        "TSLA",
        "AVGO",
        "SPY",
        "QQQ",
    ]
    refresh_token_key = "forecast_market_data_refresh_token"
    if refresh_token_key not in st.session_state:
        st.session_state[refresh_token_key] = 0

    control_col1, control_col2, control_col3, control_col4, control_col5, control_col6 = st.columns([1.2, 1.2, 1, 1, 0.8, 1])
    with control_col1:
        ticker = st.selectbox("Ticker", default_tickers, index=0)
    with control_col2:
        growth_threshold_pct = st.number_input("Growth threshold %", min_value=-50.0, max_value=200.0, value=0.0, step=1.0)

    with control_col5:
        refresh_prices = st.button("Refresh prices")
    with control_col6:
        forecast_log_y_axis = st.checkbox("Log price axis", value=False, key="forecast_log_y_axis")
    if refresh_prices:
        st.session_state[refresh_token_key] += 1

    market_prices = load_cached_forecast_market_data(
        ticker=ticker,
        prediction_start_date=str(pd.Timestamp.today().date()),
        refresh_token=int(st.session_state[refresh_token_key]),
    )
    available_dates = market_prices.index
    default_date = available_dates[-1].date()
    with control_col3:
        prediction_start_date = st.date_input(
            "Prediction start date",
            value=default_date,
            min_value=available_dates.min().date(),
            max_value=available_dates.max().date(),
        )
    with control_col4:
        forecast_horizon_days = st.slider("Forecast horizon days", min_value=1, max_value=20, value=3)

    selected_weekday = pd.Timestamp(prediction_start_date).day_name()
    st.caption(f"Selected prediction date: {prediction_start_date} ({selected_weekday})")
    st.caption(
        "Historical prices are cached in Streamlit for this tab and only reload when the inputs change or you click Refresh prices. "
        "If you pick a weekend or market holiday, the app uses the latest available trading day before it."
    )
    st.caption(
        "Decision-first mode prioritizes 1-3 day horizons. Retraining is capped to 3 days by default, "
        "while older saved artifacts can still be used for longer forecasts."
    )

    retrain_model = st.checkbox("Retrain model", value=False)
    trigger = st.button("Run forecast", type="primary")

    if not trigger:
        st.info("Choose a ticker, date, and horizon, then run the forecast.")
        return

    threshold_return = growth_threshold_pct / 100.0
    effective_forecast_horizon_days = forecast_horizon_days
    resolved_prediction_start = resolve_prediction_start_date(market_prices.index, prediction_start_date)
    prediction_start_text = str(resolved_prediction_start.date())
    if resolved_prediction_start.date() != prediction_start_date:
        st.info(
            f"{prediction_start_date} ({selected_weekday}) is not present in the downloaded market data. "
            f"Using {prediction_start_text} ({resolved_prediction_start.day_name()}) instead."
        )
    workflow_progress = st.progress(0.0)
    workflow_status = st.empty()
    try:
        workflow_status.write(f"Using cached historical prices for {ticker}.")
        workflow_progress.progress(0.15)

        if retrain_model:
            if forecast_horizon_days > 3:
                effective_forecast_horizon_days = 3
                st.info(
                    "Retraining uses the decision-first 1-3 day setup. "
                    f"This run will train and forecast {effective_forecast_horizon_days} days instead of {forecast_horizon_days}."
                )

            def progress_callback(step: int, total: int, message: str) -> None:
                progress_start = 0.2
                progress_span = 0.65
                workflow_progress.progress(progress_start + progress_span * (step / max(total, 1)))
                workflow_status.write(message)

            artifacts = train_forecast_artifacts(
                ticker=ticker,
                prices=market_prices,
                prediction_start_date=prediction_start_text,
                max_horizon_days=effective_forecast_horizon_days,
                threshold_return=threshold_return,
                progress_callback=progress_callback,
            )
            workflow_status.write(f"Saving trained forecast model for {ticker}...")
            workflow_progress.progress(0.9)
            save_forecast_artifacts(artifacts)
        else:
            workflow_status.write(f"Loading saved forecast model for {ticker}...")
            workflow_progress.progress(0.4)
            artifacts = load_forecast_artifacts(ticker)
            if abs(float(artifacts.threshold_return) - threshold_return) > 1e-12:
                raise ValueError(
                    f"Saved model for {ticker} was trained with threshold {artifacts.threshold_return:.2%}. "
                    "Enable retrain to use a different growth threshold."
                )
            effective_forecast_horizon_days = min(effective_forecast_horizon_days, int(artifacts.max_horizon_days))
            workflow_progress.progress(0.6)

        workflow_status.write(f"Generating {effective_forecast_horizon_days}-day forecast for {ticker}...")
        workflow_progress.progress(0.95)
        result = generate_forecast(
            ticker=ticker,
            prices=market_prices,
            prediction_start_date=prediction_start_text,
            forecast_horizon_days=effective_forecast_horizon_days,
            artifacts=artifacts,
            allow_news_refresh=retrain_model,
        )
        workflow_progress.progress(1.0)
        workflow_status.success(f"Forecast ready for {ticker} using data through {prediction_start_text}.")
    except FileNotFoundError as exc:
        workflow_status.error(str(exc))
        st.error(str(exc))
        return
    except Exception as exc:
        workflow_status.error(str(exc))
        st.error(str(exc))
        return

    main_col, side_col = st.columns([3.4, 1.2])
    with main_col:
        st.plotly_chart(build_forecast_price_chart(result, log_y_axis=forecast_log_y_axis), use_container_width=True)
    with side_col:
        render_forecast_summary(result, growth_threshold_pct)

    residual_chart = build_residual_chart(result)
    if residual_chart is not None:
        st.plotly_chart(residual_chart, use_container_width=True)
    else:
        st.info("No actual future prices exist yet for the selected forecast window, so residuals are not available.")

    st.subheader("Forecast Table")
    display = result["forecast"].copy()
    display["predicted_return_pct"] = (np.exp(display["predicted_log_return"]) - 1.0) * 100.0
    st.dataframe(display, use_container_width=True)

    st.subheader("Training vs Unseen Validation Metrics")
    st.caption("Validation uses nested walk-forward splits before the selected forecast start date. The compact table below is the first place to judge if the model is tradable per horizon.")
    horizon_table = build_horizon_diagnostics_table(result)
    st.dataframe(horizon_table, use_container_width=True)
    metrics_table = build_metrics_table(result)
    st.dataframe(metrics_table, use_container_width=True)


def main() -> None:
    simulation_tab, forecast_tab, stochastic_tab, ta_tab, backtrader_tab, transformer_tab = st.tabs(["Simulation Runs", "Stock Forecast", "Stochastic Models", "TA Library", "Backtrader Lab", "Transformer Lab"])

    with simulation_tab:
        st.sidebar.title("Simulation Runs")
        run_dirs = list_runs(RUN_ROOT)
        if not run_dirs:
            st.title("Advanced Pipeline Paper Agent")
            st.info("No simulation runs found yet. Start one with scripts/advanced_pipeline/paper_agent.py.")
        else:
            selected_name = st.sidebar.selectbox("Run", [path.name for path in run_dirs])
            selected_run = next(path for path in run_dirs if path.name == selected_name)
            st.sidebar.caption("Dashboard refreshes every 10 seconds while the simulator writes new snapshots.")

            @st.fragment(run_every="10s")
            def live_fragment() -> None:
                render_dashboard(selected_run)

            live_fragment()

    with forecast_tab:
        render_forecast_tab()

    with stochastic_tab:
        render_stochastic_tab()

    with ta_tab:
        render_ta_tab()

    with backtrader_tab:
        render_backtrader_tab()

    with transformer_tab:
        render_transformer_tab()


if __name__ == "__main__":
    main()
