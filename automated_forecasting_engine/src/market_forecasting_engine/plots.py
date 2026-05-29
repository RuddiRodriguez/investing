from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "market_forecasting_engine_matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from market_forecasting_engine.basing_points import magee_basing_point_history
from market_forecasting_engine.chapter_10_patterns import latest_chapter_10_patterns
from market_forecasting_engine.chapter_11_patterns import latest_chapter_11_patterns
from market_forecasting_engine.chapter_12_gaps import latest_chapter_12_gaps
from market_forecasting_engine.chapter_13_support_resistance import latest_chapter_13_support_resistance
from market_forecasting_engine.chapter_14_trendlines import latest_chapter_14_trendlines
from market_forecasting_engine.chapter_15_major_trendlines import latest_chapter_15_major_trendlines
from market_forecasting_engine.chapter_16_market_context import chapter_16_donchian_history
from market_forecasting_engine.chapter_9_patterns import latest_chapter_9_patterns
from market_forecasting_engine.data import normalize_price_frame
from market_forecasting_engine.features import build_feature_frame
from market_forecasting_engine.reversal_patterns import latest_reversal_patterns
from market_forecasting_engine.triangle_patterns import latest_triangle_patterns


def write_plot_artifacts(
    report: dict[str, Any],
    prices: pd.DataFrame,
    output_dir: str | Path,
    target_column: str = "close",
    chart_scale: str = "log",
) -> dict[str, str]:
    """Write validation and forecast plots for a completed forecast report."""

    output_path = Path(output_dir)
    plot_path = output_path / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)

    normalized_prices = normalize_price_frame(prices, target_column=target_column)
    artifacts: dict[str, str] = {}

    forecast_plot = plot_path / f"forecast_{report['ticker']}.png"
    _plot_forecast(report, normalized_prices[target_column], forecast_plot)
    artifacts["forecast_plot"] = str(forecast_plot)

    forecast_plotly = plot_path / f"forecast_{report['ticker']}.html"
    _plot_forecast_plotly(report, normalized_prices[target_column], forecast_plotly)
    artifacts["forecast_plotly"] = str(forecast_plotly)

    technical_chart = plot_path / f"technical_{report['ticker']}.png"
    _plot_technical_chart(report, normalized_prices, technical_chart, target_column=target_column, timeframe="daily", chart_scale=chart_scale)
    artifacts["technical_chart"] = str(technical_chart)

    technical_chart_plotly = plot_path / f"technical_{report['ticker']}.html"
    _plot_technical_chart_plotly(report, normalized_prices, technical_chart_plotly, target_column=target_column, timeframe="daily", chart_scale=chart_scale)
    artifacts["technical_chart_plotly"] = str(technical_chart_plotly)

    clean_signal_chart = plot_path / f"technical_clean_{report['ticker']}.png"
    _plot_clean_signal_chart(report, normalized_prices, clean_signal_chart, target_column=target_column)
    artifacts["technical_clean_chart"] = str(clean_signal_chart)

    clean_signal_chart_plotly = plot_path / f"technical_clean_{report['ticker']}.html"
    _plot_clean_signal_chart_plotly(report, normalized_prices, clean_signal_chart_plotly, target_column=target_column)
    artifacts["technical_clean_chart_plotly"] = str(clean_signal_chart_plotly)

    for timeframe in ("daily", "weekly", "monthly"):
        timeframe_prices = _timeframe_prices(normalized_prices, timeframe)
        timeframe_png = plot_path / f"technical_{report['ticker']}_{timeframe}.png"
        _plot_technical_chart(report, timeframe_prices, timeframe_png, target_column=target_column, timeframe=timeframe, chart_scale=chart_scale)
        artifacts[f"technical_{timeframe}_chart"] = str(timeframe_png)

        timeframe_html = plot_path / f"technical_{report['ticker']}_{timeframe}.html"
        _plot_technical_chart_plotly(report, timeframe_prices, timeframe_html, target_column=target_column, timeframe=timeframe, chart_scale=chart_scale)
        artifacts[f"technical_{timeframe}_chart_plotly"] = str(timeframe_html)

    diagnostics = report.get("diagnostics", {})
    validation_predictions = diagnostics.get("selected_validation_predictions", {})
    for horizon, records in validation_predictions.items():
        validation_plot = plot_path / f"validation_{report['ticker']}_{horizon}d.png"
        _plot_validation_predictions(report, int(horizon), records, validation_plot)
        artifacts[f"validation_plot_{horizon}d"] = str(validation_plot)

        validation_plotly = plot_path / f"validation_{report['ticker']}_{horizon}d.html"
        _plot_validation_predictions_plotly(report, int(horizon), records, validation_plotly)
        artifacts[f"validation_plotly_{horizon}d"] = str(validation_plotly)

    return artifacts


def write_forecast_log_plot_artifacts(
    forecast_log_path: str | Path,
    output_dir: str | Path | None = None,
    ticker: str | None = None,
) -> dict[str, str]:
    """Write plots from the compact append-only forecast log."""

    log_path = Path(forecast_log_path)
    if not log_path.exists():
        return {}
    frame = pd.read_csv(log_path)
    if frame.empty:
        return {}

    required_columns = {"ticker", "forecast_timestamp", "predicted_price", "lower_price", "upper_price", "horizon"}
    if not required_columns.issubset(frame.columns):
        return {}

    if ticker:
        frame = frame[frame["ticker"].astype(str).str.upper() == ticker.upper()]
    if frame.empty:
        return {}

    frame = frame.copy()
    frame["forecast_timestamp"] = pd.to_datetime(frame["forecast_timestamp"], errors="coerce")
    frame["as_of_timestamp"] = pd.to_datetime(frame.get("as_of_timestamp"), errors="coerce")
    for column in ("predicted_price", "lower_price", "upper_price", "current_price", "horizon"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["forecast_timestamp", "predicted_price", "horizon"])
    if frame.empty:
        return {}

    output_path = Path(output_dir) if output_dir is not None else log_path.parent
    plot_path = output_path / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    label = ticker.upper() if ticker else str(frame["ticker"].iloc[-1]).upper()

    png_path = plot_path / f"forecast_log_{label}.png"
    html_path = plot_path / f"forecast_log_{label}.html"
    _plot_forecast_log(frame, png_path, label)
    _plot_forecast_log_plotly(frame, html_path, label)
    return {
        "forecast_log_plot": str(png_path),
        "forecast_log_plotly": str(html_path),
    }


def write_daily_trade_plot_artifacts(
    report: dict[str, Any],
    prices: pd.DataFrame,
    output_dir: str | Path,
    target_column: str = "close",
) -> dict[str, str]:
    """Write same-session daily-trade charts."""

    output_path = Path(output_dir)
    plot_path = output_path / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    normalized_prices = normalize_price_frame(prices, target_column=target_column)
    ticker = str(report.get("ticker", "ticker")).upper()

    png_path = plot_path / f"daily_trade_{ticker}.png"
    html_path = plot_path / f"daily_trade_{ticker}.html"
    _plot_daily_trade(report, normalized_prices, png_path, target_column=target_column)
    _plot_daily_trade_plotly(report, normalized_prices, html_path, target_column=target_column)
    return {
        "daily_trade_plot": str(png_path),
        "daily_trade_plotly": str(html_path),
    }


def _plot_forecast(report: dict[str, Any], close: pd.Series, output_file: Path) -> None:
    history = close.tail(252)
    forecasts = sorted(report["forecasts"], key=lambda item: int(item["horizon_days"]))
    forecast_dates = pd.to_datetime([item["forecast_date"] for item in forecasts])
    predicted_prices = [float(item["predicted_price"]) for item in forecasts]
    lower_prices = [float(item["lower_price"]) for item in forecasts]
    upper_prices = [float(item["upper_price"]) for item in forecasts]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(history.index, history.values, color="#1f2937", linewidth=1.8, label="Actual close")
    ax.axvline(close.index[-1], color="#6b7280", linestyle="--", linewidth=1, label="Forecast start")
    ax.plot(
        [close.index[-1], *forecast_dates],
        [float(close.iloc[-1]), *predicted_prices],
        color="#2563eb",
        marker="o",
        linewidth=2.0,
        label="Forecast",
    )
    ax.fill_between(forecast_dates, lower_prices, upper_prices, color="#60a5fa", alpha=0.22, label="Confidence interval")
    ax.scatter(forecast_dates, predicted_prices, color="#1d4ed8", s=45, zorder=3)

    ax.set_title(f"{report['ticker']} Forecast Path")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def _daily_trade_plot_frame(prices: pd.DataFrame, target_column: str) -> pd.DataFrame:
    index = pd.DatetimeIndex(prices.index)
    latest_date = index[-1].date()
    session = prices[index.date == latest_date].copy()
    if len(session) < 2:
        session = prices.tail(120).copy()
    close = session[target_column].astype(float)
    high = session["high"].astype(float) if "high" in session.columns else close
    low = session["low"].astype(float) if "low" in session.columns else close
    volume = session["volume"].astype(float) if "volume" in session.columns else pd.Series(1.0, index=session.index)
    typical_price = (high + low + close) / 3.0
    session["plot_close"] = close
    session["plot_vwap"] = (typical_price * volume).cumsum() / volume.where(volume != 0.0).cumsum()
    return session


def _plot_daily_trade(report: dict[str, Any], prices: pd.DataFrame, output_file: Path, target_column: str) -> None:
    session = _daily_trade_plot_frame(prices, target_column)
    opening_range = report.get("opening_range", {})
    plan = report.get("trade_plan", {})
    forecasts = _daily_trade_forecast_frame(report)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(session.index, session["plot_close"], color="#111827", linewidth=1.8, label="Close")
    ax.plot(session.index, session["plot_vwap"], color="#2563eb", linewidth=1.4, label="VWAP")
    if opening_range.get("high") is not None:
        ax.axhline(float(opening_range["high"]), color="#16a34a", linestyle="--", linewidth=1, label="Opening range high")
    if opening_range.get("low") is not None:
        ax.axhline(float(opening_range["low"]), color="#dc2626", linestyle="--", linewidth=1, label="Opening range low")
    _add_trade_plan_levels(ax, plan)
    if not forecasts.empty:
        ax.plot(
            [session.index[-1], *forecasts["forecast_timestamp"]],
            [float(session["plot_close"].iloc[-1]), *forecasts["predicted_price"]],
            color="#f59e0b",
            marker="o",
            linewidth=1.8,
            label="Hourly forecast",
        )
        ax.fill_between(
            forecasts["forecast_timestamp"],
            forecasts["lower_price"],
            forecasts["upper_price"],
            color="#f59e0b",
            alpha=0.16,
            label="Forecast interval",
        )

    title_action = plan.get("action", "no_trade")
    ax.set_title(f"{report.get('ticker', '').upper()} Daily Trade Plan: {title_action}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def _add_trade_plan_levels(ax: plt.Axes, plan: dict[str, Any]) -> None:
    if plan.get("entry_reference") is not None:
        ax.axhline(float(plan["entry_reference"]), color="#7c3aed", linewidth=1.2, label="Entry reference")
    if plan.get("stop") is not None:
        ax.axhline(float(plan["stop"]), color="#b91c1c", linestyle="-.", linewidth=1.2, label="Stop")
    if plan.get("take_profit") is not None:
        ax.axhline(float(plan["take_profit"]), color="#15803d", linestyle="-.", linewidth=1.2, label="Take profit")


def _plot_daily_trade_plotly(report: dict[str, Any], prices: pd.DataFrame, output_file: Path, target_column: str) -> None:
    session = _daily_trade_plot_frame(prices, target_column)
    opening_range = report.get("opening_range", {})
    plan = report.get("trade_plan", {})
    forecasts = _daily_trade_forecast_frame(report)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=session.index,
            y=session["plot_close"],
            mode="lines",
            name="Close",
            line={"color": "#111827", "width": 2},
            hovertemplate="Time: %{x|%Y-%m-%d %H:%M}<br>Close: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=session.index,
            y=session["plot_vwap"],
            mode="lines",
            name="VWAP",
            line={"color": "#2563eb", "width": 2},
            hovertemplate="Time: %{x|%Y-%m-%d %H:%M}<br>VWAP: %{y:.2f}<extra></extra>",
        )
    )
    _add_plotly_horizontal_line(fig, opening_range.get("high"), "Opening range high", "#16a34a", "dash")
    _add_plotly_horizontal_line(fig, opening_range.get("low"), "Opening range low", "#dc2626", "dash")
    _add_plotly_horizontal_line(fig, plan.get("entry_reference"), "Entry reference", "#7c3aed", "solid")
    _add_plotly_horizontal_line(fig, plan.get("stop"), "Stop", "#b91c1c", "dashdot")
    _add_plotly_horizontal_line(fig, plan.get("take_profit"), "Take profit", "#15803d", "dashdot")
    if not forecasts.empty:
        fig.add_trace(
            go.Scatter(
                x=forecasts["forecast_timestamp"],
                y=forecasts["upper_price"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecasts["forecast_timestamp"],
                y=forecasts["lower_price"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(245, 158, 11, 0.16)",
                line={"width": 0},
                name="Forecast interval",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[session.index[-1], *forecasts["forecast_timestamp"]],
                y=[float(session["plot_close"].iloc[-1]), *forecasts["predicted_price"]],
                mode="lines+markers",
                name="Hourly forecast",
                line={"color": "#f59e0b", "width": 2},
                hovertemplate="Time: %{x|%Y-%m-%d %H:%M}<br>Forecast: %{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"{report.get('ticker', '').upper()} Daily Trade Plan: {plan.get('action', 'no_trade')}",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.write_html(output_file, include_plotlyjs=True, full_html=True)


def _add_plotly_horizontal_line(fig: go.Figure, value: object, name: str, color: str, dash: str) -> None:
    if value is None:
        return
    try:
        y = float(value)
    except (TypeError, ValueError):
        return
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=name,
            line={"color": color, "dash": dash, "width": 1.5},
            hoverinfo="skip",
        )
    )
    fig.add_hline(y=y, line_color=color, line_dash=dash, line_width=1.2)


def _daily_trade_forecast_frame(report: dict[str, Any]) -> pd.DataFrame:
    forecasts = report.get("forecasts", [])
    if not forecasts:
        return pd.DataFrame()
    frame = pd.DataFrame(forecasts)
    frame["forecast_timestamp"] = pd.to_datetime(frame["forecast_timestamp"], errors="coerce")
    for column in ("predicted_price", "lower_price", "upper_price"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["forecast_timestamp", "predicted_price", "lower_price", "upper_price"])


def _plot_forecast_log(frame: pd.DataFrame, output_file: Path, ticker: str) -> None:
    ordered = frame.sort_values(["horizon", "forecast_timestamp", "as_of_timestamp"])
    fig, ax = plt.subplots(figsize=(12, 6))
    for horizon, group in ordered.groupby("horizon"):
        label = f"{int(horizon)} bar horizon"
        group = group.sort_values("forecast_timestamp")
        ax.plot(
            group["forecast_timestamp"],
            group["predicted_price"],
            marker="o",
            linewidth=1.8,
            label=label,
        )
        if {"lower_price", "upper_price"}.issubset(group.columns):
            ax.fill_between(
                group["forecast_timestamp"],
                group["lower_price"],
                group["upper_price"],
                alpha=0.12,
            )

    latest_current = ordered.dropna(subset=["as_of_timestamp", "current_price"])
    if not latest_current.empty:
        current = latest_current.sort_values("as_of_timestamp").drop_duplicates("as_of_timestamp", keep="last")
        ax.scatter(
            current["as_of_timestamp"],
            current["current_price"],
            color="#111827",
            s=24,
            label="Observed at forecast time",
            zorder=4,
        )

    ax.set_title(f"{ticker} Forecast Log")
    ax.set_xlabel("Forecast timestamp")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def _plot_forecast_log_plotly(frame: pd.DataFrame, output_file: Path, ticker: str) -> None:
    ordered = frame.sort_values(["horizon", "forecast_timestamp", "as_of_timestamp"])
    fig = go.Figure()
    for horizon, group in ordered.groupby("horizon"):
        group = group.sort_values("forecast_timestamp")
        customdata = group[["as_of_timestamp", "lower_price", "upper_price", "expected_direction", "selected_model"]].to_numpy()
        fig.add_trace(
            go.Scatter(
                x=group["forecast_timestamp"],
                y=group["predicted_price"],
                customdata=customdata,
                mode="lines+markers",
                name=f"{int(horizon)} bar horizon",
                hovertemplate=(
                    "Forecast timestamp: %{x|%Y-%m-%d %H:%M}"
                    "<br>Predicted price: %{y:.2f}"
                    "<br>As of: %{customdata[0]|%Y-%m-%d %H:%M}"
                    "<br>Interval: %{customdata[1]:.2f} - %{customdata[2]:.2f}"
                    "<br>Direction: %{customdata[3]}"
                    "<br>Model: %{customdata[4]}"
                    "<extra></extra>"
                ),
            )
        )

    latest_current = ordered.dropna(subset=["as_of_timestamp", "current_price"])
    if not latest_current.empty:
        current = latest_current.sort_values("as_of_timestamp").drop_duplicates("as_of_timestamp", keep="last")
        fig.add_trace(
            go.Scatter(
                x=current["as_of_timestamp"],
                y=current["current_price"],
                mode="markers",
                name="Observed at forecast time",
                marker={"color": "#111827", "size": 7},
                hovertemplate="As of: %{x|%Y-%m-%d %H:%M}<br>Current price: %{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{ticker} Forecast Log",
        xaxis_title="Forecast timestamp",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.write_html(output_file, include_plotlyjs=True, full_html=True)


def _plot_forecast_plotly(report: dict[str, Any], close: pd.Series, output_file: Path) -> None:
    history = close.tail(252)
    forecasts = sorted(report["forecasts"], key=lambda item: int(item["horizon_days"]))
    forecast_dates = pd.to_datetime([item["forecast_date"] for item in forecasts])
    predicted_prices = [float(item["predicted_price"]) for item in forecasts]
    lower_prices = [float(item["lower_price"]) for item in forecasts]
    upper_prices = [float(item["upper_price"]) for item in forecasts]
    forecast_customdata = [
        [
            int(item["horizon_days"]),
            float(item["lower_price"]),
            float(item["upper_price"]),
            item["expected_direction"],
            item["selected_model"],
            float(item["directional_confidence"]),
        ]
        for item in forecasts
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines",
            name="Actual close",
            line={"color": "#1f2937", "width": 2},
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Actual close: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=upper_prices,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=lower_prices,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(96, 165, 250, 0.22)",
            line={"width": 0},
            name="Confidence interval",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Lower interval: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[close.index[-1], *forecast_dates],
            y=[float(close.iloc[-1]), *predicted_prices],
            mode="lines",
            name="Forecast path",
            line={"color": "#2563eb", "width": 2},
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Forecast path price: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=predicted_prices,
            customdata=forecast_customdata,
            mode="markers",
            name="Forecast points",
            marker={"color": "#1d4ed8", "size": 9},
            hovertemplate=(
                "Forecast date: %{x|%Y-%m-%d}"
                "<br>Horizon: %{customdata[0]} trading days"
                "<br>Predicted price: %{y:.2f}"
                "<br>Interval: %{customdata[1]:.2f} - %{customdata[2]:.2f}"
                "<br>Direction: %{customdata[3]}"
                "<br>Model: %{customdata[4]}"
                "<br>Confidence: %{customdata[5]:.1%}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=close.index[-1], line_width=1, line_dash="dash", line_color="#6b7280")
    fig.update_layout(
        title=f"{report['ticker']} Forecast Path",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.write_html(output_file, include_plotlyjs=True, full_html=True)


def _plot_technical_chart(
    report: dict[str, Any],
    prices: pd.DataFrame,
    output_file: Path,
    target_column: str,
    timeframe: str,
    chart_scale: str,
) -> None:
    features = build_feature_frame(prices, target_column=target_column)
    close = prices[target_column]
    volume = prices["volume"] if "volume" in prices.columns else pd.Series(index=prices.index, dtype=float)
    history = prices.tail(_history_length(timeframe))
    feature_history = features.reindex(history.index)
    close_history = close.reindex(history.index)
    volume_history = volume.reindex(history.index)
    sma_20 = close.rolling(20).mean().reindex(history.index)
    sma_50 = close.rolling(50).mean().reindex(history.index)
    support = feature_history.get("structure_support_63d")
    resistance = feature_history.get("structure_resistance_63d")
    basing_history = magee_basing_point_history(prices, target_column=target_column).reindex(history.index)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, height_ratios=[3.2, 1.0])
    price_ax, volume_ax = axes
    _plot_ohlc_bars(price_ax, history, target_column=target_column)
    price_ax.plot(close_history.index, close_history.values, color="#111827", linewidth=1.2, label="Close")
    price_ax.plot(sma_20.index, sma_20.values, color="#2563eb", linewidth=1.1, label="SMA 20")
    price_ax.plot(sma_50.index, sma_50.values, color="#7c3aed", linewidth=1.1, label="SMA 50")
    if support is not None:
        price_ax.plot(support.index, support.values, color="#16a34a", linewidth=1.0, linestyle="--", label="63d support")
    if resistance is not None:
        price_ax.plot(resistance.index, resistance.values, color="#dc2626", linewidth=1.0, linestyle="--", label="63d resistance")
    if "magee_variant_1_stop" in basing_history and basing_history["magee_variant_1_stop"].notna().any():
        price_ax.step(
            basing_history.index,
            basing_history["magee_variant_1_stop"],
            where="post",
            color="#0f766e",
            linewidth=1.1,
            linestyle="-.",
            label="Magee stop V1",
        )
    if "magee_variant_2_stop" in basing_history and basing_history["magee_variant_2_stop"].notna().any():
        price_ax.step(
            basing_history.index,
            basing_history["magee_variant_2_stop"],
            where="post",
            color="#9333ea",
            linewidth=1.1,
            linestyle="-.",
            label="Magee stop V2",
        )

    _scatter_signal(price_ax, feature_history, close_history, "structure_breakout_63d", "^", "#16a34a", "Breakout")
    _scatter_signal(price_ax, feature_history, close_history, "structure_breakdown_63d", "v", "#dc2626", "Breakdown")
    _scatter_signal(price_ax, feature_history, close_history, "structure_gap_up", "o", "#0891b2", "Gap up")
    _scatter_signal(price_ax, feature_history, close_history, "structure_gap_down", "o", "#ea580c", "Gap down")
    _scatter_value_signal(price_ax, basing_history, "magee_wave_low", "s", "#0f766e", "Magee wave low")
    _scatter_value_signal(price_ax, basing_history, "magee_wave_high", "D", "#9333ea", "Magee wave high")
    reversal_overlay = latest_reversal_patterns(prices, target_column=target_column, timeframe=timeframe)
    _plot_head_and_shoulders_overlay(price_ax, reversal_overlay.get("head_and_shoulders_top", {}))
    _plot_head_and_shoulders_overlay(price_ax, reversal_overlay.get("head_and_shoulders_bottom", {}))
    _plot_dormant_bottom_overlay(price_ax, reversal_overlay.get("dormant_bottom", {}))
    triangle_overlay = latest_triangle_patterns(prices, target_column=target_column, timeframe=timeframe)
    _plot_triangle_overlay(price_ax, triangle_overlay.get("preferred", {}))
    chapter_9_overlay = latest_chapter_9_patterns(prices, target_column=target_column, timeframe=timeframe)
    _plot_rectangle_overlay(price_ax, chapter_9_overlay.get("rectangle", {}))
    _plot_multi_top_bottom_overlay(price_ax, chapter_9_overlay.get("multi_top_bottom", {}))
    chapter_10_overlay = latest_chapter_10_patterns(prices, target_column=target_column, timeframe=timeframe)
    _plot_chapter_10_structural_overlay(price_ax, chapter_10_overlay.get("structural_preferred", {}))
    _plot_chapter_10_event_overlay(price_ax, chapter_10_overlay.get("short_term_events", {}).get("preferred", {}))
    chapter_11_overlay = latest_chapter_11_patterns(prices, target_column=target_column, timeframe=timeframe)
    _plot_chapter_11_continuation_overlay(price_ax, chapter_11_overlay.get("continuation_preferred", {}))
    _plot_chapter_11_hs_continuation_overlay(price_ax, chapter_11_overlay.get("head_and_shoulders_continuation", {}))
    chapter_12_overlay = latest_chapter_12_gaps(prices, target_column=target_column, timeframe=timeframe)
    _plot_chapter_12_gap_overlay(price_ax, chapter_12_overlay.get("recent_gaps", []), latest_date=chapter_12_overlay.get("end_date"))
    _plot_chapter_12_island_overlay(price_ax, chapter_12_overlay.get("island_reversal", {}))
    chapter_13_overlay = latest_chapter_13_support_resistance(prices, target_column=target_column, timeframe=timeframe)
    _plot_chapter_13_zone_overlay(price_ax, chapter_13_overlay)
    chapter_14_overlay = latest_chapter_14_trendlines(prices, target_column=target_column, timeframe=timeframe)
    _plot_chapter_14_trendline_overlay(price_ax, chapter_14_overlay)
    chapter_15_overlay = latest_chapter_15_major_trendlines(prices, target_column=target_column)
    _plot_chapter_15_major_trendline_overlay(price_ax, chapter_15_overlay)
    _plot_chapter_16_market_context_overlay(price_ax, prices, target_column=target_column)

    trend_state = report.get("technical_view", {}).get("trend_state", {}).get("state", "Trend")
    invalidation = _timeframe_level(report, timeframe, "support")
    if invalidation is not None:
        price_ax.axhline(float(invalidation), color="#16a34a", linestyle=":", linewidth=1.2, label="Invalidation")
    if chart_scale == "log" and (close_history.dropna() > 0).all():
        price_ax.set_yscale("log")
    price_ax.set_title(f"{report['ticker']} {timeframe.title()} Technical Chart | {trend_state} | {chart_scale} scale")
    price_ax.set_ylabel("Price")
    price_ax.grid(True, alpha=0.22)
    price_ax.legend(loc="best", ncols=3, fontsize=8)

    if volume_history.notna().any():
        volume_ax.bar(volume_history.index, volume_history.values, color="#9ca3af", width=1.0, label="Volume")
        volume_ax.plot(volume_history.index, volume_history.rolling(20).mean(), color="#374151", linewidth=1.0, label="Volume SMA 20")
    volume_ax.set_ylabel("Volume")
    volume_ax.grid(True, alpha=0.18)
    volume_ax.legend(loc="best", fontsize=8)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_technical_chart_plotly(
    report: dict[str, Any],
    prices: pd.DataFrame,
    output_file: Path,
    target_column: str,
    timeframe: str,
    chart_scale: str,
) -> None:
    features = build_feature_frame(prices, target_column=target_column)
    close = prices[target_column]
    volume = prices["volume"] if "volume" in prices.columns else pd.Series(index=prices.index, dtype=float)
    history = prices.tail(_history_length(timeframe))
    feature_history = features.reindex(history.index)
    close_history = close.reindex(history.index)
    volume_history = volume.reindex(history.index)
    sma_20 = close.rolling(20).mean().reindex(history.index)
    sma_50 = close.rolling(50).mean().reindex(history.index)
    basing_history = magee_basing_point_history(prices, target_column=target_column).reindex(history.index)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
        subplot_titles=("Market Action", "Volume"),
    )
    if all(column in history.columns for column in ("open", "high", "low", target_column)):
        fig.add_trace(
            go.Candlestick(
                x=history.index,
                open=history["open"],
                high=history["high"],
                low=history["low"],
                close=history[target_column],
                name="OHLC",
                increasing={"line": {"color": "#16a34a"}},
                decreasing={"line": {"color": "#dc2626"}},
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(x=close_history.index, y=close_history.values, mode="lines", name="Close", line={"color": "#111827", "width": 1.4}),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=sma_20.index, y=sma_20.values, mode="lines", name="SMA 20", line={"color": "#2563eb"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=sma_50.index, y=sma_50.values, mode="lines", name="SMA 50", line={"color": "#7c3aed"}), row=1, col=1)
    if "structure_support_63d" in feature_history:
        fig.add_trace(
            go.Scatter(x=feature_history.index, y=feature_history["structure_support_63d"], mode="lines", name="63d support", line={"color": "#16a34a", "dash": "dash"}),
            row=1,
            col=1,
        )
    if "structure_resistance_63d" in feature_history:
        fig.add_trace(
            go.Scatter(x=feature_history.index, y=feature_history["structure_resistance_63d"], mode="lines", name="63d resistance", line={"color": "#dc2626", "dash": "dash"}),
            row=1,
            col=1,
        )
    if "magee_variant_1_stop" in basing_history and basing_history["magee_variant_1_stop"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=basing_history.index,
                y=basing_history["magee_variant_1_stop"],
                mode="lines",
                name="Magee stop V1",
                line={"color": "#0f766e", "dash": "dashdot"},
                line_shape="hv",
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Magee stop V1: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if "magee_variant_2_stop" in basing_history and basing_history["magee_variant_2_stop"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=basing_history.index,
                y=basing_history["magee_variant_2_stop"],
                mode="lines",
                name="Magee stop V2",
                line={"color": "#9333ea", "dash": "dashdot"},
                line_shape="hv",
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Magee stop V2: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    _add_plotly_signal(fig, feature_history, close_history, "structure_breakout_63d", "triangle-up", "#16a34a", "Breakout")
    _add_plotly_signal(fig, feature_history, close_history, "structure_breakdown_63d", "triangle-down", "#dc2626", "Breakdown")
    _add_plotly_signal(fig, feature_history, close_history, "structure_gap_up", "circle", "#0891b2", "Gap up")
    _add_plotly_signal(fig, feature_history, close_history, "structure_gap_down", "circle", "#ea580c", "Gap down")
    _add_plotly_value_signal(fig, basing_history, "magee_wave_low", "square", "#0f766e", "Magee wave low")
    _add_plotly_value_signal(fig, basing_history, "magee_wave_high", "diamond", "#9333ea", "Magee wave high")
    reversal_overlay = latest_reversal_patterns(prices, target_column=target_column, timeframe=timeframe)
    _add_plotly_head_and_shoulders_overlay(fig, reversal_overlay.get("head_and_shoulders_top", {}))
    _add_plotly_head_and_shoulders_overlay(fig, reversal_overlay.get("head_and_shoulders_bottom", {}))
    _add_plotly_dormant_bottom_overlay(fig, reversal_overlay.get("dormant_bottom", {}))
    triangle_overlay = latest_triangle_patterns(prices, target_column=target_column, timeframe=timeframe)
    _add_plotly_triangle_overlay(fig, triangle_overlay.get("preferred", {}))
    chapter_9_overlay = latest_chapter_9_patterns(prices, target_column=target_column, timeframe=timeframe)
    _add_plotly_rectangle_overlay(fig, chapter_9_overlay.get("rectangle", {}))
    _add_plotly_multi_top_bottom_overlay(fig, chapter_9_overlay.get("multi_top_bottom", {}))
    chapter_10_overlay = latest_chapter_10_patterns(prices, target_column=target_column, timeframe=timeframe)
    _add_plotly_chapter_10_structural_overlay(fig, chapter_10_overlay.get("structural_preferred", {}))
    _add_plotly_chapter_10_event_overlay(fig, chapter_10_overlay.get("short_term_events", {}).get("preferred", {}))
    chapter_11_overlay = latest_chapter_11_patterns(prices, target_column=target_column, timeframe=timeframe)
    _add_plotly_chapter_11_continuation_overlay(fig, chapter_11_overlay.get("continuation_preferred", {}))
    _add_plotly_chapter_11_hs_continuation_overlay(fig, chapter_11_overlay.get("head_and_shoulders_continuation", {}))
    chapter_12_overlay = latest_chapter_12_gaps(prices, target_column=target_column, timeframe=timeframe)
    _add_plotly_chapter_12_gap_overlay(fig, chapter_12_overlay.get("recent_gaps", []), latest_date=chapter_12_overlay.get("end_date"))
    _add_plotly_chapter_12_island_overlay(fig, chapter_12_overlay.get("island_reversal", {}))
    chapter_13_overlay = latest_chapter_13_support_resistance(prices, target_column=target_column, timeframe=timeframe)
    _add_plotly_chapter_13_zone_overlay(fig, chapter_13_overlay)
    chapter_14_overlay = latest_chapter_14_trendlines(prices, target_column=target_column, timeframe=timeframe)
    _add_plotly_chapter_14_trendline_overlay(fig, chapter_14_overlay)
    chapter_15_overlay = latest_chapter_15_major_trendlines(prices, target_column=target_column)
    _add_plotly_chapter_15_major_trendline_overlay(fig, chapter_15_overlay)
    _add_plotly_chapter_16_market_context_overlay(fig, prices, target_column=target_column)
    if volume_history.notna().any():
        fig.add_trace(go.Bar(x=volume_history.index, y=volume_history.values, name="Volume", marker={"color": "#9ca3af"}), row=2, col=1)
        fig.add_trace(
            go.Scatter(x=volume_history.index, y=volume_history.rolling(20).mean(), mode="lines", name="Volume SMA 20", line={"color": "#374151"}),
            row=2,
            col=1,
        )
    trend_state = report.get("technical_view", {}).get("trend_state", {}).get("state", "Trend")
    invalidation = _timeframe_level(report, timeframe, "support")
    if invalidation is not None:
        fig.add_hline(y=float(invalidation), line_width=1, line_dash="dot", line_color="#16a34a", row=1, col=1)
    fig.update_layout(
        title=f"{report['ticker']} {timeframe.title()} Technical Chart | {trend_state} | {chart_scale} scale",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    if chart_scale == "log" and (close_history.dropna() > 0).all():
        fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.write_html(output_file, include_plotlyjs=True, full_html=True)


def _plot_clean_signal_chart(
    report: dict[str, Any],
    prices: pd.DataFrame,
    output_file: Path,
    target_column: str,
) -> None:
    features = build_feature_frame(prices, target_column=target_column)
    close = prices[target_column]
    volume = prices["volume"] if "volume" in prices.columns else pd.Series(index=prices.index, dtype=float)
    history = prices.tail(252)
    close_history = close.reindex(history.index)
    volume_history = volume.reindex(history.index)
    sma_20 = close.rolling(20).mean().reindex(history.index)
    sma_50 = close.rolling(50).mean().reindex(history.index)
    donchian = chapter_16_donchian_history(prices, target_column=target_column).reindex(history.index)
    signals = _clean_ranked_signals(report, prices, features, target_column=target_column)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8.6), sharex=False, height_ratios=[3.2, 0.85, 1.35])
    price_ax, volume_ax, table_ax = axes
    _plot_ohlc_bars(price_ax, history, target_column=target_column)
    price_ax.plot(close_history.index, close_history.values, color="#111827", linewidth=1.35, label="Close")
    price_ax.plot(sma_20.index, sma_20.values, color="#2563eb", linewidth=1.05, label="SMA 20")
    price_ax.plot(sma_50.index, sma_50.values, color="#7c3aed", linewidth=1.05, label="SMA 50")

    if "donchian_high_20" in donchian and donchian["donchian_high_20"].notna().any():
        price_ax.plot(
            donchian.index,
            donchian["donchian_high_20"],
            color="#475569",
            linewidth=0.95,
            linestyle=(0, (2, 2)),
            label="Donchian 20 high",
        )
    if "donchian_low_20" in donchian and donchian["donchian_low_20"].notna().any():
        price_ax.plot(
            donchian.index,
            donchian["donchian_low_20"],
            color="#475569",
            linewidth=0.95,
            linestyle=(0, (1, 3)),
            label="Donchian 20 low",
        )

    support = _timeframe_level(report, "daily", "support")
    resistance = _timeframe_level(report, "daily", "resistance")
    if support is not None:
        price_ax.axhline(support, color="#15803d", linewidth=1.0, linestyle=":", label="Daily support")
    if resistance is not None:
        price_ax.axhline(resistance, color="#b91c1c", linewidth=1.0, linestyle=":", label="Daily resistance")
    _plot_clean_signal_markers(price_ax, signals, history.index)

    trend_state = report.get("technical_view", {}).get("trend_state", {}).get("state", "Trend")
    price_ax.set_title(f"{report['ticker']} Clean Signal Chart | {trend_state} | Linear scale")
    price_ax.set_ylabel("Price")
    price_ax.grid(True, alpha=0.22)
    price_ax.legend(loc="best", ncols=3, fontsize=8)

    if volume_history.notna().any():
        volume_ax.bar(volume_history.index, volume_history.values, color="#cbd5e1", width=1.0, label="Volume")
        volume_ax.plot(volume_history.index, volume_history.rolling(20).mean(), color="#334155", linewidth=1.0, label="Volume SMA 20")
        volume_ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value / 1_000_000:.0f}M"))
    volume_ax.set_ylabel("Volume")
    volume_ax.grid(True, alpha=0.16)
    volume_ax.legend(loc="best", fontsize=8)
    volume_ax.set_xlim(history.index.min(), history.index.max())
    price_ax.set_xlim(history.index.min(), history.index.max())

    table_ax.axis("off")
    table_ax.text(
        0.01,
        0.96,
        _clean_signal_text(report, signals),
        transform=table_ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        color="#111827",
    )

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_clean_signal_chart_plotly(
    report: dict[str, Any],
    prices: pd.DataFrame,
    output_file: Path,
    target_column: str,
) -> None:
    features = build_feature_frame(prices, target_column=target_column)
    close = prices[target_column]
    volume = prices["volume"] if "volume" in prices.columns else pd.Series(index=prices.index, dtype=float)
    history = prices.tail(252)
    close_history = close.reindex(history.index)
    volume_history = volume.reindex(history.index)
    sma_20 = close.rolling(20).mean().reindex(history.index)
    sma_50 = close.rolling(50).mean().reindex(history.index)
    donchian = chapter_16_donchian_history(prices, target_column=target_column).reindex(history.index)
    signals = _clean_ranked_signals(report, prices, features, target_column=target_column)
    table_rows = signals[:7]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.62, 0.18, 0.20],
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
        subplot_titles=("Clean Market Action", "Volume", "Ranked Signals"),
    )
    if all(column in history.columns for column in ("open", "high", "low", target_column)):
        fig.add_trace(
            go.Candlestick(
                x=history.index,
                open=history["open"],
                high=history["high"],
                low=history["low"],
                close=history[target_column],
                name="OHLC",
                increasing={"line": {"color": "#16a34a"}},
                decreasing={"line": {"color": "#dc2626"}},
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=close_history.index,
            y=close_history.values,
            mode="lines",
            name="Close",
            line={"color": "#111827", "width": 1.6},
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=sma_20.index, y=sma_20.values, mode="lines", name="SMA 20", line={"color": "#2563eb", "width": 1.2}), row=1, col=1)
    fig.add_trace(go.Scatter(x=sma_50.index, y=sma_50.values, mode="lines", name="SMA 50", line={"color": "#7c3aed", "width": 1.2}), row=1, col=1)
    if "donchian_high_20" in donchian and donchian["donchian_high_20"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=donchian.index,
                y=donchian["donchian_high_20"],
                mode="lines",
                name="Donchian 20 high",
                line={"color": "#475569", "width": 1, "dash": "dot"},
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Donchian 20 high: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if "donchian_low_20" in donchian and donchian["donchian_low_20"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=donchian.index,
                y=donchian["donchian_low_20"],
                mode="lines",
                name="Donchian 20 low",
                line={"color": "#475569", "width": 1, "dash": "dash"},
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Donchian 20 low: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    support = _timeframe_level(report, "daily", "support")
    resistance = _timeframe_level(report, "daily", "resistance")
    if support is not None:
        fig.add_hline(y=support, line_width=1, line_dash="dot", line_color="#15803d", annotation_text="Daily support", row=1, col=1)
    if resistance is not None:
        fig.add_hline(y=resistance, line_width=1, line_dash="dot", line_color="#b91c1c", annotation_text="Daily resistance", row=1, col=1)
    _add_plotly_clean_signal_markers(fig, signals, history.index)

    if volume_history.notna().any():
        fig.add_trace(go.Bar(x=volume_history.index, y=volume_history.values, name="Volume", marker={"color": "#cbd5e1"}), row=2, col=1)
        fig.add_trace(
            go.Scatter(x=volume_history.index, y=volume_history.rolling(20).mean(), mode="lines", name="Volume SMA 20", line={"color": "#334155", "width": 1.1}),
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Table(
            header={
                "values": ["Rank", "Signal", "Direction", "Score", "Status"],
                "fill_color": "#e2e8f0",
                "align": "left",
                "font": {"color": "#0f172a", "size": 12},
            },
            cells={
                "values": [
                    [str(index + 1) for index, _ in enumerate(table_rows)],
                    [str(item.get("name", "")) for item in table_rows],
                    [str(item.get("direction", "")) for item in table_rows],
                    [f"{float(item.get('score', 0.0)):.2f}" for item in table_rows],
                    [str(item.get("status", "")) for item in table_rows],
                ],
                "fill_color": "#f8fafc",
                "align": "left",
                "font": {"color": "#111827", "size": 11},
                "height": 24,
            },
        ),
        row=3,
        col=1,
    )

    trend_state = report.get("technical_view", {}).get("trend_state", {}).get("state", "Trend")
    fig.update_layout(
        title=f"{report['ticker']} Clean Signal Chart | {trend_state} | Linear scale",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_yaxes(title_text="Price", type="linear", tickformat=",.2f", row=1, col=1)
    fig.update_yaxes(title_text="Volume", tickformat="~s", row=2, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.write_html(output_file, include_plotlyjs=True, full_html=True)


def _clean_ranked_signals(
    report: dict[str, Any],
    prices: pd.DataFrame,
    features: pd.DataFrame,
    target_column: str,
) -> list[dict[str, Any]]:
    close = prices[target_column]
    technical_view = report.get("technical_view", {})
    signals: list[dict[str, Any]] = []

    forecast = _preferred_forecast(report)
    if forecast:
        confidence = _safe_float(forecast.get("directional_confidence")) or 0.5
        horizon = forecast.get("horizon_days", "?")
        direction = _display_direction(forecast.get("expected_direction"))
        signals.append(
            {
                "name": "Model forecast",
                "direction": direction,
                "status": f"{horizon}d {forecast.get('expected_direction', 'Unknown')} forecast, {confidence:.0%} confidence",
                "score": max(0.0, min(confidence, 1.0)),
                "plot": False,
            }
        )

    trend_state = technical_view.get("trend_state", {})
    trend_label = str(trend_state.get("state", "Neutral"))
    trend_confidence = _safe_float(trend_state.get("confidence")) or 0.5
    signals.append(
        {
            "name": "Trend regime",
            "direction": _display_direction(trend_label),
            "status": trend_label,
            "score": min(0.95, max(0.55, trend_confidence)),
            "plot": False,
        }
    )

    ma_signal = _moving_average_signal(close)
    if ma_signal is not None:
        signals.append(ma_signal)

    breakout = _recent_binary_signal(features, close, "structure_breakout_63d", "63d breakout", "Bullish", lookback=40)
    breakdown = _recent_binary_signal(features, close, "structure_breakdown_63d", "63d breakdown", "Bearish", lookback=40)
    signals.extend(item for item in (breakout, breakdown) if item is not None)

    donchian_signal = _donchian_signal(technical_view)
    if donchian_signal is not None:
        signals.append(donchian_signal)

    support_resistance = _support_resistance_signal(report)
    if support_resistance is not None:
        signals.append(support_resistance)

    signals.extend(_evidence_matrix_signals(technical_view))
    fragility = technical_view.get("chapter_17_decision_fragility", {})
    if fragility:
        level = str(fragility.get("level", "Unknown"))
        score = _safe_float(fragility.get("score")) or 0.0
        signals.append(
            {
                "name": "Decision fragility",
                "direction": "Warning" if level in {"High", "Elevated"} else "Neutral",
                "status": f"{level} fragility ({score:.0%})",
                "score": min(0.85, max(0.45, score)),
                "plot": False,
            }
        )

    trend_direction = _display_direction(trend_label)
    ma_breakdown = _has_ma_breakdown(close)
    ma_breakout = _has_ma_breakout(close)
    adjusted = [_adjust_signal_for_regime(item, trend_direction, ma_breakdown=ma_breakdown, ma_breakout=ma_breakout) for item in signals]
    deduped = _dedupe_clean_signals(adjusted)
    return sorted(deduped, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:10]


def _preferred_forecast(report: dict[str, Any]) -> dict[str, Any]:
    forecasts = report.get("forecasts", [])
    if not forecasts:
        return {}
    return sorted(forecasts, key=lambda item: int(item.get("horizon_days", 10**9)))[0]


def _display_direction(value: object) -> str:
    text = str(value or "").lower()
    if "up" in text or "bull" in text or text in {"long", "longbreakout", "buy"}:
        return "Bullish"
    if "down" in text or "bear" in text or text in {"short", "shortbreakout", "sell"}:
        return "Bearish"
    if "warning" in text:
        return "Warning"
    return "Neutral"


def _moving_average_signal(close: pd.Series) -> dict[str, Any] | None:
    clean = close.dropna()
    if len(clean) < 50:
        return None
    latest_close = float(clean.iloc[-1])
    sma_20 = float(clean.rolling(20).mean().iloc[-1])
    sma_50 = float(clean.rolling(50).mean().iloc[-1])
    if pd.isna(sma_20) or pd.isna(sma_50):
        return None
    if latest_close > sma_20 > sma_50:
        direction = "Bullish"
        status = "Close above SMA 20 and SMA 50; SMA 20 above SMA 50"
        score = 0.86
    elif latest_close < sma_20 < sma_50:
        direction = "Bearish"
        status = "Close below SMA 20 and SMA 50; SMA 20 below SMA 50"
        score = 0.86
    elif latest_close > sma_50:
        direction = "Bullish"
        status = "Close above SMA 50, but moving-average stack is mixed"
        score = 0.66
    elif latest_close < sma_50:
        direction = "Bearish"
        status = "Close below SMA 50, but moving-average stack is mixed"
        score = 0.66
    else:
        direction = "Neutral"
        status = "Moving-average stack is mixed"
        score = 0.50
    return {"name": "MA 20/50 alignment", "direction": direction, "status": status, "score": score, "plot": False}


def _recent_binary_signal(
    features: pd.DataFrame,
    close: pd.Series,
    column: str,
    name: str,
    direction: str,
    lookback: int,
) -> dict[str, Any] | None:
    if column not in features:
        return None
    recent = features[column].tail(lookback).fillna(0.0) > 0
    if not recent.any():
        return None
    date = pd.Timestamp(recent[recent].index[-1])
    price = _safe_float(close.reindex([date]).iloc[0])
    if price is None:
        return None
    age = len(recent) - list(recent.index).index(date) - 1
    score = max(0.55, 0.78 - (age * 0.006))
    return {
        "name": name,
        "direction": direction,
        "status": f"Observed {age} bars ago",
        "score": score,
        "date": date,
        "price": price,
        "plot": True,
    }


def _donchian_signal(technical_view: dict[str, Any]) -> dict[str, Any] | None:
    context = technical_view.get("chapter_16_donchian_context") or technical_view.get("chapter_16_market_context", {}).get("donchian_context", {})
    if not context:
        return None
    state = str(context.get("overall_state") or context.get("state") or context.get("status") or "Unavailable")
    if state in {"Unavailable", "InsufficientData"}:
        return None
    direction = "Bullish" if "Long" in state else "Bearish" if "Short" in state else "Neutral"
    primary = context.get("primary", {})
    position = _safe_float(primary.get("channel_position"))
    detail = f"{state}"
    if position is not None:
        detail = f"{detail}, channel position {position:.2f}"
    return {"name": "Donchian 20 context", "direction": direction, "status": detail, "score": 0.79, "plot": False}


def _support_resistance_signal(report: dict[str, Any]) -> dict[str, Any] | None:
    daily = report.get("technical_view", {}).get("support_resistance_by_timeframe", {}).get("daily", {})
    support = _safe_float(daily.get("support"))
    resistance = _safe_float(daily.get("resistance"))
    latest = _safe_float(daily.get("latest_close")) or _safe_float(report.get("current_price"))
    if support is None and resistance is None:
        return None
    detail_parts = []
    if support is not None:
        detail_parts.append(f"support {support:.2f}")
    if resistance is not None:
        detail_parts.append(f"resistance {resistance:.2f}")
    if latest is not None:
        detail_parts.append(f"close {latest:.2f}")
    return {
        "name": "Nearest support/resistance",
        "direction": "Neutral",
        "status": ", ".join(detail_parts),
        "score": 0.61,
        "plot": False,
    }


def _evidence_matrix_signals(technical_view: dict[str, Any]) -> list[dict[str, Any]]:
    entries = (
        technical_view.get("chapter_17_governance_context", {})
        .get("technical_evidence_matrix", {})
        .get("entries", [])
    )
    if not entries:
        return []
    priority = {
        "reversal_patterns": 0.67,
        "triangle_patterns": 0.73,
        "chapter_9_rectangle": 0.69,
        "chapter_9_multi_top_bottom": 0.62,
        "chapter_10_structural": 0.60,
        "chapter_12_gap": 0.57,
        "chapter_14_trendlines": 0.70,
        "chapter_15_major_trendline": 0.72,
    }
    signals: list[dict[str, Any]] = []
    for entry in entries:
        name = str(entry.get("name", ""))
        if name not in priority:
            continue
        label = str(entry.get("label", ""))
        if not label or "NoPattern" in label or label == "Unavailable":
            continue
        strength = _safe_float(entry.get("strength"))
        score = strength if strength is not None else priority[name]
        score = min(0.84, max(priority[name], score))
        signals.append(
            {
                "name": _clean_evidence_name(name),
                "direction": _display_direction(entry.get("direction")),
                "status": label,
                "score": score,
                "plot": False,
            }
        )
    return signals


def _clean_evidence_name(name: str) -> str:
    names = {
        "reversal_patterns": "Reversal pattern",
        "triangle_patterns": "Triangle pattern",
        "chapter_9_rectangle": "Rectangle pattern",
        "chapter_9_multi_top_bottom": "Multi-top/bottom",
        "chapter_10_structural": "Structural warning",
        "chapter_12_gap": "Gap context",
        "chapter_14_trendlines": "Trendline context",
        "chapter_15_major_trendline": "Major trendline",
    }
    return names.get(name, name.replace("_", " ").title())


def _adjust_signal_for_regime(
    signal: dict[str, Any],
    trend_direction: str,
    ma_breakdown: bool,
    ma_breakout: bool,
) -> dict[str, Any]:
    adjusted = dict(signal)
    direction = str(adjusted.get("direction", "Neutral"))
    if trend_direction == "Bullish" and direction == "Bearish" and not ma_breakdown:
        adjusted["score"] = float(adjusted.get("score", 0.0)) * 0.55
        adjusted["status"] = f"{adjusted.get('status', '')} | down-ranked by bullish regime"
    if trend_direction == "Bearish" and direction == "Bullish" and not ma_breakout:
        adjusted["score"] = float(adjusted.get("score", 0.0)) * 0.55
        adjusted["status"] = f"{adjusted.get('status', '')} | down-ranked by bearish regime"
    return adjusted


def _has_ma_breakdown(close: pd.Series) -> bool:
    clean = close.dropna()
    if len(clean) < 50:
        return False
    sma_20 = clean.rolling(20).mean()
    sma_50 = clean.rolling(50).mean()
    return bool(clean.iloc[-1] < sma_20.iloc[-1] < sma_50.iloc[-1])


def _has_ma_breakout(close: pd.Series) -> bool:
    clean = close.dropna()
    if len(clean) < 50:
        return False
    sma_20 = clean.rolling(20).mean()
    sma_50 = clean.rolling(50).mean()
    return bool(clean.iloc[-1] > sma_20.iloc[-1] > sma_50.iloc[-1])


def _dedupe_clean_signals(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for signal in signals:
        key = str(signal.get("name", ""))
        if not key:
            continue
        existing = deduped.get(key)
        if existing is None or float(signal.get("score", 0.0)) > float(existing.get("score", 0.0)):
            deduped[key] = signal
    return list(deduped.values())


def _plot_clean_signal_markers(ax: Any, signals: list[dict[str, Any]], history_index: pd.Index) -> None:
    plotted = 0
    for signal in signals:
        if plotted >= 4 or not signal.get("plot"):
            continue
        date = _parse_date(signal.get("date"))
        price = _safe_float(signal.get("price"))
        if date is None or price is None or date < history_index.min() or date > history_index.max():
            continue
        direction = str(signal.get("direction", "Neutral"))
        marker = "^" if direction == "Bullish" else "v" if direction == "Bearish" else "o"
        color = _clean_direction_color(direction)
        ax.scatter([date], [price], marker=marker, color=color, s=58, label=str(signal.get("name", "Signal")), zorder=5)
        plotted += 1


def _add_plotly_clean_signal_markers(fig: Any, signals: list[dict[str, Any]], history_index: pd.Index) -> None:
    plotted = 0
    for signal in signals:
        if plotted >= 4 or not signal.get("plot"):
            continue
        date = _parse_date(signal.get("date"))
        price = _safe_float(signal.get("price"))
        if date is None or price is None or date < history_index.min() or date > history_index.max():
            continue
        direction = str(signal.get("direction", "Neutral"))
        symbol = "triangle-up" if direction == "Bullish" else "triangle-down" if direction == "Bearish" else "circle"
        color = _clean_direction_color(direction)
        name = str(signal.get("name", "Signal"))
        fig.add_trace(
            go.Scatter(
                x=[date],
                y=[price],
                mode="markers",
                name=name,
                marker={"symbol": symbol, "color": color, "size": 11},
                hovertemplate=f"{name}<br>Date: %{{x|%Y-%m-%d}}<br>Close: %{{y:.2f}}<br>{signal.get('status', '')}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        plotted += 1


def _clean_signal_text(report: dict[str, Any], signals: list[dict[str, Any]]) -> str:
    forecast = _preferred_forecast(report)
    action = report.get("suggested_action", "Unknown")
    risk = report.get("risk_level", "Unknown")
    direction = forecast.get("expected_direction", "Unknown") if forecast else "Unknown"
    confidence = _safe_float(forecast.get("directional_confidence")) if forecast else None
    confidence_text = f"{confidence:.0%}" if confidence is not None else "n/a"
    lines = [
        f"Action: {action} | Risk: {risk} | Forecast: {direction} ({confidence_text})",
        "Ranked evidence:",
    ]
    for index, signal in enumerate(signals[:7], start=1):
        status = str(signal.get("status", ""))
        if len(status) > 94:
            status = f"{status[:91]}..."
        lines.append(
            f"{index}. {signal.get('name', ''):<26} {signal.get('direction', ''):<8} "
            f"{float(signal.get('score', 0.0)):.2f}  {status}"
        )
    return "\n".join(lines)


def _clean_direction_color(direction: str) -> str:
    if direction == "Bullish":
        return "#15803d"
    if direction == "Bearish":
        return "#b91c1c"
    if direction == "Warning":
        return "#d97706"
    return "#475569"


def _plot_head_and_shoulders_overlay(ax: Any, pattern: dict[str, Any]) -> None:
    if pattern.get("status") in {None, "NoPattern", "InsufficientData"}:
        return
    is_bottom = pattern.get("pattern") == "HeadAndShouldersBottom"
    color = "#15803d" if is_bottom else "#b91c1c"
    shoulder_color = "#0f766e" if is_bottom else "#ea580c"
    marker = "v" if is_bottom else "^"
    label_prefix = "H&S bottom" if is_bottom else "H&S top"
    point_styles = {
        "left_shoulder": (marker, shoulder_color, f"{label_prefix} left shoulder"),
        "head": ("*", color, f"{label_prefix} head"),
        "right_shoulder": (marker, shoulder_color, f"{label_prefix} right shoulder"),
    }
    points = pattern.get("points", {})
    for key, (point_marker, point_color, label) in point_styles.items():
        point = points.get(key, {})
        date = _parse_date(point.get("date"))
        price = _safe_float(point.get("price"))
        if date is not None and price is not None:
            ax.scatter([date], [price], marker=point_marker, color=point_color, s=72, label=label, zorder=5)

    neckline = pattern.get("neckline", {})
    start_date = _parse_date(neckline.get("start_date"))
    latest_date = _parse_date(neckline.get("latest_date"))
    start_price = _safe_float(neckline.get("start_price"))
    latest_price = _safe_float(neckline.get("latest_price"))
    if None not in (start_date, latest_date, start_price, latest_price):
        ax.plot(
            [start_date, latest_date],
            [start_price, latest_price],
            color=color,
            linewidth=1.2,
            linestyle="--",
            label=f"{label_prefix} neckline",
        )

    objective = _safe_float(pattern.get("measured_objective"))
    if objective is not None and objective > 0:
        ax.axhline(objective, color=color, linewidth=1.0, linestyle=":", label=f"{label_prefix} objective")


def _add_plotly_head_and_shoulders_overlay(fig: Any, pattern: dict[str, Any]) -> None:
    if pattern.get("status") in {None, "NoPattern", "InsufficientData"}:
        return
    is_bottom = pattern.get("pattern") == "HeadAndShouldersBottom"
    color = "#15803d" if is_bottom else "#b91c1c"
    shoulder_color = "#0f766e" if is_bottom else "#ea580c"
    shoulder_symbol = "triangle-down" if is_bottom else "triangle-up"
    label_prefix = "H&S bottom" if is_bottom else "H&S top"
    point_styles = {
        "left_shoulder": (shoulder_symbol, shoulder_color, f"{label_prefix} left shoulder"),
        "head": ("star", color, f"{label_prefix} head"),
        "right_shoulder": (shoulder_symbol, shoulder_color, f"{label_prefix} right shoulder"),
    }
    points = pattern.get("points", {})
    for key, (symbol, point_color, name) in point_styles.items():
        point = points.get(key, {})
        date = _parse_date(point.get("date"))
        price = _safe_float(point.get("price"))
        if date is None or price is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=[date],
                y=[price],
                mode="markers",
                name=name,
                marker={"symbol": symbol, "color": point_color, "size": 11},
                hovertemplate=f"{name}<br>Date: %{{x|%Y-%m-%d}}<br>Price: %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    neckline = pattern.get("neckline", {})
    start_date = _parse_date(neckline.get("start_date"))
    latest_date = _parse_date(neckline.get("latest_date"))
    start_price = _safe_float(neckline.get("start_price"))
    latest_price = _safe_float(neckline.get("latest_price"))
    if None not in (start_date, latest_date, start_price, latest_price):
        fig.add_trace(
            go.Scatter(
                x=[start_date, latest_date],
                y=[start_price, latest_price],
                mode="lines",
                name=f"{label_prefix} neckline",
                line={"color": color, "dash": "dash", "width": 1.4},
                hovertemplate=f"{label_prefix} neckline<br>Date: %{{x|%Y-%m-%d}}<br>Price: %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    objective = _safe_float(pattern.get("measured_objective"))
    if objective is not None and objective > 0:
        fig.add_hline(
            y=objective,
            line_width=1,
            line_dash="dot",
            line_color=color,
            annotation_text=f"{label_prefix} objective",
            row=1,
            col=1,
        )


def _plot_dormant_bottom_overlay(ax: Any, pattern: dict[str, Any]) -> None:
    if pattern.get("status") in {None, "NoPattern", "InsufficientData"}:
        return
    start_date = _parse_date(pattern.get("base_start_date"))
    end_date = _parse_date(pattern.get("base_end_date"))
    base_low = _safe_float(pattern.get("base_low"))
    base_high = _safe_float(pattern.get("base_high"))
    if None in (start_date, end_date, base_low, base_high):
        return
    ax.axhspan(base_low, base_high, xmin=0.0, xmax=1.0, color="#bbf7d0", alpha=0.10, label="Dormant base zone")
    ax.plot([start_date, end_date], [base_high, base_high], color="#15803d", linewidth=1.0, linestyle=":", label="Dormant base high")
    ax.plot([start_date, end_date], [base_low, base_low], color="#15803d", linewidth=1.0, linestyle=":", label="Dormant base low")


def _add_plotly_dormant_bottom_overlay(fig: Any, pattern: dict[str, Any]) -> None:
    if pattern.get("status") in {None, "NoPattern", "InsufficientData"}:
        return
    start_date = _parse_date(pattern.get("base_start_date"))
    end_date = _parse_date(pattern.get("base_end_date"))
    base_low = _safe_float(pattern.get("base_low"))
    base_high = _safe_float(pattern.get("base_high"))
    if None in (start_date, end_date, base_low, base_high):
        return
    fig.add_trace(
        go.Scatter(
            x=[start_date, end_date, end_date, start_date, start_date],
            y=[base_low, base_low, base_high, base_high, base_low],
            mode="lines",
            name="Dormant base zone",
            fill="toself",
            fillcolor="rgba(187, 247, 208, 0.18)",
            line={"color": "#15803d", "dash": "dot", "width": 1},
            hovertemplate="Dormant base<br>Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )


def _plot_triangle_overlay(ax: Any, pattern: dict[str, Any]) -> None:
    if pattern.get("status") in {None, "NoPattern", "InsufficientData"} or pattern.get("pattern") == "NoTriangle":
        return
    boundaries = pattern.get("boundaries", {})
    start_date = _parse_date(boundaries.get("start_date"))
    latest_date = _parse_date(boundaries.get("latest_date"))
    upper_start = _safe_float(boundaries.get("upper_start"))
    upper_latest = _safe_float(boundaries.get("upper_latest"))
    lower_start = _safe_float(boundaries.get("lower_start"))
    lower_latest = _safe_float(boundaries.get("lower_latest"))
    if None in (start_date, latest_date, upper_start, upper_latest, lower_start, lower_latest):
        return
    color = _triangle_color(pattern)
    label = _triangle_label(pattern)
    ax.plot([start_date, latest_date], [upper_start, upper_latest], color=color, linewidth=1.2, linestyle="--", label=f"{label} upper")
    ax.plot([start_date, latest_date], [lower_start, lower_latest], color=color, linewidth=1.2, linestyle="--", label=f"{label} lower")

    apex_date = _parse_date(pattern.get("apex", {}).get("date"))
    if apex_date is not None:
        ax.axvline(apex_date, color=color, linewidth=0.8, linestyle=":", alpha=0.65, label=f"{label} apex")

    objective = _safe_float(pattern.get("measured_objective"))
    if objective is not None and objective > 0:
        ax.axhline(objective, color=color, linewidth=1.0, linestyle=":", label=f"{label} objective")


def _add_plotly_triangle_overlay(fig: Any, pattern: dict[str, Any]) -> None:
    if pattern.get("status") in {None, "NoPattern", "InsufficientData"} or pattern.get("pattern") == "NoTriangle":
        return
    boundaries = pattern.get("boundaries", {})
    start_date = _parse_date(boundaries.get("start_date"))
    latest_date = _parse_date(boundaries.get("latest_date"))
    upper_start = _safe_float(boundaries.get("upper_start"))
    upper_latest = _safe_float(boundaries.get("upper_latest"))
    lower_start = _safe_float(boundaries.get("lower_start"))
    lower_latest = _safe_float(boundaries.get("lower_latest"))
    if None in (start_date, latest_date, upper_start, upper_latest, lower_start, lower_latest):
        return
    color = _triangle_color(pattern)
    label = _triangle_label(pattern)
    for boundary_name, start_value, latest_value in (
        ("upper", upper_start, upper_latest),
        ("lower", lower_start, lower_latest),
    ):
        fig.add_trace(
            go.Scatter(
                x=[start_date, latest_date],
                y=[start_value, latest_value],
                mode="lines",
                name=f"{label} {boundary_name}",
                line={"color": color, "dash": "dash", "width": 1.3},
                hovertemplate=f"{label} {boundary_name}<br>Date: %{{x|%Y-%m-%d}}<br>Price: %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    objective = _safe_float(pattern.get("measured_objective"))
    if objective is not None and objective > 0:
        fig.add_hline(
            y=objective,
            line_width=1,
            line_dash="dot",
            line_color=color,
            annotation_text=f"{label} objective",
            row=1,
            col=1,
        )


def _plot_rectangle_overlay(ax: Any, pattern: dict[str, Any]) -> None:
    if pattern.get("status") in {None, "NoPattern", "InsufficientData"} or pattern.get("pattern") != "Rectangle":
        return
    boundaries = pattern.get("boundaries", {})
    start_date = _parse_date(boundaries.get("start_date"))
    latest_date = _parse_date(boundaries.get("latest_date"))
    resistance = _safe_float(boundaries.get("resistance"))
    support = _safe_float(boundaries.get("support"))
    if None in (start_date, latest_date, resistance, support):
        return
    color = _chapter_9_color(pattern)
    label = f"Rectangle {pattern.get('status', '')}".strip()
    ax.plot([start_date, latest_date], [resistance, resistance], color=color, linewidth=1.2, linestyle="-.", label=f"{label} resistance")
    ax.plot([start_date, latest_date], [support, support], color=color, linewidth=1.2, linestyle="-.", label=f"{label} support")
    ax.fill_between([start_date, latest_date], [support, support], [resistance, resistance], color=color, alpha=0.07)
    objective = _safe_float(pattern.get("measured_objective"))
    if objective is not None and objective > 0:
        ax.axhline(objective, color=color, linewidth=1.0, linestyle=":", label=f"{label} objective")


def _add_plotly_rectangle_overlay(fig: Any, pattern: dict[str, Any]) -> None:
    if pattern.get("status") in {None, "NoPattern", "InsufficientData"} or pattern.get("pattern") != "Rectangle":
        return
    boundaries = pattern.get("boundaries", {})
    start_date = _parse_date(boundaries.get("start_date"))
    latest_date = _parse_date(boundaries.get("latest_date"))
    resistance = _safe_float(boundaries.get("resistance"))
    support = _safe_float(boundaries.get("support"))
    if None in (start_date, latest_date, resistance, support):
        return
    color = _chapter_9_color(pattern)
    label = f"Rectangle {pattern.get('status', '')}".strip()
    fig.add_trace(
        go.Scatter(
            x=[start_date, latest_date, latest_date, start_date, start_date],
            y=[support, support, resistance, resistance, support],
            mode="lines",
            name=label,
            fill="toself",
            fillcolor="rgba(14, 165, 233, 0.10)",
            line={"color": color, "dash": "dashdot", "width": 1.2},
            hovertemplate=f"{label}<br>Date: %{{x|%Y-%m-%d}}<br>Price: %{{y:.2f}}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    objective = _safe_float(pattern.get("measured_objective"))
    if objective is not None and objective > 0:
        fig.add_hline(
            y=objective,
            line_width=1,
            line_dash="dot",
            line_color=color,
            annotation_text=f"{label} objective",
            row=1,
            col=1,
        )


def _plot_multi_top_bottom_overlay(ax: Any, pattern: dict[str, Any]) -> None:
    if pattern.get("status") in {None, "NoPattern", "InsufficientData"} or pattern.get("pattern") == "NoMultiTopBottom":
        return
    color = _chapter_9_color(pattern)
    is_bottom = "Bottom" in str(pattern.get("pattern"))
    marker = "v" if is_bottom else "^"
    label = f"{pattern.get('pattern')} {pattern.get('status')}"
    for point in pattern.get("points", {}).values():
        date = _parse_date(point.get("date"))
        price = _safe_float(point.get("price"))
        if date is not None and price is not None:
            ax.scatter([date], [price], marker=marker, color=color, s=62, label=label, zorder=5)
            label = "_nolegend_"
    confirmation_level = _safe_float(pattern.get("confirmation_level"))
    if confirmation_level is not None and confirmation_level > 0:
        ax.axhline(confirmation_level, color=color, linewidth=1.0, linestyle="--", label=f"{pattern.get('pattern')} confirmation")
    objective = _safe_float(pattern.get("measured_objective"))
    if objective is not None and objective > 0:
        ax.axhline(objective, color=color, linewidth=1.0, linestyle=":", label=f"{pattern.get('pattern')} objective")


def _add_plotly_multi_top_bottom_overlay(fig: Any, pattern: dict[str, Any]) -> None:
    if pattern.get("status") in {None, "NoPattern", "InsufficientData"} or pattern.get("pattern") == "NoMultiTopBottom":
        return
    color = _chapter_9_color(pattern)
    is_bottom = "Bottom" in str(pattern.get("pattern"))
    symbol = "triangle-down" if is_bottom else "triangle-up"
    label = f"{pattern.get('pattern')} {pattern.get('status')}"
    dates = []
    prices = []
    for point in pattern.get("points", {}).values():
        date = _parse_date(point.get("date"))
        price = _safe_float(point.get("price"))
        if date is not None and price is not None:
            dates.append(date)
            prices.append(price)
    if dates:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=prices,
                mode="markers",
                name=label,
                marker={"symbol": symbol, "color": color, "size": 10},
                hovertemplate=f"{label}<br>Date: %{{x|%Y-%m-%d}}<br>Price: %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    confirmation_level = _safe_float(pattern.get("confirmation_level"))
    if confirmation_level is not None and confirmation_level > 0:
        fig.add_hline(
            y=confirmation_level,
            line_width=1,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{pattern.get('pattern')} confirmation",
            row=1,
            col=1,
        )
    objective = _safe_float(pattern.get("measured_objective"))
    if objective is not None and objective > 0:
        fig.add_hline(
            y=objective,
            line_width=1,
            line_dash="dot",
            line_color=color,
            annotation_text=f"{pattern.get('pattern')} objective",
            row=1,
            col=1,
        )


def _plot_chapter_10_structural_overlay(ax: Any, pattern: dict[str, Any]) -> None:
    if pattern.get("status") in {None, "NoPattern", "InsufficientData"} or pattern.get("pattern", "").startswith("No"):
        return
    boundaries = pattern.get("boundaries", {})
    start_date = _parse_date(boundaries.get("start_date"))
    latest_date = _parse_date(boundaries.get("latest_date"))
    upper_start = _safe_float(boundaries.get("upper_start"))
    upper_latest = _safe_float(boundaries.get("upper_latest"))
    lower_start = _safe_float(boundaries.get("lower_start"))
    lower_latest = _safe_float(boundaries.get("lower_latest"))
    color = _chapter_10_color(pattern)
    label = f"{pattern.get('pattern')} {pattern.get('status')}"
    if None not in (start_date, latest_date, upper_start, upper_latest, lower_start, lower_latest):
        ax.plot([start_date, latest_date], [upper_start, upper_latest], color=color, linewidth=1.2, linestyle="--", label=f"{label} upper")
        ax.plot([start_date, latest_date], [lower_start, lower_latest], color=color, linewidth=1.2, linestyle="--", label=f"{label} lower")
    for point in pattern.get("points", {}).values():
        date = _parse_date(point.get("date"))
        price = _safe_float(point.get("price"))
        if date is not None and price is not None:
            ax.scatter([date], [price], marker="x", color=color, s=54, zorder=5)
    objective = _safe_float(pattern.get("measured_objective"))
    if objective is not None and objective > 0:
        ax.axhline(objective, color=color, linewidth=1.0, linestyle=":", label=f"{label} objective")


def _add_plotly_chapter_10_structural_overlay(fig: Any, pattern: dict[str, Any]) -> None:
    if pattern.get("status") in {None, "NoPattern", "InsufficientData"} or pattern.get("pattern", "").startswith("No"):
        return
    boundaries = pattern.get("boundaries", {})
    start_date = _parse_date(boundaries.get("start_date"))
    latest_date = _parse_date(boundaries.get("latest_date"))
    upper_start = _safe_float(boundaries.get("upper_start"))
    upper_latest = _safe_float(boundaries.get("upper_latest"))
    lower_start = _safe_float(boundaries.get("lower_start"))
    lower_latest = _safe_float(boundaries.get("lower_latest"))
    color = _chapter_10_color(pattern)
    label = f"{pattern.get('pattern')} {pattern.get('status')}"
    if None not in (start_date, latest_date, upper_start, upper_latest, lower_start, lower_latest):
        for boundary_name, start_value, latest_value in (
            ("upper", upper_start, upper_latest),
            ("lower", lower_start, lower_latest),
        ):
            fig.add_trace(
                go.Scatter(
                    x=[start_date, latest_date],
                    y=[start_value, latest_value],
                    mode="lines",
                    name=f"{label} {boundary_name}",
                    line={"color": color, "dash": "dash", "width": 1.3},
                    hovertemplate=f"{label} {boundary_name}<br>Date: %{{x|%Y-%m-%d}}<br>Price: %{{y:.2f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )
    points = pattern.get("points", {})
    if points:
        dates = []
        prices = []
        for point in points.values():
            date = _parse_date(point.get("date"))
            price = _safe_float(point.get("price"))
            if date is not None and price is not None:
                dates.append(date)
                prices.append(price)
        if dates:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=prices,
                    mode="markers",
                    name=f"{label} pivots",
                    marker={"symbol": "x", "color": color, "size": 9},
                    hovertemplate=f"{label}<br>Date: %{{x|%Y-%m-%d}}<br>Price: %{{y:.2f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )
    objective = _safe_float(pattern.get("measured_objective"))
    if objective is not None and objective > 0:
        fig.add_hline(
            y=objective,
            line_width=1,
            line_dash="dot",
            line_color=color,
            annotation_text=f"{label} objective",
            row=1,
            col=1,
        )


def _plot_chapter_10_event_overlay(ax: Any, event: dict[str, Any]) -> None:
    if event.get("status") in {None, "NoPattern", "InsufficientData"} or event.get("pattern") == "NoShortTermEvent":
        return
    date = _parse_date(event.get("date"))
    close = _safe_float(event.get("close"))
    if date is None or close is None:
        return
    color = _chapter_10_color(event)
    marker = "v" if str(event.get("direction", "")).startswith("bearish") else "^"
    ax.scatter([date], [close], marker=marker, color=color, s=72, label=f"{event.get('pattern')} event", zorder=6)


def _add_plotly_chapter_10_event_overlay(fig: Any, event: dict[str, Any]) -> None:
    if event.get("status") in {None, "NoPattern", "InsufficientData"} or event.get("pattern") == "NoShortTermEvent":
        return
    date = _parse_date(event.get("date"))
    close = _safe_float(event.get("close"))
    if date is None or close is None:
        return
    color = _chapter_10_color(event)
    symbol = "triangle-down" if str(event.get("direction", "")).startswith("bearish") else "triangle-up"
    label = f"{event.get('pattern')} event"
    fig.add_trace(
        go.Scatter(
            x=[date],
            y=[close],
            mode="markers",
            name=label,
            marker={"symbol": symbol, "color": color, "size": 11},
            hovertemplate=f"{label}<br>Date: %{{x|%Y-%m-%d}}<br>Close: %{{y:.2f}}<extra></extra>",
        ),
        row=1,
        col=1,
    )


def _plot_chapter_11_continuation_overlay(ax: Any, pattern: dict[str, Any]) -> None:
    if pattern.get("status") in {None, "NoPattern", "InsufficientData"} or pattern.get("pattern") not in {"Flag", "Pennant"}:
        return
    boundaries = pattern.get("boundaries", {})
    start_date = _parse_date(boundaries.get("start_date"))
    latest_date = _parse_date(boundaries.get("latest_date"))
    upper_start = _safe_float(boundaries.get("upper_start"))
    upper_latest = _safe_float(boundaries.get("upper_latest"))
    lower_start = _safe_float(boundaries.get("lower_start"))
    lower_latest = _safe_float(boundaries.get("lower_latest"))
    if any(value is None for value in (start_date, latest_date, upper_start, upper_latest, lower_start, lower_latest)):
        return
    color = _chapter_11_color(pattern)
    label = f"Ch11 {pattern.get('pattern')} {pattern.get('status')}"
    ax.plot([start_date, latest_date], [upper_start, upper_latest], color=color, linewidth=1.2, linestyle="-.", label=f"{label} upper")
    ax.plot([start_date, latest_date], [lower_start, lower_latest], color=color, linewidth=1.2, linestyle="-.", label=f"{label} lower")
    mast = pattern.get("mast", {})
    mast_start = _parse_date(mast.get("start_date"))
    mast_end = _parse_date(mast.get("end_date"))
    mast_start_price = _safe_float(mast.get("start_price"))
    mast_end_price = _safe_float(mast.get("end_price"))
    if None not in (mast_start, mast_end, mast_start_price, mast_end_price):
        ax.plot([mast_start, mast_end], [mast_start_price, mast_end_price], color=color, linewidth=1.4, alpha=0.70, label=f"{label} mast")
    breakout_date = _parse_date(pattern.get("breakout_date"))
    breakout_close = _safe_float(pattern.get("breakout_close"))
    if breakout_date is not None and breakout_close is not None:
        marker = "^" if str(pattern.get("direction", "")).startswith("bullish") else "v"
        ax.scatter([breakout_date], [breakout_close], marker=marker, color=color, s=74, label=f"{label} break", zorder=6)
    objective = _safe_float(pattern.get("measured_objective"))
    if objective is not None and objective > 0:
        ax.axhline(objective, color=color, linewidth=1.0, linestyle=":", label=f"{label} objective")


def _add_plotly_chapter_11_continuation_overlay(fig: Any, pattern: dict[str, Any]) -> None:
    if pattern.get("status") in {None, "NoPattern", "InsufficientData"} or pattern.get("pattern") not in {"Flag", "Pennant"}:
        return
    boundaries = pattern.get("boundaries", {})
    start_date = _parse_date(boundaries.get("start_date"))
    latest_date = _parse_date(boundaries.get("latest_date"))
    upper_start = _safe_float(boundaries.get("upper_start"))
    upper_latest = _safe_float(boundaries.get("upper_latest"))
    lower_start = _safe_float(boundaries.get("lower_start"))
    lower_latest = _safe_float(boundaries.get("lower_latest"))
    if any(value is None for value in (start_date, latest_date, upper_start, upper_latest, lower_start, lower_latest)):
        return
    color = _chapter_11_color(pattern)
    label = f"Ch11 {pattern.get('pattern')} {pattern.get('status')}"
    for boundary_name, start_value, latest_value in (
        ("upper", upper_start, upper_latest),
        ("lower", lower_start, lower_latest),
    ):
        fig.add_trace(
            go.Scatter(
                x=[start_date, latest_date],
                y=[start_value, latest_value],
                mode="lines",
                name=f"{label} {boundary_name}",
                line={"color": color, "dash": "dashdot", "width": 1.3},
                hovertemplate=f"{label} {boundary_name}<br>Date: %{{x|%Y-%m-%d}}<br>Price: %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    mast = pattern.get("mast", {})
    mast_start = _parse_date(mast.get("start_date"))
    mast_end = _parse_date(mast.get("end_date"))
    mast_start_price = _safe_float(mast.get("start_price"))
    mast_end_price = _safe_float(mast.get("end_price"))
    if None not in (mast_start, mast_end, mast_start_price, mast_end_price):
        fig.add_trace(
            go.Scatter(
                x=[mast_start, mast_end],
                y=[mast_start_price, mast_end_price],
                mode="lines",
                name=f"{label} mast",
                line={"color": color, "width": 1.5},
                hovertemplate=f"{label} mast<br>Date: %{{x|%Y-%m-%d}}<br>Price: %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    breakout_date = _parse_date(pattern.get("breakout_date"))
    breakout_close = _safe_float(pattern.get("breakout_close"))
    if breakout_date is not None and breakout_close is not None:
        symbol = "triangle-up" if str(pattern.get("direction", "")).startswith("bullish") else "triangle-down"
        fig.add_trace(
            go.Scatter(
                x=[breakout_date],
                y=[breakout_close],
                mode="markers",
                name=f"{label} break",
                marker={"symbol": symbol, "color": color, "size": 11},
                hovertemplate=f"{label} break<br>Date: %{{x|%Y-%m-%d}}<br>Close: %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    objective = _safe_float(pattern.get("measured_objective"))
    if objective is not None and objective > 0:
        fig.add_hline(
            y=objective,
            line_width=1,
            line_dash="dot",
            line_color=color,
            annotation_text=f"{label} objective",
            row=1,
            col=1,
        )


def _plot_chapter_11_hs_continuation_overlay(ax: Any, pattern: dict[str, Any]) -> None:
    if pattern.get("status") in {None, "NoPattern", "InsufficientData"} or pattern.get("pattern") != "HeadAndShouldersContinuation":
        return
    color = _chapter_11_color(pattern)
    marker = "v" if pattern.get("direction") == "bullish" else "^"
    label = f"Ch11 H&S {pattern.get('status')}"
    dates = []
    prices = []
    for point in pattern.get("points", {}).values():
        date = _parse_date(point.get("date"))
        price = _safe_float(point.get("price"))
        if date is not None and price is not None:
            dates.append(date)
            prices.append(price)
    if dates:
        ax.scatter(dates, prices, marker=marker, color=color, s=62, label=label, zorder=5)
    neckline = _safe_float(pattern.get("neckline"))
    if neckline is not None and len(dates) >= 2:
        ax.plot([min(dates), max(dates)], [neckline, neckline], color=color, linewidth=1.1, linestyle="--", label=f"{label} neckline")
    objective = _safe_float(pattern.get("measured_objective"))
    if objective is not None and objective > 0:
        ax.axhline(objective, color=color, linewidth=1.0, linestyle=":", label=f"{label} objective")


def _add_plotly_chapter_11_hs_continuation_overlay(fig: Any, pattern: dict[str, Any]) -> None:
    if pattern.get("status") in {None, "NoPattern", "InsufficientData"} or pattern.get("pattern") != "HeadAndShouldersContinuation":
        return
    color = _chapter_11_color(pattern)
    symbol = "triangle-down" if pattern.get("direction") == "bullish" else "triangle-up"
    label = f"Ch11 H&S {pattern.get('status')}"
    dates = []
    prices = []
    for point in pattern.get("points", {}).values():
        date = _parse_date(point.get("date"))
        price = _safe_float(point.get("price"))
        if date is not None and price is not None:
            dates.append(date)
            prices.append(price)
    if dates:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=prices,
                mode="markers",
                name=label,
                marker={"symbol": symbol, "color": color, "size": 10},
                hovertemplate=f"{label}<br>Date: %{{x|%Y-%m-%d}}<br>Price: %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    neckline = _safe_float(pattern.get("neckline"))
    if neckline is not None and len(dates) >= 2:
        fig.add_trace(
            go.Scatter(
                x=[min(dates), max(dates)],
                y=[neckline, neckline],
                mode="lines",
                name=f"{label} neckline",
                line={"color": color, "dash": "dash", "width": 1.2},
                hovertemplate=f"{label} neckline<br>Date: %{{x|%Y-%m-%d}}<br>Price: %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    objective = _safe_float(pattern.get("measured_objective"))
    if objective is not None and objective > 0:
        fig.add_hline(
            y=objective,
            line_width=1,
            line_dash="dot",
            line_color=color,
            annotation_text=f"{label} objective",
            row=1,
            col=1,
        )


def _plot_chapter_12_gap_overlay(ax: Any, gaps: list[dict[str, Any]], latest_date: object | None) -> None:
    if not gaps:
        return
    latest = _parse_date(latest_date)
    label_used = False
    for gap in gaps[-5:]:
        if gap.get("status") in {None, "NoPattern", "InsufficientData", "Ignored", "Excluded"}:
            continue
        zone = gap.get("gap_zone", {})
        start_date = _parse_date(gap.get("date"))
        end_date = _parse_date(gap.get("fill_state", {}).get("fill_date")) or latest
        lower = _safe_float(zone.get("lower"))
        upper = _safe_float(zone.get("upper"))
        if None in (start_date, end_date, lower, upper):
            continue
        color = _chapter_12_color(gap)
        label = f"Ch12 {gap.get('pattern')} {gap.get('status')}" if not label_used else "_nolegend_"
        ax.fill_between([start_date, end_date], [lower, lower], [upper, upper], color=color, alpha=0.10, label=label)
        ax.plot([start_date, end_date], [lower, lower], color=color, linewidth=0.9, linestyle=":", label="_nolegend_")
        ax.plot([start_date, end_date], [upper, upper], color=color, linewidth=0.9, linestyle=":", label="_nolegend_")
        label_used = True


def _add_plotly_chapter_12_gap_overlay(fig: Any, gaps: list[dict[str, Any]], latest_date: object | None) -> None:
    if not gaps:
        return
    latest = _parse_date(latest_date)
    for gap in gaps[-5:]:
        if gap.get("status") in {None, "NoPattern", "InsufficientData", "Ignored", "Excluded"}:
            continue
        zone = gap.get("gap_zone", {})
        start_date = _parse_date(gap.get("date"))
        end_date = _parse_date(gap.get("fill_state", {}).get("fill_date")) or latest
        lower = _safe_float(zone.get("lower"))
        upper = _safe_float(zone.get("upper"))
        if None in (start_date, end_date, lower, upper):
            continue
        color = _chapter_12_color(gap)
        label = f"Ch12 {gap.get('pattern')} {gap.get('status')}"
        fig.add_trace(
            go.Scatter(
                x=[start_date, end_date, end_date, start_date, start_date],
                y=[lower, lower, upper, upper, lower],
                mode="lines",
                name=label,
                fill="toself",
                fillcolor=_rgba(color, 0.10),
                line={"color": color, "dash": "dot", "width": 1.0},
                hovertemplate=f"{label}<br>Date: %{{x|%Y-%m-%d}}<br>Gap zone: %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )


def _plot_chapter_12_island_overlay(ax: Any, pattern: dict[str, Any]) -> None:
    if pattern.get("status") in {None, "NoPattern", "InsufficientData"} or pattern.get("pattern") != "IslandReversal":
        return
    start_date = _parse_date(pattern.get("start_date"))
    end_date = _parse_date(pattern.get("end_date"))
    island_range = pattern.get("island_range", {})
    low = _safe_float(island_range.get("low"))
    high = _safe_float(island_range.get("high"))
    if None in (start_date, end_date, low, high):
        return
    color = _chapter_12_color(pattern)
    label = f"Ch12 Island {pattern.get('direction')}"
    ax.fill_between([start_date, end_date], [low, low], [high, high], color=color, alpha=0.13, label=label)
    objective = _safe_float(pattern.get("measured_objective"))
    if objective is not None and objective > 0:
        ax.axhline(objective, color=color, linewidth=1.0, linestyle=":", label=f"{label} objective")


def _add_plotly_chapter_12_island_overlay(fig: Any, pattern: dict[str, Any]) -> None:
    if pattern.get("status") in {None, "NoPattern", "InsufficientData"} or pattern.get("pattern") != "IslandReversal":
        return
    start_date = _parse_date(pattern.get("start_date"))
    end_date = _parse_date(pattern.get("end_date"))
    island_range = pattern.get("island_range", {})
    low = _safe_float(island_range.get("low"))
    high = _safe_float(island_range.get("high"))
    if None in (start_date, end_date, low, high):
        return
    color = _chapter_12_color(pattern)
    label = f"Ch12 Island {pattern.get('direction')}"
    fig.add_trace(
        go.Scatter(
            x=[start_date, end_date, end_date, start_date, start_date],
            y=[low, low, high, high, low],
            mode="lines",
            name=label,
            fill="toself",
            fillcolor=_rgba(color, 0.13),
            line={"color": color, "dash": "dash", "width": 1.1},
            hovertemplate=f"{label}<br>Date: %{{x|%Y-%m-%d}}<br>Price: %{{y:.2f}}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    objective = _safe_float(pattern.get("measured_objective"))
    if objective is not None and objective > 0:
        fig.add_hline(
            y=objective,
            line_width=1,
            line_dash="dot",
            line_color=color,
            annotation_text=f"{label} objective",
            row=1,
            col=1,
        )


def _plot_chapter_13_zone_overlay(ax: Any, diagnostics: dict[str, Any]) -> None:
    zones = []
    zones.extend(diagnostics.get("support_zones", [])[:2])
    zones.extend(diagnostics.get("resistance_zones", [])[:2])
    zones.extend([diagnostics.get("round_number_support", {}), diagnostics.get("round_number_resistance", {})])
    label_seen: set[str] = set()
    for zone in zones:
        if zone.get("status") in {None, "NoPattern", "InsufficientData"}:
            continue
        lower = _safe_float(zone.get("lower"))
        upper = _safe_float(zone.get("upper"))
        center = _safe_float(zone.get("center"))
        if lower is None or upper is None or center is None:
            continue
        color = _chapter_13_color(zone)
        role = str(zone.get("role", "zone"))
        label = f"Ch13 {role}" if role not in label_seen else "_nolegend_"
        ax.axhspan(lower, upper, color=color, alpha=0.07, label=label)
        ax.axhline(center, color=color, linewidth=0.9, linestyle="--", alpha=0.85)
        label_seen.add(role)


def _add_plotly_chapter_13_zone_overlay(fig: Any, diagnostics: dict[str, Any]) -> None:
    zones = []
    zones.extend(diagnostics.get("support_zones", [])[:2])
    zones.extend(diagnostics.get("resistance_zones", [])[:2])
    zones.extend([diagnostics.get("round_number_support", {}), diagnostics.get("round_number_resistance", {})])
    for zone in zones:
        if zone.get("status") in {None, "NoPattern", "InsufficientData"}:
            continue
        lower = _safe_float(zone.get("lower"))
        upper = _safe_float(zone.get("upper"))
        center = _safe_float(zone.get("center"))
        if lower is None or upper is None or center is None:
            continue
        color = _chapter_13_color(zone)
        label = f"Ch13 {zone.get('role')} {zone.get('role_reversal', '')}".strip()
        fig.add_hrect(
            y0=lower,
            y1=upper,
            fillcolor=_rgba(color, 0.07),
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
            row=1,
            col=1,
        )
        fig.add_hline(
            y=center,
            line_width=1,
            line_dash="dash",
            line_color=color,
            row=1,
            col=1,
        )


def _plot_chapter_14_trendline_overlay(ax: Any, diagnostics: dict[str, Any]) -> None:
    trendline = diagnostics.get("preferred_trendline", {})
    if trendline.get("status") not in {None, "NoPattern", "InsufficientData"}:
        color = _chapter_14_color(trendline)
        line = trendline.get("line", {})
        _plot_segment_from_payload(
            ax,
            line,
            color=color,
            linestyle="-",
            label=f"Ch14 {trendline.get('kind', 'trendline')}",
        )
        outer = trendline.get("double_trendline", {}).get("line", {})
        if outer:
            _plot_segment_from_payload(
                ax,
                outer,
                color=color,
                linestyle="--",
                label="Ch14 outer trendline",
            )
    channel = diagnostics.get("channel", {})
    if channel.get("status") not in {None, "NoPattern", "InsufficientData"}:
        color = _chapter_14_color(channel)
        line = channel.get("line", {})
        _plot_explicit_segment(
            ax,
            line.get("start_date"),
            line.get("end_date"),
            line.get("return_start_value"),
            line.get("return_current_value"),
            color=color,
            linestyle=":",
            label="Ch14 return line",
        )
    fan = diagnostics.get("fan_lines", {})
    if fan.get("status") not in {None, "NoPattern", "InsufficientData"}:
        color = _chapter_14_color(fan)
        end_date = diagnostics.get("end_date")
        for fan_line in fan.get("lines", []):
            _plot_explicit_segment(
                ax,
                fan_line.get("start_date"),
                end_date,
                fan_line.get("start_value"),
                fan_line.get("current_value"),
                color=color,
                linestyle="-.",
                label=f"Ch14 {fan_line.get('name', 'fan')}",
            )


def _add_plotly_chapter_14_trendline_overlay(fig: Any, diagnostics: dict[str, Any]) -> None:
    trendline = diagnostics.get("preferred_trendline", {})
    if trendline.get("status") not in {None, "NoPattern", "InsufficientData"}:
        color = _chapter_14_color(trendline)
        line = trendline.get("line", {})
        _add_plotly_segment_from_payload(
            fig,
            line,
            color=color,
            dash="solid",
            label=f"Ch14 {trendline.get('kind', 'trendline')} {trendline.get('status', '')}".strip(),
        )
        outer = trendline.get("double_trendline", {}).get("line", {})
        if outer:
            _add_plotly_segment_from_payload(fig, outer, color=color, dash="dash", label="Ch14 outer trendline")
    channel = diagnostics.get("channel", {})
    if channel.get("status") not in {None, "NoPattern", "InsufficientData"}:
        color = _chapter_14_color(channel)
        line = channel.get("line", {})
        _add_plotly_explicit_segment(
            fig,
            line.get("start_date"),
            line.get("end_date"),
            line.get("return_start_value"),
            line.get("return_current_value"),
            color=color,
            dash="dot",
            label=f"Ch14 return line {channel.get('status', '')}".strip(),
        )
    fan = diagnostics.get("fan_lines", {})
    if fan.get("status") not in {None, "NoPattern", "InsufficientData"}:
        color = _chapter_14_color(fan)
        end_date = diagnostics.get("end_date")
        for fan_line in fan.get("lines", []):
            _add_plotly_explicit_segment(
                fig,
                fan_line.get("start_date"),
                end_date,
                fan_line.get("start_value"),
                fan_line.get("current_value"),
                color=color,
                dash="dashdot",
                label=f"Ch14 {fan_line.get('name', 'fan')}",
            )


def _plot_chapter_15_major_trendline_overlay(ax: Any, diagnostics: dict[str, Any]) -> None:
    trendline = diagnostics.get("major_trendline", {})
    if trendline.get("status") in {None, "NoPattern", "InsufficientData"}:
        return
    color = _chapter_15_color(trendline)
    label = f"Ch15 {trendline.get('kind', 'major')} {trendline.get('scale', '')}".strip()
    _plot_segment_from_payload(
        ax,
        trendline.get("line", {}),
        color=color,
        linestyle=(0, (5, 2, 1, 2)),
        label=label,
    )


def _add_plotly_chapter_15_major_trendline_overlay(fig: Any, diagnostics: dict[str, Any]) -> None:
    trendline = diagnostics.get("major_trendline", {})
    if trendline.get("status") in {None, "NoPattern", "InsufficientData"}:
        return
    color = _chapter_15_color(trendline)
    label = f"Ch15 {trendline.get('kind', 'major')} {trendline.get('status', '')} {trendline.get('scale', '')}".strip()
    _add_plotly_segment_from_payload(
        fig,
        trendline.get("line", {}),
        color=color,
        dash="longdashdot",
        label=label,
    )


def _plot_chapter_16_market_context_overlay(ax: Any, prices: pd.DataFrame, target_column: str) -> None:
    history = chapter_16_donchian_history(prices, target_column=target_column)
    if history.empty or "donchian_high_20" not in history or "donchian_low_20" not in history:
        return
    upper = history["donchian_high_20"].dropna()
    lower = history["donchian_low_20"].dropna()
    if upper.empty or lower.empty:
        return
    ax.plot(upper.index, upper.values, color="#64748b", linewidth=0.85, linestyle=(0, (2, 2)), label="Ch16 Donchian 20 high")
    ax.plot(lower.index, lower.values, color="#64748b", linewidth=0.85, linestyle=(0, (1, 3)), label="Ch16 Donchian 20 low")


def _add_plotly_chapter_16_market_context_overlay(fig: Any, prices: pd.DataFrame, target_column: str) -> None:
    history = chapter_16_donchian_history(prices, target_column=target_column)
    if history.empty or "donchian_high_20" not in history or "donchian_low_20" not in history:
        return
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["donchian_high_20"],
            mode="lines",
            name="Ch16 Donchian 20 high",
            line={"color": "#64748b", "width": 1, "dash": "dot"},
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Ch16 Donchian 20 high: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["donchian_low_20"],
            mode="lines",
            name="Ch16 Donchian 20 low",
            line={"color": "#64748b", "width": 1, "dash": "dash"},
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Ch16 Donchian 20 low: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )


def _plot_segment_from_payload(ax: Any, line: dict[str, Any], color: str, linestyle: Any, label: str) -> None:
    _plot_explicit_segment(
        ax,
        line.get("start_date"),
        line.get("end_date"),
        line.get("start_value"),
        line.get("current_value"),
        color=color,
        linestyle=linestyle,
        label=label,
    )


def _plot_explicit_segment(
    ax: Any,
    start_date: object,
    end_date: object,
    start_value: object,
    end_value: object,
    color: str,
    linestyle: Any,
    label: str,
) -> None:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    start_price = _safe_float(start_value)
    end_price = _safe_float(end_value)
    if None in (start, end, start_price, end_price):
        return
    ax.plot([start, end], [start_price, end_price], color=color, linewidth=1.25, linestyle=linestyle, label=label)


def _add_plotly_segment_from_payload(fig: Any, line: dict[str, Any], color: str, dash: str, label: str) -> None:
    _add_plotly_explicit_segment(
        fig,
        line.get("start_date"),
        line.get("end_date"),
        line.get("start_value"),
        line.get("current_value"),
        color=color,
        dash=dash,
        label=label,
    )


def _add_plotly_explicit_segment(
    fig: Any,
    start_date: object,
    end_date: object,
    start_value: object,
    end_value: object,
    color: str,
    dash: str,
    label: str,
) -> None:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    start_price = _safe_float(start_value)
    end_price = _safe_float(end_value)
    if None in (start, end, start_price, end_price):
        return
    fig.add_trace(
        go.Scatter(
            x=[start, end],
            y=[start_price, end_price],
            mode="lines",
            name=label,
            line={"color": color, "dash": dash, "width": 1.4},
            hovertemplate=f"{label}<br>Date: %{{x|%Y-%m-%d}}<br>Price: %{{y:.2f}}<extra></extra>",
        ),
        row=1,
        col=1,
    )


def _scatter_signal(
    ax: Any,
    features: pd.DataFrame,
    close: pd.Series,
    column: str,
    marker: str,
    color: str,
    label: str,
) -> None:
    if column not in features:
        return
    signals = features[column].fillna(0.0) > 0
    if not signals.any():
        return
    ax.scatter(close.index[signals], close.loc[signals], marker=marker, color=color, s=36, label=label, zorder=4)


def _scatter_value_signal(
    ax: Any,
    values: pd.DataFrame,
    column: str,
    marker: str,
    color: str,
    label: str,
) -> None:
    if column not in values:
        return
    points = values[column].dropna()
    if points.empty:
        return
    ax.scatter(points.index, points.values, marker=marker, color=color, s=30, label=label, zorder=4)


def _add_plotly_signal(
    fig: go.Figure,
    features: pd.DataFrame,
    close: pd.Series,
    column: str,
    symbol: str,
    color: str,
    label: str,
) -> None:
    if column not in features:
        return
    signals = features[column].fillna(0.0) > 0
    if not signals.any():
        return
    fig.add_trace(
        go.Scatter(
            x=close.index[signals],
            y=close.loc[signals],
            mode="markers",
            name=label,
            marker={"symbol": symbol, "color": color, "size": 9},
            hovertemplate=f"{label}<br>Date: %{{x|%Y-%m-%d}}<br>Close: %{{y:.2f}}<extra></extra>",
        ),
        row=1,
        col=1,
    )


def _add_plotly_value_signal(
    fig: go.Figure,
    values: pd.DataFrame,
    column: str,
    symbol: str,
    color: str,
    label: str,
) -> None:
    if column not in values:
        return
    points = values[column].dropna()
    if points.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=points.index,
            y=points.values,
            mode="markers",
            name=label,
            marker={"symbol": symbol, "color": color, "size": 8},
            hovertemplate=f"{label}<br>Date: %{{x|%Y-%m-%d}}<br>Price: %{{y:.2f}}<extra></extra>",
        ),
        row=1,
        col=1,
    )


def _timeframe_prices(prices: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "daily":
        return prices.copy()
    rule = "W-FRI" if timeframe == "weekly" else "ME"
    aggregations: dict[str, str] = {}
    if "open" in prices.columns:
        aggregations["open"] = "first"
    if "high" in prices.columns:
        aggregations["high"] = "max"
    if "low" in prices.columns:
        aggregations["low"] = "min"
    if "close" in prices.columns:
        aggregations["close"] = "last"
    if "volume" in prices.columns:
        aggregations["volume"] = "sum"
    for optional in ("dividends", "stock_splits"):
        if optional in prices.columns:
            aggregations[optional] = "sum"
    frame = prices.resample(rule).agg(aggregations)
    return frame.dropna(subset=["close"]) if "close" in frame.columns else frame.dropna(how="all")


def _history_length(timeframe: str) -> int:
    if timeframe == "monthly":
        return 120
    if timeframe == "weekly":
        return 156
    return 252


def _plot_ohlc_bars(ax: Any, history: pd.DataFrame, target_column: str) -> None:
    if not all(column in history.columns for column in ("open", "high", "low", target_column)):
        return
    width = _ohlc_tick_width(history.index)
    up = history[target_column] >= history["open"]
    colors = up.map({True: "#16a34a", False: "#dc2626"})
    ax.vlines(history.index, history["low"], history["high"], color=colors, linewidth=0.7, alpha=0.78, label="High-low range")
    for date, row in history.iterrows():
        color = "#16a34a" if row[target_column] >= row["open"] else "#dc2626"
        ax.hlines(row["open"], date - width, date, color=color, linewidth=0.8, alpha=0.75)
        ax.hlines(row[target_column], date, date + width, color=color, linewidth=0.8, alpha=0.75)


def _ohlc_tick_width(index: pd.Index) -> pd.Timedelta:
    if len(index) < 2:
        return pd.Timedelta(days=0.25)
    deltas = pd.Series(pd.DatetimeIndex(index)).diff().dropna()
    if deltas.empty:
        return pd.Timedelta(days=0.25)
    return deltas.median() * 0.28


def _timeframe_level(report: dict[str, Any], timeframe: str, level: str) -> float | None:
    value = (
        report.get("technical_view", {})
        .get("support_resistance_by_timeframe", {})
        .get(timeframe, {})
        .get(level)
    )
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _plot_validation_predictions(
    report: dict[str, Any],
    horizon: int,
    records: list[dict[str, Any]],
    output_file: Path,
) -> None:
    frame = pd.DataFrame(records)
    if frame.empty:
        return

    frame["validation_date"] = pd.to_datetime(frame["validation_date"])
    frame = frame.sort_values("validation_date")
    min_price = float(min(frame["actual_future_price"].min(), frame["predicted_future_price"].min()))
    max_price = float(max(frame["actual_future_price"].max(), frame["predicted_future_price"].max()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(
        frame["validation_date"],
        frame["actual_future_price"],
        color="#111827",
        linewidth=1.8,
        label="Actual future price",
    )
    axes[0].plot(
        frame["validation_date"],
        frame["predicted_future_price"],
        color="#dc2626",
        linewidth=1.5,
        alpha=0.9,
        label="Predicted future price",
    )
    axes[0].set_title(f"{horizon}d Walk-Forward Validation")
    axes[0].set_xlabel("Validation date")
    axes[0].set_ylabel("Future price")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].scatter(
        frame["actual_future_price"],
        frame["predicted_future_price"],
        color="#2563eb",
        alpha=0.72,
        edgecolors="none",
    )
    axes[1].plot([min_price, max_price], [min_price, max_price], color="#6b7280", linestyle="--", linewidth=1.2)
    axes[1].set_title("Actual vs Predicted")
    axes[1].set_xlabel("Actual future price")
    axes[1].set_ylabel("Predicted future price")
    axes[1].grid(True, alpha=0.25)

    selected = next(
        (item for item in report["forecasts"] if int(item["horizon_days"]) == horizon),
        None,
    )
    if selected is not None:
        fig.suptitle(
            f"{report['ticker']} selected model: {selected['selected_model']} | "
            f"MAE: {selected['validation_metrics']['mae']:.4f}",
            y=1.02,
        )

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_validation_predictions_plotly(
    report: dict[str, Any],
    horizon: int,
    records: list[dict[str, Any]],
    output_file: Path,
) -> None:
    frame = pd.DataFrame(records)
    if frame.empty:
        return

    frame["validation_date"] = pd.to_datetime(frame["validation_date"])
    frame = frame.sort_values("validation_date")
    frame["validation_date_label"] = frame["validation_date"].dt.strftime("%Y-%m-%d")
    min_price = float(min(frame["actual_future_price"].min(), frame["predicted_future_price"].min()))
    max_price = float(max(frame["actual_future_price"].max(), frame["predicted_future_price"].max()))
    selected = next(
        (item for item in report["forecasts"] if int(item["horizon_days"]) == horizon),
        None,
    )
    selected_model = selected["selected_model"] if selected is not None else "selected model"
    mae = selected["validation_metrics"]["mae"] if selected is not None else 0.0

    customdata = frame[
        [
            "base_price",
            "actual_log_return",
            "predicted_log_return",
            "split_train_end",
            "validation_date_label",
        ]
    ].to_numpy()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"{horizon}d Walk-Forward Validation", "Actual vs Predicted"),
        horizontal_spacing=0.10,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["validation_date"],
            y=frame["actual_future_price"],
            customdata=customdata,
            mode="lines+markers",
            name="Actual future price",
            line={"color": "#111827", "width": 2},
            marker={"size": 5},
            hovertemplate=(
                "Validation date: %{x|%Y-%m-%d}"
                "<br>Actual future price: %{y:.2f}"
                "<br>Base price: %{customdata[0]:.2f}"
                "<br>Actual log return: %{customdata[1]:.4f}"
                "<br>Train through: %{customdata[3]}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["validation_date"],
            y=frame["predicted_future_price"],
            customdata=customdata,
            mode="lines+markers",
            name="Predicted future price",
            line={"color": "#dc2626", "width": 2},
            marker={"size": 5},
            hovertemplate=(
                "Validation date: %{x|%Y-%m-%d}"
                "<br>Predicted future price: %{y:.2f}"
                "<br>Base price: %{customdata[0]:.2f}"
                "<br>Predicted log return: %{customdata[2]:.4f}"
                "<br>Train through: %{customdata[3]}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["actual_future_price"],
            y=frame["predicted_future_price"],
            customdata=customdata,
            mode="markers",
            name="Validation points",
            marker={"color": "#2563eb", "size": 7, "opacity": 0.72},
            hovertemplate=(
                "Validation date: %{customdata[4]}"
                "<br>Actual future price: %{x:.2f}"
                "<br>Predicted future price: %{y:.2f}"
                "<br>Base price: %{customdata[0]:.2f}"
                "<br>Actual log return: %{customdata[1]:.4f}"
                "<br>Predicted log return: %{customdata[2]:.4f}"
                "<br>Train through: %{customdata[3]}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[min_price, max_price],
            y=[min_price, max_price],
            mode="lines",
            name="Perfect prediction",
            line={"color": "#6b7280", "dash": "dash", "width": 1.4},
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Validation date", row=1, col=1)
    fig.update_yaxes(title_text="Future price", row=1, col=1)
    fig.update_xaxes(title_text="Actual future price", row=1, col=2)
    fig.update_yaxes(title_text="Predicted future price", row=1, col=2)
    fig.update_layout(
        title=f"{report['ticker']} {horizon}d validation | {selected_model} | MAE {mae:.4f}",
        template="plotly_white",
        hovermode="closest",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.write_html(output_file, include_plotlyjs=True, full_html=True)


def _parse_date(value: object) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        return pd.Timestamp(value)
    except Exception:
        return None


def _triangle_color(pattern: dict[str, Any]) -> str:
    status = pattern.get("status")
    direction = pattern.get("direction")
    if status in {"Breakout", "Retest"} and direction == "bullish":
        return "#2563eb"
    if status in {"Breakdown", "Retest"} and direction == "bearish":
        return "#dc2626"
    if status in {"FailedBreakout", "FailedBreakdown", "LateApex"}:
        return "#f59e0b"
    return "#7c3aed"


def _triangle_label(pattern: dict[str, Any]) -> str:
    pattern_name = str(pattern.get("pattern", "Triangle")).replace("Triangle", " Triangle")
    status = pattern.get("status", "Pattern")
    return f"{pattern_name} {status}"


def _chapter_9_color(pattern: dict[str, Any]) -> str:
    status = pattern.get("status")
    direction = pattern.get("direction")
    if status in {"Breakout", "Retest", "Confirmed", "PullbackToConfirmation"} and direction == "bullish":
        return "#15803d"
    if status in {"Breakdown", "Retest", "Confirmed", "PullbackToConfirmation"} and direction == "bearish":
        return "#b91c1c"
    if status in {"FalseBreakout", "FalseBreakdown", "PrematureBreakout", "PrematureBreakdown", "Suspected"}:
        return "#f59e0b"
    return "#0ea5e9"


def _chapter_10_color(pattern: dict[str, Any]) -> str:
    direction = str(pattern.get("direction", ""))
    status = pattern.get("status")
    if direction.startswith("bearish") or status == "Breakdown":
        return "#be123c"
    if direction.startswith("bullish") or status == "Breakout":
        return "#047857"
    if status in {"Candidate", "Observed", "UpsideBreakout", "OppositeBreak"}:
        return "#d97706"
    return "#334155"


def _chapter_11_color(pattern: dict[str, Any]) -> str:
    direction = str(pattern.get("direction", ""))
    status = pattern.get("status")
    if status in {"FailedBreakout", "FailedBreakdown", "Stale"}:
        return "#d97706"
    if direction.startswith("bearish") or status == "Breakdown":
        return "#b91c1c"
    if direction.startswith("bullish") or status == "Breakout":
        return "#047857"
    if status == "Candidate":
        return "#2563eb"
    return "#475569"


def _chapter_12_color(pattern: dict[str, Any]) -> str:
    name = pattern.get("pattern")
    direction = str(pattern.get("direction", ""))
    if name in {"ExhaustionGap", "IslandReversal"}:
        return "#be123c" if direction.startswith("bearish") else "#047857"
    if name == "RunawayGap":
        return "#7c3aed"
    if name == "BreakawayGap":
        return "#2563eb" if direction.startswith("bullish") else "#dc2626"
    return "#64748b"


def _chapter_13_color(zone: dict[str, Any]) -> str:
    if zone.get("pattern") == "RoundNumberZone":
        return "#64748b"
    role = zone.get("role")
    if role == "support":
        return "#047857"
    if role == "resistance":
        return "#be123c"
    return "#7c3aed"


def _chapter_14_color(pattern: dict[str, Any]) -> str:
    status = str(pattern.get("status", ""))
    direction = str(pattern.get("direction", ""))
    kind = str(pattern.get("kind", ""))
    if status in {"DecisiveBreak", "BorderlineBreak", "PullbackToBrokenLine"} and direction == "bearish_warning":
        return "#be123c"
    if status in {"DecisiveBreak", "BorderlineBreak", "PullbackToBrokenLine"} and direction == "bullish_warning":
        return "#047857"
    if status in {"InnerLineBreak", "ShakeoutWarning", "ReturnLineFailure", "FanLinesDeveloping"}:
        return "#d97706"
    if direction == "bullish" or direction == "bullish_acceleration" or kind == "uptrend":
        return "#2563eb"
    if direction == "bearish" or direction == "bearish_acceleration" or kind == "downtrend":
        return "#7c3aed"
    return "#475569"


def _chapter_15_color(pattern: dict[str, Any]) -> str:
    status = str(pattern.get("status", ""))
    direction = str(pattern.get("direction", ""))
    kind = str(pattern.get("kind", ""))
    if status == "MajorTrendlineBreak" and direction == "bearish_major_warning":
        return "#991b1b"
    if status == "MajorBearTrendlineBreakWarning" or direction == "bullish_major_warning":
        return "#047857"
    if kind == "major_uptrend":
        return "#1d4ed8"
    if kind == "major_downtrend":
        return "#7c2d12"
    return "#334155"


def _rgba(hex_color: str, alpha: float) -> str:
    clean = hex_color.lstrip("#")
    if len(clean) != 6:
        return f"rgba(100, 116, 139, {alpha:.2f})"
    red = int(clean[0:2], 16)
    green = int(clean[2:4], 16)
    blue = int(clean[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha:.2f})"


def _safe_float(value: object) -> float | None:
    try:
        output = float(value)
    except Exception:
        return None
    return output if pd.notna(output) else None
