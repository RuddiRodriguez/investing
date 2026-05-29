"""Streamlit web dashboard for all trader agents.
Run with:
    cd ai_trading_system
    streamlit run trader_dashboard.py
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from ingestion.db import get_connection, init_db
from mark_to_market.repositories import get_open_holdings
from trader.trader_agent import get_trader_status

st.set_page_config(
    page_title="AI Trading Dashboard",
    page_icon="📈",
    layout="wide",
)

init_db()


def all_trader_names() -> list[str]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT trader_name
        FROM trader_profiles
        WHERE status = 'running'
        ORDER BY trader_name ASC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    return [row["trader_name"] for row in rows]


def get_portfolio_history(trader_name: str) -> pd.DataFrame:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT created_at, total_portfolio_value
        FROM portfolio_valuation_log
        WHERE trader_name = ?
        ORDER BY created_at ASC
        """,
        (trader_name,),
    )
    history_rows = cursor.fetchall()

    cursor.execute(
        """
        SELECT created_at, initial_cash
        FROM trader_profiles
        WHERE trader_name = ?
        LIMIT 1
        """,
        (trader_name,),
    )
    profile_row = cursor.fetchone()
    conn.close()

    if profile_row is None:
        return pd.DataFrame()

    started_at = pd.to_datetime(profile_row["created_at"], utc=True, errors="coerce")
    initial_cash = float(profile_row["initial_cash"] or 0.0)

    rows = []
    if pd.notna(started_at):
        rows.append(
            {
                "timestamp": started_at,
                "total_portfolio_value": initial_cash,
            }
        )

    for row in history_rows:
        timestamp = pd.to_datetime(row["created_at"], utc=True, errors="coerce")
        if pd.isna(timestamp):
            continue
        rows.append(
            {
                "timestamp": timestamp,
                "total_portfolio_value": float(row["total_portfolio_value"] or 0.0),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    baseline = float(df.iloc[0]["total_portfolio_value"])
    if baseline <= 0:
        baseline = initial_cash if initial_cash > 0 else 1.0

    df["performance_vs_start_pct"] = (df["total_portfolio_value"] / baseline - 1.0) * 100.0
    return df


def get_pipeline_run_history(trader_name: str) -> pd.DataFrame:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT created_at
        FROM trader_run_log
        WHERE trader_name = ?
          AND event_type = 'cycle_started'
        ORDER BY created_at ASC
        """,
        (trader_name,),
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        [
            {
                "timestamp": pd.to_datetime(row["created_at"], utc=True, errors="coerce"),
            }
            for row in rows
        ]
    )
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        return pd.DataFrame()

    df["pipeline_runs_cumulative"] = range(1, len(df) + 1)
    df["run_date"] = df["timestamp"].dt.date
    return df


def get_cycle_decision_history(trader_name: str) -> pd.DataFrame:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT created_at, metadata
        FROM trader_run_log
        WHERE trader_name = ?
          AND event_type = 'cycle_finished'
        ORDER BY created_at ASC
        """,
        (trader_name,),
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame()

    parsed_rows = []
    for row in rows:
        timestamp = pd.to_datetime(row["created_at"], utc=True, errors="coerce")
        if pd.isna(timestamp):
            continue

        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        execution = metadata.get("execution") or {}
        portfolio = metadata.get("portfolio") or {}
        risk = metadata.get("risk") or {}

        parsed_rows.append(
            {
                "timestamp": timestamp,
                "execution_filled": int(execution.get("filled", 0) or 0),
                "execution_partial": int(execution.get("partial_filled", 0) or 0),
                "execution_rejected": int(execution.get("rejected", 0) or 0),
                "execution_skipped": int(execution.get("skipped", 0) or 0),
                "portfolio_opened": int(portfolio.get("opened", 0) or 0),
                "portfolio_watched": int(portfolio.get("watched", 0) or 0),
                "portfolio_reduced": int(portfolio.get("reduced", 0) or 0),
                "portfolio_skipped": int(portfolio.get("skipped", 0) or 0),
                "risk_allowed": int(risk.get("allowed", 0) or 0),
                "risk_rejected": int(risk.get("rejected", 0) or 0),
            }
        )

    if not parsed_rows:
        return pd.DataFrame()

    return pd.DataFrame(parsed_rows).sort_values("timestamp")


def get_buy_candidates() -> pd.DataFrame:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            pp.ticker,
            pp.company_name,
            pp.signal_date,
            ccs.current_price,
            ccs.trend_reading,
            ccs.support_level,
            ccs.resistance_level,
            ccs.breakout_status,
            ccs.volume_confirmation,
            ccs.volume_ratio,
            ccs.entry_quality,
            ccs.buy_trigger,
            ccs.invalid_buy_reason,
            ccs.reason_to_wait,
            ccs.current_price_stop_7_pct,
            ccs.current_price_stop_8_pct,
            ccs.breakout_entry_stop_7_pct,
            ccs.breakout_entry_stop_8_pct,
            ccs.danger_level,
            ccs.chart_decision,
            ccs.llm_chart_reason,
            pp.llm_direction,
            pp.llm_portfolio_action,
            pp.llm_position_size,
            pp.buy_probability,
            pp.portfolio_reason
        FROM portfolio_positions pp
        LEFT JOIN chart_confirmation_signals ccs
            ON pp.ticker = ccs.ticker
           AND pp.signal_date = ccs.signal_date
           AND lower(pp.sector) = lower(ccs.sector)
        WHERE pp.llm_direction = 'long'
        ORDER BY pp.buy_probability DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(row) for row in rows])


def get_open_long_positions_with_buy_probability(trader_name: str) -> pd.DataFrame:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            ph.trader_name,
            ph.ticker,
            ph.company_name,
            ph.direction,
            ph.quantity,
            ph.average_entry_price,
            ph.current_price,
            ph.market_value,
            ph.unrealized_pnl,
            ph.unrealized_pnl_pct,
            COALESCE(pp.buy_probability, 0.0) AS buy_probability,
            pp.portfolio_reason,
            ccs.current_price AS chart_current_price,
            ccs.trend_reading,
            ccs.support_level,
            ccs.resistance_level,
            ccs.breakout_status,
            ccs.volume_confirmation,
            ccs.volume_ratio,
            ccs.entry_quality,
            ccs.buy_trigger,
            ccs.invalid_buy_reason,
            ccs.reason_to_wait,
            ccs.current_price_stop_7_pct,
            ccs.current_price_stop_8_pct,
            ccs.breakout_entry_stop_7_pct,
            ccs.breakout_entry_stop_8_pct,
            ccs.danger_level,
            ccs.chart_decision,
            ccs.llm_chart_reason
        FROM portfolio_holdings ph
        LEFT JOIN portfolio_positions pp
            ON ph.ticker = pp.ticker
        LEFT JOIN chart_confirmation_signals ccs
            ON pp.ticker = ccs.ticker
           AND pp.signal_date = ccs.signal_date
           AND lower(pp.sector) = lower(ccs.sector)
        WHERE ph.direction = 'long'
          AND ph.trader_name = ?
        ORDER BY pp.buy_probability DESC
        """,
        (trader_name,),
    )
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(row) for row in rows])


def get_chart_confirmation_rows() -> pd.DataFrame:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            ticker,
            company_name,
            signal_date,
            open_price,
            current_price,
            high_price,
            low_price,
            latest_volume,
            trend_reading,
            trend_status,
            base_status,
            support_level,
            resistance_level,
            breakout_status,
            volume_confirmation,
            volume_ratio,
            entry_quality,
            extension_pct,
            buy_trigger,
            invalid_buy_reason,
            reason_to_wait,
            current_price_stop_7_pct,
            current_price_stop_8_pct,
            breakout_entry_stop_7_pct,
            breakout_entry_stop_8_pct,
            danger_level,
            sell_signal,
            chart_decision,
            chart_score,
            chart_confidence,
            llm_chart_reason,
            chart_flags
        FROM chart_confirmation_signals
        ORDER BY signal_date DESC, chart_score DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(row) for row in rows])


def render_trader(trader_name: str) -> None:
    result = get_trader_status(trader_name)
    holdings = get_open_holdings(trader_name)

    if result.get("status") == "not_found":
        st.warning(f"Trader '{trader_name}' not found.")
        return

    status = result.get("status", "-")
    profile = result.get("profile") or {}
    state = result.get("portfolio_state") or {}

    initial = profile.get("initial_cash") or 0
    total = state.get("total_portfolio_value") or 0
    cash = state.get("cash") or 0
    invested = state.get("invested_value") or 0
    pnl = total - initial
    pnl_pct = (pnl / initial * 100) if initial else 0

    status_color = "🟢" if status == "running" else "🔴"
    st.subheader(f"{status_color} {trader_name}  —  {profile.get('profile_type', '').upper()}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Value", f"${total:,.2f}", f"{pnl:+,.2f} ({pnl_pct:+.2f}%)")
    c2.metric("Cash", f"${cash:,.2f}")
    c3.metric("Invested", f"${invested:,.2f}")
    c4.metric("Open Positions", state.get("open_positions_count", 0))
    c5.metric("Initial Cash", f"${initial:,.2f}")

    if holdings:
        rows = [
            {
                "Ticker": h.ticker,
                "Company": h.company_name,
                "Direction": h.direction,
                "Qty": round(h.quantity, 4),
                "Avg Entry": round(h.average_entry_price, 4),
                "Price": round(h.current_price, 4),
                "Mkt Value": round(h.market_value, 2),
                "Unreal. P&L": round(h.unrealized_pnl, 2),
                "P&L %": round(h.unrealized_pnl_pct, 2),
                "Size": round(h.position_size, 4),
            }
            for h in holdings
        ]
        df = pd.DataFrame(rows)

        def color_pnl(val):
            color = "green" if val > 0 else ("red" if val < 0 else "gray")
            return f"color: {color}"

        styled = df.style.map(color_pnl, subset=["Unreal. P&L", "P&L %"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.caption("No open positions.")

    long_positions_df = get_open_long_positions_with_buy_probability(trader_name)
    if not long_positions_df.empty:
        display_long_positions = long_positions_df.copy()
        display_long_positions["buy_probability_pct"] = (
            display_long_positions["buy_probability"] * 100.0
        ).round(1)
        display_long_positions["unrealized_pnl_pct"] = display_long_positions["unrealized_pnl_pct"].round(2)
        st.caption("Open long positions ranked by buy probability")
        st.dataframe(
            display_long_positions[
                [
                    "ticker",
                    "company_name",
                    "chart_decision",
                    "breakout_status",
                    "volume_confirmation",
                    "volume_ratio",
                    "entry_quality",
                    "buy_trigger",
                    "invalid_buy_reason",
                    "reason_to_wait",
                    "current_price_stop_7_pct",
                    "current_price_stop_8_pct",
                    "breakout_entry_stop_7_pct",
                    "breakout_entry_stop_8_pct",
                    "resistance_level",
                    "support_level",
                    "danger_level",
                    "quantity",
                    "average_entry_price",
                    "current_price",
                    "market_value",
                    "unrealized_pnl",
                    "unrealized_pnl_pct",
                    "buy_probability_pct",
                    "llm_chart_reason",
                    "portfolio_reason",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    history_df = get_portfolio_history(trader_name)
    pipeline_df = get_pipeline_run_history(trader_name)
    decisions_df = get_cycle_decision_history(trader_name)

    top_left, top_right = st.columns(2)
    with top_left:
        if len(history_df) >= 2:
            st.caption("1) Performance vs start date (%)")
            chart_df = history_df.set_index("timestamp")[["performance_vs_start_pct"]]
            st.line_chart(chart_df, use_container_width=True)
        else:
            st.caption("1) Not enough history for performance chart.")

    with top_right:
        if not pipeline_df.empty:
            st.caption("2) Wakeups per day")
            runs_per_day = (
                pipeline_df.groupby("run_date")
                .size()
                .rename("wakeup_count")
                .to_frame()
            )
            st.bar_chart(runs_per_day, use_container_width=True)
        else:
            st.caption("2) No wakeup history yet.")

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        if not decisions_df.empty:
            st.caption("3) Execution decision counts per wakeup")
            execution_long = decisions_df[[
                "timestamp",
                "execution_filled",
                "execution_partial",
                "execution_rejected",
                "execution_skipped",
            ]].melt(
                id_vars="timestamp",
                var_name="decision_type",
                value_name="count",
            )
            fig_execution = px.bar(
                execution_long,
                x="timestamp",
                y="count",
                color="decision_type",
                barmode="stack",
            )
            fig_execution.update_layout(
                xaxis_title="Wakeup time",
                yaxis_title="Decision count",
                legend_title="Execution decision",
                height=330,
            )
            st.plotly_chart(fig_execution, use_container_width=True)
        else:
            st.caption("3) No execution decision history yet.")

    with bottom_right:
        if not decisions_df.empty:
            st.caption("4) Portfolio/Risk decisions per wakeup")
            portfolio_risk_long = decisions_df[[
                "timestamp",
                "portfolio_opened",
                "portfolio_watched",
                "portfolio_reduced",
                "portfolio_skipped",
                "risk_allowed",
                "risk_rejected",
            ]].melt(
                id_vars="timestamp",
                var_name="decision_type",
                value_name="count",
            )
            fig_portfolio_risk = px.bar(
                portfolio_risk_long,
                x="timestamp",
                y="count",
                color="decision_type",
                barmode="stack",
            )
            fig_portfolio_risk.update_layout(
                xaxis_title="Wakeup time",
                yaxis_title="Decision count",
                legend_title="Portfolio/Risk decision",
                height=330,
            )
            st.plotly_chart(fig_portfolio_risk, use_container_width=True)
        else:
            st.caption("4) No portfolio/risk decision history yet.")

    st.divider()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 AI Trader")
    st.caption("Live portfolio monitor (running traders only)")

    all_names = all_trader_names()
    selected = st.multiselect(
        "Traders to show",
        options=all_names,
        default=all_names,
    )

    refresh_interval = st.slider("Auto-refresh (seconds)", 10, 300, 30, step=10)
    manual_refresh = st.button("🔄 Refresh now", use_container_width=True)

    st.divider()
    refreshed_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.caption(f"Last refresh: {refreshed_at}")

# ── Main ───────────────────────────────────────────────────────────────────
st.title("AI Trading Dashboard")

if not selected:
    st.info("No live traders found. Start a trader to see it in this dashboard.")
else:
    for name in selected:
        render_trader(name)

buy_candidates_df = get_buy_candidates()
if not buy_candidates_df.empty:
    display_candidates = buy_candidates_df.copy()
    display_candidates["buy_probability_pct"] = (display_candidates["buy_probability"] * 100.0).round(1)
    display_candidates["llm_position_size_pct"] = (display_candidates["llm_position_size"] * 100.0).round(1)
    st.subheader("Long Buy Candidates")
    st.dataframe(
        display_candidates[
            [
                "ticker",
                "company_name",
                "signal_date",
                "current_price",
                "trend_reading",
                "support_level",
                "resistance_level",
                "chart_decision",
                "breakout_status",
                "volume_confirmation",
                "volume_ratio",
                "entry_quality",
                "buy_trigger",
                "invalid_buy_reason",
                "reason_to_wait",
                "current_price_stop_7_pct",
                "current_price_stop_8_pct",
                "breakout_entry_stop_7_pct",
                "breakout_entry_stop_8_pct",
                "danger_level",
                "llm_portfolio_action",
                "llm_position_size_pct",
                "buy_probability_pct",
                "llm_chart_reason",
                "portfolio_reason",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

chart_confirmation_df = get_chart_confirmation_rows()
if not chart_confirmation_df.empty:
    st.subheader("Chart Confirmation")
    st.dataframe(chart_confirmation_df, use_container_width=True, hide_index=True)

# Auto-refresh
time.sleep(refresh_interval)
st.rerun()
