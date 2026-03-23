"""Streamlit app for a medium-term ETF allocation model."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ai_interpreter import interpret_results_with_openai
from src.data import download_prices, load_inflation_series, parse_uploaded_csv
from src.strategy import StrategyConfig, backtest


st.set_page_config(
    page_title="ETF Rotation Lab",
    page_icon="📈",
    layout="wide",
)


DEFAULT_TICKERS = "VWCE.DE, EUNL.DE, SPYI.DE, VAGF.DE, EGLN.L"
INFLATION_OPTIONS = {
    "United States": {"code": "US", "index_label": "CPI", "source": "FRED"},
    "Netherlands": {"code": "NL", "index_label": "HICP", "source": "Eurostat"},
    "Spain": {"code": "ES", "index_label": "HICP", "source": "Eurostat"},
}


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def _load_prices(data_source: str, tickers: list[str], start_date: pd.Timestamp, uploaded_file):
    if data_source == "Yahoo Finance":
        return download_prices(tickers, start_date.strftime("%Y-%m-%d"))
    if uploaded_file is None:
        raise ValueError("Upload a CSV file to use manual data input.")
    return parse_uploaded_csv(uploaded_file.getvalue())


@st.cache_data(show_spinner=False)
def _load_cpi_data(region_code: str) -> pd.DataFrame:
    return load_inflation_series(region_code)


def main() -> None:
    st.title("ETF Investing Helper")
    st.caption(
        "A simple app to help you review ETFs for a 6 to 12 month horizon "
        "using price strength, long-term direction, and occasional portfolio updates."
    )

    analysis_tab, inflation_tab = st.tabs(["ETF Analysis", "Inflation / CPI"])

    with analysis_tab:
        with st.sidebar:
            st.header("Choose Your Data")
            data_source = st.radio("Where should prices come from?", ["Yahoo Finance", "Upload CSV"], index=0)
            tickers = st.text_area(
                "ETF codes",
                value=DEFAULT_TICKERS,
                help="Write ETF codes separated by commas. Example: VWCE.DE, EUNL.DE, SPYI.DE",
            )
            start_date = st.date_input("How far back should we look?", value=pd.Timestamp("2018-01-01"))
            uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

            st.header("How The Check Works")
            lookback_months = st.slider(
                "Lookback period in months",
                3,
                12,
                6,
                help="The app checks which ETFs have done best over this period.",
            )
            moving_average_days = st.slider(
                "Long-term trend in days",
                100,
                250,
                200,
                step=10,
                help="This helps avoid ETFs that are in a weak long-term trend.",
            )
            top_n = st.slider(
                "How many ETFs to keep",
                1,
                5,
                3,
                help="The app keeps the strongest ETFs that still look healthy.",
            )
            rebalance_frequency_label = st.selectbox(
                "How often to review the portfolio",
                ["Monthly", "Quarterly"],
                index=1,
            )
            rebalance_frequency = "M" if rebalance_frequency_label == "Monthly" else "Q"

            st.header("Optional AI Explanation")
            enable_ai = st.checkbox("Use OpenAI to explain the results", value=False)
            openai_model = st.text_input("OpenAI model", value="gpt-4.1", disabled=not enable_ai)
            use_web_search = st.checkbox(
                "Include recent market news",
                value=True,
                disabled=not enable_ai,
                help="The AI can look at recent market news and big world events before giving its view.",
            )
            user_note = st.text_area(
                "Extra note for the AI",
                value="I am a beginner investor. My horizon is 6 to 12 months. Please explain the result in simple language.",
                disabled=not enable_ai,
            )

            run_button = st.button("Analyze ETFs", type="primary", use_container_width=True)

        st.markdown(
            """
            **What this app does**

            1. It checks which ETFs have been strongest over the past few months.
            2. It removes ETFs that are in a weak long-term direction.
            3. It keeps the strongest remaining ETFs.
            4. If nothing looks strong enough, it suggests staying in cash.

            This is a learning and research tool. It is not personal financial advice.
            """
        )

        if not run_button:
            st.info("Choose your ETFs and settings in the sidebar, then click Analyze ETFs.")
        else:
            try:
                ticker_list = [ticker.strip().upper() for ticker in tickers.split(",") if ticker.strip()]
                if len(ticker_list) < 2:
                    raise ValueError("Please enter at least two ETF codes.")

                config = StrategyConfig(
                    lookback_months=lookback_months,
                    moving_average_days=moving_average_days,
                    top_n=top_n,
                    rebalance_frequency=rebalance_frequency,
                )
                prices = _load_prices(data_source, ticker_list, pd.Timestamp(start_date), uploaded_file)
                results = backtest(prices, config)
            except Exception as exc:
                st.error(str(exc))
                results = None

            if results is not None:
                metrics = results["metrics"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Total growth", _format_pct(metrics["Total Return"]))
                col2.metric("Average yearly growth", _format_pct(metrics["Annualized Return"]))
                col3.metric("Worst drop", _format_pct(metrics["Max Drawdown"]))

                col4, col5, col6 = st.columns(3)
                col4.metric("How much it moved", _format_pct(metrics["Annualized Volatility"]))
                col5.metric("Risk-adjusted score", f"{metrics['Sharpe (rf=0)']:.2f}")
                col6.metric("Positive days", _format_pct(metrics["Win Rate"]))

                equity_curve = results["equity_curve"]
                drawdown = results["drawdown"]
                rebalance_weights = results["rebalance_weights"]
                latest_allocation = results["latest_allocation"]

                performance_chart = go.Figure()
                performance_chart.add_trace(
                    go.Scatter(x=equity_curve.index, y=equity_curve.values, mode="lines", name="Strategy")
                )
                performance_chart.update_layout(
                    title="Portfolio Growth Over Time",
                    xaxis_title="Date",
                    yaxis_title="Portfolio value (started at 1.0)",
                    height=420,
                )

                drawdown_chart = go.Figure()
                drawdown_chart.add_trace(
                    go.Scatter(x=drawdown.index, y=drawdown.values, fill="tozeroy", mode="lines", name="Drawdown")
                )
                drawdown_chart.update_layout(
                    title="Drops From Previous Peaks",
                    xaxis_title="Date",
                    yaxis_title="Drop",
                    height=320,
                )

                alloc_df = latest_allocation.rename("weight").reset_index()
                alloc_df.columns = ["asset", "weight"]
                allocation_chart = px.pie(alloc_df, names="asset", values="weight", title="Current Suggested Mix")

                left, right = st.columns([2, 1])
                left.plotly_chart(performance_chart, use_container_width=True)
                right.plotly_chart(allocation_chart, use_container_width=True)
                st.plotly_chart(drawdown_chart, use_container_width=True)

                st.subheader("Current Suggested Allocation")
                st.dataframe((latest_allocation * 100).rename("weight_pct").to_frame().style.format("{:.2f}%"))

                st.subheader("Why These ETFs Were Chosen")
                signal_frame = results["latest_signal_frame"].copy()
                st.dataframe(
                    signal_frame.style.format(
                        {
                            "price": "{:.2f}",
                            "momentum": "{:.2%}",
                            "moving_average": "{:.2f}",
                        }
                    ),
                    use_container_width=True,
                )

                st.subheader("Past Portfolio Changes")
                st.dataframe((rebalance_weights * 100).style.format("{:.2f}%"), use_container_width=True)

                if enable_ai:
                    st.subheader("AI Explanation")
                    st.caption(
                        "Optional OpenAI explanation that combines the app result with recent market news and world events. "
                        "Use it as guidance only."
                    )
                    try:
                        with st.spinner("Generating AI interpretation..."):
                            interpretation = interpret_results_with_openai(
                                results=results,
                                user_note=user_note,
                                model=openai_model,
                                use_web_search=use_web_search,
                            )
                        if interpretation:
                            st.markdown(interpretation)
                        else:
                            st.warning("The AI returned an empty explanation.")
                    except Exception as exc:
                        st.error(str(exc))

                st.subheader("How To Read This")
                st.markdown(
                    """
                    - Start with a small list of ETFs you understand.
                    - Quarterly review is usually easier for beginners than monthly review.
                    - Adding a bond ETF can make the portfolio less jumpy.
                    - Use this as a guide, not as an automatic buy and sell machine.
                    - If you enable AI, double-check any claims about news or the economy before acting.
                    """
                )

    with inflation_tab:
        st.subheader("Inflation")
        selected_region = st.selectbox("Country", list(INFLATION_OPTIONS.keys()), index=0)
        region_config = INFLATION_OPTIONS[selected_region]
        st.caption(
            f"A simple view of {selected_region} inflation using {region_config['index_label']} data from {region_config['source']}."
        )

        try:
            cpi = _load_cpi_data(region_config["code"])
            latest = cpi.iloc[-1]
            prior_year = cpi["inflation_yoy"].dropna().iloc[-1]

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric(f"Latest {region_config['index_label']} index", f"{latest['cpi_index']:.1f}")
            metric_col2.metric("Year-over-year inflation", _format_pct(prior_year))
            metric_col3.metric("Latest reading", latest["date"].strftime("%B %Y"))

            cpi_chart = go.Figure()
            cpi_chart.add_trace(
                go.Scatter(x=cpi["date"], y=cpi["cpi_index"], mode="lines", name="CPI index")
            )
            cpi_chart.update_layout(
                title=f"{region_config['index_label']} Level",
                xaxis_title="Date",
                yaxis_title="Index level",
                height=360,
            )

            inflation_chart = go.Figure()
            inflation_chart.add_trace(
                go.Scatter(x=cpi["date"], y=cpi["inflation_yoy"], mode="lines", name="Inflation YoY")
            )
            inflation_chart.update_layout(
                title="Inflation Rate (Year over Year)",
                xaxis_title="Date",
                yaxis_title="Inflation",
                yaxis_tickformat=".1%",
                height=360,
            )

            left, right = st.columns(2)
            left.plotly_chart(cpi_chart, use_container_width=True)
            right.plotly_chart(inflation_chart, use_container_width=True)

            st.markdown(
                f"""
                **What this means**

                {region_config['index_label']} measures how the average price of everyday goods and services changes over time.
                When this index rises quickly, inflation is increasing and purchasing power is falling.

                The latest data shown here is for **{selected_region}** in **{latest["date"].strftime("%B %Y")}**.
                Based on that reading, annual inflation was **{prior_year:.1%}**.

                This is useful for investors because inflation can influence interest rates, bond prices, stock valuations,
                and how much real return you keep after prices rise.
                """
            )

            st.dataframe(
                cpi.tail(24).assign(
                    inflation_yoy=lambda frame: frame["inflation_yoy"].map(
                        lambda value: f"{value:.2%}" if pd.notna(value) else "-"
                    )
                ),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.error(f"Unable to load inflation data right now: {exc}")


if __name__ == "__main__":
    main()
