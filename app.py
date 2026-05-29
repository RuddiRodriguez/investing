"""Streamlit app for a medium-term ETF allocation model."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scripts.stochastic_pipeline import run_stochastic_analysis
from src.ai_interpreter import interpret_results_with_openai, summarize_daily_finance_news_with_openai
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
RESOURCE_CATEGORIES = {
    "Market Data And News": [
        {
            "name": "Investing.com",
            "url": "https://www.investing.com/",
            "description": "Live markets, economic calendar, rates, commodities, and broad news coverage.",
        },
        {
            "name": "Yahoo Finance",
            "url": "https://finance.yahoo.com/",
            "description": "Quotes, charts, ETF holdings snapshots, and headline market news.",
        },
        {
            "name": "Trading Economics",
            "url": "https://tradingeconomics.com/",
            "description": "Macro indicators, country data, central bank updates, and economic calendars.",
        },
    ],
    "ETF Research": [
        {
            "name": "Morningstar ETFs",
            "url": "https://www.morningstar.com/topics/etfs",
            "description": "ETF research, category analysis, ratings, and fund commentary.",
        },
        {
            "name": "JustETF",
            "url": "https://www.justetf.com/",
            "description": "Strong ETF screener for European investors, including UCITS funds and comparisons.",
        },
        {
            "name": "ETF.com",
            "url": "https://www.etf.com/",
            "description": "ETF education, industry news, fund data, and screening tools.",
        },
    ],
    "Official Macro Data": [
        {
            "name": "FRED",
            "url": "https://fred.stlouisfed.org/",
            "description": "Official-style macro data portal for inflation, rates, employment, and growth series.",
        },
        {
            "name": "Eurostat",
            "url": "https://ec.europa.eu/eurostat",
            "description": "European statistics on inflation, GDP, labor markets, and household data.",
        },
    ],
    "Investor Protection And Checks": [
        {
            "name": "Investor.gov",
            "url": "https://www.investor.gov/",
            "description": "U.S. SEC investor education, fraud warnings, and basic investment explainers.",
        },
        {
            "name": "FINRA BrokerCheck",
            "url": "https://brokercheck.finra.org/",
            "description": "Check whether a broker or adviser is registered and review disclosures.",
        },
    ],
}


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def _load_prices(data_source: str, tickers: list[str], start_date: pd.Timestamp, uploaded_file):
    if data_source == "Yahoo Finance":
        return download_prices(tickers, start_date.strftime("%Y-%m-%d"))
    if uploaded_file is None:
        raise ValueError("Upload a CSV file to use manual data input.")
    return parse_uploaded_csv(uploaded_file.getvalue())


def _load_single_price_series(
    data_source: str,
    ticker: str,
    start_date: pd.Timestamp,
    uploaded_file,
    uploaded_column: str | None,
) -> pd.DataFrame:
    if data_source == "Yahoo Finance":
        prices = download_prices([ticker.strip().upper()], start_date.strftime("%Y-%m-%d"))
        column = prices.columns[0]
        return prices[[column]].rename(columns={column: ticker.strip().upper()})

    if uploaded_file is None:
        raise ValueError("Upload a CSV file or switch to Yahoo Finance.")
    prices = parse_uploaded_csv(uploaded_file.getvalue())
    if prices.empty:
        raise ValueError("Uploaded CSV returned no usable price data.")
    column = uploaded_column or str(prices.columns[0])
    if column not in prices.columns:
        raise ValueError(f"Column {column} is not present in the uploaded CSV.")
    return prices[[column]].rename(columns={column: str(column).upper()})


@st.cache_data(show_spinner=False)
def _load_cpi_data(region_code: str) -> pd.DataFrame:
    return load_inflation_series(region_code)


def _build_stochastic_price_chart(result: dict) -> go.Figure:
    history = result["history"].copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["price"],
            mode="lines",
            name="History",
            line=dict(color="#1f2937", width=2),
        )
    )

    for label, color in (("gbm", "#2563eb"), ("garch", "#dc2626"), ("egarch", "#059669")):
        model_result = result[label]
        if label == "gbm":
            cone = model_result["price_cone"].copy()
        else:
            cone = model_result.price_cone.copy()
        cone = cone.reset_index().rename(columns={cone.index.name or "index": "date"})
        if "index" in cone.columns and "date" not in cone.columns:
            cone = cone.rename(columns={"index": "date"})
        cone["date"] = pd.to_datetime(cone["date"], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=cone["date"],
                y=cone["p90"],
                mode="lines",
                line=dict(color=color, width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cone["date"],
                y=cone["p10"],
                mode="lines",
                line=dict(color=color, width=0),
                fill="tonexty",
                fillcolor=f"rgba{(*tuple(int(color[i:i+2], 16) for i in (1, 3, 5)), 0.10)}",
                hoverinfo="skip",
                name=f"{model_result['model_name'] if label == 'gbm' else model_result.model_name} 10-90% band",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cone["date"],
                y=cone["p50"],
                mode="lines",
                name=f"{model_result['model_name'] if label == 'gbm' else model_result.model_name} median",
                line=dict(color=color, width=2, dash="dash" if label == "gbm" else "solid"),
            )
        )

    fig.update_layout(
        title="Stochastic Price Cones",
        xaxis_title="Date",
        yaxis_title="Price",
        height=460,
    )
    return fig


def _build_stochastic_volatility_chart(result: dict) -> go.Figure:
    fig = go.Figure()
    for model_result, color in ((result["garch"], "#dc2626"), (result["egarch"], "#059669")):
        forecast = model_result.volatility_forecast.copy().reset_index().rename(columns={"index": "date"})
        forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=forecast["date"],
                y=forecast["annualized_volatility"],
                mode="lines+markers",
                name=model_result.model_name,
                line=dict(color=color, width=2),
            )
        )

    fig.update_layout(
        title="Forward Volatility Forecast",
        xaxis_title="Date",
        yaxis_title="Annualized volatility",
        yaxis_tickformat=".1%",
        height=360,
    )
    return fig


def main() -> None:
    st.title("ETF Investing Helper")
    st.caption(
        "A simple app to help you review ETFs for a 6 to 12 month horizon "
        "using price strength, long-term direction, and occasional portfolio updates."
    )

    analysis_tab, stochastic_tab, inflation_tab, resources_tab = st.tabs(["ETF Analysis", "Stochastic Models", "Inflation / CPI", "Resources"])

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

    with stochastic_tab:
        st.subheader("Stochastic Process Modeling")
        st.caption(
            "Run a probabilistic market-dynamics view using GBM for price paths and GARCH / EGARCH for volatility clustering and persistence."
        )

        control_col1, control_col2, control_col3, control_col4 = st.columns([1.1, 1.4, 1.0, 1.0])
        with control_col1:
            stochastic_data_source = st.radio(
                "Price source",
                ["Yahoo Finance", "Upload CSV"],
                index=0,
                key="stochastic_data_source",
            )
        with control_col2:
            stochastic_ticker = st.text_input("Ticker", value="SPY", key="stochastic_ticker")
        with control_col3:
            stochastic_start_date = st.date_input(
                "History start",
                value=pd.Timestamp("2018-01-01"),
                key="stochastic_start_date",
            )
        with control_col4:
            horizon_days = st.slider("Forecast days", 5, 90, 30, key="stochastic_horizon_days")

        control_col5, control_col6, control_col7 = st.columns([1.2, 1.0, 1.0])
        with control_col5:
            uploaded_file = st.file_uploader("Upload price CSV", type=["csv"], key="stochastic_uploaded_file")
        uploaded_column = None
        if uploaded_file is not None and stochastic_data_source == "Upload CSV":
            uploaded_prices = parse_uploaded_csv(uploaded_file.getvalue())
            uploaded_column = st.selectbox(
                "Price column",
                options=[str(column) for column in uploaded_prices.columns],
                index=0,
                key="stochastic_uploaded_column",
            )
        with control_col6:
            simulation_paths = st.slider("GBM paths", 200, 5000, 1500, step=100, key="stochastic_paths")
        with control_col7:
            random_seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1, key="stochastic_seed")

        run_stochastic_button = st.button("Run stochastic models", type="primary", key="run_stochastic_button")

        st.markdown(
            """
            **What this tab does**

            1. Fits a geometric Brownian motion view of future price paths.
            2. Fits GARCH(1,1) and EGARCH(1,1) style volatility processes.
            3. Shows forward price cones and volatility forecasts instead of a single point estimate.
            """
        )

        if not run_stochastic_button:
            st.info("Choose a series and click Run stochastic models.")
        else:
            progress_bar = st.progress(0.0)
            progress_status = st.empty()

            def stochastic_progress(step: int, total: int, message: str) -> None:
                progress_bar.progress(step / max(total, 1))
                progress_status.write(message)

            try:
                price_frame = _load_single_price_series(
                    data_source=stochastic_data_source,
                    ticker=stochastic_ticker,
                    start_date=pd.Timestamp(stochastic_start_date),
                    uploaded_file=uploaded_file,
                    uploaded_column=uploaded_column,
                )
                result = run_stochastic_analysis(
                    price_frame,
                    horizon_days=horizon_days,
                    num_paths=simulation_paths,
                    seed=int(random_seed),
                    progress_callback=stochastic_progress,
                )
                progress_bar.progress(1.0)
                progress_status.success("Stochastic model run complete.")
            except Exception as exc:
                progress_status.error(str(exc))
                st.error(str(exc))
                result = None

            if result is not None:
                comparison = result["comparison"].copy()
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                metric_col1.metric("Last price", f"{result['last_price']:.2f}")
                metric_col2.metric("GBM median terminal price", f"{comparison.loc[comparison['model'] == 'GBM', 'terminal_median_price'].iloc[0]:.2f}")
                metric_col3.metric("GARCH terminal vol", f"{result['garch'].volatility_forecast.iloc[-1]['annualized_volatility']:.2%}")
                metric_col4.metric("EGARCH terminal vol", f"{result['egarch'].volatility_forecast.iloc[-1]['annualized_volatility']:.2%}")

                left, right = st.columns([2.2, 1.2])
                left.plotly_chart(_build_stochastic_price_chart(result), use_container_width=True)
                right.plotly_chart(_build_stochastic_volatility_chart(result), use_container_width=True)

                st.subheader("Model Comparison")
                st.dataframe(
                    comparison.style.format(
                        {
                            "annualized_volatility": "{:.2%}",
                            "terminal_median_price": "{:.2f}",
                            "terminal_p10_price": "{:.2f}",
                            "terminal_p90_price": "{:.2f}",
                            "log_likelihood": "{:.2f}",
                        }
                    ),
                    use_container_width=True,
                )

                parameter_rows = [
                    {"model": "GBM", **result["gbm"]["parameters"]},
                    {"model": result["garch"].model_name, **result["garch"].parameters},
                    {"model": result["egarch"].model_name, **result["egarch"].parameters},
                ]
                st.subheader("Estimated Parameters")
                st.dataframe(pd.DataFrame(parameter_rows), use_container_width=True)

                realized_vol = result["log_returns"].rolling(20).std() * (252 ** 0.5)
                realized_vol = realized_vol.dropna().rename("realized_20d_vol")
                if not realized_vol.empty:
                    realized_chart = go.Figure()
                    realized_chart.add_trace(
                        go.Scatter(
                            x=realized_vol.index,
                            y=realized_vol.values,
                            mode="lines",
                            name="Realized 20d volatility",
                            line=dict(color="#1f2937", width=2),
                        )
                    )
                    st.subheader("Recent Realized Volatility")
                    realized_chart.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Annualized volatility",
                        yaxis_tickformat=".1%",
                        height=320,
                    )
                    st.plotly_chart(realized_chart, use_container_width=True)

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

    with resources_tab:
        st.subheader("Useful Investing Resources")
        st.caption(
            "A short list of widely used websites for market data, ETF research, macro data, and investor safety checks."
        )

        st.markdown("### AI Finance News")
        st.caption(
            "Generate a fresh AI summary of the most important finance news for investors. This runs only when you click the button."
        )
        news_focus = st.text_input(
            "Optional focus",
            value="ETFs, inflation, interest rates, and global markets",
            key="resources_ai_focus",
            help="Example: European ETFs, technology stocks, bonds, or central banks.",
        )
        summarize_news = st.button("Summarize Today's Finance News", key="resources_ai_news_button")

        if summarize_news:
            try:
                with st.spinner("Summarizing current finance news..."):
                    news_summary = summarize_daily_finance_news_with_openai(
                        focus_note=news_focus,
                    )
                st.markdown(news_summary)
            except Exception as exc:
                st.error(str(exc))

        st.markdown(
            """
            Use these sites to cross-check ideas before investing:

            - compare the same ETF across more than one source
            - verify macro claims with official data when possible
            - check fees, holdings, and region/currency details before buying
            - verify brokers and avoid acting on unverified social media tips
            """
        )

        for category, resources in RESOURCE_CATEGORIES.items():
            st.markdown(f"### {category}")
            for resource in resources:
                st.markdown(
                    f"- [{resource['name']}]({resource['url']})  \n"
                    f"  {resource['description']}"
                )


if __name__ == "__main__":
    main()
