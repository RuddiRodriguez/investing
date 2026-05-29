from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from market_forecasting_engine.calendar import summarize_calendar_alignment
from market_forecasting_engine.chapter_19_validation import apply_chapter_19_validation
from market_forecasting_engine.chapter_20_selection import apply_chapter_20_ticker_suitability
from market_forecasting_engine.chapter_21_chart_selection import apply_chapter_21_chart_selection
from market_forecasting_engine.chapter_23_30_trade_risk import apply_chapter_23_30_trade_risk_plan
from market_forecasting_engine.chapter_31_42_portfolio_risk import apply_chapter_31_42_portfolio_capital_risk
from market_forecasting_engine.chapter_39_43_discipline import apply_chapter_39_43_discipline_governance
from market_forecasting_engine.data import load_event_indicators, load_indicator_csv, normalize_price_frame
from market_forecasting_engine.data_manifest import build_data_manifest, point_in_time_policy_summary
from market_forecasting_engine.data_providers import DataRequest, load_prices_with_provider
from market_forecasting_engine.data_quality import build_data_quality_report
from market_forecasting_engine.data_store import MarketDataStore, request_key
from market_forecasting_engine.alternative_data import AlternativeNewsRequest, collect_alternative_news_features
from market_forecasting_engine.governance import write_audit_bundle
from market_forecasting_engine.panel import (
    build_cross_sectional_panel_features,
    build_panel_frame,
    load_universe_csv,
    parse_universe_tickers,
    rank_universe_from_panel,
    select_ticker_panel_features,
    summarize_universe,
)
from market_forecasting_engine.openai_models import DEFAULT_OPENAI_MODEL, DEFAULT_REASONING_EFFORT
from market_forecasting_engine.pipeline import ForecastingEngine
from market_forecasting_engine.plots import write_plot_artifacts
from market_forecasting_engine.schema import ForecastConfig
from market_forecasting_engine.security_master import load_security_master, resolve_security_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the automated stock forecasting engine.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, for example AAPL.")
    parser.add_argument("--csv", help="Optional local CSV file with historical OHLCV data.")
    parser.add_argument("--provider", default=None, help="Data provider: yahoo, csv, polygon, alpha-vantage, or nasdaq-data-link.")
    parser.add_argument("--start", default="2020-01-01", help="Yahoo Finance start date when --csv is not provided.")
    parser.add_argument("--end", default=None, help="Yahoo Finance end date when --csv is not provided.")
    parser.add_argument("--interval", default="1d", help="Provider bar interval, for example 1d, 1h, 15m, or 5m.")
    parser.add_argument(
        "--as-of",
        default=None,
        help="Optional forecast cutoff timestamp, for example 2026-05-29 or 2026-05-29T15:00:00.",
    )
    parser.add_argument("--adjustment-policy", default="auto_adjust", help="Price adjustment policy recorded in governance metadata.")
    parser.add_argument("--target-column", default="close", help="Price column to forecast.")
    parser.add_argument("--horizons", default="1,5,30", help="Comma-separated forecast horizons in trading days.")
    parser.add_argument("--selection-metric", default="mae", help="mae, rmse, bic, aic, directional_accuracy, or composite.")
    parser.add_argument("--confidence-level", type=float, default=0.80, help="Confidence interval level.")
    parser.add_argument("--purge-window", type=int, default=None, help="Rows to purge before each validation window. Defaults to horizon.")
    parser.add_argument("--embargo-window", type=int, default=0, help="Extra rows to embargo before each validation window.")
    parser.add_argument("--final-holdout-fraction", type=float, default=0.15, help="Final untouched holdout fraction for post-selection diagnostics.")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0, help="Backtest transaction cost in basis points per signal change.")
    parser.add_argument("--output-dir", default=None, help="Directory for forecast_report.json and governance artifacts.")
    parser.add_argument("--chart-scale", choices=("log", "linear"), default="log", help="Technical chart price scale.")
    parser.add_argument("--data-dir", default=None, help="Optional local data lake root for raw, normalized, panel, and metadata files.")
    parser.add_argument("--no-data-cache", action="store_true", help="Disable reading cached normalized provider data.")
    parser.add_argument("--refresh-data-cache", action="store_true", help="Fetch provider data even when a normalized cache file exists.")
    parser.add_argument(
        "--allow-live-cache",
        action="store_true",
        help="Allow cached provider data when --end is omitted. By default live Yahoo runs refresh on every execution.",
    )
    parser.add_argument(
        "--forecast-log",
        default=None,
        help="Optional CSV path where one compact forecast row per horizon is appended after each run.",
    )
    parser.add_argument("--no-forecast-log", action="store_true", help="Disable the default output-dir forecast log.")
    parser.add_argument("--calendar", default="XNYS", help="Exchange calendar used for data-quality session checks.")
    parser.add_argument("--security-master-csv", default=None, help="Optional security master CSV with ticker metadata.")
    parser.add_argument("--no-lightgbm", action="store_true", help="Disable optional LightGBM candidate.")
    parser.add_argument("--no-statistical-models", action="store_true", help="Disable optional ARIMA/SARIMA/GARCH/VAR candidates.")
    parser.add_argument("--include-lstm", action="store_true", help="Enable the optional LSTM candidate. Slower because it trains during validation.")
    parser.add_argument("--search-level", choices=("fast", "expanded"), default="fast", help="Candidate tuning breadth.")
    parser.add_argument("--tune", choices=("fixed", "optuna"), default="fixed", help="Candidate tuning mode. `optuna` adds nested time-series optimization candidates.")
    parser.add_argument("--optuna-trials", type=int, default=25, help="Optuna trials per tuned candidate fit.")
    parser.add_argument("--optuna-timeout", type=int, default=None, help="Optional Optuna timeout in seconds per tuned candidate fit.")
    parser.add_argument("--optuna-inner-splits", type=int, default=3, help="Inner walk-forward splits used inside each Optuna objective.")
    parser.add_argument(
        "--optuna-families",
        default="lightgbm,elastic_net,random_forest,gradient_boosting",
        help="Comma-separated Optuna model families to tune.",
    )
    parser.add_argument(
        "--tactical-profile",
        choices=("short_term", "intermediate", "long_term"),
        default="intermediate",
        help="Chapter 18 tactical profile for stops, reward/risk gates, and holding horizon.",
    )
    parser.add_argument("--enable-llm-review", action="store_true", help="Enable the optional governed OpenAI tactical reviewer.")
    parser.add_argument("--llm-provider", choices=("openai",), default="openai", help="LLM provider for optional tactical review.")
    parser.add_argument(
        "--llm-model",
        default=None,
        help=f"Optional OpenAI model name. Defaults to OPENAI_MODEL or {DEFAULT_OPENAI_MODEL}.",
    )
    parser.add_argument("--llm-temperature", type=float, default=0.0, help="Temperature for optional LLM review.")
    parser.add_argument(
        "--llm-reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        help="Reasoning effort for Responses API LLM calls. Default: none.",
    )
    parser.add_argument("--llm-timeout", type=int, default=30, help="Timeout in seconds for optional LLM review.")
    parser.add_argument("--llm-env-file", default=None, help="Optional .env file containing OPENAI_API_KEY and optionally OPENAI_MODEL.")
    parser.add_argument("--enable-bayesian-heavy", action="store_true", help="Enable optional PyMC/MCMC Bayesian diagnostics. Disabled by default.")
    parser.add_argument("--bayesian-mcmc-draws", type=int, default=300, help="MCMC draws for optional heavy Bayesian diagnostics.")
    parser.add_argument("--bayesian-mcmc-tune", type=int, default=300, help="MCMC tuning draws for optional heavy Bayesian diagnostics.")
    parser.add_argument("--benchmark", default=None, help="Optional Yahoo benchmark ticker, for example SPY.")
    parser.add_argument("--sector", default=None, help="Optional Yahoo sector/index ticker, for example XLK.")
    parser.add_argument("--vix", default=None, help="Optional Yahoo volatility ticker, for example ^VIX.")
    parser.add_argument("--rates-csv", default=None, help="Optional rates CSV with date column first and numeric indicators.")
    parser.add_argument("--macro-csv", default=None, help="Optional macro CSV with date column first and numeric indicators.")
    parser.add_argument("--events-csv", default=None, help="Optional corporate events CSV with date column first.")
    parser.add_argument("--macro-release-lag-days", type=int, default=0, help="Business-day lag before macro CSV observations are model-available.")
    parser.add_argument("--rates-release-lag-days", type=int, default=0, help="Business-day lag before rates CSV observations are model-available.")
    parser.add_argument("--events-release-lag-days", type=int, default=0, help="Business-day lag before event CSV rows are model-available.")
    parser.add_argument("--enable-alt-news", action="store_true", help="Download/scrape alternative news and add rolling sentiment features.")
    parser.add_argument("--alt-news-provider", default="yahoo_rss", help="Alternative news provider: yahoo_rss, yahoo_news, or openai_web.")
    parser.add_argument("--alt-news-lookback-days", type=int, default=30, help="Lookback window for downloaded/scraped news.")
    parser.add_argument("--alt-news-max-items", type=int, default=40, help="Maximum downloaded/scraped news items.")
    parser.add_argument(
        "--alt-news-sentiment-mode",
        choices=("lexicon", "llm", "hybrid"),
        default="lexicon",
        help="News sentiment scoring mode. Hybrid blends deterministic scoring with optional LLM classification.",
    )
    parser.add_argument("--universe-csv", default=None, help="Optional universe CSV with ticker/symbol column for panel metadata.")
    parser.add_argument("--universe-tickers", default=None, help="Optional comma-separated universe tickers for panel metadata.")
    parser.add_argument("--build-panel", action="store_true", help="Fetch and store a date/ticker panel for the supplied universe.")
    parser.add_argument("--rank-universe", action="store_true", help="Rank supplied universe members using market-action and cross-sectional evidence.")
    parser.add_argument("--universe-top-n", type=int, default=25, help="Maximum universe ranking rows to include.")
    args = parser.parse_args()

    horizons = tuple(int(value.strip()) for value in args.horizons.split(",") if value.strip())
    forecast_interval_minutes = _interval_to_minutes(args.interval)
    config = ForecastConfig(
        ticker=args.ticker,
        horizons=horizons,
        target_column=args.target_column.lower(),
        selection_metric=args.selection_metric,
        confidence_level=args.confidence_level,
        purge_window=args.purge_window,
        embargo_window=args.embargo_window,
        final_holdout_fraction=args.final_holdout_fraction,
        transaction_cost_bps=args.transaction_cost_bps,
        include_lightgbm=not args.no_lightgbm,
        include_statistical_models=not args.no_statistical_models,
        include_lstm=args.include_lstm,
        search_level=args.search_level,
        tuning_mode=args.tune,
        optuna_trials=args.optuna_trials,
        optuna_timeout_seconds=args.optuna_timeout,
        optuna_inner_splits=args.optuna_inner_splits,
        optuna_families=tuple(value.strip() for value in args.optuna_families.split(",") if value.strip()),
        tactical_profile=args.tactical_profile,
        enable_llm_review=args.enable_llm_review,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_reasoning_effort=args.llm_reasoning_effort,
        llm_timeout_seconds=args.llm_timeout,
        llm_env_file=args.llm_env_file,
        enable_bayesian_heavy=args.enable_bayesian_heavy,
        bayesian_mcmc_draws=args.bayesian_mcmc_draws,
        bayesian_mcmc_tune=args.bayesian_mcmc_tune,
        forecast_interval=args.interval,
        forecast_interval_minutes=forecast_interval_minutes,
    )

    output_dir = Path(args.output_dir) if args.output_dir else None
    data_store = _data_store(args.data_dir, output_dir)
    provider_name = (args.provider or ("csv" if args.csv else "yahoo")).lower()
    if args.csv:
        provider_name = "csv"
    request = DataRequest(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        target_column=config.target_column,
        interval=args.interval,
        adjustment_policy=args.adjustment_policy,
        source_path=args.csv,
    )
    primary_result = load_prices_with_provider(
        provider_name,
        request,
        store=data_store,
        use_cache=not args.no_data_cache,
        refresh_cache=_refresh_cache_for_request(args, provider_name),
    )
    prices = primary_result.frame
    prices = _filter_prices_as_of(prices, args.as_of, target_column=config.target_column)
    data_artifacts: dict[str, object] = {"primary": primary_result.metadata.get("artifacts", {})}
    context_sources: list[dict[str, object]] = []
    indicator_sources: list[dict[str, object]] = []
    event_sources: list[dict[str, object]] = []
    alternative_sources: list[dict[str, object]] = []

    yahoo_context = {}
    if args.benchmark:
        yahoo_context["benchmark"] = args.benchmark
    if args.sector:
        yahoo_context["sector"] = args.sector
    if args.vix:
        yahoo_context["volatility"] = args.vix

    for label, ticker in yahoo_context.items():
        context_result = load_prices_with_provider(
            "yahoo",
            DataRequest(
                ticker=ticker,
                start=args.start,
                end=args.end,
                target_column="close",
                interval=args.interval,
                adjustment_policy=args.adjustment_policy,
            ),
            store=data_store,
            use_cache=not args.no_data_cache,
            refresh_cache=_refresh_cache_for_request(args, "yahoo"),
        )
        context_frame = _filter_prices_as_of(context_result.frame, args.as_of, target_column="close")
        column = context_frame["close"].rename(f"{_safe_label(label)}_{_safe_label(ticker)}")
        prices = prices.join(column, how="left")
        context_sources.append(
            {
                "label": label,
                "ticker": ticker,
                "provider": context_result.metadata.get("provider"),
                "cache_hit": context_result.metadata.get("cache_hit"),
                "normalized_data_hash": context_result.metadata.get("normalized_data_hash"),
                "artifacts": context_result.metadata.get("artifacts", {}),
            }
        )

    if args.rates_csv:
        rates = load_indicator_csv(args.rates_csv, prefix="rates", release_lag_days=args.rates_release_lag_days)
        prices = prices.join(rates, how="left")
        indicator_sources.append(_csv_source("rates", args.rates_csv, args.rates_release_lag_days, rates))
    if args.macro_csv:
        macro = load_indicator_csv(args.macro_csv, prefix="macro", release_lag_days=args.macro_release_lag_days)
        prices = prices.join(macro, how="left")
        indicator_sources.append(_csv_source("macro", args.macro_csv, args.macro_release_lag_days, macro))
    if args.events_csv:
        events = load_event_indicators(
            args.events_csv,
            prices.index,
            prefix="event",
            release_lag_days=args.events_release_lag_days,
        )
        prices = prices.join(events, how="left")
        event_sources.append(_csv_source("event", args.events_csv, args.events_release_lag_days, events))
    if args.enable_alt_news:
        alt_features, alt_metadata = collect_alternative_news_features(
            AlternativeNewsRequest(
                ticker=args.ticker,
                provider=args.alt_news_provider,
                lookback_days=args.alt_news_lookback_days,
                max_items=args.alt_news_max_items,
                sentiment_mode=args.alt_news_sentiment_mode,
                llm_model=args.llm_model,
                llm_reasoning_effort=args.llm_reasoning_effort,
                llm_env_file=args.llm_env_file,
                llm_timeout_seconds=args.llm_timeout,
            ),
            target_index=pd.DatetimeIndex(prices.index),
            store=data_store,
        )
        prices = prices.join(alt_features, how="left")
        alternative_sources.append(alt_metadata)

    prices = _finalize_enriched_prices(prices, target_column=config.target_column)
    universe_summary, panel_artifacts, panel_sources, panel_features, universe_ranking = _build_panel_if_requested(
        args=args,
        primary_prices=prices,
        provider_name=provider_name,
        data_store=data_store,
    )
    if not panel_features.empty:
        prices = _finalize_enriched_prices(prices.join(panel_features, how="left"), target_column=config.target_column)

    security_master = load_security_master(args.security_master_csv) if args.security_master_csv else None
    security_metadata = resolve_security_metadata(
        ticker=args.ticker,
        prices=prices,
        security_master=security_master,
        provider_metadata=primary_result.metadata,
        calendar=args.calendar,
        adjustment_policy=args.adjustment_policy,
    )
    data_quality_report = build_data_quality_report(prices, target_column=config.target_column, calendar=args.calendar)
    calendar_summary = summarize_calendar_alignment(prices, calendar=args.calendar)
    data_artifacts["panel"] = panel_artifacts
    context_sources.extend(panel_sources)
    data_manifest = build_data_manifest(
        prices=prices,
        ticker=args.ticker,
        target_column=config.target_column,
        provider=provider_name,
        source=args.csv,
        request={**request.to_dict(), "as_of": args.as_of, "live_cache_refresh": _refresh_cache_for_request(args, provider_name)},
        artifacts=data_artifacts,
        context_sources=context_sources,
        indicator_sources=indicator_sources,
        event_sources=event_sources,
        alternative_sources=alternative_sources,
        point_in_time_policy=point_in_time_policy_summary(
            macro_release_lag_days=args.macro_release_lag_days,
            rates_release_lag_days=args.rates_release_lag_days,
            events_release_lag_days=args.events_release_lag_days,
        ),
        security_metadata=security_metadata,
        calendar_summary=calendar_summary,
        universe=universe_summary,
    )
    if data_store is not None:
        data_store.write_json(
            "manifests",
            provider_name,
            args.ticker,
            request_key({"ticker": args.ticker, "start": args.start, "end": args.end, "kind": "run_manifest"}),
            data_manifest,
        )

    report = ForecastingEngine(config).run(
        prices,
        data_manifest=data_manifest,
        data_quality_report=data_quality_report,
        security_metadata=security_metadata,
    )
    if universe_ranking:
        report.setdefault("diagnostics", {})["universe_ranking"] = universe_ranking
        report.setdefault("technical_view", {})["universe_ranking"] = universe_ranking
        report.setdefault("data_manifest", {}).setdefault("universe", {}).update(
            {
                "ranking_method": "market_action_cross_sectional_score",
                "ranking_top_n": int(args.universe_top_n),
                "ranking": universe_ranking,
            }
        )
    report.setdefault("technical_view", {}).setdefault("chart_metadata", {})["actual_scale"] = args.chart_scale
    report["technical_view"]["chart_metadata"]["chart_artifact_timeframes"] = ["daily", "weekly", "monthly"]
    if output_dir:
        plot_artifacts = write_plot_artifacts(
            report,
            prices,
            output_dir,
            target_column=config.target_column,
            chart_scale=args.chart_scale,
        )
        report["artifacts"] = {"plots": plot_artifacts, "data": data_artifacts}
        report["artifacts"].update(write_audit_bundle(report, output_dir))
        apply_chapter_19_validation(
            report,
            prices=prices,
            output_dir=output_dir,
            target_column=config.target_column,
        )
        apply_chapter_20_ticker_suitability(
            report,
            prices=prices,
            target_column=config.target_column,
        )
        apply_chapter_21_chart_selection(report)
        apply_chapter_23_30_trade_risk_plan(
            report,
            prices=prices,
            target_column=config.target_column,
        )
        apply_chapter_31_42_portfolio_capital_risk(
            report,
            prices=prices,
            target_column=config.target_column,
        )
        apply_chapter_39_43_discipline_governance(report)
        report["artifacts"].update(write_audit_bundle(report, output_dir))
    forecast_log_path = _forecast_log_path(args, output_dir)
    if forecast_log_path is not None:
        report.setdefault("artifacts", {})["forecast_log"] = str(forecast_log_path)
        _append_forecast_log(report, forecast_log_path)

    print(_summary(report))
    print(
        json.dumps(
            {
                "suggested_action": report["suggested_action"],
                "part_i_suggested_action": report.get("part_i_suggested_action"),
                "risk_level": report["risk_level"],
                "llm_review_status": report.get("decision_view", {}).get("llm_review", {}).get("status"),
                "chapter_19_validation_status": report.get("operations_view", {})
                .get("chapter_19_validation", {})
                .get("status"),
                "chapter_20_profile": report.get("selection_view", {})
                .get("chapter_20_ticker_suitability", {})
                .get("profile_fit", {})
                .get("primary_profile"),
                "chapter_21_bucket": report.get("selection_view", {})
                .get("chapter_21_chart_selection", {})
                .get("chart_selection", {})
                .get("chart_book_bucket"),
                "trade_risk_commitment": report.get("trade_risk_view", {})
                .get("chapter_23_30_trade_risk_plan", {})
                .get("commitment", {})
                .get("commitment_type"),
                "portfolio_capital_status": report.get("portfolio_view", {})
                .get("chapter_31_42_portfolio_capital_risk", {})
                .get("portfolio_capital_gate", {})
                .get("allocation_status"),
                "discipline_status": report.get("discipline_view", {})
                .get("chapter_39_43_discipline_governance", {})
                .get("status"),
            },
            indent=2,
        )
    )


def _summary(report: dict) -> str:
    trend = report.get("technical_view", {}).get("trend_state", {})
    dow = report.get("technical_view", {}).get("dow_theory", {})
    dow_primary = dow.get("primary_trend", {})
    dow_confirmation = dow.get("trend_confirmation", {})
    dow_defects = dow.get("chapter_4_defect_diagnostics", {})
    magee = report.get("technical_view", {}).get("magee_basing_points", {}).get("preferred", {})
    reversal_view = report.get("technical_view", {}).get("reversal_patterns", {})
    reversal = reversal_view.get("preferred", {})
    triangle = report.get("technical_view", {}).get("triangle_patterns", {}).get("preferred", {})
    chapter_9 = report.get("technical_view", {}).get("chapter_9_patterns", {})
    rectangle = chapter_9.get("rectangle_patterns", {}).get("preferred", {})
    multi_top_bottom = chapter_9.get("multi_top_bottom_patterns", {}).get("preferred", {})
    chapter_10 = report.get("technical_view", {}).get("chapter_10_patterns", {})
    chapter_10_structural = chapter_10.get("structural_patterns", {}).get("preferred", {})
    chapter_10_event = chapter_10.get("short_term_events", {}).get("preferred", {})
    chapter_11 = report.get("technical_view", {}).get("chapter_11_patterns", {})
    chapter_11_continuation = chapter_11.get("continuation_patterns", {}).get("preferred", {})
    chapter_11_hs = chapter_11.get("head_and_shoulders_continuation", {}).get("preferred", {})
    chapter_12 = report.get("technical_view", {}).get("chapter_12_gaps", {})
    chapter_12_gap = chapter_12.get("classified_gaps", {}).get("preferred", {})
    chapter_12_island = chapter_12.get("island_reversals", {}).get("preferred", {})
    chapter_13 = report.get("technical_view", {}).get("chapter_13_support_resistance", {})
    chapter_13_support = chapter_13.get("support_zones", {}).get("nearest", {})
    chapter_13_resistance = chapter_13.get("resistance_zones", {}).get("nearest", {})
    chapter_14 = report.get("technical_view", {}).get("chapter_14_trendlines", {})
    chapter_14_trendline = chapter_14.get("trendlines", {}).get("preferred", {})
    chapter_14_channel = chapter_14.get("channels", {}).get("preferred", {})
    chapter_14_fan = chapter_14.get("fan_lines", {}).get("preferred", {})
    chapter_15 = report.get("technical_view", {}).get("chapter_15_major_trendlines", {})
    chapter_15_stock = chapter_15.get("stock_major_trend", {})
    chapter_15_line = chapter_15_stock.get("major_trendline", {})
    chapter_15_confirmation = chapter_15.get("broad_market_confirmation", {})
    chapter_16 = report.get("technical_view", {}).get("chapter_16_market_context", {})
    chapter_16_market = chapter_16.get("market_character", {})
    chapter_16_donchian = chapter_16.get("donchian_context", {})
    chapter_16_risk = chapter_16.get("futures_risk_context", {})
    chapter_17 = report.get("technical_view", {}).get("chapter_17_governance_context", {})
    chapter_17_fragility = chapter_17.get("computer_humility", {}).get("decision_fragility", {})
    chapter_17_conflict = chapter_17.get("computer_humility", {}).get("method_conflict_score", {})
    decision_view = report.get("decision_view", {})
    chapter_18 = decision_view.get("chapter_18_tactical_problem", {})
    chapter_18_plan = chapter_18.get("trade_plan", {})
    chapter_18_llm = chapter_18.get("llm_review", {})
    chapter_18_safety = chapter_18.get("llm_safety_gate", {})
    chapter_19 = report.get("operations_view", {}).get("chapter_19_validation", {})
    chapter_19_gate = chapter_19.get("action_gate", {})
    chapter_20 = report.get("selection_view", {}).get("chapter_20_ticker_suitability", {})
    chapter_20_fit = chapter_20.get("profile_fit", {})
    chapter_21 = report.get("selection_view", {}).get("chapter_21_chart_selection", {})
    chapter_21_selection = chapter_21.get("chart_selection", {})
    trade_risk = report.get("trade_risk_view", {}).get("chapter_23_30_trade_risk_plan", {})
    trade_commitment = trade_risk.get("commitment", {})
    portfolio_capital = report.get("portfolio_view", {}).get("chapter_31_42_portfolio_capital_risk", {})
    portfolio_gate = portfolio_capital.get("portfolio_capital_gate", {})
    discipline = report.get("discipline_view", {}).get("chapter_39_43_discipline_governance", {})
    discipline_gate = discipline.get("discipline_gate", {})
    dormant = reversal_view.get("optional_methods", {}).get("dormant_bottoms", {}).get("preferred", {})
    decision = report.get("technical_view", {}).get("decision_diagnostics", {})
    lines = [
        f"Ticker: {report['ticker']}",
        f"As of: {report['as_of_date']}",
        f"Current Price: {report['current_price']:.2f}",
        f"Technical Trend: {trend.get('state', 'Unknown')}",
        f"Dow Primary Trend: {dow_primary.get('state', 'Unknown')}",
        f"Dow Confirmation: {dow_confirmation.get('status', 'Unknown')}",
        f"Dow Ambiguity: {dow_defects.get('ambiguity_score', 0):.0%}",
        f"Magee Basing Trend: {magee.get('trend_state', 'Unknown')}",
        f"Magee Stop Distance: {_format_pct(magee.get('stop_distance_pct'))}",
        f"Reversal Pattern: {reversal.get('pattern', 'Unknown')} {reversal.get('status', 'Unknown')}",
        f"Reversal Objective: {_format_price(reversal.get('measured_objective'))}",
        f"Triangle Pattern: {triangle.get('pattern', 'Unknown')} {triangle.get('status', 'Unknown')}",
        f"Triangle Objective: {_format_price(triangle.get('measured_objective'))}",
        f"Rectangle Pattern: {rectangle.get('pattern', 'Unknown')} {rectangle.get('status', 'Unknown')}",
        f"Rectangle Objective: {_format_price(rectangle.get('measured_objective'))}",
        f"Double/Triple Pattern: {multi_top_bottom.get('pattern', 'Unknown')} {multi_top_bottom.get('status', 'Unknown')}",
        f"Double/Triple Objective: {_format_price(multi_top_bottom.get('measured_objective'))}",
        f"Chapter 10 Pattern: {chapter_10_structural.get('pattern', 'Unknown')} {chapter_10_structural.get('status', 'Unknown')}",
        f"Chapter 10 Objective: {_format_price(chapter_10_structural.get('measured_objective'))}",
        f"Short-Term Event: {chapter_10_event.get('pattern', 'Unknown')} {chapter_10_event.get('status', 'Unknown')}",
        f"Chapter 11 Continuation: {chapter_11_continuation.get('pattern', 'Unknown')} {chapter_11_continuation.get('status', 'Unknown')}",
        f"Chapter 11 Objective: {_format_price(chapter_11_continuation.get('measured_objective'))}",
        f"Chapter 11 H&S Continuation: {chapter_11_hs.get('status', 'Unavailable')}",
        f"Chapter 12 Gap: {chapter_12_gap.get('pattern', 'Unknown')} {chapter_12_gap.get('status', 'Unknown')}",
        f"Chapter 12 Gap Objective: {_format_price(chapter_12_gap.get('measured_objective'))}",
        f"Chapter 12 Island: {chapter_12_island.get('status', 'Unavailable')}",
        f"Chapter 13 Support: {_format_price(chapter_13_support.get('center'))} {chapter_13_support.get('role_reversal', '')}",
        f"Chapter 13 Resistance: {_format_price(chapter_13_resistance.get('center'))} {chapter_13_resistance.get('role_reversal', '')}",
        f"Chapter 14 Trendline: {chapter_14_trendline.get('kind', 'Unknown')} {chapter_14_trendline.get('status', 'Unknown')}",
        f"Chapter 14 Channel: {chapter_14_channel.get('status', 'Unknown')}",
        f"Chapter 14 Fan Lines: {chapter_14_fan.get('status', 'Unknown')}",
        f"Chapter 15 Major Trendline: {chapter_15_line.get('kind', 'Unknown')} {chapter_15_line.get('status', 'Unknown')}",
        f"Chapter 15 Scale: {chapter_15_line.get('scale', 'Unavailable')}",
        f"Chapter 15 Broad Market: {chapter_15_confirmation.get('status', 'Unavailable')}",
        f"Chapter 16 Market Context: {chapter_16_market.get('state', 'Unknown')}",
        f"Chapter 16 Donchian: {chapter_16_donchian.get('overall_state', 'Unavailable')}",
        f"Chapter 16 Risk Context: {chapter_16_risk.get('risk_state', 'Unavailable')} report-only",
        f"Chapter 17 Decision Fragility: {chapter_17_fragility.get('level', 'Unavailable')} report-only",
        f"Chapter 17 Method Conflict: {chapter_17_conflict.get('level', 'Unavailable')} report-only",
        f"Chapter 18 Tactical Profile: {chapter_18.get('tactical_profile', {}).get('name', 'Unavailable')}",
        (
            "Chapter 18 Tactical Action: "
            f"{chapter_18.get('part_i_action', 'Unknown')} -> "
            f"{chapter_18.get('rule_based_action', 'Unknown')} -> "
            f"{chapter_18.get('final_action', 'Unknown')}"
        ),
        f"Chapter 18 Reward/Risk: {_format_number(chapter_18_plan.get('reward_to_risk'))}",
        f"Chapter 18 LLM Review: {chapter_18_llm.get('status', 'Unavailable')} safety={chapter_18_safety.get('status', 'Unavailable')}",
        (
            "Chapter 19 Validation: "
            f"{chapter_19.get('status', 'Unavailable')} "
            f"action={chapter_19_gate.get('input_action', 'Unknown')} -> {chapter_19_gate.get('validated_action', 'Unknown')}"
        ),
        (
            "Chapter 20 Suitability: "
            f"{chapter_20_fit.get('primary_profile', 'Unavailable')} "
            f"{chapter_20_fit.get('classification', 'Unavailable')} "
            f"score={_format_number(chapter_20_fit.get('suitability_score'))}"
        ),
        (
            "Chapter 21 Chart Selection: "
            f"{chapter_21_selection.get('chart_book_bucket', 'Unavailable')} "
            f"{chapter_21_selection.get('chart_book_action', 'Unavailable')}"
        ),
        (
            "Chapters 23-30 Trade/Risk: "
            f"{trade_commitment.get('commitment_type', 'Unavailable')} "
            f"{trade_commitment.get('entry_plan', 'Unavailable')}"
        ),
        (
            "Chapters 31/38/40-42 Portfolio Capital: "
            f"{portfolio_gate.get('allocation_status', 'Unavailable')}"
        ),
        (
            "Chapters 39/43 Discipline: "
            f"{discipline.get('status', 'Unavailable')} "
            f"{discipline_gate.get('plan_adherence', 'Unavailable')}"
        ),
        f"Dormant Bottom: {dormant.get('status', 'Unavailable')}",
        f"Suggested Action: {report['suggested_action']}",
        f"Risk Level: {report['risk_level']}",
        "",
    ]
    if decision.get("hold_reason"):
        lines.extend([f"Hold Reason: {decision['hold_reason']}", ""])
    blockers = decision.get("blocking_reasons", [])
    if blockers:
        lines.append("Decision blockers:")
        lines.extend(f"- {reason}" for reason in blockers[:3])
        lines.append("")
    for forecast in report["forecasts"]:
        lines.extend(
            [
                f"Horizon: {forecast['horizon_days']} trading days",
                f"Selected Model: {forecast['selected_model']}",
                f"Selection Metric: {forecast['selection_metric']}",
                f"Expected Direction: {forecast['expected_direction']}",
                f"Forecast Confidence: {forecast['directional_confidence']:.0%}",
                f"Predicted Price: {forecast['predicted_price']:.2f}",
                f"Interval: {forecast['lower_price']:.2f} - {forecast['upper_price']:.2f}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def _filter_prices_as_of(prices: pd.DataFrame, as_of: str | None, target_column: str) -> pd.DataFrame:
    frame = normalize_price_frame(prices, target_column=target_column)
    if not as_of:
        return frame
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert(None)
    filtered = frame.loc[frame.index <= cutoff]
    if filtered.empty:
        raise ValueError(f"No price rows are available at or before --as-of {as_of}.")
    return filtered


def _interval_to_minutes(interval: str) -> float | None:
    value = interval.strip().lower()
    if not value:
        return None
    units = {
        "m": 1.0,
        "min": 1.0,
        "h": 60.0,
        "hr": 60.0,
        "d": 24.0 * 60.0,
        "wk": 7.0 * 24.0 * 60.0,
        "mo": 30.0 * 24.0 * 60.0,
    }
    for suffix, multiplier in sorted(units.items(), key=lambda item: len(item[0]), reverse=True):
        if value.endswith(suffix):
            number = value[: -len(suffix)]
            try:
                return float(number) * multiplier
            except ValueError:
                return None
    return None


def _refresh_cache_for_request(args: argparse.Namespace, provider_name: str) -> bool:
    if args.refresh_data_cache:
        return True
    if args.allow_live_cache:
        return False
    return provider_name.lower() == "yahoo" and args.end is None and not args.csv


def _forecast_log_path(args: argparse.Namespace, output_dir: Path | None) -> Path | None:
    if args.no_forecast_log:
        return None
    if args.forecast_log:
        return Path(args.forecast_log)
    if output_dir is not None:
        return output_dir / "forecast_log.csv"
    return None


def _append_forecast_log(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for forecast in report.get("forecasts", []):
        rows.append(
            {
                "generated_at_utc": report.get("generated_at_utc"),
                "ticker": report.get("ticker"),
                "as_of_timestamp": report.get("as_of_timestamp", report.get("as_of_date")),
                "forecast_interval": report.get("forecast_interval"),
                "horizon": forecast.get("horizon_days"),
                "forecast_timestamp": forecast.get("forecast_date"),
                "current_price": report.get("current_price"),
                "predicted_price": forecast.get("predicted_price"),
                "lower_price": forecast.get("lower_price"),
                "upper_price": forecast.get("upper_price"),
                "expected_return": forecast.get("expected_return"),
                "expected_direction": forecast.get("expected_direction"),
                "directional_confidence": forecast.get("directional_confidence"),
                "selected_model": forecast.get("selected_model"),
                "suggested_action": report.get("suggested_action"),
                "risk_level": report.get("risk_level"),
            }
        )
    if not rows:
        return
    frame = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        frame = pd.concat([existing, frame], ignore_index=True)
        frame = frame.drop_duplicates(
            subset=["ticker", "as_of_timestamp", "forecast_interval", "horizon"],
            keep="last",
        )
    frame.to_csv(path, index=False)


def _format_pct(value: object) -> str:
    try:
        return f"{float(value):.1%}"
    except Exception:
        return "Unavailable"


def _format_price(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "Unavailable"


def _format_number(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "Unavailable"


def _data_store(data_dir: str | None, output_dir: Path | None) -> MarketDataStore | None:
    if data_dir:
        return MarketDataStore(data_dir)
    if output_dir:
        return MarketDataStore(output_dir / "data")
    return None


def _finalize_enriched_prices(prices: pd.DataFrame, target_column: str) -> pd.DataFrame:
    frame = normalize_price_frame(prices, target_column=target_column)
    base_columns = {"open", "high", "low", "close", "volume", "dividends", "stock_splits", target_column.lower()}
    external_columns = [column for column in frame.columns if str(column).lower() not in base_columns]
    if external_columns:
        frame[external_columns] = frame[external_columns].ffill()
    return frame.dropna(subset=[target_column.lower()])


def _csv_source(prefix: str, path: str, release_lag_days: int, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "prefix": prefix,
        "path": str(path),
        "release_lag_days": int(release_lag_days),
        "columns": [str(column) for column in frame.columns],
        "rows": int(len(frame)),
    }


def _build_panel_if_requested(
    args: argparse.Namespace,
    primary_prices: pd.DataFrame,
    provider_name: str,
    data_store: MarketDataStore | None,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]], pd.DataFrame, list[dict[str, object]]]:
    tickers = parse_universe_tickers(args.universe_tickers)
    universe = pd.DataFrame()
    if args.universe_csv:
        universe = load_universe_csv(args.universe_csv)
        tickers.extend(universe["ticker"].astype(str).str.upper().tolist())
    tickers = sorted({args.ticker.upper(), *tickers})
    if len(tickers) == 1 and not args.universe_csv and not args.universe_tickers:
        return {}, {}, [], pd.DataFrame(), []
    if not args.build_panel and not args.rank_universe:
        return summarize_universe(tickers), {}, [], pd.DataFrame(), []

    panel_provider = "yahoo" if provider_name == "csv" else provider_name
    frames = {args.ticker.upper(): primary_prices.copy()}
    sources: list[dict[str, object]] = []
    for ticker in tickers:
        if ticker == args.ticker.upper():
            continue
        result = load_prices_with_provider(
            panel_provider,
            DataRequest(
                ticker=ticker,
                start=args.start,
                end=args.end,
                target_column=args.target_column.lower(),
                interval=args.interval,
                adjustment_policy=args.adjustment_policy,
            ),
            store=data_store,
            use_cache=not args.no_data_cache,
            refresh_cache=_refresh_cache_for_request(args, panel_provider),
        )
        frames[ticker] = _filter_prices_as_of(result.frame, args.as_of, target_column=args.target_column.lower()).copy()
        sources.append(
            {
                "label": "universe_panel",
                "ticker": ticker,
                "provider": result.metadata.get("provider"),
                "cache_hit": result.metadata.get("cache_hit"),
                "normalized_data_hash": result.metadata.get("normalized_data_hash"),
                "artifacts": result.metadata.get("artifacts", {}),
            }
        )

    if not universe.empty and "sector" in universe.columns:
        sector_by_ticker = universe.set_index("ticker")["sector"].astype(str).to_dict()
        for ticker, frame in frames.items():
            if ticker in sector_by_ticker:
                frame["sector"] = sector_by_ticker[ticker]

    panel = build_panel_frame(frames)
    panel_features = build_cross_sectional_panel_features(panel, price_column=args.target_column.lower())
    primary_panel_features = select_ticker_panel_features(panel_features, args.ticker)
    universe_ranking = (
        rank_universe_from_panel(
            panel,
            panel_features,
            price_column=args.target_column.lower(),
            top_n=args.universe_top_n,
        )
        if args.rank_universe
        else []
    )
    artifacts: dict[str, object] = {}
    if data_store is not None:
        key = request_key({"kind": "panel", "tickers": tickers, "start": args.start, "end": args.end, "provider": panel_provider})
        artifacts["panel"] = data_store.write_frame("panel", panel_provider, "universe", key, panel).to_dict()
        artifacts["panel_features"] = data_store.write_frame("features", panel_provider, "universe", f"{key}_cross_sectional", panel_features).to_dict()
    return summarize_universe(tickers, panel), artifacts, sources, primary_panel_features, universe_ranking


def _safe_label(label: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in label).strip("_")


if __name__ == "__main__":
    main()
