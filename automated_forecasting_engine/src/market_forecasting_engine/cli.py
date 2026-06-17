from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
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
from market_forecasting_engine.long_term_enrichment import (
    DEFAULT_LONG_TERM_SOURCE_PROVIDERS,
    enrich_prices_with_long_term_sources,
    long_term_context_manifest_entry,
    long_term_context_source_entry,
    parse_long_term_source_providers,
)
from market_forecasting_engine.llm_trader.prompts import autonomous_trader
from market_forecasting_engine.llm_trader.profiles import trader_profiles
from market_forecasting_engine.llm_trader.responses_api import call_response, response_payload
from market_forecasting_engine.llm_trader.run import (
    build_technical_packet,
    load_env,
    openai_client_for_provider,
    resolve_llm_model,
    resolve_llm_provider,
)
from market_forecasting_engine.llm_trader.source_synthesis import (
    attach_long_term_source_synthesis,
    run_long_term_source_synthesis,
)
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
from market_forecasting_engine.plots import write_forecast_log_plot_artifacts, write_plot_artifacts
from market_forecasting_engine.schema import ForecastConfig
from market_forecasting_engine.security_master import load_security_master, resolve_security_metadata
from market_forecasting_engine.strategy_knowledge import (
    DEFAULT_STRATEGY_CORPUS_DIR,
    DEFAULT_STRATEGY_INDEX_PATH,
    StrategyKnowledgeRequest,
    attach_strategy_knowledge_context,
    build_strategy_knowledge_context,
)


class TerminalProgress:
    """Small terminal progress reporter for long forecast runs."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    def step(self, message: str) -> None:
        if self.enabled:
            print(f"[forecast] {datetime.now().strftime('%H:%M:%S')} {message}", flush=True)

    def __call__(self, event: dict[str, object]) -> None:
        if not self.enabled:
            return
        name = str(event.get("event") or "")
        horizon = event.get("horizon_days")
        if name == "horizon_started":
            self.step(f"horizon {horizon}d: started")
        elif name == "horizon_completed":
            self.step(
                "horizon "
                f"{horizon}d: selected {event.get('selected_model')} "
                f"direction={event.get('expected_direction')} "
                f"confidence={_format_number(event.get('directional_confidence'))}"
            )
        elif name == "validation_started":
            self.step(
                "horizon "
                f"{horizon}d: validating {event.get('candidate_count')} candidates "
                f"with {event.get('workers')} worker(s)"
            )
        elif name == "candidate_completed":
            completed = int(event.get("completed") or 0)
            total = int(event.get("total") or 0)
            succeeded = "ok" if event.get("succeeded") else "failed"
            bar = _progress_bar(completed, total)
            self.step(
                "horizon "
                f"{horizon}d: {bar} {completed}/{total} "
                f"{event.get('candidate')} ({event.get('family')}) {succeeded}"
            )
        elif name == "validation_completed":
            self.step(
                "horizon "
                f"{horizon}d: validation complete, "
                f"{event.get('successful_candidate_count')} successful candidates"
            )


def _progress_bar(completed: int, total: int, *, width: int = 20) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = max(0, min(width, round(width * completed / total)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


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
    parser.add_argument("--min-training-rows", type=int, default=180, help="Minimum price rows required before model validation starts.")
    parser.add_argument("--validation-window", type=int, default=45, help="Rows in each walk-forward validation window.")
    parser.add_argument("--step-size", type=int, default=20, help="Rows advanced between validation splits.")
    parser.add_argument("--max-splits", type=int, default=8, help="Maximum walk-forward validation splits.")
    parser.add_argument(
        "--validation-workers",
        type=int,
        default=0,
        help="Parallel workers for CPU-safe model validation. 0=auto bounded parallelism; -1=serial. Deep-learning/GPU candidates run serially.",
    )
    parser.add_argument("--purge-window", type=int, default=None, help="Rows to purge before each validation window. Defaults to horizon.")
    parser.add_argument("--embargo-window", type=int, default=0, help="Extra rows to embargo before each validation window.")
    parser.add_argument("--final-holdout-fraction", type=float, default=0.15, help="Final untouched holdout fraction for post-selection diagnostics.")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0, help="Backtest transaction cost in basis points per signal change.")
    parser.add_argument("--output-dir", default=None, help="Directory for forecast_report.json and governance artifacts.")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress updates during long forecast stages.")
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
    parser.add_argument("--include-lstm", action="store_true", help="Backward-compatible flag to enable the optional LSTM candidate. Slower because it trains during validation.")
    parser.add_argument(
        "--deep-learning-profile",
        choices=("off", "fast", "research"),
        default="off",
        help="Chapter 17 optional deep-learning candidate profile. Off by default; research includes the slower LSTM path.",
    )
    parser.add_argument("--search-level", choices=("fast", "expanded"), default="fast", help="Candidate tuning breadth.")
    parser.add_argument("--tune", choices=("fixed", "optuna"), default="fixed", help="Candidate tuning mode. `optuna` adds nested time-series optimization candidates.")
    parser.add_argument("--optuna-trials", type=int, default=25, help="Optuna trials per tuned candidate fit.")
    parser.add_argument("--optuna-timeout", type=int, default=None, help="Optional Optuna timeout in seconds per tuned candidate fit.")
    parser.add_argument("--optuna-inner-splits", type=int, default=3, help="Inner walk-forward splits used inside each Optuna objective.")
    parser.add_argument(
        "--optuna-families",
        default="lightgbm,xgboost,elastic_net,random_forest,extra_trees,gradient_boosting",
        help="Comma-separated Optuna model families to tune.",
    )
    parser.add_argument(
        "--tactical-profile",
        choices=("short_term", "intermediate", "long_term"),
        default="intermediate",
        help="Chapter 18 tactical profile for stops, reward/risk gates, and holding horizon.",
    )
    parser.add_argument("--enable-llm-review", action="store_true", default=False, help="Enable the optional Chapter 18 LLM tactical reviewer.")
    parser.add_argument("--disable-llm-review", action="store_false", dest="enable_llm_review", help="Disable the optional Chapter 18 LLM tactical reviewer.")
    parser.add_argument(
        "--enable-autonomous-llm-decision",
        action="store_true",
        default=True,
        help="Run the full autonomous trader LLM as the final advice layer. Enabled by default.",
    )
    parser.add_argument(
        "--disable-autonomous-llm-decision",
        action="store_false",
        dest="enable_autonomous_llm_decision",
        help="Disable the default autonomous trader LLM final advice layer.",
    )
    parser.add_argument("--llm-dry-run", action="store_true", help="Build the autonomous LLM payload without calling the model.")
    parser.add_argument("--trader-profile", choices=("aggressive", "medium", "conservative"), default="medium", help="Trader profile for autonomous LLM final advice.")
    parser.add_argument("--trader-name", default="forecast_autonomous_trader", help="Trader name for autonomous LLM final advice artifacts.")
    parser.add_argument("--holding-status", choices=("not_owned", "owned"), default="not_owned", help="Optional position context for autonomous final advice.")
    parser.add_argument("--entry-price", type=float, default=None, help="Optional owned-position average entry/cost basis.")
    parser.add_argument("--quantity", type=float, default=None, help="Optional owned-position quantity.")
    parser.add_argument("--position-value", type=float, default=None, help="Optional owned-position market value.")
    parser.add_argument("--account-equity", type=float, default=None, help="Optional account equity for sizing context.")
    parser.add_argument("--portfolio-notes", default="", help="Optional portfolio notes for the autonomous final advice layer.")
    parser.add_argument("--llm-provider", choices=("openai", "huggingface", "bedrock", "llm_studio"), default="openai", help="LLM provider for optional tactical review. Defaults to OpenAI.")
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
    parser.add_argument("--llm-no-web-search", action="store_true", help="Disable autonomous LLM web search.")
    parser.add_argument("--llm-search-context-size", choices=("low", "medium", "high"), default="medium", help="Web-search context size for autonomous LLM final advice.")
    parser.add_argument(
        "--disable-strategy-knowledge",
        action="store_true",
        help="Disable default FAISS-backed durable strategy knowledge retrieval for the CEO LLM.",
    )
    parser.add_argument(
        "--strategy-knowledge-corpus-dir",
        default=str(DEFAULT_STRATEGY_CORPUS_DIR),
        help="Directory containing strategy knowledge documents to embed and retrieve for the CEO LLM.",
    )
    parser.add_argument(
        "--strategy-knowledge-index",
        default=str(DEFAULT_STRATEGY_INDEX_PATH),
        help="FAISS index path for durable strategy knowledge.",
    )
    parser.add_argument("--strategy-knowledge-max-chunks", type=int, default=8, help="Maximum retrieved strategy chunks passed into CEO context.")
    parser.add_argument("--strategy-knowledge-rebuild-index", action="store_true", help="Force rebuild of the strategy FAISS index before retrieval.")
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
    parser.add_argument("--enable-alt-news", action="store_true", default=True, help="Download/scrape alternative news and add rolling sentiment features. Enabled by default.")
    parser.add_argument("--disable-alt-news", action="store_false", dest="enable_alt_news", help="Disable default alternative news collection for this run.")
    parser.add_argument("--alt-news-provider", default="yahoo_rss", help="Alternative news provider: yahoo_rss, yahoo_news, or openai_web.")
    parser.add_argument("--alt-news-lookback-days", type=int, default=30, help="Lookback window for downloaded/scraped news.")
    parser.add_argument("--alt-news-max-items", type=int, default=40, help="Maximum downloaded/scraped news items.")
    parser.add_argument(
        "--alt-news-sentiment-mode",
        choices=("lexicon", "llm", "hybrid"),
        default="lexicon",
        help="News sentiment scoring mode. Hybrid blends deterministic scoring with optional LLM classification.",
    )
    parser.add_argument(
        "--alt-news-topic-mode",
        choices=("none", "llm"),
        default="llm",
        help="Chapter 15 topic extraction mode. Defaults to llm; use none to disable controlled financial topic extraction.",
    )
    parser.add_argument("--alt-news-topic-max-articles", type=int, default=24)
    parser.add_argument("--alt-news-topic-max-topics-per-article", type=int, default=3)
    parser.add_argument(
        "--alt-news-embedding-mode",
        choices=("openai", "none"),
        default="openai",
        help="Chapter 16 text embedding mode. Defaults to openai; use none to disable semantic embedding features.",
    )
    parser.add_argument("--alt-news-embedding-model", default=None, help="Embedding model for Chapter 16 semantic features.")
    parser.add_argument("--alt-news-embedding-dimensions", type=int, default=256)
    parser.add_argument("--alt-news-embedding-max-articles", type=int, default=24)
    parser.add_argument(
        "--alt-news-embedding-finance-knowledge-json",
        default=None,
        help="Optional JSON object of finance-domain prototype labels to text descriptions for Chapter 16 similarity features.",
    )
    parser.add_argument(
        "--enable-long-term-sources",
        action="store_true",
        default=True,
        help="Call and consolidate all configured long-term data sources into the governed decision context.",
    )
    parser.add_argument(
        "--disable-long-term-sources",
        action="store_false",
        dest="enable_long_term_sources",
        help="Disable default long-term source collection for this run.",
    )
    parser.add_argument(
        "--long-term-source-providers",
        default=",".join(DEFAULT_LONG_TERM_SOURCE_PROVIDERS),
        help="Comma-separated long-term source providers to call. Default calls all supported providers.",
    )
    parser.add_argument(
        "--long-term-source-output-dir",
        default=None,
        help="Optional artifact directory for raw long-term source payloads. Defaults under --output-dir when supplied.",
    )
    parser.add_argument(
        "--long-term-source-env-file",
        default=None,
        help="Optional .env file for long-term source API keys. Defaults to --llm-env-file or process env.",
    )
    parser.add_argument(
        "--long-term-source-snapshot-dir",
        default=None,
        help="Durable directory for point-in-time long-term source snapshots used as leak-safe model features.",
    )
    parser.add_argument("--universe-csv", default=None, help="Optional universe CSV with ticker/symbol column for panel metadata.")
    parser.add_argument("--universe-tickers", default=None, help="Optional comma-separated universe tickers for panel metadata.")
    parser.add_argument("--build-panel", action="store_true", help="Fetch and store a date/ticker panel for the supplied universe.")
    parser.add_argument("--rank-universe", action="store_true", help="Rank supplied universe members using market-action and cross-sectional evidence.")
    parser.add_argument("--universe-top-n", type=int, default=25, help="Maximum universe ranking rows to include.")
    args = parser.parse_args()
    progress = TerminalProgress(enabled=not args.no_progress)

    horizons = tuple(int(value.strip()) for value in args.horizons.split(",") if value.strip())
    forecast_interval_minutes = _interval_to_minutes(args.interval)
    long_term_source_providers = parse_long_term_source_providers(args.long_term_source_providers)
    progress.step(
        f"starting forecast ticker={args.ticker.upper()} horizons={','.join(str(value) for value in horizons)} "
        f"provider={args.provider or ('csv' if args.csv else 'yahoo')}"
    )
    config = ForecastConfig(
        ticker=args.ticker,
        horizons=horizons,
        target_column=args.target_column.lower(),
        min_training_rows=args.min_training_rows,
        validation_window=args.validation_window,
        step_size=args.step_size,
        max_splits=args.max_splits,
        validation_workers=args.validation_workers,
        selection_metric=args.selection_metric,
        confidence_level=args.confidence_level,
        purge_window=args.purge_window,
        embargo_window=args.embargo_window,
        final_holdout_fraction=args.final_holdout_fraction,
        transaction_cost_bps=args.transaction_cost_bps,
        include_lightgbm=not args.no_lightgbm,
        include_statistical_models=not args.no_statistical_models,
        include_lstm=args.include_lstm,
        deep_learning_profile=args.deep_learning_profile,
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
        enable_long_term_sources=args.enable_long_term_sources,
        long_term_source_providers=long_term_source_providers,
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
    progress.step(f"loading primary price data from {provider_name}")
    primary_result = load_prices_with_provider(
        provider_name,
        request,
        store=data_store,
        use_cache=not args.no_data_cache,
        refresh_cache=_refresh_cache_for_request(args, provider_name),
    )
    prices = primary_result.frame
    prices = _filter_prices_as_of(prices, args.as_of, target_column=config.target_column)
    progress.step(f"primary price data loaded rows={len(prices)} columns={len(prices.columns)}")
    data_artifacts: dict[str, object] = {"primary": primary_result.metadata.get("artifacts", {})}
    context_sources: list[dict[str, object]] = []
    indicator_sources: list[dict[str, object]] = []
    event_sources: list[dict[str, object]] = []
    alternative_sources: list[dict[str, object]] = []
    long_term_context: dict[str, object] | None = None
    long_term_snapshot_feature_metadata: dict[str, object] | None = None

    yahoo_context = {}
    if args.benchmark:
        yahoo_context["benchmark"] = args.benchmark
    if args.sector:
        yahoo_context["sector"] = args.sector
    if args.vix:
        yahoo_context["volatility"] = args.vix

    for label, ticker in yahoo_context.items():
        progress.step(f"loading context ticker {ticker} for {label}")
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
        progress.step("loading rates indicator CSV")
        rates = load_indicator_csv(args.rates_csv, prefix="rates", release_lag_days=args.rates_release_lag_days)
        prices = prices.join(rates, how="left")
        indicator_sources.append(_csv_source("rates", args.rates_csv, args.rates_release_lag_days, rates))
    if args.macro_csv:
        progress.step("loading macro indicator CSV")
        macro = load_indicator_csv(args.macro_csv, prefix="macro", release_lag_days=args.macro_release_lag_days)
        prices = prices.join(macro, how="left")
        indicator_sources.append(_csv_source("macro", args.macro_csv, args.macro_release_lag_days, macro))
    if args.events_csv:
        progress.step("loading event indicator CSV")
        events = load_event_indicators(
            args.events_csv,
            prices.index,
            prefix="event",
            release_lag_days=args.events_release_lag_days,
        )
        prices = prices.join(events, how="left")
        event_sources.append(_csv_source("event", args.events_csv, args.events_release_lag_days, events))
    if args.enable_alt_news:
        progress.step(
            f"collecting alternative news provider={args.alt_news_provider} "
            f"sentiment={args.alt_news_sentiment_mode} topic={args.alt_news_topic_mode}"
        )
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
                topic_mode=args.alt_news_topic_mode,
                topic_max_articles=args.alt_news_topic_max_articles,
                topic_max_topics_per_article=args.alt_news_topic_max_topics_per_article,
                embedding_mode=args.alt_news_embedding_mode,
                embedding_model=args.alt_news_embedding_model,
                embedding_dimensions=args.alt_news_embedding_dimensions,
                embedding_max_articles=args.alt_news_embedding_max_articles,
                embedding_finance_knowledge_path=args.alt_news_embedding_finance_knowledge_json,
            ),
            target_index=pd.DatetimeIndex(prices.index),
            store=data_store,
        )
        prices = prices.join(alt_features, how="left")
        alternative_sources.append(alt_metadata)
        progress.step(f"alternative news features added columns={len(alt_features.columns)}")

    if args.enable_long_term_sources:
        progress.step(f"collecting long-term sources providers={','.join(long_term_source_providers)}")
        long_term_output_dir = (
            Path(args.long_term_source_output_dir)
            if args.long_term_source_output_dir
            else ((output_dir / "long_term_sources") if output_dir else None)
        )
        prices, long_term_context, long_term_snapshot_feature_metadata = enrich_prices_with_long_term_sources(
            ticker=args.ticker,
            prices=prices,
            target_column=config.target_column,
            enabled=True,
            providers=long_term_source_providers,
            env_file=args.long_term_source_env_file or args.llm_env_file,
            output_dir=long_term_output_dir,
            snapshot_dir=args.long_term_source_snapshot_dir,
            data_store=data_store,
            start_date=args.start,
            end_date=args.end,
        )
        source_entry = long_term_context_source_entry(long_term_context, long_term_snapshot_feature_metadata)
        if source_entry:
            context_sources.append(source_entry)
        provider_status = (long_term_context or {}).get("provider_status", {}) if isinstance(long_term_context, dict) else {}
        progress.step(f"long-term sources complete status={json.dumps(provider_status, sort_keys=True)}")

    progress.step("finalizing enriched feature frame")
    prices = _finalize_enriched_prices(prices, target_column=config.target_column)
    progress.step(f"feature frame ready rows={len(prices)} columns={len(prices.columns)}")
    progress.step("building optional universe/panel features")
    universe_summary, panel_artifacts, panel_sources, panel_features, universe_ranking = _build_panel_if_requested(
        args=args,
        primary_prices=prices,
        provider_name=provider_name,
        data_store=data_store,
    )
    if not panel_features.empty:
        prices = _finalize_enriched_prices(prices.join(panel_features, how="left"), target_column=config.target_column)
        progress.step(f"panel features added columns={len(panel_features.columns)}")

    progress.step("building security metadata and data-quality report")
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
    if long_term_context:
        data_manifest["long_term_sources"] = long_term_context_manifest_entry(
            long_term_context,
            long_term_snapshot_feature_metadata,
        )
    if data_store is not None:
        data_store.write_json(
            "manifests",
            provider_name,
            args.ticker,
            request_key({"ticker": args.ticker, "start": args.start, "end": args.end, "kind": "run_manifest"}),
            data_manifest,
        )

    progress.step("running forecast engine: feature analysis, model validation, selection, tactical gates")
    report = ForecastingEngine(config, progress_callback=progress).run(
        prices,
        data_manifest=data_manifest,
        data_quality_report=data_quality_report,
        security_metadata=security_metadata,
        long_term_context=long_term_context,
    )
    progress.step(f"forecast engine complete initial_action={report.get('suggested_action')} risk={report.get('risk_level')}")
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
        progress.step("writing plots, governance bundle, validation, trade-risk, and portfolio-risk views")
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
        progress.step("plots and governance views complete")
    if args.enable_autonomous_llm_decision:
        progress.step("running CEO LLM final decision layer")
        autonomous = _run_autonomous_llm_decision(report, args, dry_run=bool(args.llm_dry_run), progress=progress)
        _apply_autonomous_llm_to_report(report, autonomous)
        progress.step(
            "CEO LLM complete "
            f"status={autonomous.get('status')} "
            f"decision={(autonomous.get('decision') or {}).get('decision') if isinstance(autonomous.get('decision'), dict) else None}"
        )
    if output_dir:
        progress.step("writing final audit bundle")
        report.setdefault("artifacts", {}).update(write_audit_bundle(report, output_dir))
    forecast_log_path = _forecast_log_path(args, output_dir)
    if forecast_log_path is not None:
        progress.step("updating forecast log and forecast-log chart")
        report.setdefault("artifacts", {})["forecast_log"] = str(forecast_log_path)
        _append_forecast_log(report, forecast_log_path)
        report["artifacts"].update(
            write_forecast_log_plot_artifacts(
                forecast_log_path,
                output_dir=output_dir or forecast_log_path.parent,
                ticker=report.get("ticker"),
            )
        )

    progress.step("forecast run complete")
    print(_summary(report))
    print(_final_ceo_decision_block(report))
    print(
        json.dumps(
            {
                "suggested_action": report["suggested_action"],
                "part_i_suggested_action": report.get("part_i_suggested_action"),
                "risk_level": report["risk_level"],
                "llm_review_status": report.get("decision_view", {}).get("llm_review", {}).get("status"),
                "autonomous_llm_status": report.get("autonomous_llm_trader", {}).get("status"),
                "autonomous_llm_decision": (report.get("autonomous_llm_trader", {}).get("decision") or {}).get("decision"),
                "autonomous_execution_allowed": report.get("decision_view", {})
                .get("autonomous_execution_gate", {})
                .get("execution_allowed"),
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


def _run_autonomous_llm_decision(
    report: dict,
    args: argparse.Namespace,
    *,
    dry_run: bool,
    progress: TerminalProgress | None = None,
) -> dict:
    if progress is not None:
        progress.step("CEO source synthesis: started")
    source_synthesis = run_long_term_source_synthesis(
        report=report,
        llm_provider=getattr(args, "llm_provider", None),
        llm_model=args.llm_model,
        reasoning_effort=args.llm_reasoning_effort,
        llm_env_file=args.llm_env_file,
        timeout_seconds=float(args.llm_timeout),
        dry_run=dry_run,
    )
    attach_long_term_source_synthesis(report, source_synthesis)
    if progress is not None:
        progress.step(f"CEO source synthesis: status={source_synthesis.get('status')}")
    if not getattr(args, "disable_strategy_knowledge", False):
        if progress is not None:
            progress.step("CEO strategy knowledge retrieval: started")
        strategy_context = build_strategy_knowledge_context(
            report,
            StrategyKnowledgeRequest(
                ticker=str(report.get("ticker") or args.ticker).upper(),
                corpus_dir=args.strategy_knowledge_corpus_dir,
                index_path=args.strategy_knowledge_index,
                llm_env_file=args.llm_env_file,
                max_chunks=args.strategy_knowledge_max_chunks,
                rebuild_index=args.strategy_knowledge_rebuild_index,
                timeout_seconds=int(args.llm_timeout),
            ),
        )
        if progress is not None:
            progress.step(
                "CEO strategy knowledge retrieval: "
                f"status={strategy_context.get('status')} "
                f"chunks={len(strategy_context.get('retrieved_chunks', []) or [])}"
            )
    else:
        strategy_context = {
            "status": "disabled",
            "reason": "disabled by --disable-strategy-knowledge",
            "decision_policy": {
                "feeds_ceo_llm": False,
                "overrides_model_validation": False,
                "overrides_risk_gates": False,
            },
        }
    attach_strategy_knowledge_context(report, strategy_context)
    profile = trader_profiles[args.trader_profile]
    portfolio_context = {
        "holding_status": args.holding_status,
        "entry_price": args.entry_price,
        "quantity": args.quantity,
        "position_value": args.position_value,
        "account_equity": args.account_equity,
        "notes": args.portfolio_notes,
    }
    item = {
        "today": datetime.now(UTC).date().isoformat(),
        "ticker": str(report.get("ticker") or args.ticker).upper(),
        "trader_name": args.trader_name,
        "trader_profile_json": json.dumps(profile, indent=2, sort_keys=True),
        "portfolio_context_json": json.dumps(portfolio_context, indent=2, sort_keys=True),
        "technical_packet_json": json.dumps(build_technical_packet(report), indent=2, sort_keys=True, default=str),
    }
    provider = resolve_llm_provider(getattr(args, "llm_provider", None))
    model = resolve_llm_model(args.llm_model, provider=provider)
    if progress is not None:
        progress.step(f"CEO payload build: provider={provider} model={model}")
    payload = response_payload(
        model=model,
        system_message=autonomous_trader.system_message,
        user_message=autonomous_trader.user_message,
        json_schema=autonomous_trader.json_schema,
        reasoning_effort=args.llm_reasoning_effort,
        item=item,
        use_web_search=not args.llm_no_web_search and provider == "openai",
        search_context_size=args.llm_search_context_size,
        require_web_search=not args.llm_no_web_search and provider == "openai",
    )
    if dry_run:
        if progress is not None:
            progress.step("CEO LLM call skipped: dry run")
        return {
            "status": "dry_run",
            "model": model,
            "provider": provider,
            "decision": None,
            "portfolio_context": portfolio_context,
            "long_term_source_synthesis": source_synthesis,
            "strategy_knowledge_context": strategy_context,
            "llm_prompt_payload": payload,
            "reason": "Autonomous LLM decision packet was built but no model call was made.",
        }
    try:
        if progress is not None:
            progress.step("CEO LLM call: started")
        load_env(args.llm_env_file)
        client = openai_client_for_provider(provider, timeout=float(args.llm_timeout))
        payload, raw_response, decision = call_response(
            client=client,
            provider=provider,
            model=model,
            system_message=autonomous_trader.system_message,
            user_message=autonomous_trader.user_message,
            json_schema=autonomous_trader.json_schema,
            reasoning_effort=args.llm_reasoning_effort,
            item=item,
            use_web_search=not args.llm_no_web_search and provider == "openai",
            search_context_size=args.llm_search_context_size,
            require_web_search=not args.llm_no_web_search and provider == "openai",
            usage_context={
                "purpose": "forecast_cli_autonomous_final_decision",
                "ticker": str(report.get("ticker") or args.ticker).upper(),
                "profile": args.trader_profile,
                "provider": provider,
            },
        )
        if progress is not None:
            progress.step("CEO LLM call: completed")
    except Exception as exc:
        if progress is not None:
            progress.step(f"CEO LLM call: error={exc}")
        return {
            "status": "error",
            "model": model,
            "provider": provider,
            "decision": None,
            "portfolio_context": portfolio_context,
            "long_term_source_synthesis": source_synthesis,
            "strategy_knowledge_context": strategy_context,
            "llm_prompt_payload": payload,
            "reason": str(exc),
        }
    return {
        "status": "executed",
        "model": model,
        "provider": provider,
        "decision": decision,
        "portfolio_context": portfolio_context,
        "long_term_source_synthesis": source_synthesis,
        "strategy_knowledge_context": strategy_context,
        "llm_prompt_payload": payload,
        "llm_raw_response": raw_response,
        "policy": (
            "Autonomous LLM is the final advice/action layer for the forecast report. "
            "Deterministic risk, validation, market-hours, and broker gates are reported as execution constraints."
        ),
    }


def _apply_autonomous_llm_to_report(report: dict, autonomous: dict) -> None:
    decision = autonomous.get("decision") if isinstance(autonomous.get("decision"), dict) else {}
    execution_gate = _autonomous_execution_gate(report, decision)
    report["autonomous_llm_trader"] = autonomous
    report.setdefault("decision_view", {})["autonomous_llm_trader"] = autonomous
    report["decision_view"]["autonomous_execution_gate"] = execution_gate
    if autonomous.get("status") == "executed" and decision.get("decision") in {"Buy", "Hold", "Sell"}:
        report["suggested_action"] = decision["decision"]
        report["decision_view"]["final_governed_action"] = decision["decision"]
        report["llm_final_decision"] = decision
        report["final_advice"] = decision.get("final_advice", {})
        reasoning = report.setdefault("final_decision_reasoning", {})
        reasoning.update(
            {
                "final_action": decision["decision"],
                "decision_source": "autonomous_llm_trader",
                "autonomous_llm_status": autonomous.get("status"),
                "autonomous_llm_model": autonomous.get("model"),
                "autonomous_llm_confidence": decision.get("confidence"),
                "autonomous_llm_analysis": decision.get("analysis"),
                "autonomous_llm_rationale": decision.get("llm_rationale") or decision.get("decision_reasoning"),
                "llm_technical_read": decision.get("technical_read"),
                "llm_market_context_read": decision.get("market_context_read"),
                "llm_sentiment": decision.get("sentiment"),
                "final_advice": decision.get("final_advice", {}),
                "execution_gate": execution_gate,
            }
        )
    else:
        report["decision_view"].setdefault("final_governed_action", report.get("suggested_action"))
        report.setdefault("final_decision_reasoning", {})["autonomous_llm_status"] = autonomous.get("status")
        report["final_decision_reasoning"]["autonomous_llm_error"] = autonomous.get("reason")


def _autonomous_execution_gate(report: dict, decision: dict) -> dict:
    action = str(decision.get("decision") or "").title()
    blocks: list[str] = []
    warnings: list[str] = []
    chapter_18 = report.get("decision_view", {}).get("chapter_18_tactical_problem", {})
    rule_gate = chapter_18.get("rule_gate", {}) if isinstance(chapter_18.get("rule_gate"), dict) else {}
    blocks.extend(str(item) for item in rule_gate.get("hard_blockers", []) or [])
    diagnostics = report.get("technical_view", {}).get("decision_diagnostics", {})
    if action in {"Buy", "Sell"}:
        blocks.extend(str(item) for item in diagnostics.get("blocking_reasons", []) or [])
    chapter_19 = report.get("operations_view", {}).get("chapter_19_validation", {})
    if chapter_19.get("status") == "fail":
        blocks.append("Chapter 19 validation failed; execution is blocked even though autonomous advice is final.")
    portfolio_gate = report.get("portfolio_view", {}).get("chapter_31_42_portfolio_capital_risk", {}).get("portfolio_capital_gate", {})
    allocation_status = portfolio_gate.get("allocation_status")
    if allocation_status in {"blocked", "not_allowed"}:
        blocks.append(f"Portfolio capital gate blocks execution: {allocation_status}.")
    if report.get("risk_level") == "High" and action in {"Buy", "Sell"}:
        warnings.append("Overall risk is High; execution should require explicit confirmation.")
    return {
        "advice_action": action or None,
        "execution_allowed": not blocks,
        "execution_blocks": blocks,
        "warnings": warnings,
        "policy": "Autonomous LLM decision is final advice. This gate only states whether execution is blocked by deterministic safety constraints.",
    }


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
    autonomous = report.get("autonomous_llm_trader", {})
    autonomous_decision = autonomous.get("decision", {}) if isinstance(autonomous.get("decision"), dict) else {}
    final_advice = autonomous_decision.get("final_advice", {}) if isinstance(autonomous_decision.get("final_advice"), dict) else {}
    autonomous_gate = decision_view.get("autonomous_execution_gate", {})
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
            "Autonomous LLM CEO: "
            f"{autonomous.get('status', 'Unavailable')} "
            f"decision={autonomous_decision.get('decision', 'Unavailable')} "
            f"confidence={_format_number(autonomous_decision.get('confidence'))}"
        ),
        f"Autonomous Advice: {final_advice.get('headline', 'Unavailable')}",
        (
            "Autonomous Levels: "
            f"buy_now={_format_price(final_advice.get('buy_now_price'))} "
            f"buy_lower={_format_price(final_advice.get('buy_lower_price'))} "
            f"buy_above={_format_price(final_advice.get('buy_above_breakout_price'))} "
            f"sell_trim={_format_price(final_advice.get('sell_or_trim_price'))} "
            f"take_profit={_format_price(final_advice.get('take_profit_price'))} "
            f"stop={_format_price(final_advice.get('stop_loss_price'))}"
        ),
        f"Execution Allowed: {autonomous_gate.get('execution_allowed', 'Unavailable')}",
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
    lines.extend(_autonomous_llm_reasoning_lines(autonomous_decision, autonomous))
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


def _autonomous_llm_reasoning_lines(decision: dict, autonomous: dict) -> list[str]:
    if not decision:
        return []
    source_synthesis = autonomous.get("long_term_source_synthesis", {}) if isinstance(autonomous.get("long_term_source_synthesis"), dict) else {}
    synthesis_payload = source_synthesis.get("synthesis", {}) if isinstance(source_synthesis.get("synthesis"), dict) else {}
    strategy_context = autonomous.get("strategy_knowledge_context", {}) if isinstance(autonomous.get("strategy_knowledge_context"), dict) else {}
    strategy_synthesis = strategy_context.get("synthesis", {}) if isinstance(strategy_context.get("synthesis"), dict) else {}
    lines = ["Autonomous LLM Reasoning:"]
    if source_synthesis:
        lines.append(
            "Source Synthesis: "
            f"status={source_synthesis.get('status', 'Unavailable')} "
            f"model={source_synthesis.get('model', 'Unavailable')} "
            f"coverage_all_passed={source_synthesis.get('llm_evidence_manifest', {}).get('all_scraped_sections_passed', 'Unavailable')}"
        )
        if synthesis_payload:
            lines.extend(
                [
                    _json_block("Source Synthesis Output", synthesis_payload),
                ]
            )
    if strategy_context:
        index_status = strategy_context.get("index_status", {}) if isinstance(strategy_context.get("index_status"), dict) else {}
        lines.append(
            "Strategy Knowledge: "
            f"status={strategy_context.get('status', 'Unavailable')} "
            f"backend={index_status.get('backend', 'Unavailable')} "
            f"chunks={len(strategy_context.get('retrieved_chunks', []) or [])}"
        )
        if strategy_synthesis:
            lines.append(_json_block("Strategy Knowledge Synthesis", strategy_synthesis))
    analysis = decision.get("analysis")
    if analysis:
        lines.append(_json_block("CEO Analysis", analysis))
    rationale = decision.get("llm_rationale")
    if rationale:
        lines.extend(["CEO LLM Rationale:", str(rationale)])
    reasoning = decision.get("decision_reasoning")
    if reasoning:
        lines.extend(["CEO Decision Reasoning:", str(reasoning)])
    final_advice = decision.get("final_advice")
    if final_advice:
        lines.append(_json_block("CEO Final Advice", final_advice))
    lines.append("")
    return lines


def _final_ceo_decision_block(report: dict) -> str:
    autonomous = report.get("autonomous_llm_trader", {})
    decision = report.get("llm_final_decision")
    if not isinstance(decision, dict):
        decision = autonomous.get("decision") if isinstance(autonomous.get("decision"), dict) else {}
    final_advice = decision.get("final_advice", {}) if isinstance(decision.get("final_advice"), dict) else {}
    gate = report.get("decision_view", {}).get("autonomous_execution_gate", {})
    reasoning = decision.get("decision_reasoning") or report.get("final_decision_reasoning", {}).get("autonomous_llm_rationale")
    rationale = decision.get("llm_rationale")
    analysis = decision.get("analysis")
    lines = [
        "",
        "=" * 72,
        "CEO FINAL DECISION",
        "=" * 72,
        f"Status: {autonomous.get('status', 'Unavailable')}",
        f"Model: {autonomous.get('model', 'Unavailable')}",
        f"Decision: {decision.get('decision', report.get('suggested_action', 'Unavailable'))}",
        f"Confidence: {_format_number(decision.get('confidence'))}",
        f"Risk Level: {report.get('risk_level', 'Unavailable')}",
        f"Execution Allowed: {gate.get('execution_allowed', 'Unavailable')}",
    ]
    blocks = gate.get("execution_blocks") if isinstance(gate, dict) else None
    if blocks:
        lines.append("Execution Blocks:")
        lines.extend(f"- {item}" for item in blocks)
    warnings = gate.get("warnings") if isinstance(gate, dict) else None
    if warnings:
        lines.append("Execution Warnings:")
        lines.extend(f"- {item}" for item in warnings)
    if analysis:
        lines.append(_json_block("Analysis", analysis))
    if rationale:
        lines.extend(["Rationale:", str(rationale)])
    if reasoning:
        lines.extend(["Decision Reasoning:", str(reasoning)])
    if final_advice:
        lines.append(_json_block("Final Advice", final_advice))
    else:
        fallback_advice = report.get("final_decision_reasoning", {}).get("final_advice")
        if fallback_advice:
            lines.append(_json_block("Final Advice", fallback_advice))
    lines.append("=" * 72)
    return "\n".join(lines)


def _json_block(title: str, payload: object) -> str:
    return f"{title}:\n{json.dumps(payload, indent=2, sort_keys=True, default=str)}"


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
