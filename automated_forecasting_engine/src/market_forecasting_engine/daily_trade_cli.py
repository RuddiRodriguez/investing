from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from market_forecasting_engine.calendar import summarize_calendar_alignment
from market_forecasting_engine.daily_trade import (
    DailyTradeConfig,
    add_trading_bars,
    build_daily_trade_plan,
    build_intraday_chart_confirmation,
    build_intraday_feature_frame,
    build_intraday_risk_context,
    infer_bar_interval_minutes,
)
from market_forecasting_engine.data import normalize_price_frame
from market_forecasting_engine.data_manifest import build_data_manifest
from market_forecasting_engine.data_providers import DataRequest, load_prices_with_provider
from market_forecasting_engine.data_quality import build_data_quality_report
from market_forecasting_engine.dip_buy import annotate_mean_reversion_dip_buy, historical_reversal_stats
from market_forecasting_engine.governance import write_audit_bundle
from market_forecasting_engine.long_term_enrichment import (
    DEFAULT_LONG_TERM_SOURCE_PROVIDERS,
    enrich_prices_with_long_term_sources,
    long_term_context_manifest_entry,
    parse_long_term_source_providers,
)
from market_forecasting_engine.llm_trader.prompts import autonomous_trader
from market_forecasting_engine.llm_trader.profiles import trader_profiles
from market_forecasting_engine.llm_trader.responses_api import call_response, response_payload
from market_forecasting_engine.llm_trader.run import build_technical_packet, load_env, openai_client_for_provider, resolve_llm_model, resolve_llm_provider
from market_forecasting_engine.openai_models import DEFAULT_OPENAI_MODEL, DEFAULT_REASONING_EFFORT
from market_forecasting_engine.models import HistoricalMeanReturn, RecentMeanReturn, default_candidates
from market_forecasting_engine.options_decision import (
    OptionsDecisionConfig,
    build_options_decision,
    write_options_artifacts,
)
from market_forecasting_engine.pipeline import ForecastingEngine
from market_forecasting_engine.plots import write_daily_trade_plot_artifacts, write_plot_artifacts
from market_forecasting_engine.risk_profiles import risk_profile_for_name
from market_forecasting_engine.schema import ForecastConfig
from market_forecasting_engine.security_master import resolve_security_metadata
from market_forecasting_engine.validation import select_candidate, validate_candidates, validation_summaries_as_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a same-session intraday trade plan.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, for example AAPL.")
    parser.add_argument("--csv", help="Optional local intraday OHLCV CSV.")
    parser.add_argument("--provider", default=None, help="Data provider: yahoo or csv.")
    parser.add_argument("--start", default=None, help="Provider start datetime/date.")
    parser.add_argument("--end", default=None, help="Provider end datetime/date.")
    parser.add_argument("--interval", default="5m", help="Intraday interval, for example 1m, 5m, or 15m.")
    parser.add_argument("--target-column", default="close", help="Price column used for decisions.")
    parser.add_argument("--opening-range-bars", type=int, default=6, help="Opening range length in bars.")
    parser.add_argument("--minimum-score-to-trade", type=float, default=2.0, help="Absolute signal score needed to trade.")
    parser.add_argument("--risk-reward", type=float, default=1.8, help="Take-profit distance divided by stop distance.")
    parser.add_argument("--stop-atr-multiple", type=float, default=1.2, help="ATR multiple used for stop distance.")
    parser.add_argument("--max-hold-bars", type=int, default=24, help="Maximum same-session hold length in bars.")
    parser.add_argument("--forecast-hours", default="1,2,4", help="Comma-separated hourly forecast horizons.")
    parser.add_argument("--training-lookback-days", type=int, default=45, help="Minimum Yahoo intraday lookback used when the requested start has too few rows for modelling.")
    parser.add_argument("--max-training-rows", type=int, default=3500, help="Maximum most-recent intraday rows used for modelling; set 0 to disable.")
    parser.add_argument("--selection-metric", default="mae", help="Model selection metric for the advanced hourly forecasting engine.")
    parser.add_argument("--no-lightgbm", action="store_true", help="Disable optional LightGBM candidate.")
    parser.add_argument("--no-statistical-models", action="store_true", help="Disable optional ARIMA/SARIMA/GARCH/VAR candidates.")
    parser.add_argument("--search-level", choices=("fast", "expanded"), default="fast", help="Candidate tuning breadth.")
    parser.add_argument("--risk-profile", choices=("conservative", "medium", "aggressive"), default="medium", help="Risk profile used by production and options gates.")
    parser.add_argument("--transaction-cost-bps", type=float, default=2.0, help="Estimated transaction cost in bps.")
    parser.add_argument("--enable-options-decision", action="store_true", help="Score options candidates against the forecast distribution.")
    parser.add_argument("--options-chain-csv", default=None, help="Optional options chain CSV with strike, option_type, expiry, bid/ask or mid.")
    parser.add_argument("--options-strike-pct-range", type=float, default=0.04, help="Synthetic options chain strike range around spot.")
    parser.add_argument("--options-strike-count", type=int, default=17, help="Synthetic options chain strike count.")
    parser.add_argument("--options-min-edge-pct", type=float, default=None, help="Minimum expected value over ask for options candidates; defaults from risk profile.")
    parser.add_argument("--options-max-spread-pct", type=float, default=None, help="Maximum bid/ask spread over mid for options candidates; defaults from risk profile.")
    parser.add_argument("--options-min-probability-breakeven", type=float, default=None, help="Minimum probability of finishing beyond option breakeven; defaults from risk profile.")
    parser.add_argument("--enable-llm-decision", action="store_true", default=True, help="Run the autonomous LLM trader as a final governed decision layer. Enabled by default.")
    parser.add_argument("--disable-llm-decision", action="store_false", dest="enable_llm_decision", help="Disable the default autonomous LLM decision layer for this run.")
    parser.add_argument("--llm-dry-run", action="store_true", help="Build the LLM decision packet without calling the LLM.")
    parser.add_argument("--llm-provider", choices=("openai", "huggingface", "bedrock", "llm_studio"), default="openai")
    parser.add_argument("--llm-model", default=None, help=f"Defaults to OPENAI_MODEL or {DEFAULT_OPENAI_MODEL}.")
    parser.add_argument("--llm-reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument("--llm-env-file", default=None)
    parser.add_argument("--llm-no-web-search", action="store_true")
    parser.add_argument("--llm-search-context-size", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--disable-long-term-sources", action="store_true", help="Disable default long-term source enrichment for this run.")
    parser.add_argument(
        "--long-term-source-providers",
        default=",".join(DEFAULT_LONG_TERM_SOURCE_PROVIDERS),
        help="Comma-separated long-term source providers to call.",
    )
    parser.add_argument("--long-term-source-env-file", default=None, help="Optional .env file for long-term source API keys.")
    parser.add_argument("--long-term-source-output-dir", default=None, help="Optional artifact directory for raw long-term source payloads.")
    parser.add_argument("--long-term-source-snapshot-dir", default=None, help="Durable directory for point-in-time long-term source snapshots.")
    parser.add_argument("--trader-name", default="daily_intraday_trader")
    parser.add_argument("--holding-status", choices=("not_owned", "owned"), default="not_owned")
    parser.add_argument("--entry-price", type=float, default=None)
    parser.add_argument("--quantity", type=float, default=None)
    parser.add_argument("--position-value", type=float, default=None)
    parser.add_argument("--account-equity", type=float, default=None)
    parser.add_argument("--portfolio-notes", default="")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--output-dir", default=None, help="Optional directory for JSON and daily-trade plot artifacts.")
    parser.add_argument("--no-plot", action="store_true", help="Disable daily-trade chart artifact generation.")
    args = parser.parse_args()

    provider_name = (args.provider or ("csv" if args.csv else "yahoo")).lower()
    if args.csv:
        provider_name = "csv"
    forecast_hours = tuple(float(value.strip()) for value in args.forecast_hours.split(",") if value.strip())
    result = load_prices_with_provider(
        provider_name,
        DataRequest(
            ticker=args.ticker,
            start=args.start,
            end=args.end,
            interval=args.interval,
            target_column=args.target_column,
            source_path=args.csv,
        ),
        store=None,
        use_cache=False,
        refresh_cache=True,
    )
    result = _ensure_training_history(result, args, provider_name, forecast_hours)
    prices = normalize_price_frame(result.frame, target_column=args.target_column)
    prices = _limit_training_rows(prices, args.max_training_rows)
    config = DailyTradeConfig(
        ticker=args.ticker,
        interval=args.interval,
        target_column=args.target_column,
        opening_range_bars=args.opening_range_bars,
        minimum_score_to_trade=args.minimum_score_to_trade,
        risk_reward=args.risk_reward,
        stop_atr_multiple=args.stop_atr_multiple,
        max_hold_bars=args.max_hold_bars,
        forecast_hours=forecast_hours,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    trade_report = build_daily_trade_plan(prices, config)
    trade_report["data_provider"] = result.metadata
    output_dir = _resolve_output_dir(args.output_dir, args.output)
    report = _run_advanced_hourly_forecast(prices, trade_report, args, result.metadata, forecast_hours)
    risk_profile = risk_profile_for_name(args.risk_profile)
    report["risk_profile"] = risk_profile.to_dict()
    _annotate_production_decision(report, prices, args.target_column, risk_profile_name=args.risk_profile)
    _annotate_mean_reversion_dip_buy(report, prices, args.target_column, risk_profile_name=args.risk_profile)
    if args.enable_options_decision or args.options_chain_csv:
        report["options_decision"] = build_options_decision(
            report,
            prices,
            target_column=args.target_column,
            config=OptionsDecisionConfig(
                ticker=args.ticker,
                risk_profile=args.risk_profile,
                chain_csv=args.options_chain_csv,
                strike_pct_range=args.options_strike_pct_range,
                strike_count=args.options_strike_count,
                min_edge_pct=args.options_min_edge_pct if args.options_min_edge_pct is not None else risk_profile.options_min_edge_pct,
                max_spread_pct=args.options_max_spread_pct if args.options_max_spread_pct is not None else risk_profile.options_max_spread_pct,
                min_probability_above_breakeven=(
                    args.options_min_probability_breakeven
                    if args.options_min_probability_breakeven is not None
                    else risk_profile.options_min_probability_breakeven
                ),
            ),
        )
    if args.enable_llm_decision:
        report["llm_trader"] = _run_daily_llm_decision(report, args, dry_run=bool(args.llm_dry_run))
    report.setdefault("decision_view", {})["final_decision_reasoning"] = _daily_final_decision_reasoning(report, args)
    report["final_decision_reasoning"] = report["decision_view"]["final_decision_reasoning"]
    if output_dir is not None and not args.no_plot:
        report.setdefault("artifacts", {}).update(write_plot_artifacts(report, prices, output_dir, target_column=args.target_column))
        report["artifacts"].update(_write_hour_named_validation_aliases(report, output_dir))
        report["artifacts"].update(
            write_daily_trade_plot_artifacts(
                trade_report,
                prices,
                output_dir=output_dir,
                target_column=args.target_column,
            )
        )
        report["daily_trade_view"]["artifacts"] = {
            key: value for key, value in report["artifacts"].items() if key.startswith("daily_trade")
        }
    if output_dir is not None:
        if "options_decision" in report:
            report.setdefault("artifacts", {}).update(write_options_artifacts(report["options_decision"], output_dir))
        report.setdefault("artifacts", {}).update(_update_forecast_ledger(report, prices, output_dir, args.target_column))
        report.setdefault("artifacts", {}).update(write_audit_bundle(report, output_dir))

    text = json.dumps(report, indent=2, default=str)
    output_path = Path(args.output) if args.output else (output_dir / "daily_trade_report.json" if output_dir else None)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        _write_forecast_csv(report, output_path.parent / "daily_trade_forecasts.csv")
    print(text)


def _ensure_training_history(result, args: argparse.Namespace, provider_name: str, forecast_hours: tuple[float, ...]):
    if provider_name != "yahoo" or args.csv:
        return result
    prices = normalize_price_frame(result.frame, target_column=args.target_column)
    interval_minutes = infer_bar_interval_minutes(prices.index) or _interval_to_minutes(args.interval) or 5.0
    max_horizon_bars = max(1, max(int(round(hours * 60.0 / interval_minutes)) for hours in forecast_hours))
    required_rows = max(600, max_horizon_bars * 12)
    fallback_start = (pd.Timestamp.now(tz=UTC).tz_convert(None) - pd.Timedelta(days=_safe_yahoo_intraday_lookback_days(args))).date().isoformat()
    if len(prices) >= required_rows and (args.start is None or pd.Timestamp(args.start) <= pd.Timestamp(fallback_start)):
        return result

    try:
        expanded = load_prices_with_provider(
            provider_name,
            DataRequest(
                ticker=args.ticker,
                start=fallback_start,
                end=args.end,
                interval=args.interval,
                target_column=args.target_column,
            ),
            store=None,
            use_cache=False,
            refresh_cache=True,
        )
    except Exception as exc:
        result.metadata["training_history_expanded"] = False
        result.metadata["training_history_expand_error"] = str(exc)
        result.metadata["training_history_required_rows"] = int(required_rows)
        return result
    expanded.metadata["initial_request_rows"] = int(len(prices))
    expanded.metadata["training_history_expanded"] = True
    expanded.metadata["training_history_start"] = fallback_start
    expanded.metadata["training_history_required_rows"] = int(required_rows)
    return expanded


def _limit_training_rows(prices: pd.DataFrame, max_training_rows: int) -> pd.DataFrame:
    if max_training_rows <= 0 or len(prices) <= max_training_rows:
        return prices
    return prices.tail(max_training_rows).copy()


def _safe_yahoo_intraday_lookback_days(args: argparse.Namespace) -> int:
    interval_minutes = _interval_to_minutes(args.interval) or 1440.0
    requested = max(1, int(args.training_lookback_days))
    if interval_minutes < 60:
        return min(requested, 45)
    if interval_minutes < 1440:
        return min(requested, 700)
    return requested


def _run_advanced_hourly_forecast(
    prices: pd.DataFrame,
    trade_report: dict,
    args: argparse.Namespace,
    provider_metadata: dict,
    forecast_hours: tuple[float, ...],
) -> dict:
    output_dir = _resolve_output_dir(args.output_dir, args.output)
    long_term_context = None
    long_term_snapshot_metadata = None
    if not args.disable_long_term_sources:
        long_term_output_dir = (
            Path(args.long_term_source_output_dir)
            if args.long_term_source_output_dir
            else ((output_dir / "long_term_sources") if output_dir else None)
        )
        prices, long_term_context, long_term_snapshot_metadata = enrich_prices_with_long_term_sources(
            ticker=args.ticker,
            prices=prices,
            target_column=args.target_column,
            enabled=True,
            providers=parse_long_term_source_providers(args.long_term_source_providers),
            env_file=args.long_term_source_env_file or args.llm_env_file,
            output_dir=long_term_output_dir,
            snapshot_dir=args.long_term_source_snapshot_dir,
            data_store=None,
            start_date=args.start,
            end_date=args.end,
        )
    interval_minutes = trade_report.get("interval_minutes") or infer_bar_interval_minutes(prices.index) or _interval_to_minutes(args.interval)
    if interval_minutes is None:
        interval_minutes = 5.0
    horizons = tuple(max(1, int(round(hours * 60.0 / float(interval_minutes)))) for hours in forecast_hours)
    n_rows = len(prices)
    max_horizon = max(horizons)
    min_training_rows = max(30, min(180, max(20, n_rows // 2)))
    validation_window = max(max_horizon, min(max_horizon * 3, max(10, n_rows // 5)))
    step_size = max_horizon
    security_metadata = resolve_security_metadata(ticker=args.ticker, prices=prices, provider_metadata=provider_metadata)
    data_quality_report = build_data_quality_report(prices, target_column=args.target_column)
    calendar_summary = summarize_calendar_alignment(prices)
    data_manifest = build_data_manifest(
        prices=prices,
        ticker=args.ticker,
        target_column=args.target_column,
        provider=provider_metadata.get("provider", "unknown"),
        request={**provider_metadata.get("request", {}), "forecast_hours": list(forecast_hours), "advanced_hourly_model": True},
        security_metadata=security_metadata,
        calendar_summary=calendar_summary,
    )
    if long_term_context:
        data_manifest["long_term_sources"] = long_term_context_manifest_entry(
            long_term_context,
            long_term_snapshot_metadata,
        )
    config = ForecastConfig(
        ticker=args.ticker,
        horizons=horizons,
        target_column=args.target_column,
        min_training_rows=min_training_rows,
        validation_window=validation_window,
        step_size=step_size,
        max_splits=6,
        selection_metric=args.selection_metric,
        transaction_cost_bps=args.transaction_cost_bps,
        include_lightgbm=not args.no_lightgbm,
        include_statistical_models=not args.no_statistical_models,
        search_level=args.search_level,
        tactical_profile="short_term",
        forecast_interval=args.interval,
        forecast_interval_minutes=float(interval_minutes),
        enable_long_term_sources=not args.disable_long_term_sources,
        long_term_source_providers=parse_long_term_source_providers(args.long_term_source_providers),
    )
    try:
        report = ForecastingEngine(config).run(
            prices,
            data_manifest=data_manifest,
            data_quality_report=data_quality_report,
            security_metadata=security_metadata,
            long_term_context=long_term_context,
        )
    except Exception as exc:
        report = _run_compact_intraday_model_report(
            prices=prices,
            trade_report=trade_report,
            args=args,
            provider_metadata=provider_metadata,
            data_manifest=data_manifest,
            data_quality_report=data_quality_report,
            security_metadata=security_metadata,
            forecast_hours=forecast_hours,
            horizons=horizons,
            interval_minutes=float(interval_minutes),
            fallback_reason=str(exc),
        )
    report["mode"] = "advanced_intraday_hourly_forecast"
    report["daily_trade_view"] = trade_report
    report["data_provider"] = provider_metadata
    report["forecast_hours"] = list(forecast_hours)
    _annotate_validation_gates(report, forecast_hours=forecast_hours)
    report["generated_at_utc"] = datetime.now(UTC).isoformat()
    return report


def _run_compact_intraday_model_report(
    *,
    prices: pd.DataFrame,
    trade_report: dict,
    args: argparse.Namespace,
    provider_metadata: dict,
    data_manifest: dict,
    data_quality_report: dict,
    security_metadata: dict,
    forecast_hours: tuple[float, ...],
    horizons: tuple[int, ...],
    interval_minutes: float,
    fallback_reason: str,
) -> dict:
    close = prices[args.target_column].astype(float)
    features = _intraday_model_features(prices, args.target_column)
    log_close = np.log(close.replace(0, np.nan))
    latest_features = features.iloc[[-1]].ffill().bfill()
    forecasts = []
    candidate_results: dict[str, list[dict[str, object]]] = {}
    selected_validation_predictions: dict[str, list[dict[str, object]]] = {}
    model_cards: dict[str, object] = {}

    for hours, horizon in zip(forecast_hours, horizons):
        target = log_close.shift(-horizon) - log_close
        training_frame = pd.concat([features, target.rename("target")], axis=1).replace([np.inf, -np.inf], np.nan)
        training_frame = training_frame.ffill().dropna(subset=["target"])
        training_frame = training_frame.dropna(axis=1, how="all").dropna()
        feature_columns = [column for column in training_frame.columns if column != "target"]
        if len(training_frame) < 30 or not feature_columns:
            selected_model = HistoricalMeanReturn().fit(pd.DataFrame(index=training_frame.index), training_frame["target"])
            selected_name = selected_model.name
            selected_family = selected_model.family
            prediction = float(selected_model.predict(latest_features.iloc[:, 0:0])[0])
            validation_metrics = {"mae": float(training_frame["target"].abs().mean() if len(training_frame) else 0.0)}
            residual_std = float(training_frame["target"].std() if len(training_frame) > 1 else 0.002)
            candidate_results[str(horizon)] = []
            selected_validation_predictions[str(horizon)] = []
        else:
            candidate_pool = default_candidates(
                include_lightgbm=not args.no_lightgbm,
                include_statistical_models=False,
                search_level=args.search_level,
            )
            candidate_pool.extend([HistoricalMeanReturn(), RecentMeanReturn(window=12, name="recent_mean_return_12_bars")])
            validation_window = max(horizon, min(horizon * 3, max(12, len(training_frame) // 5)))
            step_size = max(horizon, 1)
            validation_results = validate_candidates(
                candidates=candidate_pool,
                features=training_frame[feature_columns],
                target=training_frame["target"],
                horizon_days=horizon,
                min_training_rows=max(40, min(240, len(training_frame) // 2)),
                validation_window=validation_window,
                step_size=step_size,
                max_splits=5,
                purge_window=horizon,
                embargo_window=0,
                final_holdout_fraction=0.10,
            )
            if not validation_results:
                validation_results = validate_candidates(
                    candidates=[HistoricalMeanReturn(), RecentMeanReturn(window=12, name="recent_mean_return_12_bars")],
                    features=training_frame[feature_columns],
                    target=training_frame["target"],
                    horizon_days=horizon,
                    min_training_rows=max(20, min(120, len(training_frame) // 2)),
                    validation_window=max(horizon, min(horizon * 2, len(training_frame) // 4)),
                    step_size=max(horizon, 1),
                    max_splits=3,
                    final_holdout_fraction=0.0,
                )
            selected_candidate, selected_summary, selected_predictions = select_candidate(validation_results, args.selection_metric)
            fitted = selected_candidate.clone().fit(training_frame[feature_columns], training_frame["target"])
            prediction = float(fitted.predict(latest_features.reindex(columns=feature_columns))[0])
            selected_name = selected_summary.model_name
            selected_family = selected_summary.model_family
            validation_metrics = selected_summary.metrics
            residuals = selected_predictions["actual"] - selected_predictions["predicted"]
            residual_std = float(residuals.std()) if len(residuals) > 1 else float(training_frame["target"].std())
            candidate_results[str(horizon)] = validation_summaries_as_dict(validation_results)
            selected_validation_predictions[str(horizon)] = _compact_validation_records(
                selected_predictions,
                close=close,
                horizon=horizon,
            )

        forecast_timestamp = add_trading_bars(pd.DatetimeIndex(prices.index), pd.Timestamp(prices.index[-1]), horizon, interval_minutes)
        interval_width = max(1.28 * residual_std, 0.001)
        predicted_price = float(close.iloc[-1] * np.exp(prediction))
        lower_price = float(close.iloc[-1] * np.exp(prediction - interval_width))
        upper_price = float(close.iloc[-1] * np.exp(prediction + interval_width))
        validation_gate = _validation_gate(validation_metrics)
        forecasts.append(
            {
                "horizon_days": int(horizon),
                "horizon_hours": float(hours),
                "forecast_date": forecast_timestamp.isoformat(),
                "selected_model": selected_name,
                "selected_model_family": selected_family,
                "selection_metric": args.selection_metric,
                "expected_log_return": float(prediction),
                "expected_return": float(np.expm1(prediction)),
                "predicted_price": predicted_price,
                "lower_price": lower_price,
                "upper_price": upper_price,
                "expected_direction": "Up" if prediction > 0 else "Down" if prediction < 0 else "Flat",
                "directional_confidence": float(min(0.95, max(0.50, 0.50 + abs(prediction) / max(interval_width, 1e-6) * 0.15))),
                "confidence_interval_method": "compact_intraday_residual_std",
                "calibration_sample_size": int(len(training_frame)),
                "trade_quality": {},
                "validation_gate": validation_gate,
                "trade_allowed": bool(validation_gate["trade_allowed"]),
                "validation_metrics": validation_metrics,
            }
        )
        model_cards[str(horizon)] = {
            "ticker": args.ticker.upper(),
            "horizon_bars": int(horizon),
            "horizon_hours": float(hours),
            "selected_model": selected_name,
            "selected_model_family": selected_family,
            "selection_metric": args.selection_metric,
            "training_rows": int(len(training_frame)),
            "created_at_utc": datetime.now(UTC).isoformat(),
        }

    return {
        "ticker": args.ticker.upper(),
        "mode": "advanced_intraday_hourly_forecast_compact",
        "as_of_date": str(pd.Timestamp(prices.index[-1]).date()),
        "as_of_timestamp": pd.Timestamp(prices.index[-1]).isoformat(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "current_price": float(close.iloc[-1]),
        "horizons": list(horizons),
        "forecast_hours": list(forecast_hours),
        "forecast_interval": args.interval,
        "forecast_interval_minutes": float(interval_minutes),
        "forecasts": forecasts,
        "suggested_action": trade_report.get("trade_plan", {}).get("action", "no_trade"),
        "risk_level": "Intraday",
        "risk_warning": "Intraday forecasts are short-horizon estimates and should be treated as tactical, not investment advice.",
        "daily_trade_view": trade_report,
        "data_provider": provider_metadata,
        "data_manifest": data_manifest,
        "diagnostics": {
            "data_quality": data_quality_report,
            "fallback_from_full_engine": True,
            "fallback_reason": fallback_reason,
            "candidate_results": candidate_results,
            "selected_validation_predictions": selected_validation_predictions,
            "validation_gates": {str(item["horizon_days"]): item["validation_gate"] for item in forecasts},
        },
        "governance": {"model_cards": model_cards},
        "technical_view": {},
        "decision_view": {},
        "portfolio_view": {},
        "operations_view": {},
        "selection_view": {},
        "trade_risk_view": {},
        "discipline_view": {},
    }


def _intraday_model_features(prices: pd.DataFrame, target_column: str) -> pd.DataFrame:
    return build_intraday_feature_frame(prices, target_column=target_column)


def _validation_gate(metrics: dict[str, object]) -> dict[str, object]:
    directional_accuracy = float(metrics.get("directional_accuracy", 0.0) or 0.0)
    mae = float(metrics.get("mae", 999.0) or 999.0)
    holdout_directional_accuracy = float(metrics.get("holdout_directional_accuracy", directional_accuracy) or 0.0)
    holdout_mae = float(metrics.get("holdout_mae", mae) or mae)
    reasons = []
    if directional_accuracy < 0.45:
        reasons.append("low_walk_forward_directional_accuracy")
    if holdout_directional_accuracy < 0.45:
        reasons.append("low_holdout_directional_accuracy")
    if mae > 0.012:
        reasons.append("high_walk_forward_mae")
    if holdout_mae > 0.012:
        reasons.append("high_holdout_mae")
    return {
        "trade_allowed": not reasons,
        "status": "pass" if not reasons else "weak_validation",
        "reasons": reasons,
        "directional_accuracy": directional_accuracy,
        "holdout_directional_accuracy": holdout_directional_accuracy,
        "mae": mae,
        "holdout_mae": holdout_mae,
    }


def _annotate_validation_gates(report: dict, forecast_hours: tuple[float, ...]) -> None:
    gates = {}
    for forecast, hours in zip(report.get("forecasts", []), forecast_hours):
        forecast["horizon_hours"] = float(hours)
        gate = _validation_gate(forecast.get("validation_metrics", {}))
        forecast["validation_gate"] = gate
        forecast["trade_allowed"] = bool(gate["trade_allowed"])
        gates[str(forecast.get("horizon_days"))] = gate
    report.setdefault("diagnostics", {})["validation_gates"] = gates


def _annotate_production_decision(report: dict, prices: pd.DataFrame, target_column: str, risk_profile_name: str = "medium") -> None:
    risk_profile = risk_profile_for_name(risk_profile_name)
    risk_context = build_intraday_risk_context(prices, target_column=target_column)
    chart_confirmation = build_intraday_chart_confirmation(prices, target_column=target_column)
    provider = str(report.get("data_provider", {}).get("provider", "")).lower()
    provider_warning = _provider_quality_warning(provider, str(report.get("forecast_interval", "")))
    allowed_count = 0
    for forecast in report.get("forecasts", []):
        gate = forecast.get("validation_gate", {})
        reasons = list(gate.get("reasons", []))
        if risk_context["risk_score"] > risk_profile.maximum_risk_score:
            reasons.append("intraday_risk_score_too_high")
        if forecast.get("directional_confidence", 0.0) < risk_profile.minimum_directional_confidence:
            reasons.append("low_directional_confidence")
        expected_return = abs(float(forecast.get("expected_return", 0.0) or 0.0))
        if expected_return < max(
            risk_profile.minimum_edge_fraction,
            float(forecast.get("validation_metrics", {}).get("mae", 0.0)) * risk_profile.validation_mae_edge_multiplier,
        ):
            reasons.append("forecast_edge_too_small")
        if provider_warning:
            reasons.append("provider_not_production_intraday_grade")
        production_gate = {
            "trade_allowed": not reasons,
            "status": "pass" if not reasons else "blocked",
            "reasons": sorted(set(reasons)),
            "risk_score": risk_context["risk_score"],
            "uncertainty": risk_context["uncertainty"],
        }
        forecast["production_gate"] = production_gate
        forecast["trade_allowed"] = bool(production_gate["trade_allowed"])
        forecast["risk_adjusted_expected_return"] = float(forecast.get("expected_return", 0.0) or 0.0) - risk_context["risk_penalty"]
        if forecast["trade_allowed"]:
            allowed_count += 1

    report.setdefault("decision_view", {})["production_gate"] = {
        "allowed_forecast_count": allowed_count,
        "risk_context": risk_context,
        "chart_confirmation": chart_confirmation,
        "provider_quality_warning": provider_warning,
        "risk_profile": risk_profile.to_dict(),
        "policy": "A forecast must pass walk-forward validation, confidence, edge, intraday risk, and provider-quality checks before it can be used for trading.",
    }
    report.setdefault("diagnostics", {})["intraday_risk_context"] = risk_context
    report.setdefault("technical_view", {})["intraday_chart_confirmation"] = chart_confirmation
    if allowed_count == 0:
        report["suggested_action"] = "Hold"


def _annotate_mean_reversion_dip_buy(report: dict, prices: pd.DataFrame, target_column: str, risk_profile_name: str = "medium") -> None:
    """Add conditional dip-buy setups that are separate from momentum forecast gates."""

    annotate_mean_reversion_dip_buy(report, prices, target_column, risk_profile_name)


def _run_daily_llm_decision(report: dict, args: argparse.Namespace, *, dry_run: bool) -> dict[str, Any]:
    profile = trader_profiles[args.risk_profile]
    technical_packet = build_technical_packet(report)
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
        "ticker": args.ticker.upper(),
        "trader_name": args.trader_name,
        "trader_profile_json": json.dumps(profile, indent=2, sort_keys=True),
        "portfolio_context_json": json.dumps(portfolio_context, indent=2, sort_keys=True),
        "technical_packet_json": json.dumps(technical_packet, indent=2, sort_keys=True, default=str),
    }
    provider = resolve_llm_provider(getattr(args, "llm_provider", None))
    model = resolve_llm_model(args.llm_model, provider=provider)
    if dry_run:
        payload = response_payload(
            model=model,
            system_message=autonomous_trader.system_message,
            user_message=autonomous_trader.user_message,
            json_schema=autonomous_trader.json_schema,
            reasoning_effort=args.llm_reasoning_effort,
            item=item,
            use_web_search=not args.llm_no_web_search,
            search_context_size=args.llm_search_context_size,
        )
        return {
            "status": "dry_run",
            "decision": None,
            "reason": "LLM decision packet was built but no LLM call was made.",
            "model": model,
            "trader_profile": profile,
            "portfolio_context": portfolio_context,
            "technical_packet": technical_packet,
            "llm_prompt_payload": payload,
        }

    load_env(args.llm_env_file)
    try:
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
            usage_context={"purpose": "daily_trade_llm_decision", "ticker": args.ticker.upper(), "risk_profile": args.risk_profile, "provider": provider},
        )
    except Exception as exc:
        return {
            "status": "error",
            "decision": None,
            "reason": str(exc),
            "model": model,
            "trader_profile": profile,
            "portfolio_context": portfolio_context,
            "technical_packet": technical_packet,
        }
    return {
        "status": "executed",
        "model": model,
        "trader_profile": profile,
        "portfolio_context": portfolio_context,
        "technical_packet": technical_packet,
        "decision": decision,
        "llm_prompt_payload": payload,
        "llm_raw_response": raw_response,
        "policy": "LLM decision is advisory/governed context. Deterministic risk and broker gates still control execution.",
    }


def _daily_final_decision_reasoning(report: dict, args: argparse.Namespace) -> dict[str, Any]:
    llm = report.get("llm_trader") if isinstance(report.get("llm_trader"), dict) else {}
    decision = llm.get("decision") if isinstance(llm.get("decision"), dict) else {}
    production_gate = report.get("decision_view", {}).get("production_gate", {})
    dip_buy = report.get("decision_view", {}).get("mean_reversion_dip_buy", {})
    long_term_context = report.get("decision_view", {}).get("long_term_context", {})
    provider_summaries = (
        long_term_context.get("provider_summaries", {})
        if isinstance(long_term_context.get("provider_summaries"), dict)
        else {}
    )
    return {
        "final_action": decision.get("decision") or report.get("suggested_action"),
        "decision_source": "autonomous_llm_trader" if llm.get("status") == "executed" else "deterministic_daily_gate",
        "llm_status": llm.get("status") or ("disabled" if not getattr(args, "enable_llm_decision", False) else None),
        "llm_model": llm.get("model"),
        "llm_confidence": decision.get("confidence"),
        "llm_technical_read": decision.get("technical_read"),
        "llm_market_context_read": decision.get("market_context_read"),
        "llm_decision_reasoning": decision.get("decision_reasoning"),
        "llm_sentiment": decision.get("sentiment"),
        "final_advice": decision.get("final_advice") or {},
        "entry_plan": decision.get("entry_plan") or {},
        "portfolio_plan": decision.get("portfolio_plan") or {},
        "rule_blocks": decision.get("rule_blocks") or [],
        "risks": decision.get("risks") or [],
        "change_triggers": decision.get("change_triggers") or [],
        "production_gate": {
            "allowed_forecast_count": production_gate.get("allowed_forecast_count"),
            "provider_quality_warning": production_gate.get("provider_quality_warning"),
            "policy": production_gate.get("policy"),
        },
        "mean_reversion_dip_buy": {
            "status": dip_buy.get("status"),
            "best_setup": dip_buy.get("best_setup"),
            "policy": dip_buy.get("policy"),
        },
        "long_term_sources": {
            "status": long_term_context.get("status"),
            "providers_requested": long_term_context.get("providers_requested", []),
            "provider_status": {
                provider: summary.get("status")
                for provider, summary in provider_summaries.items()
                if isinstance(summary, dict)
            },
            "decision_relevance": long_term_context.get("decision_relevance", {}),
        },
        "audit_note": (
            "The autonomous LLM receives the technical packet, portfolio context, news/sentiment, "
            "long-term sources, and dip-buy context. Deterministic broker/risk gates still control execution."
        ),
    }


def _historical_reversal_stats(
    *,
    prices: pd.DataFrame,
    target_column: str,
    current_price: float,
    entry_price: float,
    stop_price: float,
    target_price: float,
    horizon_bars: int,
) -> dict[str, Any]:
    return historical_reversal_stats(
        prices=prices,
        target_column=target_column,
        current_price=current_price,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        horizon_bars=horizon_bars,
    )


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def _float_or_none(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _provider_quality_warning(provider: str, interval: str) -> str | None:
    interval_minutes = _interval_to_minutes(interval) or 1440.0
    if provider == "yahoo" and interval_minutes < 1440:
        return (
            "Yahoo intraday data is useful for validation runs but has limited history and no explicit production SLA. "
            "Use Polygon, Alpaca, IEX, or another configured low-latency provider for live trading."
        )
    return None


def _update_forecast_ledger(report: dict, prices: pd.DataFrame, output_dir: Path, target_column: str) -> dict[str, str]:
    ledger_path = output_dir / "forecast_ledger.csv"
    existing = pd.read_csv(ledger_path) if ledger_path.exists() else pd.DataFrame()
    now = datetime.now(UTC).isoformat()
    run_id = f"{report.get('ticker', 'TICKER')}_{report.get('as_of_timestamp', now)}"
    rows = []
    for forecast in report.get("forecasts", []):
        rows.append(
            {
                "run_id": run_id,
                "ticker": report.get("ticker"),
                "as_of_timestamp": report.get("as_of_timestamp"),
                "created_at_utc": now,
                "forecast_timestamp": forecast.get("forecast_date"),
                "horizon_bars": forecast.get("horizon_days"),
                "horizon_hours": forecast.get("horizon_hours"),
                "current_price": report.get("current_price"),
                "predicted_price": forecast.get("predicted_price"),
                "lower_price": forecast.get("lower_price"),
                "upper_price": forecast.get("upper_price"),
                "expected_direction": forecast.get("expected_direction"),
                "selected_model": forecast.get("selected_model"),
                "trade_allowed": forecast.get("trade_allowed"),
                "gate_status": forecast.get("production_gate", forecast.get("validation_gate", {})).get("status"),
                "actual_price": np.nan,
                "actual_timestamp": "",
                "absolute_error": np.nan,
                "absolute_pct_error": np.nan,
                "direction_correct": np.nan,
                "status": "pending",
            }
        )
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if not existing.empty else pd.DataFrame(rows)
    combined = _score_matured_ledger_rows(combined, prices, target_column)
    combined = combined.drop_duplicates(subset=["run_id", "horizon_bars"], keep="last")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(ledger_path, index=False)
    summary_path = output_dir / "forecast_ledger_summary.json"
    summary_path.write_text(json.dumps(_ledger_summary(combined), indent=2, default=str) + "\n", encoding="utf-8")
    return {"forecast_ledger": str(ledger_path), "forecast_ledger_summary": str(summary_path)}


def _score_matured_ledger_rows(ledger: pd.DataFrame, prices: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    output = ledger.copy()
    price_series = prices[target_column].astype(float).sort_index()
    price_index = pd.DatetimeIndex(price_series.index)
    output["forecast_timestamp"] = pd.to_datetime(output["forecast_timestamp"], errors="coerce")
    for index, row in output.iterrows():
        if str(row.get("status", "")) == "matured" and pd.notna(row.get("actual_price")):
            continue
        forecast_timestamp = row.get("forecast_timestamp")
        if pd.isna(forecast_timestamp):
            output.at[index, "status"] = "invalid_timestamp"
            continue
        future_positions = np.where(price_index >= pd.Timestamp(forecast_timestamp))[0]
        if len(future_positions) == 0:
            output.at[index, "status"] = "pending"
            continue
        actual_idx = future_positions[0]
        actual_timestamp = price_index[actual_idx]
        actual_price = float(price_series.iloc[actual_idx])
        predicted_price = float(row.get("predicted_price", np.nan))
        current_price = float(row.get("current_price", np.nan))
        output.at[index, "actual_price"] = actual_price
        output.at[index, "actual_timestamp"] = actual_timestamp.isoformat()
        output.at[index, "absolute_error"] = abs(predicted_price - actual_price)
        output.at[index, "absolute_pct_error"] = abs(predicted_price / actual_price - 1.0) if actual_price else np.nan
        predicted_direction = np.sign(predicted_price - current_price)
        actual_direction = np.sign(actual_price - current_price)
        output.at[index, "direction_correct"] = bool(predicted_direction == actual_direction)
        output.at[index, "status"] = "matured"
    return output


def _ledger_summary(ledger: pd.DataFrame) -> dict[str, object]:
    matured = ledger[ledger.get("status") == "matured"] if "status" in ledger else pd.DataFrame()
    if matured.empty:
        return {"rows": int(len(ledger)), "matured_rows": 0}
    return {
        "rows": int(len(ledger)),
        "matured_rows": int(len(matured)),
        "mean_absolute_error": float(pd.to_numeric(matured["absolute_error"], errors="coerce").mean()),
        "mean_absolute_pct_error": float(pd.to_numeric(matured["absolute_pct_error"], errors="coerce").mean()),
        "directional_accuracy": float(pd.to_numeric(matured["direction_correct"], errors="coerce").mean()),
        "by_horizon_hours": matured.groupby("horizon_hours")
        .agg(
            matured_rows=("status", "count"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_absolute_pct_error=("absolute_pct_error", "mean"),
            directional_accuracy=("direction_correct", "mean"),
        )
        .reset_index()
        .to_dict(orient="records"),
    }


def _write_hour_named_validation_aliases(report: dict, output_dir: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    plot_dir = output_dir / "plots"
    for forecast in report.get("forecasts", []):
        horizon = int(forecast.get("horizon_days", 0))
        hours = forecast.get("horizon_hours")
        if not horizon or hours is None:
            continue
        hour_label = _hour_label(float(hours))
        for suffix in ("png", "html"):
            source = plot_dir / f"validation_{report['ticker']}_{horizon}d.{suffix}"
            if not source.exists():
                continue
            target = plot_dir / f"validation_{report['ticker']}_{hour_label}.{suffix}"
            shutil.copyfile(source, target)
            artifacts[f"validation_{hour_label}_{suffix}"] = str(target)
    return artifacts


def _hour_label(hours: float) -> str:
    if float(hours).is_integer():
        return f"{int(hours)}h"
    return f"{str(hours).replace('.', '_')}h"


def _compact_validation_records(
    predictions: pd.DataFrame,
    *,
    close: pd.Series,
    horizon: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, row in predictions.iterrows():
        if index not in close.index:
            continue
        base_price = float(close.loc[index])
        forecast_timestamp = pd.Timestamp(index) + pd.Timedelta(minutes=0)
        records.append(
            {
                "validation_date": pd.Timestamp(index).isoformat(),
                "forecast_date": forecast_timestamp.isoformat(),
                "horizon_days": int(horizon),
                "base_price": base_price,
                "actual_log_return": float(row["actual"]),
                "predicted_log_return": float(row["predicted"]),
                "actual_future_price": float(base_price * np.exp(float(row["actual"]))),
                "predicted_future_price": float(base_price * np.exp(float(row["predicted"]))),
                "split_train_end": str(row.get("split_train_end", "")),
            }
        )
    return records


def _resolve_output_dir(output_dir: str | None, output: str | None) -> Path | None:
    if output_dir:
        return Path(output_dir)
    if output:
        return Path(output).parent
    return None


def _write_forecast_csv(report: dict, path: Path) -> None:
    forecasts = report.get("forecasts", [])
    if not forecasts:
        return
    rows = []
    for forecast in forecasts:
        rows.append(
            {
                "ticker": report.get("ticker"),
                "as_of": report.get("as_of_timestamp", report.get("as_of_date")),
                "horizon_bars": forecast.get("horizon_days"),
                "forecast_timestamp": forecast.get("forecast_date"),
                **forecast,
                "trade_action": report.get("daily_trade_view", {}).get("trade_plan", {}).get("action"),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _interval_to_minutes(interval: str) -> float | None:
    value = interval.strip().lower()
    units = {"m": 1.0, "h": 60.0, "d": 1440.0}
    for suffix, multiplier in sorted(units.items(), key=lambda item: len(item[0]), reverse=True):
        if value.endswith(suffix):
            try:
                return float(value[: -len(suffix)]) * multiplier
            except ValueError:
                return None
    return None


if __name__ == "__main__":
    main()
