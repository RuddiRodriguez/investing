from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from market_forecasting_engine.calendar import summarize_calendar_alignment
from market_forecasting_engine.daily_trade import DailyTradeConfig, build_daily_trade_plan, infer_bar_interval_minutes
from market_forecasting_engine.data import normalize_price_frame
from market_forecasting_engine.data_manifest import build_data_manifest
from market_forecasting_engine.data_providers import DataRequest, load_prices_with_provider
from market_forecasting_engine.data_quality import build_data_quality_report
from market_forecasting_engine.governance import write_audit_bundle
from market_forecasting_engine.models import HistoricalMeanReturn, RecentMeanReturn, default_candidates
from market_forecasting_engine.pipeline import ForecastingEngine
from market_forecasting_engine.plots import write_daily_trade_plot_artifacts, write_plot_artifacts
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
    parser.add_argument("--training-lookback-days", type=int, default=30, help="Minimum Yahoo intraday lookback used when the requested start has too few rows for modelling.")
    parser.add_argument("--selection-metric", default="mae", help="Model selection metric for the advanced hourly forecasting engine.")
    parser.add_argument("--no-lightgbm", action="store_true", help="Disable optional LightGBM candidate.")
    parser.add_argument("--no-statistical-models", action="store_true", help="Disable optional ARIMA/SARIMA/GARCH/VAR candidates.")
    parser.add_argument("--search-level", choices=("fast", "expanded"), default="fast", help="Candidate tuning breadth.")
    parser.add_argument("--transaction-cost-bps", type=float, default=2.0, help="Estimated transaction cost in bps.")
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
    if output_dir is not None and not args.no_plot:
        report.setdefault("artifacts", {}).update(write_plot_artifacts(report, prices, output_dir, target_column=args.target_column))
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
    required_rows = max_horizon_bars + 60
    if len(prices) >= required_rows:
        return result

    fallback_start = (pd.Timestamp.now(tz=UTC).tz_convert(None) - pd.Timedelta(days=args.training_lookback_days)).date().isoformat()
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
    expanded.metadata["initial_request_rows"] = int(len(prices))
    expanded.metadata["training_history_expanded"] = True
    expanded.metadata["training_history_start"] = fallback_start
    expanded.metadata["training_history_required_rows"] = int(required_rows)
    return expanded


def _run_advanced_hourly_forecast(
    prices: pd.DataFrame,
    trade_report: dict,
    args: argparse.Namespace,
    provider_metadata: dict,
    forecast_hours: tuple[float, ...],
) -> dict:
    interval_minutes = trade_report.get("interval_minutes") or infer_bar_interval_minutes(prices.index) or _interval_to_minutes(args.interval)
    if interval_minutes is None:
        interval_minutes = 5.0
    horizons = tuple(max(1, int(round(hours * 60.0 / float(interval_minutes)))) for hours in forecast_hours)
    n_rows = len(prices)
    min_training_rows = max(30, min(180, max(20, n_rows // 2)))
    validation_window = max(10, min(45, max(10, n_rows // 5)))
    step_size = max(5, validation_window // 2)
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
    )
    try:
        report = ForecastingEngine(config).run(
            prices,
            data_manifest=data_manifest,
            data_quality_report=data_quality_report,
            security_metadata=security_metadata,
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
            validation_results = validate_candidates(
                candidates=candidate_pool,
                features=training_frame[feature_columns],
                target=training_frame["target"],
                horizon_days=horizon,
                min_training_rows=max(20, min(60, len(training_frame) // 2)),
                validation_window=max(8, min(24, len(training_frame) // 4)),
                step_size=max(4, min(12, len(training_frame) // 8)),
                max_splits=5,
                purge_window=min(horizon, 12),
                embargo_window=0,
                final_holdout_fraction=0.10,
            )
            if not validation_results:
                validation_results = validate_candidates(
                    candidates=[HistoricalMeanReturn(), RecentMeanReturn(window=12, name="recent_mean_return_12_bars")],
                    features=training_frame[feature_columns],
                    target=training_frame["target"],
                    horizon_days=horizon,
                    min_training_rows=20,
                    validation_window=max(8, min(16, len(training_frame) // 4)),
                    step_size=4,
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

        forecast_timestamp = pd.Timestamp(prices.index[-1]) + pd.Timedelta(minutes=interval_minutes * horizon)
        interval_width = max(1.28 * residual_std, 0.001)
        predicted_price = float(close.iloc[-1] * np.exp(prediction))
        lower_price = float(close.iloc[-1] * np.exp(prediction - interval_width))
        upper_price = float(close.iloc[-1] * np.exp(prediction + interval_width))
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
    close = prices[target_column].astype(float)
    high = prices["high"].astype(float) if "high" in prices.columns else close
    low = prices["low"].astype(float) if "low" in prices.columns else close
    volume = prices["volume"].astype(float) if "volume" in prices.columns else pd.Series(1.0, index=prices.index)
    log_close = np.log(close.replace(0, np.nan))
    returns = log_close.diff()
    typical_price = (high + low + close) / 3.0
    vwap = (typical_price * volume).groupby(pd.DatetimeIndex(prices.index).date).cumsum() / volume.where(volume != 0).groupby(pd.DatetimeIndex(prices.index).date).cumsum()
    features = pd.DataFrame(index=prices.index)
    for window in (1, 3, 6, 12, 24, 48):
        features[f"log_return_{window}_bars"] = log_close.diff(window)
        features[f"momentum_{window}_bars"] = close.pct_change(window)
    for window in (6, 12, 24):
        features[f"realized_vol_{window}_bars"] = returns.rolling(window).std()
        features[f"volume_z_{window}_bars"] = (volume - volume.rolling(window).mean()) / volume.rolling(window).std()
    features["close_to_vwap"] = close / vwap - 1
    features["ema_9_to_21"] = close.ewm(span=9, adjust=False).mean() / close.ewm(span=21, adjust=False).mean() - 1
    features["range_pct"] = (high - low) / close
    features["minute_of_day"] = pd.DatetimeIndex(prices.index).hour * 60 + pd.DatetimeIndex(prices.index).minute
    features["day_of_week"] = pd.DatetimeIndex(prices.index).dayofweek
    return features.replace([np.inf, -np.inf], np.nan).ffill()


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
