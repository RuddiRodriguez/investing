from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_forecasting_engine.calendar import summarize_calendar_alignment
from market_forecasting_engine.data_manifest import build_data_manifest
from market_forecasting_engine.data_providers import DataRequest, ProviderResult, load_prices_with_provider
from market_forecasting_engine.data_quality import build_data_quality_report
from market_forecasting_engine.data_store import MarketDataStore, request_key
from market_forecasting_engine.pipeline import ForecastingEngine
from market_forecasting_engine.portfolio import (
    PortfolioHolding,
    extract_portfolio_holdings,
    portfolio_projection_frame,
    portfolio_totals,
    write_projection_artifacts,
)
from market_forecasting_engine.schema import ForecastConfig
from market_forecasting_engine.security_master import resolve_security_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forecasts for every position in a portfolio PDF.")
    parser.add_argument("--pdf", required=True, help="Portfolio/net-worth PDF to extract.")
    parser.add_argument("--start", default="2020-01-01", help="Yahoo Finance start date.")
    parser.add_argument("--end", default=None, help="Yahoo Finance end date.")
    parser.add_argument("--interval", default="1d", help="Provider bar interval.")
    parser.add_argument("--horizons", default="1,5,30", help="Comma-separated forecast horizons in trading days.")
    parser.add_argument("--projection-horizon", type=int, default=5, help="Forecast horizon to use for portfolio value projection.")
    parser.add_argument("--selection-metric", default="mae", help="Model-selection metric.")
    parser.add_argument("--confidence-level", type=float, default=0.80, help="Forecast confidence interval level.")
    parser.add_argument("--min-training-rows", type=int, default=180, help="Minimum training rows per ticker.")
    parser.add_argument("--validation-window", type=int, default=45, help="Walk-forward validation window.")
    parser.add_argument("--step-size", type=int, default=20, help="Walk-forward validation step size.")
    parser.add_argument("--max-splits", type=int, default=8, help="Maximum validation splits.")
    parser.add_argument("--final-holdout-fraction", type=float, default=0.15, help="Untouched final holdout fraction.")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0, help="Transaction cost for validation backtests.")
    parser.add_argument("--output-dir", required=True, help="Directory for portfolio projection outputs.")
    parser.add_argument("--data-dir", default=None, help="Optional shared data lake/cache directory.")
    parser.add_argument("--refresh-data-cache", action="store_true", help="Download data even if cache exists.")
    parser.add_argument("--no-data-cache", action="store_true", help="Disable cache reads.")
    parser.add_argument("--no-lightgbm", action="store_true", help="Disable optional LightGBM candidate.")
    parser.add_argument("--no-statistical-models", action="store_true", help="Disable optional ARIMA/SARIMA/GARCH/VAR candidates.")
    parser.add_argument("--include-lstm", action="store_true", help="Enable optional LSTM candidate.")
    parser.add_argument("--deep-learning-profile", choices=("off", "fast", "research"), default="off", help="Optional Chapter 17 deep-learning candidate profile.")
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
    args = parser.parse_args()

    horizons = tuple(int(value.strip()) for value in args.horizons.split(",") if value.strip())
    if args.projection_horizon not in horizons:
        raise ValueError("--projection-horizon must be one of --horizons.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ticker_report_dir = output_dir / "ticker_reports"
    ticker_report_dir.mkdir(parents=True, exist_ok=True)
    data_store = MarketDataStore(args.data_dir) if args.data_dir else MarketDataStore(output_dir / "data")

    holdings = extract_portfolio_holdings(args.pdf)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for holding in holdings:
        row = _forecast_holding(
            holding=holding,
            args=args,
            horizons=horizons,
            projection_horizon=args.projection_horizon,
            ticker_report_dir=ticker_report_dir,
            data_store=data_store,
        )
        rows.append(row)
        if row["status"] == "failed":
            failures.append(row)

    cash = _cash_row(args.pdf)
    if cash is not None:
        rows.append(cash)

    projection = portfolio_projection_frame(rows)
    totals = portfolio_totals(projection)
    artifacts = write_projection_artifacts(projection, totals, output_dir)
    extracted_path = output_dir / "extracted_holdings.json"
    extracted_path.write_text(
        json.dumps(
            {
                "source_pdf": str(Path(args.pdf)),
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "holdings": [holding.to_dict() for holding in holdings],
                "cash": cash,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    artifacts["extracted_holdings"] = str(extracted_path)

    summary = {
        "source_pdf": str(Path(args.pdf)),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "projection_horizon_days": args.projection_horizon,
        "totals": totals,
        "artifacts": artifacts,
        "failures": failures,
    }
    summary_path = output_dir / "portfolio_projection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(_summary_text(summary, projection))


def _forecast_holding(
    holding: PortfolioHolding,
    args: argparse.Namespace,
    horizons: tuple[int, ...],
    projection_horizon: int,
    ticker_report_dir: Path,
    data_store: MarketDataStore,
) -> dict[str, Any]:
    if not holding.symbol_candidates:
        return _failed_row(holding, None, "No Yahoo symbol mapping was available.")

    errors = []
    for symbol in holding.symbol_candidates:
        try:
            provider_result = _load_symbol_prices(symbol, args, data_store)
            report = _run_symbol_forecast(
                symbol=symbol,
                prices=provider_result.frame,
                provider_result=provider_result,
                args=args,
                horizons=horizons,
                data_store=data_store,
            )
            symbol_dir = ticker_report_dir / _safe_path(symbol)
            symbol_dir.mkdir(parents=True, exist_ok=True)
            report_path = symbol_dir / "forecast_report.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            return _projection_row(holding, symbol, report, projection_horizon, report_path)
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")
            continue
    return _failed_row(holding, holding.symbol_candidates[0], "; ".join(errors))


def _load_symbol_prices(symbol: str, args: argparse.Namespace, data_store: MarketDataStore) -> ProviderResult:
    return load_prices_with_provider(
        "yahoo",
        DataRequest(
            ticker=symbol,
            start=args.start,
            end=args.end,
            target_column="close",
            interval=args.interval,
            adjustment_policy="auto_adjust",
        ),
        store=data_store,
        use_cache=not args.no_data_cache,
        refresh_cache=args.refresh_data_cache,
    )


def _run_symbol_forecast(
    symbol: str,
    prices: pd.DataFrame,
    provider_result: ProviderResult,
    args: argparse.Namespace,
    horizons: tuple[int, ...],
    data_store: MarketDataStore,
) -> dict[str, Any]:
    config = ForecastConfig(
        ticker=symbol,
        horizons=horizons,
        target_column="close",
        min_training_rows=args.min_training_rows,
        validation_window=args.validation_window,
        step_size=args.step_size,
        max_splits=args.max_splits,
        selection_metric=args.selection_metric,
        confidence_level=args.confidence_level,
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
    )
    security_metadata = resolve_security_metadata(
        ticker=symbol,
        prices=prices,
        provider_metadata=provider_result.metadata,
        adjustment_policy="auto_adjust",
    )
    data_quality_report = build_data_quality_report(prices, target_column="close")
    data_manifest = build_data_manifest(
        prices=prices,
        ticker=symbol,
        target_column="close",
        provider="yahoo",
        request=DataRequest(ticker=symbol, start=args.start, end=args.end).to_dict(),
        artifacts=provider_result.metadata.get("artifacts", {}),
        security_metadata=security_metadata,
        calendar_summary=summarize_calendar_alignment(prices),
    )
    data_store.write_json(
        "manifests",
        "yahoo",
        symbol,
        request_key({"ticker": symbol, "start": args.start, "end": args.end, "kind": "portfolio_run_manifest"}),
        data_manifest,
    )
    return ForecastingEngine(config).run(
        prices,
        data_manifest=data_manifest,
        data_quality_report=data_quality_report,
        security_metadata=security_metadata,
    )


def _projection_row(
    holding: PortfolioHolding,
    symbol: str,
    report: dict[str, Any],
    projection_horizon: int,
    report_path: Path,
) -> dict[str, Any]:
    forecast = next(
        item for item in report["forecasts"] if int(item["horizon_days"]) == int(projection_horizon)
    )
    expected_return = float(forecast["expected_return"])
    current_value = float(holding.statement_value_eur)
    projected_value = current_value * (1.0 + expected_return)
    decision = report.get("technical_view", {}).get("decision_diagnostics", {})
    return {
        "asset_type": holding.asset_type,
        "security_name": holding.security_name,
        "isin": holding.isin,
        "symbol": symbol,
        "quantity": holding.quantity,
        "current_value_eur": current_value,
        "projection_horizon_days": int(projection_horizon),
        "expected_return": expected_return,
        "projected_value_eur": projected_value,
        "projected_change_eur": projected_value - current_value,
        "expected_direction": forecast["expected_direction"],
        "directional_confidence": float(forecast["directional_confidence"]),
        "suggested_action": report["suggested_action"],
        "risk_level": report["risk_level"],
        "hold_reason": decision.get("hold_reason"),
        "selected_model": forecast["selected_model"],
        "forecast_price": float(forecast["predicted_price"]),
        "forecast_lower_price": float(forecast["lower_price"]),
        "forecast_upper_price": float(forecast["upper_price"]),
        "status": "forecasted",
        "error": None,
        "forecast_report": str(report_path),
    }


def _failed_row(holding: PortfolioHolding, symbol: str | None, error: str) -> dict[str, Any]:
    return {
        "asset_type": holding.asset_type,
        "security_name": holding.security_name,
        "isin": holding.isin,
        "symbol": symbol,
        "quantity": holding.quantity,
        "current_value_eur": holding.statement_value_eur,
        "projection_horizon_days": None,
        "expected_return": 0.0,
        "projected_value_eur": holding.statement_value_eur,
        "projected_change_eur": 0.0,
        "expected_direction": "Unavailable",
        "directional_confidence": 0.0,
        "suggested_action": "Hold",
        "risk_level": "Unavailable",
        "hold_reason": "NoForecast",
        "selected_model": None,
        "forecast_price": None,
        "forecast_lower_price": None,
        "forecast_upper_price": None,
        "status": "failed",
        "error": error,
    }


def _cash_row(pdf_path: str) -> dict[str, Any] | None:
    text = Path(pdf_path)
    try:
        import pdfplumber
    except ImportError:
        return None
    extracted = []
    with pdfplumber.open(str(text)) as pdf:
        for page in pdf.pages:
            extracted.append(page.extract_text(x_tolerance=1, y_tolerance=3) or "")
    match = None
    for line in "\n".join(extracted).splitlines():
        clean = line.strip()
        if clean.startswith("Current account "):
            match = clean
    if match is None:
        return None
    value = float(match.replace("Current account", "").replace("EUR", "").strip().replace(",", ""))
    return {
        "asset_type": "cash",
        "security_name": "Current account",
        "isin": None,
        "symbol": "CASH",
        "quantity": 1.0,
        "current_value_eur": value,
        "projection_horizon_days": 0,
        "expected_return": 0.0,
        "projected_value_eur": value,
        "projected_change_eur": 0.0,
        "expected_direction": "Flat",
        "directional_confidence": 1.0,
        "suggested_action": "Hold",
        "risk_level": "Low",
        "hold_reason": None,
        "selected_model": "cash_identity",
        "forecast_price": value,
        "forecast_lower_price": value,
        "forecast_upper_price": value,
        "status": "cash",
        "error": None,
    }


def _summary_text(summary: dict[str, Any], projection: pd.DataFrame) -> str:
    totals = summary["totals"]
    lines = [
        "Portfolio Projection",
        f"Projection horizon: {summary['projection_horizon_days']} trading days",
        f"Current value: EUR {totals['current_value_eur']:.2f}",
        f"Projected value: EUR {totals['projected_value_eur']:.2f}",
        f"Projected change: EUR {totals['projected_change_eur']:.2f} ({totals['projected_return']:.2%})",
        "",
        "Top projected movers:",
    ]
    movers = projection[projection["asset_type"] != "cash"].copy()
    movers["abs_change"] = pd.to_numeric(movers["projected_change_eur"], errors="coerce").abs()
    for _, row in movers.sort_values("abs_change", ascending=False).head(8).iterrows():
        lines.append(
            f"- {row['symbol']}: EUR {float(row['projected_change_eur']):+.2f} "
            f"({float(row['expected_return']):+.2%}, {row['expected_direction']})"
        )
    lines.extend(["", "Artifacts:"])
    for label, path in summary["artifacts"].items():
        lines.append(f"- {label}: {path}")
    if summary["failures"]:
        lines.extend(["", "Failures:"])
        for failure in summary["failures"]:
            lines.append(f"- {failure['security_name']}: {failure['error']}")
    return "\n".join(lines)


def _safe_path(symbol: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in symbol)


if __name__ == "__main__":
    main()
