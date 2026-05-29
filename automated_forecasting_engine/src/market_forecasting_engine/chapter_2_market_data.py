from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


MARKET_COLUMNS = {"open", "high", "low", "close", "volume", "dividends", "stock_splits"}


def analyze_chapter_2_market_data(
    *,
    prices: pd.DataFrame,
    data_manifest: dict[str, Any],
    data_quality_report: dict[str, Any],
    target_column: str = "close",
) -> dict[str, Any]:
    """Assess market data source, corporate-action, and tradability readiness."""

    target = target_column.lower()
    source = {
        "provider": data_manifest.get("provider", "unknown"),
        "source": data_manifest.get("source"),
        "start_date": data_manifest.get("start_date"),
        "end_date": data_manifest.get("end_date"),
        "row_count": int(data_manifest.get("row_count") or len(prices)),
        "column_count": int(data_manifest.get("column_count") or len(prices.columns)),
        "fields_available": [str(column) for column in prices.columns],
    }
    field_coverage = _field_coverage(prices, target)
    corporate_actions = _corporate_action_diagnostics(prices, target)
    liquidity = _liquidity_diagnostics(prices, target)
    tradability = _tradability_gate(field_coverage, corporate_actions, liquidity, data_quality_report)

    return {
        "chapter": 2,
        "name": "Market and Fundamental Data - Sources and Techniques",
        "status": tradability["status"],
        "decision_policy": {
            "influences_final_action": False,
            "mode": "diagnostic_only",
            "reason": "Market-data diagnostics disclose data and tradability limits without overriding the forecast action.",
        },
        "source": source,
        "field_coverage": field_coverage,
        "corporate_actions": corporate_actions,
        "liquidity_tradability": liquidity,
        "tradability_gate": tradability,
        "technical_method_card": chapter_2_market_data_method_card(),
    }


def chapter_2_market_data_method_card() -> dict[str, Any]:
    return {
        "name": "jansen_ml4t_chapter_2_market_data_diagnostics",
        "version": "chapter_2_market_data_v1",
        "source": "Machine Learning for Algorithmic Trading, 2nd ed., Chapter 2",
        "purpose": "Audit market-data source coverage, corporate-action evidence, liquidity, and model-ready tradability.",
        "decision_policy": "diagnostic_only",
    }


def _field_coverage(prices: pd.DataFrame, target: str) -> dict[str, Any]:
    available = {str(column).lower() for column in prices.columns}
    required = {"open", "high", "low", target, "volume"}
    missing = sorted(required - available)
    external = sorted(str(column) for column in prices.columns if str(column).lower() not in MARKET_COLUMNS)
    return {
        "has_ohlc": all(column in available for column in ("open", "high", "low", target)),
        "has_volume": "volume" in available,
        "missing_core_fields": missing,
        "external_context_fields": external,
        "market_data_completeness": "complete_ohlcv" if not missing else "partial_market_data",
    }


def _corporate_action_diagnostics(prices: pd.DataFrame, target: str) -> dict[str, Any]:
    close = pd.to_numeric(prices[target], errors="coerce") if target in prices.columns else pd.Series(dtype=float)
    log_returns = np.log(close.replace(0, np.nan)).diff().replace([np.inf, -np.inf], np.nan)
    suspicious_jumps = log_returns.abs() > 0.30
    split_column = _first_existing(prices, ("stock_splits", "splits", "split"))
    dividend_column = _first_existing(prices, ("dividends", "dividend"))
    split_events = _positive_event_count(prices, split_column)
    dividend_events = _positive_event_count(prices, dividend_column)
    warnings = []
    if int(suspicious_jumps.sum()) > 0 and split_events == 0:
        warnings.append("Large price jumps exist but no split column/events are present.")
    if split_column is None:
        warnings.append("No split column is available after normalization.")
    if dividend_column is None:
        warnings.append("No dividend column is available after normalization.")
    return {
        "split_column": split_column,
        "dividend_column": dividend_column,
        "split_event_count": split_events,
        "dividend_event_count": dividend_events,
        "suspicious_large_return_count": int(suspicious_jumps.sum()),
        "suspicious_large_return_dates": [str(pd.Timestamp(date).date()) for date in log_returns.index[suspicious_jumps][:10]],
        "adjustment_policy": "normalized_close_used_as_model_target",
        "warnings": warnings,
    }


def _liquidity_diagnostics(prices: pd.DataFrame, target: str) -> dict[str, Any]:
    close = pd.to_numeric(prices[target], errors="coerce") if target in prices.columns else pd.Series(dtype=float)
    volume = pd.to_numeric(prices["volume"], errors="coerce") if "volume" in prices.columns else pd.Series(index=prices.index, dtype=float)
    dollar_volume = close * volume
    high = pd.to_numeric(prices["high"], errors="coerce") if "high" in prices.columns else close
    low = pd.to_numeric(prices["low"], errors="coerce") if "low" in prices.columns else close
    spread_proxy = ((high - low) / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    log_return = np.log(close.replace(0, np.nan)).diff().abs()
    amihud = (log_return / dollar_volume.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    zero_volume_ratio = float((volume.fillna(0) <= 0).mean()) if len(volume) else 1.0
    median_dollar_volume = _finite(dollar_volume.tail(63).median())
    average_dollar_volume = _finite(dollar_volume.tail(63).mean())
    warnings = []
    if "volume" not in prices.columns:
        warnings.append("Volume is missing, so tradability cannot be verified.")
    elif zero_volume_ratio > 0.05:
        warnings.append("More than 5% of rows have zero or missing volume.")
    if median_dollar_volume and median_dollar_volume < 1_000_000:
        warnings.append("Recent median dollar volume is below 1M, which may limit practical tradability.")
    return {
        "zero_volume_ratio": zero_volume_ratio,
        "average_dollar_volume_63d": average_dollar_volume,
        "median_dollar_volume_63d": median_dollar_volume,
        "median_spread_proxy_63d": _finite(spread_proxy.tail(63).median()),
        "amihud_illiquidity_63d": _finite(amihud.tail(63).mean()),
        "forecastable_but_not_tradable": bool(warnings),
        "warnings": warnings,
    }


def _tradability_gate(
    field_coverage: dict[str, Any],
    corporate_actions: dict[str, Any],
    liquidity: dict[str, Any],
    data_quality_report: dict[str, Any],
) -> dict[str, Any]:
    blocking = []
    warnings = []
    if not field_coverage["has_ohlc"]:
        warnings.append("Incomplete OHLC data reduces range, volatility, and execution diagnostics.")
    if not field_coverage["has_volume"]:
        blocking.append("Volume is missing.")
    if data_quality_report.get("status") == "fail":
        blocking.append("Data quality report has high-severity failures.")
    warnings.extend(corporate_actions.get("warnings", []))
    warnings.extend(liquidity.get("warnings", []))
    if blocking:
        status = "fail"
    elif warnings:
        status = "warn"
    else:
        status = "pass"
    return {
        "status": status,
        "blocking_reasons": blocking,
        "warnings": warnings,
        "model_ready": not blocking,
        "tradability_ready": not blocking and not liquidity.get("forecastable_but_not_tradable", False),
    }


def _first_existing(prices: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    available = {str(column).lower(): str(column) for column in prices.columns}
    for column in columns:
        if column in available:
            return available[column]
    return None


def _positive_event_count(prices: pd.DataFrame, column: str | None) -> int:
    if column is None:
        return 0
    values = pd.to_numeric(prices[column], errors="coerce").fillna(0.0)
    return int((values != 0).sum())


def _finite(value: Any) -> float:
    numeric = float(value or 0.0)
    return numeric if np.isfinite(numeric) else 0.0
