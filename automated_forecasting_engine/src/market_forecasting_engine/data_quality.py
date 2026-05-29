from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from market_forecasting_engine.calendar import summarize_calendar_alignment


PRICE_COLUMNS = ("open", "high", "low", "close")


def build_data_quality_report(
    prices: pd.DataFrame,
    target_column: str = "close",
    calendar: str = "XNYS",
    max_missing_session_examples: int = 25,
) -> dict[str, Any]:
    """Build auditable quality diagnostics for the model-ready data frame."""

    frame = prices.copy()
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_convert(None)
    frame.index = index
    target = target_column.lower()
    warnings: list[dict[str, Any]] = []

    duplicate_count = int(frame.index.duplicated().sum())
    if duplicate_count:
        warnings.append(_warning("duplicate_dates", "high", f"{duplicate_count} duplicated index rows."))

    missing_by_column = {str(column): int(frame[column].isna().sum()) for column in frame.columns}
    missing_pct_by_column = {
        str(column): float(frame[column].isna().mean()) if len(frame) else 0.0 for column in frame.columns
    }
    heavily_missing = [column for column, pct in missing_pct_by_column.items() if pct > 0.20]
    if heavily_missing:
        warnings.append(
            _warning(
                "high_missingness",
                "medium",
                f"Columns over 20% missing: {', '.join(heavily_missing[:20])}.",
            )
        )

    non_positive_prices: dict[str, int] = {}
    for column in PRICE_COLUMNS:
        if column in frame.columns:
            count = int((pd.to_numeric(frame[column], errors="coerce") <= 0).sum())
            if count:
                non_positive_prices[column] = count
    if non_positive_prices:
        warnings.append(_warning("non_positive_prices", "high", "Non-positive OHLC prices detected."))

    non_positive_volume = 0
    if "volume" in frame.columns:
        non_positive_volume = int((pd.to_numeric(frame["volume"], errors="coerce") <= 0).sum())
        if non_positive_volume:
            warnings.append(_warning("non_positive_volume", "low", "Non-positive volume rows detected."))

    return_stats = _return_outlier_stats(frame[target]) if target in frame.columns else {}
    if return_stats.get("extreme_return_rows", 0) > 0:
        warnings.append(
            _warning(
                "extreme_returns",
                "medium",
                f"{return_stats['extreme_return_rows']} rows exceed return outlier thresholds.",
            )
        )

    stale_stats = _stale_price_stats(frame[target]) if target in frame.columns else {}
    if stale_stats.get("stale_runs_ge_5", 0) > 0:
        warnings.append(
            _warning(
                "stale_prices",
                "medium",
                f"{stale_stats['stale_runs_ge_5']} runs have at least 5 unchanged target prices.",
            )
        )

    calendar_summary = summarize_calendar_alignment(frame, calendar=calendar)
    missing_sessions = calendar_summary.get("missing_sessions", [])
    if missing_sessions:
        warnings.append(
            _warning(
                "missing_trading_sessions",
                "medium",
                f"{len(missing_sessions)} expected trading sessions are missing.",
            )
        )

    external_columns = [
        column
        for column in frame.columns
        if str(column).lower() not in {"open", "high", "low", "close", "volume", "dividends", "stock_splits", target}
    ]

    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target_column": target,
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "start_date": str(frame.index.min().date()) if len(frame) else None,
        "end_date": str(frame.index.max().date()) if len(frame) else None,
        "duplicate_dates": duplicate_count,
        "missing_values": missing_by_column,
        "missing_pct": missing_pct_by_column,
        "non_positive_prices": non_positive_prices,
        "non_positive_volume": non_positive_volume,
        "return_outliers": return_stats,
        "stale_prices": stale_stats,
        "external_columns": [str(column) for column in external_columns],
        "calendar_alignment": {
            **calendar_summary,
            "missing_sessions": list(calendar_summary.get("missing_sessions", []))[:max_missing_session_examples],
        },
        "warnings": warnings,
        "status": "fail" if any(item["severity"] == "high" for item in warnings) else "warn" if warnings else "pass",
    }


def _return_outlier_stats(close: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(close, errors="coerce")
    returns = np.log(numeric.replace(0, np.nan)).diff().replace([np.inf, -np.inf], np.nan)
    finite = returns.dropna()
    if finite.empty:
        return {"extreme_return_rows": 0, "largest_abs_log_return": 0.0, "threshold": None}
    robust_sigma = float((finite - finite.median()).abs().median() * 1.4826)
    threshold = max(0.20, 8.0 * robust_sigma) if robust_sigma > 0 else 0.20
    extreme = finite.abs() > threshold
    return {
        "extreme_return_rows": int(extreme.sum()),
        "largest_abs_log_return": float(finite.abs().max()),
        "threshold": float(threshold),
        "example_dates": [str(pd.Timestamp(value).date()) for value in finite.index[extreme][:10]],
    }


def _stale_price_stats(close: pd.Series) -> dict[str, int]:
    numeric = pd.to_numeric(close, errors="coerce")
    unchanged = numeric.diff().fillna(np.nan) == 0
    run_lengths: list[int] = []
    current = 0
    for value in unchanged:
        if bool(value):
            current += 1
        else:
            if current:
                run_lengths.append(current)
            current = 0
    if current:
        run_lengths.append(current)
    return {
        "stale_runs_ge_3": int(sum(length >= 3 for length in run_lengths)),
        "stale_runs_ge_5": int(sum(length >= 5 for length in run_lengths)),
        "max_stale_run": int(max(run_lengths) if run_lengths else 0),
    }


def _warning(code: str, severity: str, message: str) -> dict[str, str]:
    return {"code": code, "severity": severity, "message": message}
