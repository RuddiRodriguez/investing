from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from market_forecasting_engine.data_store import frame_sha256


def build_data_manifest(
    prices: pd.DataFrame,
    ticker: str,
    target_column: str = "close",
    provider: str = "in_memory",
    source: str | None = None,
    request: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    context_sources: list[dict[str, Any]] | None = None,
    indicator_sources: list[dict[str, Any]] | None = None,
    event_sources: list[dict[str, Any]] | None = None,
    alternative_sources: list[dict[str, Any]] | None = None,
    point_in_time_policy: dict[str, Any] | None = None,
    security_metadata: dict[str, Any] | None = None,
    calendar_summary: dict[str, Any] | None = None,
    universe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    index = pd.DatetimeIndex(prices.index)
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "ticker": ticker.upper(),
        "provider": provider,
        "source": source,
        "request": request or {},
        "target_column": target_column.lower(),
        "row_count": int(len(prices)),
        "column_count": int(len(prices.columns)),
        "columns": [str(column) for column in prices.columns],
        "start_date": str(index.min().date()) if len(index) else None,
        "end_date": str(index.max().date()) if len(index) else None,
        "normalized_data_hash": frame_sha256(prices),
        "artifacts": artifacts or {},
        "context_sources": context_sources or [],
        "indicator_sources": indicator_sources or [],
        "event_sources": event_sources or [],
        "alternative_sources": alternative_sources or [],
        "point_in_time_policy": point_in_time_policy or {},
        "security_metadata": security_metadata or {},
        "calendar": calendar_summary or {},
        "universe": universe or {},
    }


def point_in_time_policy_summary(
    macro_release_lag_days: int = 0,
    rates_release_lag_days: int = 0,
    events_release_lag_days: int = 0,
) -> dict[str, Any]:
    return {
        "principle": "Features are indexed by the date they are available to the model, not necessarily the economic observation date.",
        "macro_release_lag_days": int(macro_release_lag_days),
        "rates_release_lag_days": int(rates_release_lag_days),
        "events_release_lag_days": int(events_release_lag_days),
        "recognized_availability_columns": ["available_at", "availability_date", "release_date", "published_at"],
        "revision_handling": "If revision_as_of is present, later rows for the same available date replace earlier rows.",
    }
