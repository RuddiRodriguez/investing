from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from market_forecasting_engine.data import normalize_price_frame
from market_forecasting_engine.data_store import MarketDataStore
from market_forecasting_engine.long_term_sources import (
    DEFAULT_LONG_TERM_SOURCE_PROVIDERS,
    LongTermSourceRequest,
    append_long_term_source_snapshot,
    collect_long_term_source_context,
    load_long_term_source_snapshot_features,
    parse_long_term_source_providers,
)


def enrich_prices_with_long_term_sources(
    *,
    ticker: str,
    prices: pd.DataFrame,
    target_column: str,
    enabled: bool = True,
    providers: str | tuple[str, ...] | None = None,
    env_file: str | None = None,
    output_dir: str | Path | None = None,
    snapshot_dir: str | Path | None = None,
    data_store: MarketDataStore | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None, dict[str, Any] | None]:
    if not enabled:
        return prices, None, None

    provider_tuple = parse_long_term_source_providers(providers) if isinstance(providers, str) else (providers or DEFAULT_LONG_TERM_SOURCE_PROVIDERS)
    artifact_dir = Path(output_dir).expanduser() if output_dir else None
    durable_snapshot_dir = _long_term_snapshot_dir(snapshot_dir, data_store, artifact_dir)
    context = collect_long_term_source_context(
        LongTermSourceRequest(
            ticker=ticker,
            providers=tuple(provider_tuple),
            env_file=env_file,
            output_dir=artifact_dir,
            start_date=start_date,
            end_date=end_date,
        )
    )
    snapshot_path = append_long_term_source_snapshot(context, durable_snapshot_dir, ticker=ticker)
    features, metadata = load_long_term_source_snapshot_features(
        ticker,
        durable_snapshot_dir,
        pd.DatetimeIndex(prices.index),
    )
    metadata["snapshot_path"] = str(snapshot_path)
    context.setdefault("model_feature_policy", {})["snapshot_feature_metadata"] = metadata
    if not features.empty:
        prices = normalize_price_frame(prices.join(features, how="left"), target_column=target_column)
    return prices, context, metadata


def long_term_context_manifest_entry(context: dict[str, Any] | None, snapshot_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    if not context:
        return {}
    return {
        "status": context.get("status"),
        "providers_requested": context.get("providers_requested", []),
        "provider_summaries": context.get("provider_summaries", {}),
        "artifacts": context.get("artifacts", {}),
        "model_feature_policy": context.get("model_feature_policy", {}),
        "snapshot_feature_metadata": snapshot_metadata or context.get("model_feature_policy", {}).get("snapshot_feature_metadata", {}),
    }


def long_term_context_source_entry(context: dict[str, Any] | None, snapshot_metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if not context:
        return None
    metadata = snapshot_metadata or context.get("model_feature_policy", {}).get("snapshot_feature_metadata", {})
    return {
        "label": "long_term_sources",
        "provider": "call_all_consolidated",
        "providers": list(context.get("providers_requested", [])),
        "status": context.get("status"),
        "provider_health": context.get("consolidated", {}).get("provider_health", {}),
        "artifacts": context.get("artifacts", {}),
        "snapshot_path": metadata.get("snapshot_path"),
        "snapshot_feature_metadata": metadata,
        "model_training_included": bool(metadata.get("model_training_included")),
    }


def _long_term_snapshot_dir(
    snapshot_dir: str | Path | None,
    data_store: MarketDataStore | None,
    output_dir: Path | None,
) -> Path:
    if snapshot_dir:
        return Path(snapshot_dir)
    if data_store is not None:
        return data_store.root / "long_term_source_snapshots"
    if output_dir is not None:
        return output_dir / "data" / "long_term_source_snapshots"
    return Path("automated_forecasting_engine/runs/long_term_source_snapshots")
