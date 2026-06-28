from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_forecasting_engine.eodhd_fundamentals import EodhdFundamentalsClient
from market_forecasting_engine.fmp_fundamentals import FinancialModelingPrepClient
from market_forecasting_engine.massive_fundamentals import MassiveFundamentalsClient
from market_forecasting_engine.nasdaq_data_link_fundamentals import NasdaqDataLinkFundamentalsClient
from market_forecasting_engine.stockanalysis_fundamentals import StockAnalysisFundamentalsClient
from market_forecasting_engine.tiingo_fundamentals import TiingoFundamentalsClient


DEFAULT_LONG_TERM_SOURCE_PROVIDERS = ("fmp", "tiingo", "massive", "nasdaq_data_link", "eodhd", "stockanalysis")
SNAPSHOT_VERSION = "long_term_source_snapshot_v1"
NUMERIC_FIELD_PATHS: dict[str, tuple[str, str]] = {
    "price": ("quote", "price"),
    "close": ("quote", "close"),
    "market_cap": ("valuation", "market_cap"),
    "enterprise_value": ("valuation", "enterprise_value"),
    "pe": ("valuation", "pe"),
    "forward_pe": ("valuation", "forward_pe"),
    "pb": ("valuation", "pb"),
    "ps": ("valuation", "ps"),
    "ev_to_ebitda": ("valuation", "ev_to_ebitda"),
    "revenue": ("income_statement", "revenue"),
    "net_income": ("income_statement", "net_income"),
    "free_cash_flow": ("cash_flow", "free_cash_flow"),
    "total_debt": ("balance_sheet", "total_debt"),
    "total_assets": ("balance_sheet", "total_assets"),
    "total_equity": ("balance_sheet", "total_equity"),
    "gross_margin": ("profitability", "gross_margin"),
    "operating_margin": ("profitability", "operating_margin"),
    "net_margin": ("profitability", "net_margin"),
    "profit_margin": ("profitability", "profit_margin"),
    "return_on_assets": ("profitability", "return_on_assets"),
    "return_on_equity": ("profitability", "return_on_equity"),
    "roic": ("profitability", "roic"),
    "target_consensus": ("analyst", "target_consensus"),
    "target_high": ("analyst", "target_high"),
    "target_low": ("analyst", "target_low"),
    "short_interest_days_to_cover": ("short_interest", "days_to_cover"),
}
IDENTITY_KEYS = ("name", "sector", "industry", "country", "currency", "exchange", "primary_exchange", "website", "homepage_url")


@dataclass(frozen=True)
class LongTermSourceRequest:
    ticker: str
    providers: tuple[str, ...] = DEFAULT_LONG_TERM_SOURCE_PROVIDERS
    env_file: str | None = None
    output_dir: str | Path | None = None
    start_date: str | None = None
    end_date: str | None = None
    max_news_items: int = 12


def collect_long_term_source_context(request: LongTermSourceRequest) -> dict[str, Any]:
    """Call all configured long-term data sources and consolidate current context.

    The resulting context is designed for report, audit, and LLM decision use. Current
    point-in-time snapshots are intentionally not backfilled across the historical
    supervised feature matrix.
    """

    fetched_at = datetime.now(UTC).isoformat()
    providers = _normalize_provider_list(request.providers)
    output_dir = Path(request.output_dir).expanduser() if request.output_dir else None
    raw_payloads: dict[str, dict[str, Any]] = {}
    provider_summaries: dict[str, dict[str, Any]] = {}

    for provider in providers:
        try:
            payload = _fetch_provider_payload(provider, request)
            raw_payloads[provider] = payload
            provider_summaries[provider] = _provider_summary(provider, payload)
        except Exception as exc:  # provider isolation is required for call-all collection
            provider_summaries[provider] = {
                "provider": provider,
                "status": "error",
                "error": _safe_error(exc),
                "ok_endpoints": 0,
                "total_endpoints": 0,
                "normalized_sections": [],
            }

    artifacts = _write_artifacts(output_dir, request.ticker, raw_payloads, provider_summaries) if output_dir else {}
    source_payloads = {
        provider: _compact_provider_payload(payload)
        for provider, payload in raw_payloads.items()
    }
    consolidated = _consolidate_sources(request.ticker, source_payloads, provider_summaries, fetched_at, request.max_news_items)
    if output_dir:
        consolidated_path = output_dir / _safe_symbol(request.ticker) / "consolidated.json"
        consolidated_path.parent.mkdir(parents=True, exist_ok=True)
        consolidated_path.write_text(json.dumps(consolidated, indent=2, sort_keys=True, default=str), encoding="utf-8")
        artifacts["consolidated"] = str(consolidated_path)

    return {
        "ticker": request.ticker.upper(),
        "source_type": "call_all_consolidated_long_term_context",
        "status": consolidated["status"],
        "fetched_at": fetched_at,
        "providers_requested": list(providers),
        "provider_summaries": provider_summaries,
        "source_payloads": source_payloads,
        "consolidated": consolidated,
        "artifacts": artifacts,
        "model_feature_policy": {
            "status": "current_snapshot_context_only",
            "model_training_included": False,
            "reason": (
                "Provider fundamentals, analyst data, filings, and news are current-run snapshots. "
                "They are passed into the governed decision context and audit report, but are not "
                "backfilled into historical supervised training until dated point-in-time snapshots exist."
            ),
            "future_promotion_path": [
                "store repeated dated source snapshots",
                "build as-of joined point-in-time historical features",
                "validate enriched models against market-only baselines",
                "promote only when validation improvement is stable",
            ],
        },
        "technical_method_card": long_term_sources_method_card(),
    }


def long_term_sources_method_card() -> dict[str, Any]:
    return {
        "name": "long_term_source_consolidation",
        "version": "call_all_consensus_v1",
        "decision_policy": "context_contributes_to_governed_llm_review_without_overriding_rule_gates",
        "providers": list(DEFAULT_LONG_TERM_SOURCE_PROVIDERS),
        "implemented_controls": [
            "call_all_provider_collection",
            "per_provider_error_isolation",
            "raw_payload_artifacts",
            "endpoint_status_audit",
            "numeric_consensus_median",
            "staleness_exclusion",
            "cross_provider_conflict_detection",
            "news_deduplication",
            "lookahead_leakage_block_for_training_features",
        ],
    }


def parse_long_term_source_providers(value: str | None) -> tuple[str, ...]:
    if not value:
        return DEFAULT_LONG_TERM_SOURCE_PROVIDERS
    providers = tuple(item.strip().lower().replace("-", "_") for item in value.split(",") if item.strip())
    unknown = [provider for provider in providers if provider not in DEFAULT_LONG_TERM_SOURCE_PROVIDERS]
    if unknown:
        raise ValueError(f"Unknown long-term source provider(s): {', '.join(unknown)}")
    return providers or DEFAULT_LONG_TERM_SOURCE_PROVIDERS


def compact_long_term_context_for_llm(context: dict[str, Any] | None) -> dict[str, Any]:
    if not context:
        return {}
    consolidated = context.get("consolidated", {}) if isinstance(context.get("consolidated"), dict) else {}
    return {
        "status": context.get("status"),
        "fetched_at": context.get("fetched_at"),
        "providers_requested": context.get("providers_requested", []),
        "provider_health": consolidated.get("provider_health", {}),
        "identity": consolidated.get("identity", {}),
        "numeric_consensus": consolidated.get("numeric_consensus", {}),
        "conflicts": consolidated.get("conflicts", [])[:8],
        "stale_fields": consolidated.get("stale_fields", [])[:8],
        "recent_news": consolidated.get("recent_news", [])[:8],
        "filing_context": consolidated.get("filing_context", {}),
        "provider_contexts": consolidated.get("provider_contexts", {}),
        "llm_evidence_manifest": consolidated.get("llm_evidence_manifest", {}),
        "llm_source_synthesis": context.get("llm_source_synthesis") or consolidated.get("llm_source_synthesis", {}),
        "model_feature_policy": context.get("model_feature_policy", {}),
        "decision_relevance": consolidated.get("decision_relevance", {}),
    }


def compact_long_term_context_for_source_synthesis(context: dict[str, Any] | None) -> dict[str, Any]:
    if not context:
        return {}
    consolidated = context.get("consolidated", {}) if isinstance(context.get("consolidated"), dict) else {}
    source_payloads = context.get("source_payloads", {}) if isinstance(context.get("source_payloads"), dict) else {}
    normalized_payloads: dict[str, Any] = {}
    for provider, payload in source_payloads.items():
        if not isinstance(payload, dict):
            continue
        normalized = payload.get("normalized", {}) if isinstance(payload.get("normalized"), dict) else {}
        normalized_payloads[str(provider)] = {
            "source": payload.get("source"),
            "source_base_url": payload.get("source_base_url"),
            "fetched_at": payload.get("fetched_at"),
            "endpoint_status": payload.get("endpoint_status", {}),
            "normalized": normalized,
            "normalized_sections": sorted([key for key, value in normalized.items() if value not in (None, {}, [])]),
            "raw_sha256": payload.get("raw_sha256"),
        }
    return {
        "status": context.get("status"),
        "fetched_at": context.get("fetched_at"),
        "providers_requested": context.get("providers_requested", []),
        "provider_summaries": context.get("provider_summaries", {}),
        "provider_health": consolidated.get("provider_health", {}),
        "identity": consolidated.get("identity", {}),
        "numeric_consensus": consolidated.get("numeric_consensus", {}),
        "conflicts": consolidated.get("conflicts", []),
        "stale_fields": consolidated.get("stale_fields", []),
        "recent_news": consolidated.get("recent_news", []),
        "filing_context": consolidated.get("filing_context", {}),
        "source_payloads_normalized": normalized_payloads,
        "llm_evidence_manifest": consolidated.get("llm_evidence_manifest", {}),
        "model_feature_policy": context.get("model_feature_policy", {}),
        "decision_relevance": consolidated.get("decision_relevance", {}),
        "policy": (
            "This synthesis context passes every non-empty normalized provider section. "
            "Raw HTML is excluded from prompts; raw payload artifacts remain available by path and checksum."
        ),
    }


def append_long_term_source_snapshot(
    context: dict[str, Any],
    snapshot_dir: str | Path,
    *,
    ticker: str | None = None,
) -> Path:
    """Append one compact point-in-time source snapshot for future as-of features."""

    directory = Path(snapshot_dir).expanduser() / _safe_symbol(ticker or str(context.get("ticker") or "ticker"))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "snapshots.jsonl"
    row = build_long_term_source_snapshot_row(context)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return path


def build_long_term_source_snapshot_row(context: dict[str, Any]) -> dict[str, Any]:
    consolidated = context.get("consolidated", {}) if isinstance(context.get("consolidated"), dict) else {}
    numeric = consolidated.get("numeric_consensus", {}) if isinstance(consolidated.get("numeric_consensus"), dict) else {}
    health = consolidated.get("provider_health", {}) if isinstance(consolidated.get("provider_health"), dict) else {}
    features: dict[str, float] = {}
    for field, payload in numeric.items():
        if not isinstance(payload, dict) or payload.get("status") not in {"ok", "stale_only"}:
            continue
        value = _to_float(payload.get("value"))
        if value is not None:
            features[f"lts_{_safe_symbol(field)}"] = value
        dispersion = _to_float(payload.get("relative_dispersion"))
        if dispersion is not None:
            features[f"lts_{_safe_symbol(field)}_dispersion"] = dispersion
        features[f"lts_{_safe_symbol(field)}_provider_count"] = float(payload.get("provider_count") or 0)
        features[f"lts_{_safe_symbol(field)}_stale_only"] = 1.0 if payload.get("status") == "stale_only" else 0.0
    endpoint_ratio = _to_float(health.get("endpoint_success_ratio"))
    if endpoint_ratio is not None:
        features["lts_endpoint_success_ratio"] = endpoint_ratio
    features["lts_provider_ok_count"] = float(len(health.get("providers_ok") or []))
    features["lts_provider_error_count"] = float(len(health.get("providers_error") or []))
    features["lts_conflict_count"] = float(len(consolidated.get("conflicts") or []))
    features["lts_stale_field_count"] = float(len(consolidated.get("stale_fields") or []))
    features["lts_recent_news_count"] = float(len(consolidated.get("recent_news") or []))
    return {
        "version": SNAPSHOT_VERSION,
        "ticker": str(context.get("ticker") or "").upper(),
        "fetched_at": context.get("fetched_at") or datetime.now(UTC).isoformat(),
        "status": context.get("status"),
        "providers_requested": context.get("providers_requested", []),
        "provider_health": health,
        "features": features,
        "model_feature_policy": {
            "model_training_included": True,
            "point_in_time_join": "merge_asof_backward",
            "source_context_status": context.get("model_feature_policy", {}).get("status")
            if isinstance(context.get("model_feature_policy"), dict)
            else None,
        },
    }


def load_long_term_source_snapshot_features(
    ticker: str,
    snapshot_dir: str | Path,
    target_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = Path(snapshot_dir).expanduser() / _safe_symbol(ticker)
    path = directory / "snapshots.jsonl"
    if not path.exists():
        return pd.DataFrame(index=target_index), {
            "status": "missing",
            "path": str(path),
            "rows": 0,
            "model_training_included": False,
            "reason": "No long-term source snapshots exist yet for this ticker.",
        }

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        fetched_at = pd.to_datetime(payload.get("fetched_at"), utc=True, errors="coerce")
        if pd.isna(fetched_at):
            continue
        features = payload.get("features") if isinstance(payload.get("features"), dict) else {}
        row = {"snapshot_fetched_at": fetched_at}
        row.update({key: _to_float(value) for key, value in features.items() if _to_float(value) is not None})
        rows.append(row)
    if not rows:
        return pd.DataFrame(index=target_index), {
            "status": "empty",
            "path": str(path),
            "rows": 0,
            "model_training_included": False,
            "reason": "Snapshot file exists but contains no usable dated feature rows.",
        }

    snapshots = pd.DataFrame(rows).drop_duplicates(subset=["snapshot_fetched_at"], keep="last").sort_values("snapshot_fetched_at")
    target = pd.DataFrame({"market_timestamp": pd.DatetimeIndex(target_index)})
    target["market_timestamp"] = pd.to_datetime(target["market_timestamp"], utc=True, errors="coerce")
    target = target.sort_values("market_timestamp")
    joined = pd.merge_asof(
        target,
        snapshots,
        left_on="market_timestamp",
        right_on="snapshot_fetched_at",
        direction="backward",
        allow_exact_matches=True,
    )
    joined.index = target_index
    feature_columns = [column for column in joined.columns if column.startswith("lts_")]
    features = joined[feature_columns].astype(float) if feature_columns else pd.DataFrame(index=target_index)
    usable_rows = int(features.notna().any(axis=1).sum()) if not features.empty else 0
    return features, {
        "status": "ok" if usable_rows else "no_asof_rows",
        "path": str(path),
        "rows": int(len(snapshots)),
        "feature_columns": feature_columns,
        "usable_price_rows": usable_rows,
        "first_snapshot_at": snapshots["snapshot_fetched_at"].iloc[0].isoformat(),
        "last_snapshot_at": snapshots["snapshot_fetched_at"].iloc[-1].isoformat(),
        "model_training_included": bool(usable_rows),
        "join_policy": "merge_asof_backward_no_future_snapshots",
    }


def _fetch_provider_payload(provider: str, request: LongTermSourceRequest) -> dict[str, Any]:
    env_file = request.env_file
    ticker = request.ticker
    start = request.start_date or _default_start_date()
    if provider == "fmp":
        return FinancialModelingPrepClient(env_file=env_file).fetch_company_fundamentals(ticker)
    if provider == "tiingo":
        return TiingoFundamentalsClient(env_file=env_file).fetch_company_fundamentals(ticker, start_date=start, end_date=request.end_date)
    if provider == "massive":
        return MassiveFundamentalsClient(env_file=env_file).fetch_company_context(ticker, limit=request.max_news_items)
    if provider == "nasdaq_data_link":
        return NasdaqDataLinkFundamentalsClient(env_file=env_file).fetch_company_context(ticker)
    if provider == "eodhd":
        return EodhdFundamentalsClient(env_file=env_file).fetch_company_context(ticker, from_date=start, to_date=request.end_date)
    if provider == "stockanalysis":
        return StockAnalysisFundamentalsClient().fetch_company_context(ticker)
    raise ValueError(f"Unsupported provider: {provider}")


def _consolidate_sources(
    ticker: str,
    source_payloads: dict[str, dict[str, Any]],
    provider_summaries: dict[str, dict[str, Any]],
    fetched_at: str,
    max_news_items: int,
) -> dict[str, Any]:
    observations: dict[str, list[dict[str, Any]]] = {field: [] for field in NUMERIC_FIELD_PATHS}
    identity_votes: dict[str, list[dict[str, str]]] = {key: [] for key in IDENTITY_KEYS}
    news_items: list[dict[str, Any]] = []
    filing_context: dict[str, Any] = {}
    provider_contexts: dict[str, Any] = {}
    stale_fields: list[dict[str, Any]] = []

    for provider, payload in source_payloads.items():
        normalized = payload.get("normalized", {}) if isinstance(payload.get("normalized"), dict) else {}
        for key in IDENTITY_KEYS:
            value = _lookup_identity(normalized, key)
            if value:
                identity_votes[key].append({"provider": provider, "value": str(value)})
        for field, path in NUMERIC_FIELD_PATHS.items():
            value = _nested_get(normalized, path)
            numeric = _to_float(value)
            if numeric is None:
                continue
            as_of = _as_of_for_path(normalized, path)
            stale = _is_stale(as_of, field, fetched_at)
            observation = {"provider": provider, "value": numeric, "as_of": as_of, "stale": stale}
            observations[field].append(observation)
            if stale:
                stale_fields.append({"field": field, **observation})
        news_items.extend(_recent_news(provider, normalized))
        provider_filing = _filing_context(provider, normalized)
        if provider_filing:
            filing_context[provider] = provider_filing
        provider_context = _provider_decision_context(provider, normalized)
        if provider_context:
            provider_contexts[provider] = provider_context

    numeric_consensus = {
        field: _field_consensus(field, values)
        for field, values in observations.items()
        if values
    }
    conflicts = [
        {
            "field": field,
            "dispersion": consensus.get("relative_dispersion"),
            "providers": [item["provider"] for item in consensus.get("used_observations", [])],
            "values": [item["value"] for item in consensus.get("used_observations", [])],
        }
        for field, consensus in numeric_consensus.items()
        if consensus.get("conflict")
    ]
    identity = _identity_consensus(identity_votes)
    recent_news = _dedupe_news(news_items, max_news_items)
    provider_health = _provider_health(provider_summaries)
    llm_evidence_manifest = _llm_evidence_manifest(source_payloads, provider_contexts)
    status = "ok" if provider_health["providers_ok"] else "blocked"
    if provider_health["providers_ok"] and provider_health["providers_error"]:
        status = "partial"

    return {
        "ticker": ticker.upper(),
        "status": status,
        "fetched_at": fetched_at,
        "provider_health": provider_health,
        "identity": identity,
        "numeric_consensus": numeric_consensus,
        "conflicts": conflicts,
        "stale_fields": stale_fields[:25],
        "recent_news": recent_news,
        "filing_context": filing_context,
        "provider_contexts": provider_contexts,
        "llm_evidence_manifest": llm_evidence_manifest,
        "decision_relevance": _decision_relevance(numeric_consensus, conflicts, stale_fields, recent_news),
    }


def _field_consensus(field: str, observations: list[dict[str, Any]]) -> dict[str, Any]:
    current = [item for item in observations if not item.get("stale")]
    used = current or observations
    values = [float(item["value"]) for item in used if _to_float(item.get("value")) is not None]
    if not values:
        return {"status": "unavailable", "observations": observations}
    median = float(pd.Series(values).median())
    max_value = max(values)
    min_value = min(values)
    denom = max(abs(median), 1e-12)
    dispersion = float((max_value - min_value) / denom) if len(values) > 1 else 0.0
    return {
        "status": "ok" if current else "stale_only",
        "value": median,
        "provider_count": len({item["provider"] for item in used}),
        "observation_count": len(used),
        "relative_dispersion": dispersion,
        "conflict": bool(len(values) > 1 and dispersion > _conflict_threshold(field)),
        "used_observations": used,
        "all_observations": observations,
    }


def _provider_summary(provider: str, payload: dict[str, Any]) -> dict[str, Any]:
    endpoint_status = payload.get("endpoint_status", {}) if isinstance(payload.get("endpoint_status"), dict) else {}
    total = len(endpoint_status)
    ok = 0
    for status in endpoint_status.values():
        if isinstance(status, dict) and str(status.get("status", "")).lower() in {"ok", "success", "non_empty"}:
            ok += 1
        elif isinstance(status, dict) and int(status.get("items") or status.get("rows") or 0) > 0:
            ok += 1
    normalized = payload.get("normalized", {}) if isinstance(payload.get("normalized"), dict) else {}
    return {
        "provider": provider,
        "status": "ok" if normalized else "empty",
        "source": payload.get("source"),
        "fetched_at": payload.get("fetched_at"),
        "api_key_fingerprint": payload.get("api_key_fingerprint"),
        "raw_sha256": payload.get("raw_sha256"),
        "ok_endpoints": ok,
        "total_endpoints": total,
        "normalized_sections": sorted([key for key, value in normalized.items() if value not in ({}, [], None)]),
    }


def _provider_health(provider_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    providers_ok = [name for name, summary in provider_summaries.items() if summary.get("status") == "ok"]
    providers_error = [name for name, summary in provider_summaries.items() if summary.get("status") == "error"]
    total_endpoints = sum(int(summary.get("total_endpoints") or 0) for summary in provider_summaries.values())
    ok_endpoints = sum(int(summary.get("ok_endpoints") or 0) for summary in provider_summaries.values())
    return {
        "providers_requested": len(provider_summaries),
        "providers_ok": providers_ok,
        "providers_error": providers_error,
        "endpoint_success_ratio": (ok_endpoints / total_endpoints) if total_endpoints else None,
    }


def _compact_provider_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": payload.get("symbol"),
        "source": payload.get("source"),
        "source_base_url": payload.get("source_base_url"),
        "fetched_at": payload.get("fetched_at"),
        "api_key_fingerprint": payload.get("api_key_fingerprint"),
        "endpoint_status": payload.get("endpoint_status", {}),
        "normalized": payload.get("normalized", {}),
        "raw_sha256": payload.get("raw_sha256"),
    }


def _write_artifacts(
    output_dir: Path,
    ticker: str,
    raw_payloads: dict[str, dict[str, Any]],
    provider_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ticker_dir = output_dir / _safe_symbol(ticker)
    ticker_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {"providers": {}}
    for provider, payload in raw_payloads.items():
        path = ticker_dir / f"{provider}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        artifacts["providers"][provider] = str(path)
    summary_path = ticker_dir / "provider_summaries.json"
    summary_path.write_text(json.dumps(provider_summaries, indent=2, sort_keys=True, default=str), encoding="utf-8")
    artifacts["provider_summaries"] = str(summary_path)
    return artifacts


def _recent_news(provider: str, normalized: dict[str, Any]) -> list[dict[str, Any]]:
    news = normalized.get("recent_news") if isinstance(normalized.get("recent_news"), list) else []
    result = []
    for item in news:
        if not isinstance(item, dict):
            continue
        result.append(
            {
                "provider": provider,
                "title": item.get("title"),
                "published_at": item.get("published_at") or item.get("date") or item.get("published_utc"),
                "url": item.get("url") or item.get("article_url") or item.get("link"),
                "summary": item.get("summary") or item.get("description"),
                "sentiment": item.get("sentiment"),
            }
        )
    return result


def _dedupe_news(news_items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in sorted(news_items, key=lambda row: str(row.get("published_at") or ""), reverse=True):
        key = str(item.get("url") or item.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append({key: value for key, value in item.items() if value not in (None, "", [], {})})
        if len(deduped) >= limit:
            break
    return deduped


def _filing_context(provider: str, normalized: dict[str, Any]) -> dict[str, Any]:
    filings = normalized.get("filings") if isinstance(normalized.get("filings"), dict) else {}
    if not filings:
        return {}
    compact = {}
    for key, value in filings.items():
        if isinstance(value, dict):
            text = value.get("text") or value.get("summary") or value.get("content")
            compact[key] = {item_key: item_value for item_key, item_value in value.items() if item_key != "text"}
            if text:
                compact[key]["excerpt"] = str(text)[:1200]
        else:
            compact[key] = value
    return {"provider": provider, "filings": compact}


def _provider_decision_context(provider: str, normalized: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for section, value in normalized.items():
        if value in (None, {}, []):
            continue
        if section == "recent_news" and isinstance(value, list):
            context[section] = _bounded_context(value[:8])
        else:
            context[str(section)] = _bounded_context(value)
    if context:
        context["provider"] = provider
    return context


def _llm_evidence_manifest(source_payloads: dict[str, dict[str, Any]], provider_contexts: dict[str, Any]) -> dict[str, Any]:
    providers: dict[str, Any] = {}
    omitted_total = 0
    for provider, payload in source_payloads.items():
        normalized = payload.get("normalized", {}) if isinstance(payload.get("normalized"), dict) else {}
        scraped_sections = sorted([key for key, value in normalized.items() if value not in (None, {}, [])])
        passed_context = provider_contexts.get(provider, {}) if isinstance(provider_contexts.get(provider), dict) else {}
        passed_sections = sorted([key for key in passed_context if key != "provider"])
        omitted = [section for section in scraped_sections if section not in passed_sections]
        omitted_total += len(omitted)
        providers[provider] = {
            "scraped_sections": scraped_sections,
            "passed_sections": passed_sections,
            "omitted_sections": omitted,
            "all_scraped_sections_passed": not omitted,
        }
    return {
        "policy": (
            "All non-empty normalized provider sections are passed to the autonomous LLM context. "
            "Raw HTML is not sent directly; extracted and bounded normalized evidence is sent with provider provenance."
        ),
        "providers": providers,
        "omitted_section_count": omitted_total,
        "all_scraped_sections_passed": omitted_total == 0,
    }


def _bounded_context(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return str(value)[:500]
    if isinstance(value, dict):
        result = {}
        for key, item in list(value.items())[:25]:
            if item in (None, "", [], {}):
                continue
            result[str(key)] = _bounded_context(item, depth=depth + 1)
        return result
    if isinstance(value, list):
        return [_bounded_context(item, depth=depth + 1) for item in value[:10]]
    if isinstance(value, str):
        return value[:1200]
    return value


def _identity_consensus(identity_votes: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, votes in identity_votes.items():
        if not votes:
            continue
        counts: dict[str, int] = {}
        for vote in votes:
            value = vote["value"].strip()
            counts[value] = counts.get(value, 0) + 1
        value = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        result[key] = {"value": value, "providers": [vote["provider"] for vote in votes if vote["value"] == value]}
    return result


def _decision_relevance(
    numeric_consensus: dict[str, dict[str, Any]],
    conflicts: list[dict[str, Any]],
    stale_fields: list[dict[str, Any]],
    recent_news: list[dict[str, Any]],
) -> dict[str, Any]:
    notes: list[str] = []
    if conflicts:
        notes.append("cross_provider_metric_conflicts_require_conservative_interpretation")
    if stale_fields:
        notes.append("some_provider_values_are_stale_and_excluded_when_current_values_exist")
    if recent_news:
        notes.append("recent_news_available_for_llm_context")
    if any(field in numeric_consensus for field in ("target_consensus", "target_high", "target_low")):
        notes.append("analyst_target_context_available")
    return {
        "can_influence_llm_review": True,
        "can_override_rule_gate": False,
        "can_enter_historical_model_training": False,
        "notes": notes,
    }


def _nested_get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _as_of_for_path(normalized: dict[str, Any], path: tuple[str, ...]) -> str | None:
    section = normalized.get(path[0]) if path else None
    if not isinstance(section, dict):
        return None
    for key in ("date", "period_end", "timestamp", "timestamp_ms", "year", "fiscal_date_ending"):
        if section.get(key):
            return str(section.get(key))
    return None


def _lookup_identity(normalized: dict[str, Any], key: str) -> Any:
    identity = normalized.get("identity") if isinstance(normalized.get("identity"), dict) else {}
    return identity.get(key)


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _is_stale(as_of: str | None, field: str, fetched_at: str) -> bool:
    if not as_of:
        return False
    timestamp = pd.to_datetime(_normalize_timestamp_text(as_of), utc=True, errors="coerce")
    fetched = pd.to_datetime(fetched_at, utc=True, errors="coerce")
    if pd.isna(timestamp) or pd.isna(fetched):
        return False
    age_days = int((fetched - timestamp).days)
    max_age = 10 if field in {"price", "close"} else 540
    return age_days > max_age


def _normalize_timestamp_text(value: str) -> str:
    return str(value).replace(" EDT", "").replace(" EST", "").replace(" CDT", "").replace(" CST", "").replace(" MDT", "").replace(" MST", "").replace(" PDT", "").replace(" PST", "")


def _conflict_threshold(field: str) -> float:
    if field in {"price", "close"}:
        return 0.03
    if field in {"market_cap", "enterprise_value", "revenue", "net_income", "free_cash_flow"}:
        return 0.12
    return 0.20


def _normalize_provider_list(providers: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(provider.strip().lower().replace("-", "_") for provider in providers if provider.strip())
    return normalized or DEFAULT_LONG_TERM_SOURCE_PROVIDERS


def _safe_symbol(symbol: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in symbol.strip()) or "ticker"


def _safe_error(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {str(exc)[:500]}"


def _default_start_date() -> str:
    return str((pd.Timestamp.now(tz=UTC) - pd.Timedelta(days=120)).date())
