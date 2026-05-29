from __future__ import annotations

from typing import Any

import numpy as np


def analyze_chapter_3_alternative_data(
    *,
    data_manifest: dict[str, Any],
    data_quality_report: dict[str, Any],
    market_feature_comparison: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate alternative data coverage, safety, and incremental value."""

    registry = _registry_entries(data_manifest)
    value_test = _value_test(market_feature_comparison)
    quality = [_quality_score(entry, data_quality_report, value_test) for entry in registry]
    status = "not_supplied"
    if registry:
        status = "pass" if all(item["status"] == "pass" for item in quality) else "warn"
    return {
        "chapter": 3,
        "name": "Alternative Data for Finance - Categories and Use Cases",
        "status": status,
        "decision_policy": {
            "influences_final_action": False,
            "influences_model_fitting": True,
            "mode": "feature_safety_filter",
            "reason": "Unsafe alternative-data sources can exclude alternative/news/sentiment features before model validation; validated sources still compete through normal walk-forward selection.",
        },
        "hybrid_collection_design": {
            "retrieval_layer": "Download or scrape public/provider data such as Yahoo RSS news, filings, or social/forum pages.",
            "extraction_layer": "Normalize records into dated article/post/event rows with source, URL, title/body, and entity/ticker mapping.",
            "sentiment_layer": "Use deterministic lexicon scoring by default; optionally blend with LLM classification when an API key is configured.",
            "feature_layer": "Aggregate point-in-time sentiment, attention volume, positive/negative share, and relevance into rolling model features.",
            "governance_layer": "Store raw records, feature hashes, provider metadata, point-in-time policy, and market-only value tests.",
        },
        "registry": registry,
        "quality_scores": quality,
        "value_test": value_test,
        "technical_method_card": chapter_3_alternative_data_method_card(),
    }


def chapter_3_alternative_data_method_card() -> dict[str, Any]:
    return {
        "name": "jansen_ml4t_chapter_3_alternative_data_registry",
        "version": "chapter_3_alternative_data_v1",
        "source": "Machine Learning for Algorithmic Trading, 2nd ed., Chapter 3",
        "purpose": "Register downloaded/scraped alternative datasets and evaluate quality, point-in-time safety, and incremental value.",
        "decision_policy": "feature_safety_filter",
    }


def _registry_entries(data_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for source in data_manifest.get("alternative_sources", []) or []:
        if isinstance(source, dict):
            registry = source.get("registry") if isinstance(source.get("registry"), dict) else source
            entries.append(registry)
    return entries


def _value_test(market_feature_comparison: dict[str, Any]) -> dict[str, Any]:
    if market_feature_comparison.get("status") != "compared":
        return {
            "status": market_feature_comparison.get("status", "not_applicable"),
            "reason": market_feature_comparison.get("reason", "No enriched-vs-market-only comparison is available."),
            "alternative_features_helped_any_horizon": False,
            "alternative_features_helped_all_horizons": False,
            "horizons": [],
        }
    horizons = []
    for item in market_feature_comparison.get("horizons", []):
        horizons.append(
            {
                "horizon_days": item.get("horizon_days"),
                "selection_metric": item.get("selection_metric"),
                "enriched_features_helped": bool(item.get("enriched_features_helped")),
                "metric_delta_enriched_minus_market": _finite(item.get("metric_delta_enriched_minus_market")),
            }
        )
    helped = [item["enriched_features_helped"] for item in horizons]
    return {
        "status": "compared",
        "principle": "Alternative data must improve walk-forward validation against a market-only baseline to justify production use.",
        "alternative_features_helped_any_horizon": bool(any(helped)),
        "alternative_features_helped_all_horizons": bool(helped and all(helped)),
        "horizons": horizons,
    }


def _quality_score(entry: dict[str, Any], data_quality_report: dict[str, Any], value_test: dict[str, Any]) -> dict[str, Any]:
    quality = entry.get("quality", {}) if isinstance(entry.get("quality"), dict) else {}
    score = 0
    warnings = []
    if entry.get("point_in_time_safe"):
        score += 25
    else:
        warnings.append("Dataset lacks point-in-time safety evidence.")
    if int(entry.get("article_count") or 0) >= 10 or int(quality.get("history_days") or 0) >= 7:
        score += 20
    else:
        warnings.append("Dataset has limited history or too few observations.")
    if entry.get("provider_status") == "ok":
        score += 15
    else:
        warnings.append("Provider retrieval did not report ok status.")
    if value_test.get("alternative_features_helped_any_horizon"):
        score += 25
    else:
        warnings.append("No validation improvement over market-only features has been shown yet.")
    if data_quality_report.get("status") != "fail":
        score += 15
    else:
        warnings.append("Run-level data quality has high-severity failures.")
    return {
        "name": entry.get("name", "alternative_dataset"),
        "score": int(score),
        "status": "pass" if score >= 70 else "warn",
        "warnings": warnings,
    }


def _finite(value: Any) -> float:
    try:
        numeric = float(value or 0.0)
    except Exception:
        return 0.0
    return numeric if np.isfinite(numeric) else 0.0
