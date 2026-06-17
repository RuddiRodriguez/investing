from __future__ import annotations

from market_forecasting_engine.chapter_18_tactics import analyze_chapter_18_tactical_problem
from market_forecasting_engine.long_term_sources import (
    _consolidate_sources,
    append_long_term_source_snapshot,
    build_long_term_source_snapshot_row,
    compact_long_term_context_for_llm,
    load_long_term_source_snapshot_features,
    parse_long_term_source_providers,
)


def test_long_term_source_consolidation_uses_current_values_and_marks_stale() -> None:
    source_payloads = {
        "fmp": {
            "normalized": {
                "identity": {"name": "Example Corp", "sector": "Technology", "currency": "USD"},
                "quote": {"price": 100.0},
                "valuation": {"market_cap": 1_000_000_000, "pe": 20.0},
                "income_statement": {"date": "2025-12-31", "revenue": 500_000_000},
                "recent_news": [{"title": "Example raises guidance", "published_at": "2026-06-12", "url": "https://x/news"}],
            }
        },
        "tiingo": {
            "normalized": {
                "identity": {"name": "Example Corp", "sector": "Technology"},
                "quote": {"date": "2026-06-13", "close": 101.0},
                "valuation": {"date": "2026-06-13", "market_cap": 1_050_000_000, "pe": 25.0},
                "income_statement": {"date": "2018-12-31", "revenue": 100_000_000},
            }
        },
    }
    provider_summaries = {
        "fmp": {"status": "ok", "ok_endpoints": 3, "total_endpoints": 3},
        "tiingo": {"status": "ok", "ok_endpoints": 2, "total_endpoints": 2},
    }

    consolidated = _consolidate_sources(
        "EXM",
        source_payloads,
        provider_summaries,
        "2026-06-14T10:00:00+00:00",
        max_news_items=5,
    )

    assert consolidated["status"] == "ok"
    assert consolidated["identity"]["name"]["value"] == "Example Corp"
    assert consolidated["numeric_consensus"]["market_cap"]["value"] == 1_025_000_000
    assert consolidated["numeric_consensus"]["pe"]["conflict"] is True
    assert consolidated["numeric_consensus"]["revenue"]["value"] == 500_000_000
    assert consolidated["numeric_consensus"]["revenue"]["used_observations"][0]["provider"] == "fmp"
    assert consolidated["stale_fields"][0]["provider"] == "tiingo"
    assert consolidated["recent_news"][0]["title"] == "Example raises guidance"


def test_compact_long_term_context_for_llm_removes_provider_raw_payloads() -> None:
    context = {
        "status": "partial",
        "fetched_at": "2026-06-14T10:00:00+00:00",
        "providers_requested": ["fmp"],
        "source_payloads": {"fmp": {"raw": {"large": "payload"}}},
        "consolidated": {
            "provider_health": {"providers_ok": ["fmp"]},
            "identity": {"name": {"value": "Example Corp"}},
            "numeric_consensus": {"pe": {"value": 20.0}},
            "conflicts": [],
            "recent_news": [],
            "decision_relevance": {"can_influence_llm_review": True},
        },
        "model_feature_policy": {"model_training_included": False},
    }

    compact = compact_long_term_context_for_llm(context)

    assert compact["numeric_consensus"]["pe"]["value"] == 20.0
    assert "source_payloads" not in compact
    assert compact["model_feature_policy"]["model_training_included"] is False


def test_long_term_snapshots_join_asof_without_future_leakage(tmp_path) -> None:
    import pandas as pd

    base_context = {
        "ticker": "EXM",
        "status": "ok",
        "providers_requested": ["fmp"],
        "consolidated": {
            "provider_health": {"providers_ok": ["fmp"], "providers_error": [], "endpoint_success_ratio": 1.0},
            "numeric_consensus": {
                "pe": {
                    "status": "ok",
                    "value": 20.0,
                    "relative_dispersion": 0.0,
                    "provider_count": 1,
                }
            },
            "conflicts": [],
            "stale_fields": [],
            "recent_news": [],
        },
        "model_feature_policy": {"status": "current_snapshot_context_only"},
    }
    first = {**base_context, "fetched_at": "2026-01-05T12:00:00+00:00"}
    second = {
        **base_context,
        "fetched_at": "2026-01-08T12:00:00+00:00",
        "consolidated": {
            **base_context["consolidated"],
            "numeric_consensus": {
                "pe": {
                    "status": "ok",
                    "value": 30.0,
                    "relative_dispersion": 0.0,
                    "provider_count": 1,
                }
            },
        },
    }

    append_long_term_source_snapshot(first, tmp_path, ticker="EXM")
    append_long_term_source_snapshot(second, tmp_path, ticker="EXM")
    index = pd.DatetimeIndex(
        [
            "2026-01-05T11:00:00+00:00",
            "2026-01-06T16:00:00+00:00",
            "2026-01-08T11:00:00+00:00",
            "2026-01-09T16:00:00+00:00",
        ]
    )

    features, metadata = load_long_term_source_snapshot_features("EXM", tmp_path, index)

    assert metadata["status"] == "ok"
    assert pd.isna(features.loc[index[0], "lts_pe"])
    assert features.loc[index[1], "lts_pe"] == 20.0
    assert features.loc[index[2], "lts_pe"] == 20.0
    assert features.loc[index[3], "lts_pe"] == 30.0


def test_build_snapshot_row_contains_model_feature_audit_fields() -> None:
    context = {
        "ticker": "EXM",
        "fetched_at": "2026-01-05T12:00:00+00:00",
        "status": "ok",
        "providers_requested": ["fmp", "tiingo"],
        "consolidated": {
            "provider_health": {"providers_ok": ["fmp"], "providers_error": ["tiingo"], "endpoint_success_ratio": 0.5},
            "numeric_consensus": {
                "market_cap": {"status": "ok", "value": 1000, "relative_dispersion": 0.1, "provider_count": 2}
            },
            "conflicts": [{"field": "market_cap"}],
            "stale_fields": [{"field": "revenue"}],
            "recent_news": [{"title": "x"}],
        },
        "model_feature_policy": {"status": "current_snapshot_context_only"},
    }

    row = build_long_term_source_snapshot_row(context)

    assert row["version"] == "long_term_source_snapshot_v1"
    assert row["features"]["lts_market_cap"] == 1000.0
    assert row["features"]["lts_endpoint_success_ratio"] == 0.5
    assert row["features"]["lts_conflict_count"] == 1.0
    assert row["model_feature_policy"]["model_training_included"] is True


def test_parse_long_term_source_providers_rejects_unknown_provider() -> None:
    try:
        parse_long_term_source_providers("fmp,bad_source")
    except ValueError as exc:
        assert "bad_source" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_chapter_18_llm_packet_includes_long_term_context() -> None:
    import pandas as pd

    prices = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0], "volume": [1000, 1100, 1200]},
        index=pd.date_range("2026-06-10", periods=3),
    )
    forecasts = [
        {
            "horizon_days": 1,
            "expected_direction": "Up",
            "expected_return": 0.02,
            "directional_confidence": 0.58,
            "predicted_price": 104.0,
        }
    ]
    context = {
        "status": "ok",
        "fetched_at": "2026-06-14T10:00:00+00:00",
        "providers_requested": ["fmp"],
        "consolidated": {
            "provider_health": {"providers_ok": ["fmp"]},
            "numeric_consensus": {"target_consensus": {"value": 120.0}},
            "decision_relevance": {"can_influence_llm_review": True},
        },
        "model_feature_policy": {"model_training_included": False},
    }

    result = analyze_chapter_18_tactical_problem(
        prices=prices,
        forecasts=forecasts,
        part_i_action="Buy",
        raw_action="Buy",
        risk_level="Medium",
        decision_diagnostics={},
        technical_contexts={},
        long_term_context=context,
    )

    packet_context = result["llm_review_packet"]["long_term_context"]
    assert packet_context["numeric_consensus"]["target_consensus"]["value"] == 120.0
    assert packet_context["model_feature_policy"]["model_training_included"] is False
