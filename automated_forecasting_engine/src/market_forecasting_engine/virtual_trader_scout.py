from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from market_forecasting_engine.data import normalize_price_frame
from market_forecasting_engine.data_providers import DataRequest, load_prices_with_provider
from market_forecasting_engine.data_store import MarketDataStore
from market_forecasting_engine.openai_models import DEFAULT_OPENAI_MODEL, DEFAULT_REASONING_EFFORT
from market_forecasting_engine.stockanalysis_analyst_flow import (
    AnalystTickerCandidate,
    BASE_URL as STOCKANALYSIS_BASE_URL,
    scrape_visible_analyst_tickers,
    scrape_visible_analysts,
)


DEFAULT_SCOUT_UNIVERSE_MAX_TICKERS = 350
DEFAULT_FINAL_CANDIDATES = 3
DEFAULT_MIN_AVG_DOLLAR_VOLUME = 20_000_000.0
DEFAULT_MIN_PRICE = 5.0
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
SCOUT_LLM_SYSTEM_MESSAGE = """
# Role: Virtual Trader Cheap Subjective Ranker

You are the subjective ranking layer for a serious autonomous stock-screening pipeline.

The deterministic system already scored liquidity, price trend, setup shape, volatility, and portfolio diversification from numeric data. Your job is only to assess the parts that need current market judgment:

- latest catalyst/news quality,
- valuation sanity at a high level,
- whether the setup narrative is coherent or crowded,
- red flags that should reduce priority before expensive forecasting,
- portfolio diversification concerns when portfolio context is supplied.

## Rules

- Use web search when available to check current information.
- Do not make a final Buy/Hold/Sell trading decision.
- Do not recommend orders.
- Do not override hard eligibility gates.
- Do not promote a ticker purely because it is popular; explain the catalyst and risk.
- Return one object matching the schema.
- Scores must be decimals from 0.0 to 1.0 where 1.0 is best, except red flags which are text.
- If current information is thin, say so and use moderate confidence.
""".strip()

SCOUT_LLM_USER_MESSAGE = """
Today:
{{ item.today }}

Portfolio context:
{{ item.portfolio_context_json }}

Candidate shortlist from deterministic scout:
{{ item.shortlist_json }}

Task:
For each ticker in the shortlist, assess only the subjective parts that benefit from current context and web search:
1. catalyst/news quality,
2. valuation sanity,
3. setup quality narrative,
4. risk/red flags,
5. portfolio diversification concerns.

Return structured scores and concise rationale for every supplied ticker.
""".strip()

SCOUT_LLM_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "virtual_trader_subjective_ranking",
    "description": "Subjective web-aware ranking overlay for the virtual trader cheap ranking layer.",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "analysis": {"type": "string"},
            "market_regime_note": {"type": "string"},
            "ranked_tickers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "valuation_sanity_score": {"type": "number"},
                        "catalyst_news_score": {"type": "number"},
                        "setup_quality_score": {"type": "number"},
                        "subjective_conviction_score": {"type": "number"},
                        "confidence": {"type": "number"},
                        "setup_quality_comment": {"type": "string"},
                        "valuation_comment": {"type": "string"},
                        "catalyst_summary": {"type": "string"},
                        "risk_comment": {"type": "string"},
                        "portfolio_diversification_comment": {"type": "string"},
                        "positive_evidence": {"type": "array", "items": {"type": "string"}},
                        "red_flags": {"type": "array", "items": {"type": "string"}},
                        "web_sources_used": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "ticker",
                        "valuation_sanity_score",
                        "catalyst_news_score",
                        "setup_quality_score",
                        "subjective_conviction_score",
                        "confidence",
                        "setup_quality_comment",
                        "valuation_comment",
                        "catalyst_summary",
                        "risk_comment",
                        "portfolio_diversification_comment",
                        "positive_evidence",
                        "red_flags",
                        "web_sources_used",
                    ],
                    "additionalProperties": False,
                },
            },
            "cross_candidate_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["analysis", "market_regime_note", "ranked_tickers", "cross_candidate_notes"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class ScoutConfig:
    output_dir: str | Path
    start: str = "2025-01-01"
    end: str | None = None
    provider: str = "yahoo"
    data_dir: str | Path | None = None
    env_file: str | Path | None = None
    max_universe_tickers: int = DEFAULT_SCOUT_UNIVERSE_MAX_TICKERS
    final_candidates: int = DEFAULT_FINAL_CANDIDATES
    analyst_pages: int = 12
    analyst_timeout_seconds: float = 20.0
    include_stockanalysis_market_pages: bool = True
    include_fmp_market_movers: bool = True
    include_fmp_news: bool = True
    include_fmp_earnings: bool = True
    min_price: float = DEFAULT_MIN_PRICE
    min_avg_dollar_volume: float = DEFAULT_MIN_AVG_DOLLAR_VOLUME
    max_realized_volatility_20d: float = 1.50
    enable_llm_ranking: bool = True
    llm_rank_top_n: int = 12
    llm_provider: str = "openai"
    llm_model: str | None = None
    llm_reasoning_effort: str = DEFAULT_REASONING_EFFORT
    llm_timeout_seconds: float = 90.0
    llm_search_context_size: str = "low"
    llm_use_web_search: bool = True
    llm_require_web_search: bool = False
    portfolio_tickers: tuple[str, ...] = ()
    portfolio_sectors: tuple[str, ...] = ()
    refresh_data_cache: bool = False
    use_data_cache: bool = True
    progress: bool = True


@dataclass(frozen=True)
class DiscoveryRecord:
    ticker: str
    source: str
    reason: str
    metadata: dict[str, Any]


def run_virtual_trader_scout(config: ScoutConfig) -> dict[str, Any]:
    """Run the cheap daily discovery and ranking layer for the virtual trader.

    This layer is intentionally forecast-free. It scans dynamic public/API sources,
    pulls only cheap historical bars for discovered names, applies deterministic
    ranking where the data is numeric, and can use a bounded LLM/web-search overlay
    only for subjective catalyst, valuation, setup, and red-flag interpretation.
    """

    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    _load_env(config.env_file)
    data_store = MarketDataStore(Path(config.data_dir).expanduser()) if config.data_dir else MarketDataStore(output_dir / "data")
    started_at = datetime.now(UTC)
    _progress(config, f"virtual trader scout started output={output_dir}")

    discovery_records, source_audit = discover_candidate_universe(config)
    universe = _build_stratified_universe(discovery_records, config.max_universe_tickers)
    _progress(config, f"discovery complete records={len(discovery_records)} unique_tickers={len(universe)}")

    price_frames, price_audit = load_scout_price_frames(universe, config, data_store)
    _progress(config, f"price scan complete ok={len(price_frames)} failed={len(price_audit.get('failed', []))}")

    scored = score_scout_candidates(universe, discovery_records, price_frames, config)
    ranked, ranking_audit = rank_scout_candidates(scored, config, output_dir=output_dir)
    selected = [row for row in ranked if row.get("eligible")][: config.final_candidates]
    selected_tickers = {item["ticker"] for item in selected}
    for row in ranked:
        row["selected"] = row["ticker"] in selected_tickers
    rejected = [row for row in ranked if not row.get("selected")]

    summary = {
        "run_type": "virtual_trader_cheap_discovery_and_ranking_layer",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "policy": {
            "purpose": "Find a small candidate set for the expensive forecast/CEO pipeline.",
            "forecast_free": True,
            "llm_free": not bool(config.enable_llm_ranking),
            "llm_usage": "Only bounded shortlist ranking for subjective catalyst/valuation/setup interpretation when enabled.",
            "execution_allowed": False,
            "output_use": "Candidate discovery only; selected tickers still require full forecast, CEO decision, and risk gates.",
        },
        "counts": {
            "discovery_records": len(discovery_records),
            "unique_discovered_tickers": len(universe),
            "price_frames_loaded": len(price_frames),
            "scored_candidates": len(ranked),
            "eligible_candidates": len([row for row in ranked if row.get("eligible")]),
            "selected_candidates": len(selected),
            "rejected_candidates": len(rejected),
        },
        "source_audit": source_audit,
        "price_audit": price_audit,
        "ranking_audit": ranking_audit,
        "selected_candidates": selected,
        "top_candidates": ranked[: min(25, len(ranked))],
        "artifact_paths": {},
    }

    artifacts = {
        "discovery_sources": _write_json(output_dir / "discovery_sources.json", [_record_to_dict(record) for record in discovery_records]),
        "source_audit": _write_json(output_dir / "source_audit.json", source_audit),
        "price_audit": _write_json(output_dir / "price_audit.json", price_audit),
        "candidate_scores": _write_json(output_dir / "candidate_scores.json", ranked),
        "cheap_ranking": _write_json(output_dir / "cheap_ranking.json", ranked),
        "ranking_audit": _write_json(output_dir / "ranking_audit.json", ranking_audit),
        "selected_candidates": _write_json(output_dir / "selected_candidates.json", selected),
        "rejected_candidates": _write_json(output_dir / "rejected_candidates.json", rejected),
        "run_summary": str(output_dir / "run_summary.md"),
    }
    llm_ranking_audit = ranking_audit.get("llm_ranking", {}) if isinstance(ranking_audit.get("llm_ranking"), dict) else {}
    for key in ("payload_path", "raw_response_path", "parsed_path"):
        if llm_ranking_audit.get(key):
            artifacts[f"llm_ranking_{key.removesuffix('_path')}"] = llm_ranking_audit[key]
    _write_csv(output_dir / "candidate_scores.csv", ranked)
    _write_csv(output_dir / "cheap_ranking.csv", ranked)
    _write_csv(output_dir / "selected_candidates.csv", selected)
    _write_markdown_summary(output_dir / "run_summary.md", summary)
    summary["artifact_paths"] = artifacts
    _write_json(output_dir / "scout_summary.json", summary)
    _progress(config, f"scout complete selected={','.join(item['ticker'] for item in selected) or 'none'}")
    return summary


def discover_candidate_universe(config: ScoutConfig) -> tuple[list[DiscoveryRecord], dict[str, Any]]:
    records: list[DiscoveryRecord] = []
    audit: dict[str, Any] = {"sources": {}, "errors": []}

    try:
        analyst_records = _discover_stockanalysis_analyst_records(config)
        records.extend(analyst_records)
        audit["sources"]["stockanalysis_visible_analysts"] = {"status": "ok", "records": len(analyst_records)}
    except Exception as exc:
        audit["sources"]["stockanalysis_visible_analysts"] = {"status": "error", "error": _safe_error(exc), "records": 0}
        audit["errors"].append({"source": "stockanalysis_visible_analysts", "error": _safe_error(exc)})

    if config.include_stockanalysis_market_pages:
        try:
            market_records = _discover_stockanalysis_market_pages(config)
            records.extend(market_records)
            audit["sources"]["stockanalysis_market_pages"] = {"status": "ok", "records": len(market_records)}
        except Exception as exc:
            audit["sources"]["stockanalysis_market_pages"] = {"status": "error", "error": _safe_error(exc), "records": 0}
            audit["errors"].append({"source": "stockanalysis_market_pages", "error": _safe_error(exc)})

    if config.include_fmp_market_movers:
        try:
            mover_records = _discover_fmp_market_movers(config)
            records.extend(mover_records)
            audit["sources"]["fmp_market_movers"] = {"status": "ok", "records": len(mover_records)}
        except Exception as exc:
            audit["sources"]["fmp_market_movers"] = {"status": "error", "error": _safe_error(exc), "records": 0}
            audit["errors"].append({"source": "fmp_market_movers", "error": _safe_error(exc)})

    if config.include_fmp_news:
        try:
            news_records = _discover_fmp_news_activity(config)
            records.extend(news_records)
            audit["sources"]["fmp_news_activity"] = {"status": "ok", "records": len(news_records)}
        except Exception as exc:
            audit["sources"]["fmp_news_activity"] = {"status": "error", "error": _safe_error(exc), "records": 0}
            audit["errors"].append({"source": "fmp_news_activity", "error": _safe_error(exc)})

    if config.include_fmp_earnings:
        try:
            earnings_records = _discover_fmp_earnings_activity(config)
            records.extend(earnings_records)
            audit["sources"]["fmp_earnings_activity"] = {"status": "ok", "records": len(earnings_records)}
        except Exception as exc:
            audit["sources"]["fmp_earnings_activity"] = {"status": "error", "error": _safe_error(exc), "records": 0}
            audit["errors"].append({"source": "fmp_earnings_activity", "error": _safe_error(exc)})

    return records, audit


def load_scout_price_frames(
    tickers: list[str],
    config: ScoutConfig,
    data_store: MarketDataStore | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    frames: dict[str, pd.DataFrame] = {}
    failed: list[dict[str, Any]] = []
    for position, ticker in enumerate(tickers, start=1):
        _progress(config, f"price scan {position}/{len(tickers)} {ticker}")
        try:
            result = load_prices_with_provider(
                config.provider,
                DataRequest(ticker=ticker, start=config.start, end=config.end, interval="1d", target_column="close"),
                store=data_store,
                use_cache=config.use_data_cache,
                refresh_cache=config.refresh_data_cache,
            )
            frame = normalize_price_frame(result.frame, target_column="close")
            if len(frame) < 65:
                failed.append({"ticker": ticker, "reason": "insufficient_price_rows", "rows": len(frame)})
                continue
            frames[ticker] = frame
        except Exception as exc:
            failed.append({"ticker": ticker, "reason": "price_fetch_failed", "error": _safe_error(exc)})
    return frames, {"provider": config.provider, "ok": sorted(frames), "failed": failed}


def score_scout_candidates(
    tickers: list[str],
    discovery_records: list[DiscoveryRecord],
    price_frames: dict[str, pd.DataFrame],
    config: ScoutConfig,
) -> list[dict[str, Any]]:
    discovery_by_ticker: dict[str, list[DiscoveryRecord]] = {}
    for record in discovery_records:
        discovery_by_ticker.setdefault(record.ticker, []).append(record)

    feature_rows = []
    for ticker in tickers:
        frame = price_frames.get(ticker)
        if frame is None or frame.empty:
            feature_rows.append(_missing_price_row(ticker, discovery_by_ticker.get(ticker, []), config))
            continue
        feature_rows.append(_score_one_ticker(ticker, frame, discovery_by_ticker.get(ticker, []), config))

    _add_sector_strength(feature_rows)
    for row in feature_rows:
        row["score_components"]["sector_strength"] = row.get("sector_strength_score", 0.5)
        row["score"] = _weighted_score(row["score_components"]) - row.get("risk_penalty", 0.0)
        row["score"] = round(float(max(0.0, min(1.0, row["score"]))), 4)
        row["eligible"] = bool(row["eligible"] and row["score"] >= 0.35)
        if not row["eligible"] and "low_total_score" not in row["rejection_reasons"] and row["score"] < 0.35:
            row["rejection_reasons"].append("low_total_score")
    ranked = sorted(feature_rows, key=lambda item: (bool(item.get("eligible")), item.get("score", 0.0)), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["selected"] = False
    return ranked


def rank_scout_candidates(
    scored_rows: list[dict[str, Any]],
    config: ScoutConfig,
    *,
    output_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the cheap ranking layer used to decide which tickers get expensive forecasts."""

    ranked_rows = [dict(row) for row in scored_rows]
    sector_counts = _sector_counts(ranked_rows)
    portfolio_tickers = {_normalize_ticker(ticker) for ticker in config.portfolio_tickers if _normalize_ticker(ticker)}
    portfolio_sectors = {str(sector).strip().lower() for sector in config.portfolio_sectors if str(sector).strip()}
    for row in ranked_rows:
        components = _ranking_components(row, sector_counts, portfolio_tickers, portfolio_sectors)
        row["ranking_components"] = components
        row["ranking_score"] = _weighted_ranking_score(components)
        row["ranking_policy"] = {
            "numeric_components": [
                "liquidity_score",
                "trend_score",
                "setup_quality_score",
                "valuation_sanity_score",
                "catalyst_news_score",
                "risk_score",
                "portfolio_diversification_score",
            ],
            "llm_components": [],
            "forecast_free": True,
            "execution_allowed": False,
        }
    audit: dict[str, Any] = {
        "method": "cheap_numeric_ranker",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "llm_ranking": {
            "enabled": bool(config.enable_llm_ranking),
            "status": "not_requested" if not config.enable_llm_ranking else "pending",
            "rank_top_n": int(config.llm_rank_top_n),
            "provider": config.llm_provider,
            "model": config.llm_model or DEFAULT_OPENAI_MODEL,
            "web_search": bool(config.llm_use_web_search),
        },
        "portfolio_context": {
            "portfolio_tickers": sorted(portfolio_tickers),
            "portfolio_sectors": sorted(portfolio_sectors),
        },
        "weights": _ranking_weights(),
    }

    if config.enable_llm_ranking:
        llm_audit = apply_llm_subjective_ranking(ranked_rows, config, output_dir=output_dir)
        audit["llm_ranking"] = llm_audit

    ranked_rows = sorted(
        ranked_rows,
        key=lambda item: (bool(item.get("eligible")), float(item.get("ranking_score", 0.0)), float(item.get("score", 0.0))),
        reverse=True,
    )
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank
    audit["selected_sort"] = "eligible desc, ranking_score desc, original_score desc"
    return ranked_rows, audit


def apply_llm_subjective_ranking(
    rows: list[dict[str, Any]],
    config: ScoutConfig,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    eligible_rows = [row for row in rows if row.get("eligible")]
    shortlist = sorted(eligible_rows, key=lambda item: float(item.get("ranking_score", 0.0)), reverse=True)[: max(0, int(config.llm_rank_top_n))]
    audit = {
        "enabled": True,
        "status": "skipped_no_eligible_shortlist" if not shortlist else "pending",
        "rank_top_n": int(config.llm_rank_top_n),
        "shortlist": [row.get("ticker") for row in shortlist],
        "provider": config.llm_provider,
        "model": config.llm_model or DEFAULT_OPENAI_MODEL,
        "web_search": bool(config.llm_use_web_search),
        "require_web_search": bool(config.llm_require_web_search),
        "bounded_effect_policy": "LLM contributes bounded subjective scores and red flags; it cannot bypass eligibility gates.",
    }
    if not shortlist:
        return audit
    try:
        from market_forecasting_engine.llm_trader.responses_api import call_response
        from market_forecasting_engine.llm_trader.run import (
            openai_client_for_provider,
            resolve_llm_model,
            resolve_llm_provider,
        )

        provider = resolve_llm_provider(config.llm_provider)
        model = resolve_llm_model(config.llm_model, provider=provider)
        use_web_search = bool(config.llm_use_web_search and provider == "openai")
        client = openai_client_for_provider(provider, timeout=config.llm_timeout_seconds)
        item = {
            "today": datetime.now(UTC).date().isoformat(),
            "shortlist_json": json.dumps(_llm_shortlist_payload(shortlist), indent=2, sort_keys=True, default=str),
            "portfolio_context_json": json.dumps(
                {
                    "portfolio_tickers": sorted(_normalize_ticker(ticker) for ticker in config.portfolio_tickers if _normalize_ticker(ticker)),
                    "portfolio_sectors": sorted(str(sector) for sector in config.portfolio_sectors if str(sector).strip()),
                },
                indent=2,
                sort_keys=True,
            ),
        }
        payload, raw_response, parsed = call_response(
            model=model,
            system_message=SCOUT_LLM_SYSTEM_MESSAGE,
            user_message=SCOUT_LLM_USER_MESSAGE,
            json_schema=SCOUT_LLM_JSON_SCHEMA,
            item=item,
            use_web_search=use_web_search,
            search_context_size=config.llm_search_context_size,
            client=client,
            provider=provider,
            reasoning_effort=config.llm_reasoning_effort,
            usage_context={
                "process": "virtual_trader_scout",
                "purpose": "cheap_subjective_candidate_ranking",
                "rank_top_n": len(shortlist),
                "web_search": use_web_search,
            },
            require_web_search=bool(config.llm_require_web_search),
        )
        if output_dir:
            _write_json(output_dir / "llm_ranking_payload.json", payload)
            _write_json(output_dir / "llm_ranking_raw_response.json", raw_response)
            _write_json(output_dir / "llm_ranking.json", parsed)
        _merge_llm_ranking(rows, parsed)
        audit.update(
            {
                "status": "ok",
                "provider": provider,
                "model": model,
                "payload_path": str(output_dir / "llm_ranking_payload.json") if output_dir else None,
                "raw_response_path": str(output_dir / "llm_ranking_raw_response.json") if output_dir else None,
                "parsed_path": str(output_dir / "llm_ranking.json") if output_dir else None,
                "llm_output_summary": {
                    "market_regime_note": parsed.get("market_regime_note"),
                    "ranked_tickers": [item.get("ticker") for item in parsed.get("ranked_tickers", []) if isinstance(item, dict)],
                },
            }
        )
        return audit
    except Exception as exc:
        audit.update({"status": "error", "error": _safe_error(exc)})
        if output_dir:
            _write_json(output_dir / "llm_ranking_error.json", audit)
        return audit


def _ranking_components(
    row: dict[str, Any],
    sector_counts: dict[str, int],
    portfolio_tickers: set[str],
    portfolio_sectors: set[str],
) -> dict[str, float]:
    components = row.get("score_components", {}) if isinstance(row.get("score_components"), dict) else {}
    metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
    liquidity_score = float(components.get("liquidity", 0.0))
    trend_score = _trend_rank_score(components, metrics)
    setup_quality_score = _setup_quality_rank_score(components, metrics)
    valuation_sanity_score = _deterministic_valuation_sanity(row)
    catalyst_news_score = _catalyst_news_rank_score(row)
    risk_score = _risk_rank_score(row)
    diversification_score = _portfolio_diversification_rank_score(row, sector_counts, portfolio_tickers, portfolio_sectors)
    return {
        "liquidity_score": round(_clip(liquidity_score), 4),
        "trend_score": round(_clip(trend_score), 4),
        "setup_quality_score": round(_clip(setup_quality_score), 4),
        "valuation_sanity_score": round(_clip(valuation_sanity_score), 4),
        "catalyst_news_score": round(_clip(catalyst_news_score), 4),
        "risk_score": round(_clip(risk_score), 4),
        "portfolio_diversification_score": round(_clip(diversification_score), 4),
        "llm_subjective_score": 0.0,
        "llm_red_flag_penalty": 0.0,
    }


def _weighted_ranking_score(components: dict[str, float]) -> float:
    score = sum(_ranking_weights()[key] * float(components.get(key, 0.0)) for key in _ranking_weights())
    score -= 0.08 * float(components.get("llm_red_flag_penalty", 0.0))
    return round(float(max(0.0, min(1.0, score))), 4)


def _ranking_weights() -> dict[str, float]:
    return {
        "liquidity_score": 0.14,
        "trend_score": 0.15,
        "setup_quality_score": 0.16,
        "valuation_sanity_score": 0.11,
        "catalyst_news_score": 0.14,
        "risk_score": 0.15,
        "portfolio_diversification_score": 0.07,
        "llm_subjective_score": 0.08,
    }


def _trend_rank_score(components: dict[str, Any], metrics: dict[str, Any]) -> float:
    momentum = float(components.get("momentum", 0.0))
    sector = float(components.get("sector_strength", 0.5))
    sma_200_distance = float(metrics.get("distance_to_sma_200_pct") or 0.0)
    range_position = float(metrics.get("range_position_252d") or 0.5)
    long_trend = _clip(0.5 + np.tanh(sma_200_distance * 4.0) / 2.0)
    range_health = _band_score(range_position, 0.05, 0.35, 0.88, 0.99)
    return float(0.50 * momentum + 0.20 * sector + 0.20 * long_trend + 0.10 * range_health)


def _setup_quality_rank_score(components: dict[str, Any], metrics: dict[str, Any]) -> float:
    breakout = float(components.get("breakout", 0.0))
    pullback = float(components.get("pullback", 0.0))
    volume = float(components.get("unusual_volume", 0.0))
    atr_pct = float(metrics.get("atr_pct_20d") or 0.0)
    tradable_range = _band_score(atr_pct, 0.004, 0.012, 0.055, 0.095)
    return float(0.42 * max(breakout, pullback) + 0.20 * min(1.0, breakout + pullback) + 0.18 * volume + 0.20 * tradable_range)


def _deterministic_valuation_sanity(row: dict[str, Any]) -> float:
    records = row.get("latest_records", []) if isinstance(row.get("latest_records"), list) else []
    market_caps = []
    revenues = []
    upside_values = []
    for record in records:
        metadata = record.get("metadata", {}) if isinstance(record, dict) else {}
        market_cap = _to_float(metadata.get("market_cap"))
        revenue = _to_float(metadata.get("revenue"))
        upside = _to_float(metadata.get("upside"))
        if market_cap:
            market_caps.append(market_cap)
        if revenue:
            revenues.append(revenue)
        if upside is not None:
            upside_values.append(upside / 100.0 if abs(upside) > 1.5 else upside)
    score = 0.50
    if market_caps:
        cap = max(market_caps)
        if cap >= 2_000_000_000:
            score += 0.08
        if cap >= 25_000_000_000:
            score += 0.04
        if cap < 500_000_000:
            score -= 0.12
    if market_caps and revenues:
        ps_ratio = _safe_ratio(max(market_caps), max(revenues))
        score += 0.10 * _band_score(ps_ratio, 0.2, 0.6, 9.0, 22.0)
        if ps_ratio > 18:
            score -= 0.08
    if upside_values:
        median_upside = float(np.nanmedian(upside_values))
        score += 0.14 * _clip((median_upside + 0.05) / 0.35)
        if median_upside < -0.05:
            score -= 0.10
    return _clip(score)


def _catalyst_news_rank_score(row: dict[str, Any]) -> float:
    components = row.get("score_components", {}) if isinstance(row.get("score_components"), dict) else {}
    source_counts = row.get("source_counts", {}) if isinstance(row.get("source_counts"), dict) else {}
    analyst = float(components.get("analyst_activity", 0.0))
    news = float(components.get("news_activity", 0.0))
    earnings = float(components.get("earnings_activity", 0.0))
    mover = float(components.get("market_mover", 0.0))
    market_pages = min(
        1.0,
        (
            int(source_counts.get("stockanalysis_market_gainers", 0))
            + int(source_counts.get("stockanalysis_market_active", 0))
            + int(source_counts.get("stockanalysis_market_losers", 0))
        )
        / 2.0,
    )
    return float(0.32 * analyst + 0.22 * news + 0.18 * earnings + 0.18 * mover + 0.10 * market_pages)


def _risk_rank_score(row: dict[str, Any]) -> float:
    metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
    components = row.get("score_components", {}) if isinstance(row.get("score_components"), dict) else {}
    realized_vol = float(metrics.get("realized_volatility_20d") or 0.0)
    atr_pct = float(metrics.get("atr_pct_20d") or 0.0)
    liquidity = float(components.get("liquidity", 0.0))
    range_position = float(metrics.get("range_position_252d") or 0.5)
    volatility_quality = _volatility_fit_score(realized_vol, atr_pct)
    extension_penalty = 0.18 if range_position > 0.96 else 0.08 if range_position > 0.91 else 0.0
    existing_penalty = float(row.get("risk_penalty", 0.0))
    return _clip(0.55 * volatility_quality + 0.25 * liquidity + 0.20 * (1.0 - min(1.0, existing_penalty / 0.20)) - extension_penalty)


def _portfolio_diversification_rank_score(
    row: dict[str, Any],
    sector_counts: dict[str, int],
    portfolio_tickers: set[str],
    portfolio_sectors: set[str],
) -> float:
    ticker = _normalize_ticker(row.get("ticker"))
    sector = str(row.get("sector") or "Unknown")
    sector_key = sector.strip().lower()
    score = 0.70
    if ticker and ticker in portfolio_tickers:
        score -= 0.35
    if sector_key and sector_key in portfolio_sectors:
        score -= 0.20
    if sector and sector != "Unknown":
        count = sector_counts.get(sector, 0)
        if count > 1:
            score -= min(0.20, 0.035 * (count - 1))
        else:
            score += 0.10
    else:
        score -= 0.05
    return _clip(score)


def _sector_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        sector = str(row.get("sector") or "Unknown")
        counts[sector] = counts.get(sector, 0) + 1
    return counts


def _llm_shortlist_payload(shortlist: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = []
    for row in shortlist:
        payload.append(
            {
                "ticker": row.get("ticker"),
                "sector": row.get("sector"),
                "latest_close": row.get("latest_close"),
                "eligible": row.get("eligible"),
                "discovery_sources": row.get("discovery_sources"),
                "source_counts": row.get("source_counts"),
                "reasons": row.get("reasons"),
                "metrics": row.get("metrics"),
                "original_score": row.get("score"),
                "ranking_score_before_llm": row.get("ranking_score"),
                "ranking_components_before_llm": row.get("ranking_components"),
                "latest_records": row.get("latest_records", [])[:6],
            }
        )
    return payload


def _merge_llm_ranking(rows: list[dict[str, Any]], parsed: dict[str, Any]) -> None:
    by_ticker = {_normalize_ticker(row.get("ticker")): row for row in rows}
    ranked_tickers = parsed.get("ranked_tickers", [])
    if not isinstance(ranked_tickers, list):
        return
    for item in ranked_tickers:
        if not isinstance(item, dict):
            continue
        ticker = _normalize_ticker(item.get("ticker"))
        row = by_ticker.get(ticker)
        if row is None:
            continue
        components = row.setdefault("ranking_components", {})
        components["valuation_sanity_score"] = _bounded_llm_component(
            components.get("valuation_sanity_score", 0.5),
            item.get("valuation_sanity_score"),
            max_delta=0.18,
        )
        components["catalyst_news_score"] = _bounded_llm_component(
            components.get("catalyst_news_score", 0.0),
            item.get("catalyst_news_score"),
            max_delta=0.22,
        )
        components["setup_quality_score"] = _bounded_llm_component(
            components.get("setup_quality_score", 0.5),
            item.get("setup_quality_score"),
            max_delta=0.12,
        )
        subjective_values = [
            _to_float(item.get("valuation_sanity_score")),
            _to_float(item.get("catalyst_news_score")),
            _to_float(item.get("setup_quality_score")),
            _to_float(item.get("subjective_conviction_score")),
        ]
        subjective_clean = [float(value) for value in subjective_values if value is not None]
        components["llm_subjective_score"] = round(_clip(float(np.nanmean(subjective_clean)) if subjective_clean else 0.0), 4)
        red_flags = item.get("red_flags") if isinstance(item.get("red_flags"), list) else []
        components["llm_red_flag_penalty"] = round(_clip(len(red_flags) / 4.0), 4)
        row["llm_subjective_ranker"] = {
            "status": "applied",
            "confidence": _round(_to_float(item.get("confidence"))),
            "setup_quality_comment": item.get("setup_quality_comment"),
            "valuation_comment": item.get("valuation_comment"),
            "catalyst_summary": item.get("catalyst_summary"),
            "risk_comment": item.get("risk_comment"),
            "portfolio_diversification_comment": item.get("portfolio_diversification_comment"),
            "red_flags": red_flags,
            "positive_evidence": item.get("positive_evidence") if isinstance(item.get("positive_evidence"), list) else [],
            "web_sources_used": item.get("web_sources_used") if isinstance(item.get("web_sources_used"), list) else [],
        }
        row["ranking_policy"]["llm_components"] = [
            "valuation_sanity_score",
            "catalyst_news_score",
            "setup_quality_score",
            "llm_subjective_score",
            "llm_red_flag_penalty",
        ]
        row["ranking_score"] = _weighted_ranking_score(components)


def _bounded_llm_component(base_value: Any, llm_value: Any, *, max_delta: float) -> float:
    base = _clip(float(base_value or 0.0))
    parsed = _to_float(llm_value)
    if parsed is None:
        return round(base, 4)
    target = _clip(parsed)
    delta = max(-max_delta, min(max_delta, target - base))
    return round(_clip(base + delta), 4)


def _discover_stockanalysis_analyst_records(config: ScoutConfig) -> list[DiscoveryRecord]:
    analysts = scrape_visible_analysts(timeout=config.analyst_timeout_seconds)[: max(0, config.analyst_pages)]
    records: list[DiscoveryRecord] = []
    for analyst in analysts:
        try:
            candidates = scrape_visible_analyst_tickers(analyst, timeout=config.analyst_timeout_seconds)
        except Exception:
            continue
        for candidate in candidates:
            records.append(_analyst_candidate_record(candidate))
    return records


def _discover_stockanalysis_market_pages(config: ScoutConfig) -> list[DiscoveryRecord]:
    pages = {
        "gainers": "/markets/gainers/",
        "losers": "/markets/losers/",
        "active": "/markets/active/",
        "biggest_companies": "/list/biggest-companies/",
    }
    records: list[DiscoveryRecord] = []
    for label, path in pages.items():
        url = f"{STOCKANALYSIS_BASE_URL}{path}"
        html = _get_html(url, timeout=config.analyst_timeout_seconds)
        records.extend(_stockanalysis_market_records_from_html(label, url, html))
    return records


def _stockanalysis_market_records_from_html(label: str, url: str, html: str) -> list[DiscoveryRecord]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return []
    header = [cell.get_text(" ", strip=True).lower() for cell in table.find_all("tr")[0].find_all(["th", "td"])]
    records: list[DiscoveryRecord] = []
    for tr in table.find_all("tr")[1:80]:
        cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["td", "th"])]
        link = tr.find("a", href=True)
        if len(cells) < 3 or link is None:
            continue
        ticker = _normalize_ticker(link.get_text(" ", strip=True))
        if not ticker:
            continue
        row = {header[index] if index < len(header) else f"column_{index}": value for index, value in enumerate(cells)}
        records.append(
            DiscoveryRecord(
                ticker=ticker,
                source=f"stockanalysis_market_{label}",
                reason=f"StockAnalysis market page: {label}",
                metadata={
                    "source_url": url,
                    "company_name": row.get("company name"),
                    "rank": _to_float(row.get("no.")),
                    "price": _to_float(row.get("stock price")),
                    "change_pct": _to_float(row.get("% change")),
                    "volume": _parse_compact_number(row.get("volume")),
                    "market_cap": _parse_compact_number(row.get("market cap")),
                    "revenue": _parse_compact_number(row.get("revenue")),
                    "raw_row": row,
                },
            )
        )
    return records


def _analyst_candidate_record(candidate: AnalystTickerCandidate) -> DiscoveryRecord:
    ticker = _normalize_ticker(candidate.forecast_symbol)
    rating_text = " ".join(part for part in [candidate.rating_action, candidate.rating] if part).strip()
    return DiscoveryRecord(
        ticker=ticker,
        source="stockanalysis_visible_analyst_rating",
        reason=f"visible analyst rating: {rating_text or 'rating'} updated {candidate.updated}",
        metadata={
            "source_symbol": candidate.source_symbol,
            "forecast_symbol": candidate.forecast_symbol,
            "company_name": candidate.company_name,
            "rating_action": candidate.rating_action,
            "rating": candidate.rating,
            "price_target": candidate.price_target,
            "current_price": candidate.current_price,
            "upside": candidate.upside,
            "updated": candidate.updated,
            "source_url": candidate.source_url,
            "analyst": {
                "rank": candidate.analyst.rank,
                "name": candidate.analyst.name,
                "company": candidate.analyst.company,
                "sector": candidate.analyst.sector,
                "success_rate": candidate.analyst.success_rate,
                "average_return": candidate.analyst.average_return,
                "ratings": candidate.analyst.ratings,
                "last_rating": candidate.analyst.last_rating,
                "url": candidate.analyst.url,
            },
        },
    )


def _discover_fmp_market_movers(config: ScoutConfig) -> list[DiscoveryRecord]:
    api_key = _fmp_api_key()
    records: list[DiscoveryRecord] = []
    endpoints = {
        "gainers": "stock_market/gainers",
        "losers": "stock_market/losers",
        "actives": "stock_market/actives",
    }
    for label, path in endpoints.items():
        payload = _get_json(f"{FMP_BASE_URL}/{path}?" + urlencode({"apikey": api_key}))
        rows = payload if isinstance(payload, list) else []
        for row in rows[:80]:
            ticker = _normalize_ticker(row.get("symbol"))
            if not ticker:
                continue
            records.append(
                DiscoveryRecord(
                    ticker=ticker,
                    source=f"fmp_market_{label}",
                    reason=f"FMP market mover list: {label}",
                    metadata={
                        "symbol": ticker,
                        "name": row.get("name"),
                        "price": _to_float(row.get("price")),
                        "change": _to_float(row.get("change")),
                        "changes_percentage": _to_float(row.get("changesPercentage")),
                    },
                )
            )
    return records


def _discover_fmp_news_activity(config: ScoutConfig) -> list[DiscoveryRecord]:
    api_key = _fmp_api_key()
    payload = _get_json(f"{FMP_BASE_URL}/stock_news?" + urlencode({"limit": 200, "apikey": api_key}))
    records: list[DiscoveryRecord] = []
    seen: set[tuple[str, str]] = set()
    rows = payload if isinstance(payload, list) else []
    for row in rows:
        ticker = _normalize_ticker(row.get("symbol"))
        title = str(row.get("title") or "")
        if not ticker or not title:
            continue
        key = (ticker, title)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            DiscoveryRecord(
                ticker=ticker,
                source="fmp_recent_news",
                reason="recent news activity",
                metadata={
                    "title": title,
                    "published_date": row.get("publishedDate"),
                    "site": row.get("site"),
                    "url": row.get("url"),
                },
            )
        )
    return records


def _discover_fmp_earnings_activity(config: ScoutConfig) -> list[DiscoveryRecord]:
    api_key = _fmp_api_key()
    today = datetime.now(UTC).date()
    date_from = (pd.Timestamp(today) - pd.Timedelta(days=7)).date().isoformat()
    date_to = (pd.Timestamp(today) + pd.Timedelta(days=14)).date().isoformat()
    payload = _get_json(f"{FMP_BASE_URL}/earning_calendar?" + urlencode({"from": date_from, "to": date_to, "apikey": api_key}))
    records: list[DiscoveryRecord] = []
    rows = payload if isinstance(payload, list) else []
    for row in rows[:300]:
        ticker = _normalize_ticker(row.get("symbol"))
        if not ticker:
            continue
        records.append(
            DiscoveryRecord(
                ticker=ticker,
                source="fmp_earnings_calendar",
                reason=f"earnings activity {row.get('date')}",
                metadata={
                    "date": row.get("date"),
                    "eps": _to_float(row.get("eps")),
                    "eps_estimated": _to_float(row.get("epsEstimated")),
                    "revenue": _to_float(row.get("revenue")),
                    "revenue_estimated": _to_float(row.get("revenueEstimated")),
                    "time": row.get("time"),
                },
            )
        )
    return records


def _score_one_ticker(
    ticker: str,
    frame: pd.DataFrame,
    records: list[DiscoveryRecord],
    config: ScoutConfig,
) -> dict[str, Any]:
    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    high = pd.to_numeric(frame.get("high", close), errors="coerce").reindex(close.index).ffill()
    low = pd.to_numeric(frame.get("low", close), errors="coerce").reindex(close.index).ffill()
    volume = pd.to_numeric(frame.get("volume", pd.Series(index=close.index, dtype=float)), errors="coerce").reindex(close.index)
    latest = float(close.iloc[-1])
    returns = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    avg_dollar_volume_20d = float((close * volume).rolling(20).mean().iloc[-1]) if volume.notna().any() else 0.0
    volume_ratio = _safe_ratio(float(volume.rolling(5).mean().iloc[-1]), float(volume.rolling(60).mean().iloc[-1])) if volume.notna().sum() >= 60 else 1.0
    ret_1d = _log_return(close, 1)
    ret_5d = _log_return(close, 5)
    ret_21d = _log_return(close, 21)
    ret_63d = _log_return(close, 63)
    realized_vol_20d = float(returns.rolling(20).std().iloc[-1] * math.sqrt(252)) if returns.notna().sum() >= 20 else 0.0
    atr_pct_20d = _atr_pct(high, low, close, 20)
    high_20 = float(close.rolling(20).max().iloc[-1])
    high_55 = float(close.rolling(55).max().iloc[-1])
    low_20 = float(close.rolling(20).min().iloc[-1])
    sma_20 = float(close.rolling(20).mean().iloc[-1])
    sma_50 = float(close.rolling(50).mean().iloc[-1])
    sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.rolling(min(120, len(close))).mean().iloc[-1])
    range_252_high = float(close.rolling(min(252, len(close))).max().iloc[-1])
    range_252_low = float(close.rolling(min(252, len(close))).min().iloc[-1])
    range_position = _safe_ratio(latest - range_252_low, range_252_high - range_252_low)

    analyst_score = _analyst_score(records)
    source_counts = _source_counts(records)
    news_score = min(1.0, source_counts.get("fmp_recent_news", 0) / 4.0)
    earnings_score = 1.0 if source_counts.get("fmp_earnings_calendar", 0) else 0.0
    unusual_volume_score = _clip((volume_ratio - 1.0) / 2.0)
    liquidity_score = _liquidity_score(avg_dollar_volume_20d, config.min_avg_dollar_volume)
    momentum_score = _momentum_score(ret_5d, ret_21d, ret_63d)
    breakout_score = _breakout_score(latest, high_20, high_55, ret_21d, volume_ratio)
    pullback_score = _pullback_score(latest, sma_20, sma_50, sma_200, low_20, ret_63d)
    volatility_score = _volatility_fit_score(realized_vol_20d, atr_pct_20d)
    market_mover_score = min(
        1.0,
        (
            source_counts.get("fmp_market_gainers", 0)
            + source_counts.get("fmp_market_actives", 0)
            + source_counts.get("stockanalysis_market_gainers", 0)
            + source_counts.get("stockanalysis_market_active", 0)
            + 0.5 * source_counts.get("stockanalysis_market_biggest_companies", 0)
        )
        / 2.0,
    )

    rejection_reasons = []
    eligible = True
    if latest < config.min_price:
        eligible = False
        rejection_reasons.append("price_below_minimum")
    if avg_dollar_volume_20d < config.min_avg_dollar_volume:
        eligible = False
        rejection_reasons.append("liquidity_below_minimum")
    if realized_vol_20d > config.max_realized_volatility_20d:
        eligible = False
        rejection_reasons.append("volatility_above_limit")
    if not records:
        rejection_reasons.append("no_dynamic_discovery_source")

    risk_penalty = 0.0
    if realized_vol_20d > 0.90:
        risk_penalty += 0.08
    if latest < sma_200 and ret_63d < 0:
        risk_penalty += 0.08
    if range_position > 0.92 and breakout_score < 0.65:
        risk_penalty += 0.03

    return {
        "ticker": ticker,
        "as_of_date": str(close.index[-1].date()),
        "latest_close": round(latest, 4),
        "eligible": eligible,
        "selected": False,
        "rejection_reasons": rejection_reasons,
        "reasons": _candidate_reasons(records, breakout_score, pullback_score, unusual_volume_score, momentum_score, earnings_score, news_score),
        "discovery_sources": sorted(source_counts),
        "source_counts": source_counts,
        "sector": _sector_from_records(records),
        "metrics": {
            "return_1d": _round(ret_1d),
            "return_5d": _round(ret_5d),
            "return_21d": _round(ret_21d),
            "return_63d": _round(ret_63d),
            "avg_dollar_volume_20d": round(avg_dollar_volume_20d, 2),
            "volume_ratio_5d_vs_60d": _round(volume_ratio),
            "realized_volatility_20d": _round(realized_vol_20d),
            "atr_pct_20d": _round(atr_pct_20d),
            "range_position_252d": _round(range_position),
            "distance_to_20d_high_pct": _round(_safe_ratio(latest, high_20) - 1.0),
            "distance_to_55d_high_pct": _round(_safe_ratio(latest, high_55) - 1.0),
            "distance_to_sma_20_pct": _round(_safe_ratio(latest, sma_20) - 1.0),
            "distance_to_sma_50_pct": _round(_safe_ratio(latest, sma_50) - 1.0),
            "distance_to_sma_200_pct": _round(_safe_ratio(latest, sma_200) - 1.0),
        },
        "score_components": {
            "liquidity": liquidity_score,
            "momentum": momentum_score,
            "breakout": breakout_score,
            "pullback": pullback_score,
            "analyst_activity": analyst_score,
            "unusual_volume": unusual_volume_score,
            "news_activity": news_score,
            "earnings_activity": earnings_score,
            "market_mover": market_mover_score,
            "volatility_fit": volatility_score,
            "sector_strength": 0.5,
        },
        "risk_penalty": round(risk_penalty, 4),
        "latest_records": [_record_to_dict(record) for record in records[:8]],
    }


def _missing_price_row(ticker: str, records: list[DiscoveryRecord], config: ScoutConfig) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "as_of_date": None,
        "latest_close": None,
        "eligible": False,
        "selected": False,
        "score": 0.0,
        "rejection_reasons": ["missing_price_data"],
        "reasons": _candidate_reasons(records, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "discovery_sources": sorted(_source_counts(records)),
        "source_counts": _source_counts(records),
        "sector": _sector_from_records(records),
        "metrics": {},
        "score_components": {
            "liquidity": 0.0,
            "momentum": 0.0,
            "breakout": 0.0,
            "pullback": 0.0,
            "analyst_activity": _analyst_score(records),
            "unusual_volume": 0.0,
            "news_activity": 0.0,
            "earnings_activity": 0.0,
            "market_mover": 0.0,
            "volatility_fit": 0.0,
            "sector_strength": 0.5,
        },
        "risk_penalty": 0.0,
        "latest_records": [_record_to_dict(record) for record in records[:8]],
    }


def _add_sector_strength(rows: list[dict[str, Any]]) -> None:
    by_sector: dict[str, list[float]] = {}
    for row in rows:
        sector = row.get("sector") or "Unknown"
        momentum = row.get("metrics", {}).get("return_21d")
        if momentum is not None:
            by_sector.setdefault(str(sector), []).append(float(momentum))
    sector_mean = {sector: float(np.nanmean(values)) for sector, values in by_sector.items() if values}
    if not sector_mean:
        return
    values = sorted(sector_mean.values())
    for row in rows:
        sector = str(row.get("sector") or "Unknown")
        value = sector_mean.get(sector)
        if value is None:
            row["sector_strength_score"] = 0.5
        else:
            row["sector_strength_score"] = _percentile_rank(values, value)


def _weighted_score(components: dict[str, float]) -> float:
    weights = {
        "liquidity": 0.15,
        "momentum": 0.14,
        "breakout": 0.12,
        "pullback": 0.10,
        "analyst_activity": 0.14,
        "unusual_volume": 0.10,
        "news_activity": 0.08,
        "earnings_activity": 0.06,
        "market_mover": 0.06,
        "volatility_fit": 0.07,
        "sector_strength": 0.08,
    }
    return float(sum(weights[key] * float(components.get(key, 0.0)) for key in weights))


def _candidate_reasons(
    records: list[DiscoveryRecord],
    breakout_score: float,
    pullback_score: float,
    unusual_volume_score: float,
    momentum_score: float,
    earnings_score: float,
    news_score: float,
) -> list[str]:
    reasons = []
    if records:
        source_counts = _source_counts(records)
        top_sources = ", ".join(f"{source}:{count}" for source, count in sorted(source_counts.items())[:4])
        reasons.append(f"dynamic discovery sources: {top_sources}")
    if breakout_score >= 0.65:
        reasons.append("breakout pressure near recent highs")
    if pullback_score >= 0.65:
        reasons.append("constructive pullback into moving-average support")
    if unusual_volume_score >= 0.50:
        reasons.append("unusual volume expansion")
    if momentum_score >= 0.65:
        reasons.append("positive multi-week momentum")
    if earnings_score > 0:
        reasons.append("recent/upcoming earnings activity")
    if news_score >= 0.25:
        reasons.append("recent news activity")
    return reasons[:8]


def _analyst_score(records: list[DiscoveryRecord]) -> float:
    scores = []
    for record in records:
        if record.source != "stockanalysis_visible_analyst_rating":
            continue
        analyst = record.metadata.get("analyst", {}) if isinstance(record.metadata.get("analyst"), dict) else {}
        rank = _to_float(analyst.get("rank")) or 999.0
        success = _to_float(analyst.get("success_rate")) or 50.0
        avg_return = _to_float(analyst.get("average_return")) or 0.0
        rating = str(record.metadata.get("rating") or "").lower()
        rating_bias = 0.15 if "buy" in rating or "outperform" in rating else -0.05 if "sell" in rating or "underperform" in rating else 0.05
        rank_score = _clip(1.0 - (rank - 1.0) / 150.0)
        success_score = _clip((success - 45.0) / 35.0)
        return_score = _clip((avg_return + 5.0) / 30.0)
        scores.append(_clip(0.45 * rank_score + 0.30 * success_score + 0.20 * return_score + rating_bias))
    return round(float(max(scores) if scores else 0.0), 4)


def _liquidity_score(avg_dollar_volume: float, minimum: float) -> float:
    if avg_dollar_volume <= 0:
        return 0.0
    return round(_clip(math.log10(max(avg_dollar_volume, 1.0) / max(minimum, 1.0)) / 2.0 + 0.5), 4)


def _momentum_score(ret_5d: float, ret_21d: float, ret_63d: float) -> float:
    raw = 0.25 * np.tanh(ret_5d * 12.0) + 0.45 * np.tanh(ret_21d * 5.0) + 0.30 * np.tanh(ret_63d * 2.5)
    return round(_clip(0.5 + 0.5 * float(raw)), 4)


def _breakout_score(latest: float, high_20: float, high_55: float, ret_21d: float, volume_ratio: float) -> float:
    near_20 = _clip(1.0 - abs(_safe_ratio(latest, high_20) - 1.0) / 0.04)
    near_55 = _clip(1.0 - abs(_safe_ratio(latest, high_55) - 1.0) / 0.07)
    trend = _clip(0.5 + np.tanh(ret_21d * 6.0) / 2.0)
    volume = _clip((volume_ratio - 0.8) / 1.7)
    return round(float(0.35 * near_20 + 0.25 * near_55 + 0.25 * trend + 0.15 * volume), 4)


def _pullback_score(latest: float, sma_20: float, sma_50: float, sma_200: float, low_20: float, ret_63d: float) -> float:
    trend_ok = 1.0 if latest > sma_200 and ret_63d > 0 else 0.35 if latest > sma_200 else 0.0
    near_20 = _clip(1.0 - abs(_safe_ratio(latest, sma_20) - 1.0) / 0.035)
    near_50 = _clip(1.0 - abs(_safe_ratio(latest, sma_50) - 1.0) / 0.06)
    above_low = _clip((_safe_ratio(latest, low_20) - 1.0) / 0.08)
    return round(float(0.35 * trend_ok + 0.25 * near_20 + 0.25 * near_50 + 0.15 * above_low), 4)


def _volatility_fit_score(realized_vol_20d: float, atr_pct_20d: float) -> float:
    if realized_vol_20d <= 0:
        return 0.3
    vol_band = _band_score(realized_vol_20d, 0.12, 0.22, 0.55, 0.95)
    atr_band = _band_score(atr_pct_20d, 0.008, 0.015, 0.045, 0.075)
    return round(float(0.65 * vol_band + 0.35 * atr_band), 4)


def _band_score(value: float, low_bad: float, low_good: float, high_good: float, high_bad: float) -> float:
    if value <= low_bad or value >= high_bad:
        return 0.0
    if low_good <= value <= high_good:
        return 1.0
    if value < low_good:
        return _clip((value - low_bad) / (low_good - low_bad))
    return _clip((high_bad - value) / (high_bad - high_good))


def _atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> float:
    prev_close = close.shift(1)
    true_range = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = float(true_range.rolling(window).mean().iloc[-1])
    latest = float(close.iloc[-1])
    return _safe_ratio(atr, latest)


def _log_return(close: pd.Series, periods: int) -> float:
    if len(close) <= periods:
        return 0.0
    value = float(np.log(close.iloc[-1] / close.iloc[-periods - 1]))
    return value if np.isfinite(value) else 0.0


def _source_counts(records: list[DiscoveryRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.source] = counts.get(record.source, 0) + 1
    return counts


def _sector_from_records(records: list[DiscoveryRecord]) -> str:
    for record in records:
        analyst = record.metadata.get("analyst", {}) if isinstance(record.metadata.get("analyst"), dict) else {}
        sector = analyst.get("sector")
        if sector:
            return str(sector)
    return "Unknown"


def _build_stratified_universe(records: list[DiscoveryRecord], max_tickers: int) -> list[str]:
    """Build a source-balanced ticker list so one discovery feed cannot crowd out others."""

    limit = max(0, int(max_tickers))
    if limit <= 0:
        return []
    source_order = [
        "stockanalysis_visible_analyst_rating",
        "stockanalysis_market_gainers",
        "stockanalysis_market_active",
        "stockanalysis_market_biggest_companies",
        "stockanalysis_market_losers",
        "fmp_market_gainers",
        "fmp_market_actives",
        "fmp_recent_news",
        "fmp_earnings_calendar",
        "fmp_market_losers",
    ]
    by_source: dict[str, list[str]] = {}
    for record in records:
        by_source.setdefault(record.source, [])
        if record.ticker not in by_source[record.source]:
            by_source[record.source].append(record.ticker)
    ordered_sources = [source for source in source_order if source in by_source]
    ordered_sources.extend(sorted(source for source in by_source if source not in set(ordered_sources)))

    selected: list[str] = []
    seen: set[str] = set()
    cursor = 0
    while len(selected) < limit:
        added = False
        for source in ordered_sources:
            tickers = by_source.get(source, [])
            if cursor >= len(tickers):
                continue
            ticker = tickers[cursor]
            if ticker not in seen:
                selected.append(ticker)
                seen.add(ticker)
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        cursor += 1
    return selected


def _record_to_dict(record: DiscoveryRecord) -> dict[str, Any]:
    return {"ticker": record.ticker, "source": record.source, "reason": record.reason, "metadata": record.metadata}


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return str(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    flattened = []
    for row in rows:
        flattened.append(
            {
                "rank": row.get("rank"),
                "ticker": row.get("ticker"),
                "score": row.get("score"),
                "ranking_score": row.get("ranking_score"),
                "selected": row.get("selected"),
                "eligible": row.get("eligible"),
                "latest_close": row.get("latest_close"),
                "sector": row.get("sector"),
                "reasons": " | ".join(row.get("reasons", [])),
                "rejection_reasons": " | ".join(row.get("rejection_reasons", [])),
                "discovery_sources": ",".join(row.get("discovery_sources", [])),
                "avg_dollar_volume_20d": row.get("metrics", {}).get("avg_dollar_volume_20d"),
                "return_21d": row.get("metrics", {}).get("return_21d"),
                "return_63d": row.get("metrics", {}).get("return_63d"),
                "realized_volatility_20d": row.get("metrics", {}).get("realized_volatility_20d"),
                "liquidity_score": row.get("ranking_components", {}).get("liquidity_score"),
                "trend_score": row.get("ranking_components", {}).get("trend_score"),
                "setup_quality_score": row.get("ranking_components", {}).get("setup_quality_score"),
                "valuation_sanity_score": row.get("ranking_components", {}).get("valuation_sanity_score"),
                "catalyst_news_score": row.get("ranking_components", {}).get("catalyst_news_score"),
                "risk_score": row.get("ranking_components", {}).get("risk_score"),
                "portfolio_diversification_score": row.get("ranking_components", {}).get("portfolio_diversification_score"),
                "llm_subjective_score": row.get("ranking_components", {}).get("llm_subjective_score"),
                "llm_red_flag_penalty": row.get("ranking_components", {}).get("llm_red_flag_penalty"),
            }
        )
    pd.DataFrame(flattened).to_csv(path, index=False)


def _write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    selected = summary.get("selected_candidates", [])
    lines = [
        "# Virtual Trader Scout Summary",
        "",
        f"Generated: {summary.get('generated_at_utc')}",
        "",
        "## Policy",
        "",
        "- Forecast-free discovery only.",
        f"- LLM subjective ranking: {summary.get('ranking_audit', {}).get('llm_ranking', {}).get('status', 'not_requested')}.",
        "- No execution is allowed from this layer.",
        "- Selected tickers must still run the full forecast and CEO decision pipeline.",
        "",
        "## Counts",
        "",
    ]
    for key, value in summary.get("counts", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Selected Candidates", ""])
    if not selected:
        lines.append("No candidates selected.")
    else:
        for row in selected:
            lines.append(
                f"- {row.get('rank')}. {row.get('ticker')} "
                f"ranking_score={row.get('ranking_score')} scout_score={row.get('score')} "
                f"price={row.get('latest_close')} reasons={'; '.join(row.get('reasons', [])[:3])}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _get_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 market-forecasting-engine/virtual-trader"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_html(url: str, *, timeout: float) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 market-forecasting-engine/virtual-trader"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def _fmp_api_key() -> str:
    key = os.getenv("FMP_API_KEY") or os.getenv("FINANCIAL_MODELING_PREP_API_KEY")
    if not key:
        raise RuntimeError("FMP_API_KEY is required for FMP discovery sources.")
    return key


def _load_env(path: str | Path | None) -> None:
    candidates = [Path(path).expanduser()] if path else [Path.cwd() / ".env", *[parent / ".env" for parent in Path.cwd().parents[:4]]]
    for candidate in candidates:
        if not candidate.exists():
            continue
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return


def _normalize_ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text or text in {"N/A", "NA", "NONE", "NULL"}:
        return ""
    return text.replace("/", "-")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("$", "").replace(",", "").replace("%", "")
    if not text or text in {"-", "N/A", "nan"}:
        return None
    try:
        number = float(text)
    except Exception:
        return None
    return number if np.isfinite(number) else None


def _parse_compact_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("$", "").replace(",", "")
    if not text or text in {"-", "N/A"}:
        return None
    multiplier = 1.0
    suffix = text[-1:].upper()
    if suffix in {"K", "M", "B", "T"}:
        text = text[:-1]
        multiplier = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[suffix]
    try:
        number = float(text) * multiplier
    except Exception:
        return None
    return number if np.isfinite(number) else None


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not denominator or not np.isfinite(denominator):
        return 0.0
    value = numerator / denominator
    return float(value) if np.isfinite(value) else 0.0


def _clip(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def _percentile_rank(sorted_values: list[float], value: float) -> float:
    if not sorted_values:
        return 0.5
    less_equal = sum(1 for item in sorted_values if item <= value)
    return round(less_equal / len(sorted_values), 4)


def _safe_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _progress(config: ScoutConfig, message: str) -> None:
    if config.progress:
        print(f"[scout] {datetime.now().strftime('%H:%M:%S')} {message}", flush=True)
