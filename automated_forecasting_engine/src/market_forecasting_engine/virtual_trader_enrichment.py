from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from market_forecasting_engine.llm_trader.source_synthesis import (
    attach_long_term_source_synthesis,
    run_long_term_source_synthesis,
)
from market_forecasting_engine.long_term_sources import (
    DEFAULT_LONG_TERM_SOURCE_PROVIDERS,
    LongTermSourceRequest,
    append_long_term_source_snapshot,
    collect_long_term_source_context,
    parse_long_term_source_providers,
)
from market_forecasting_engine.openai_models import DEFAULT_OPENAI_MODEL, DEFAULT_REASONING_EFFORT
from market_forecasting_engine.strategy_knowledge import (
    DEFAULT_STRATEGY_CORPUS_DIR,
    DEFAULT_STRATEGY_INDEX_PATH,
    StrategyKnowledgeRequest,
    attach_strategy_knowledge_context,
    build_strategy_knowledge_context,
)


@dataclass(frozen=True)
class VirtualTraderEnrichmentConfig:
    selected_candidates_path: str | Path
    output_dir: str | Path
    max_candidates: int = 3
    providers: tuple[str, ...] = DEFAULT_LONG_TERM_SOURCE_PROVIDERS
    env_file: str | Path | None = None
    max_news_items: int = 12
    enable_source_synthesis: bool = True
    llm_provider: str = "openai"
    llm_model: str | None = None
    llm_reasoning_effort: str = DEFAULT_REASONING_EFFORT
    llm_timeout_seconds: float = 90.0
    source_synthesis_dry_run: bool = False
    enable_strategy_knowledge: bool = True
    strategy_knowledge_corpus_dir: str | Path = DEFAULT_STRATEGY_CORPUS_DIR
    strategy_knowledge_index: str | Path = DEFAULT_STRATEGY_INDEX_PATH
    strategy_knowledge_max_chunks: int = 8
    strategy_knowledge_rebuild_index: bool = False
    snapshot_dir: str | Path | None = None
    progress: bool = True


def run_virtual_trader_enrichment(config: VirtualTraderEnrichmentConfig) -> dict[str, Any]:
    """Build deep evidence packets for ranker-selected tickers without running forecasts."""

    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_candidates = load_selected_candidates(config.selected_candidates_path, max_candidates=config.max_candidates)
    started_at = datetime.now(UTC)
    _progress(config, f"virtual trader enrichment started candidates={len(selected_candidates)} output={output_dir}")

    ticker_results = []
    for position, candidate in enumerate(selected_candidates, start=1):
        ticker = str(candidate.get("ticker") or "").upper()
        if not ticker:
            continue
        _progress(config, f"enriching {position}/{len(selected_candidates)} {ticker}")
        ticker_results.append(_enrich_one_candidate(candidate, config, output_dir))

    board = {
        "run_type": "virtual_trader_evidence_enrichment_layer",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "config": _config_for_report(config),
        "policy": {
            "forecast_free": True,
            "execution_allowed": False,
            "role": (
                "Deep evidence packets for top-ranked virtual trader candidates. "
                "The normal forecast engine remains independent and can run without this layer."
            ),
            "next_stage": "Run full forecast suite and autonomous CEO decision for enriched candidates.",
        },
        "counts": {
            "selected_candidates_loaded": len(selected_candidates),
            "enriched_candidates": len(ticker_results),
            "source_synthesis_executed": sum(1 for row in ticker_results if row.get("source_synthesis", {}).get("status") == "executed"),
            "strategy_context_executed": sum(1 for row in ticker_results if row.get("strategy_knowledge_context", {}).get("status") == "executed"),
        },
        "candidates": ticker_results,
        "artifact_paths": {},
    }
    board["artifact_paths"] = {
        "enrichment_board": _write_json(output_dir / "enrichment_board.json", board),
        "enrichment_board_markdown": str(output_dir / "enrichment_board.md"),
    }
    _write_markdown_board(output_dir / "enrichment_board.md", board)
    _write_json(output_dir / "enrichment_board.json", board)
    _progress(config, f"enrichment complete tickers={','.join(row.get('ticker', '') for row in ticker_results) or 'none'}")
    return board


def load_selected_candidates(path: str | Path, *, max_candidates: int) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("selected_candidates") or payload.get("candidates") or payload.get("top_candidates") or []
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError(f"selected candidates file must contain a list or candidate container: {path}")
    candidates = [row for row in rows if isinstance(row, dict) and row.get("ticker")]
    candidates = sorted(candidates, key=lambda row: int(row.get("rank") or 999999))
    return candidates[: max(0, int(max_candidates))]


def _enrich_one_candidate(
    candidate: dict[str, Any],
    config: VirtualTraderEnrichmentConfig,
    output_dir: Path,
) -> dict[str, Any]:
    ticker = str(candidate.get("ticker") or "").upper()
    ticker_dir = output_dir / "tickers" / _safe_symbol(ticker)
    source_dir = ticker_dir / "long_term_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_context = collect_long_term_source_context(
        LongTermSourceRequest(
            ticker=ticker,
            providers=config.providers,
            env_file=str(config.env_file) if config.env_file else None,
            output_dir=source_dir,
            max_news_items=int(config.max_news_items),
        )
    )
    snapshot_path = None
    try:
        snapshot_path = append_long_term_source_snapshot(
            source_context,
            Path(config.snapshot_dir).expanduser() if config.snapshot_dir else output_dir / "data" / "long_term_source_snapshots",
            ticker=ticker,
        )
    except Exception as exc:
        source_context.setdefault("snapshot_error", _safe_error(exc))

    report = _minimal_report_for_candidate(candidate, source_context)
    source_synthesis = {
        "status": "skipped",
        "reason": "source synthesis disabled for this enrichment run.",
    }
    if config.enable_source_synthesis:
        source_synthesis = run_long_term_source_synthesis(
            report=report,
            llm_provider=config.llm_provider,
            llm_model=config.llm_model or DEFAULT_OPENAI_MODEL,
            reasoning_effort=config.llm_reasoning_effort,
            llm_env_file=str(config.env_file) if config.env_file else None,
            timeout_seconds=float(config.llm_timeout_seconds),
            dry_run=bool(config.source_synthesis_dry_run),
        )
        attach_long_term_source_synthesis(report, source_synthesis)

    strategy_context = {
        "status": "skipped",
        "reason": "strategy knowledge disabled for this enrichment run.",
    }
    if config.enable_strategy_knowledge:
        strategy_context = build_strategy_knowledge_context(
            report,
            StrategyKnowledgeRequest(
                ticker=ticker,
                corpus_dir=config.strategy_knowledge_corpus_dir,
                index_path=config.strategy_knowledge_index,
                llm_env_file=str(config.env_file) if config.env_file else None,
                max_chunks=int(config.strategy_knowledge_max_chunks),
                rebuild_index=bool(config.strategy_knowledge_rebuild_index),
                timeout_seconds=int(config.llm_timeout_seconds),
            ),
        )
        attach_strategy_knowledge_context(report, strategy_context)

    ticker_packet = {
        "ticker": ticker,
        "candidate": candidate,
        "long_term_source_context": source_context,
        "source_synthesis": source_synthesis,
        "strategy_knowledge_context": strategy_context,
        "forecast_engine_integration": {
            "independent_forecast_engine": True,
            "recommended_next_input": "Pass this packet beside the normal forecast report; do not require the forecast engine to depend on virtual trader state.",
            "ticker_artifact": str(ticker_dir / "evidence_packet.json"),
        },
        "artifacts": {
            "ticker_dir": str(ticker_dir),
            "long_term_sources_dir": str(source_dir),
            "source_context": str(ticker_dir / "long_term_source_context.json"),
            "source_synthesis": str(ticker_dir / "source_synthesis.json"),
            "strategy_knowledge": str(ticker_dir / "strategy_knowledge_context.json"),
            "evidence_packet": str(ticker_dir / "evidence_packet.json"),
            "snapshot_path": str(snapshot_path) if snapshot_path else None,
        },
    }
    _write_json(ticker_dir / "long_term_source_context.json", source_context)
    _write_json(ticker_dir / "source_synthesis.json", source_synthesis)
    _write_json(ticker_dir / "strategy_knowledge_context.json", strategy_context)
    _write_json(ticker_dir / "evidence_packet.json", ticker_packet)
    return _candidate_board_row(ticker_packet)


def _minimal_report_for_candidate(candidate: dict[str, Any], source_context: dict[str, Any]) -> dict[str, Any]:
    ticker = str(candidate.get("ticker") or "").upper()
    return {
        "ticker": ticker,
        "suggested_action": "enrichment_only",
        "risk_level": "unknown_until_forecast",
        "current_price": candidate.get("latest_close"),
        "forecasts": [],
        "technical_view": {
            "long_term_source_context": source_context,
            "virtual_trader_candidate": candidate,
            "trend_state": {
                "trend_score": candidate.get("ranking_components", {}).get("trend_score"),
                "setup_quality_score": candidate.get("ranking_components", {}).get("setup_quality_score"),
                "risk_score": candidate.get("ranking_components", {}).get("risk_score"),
            },
        },
        "decision_view": {
            "long_term_context": source_context,
            "virtual_trader_candidate": candidate,
        },
        "final_decision_reasoning": {
            "stage": "virtual_trader_evidence_enrichment_only",
            "final_ceo_decision_made": False,
        },
    }


def _candidate_board_row(packet: dict[str, Any]) -> dict[str, Any]:
    source_context = packet.get("long_term_source_context", {})
    provider_summaries = source_context.get("provider_summaries", {}) if isinstance(source_context.get("provider_summaries"), dict) else {}
    provider_status = {
        provider: summary.get("status")
        for provider, summary in provider_summaries.items()
        if isinstance(summary, dict)
    }
    synthesis = packet.get("source_synthesis", {})
    strategy = packet.get("strategy_knowledge_context", {})
    candidate = packet.get("candidate", {})
    return {
        "ticker": packet.get("ticker"),
        "rank": candidate.get("rank"),
        "ranking_score": candidate.get("ranking_score"),
        "scout_score": candidate.get("score"),
        "provider_status": provider_status,
        "source_status": source_context.get("status"),
        "source_synthesis": {
            "status": synthesis.get("status"),
            "model": synthesis.get("model"),
            "coverage_all_passed": synthesis.get("llm_evidence_manifest", {}).get("all_scraped_sections_passed")
            if isinstance(synthesis.get("llm_evidence_manifest"), dict)
            else None,
            "synthesis": synthesis.get("synthesis"),
            "reason": synthesis.get("reason"),
        },
        "strategy_knowledge_context": {
            "status": strategy.get("status"),
            "retrieved_chunks": len(strategy.get("retrieved_chunks", [])) if isinstance(strategy.get("retrieved_chunks"), list) else 0,
            "synthesis": strategy.get("synthesis"),
            "reason": strategy.get("reason"),
        },
        "artifacts": packet.get("artifacts", {}),
    }


def _write_markdown_board(path: Path, board: dict[str, Any]) -> None:
    lines = [
        "# Virtual Trader Evidence Enrichment",
        "",
        f"Generated: {board.get('generated_at_utc')}",
        "",
        "## Policy",
        "",
        "- Forecast-free evidence enrichment only.",
        "- No orders or final Buy/Hold/Sell decisions are produced here.",
        "- Normal forecast engine remains independent and can run without this layer.",
        "",
        "## Candidates",
        "",
    ]
    for row in board.get("candidates", []):
        providers = ", ".join(f"{key}:{value}" for key, value in sorted(row.get("provider_status", {}).items()))
        lines.append(
            f"- {row.get('rank')}. {row.get('ticker')} ranking_score={row.get('ranking_score')} "
            f"sources={row.get('source_status')} synthesis={row.get('source_synthesis', {}).get('status')} "
            f"strategy={row.get('strategy_knowledge_context', {}).get('status')} providers=({providers})"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _config_for_report(config: VirtualTraderEnrichmentConfig) -> dict[str, Any]:
    data = asdict(config)
    for key in ("selected_candidates_path", "output_dir", "env_file", "strategy_knowledge_corpus_dir", "strategy_knowledge_index", "snapshot_dir"):
        if data.get(key) is not None:
            data[key] = str(data[key])
    data["llm_model"] = data.get("llm_model") or DEFAULT_OPENAI_MODEL
    data["providers"] = list(config.providers)
    return data


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return str(path)


def _safe_symbol(ticker: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in ticker).strip("_") or "unknown"


def _safe_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _progress(config: VirtualTraderEnrichmentConfig, message: str) -> None:
    if config.progress:
        print(f"[virtual-enrich] {datetime.now().strftime('%H:%M:%S')} {message}", flush=True)

