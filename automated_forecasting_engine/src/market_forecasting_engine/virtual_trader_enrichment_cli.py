from __future__ import annotations

import argparse
import json
from pathlib import Path

from market_forecasting_engine.long_term_sources import DEFAULT_LONG_TERM_SOURCE_PROVIDERS, parse_long_term_source_providers
from market_forecasting_engine.virtual_trader_enrichment import (
    VirtualTraderEnrichmentConfig,
    run_virtual_trader_enrichment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run virtual trader evidence enrichment for scout-selected tickers.")
    parser.add_argument("--selected-candidates", required=True, help="Path to selected_candidates.json from virtual_trader_scout.")
    parser.add_argument("--output-dir", required=True, help="Directory where enrichment artifacts are written.")
    parser.add_argument("--max-candidates", type=int, default=3, help="Maximum selected candidates to enrich.")
    parser.add_argument(
        "--providers",
        default=",".join(DEFAULT_LONG_TERM_SOURCE_PROVIDERS),
        help="Comma-separated long-term source providers.",
    )
    parser.add_argument("--env-file", default=None, help="Optional .env file for provider and LLM API keys.")
    parser.add_argument("--max-news-items", type=int, default=12, help="Maximum news items preserved in consolidated source context.")
    parser.add_argument("--disable-source-synthesis", action="store_true", help="Collect sources but skip source-synthesis LLM.")
    parser.add_argument("--source-synthesis-dry-run", action="store_true", help="Build source-synthesis payloads without calling the LLM.")
    parser.add_argument("--llm-provider", default="openai", help="LLM provider for source synthesis.")
    parser.add_argument("--llm-model", default=None, help="Optional source-synthesis model override.")
    parser.add_argument("--llm-reasoning-effort", default="none", help="Reasoning effort for reasoning-capable models.")
    parser.add_argument("--llm-timeout-seconds", type=float, default=90.0, help="LLM and strategy embedding timeout.")
    parser.add_argument("--disable-strategy-knowledge", action="store_true", help="Skip FAISS/book strategy context retrieval.")
    parser.add_argument("--strategy-knowledge-corpus-dir", default="automated_forecasting_engine/strategy_knowledge/corpus")
    parser.add_argument("--strategy-knowledge-index", default="automated_forecasting_engine/strategy_knowledge/indexes/strategy_knowledge.faiss")
    parser.add_argument("--strategy-knowledge-max-chunks", type=int, default=8)
    parser.add_argument("--strategy-knowledge-rebuild-index", action="store_true")
    parser.add_argument("--snapshot-dir", default=None, help="Optional long-term source snapshot directory.")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress output.")
    args = parser.parse_args()

    board = run_virtual_trader_enrichment(
        VirtualTraderEnrichmentConfig(
            selected_candidates_path=Path(args.selected_candidates),
            output_dir=Path(args.output_dir),
            max_candidates=args.max_candidates,
            providers=parse_long_term_source_providers(args.providers),
            env_file=args.env_file,
            max_news_items=args.max_news_items,
            enable_source_synthesis=not args.disable_source_synthesis,
            source_synthesis_dry_run=args.source_synthesis_dry_run,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_reasoning_effort=args.llm_reasoning_effort,
            llm_timeout_seconds=args.llm_timeout_seconds,
            enable_strategy_knowledge=not args.disable_strategy_knowledge,
            strategy_knowledge_corpus_dir=args.strategy_knowledge_corpus_dir,
            strategy_knowledge_index=args.strategy_knowledge_index,
            strategy_knowledge_max_chunks=args.strategy_knowledge_max_chunks,
            strategy_knowledge_rebuild_index=args.strategy_knowledge_rebuild_index,
            snapshot_dir=args.snapshot_dir,
            progress=not args.no_progress,
        )
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "tickers": [row.get("ticker") for row in board.get("candidates", [])],
                "counts": board.get("counts", {}),
                "artifact_paths": board.get("artifact_paths", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

