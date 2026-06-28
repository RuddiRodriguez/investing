#!/usr/bin/env python3
"""Archive old agent run directories out of the repo.

The repo keeps code, configs, data, models, current state, and the newest run
artifacts. Historical runs move to a Dropbox-backed archive so the working tree
stays small without losing audit evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_RUNS_DIR = Path("automated_forecasting_engine/runs")
DEFAULT_ARCHIVE_DIR = Path.home() / "Dropbox" / "invest-agent-archive" / "automated_forecasting_engine" / "runs"
DEFAULT_KEEP_LATEST = 3

ALWAYS_KEEP_NAMES = {
    "_archive_manifests",
    "archive",
    "deribit_live_strategy_testnet",
    "deribit_scalping_testnet",
    "emergency_close_alpaca_paper_options",
    "live_alpaca_breakout_iwm",
    "live_alpaca_breakout_spy",
    "live_alpaca_breakout_tlt",
    "live_deribit_eth_usdc_daily_agent",
    "live_trading",
    "llm_options_trader_live_shadow",
    "llm_options_trader_live_shadow_hf",
    "llm_options_trader_testnet",
    "logs",
    "openai_usage",
    "paper_options_performance_dashboard",
    "portfolio_brain",
    "portfolio_projection_latest",
    "trade_republic_eod_dashboard",
    "watch_agent_state",
}

FORBIDDEN_NAMES = {
    ".env",
    ".venv",
    "__pycache__",
    "venv",
}


@dataclass
class ArchiveDecision:
    name: str
    source: str
    destination: str | None
    modified_at_utc: str
    size_bytes: int
    action: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move old agent run artifacts to Dropbox archive.")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--keep-latest", type=int, default=DEFAULT_KEEP_LATEST)
    parser.add_argument("--execute", action="store_true", help="Move files. Without this flag, only writes a dry-run plan.")
    return parser.parse_args()


def dir_size(path: Path) -> int:
    total = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [name for name in dirs if name not in FORBIDDEN_NAMES]
        for file_name in files:
            file_path = Path(root) / file_name
            if file_path.name.startswith(".env") or file_path.name in FORBIDDEN_NAMES:
                continue
            try:
                total += file_path.stat().st_size
            except FileNotFoundError:
                continue
    return total


def modified_at(path: Path) -> float:
    newest = path.stat().st_mtime
    for root, dirs, files in os.walk(path):
        dirs[:] = [name for name in dirs if name not in FORBIDDEN_NAMES]
        for entry in files:
            file_path = Path(root) / entry
            try:
                newest = max(newest, file_path.stat().st_mtime)
            except FileNotFoundError:
                continue
    return newest


def utc_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def unique_destination(archive_dir: Path, source: Path) -> Path:
    destination = archive_dir / source.name
    if not destination.exists():
        return destination
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return archive_dir / f"{source.name}_{suffix}"


def build_plan(runs_dir: Path, archive_dir: Path, keep_latest: int) -> list[ArchiveDecision]:
    candidates = [path for path in runs_dir.iterdir() if path.is_dir() and path.name not in FORBIDDEN_NAMES]
    metadata = [(path, modified_at(path), dir_size(path)) for path in candidates]
    recency_candidates = [item for item in metadata if item[0].name not in ALWAYS_KEEP_NAMES]
    keep_by_recency = {
        path for path, _, _ in sorted(recency_candidates, key=lambda item: item[1], reverse=True)[:keep_latest]
    }

    decisions: list[ArchiveDecision] = []
    for path, mtime, size in sorted(metadata, key=lambda item: item[1], reverse=True):
        destination: Path | None = None
        action = "archive"
        reason = "historical run artifact"
        if path.name in ALWAYS_KEEP_NAMES:
            action = "keep"
            reason = "operational state or local index"
        elif path in keep_by_recency:
            action = "keep"
            reason = f"newest {keep_latest} run dirs kept local"
        else:
            destination = unique_destination(archive_dir, path)

        decisions.append(
            ArchiveDecision(
                name=path.name,
                source=str(path),
                destination=str(destination) if destination else None,
                modified_at_utc=utc_from_timestamp(mtime),
                size_bytes=size,
                action=action,
                reason=reason,
            )
        )
    return decisions


def write_manifest(
    runs_dir: Path,
    archive_dir: Path,
    decisions: list[ArchiveDecision],
    executed: bool,
    keep_latest: int,
) -> Path:
    manifest_dir = runs_dir / "_archive_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "created_at_utc": utc_from_timestamp(datetime.now(timezone.utc).timestamp()),
        "executed": executed,
        "runs_dir": str(runs_dir),
        "archive_dir": str(archive_dir),
        "policy": {
            "keep_latest": keep_latest,
            "always_keep_names": sorted(ALWAYS_KEEP_NAMES),
            "forbidden_names": sorted(FORBIDDEN_NAMES),
            "note": "Dropbox archive can be online-only. Restore a run by moving its folder back under runs/.",
        },
        "decisions": [asdict(decision) for decision in decisions],
    }
    local_manifest = manifest_dir / f"archive_manifest_{stamp}.json"
    local_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_manifest = archive_dir / "_archive_manifests" / local_manifest.name
    archive_manifest.parent.mkdir(parents=True, exist_ok=True)
    archive_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return local_manifest


def execute_plan(decisions: list[ArchiveDecision], archive_dir: Path) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    for decision in decisions:
        if decision.action != "archive" or not decision.destination:
            continue
        source = Path(decision.source)
        destination = Path(decision.destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            continue
        shutil.move(str(source), str(destination))


def main() -> int:
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    archive_dir = args.archive_dir.expanduser().resolve()
    if not runs_dir.exists():
        raise SystemExit(f"runs dir not found: {runs_dir}")
    if args.keep_latest < 0:
        raise SystemExit("--keep-latest must be >= 0")

    decisions = build_plan(runs_dir, archive_dir, args.keep_latest)
    if args.execute:
        execute_plan(decisions, archive_dir)
    manifest = write_manifest(runs_dir, archive_dir, decisions, args.execute, args.keep_latest)

    archived = [decision for decision in decisions if decision.action == "archive"]
    kept = [decision for decision in decisions if decision.action == "keep"]
    archived_bytes = sum(decision.size_bytes for decision in archived)
    print(
        json.dumps(
            {
                "executed": args.execute,
                "manifest": str(manifest),
                "archive_dir": str(archive_dir),
                "archived_count": len(archived),
                "kept_count": len(kept),
                "archived_bytes": archived_bytes,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
