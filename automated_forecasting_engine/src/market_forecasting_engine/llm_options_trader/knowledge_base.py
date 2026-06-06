from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"


def load_strategy_knowledge(*, knowledge_dir: Path | None = None) -> dict[str, Any]:
    root = knowledge_dir or DEFAULT_KNOWLEDGE_DIR
    if not root.exists():
        return {"status": "empty", "topics": []}
    topics: list[dict[str, str]] = []
    for path in sorted(root.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        topics.append(
            {
                "id": path.stem,
                "title": _title_from_markdown(text, fallback=path.stem.replace("_", " ").title()),
                "body": text,
            }
        )
    return {"status": "ok" if topics else "empty", "topics": topics}


def _title_from_markdown(text: str, *, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
    return fallback
