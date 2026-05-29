"""Persistent cache for live market data requests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


class DataCache:
    """Small pickle/json cache keyed by normalized request parameters."""

    def __init__(self, root: Path, ttl_hours: float | None = 24.0):
        self.root = Path(root)
        self.ttl_hours = ttl_hours
        self.root.mkdir(parents=True, exist_ok=True)

    def key_for(self, namespace: str, params: dict[str, Any]) -> str:
        payload = json.dumps(params, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"{namespace}_{digest}"

    def read_frame(self, key: str) -> pd.DataFrame | None:
        data_path = self.root / f"{key}.pkl"
        meta_path = self.root / f"{key}.json"
        if not data_path.exists() or not meta_path.exists():
            return None
        if self._expired(meta_path):
            return None
        return pd.read_pickle(data_path)

    def write_frame(self, key: str, frame: pd.DataFrame, params: dict[str, Any]) -> None:
        data_path = self.root / f"{key}.pkl"
        meta_path = self.root / f"{key}.json"
        frame.to_pickle(data_path)
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rows": int(len(frame)),
            "columns": list(map(str, frame.columns)),
            "params": params,
        }
        meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def _expired(self, meta_path: Path) -> bool:
        if self.ttl_hours is None:
            return False
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            created_at = datetime.fromisoformat(metadata["created_at"])
        except (KeyError, ValueError, json.JSONDecodeError):
            return True
        age_seconds = (datetime.now(timezone.utc) - created_at).total_seconds()
        return age_seconds > self.ttl_hours * 3600
