from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DataArtifact:
    layer: str
    provider: str
    ticker: str
    path: str
    format: str
    rows: int
    columns: list[str]
    sha256: str
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "provider": self.provider,
            "ticker": self.ticker,
            "path": self.path,
            "format": self.format,
            "rows": self.rows,
            "columns": self.columns,
            "sha256": self.sha256,
            "created_at_utc": self.created_at_utc,
        }


class MarketDataStore:
    """Small local data lake for raw, normalized, feature, and panel datasets."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def find_frame(self, layer: str, provider: str, ticker: str, request_key: str) -> Path | None:
        directory = self._dataset_dir(layer, provider, ticker)
        if not directory.exists():
            return None
        for suffix in ("parquet", "pkl", "csv"):
            candidate = directory / f"{request_key}.{suffix}"
            if candidate.exists():
                return candidate
        return None

    def read_frame(self, path: str | Path) -> pd.DataFrame:
        file_path = Path(path)
        if file_path.suffix == ".parquet":
            return pd.read_parquet(file_path)
        if file_path.suffix == ".csv":
            frame = pd.read_csv(file_path)
            if frame.columns[0] == "date":
                frame.index = pd.to_datetime(frame.pop("date"), errors="coerce")
            return frame
        return pd.read_pickle(file_path)

    def write_frame(
        self,
        layer: str,
        provider: str,
        ticker: str,
        request_key: str,
        frame: pd.DataFrame,
        preferred_format: str = "parquet",
    ) -> DataArtifact:
        directory = self._dataset_dir(layer, provider, ticker)
        directory.mkdir(parents=True, exist_ok=True)
        clean_frame = frame.copy()
        if isinstance(clean_frame.index, pd.MultiIndex):
            clean_frame.index.names = [name or f"level_{position}" for position, name in enumerate(clean_frame.index.names)]
        else:
            clean_frame.index.name = clean_frame.index.name or "date"

        file_format = preferred_format
        path = directory / f"{request_key}.{file_format}"
        try:
            if file_format == "parquet":
                clean_frame.to_parquet(path)
            elif file_format == "csv":
                clean_frame.to_csv(path)
            else:
                file_format = "pkl"
                path = directory / f"{request_key}.pkl"
                clean_frame.to_pickle(path)
        except Exception:
            file_format = "pkl"
            path = directory / f"{request_key}.pkl"
            clean_frame.to_pickle(path)

        artifact = DataArtifact(
            layer=layer,
            provider=provider,
            ticker=ticker,
            path=str(path),
            format=file_format,
            rows=int(len(clean_frame)),
            columns=[str(column) for column in clean_frame.columns],
            sha256=frame_sha256(clean_frame),
            created_at_utc=datetime.now(UTC).isoformat(),
        )
        self.write_json(
            "metadata",
            provider,
            ticker,
            f"{request_key}_{layer}_artifact",
            artifact.to_dict(),
        )
        return artifact

    def write_json(self, layer: str, provider: str, ticker: str, name: str, payload: Any) -> Path:
        directory = self._dataset_dir(layer, provider, ticker)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    def _dataset_dir(self, layer: str, provider: str, ticker: str) -> Path:
        return self.root / _safe_part(layer) / _safe_part(provider) / _safe_part(ticker.upper())


def frame_sha256(frame: pd.DataFrame) -> str:
    payload = frame.copy()
    if not isinstance(payload.index, pd.MultiIndex):
        try:
            payload.index = pd.to_datetime(payload.index)
        except Exception:
            pass
    payload = payload.sort_index()
    data = payload.to_csv(index=True, float_format="%.12g").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def request_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _safe_part(value: str) -> str:
    safe = "".join(character.lower() if character.isalnum() else "_" for character in str(value)).strip("_")
    return safe or "unknown"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
