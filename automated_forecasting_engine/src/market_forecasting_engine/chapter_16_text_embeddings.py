from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from market_forecasting_engine.data_store import frame_sha256
from market_forecasting_engine.llm_usage import log_openai_embedding_usage, monotonic_ms, new_llm_call_id


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 256
PROMPT_VERSION = "chapter_16_text_embeddings_v1"

FINANCE_KNOWLEDGE_PROTOTYPES: dict[str, str] = {
    "earnings_upside": "earnings beat, raised guidance, expanding margins, stronger demand, positive operating leverage",
    "earnings_downside": "earnings miss, guidance cut, margin pressure, revenue shortfall, weak demand",
    "liquidity_stress": "debt refinancing risk, liquidity pressure, cash burn, covenant pressure, funding stress",
    "regulatory_risk": "regulatory investigation, export controls, antitrust pressure, compliance risk, sanctions exposure",
    "product_innovation": "new product launch, technology breakthrough, platform upgrade, customer adoption, innovation cycle",
    "competitive_pressure": "market share loss, pricing pressure, new competitor, substitution risk, commoditization",
    "macro_rate_headwind": "higher interest rates, tighter financial conditions, stronger dollar, macro slowdown",
    "capital_return": "share buyback, dividend increase, capital return, balance sheet strength, free cash flow",
}


@dataclass(frozen=True)
class Chapter16EmbeddingRequest:
    ticker: str
    model: str | None = None
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    max_articles: int = 24
    llm_env_file: str | None = None
    timeout_seconds: int = 30
    finance_knowledge_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ticker"] = self.ticker.upper()
        return data


def extract_text_embeddings_for_articles(
    articles: list[dict[str, Any]],
    request: Chapter16EmbeddingRequest,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_articles = _article_payload(articles[: max(1, int(request.max_articles))])
    if not selected_articles:
        return [], _metadata(request, "skipped", reason="no_articles")
    api_key = os.environ.get("OPENAI_API_KEY") or _read_env_value(request.llm_env_file, "OPENAI_API_KEY")
    model = request.model or os.environ.get("OPENAI_EMBEDDING_MODEL") or _read_env_value(request.llm_env_file, "OPENAI_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    dimensions = max(16, int(request.dimensions or DEFAULT_EMBEDDING_DIMENSIONS))
    if not api_key:
        return [], _metadata(request, "skipped", model=model, reason="OPENAI_API_KEY is not available.")

    prototypes = _load_finance_knowledge_prototypes(request.finance_knowledge_path)
    prototype_items = [{"id": key, "text": value} for key, value in prototypes.items()]
    texts = [item["text"] for item in selected_articles] + [item["text"] for item in prototype_items]
    payload: dict[str, Any] = {
        "model": model,
        "input": texts,
        "encoding_format": "float",
    }
    if model.startswith("text-embedding-3"):
        payload["dimensions"] = dimensions

    call_id = new_llm_call_id()
    started_ms = monotonic_ms()
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=float(request.timeout_seconds))
        response = client.embeddings.create(**payload)
        response_data = _response_json(response)
        log_openai_embedding_usage(
            call_id=call_id,
            model=model,
            payload=payload,
            response_data=response_data,
            started_ms=started_ms,
            status="ok",
            context={"purpose": "chapter_16_text_embeddings", "ticker": request.ticker.upper(), "process": "alternative_data"},
            api_key=getattr(client, "api_key", None),
        )
        vectors = _extract_vectors(response_data)
        article_count = len(selected_articles)
        article_vectors = vectors[:article_count]
        prototype_vectors = vectors[article_count:]
        rows = _embedding_rows(
            selected_articles,
            article_vectors,
            prototype_items,
            prototype_vectors,
            model=model,
            dimensions=dimensions,
        )
        return rows, _metadata(
            request,
            "executed",
            model=model,
            dimensions=dimensions,
            article_count=article_count,
            prototype_count=len(prototype_items),
            embedding_count=len(vectors),
            response_id=response_data.get("id"),
            finance_knowledge_hash=_finance_knowledge_hash(prototypes),
        )
    except Exception as exc:
        log_openai_embedding_usage(
            call_id=call_id,
            model=model,
            payload=payload,
            started_ms=started_ms,
            status="error",
            error=str(exc),
            context={"purpose": "chapter_16_text_embeddings", "ticker": request.ticker.upper(), "process": "alternative_data"},
            api_key=api_key,
        )
        return [], _metadata(request, "failed", model=model, dimensions=dimensions, reason=f"{type(exc).__name__}: {exc}")


def aggregate_embedding_features(
    embedding_rows: list[dict[str, Any]] | pd.DataFrame,
    target_index: pd.DatetimeIndex,
    *,
    windows: tuple[int, ...] = (1, 3, 7, 14),
    n_components: int = 6,
) -> pd.DataFrame:
    index = pd.DatetimeIndex(target_index)
    frame = pd.DataFrame(embedding_rows) if not isinstance(embedding_rows, pd.DataFrame) else embedding_rows.copy()
    base_columns: dict[str, float] = {}
    for component in range(n_components):
        for window in windows:
            base_columns[f"alt_embed_component_{component}_{window}d"] = 0.0
    for window in windows:
        base_columns[f"alt_embed_novelty_{window}d"] = 0.0
        base_columns[f"alt_embed_dispersion_{window}d"] = 0.0
        base_columns[f"alt_embed_article_count_{window}d"] = 0.0
    for prototype_id in FINANCE_KNOWLEDGE_PROTOTYPES:
        for window in (3, 7, 14):
            base_columns[f"alt_embed_{prototype_id}_similarity_{window}d"] = 0.0
    output = pd.DataFrame(base_columns, index=index)
    if frame.empty:
        return output
    frame["date"] = pd.to_datetime(frame.get("date", frame.get("published_at")), errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date"])
    if frame.empty or "embedding" not in frame:
        return output

    vectors = _vectors_from_frame(frame)
    if vectors.size == 0:
        return output
    n_components = min(n_components, vectors.shape[0], vectors.shape[1])
    components = (
        PCA(n_components=n_components, random_state=42).fit_transform(vectors)
        if n_components > 0 and vectors.shape[0] >= 2
        else np.zeros((len(frame), n_components))
    )
    frame = frame.reset_index(drop=True)
    for component in range(n_components):
        frame[f"component_{component}"] = components[:, component]
    similarities = [column for column in frame.columns if str(column).endswith("_similarity") and str(column).startswith("finance_")]

    daily_index = pd.DatetimeIndex(index.normalize().unique()).sort_values()
    daily = pd.DataFrame(index=daily_index)
    grouped = frame.groupby("date", sort=True)
    for component in range(n_components):
        daily[f"component_{component}"] = grouped[f"component_{component}"].mean()
    daily["article_count"] = grouped.size().astype(float)
    daily["novelty"] = grouped["semantic_novelty"].mean() if "semantic_novelty" in frame else 0.0
    daily["dispersion"] = grouped["semantic_dispersion"].mean() if "semantic_dispersion" in frame else 0.0
    for column in similarities:
        daily[column] = grouped[column].mean()
    daily = daily.reindex(daily_index).fillna(0.0)

    columns: dict[str, np.ndarray] = {}
    for component in range(n_components):
        series = daily[f"component_{component}"]
        for window in windows:
            columns[f"alt_embed_component_{component}_{window}d"] = series.rolling(window, min_periods=1).mean().reindex(index.normalize()).to_numpy()
    for window in windows:
        columns[f"alt_embed_novelty_{window}d"] = daily["novelty"].rolling(window, min_periods=1).mean().reindex(index.normalize()).to_numpy()
        columns[f"alt_embed_dispersion_{window}d"] = daily["dispersion"].rolling(window, min_periods=1).mean().reindex(index.normalize()).to_numpy()
        columns[f"alt_embed_article_count_{window}d"] = daily["article_count"].rolling(window, min_periods=1).sum().reindex(index.normalize()).to_numpy()
    for prototype_id in FINANCE_KNOWLEDGE_PROTOTYPES:
        source = f"finance_{prototype_id}_similarity"
        if source not in daily:
            continue
        for window in (3, 7, 14):
            weighted_sum = (daily[source] * daily["article_count"]).rolling(window, min_periods=1).sum()
            article_count = daily["article_count"].rolling(window, min_periods=1).sum()
            weighted_mean = weighted_sum.divide(article_count.replace(0.0, np.nan)).fillna(0.0)
            columns[f"alt_embed_{prototype_id}_similarity_{window}d"] = weighted_mean.reindex(index.normalize()).to_numpy()
    if columns:
        output.update(pd.DataFrame(columns, index=index))
    return output.copy()


def chapter_16_embedding_registry_entry(
    request: Chapter16EmbeddingRequest,
    embeddings: pd.DataFrame,
    features: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": "chapter_16_text_embeddings",
        "version": PROMPT_VERSION,
        "source": "Machine Learning for Algorithmic Trading, 2nd ed., Chapter 16",
        "ticker": request.ticker.upper(),
        "status": metadata.get("status"),
        "embedding_model": metadata.get("model") or request.model or DEFAULT_EMBEDDING_MODEL,
        "dimensions": metadata.get("dimensions") or request.dimensions,
        "document_count": int(len(embeddings)),
        "feature_count": int(features.shape[1]) if not features.empty else 0,
        "expected_signal_type": "semantic_embedding_novelty_finance_prototype_similarity",
        "decision_policy": {
            "influences_final_action": False,
            "influences_model_selection": True,
            "mode": "feature_validation",
            "reason": "Embedding features enter the supervised feature matrix and affect model selection only if walk-forward validation supports them.",
        },
        "finance_knowledge": {
            "mode": "built_in_or_json_prototypes",
            "prototype_count": len(_load_finance_knowledge_prototypes(request.finance_knowledge_path)),
            "hash": metadata.get("finance_knowledge_hash"),
        },
        "provenance": {
            "raw_embedding_hash": frame_sha256(embeddings) if not embeddings.empty else None,
            "feature_hash": frame_sha256(features) if not features.empty else None,
        },
    }


def _article_payload(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = []
    for index, article in enumerate(articles):
        title = str(article.get("title") or "").strip()
        summary = str(article.get("summary") or article.get("description") or "").strip()
        text = " ".join(part for part in [title, summary] if part).strip()
        if not text:
            continue
        payload.append(
            {
                "article_id": str(article.get("article_id") or _article_id(article, index)),
                "ticker": str(article.get("ticker") or "").upper(),
                "date": str(article.get("date") or "")[:10] or None,
                "published_at": article.get("published_at"),
                "source": article.get("source"),
                "title": title[:500],
                "summary": summary[:2000],
                "text": text[:6000],
                "raw_text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    return payload


def _embedding_rows(
    articles: list[dict[str, Any]],
    article_vectors: list[list[float]],
    prototypes: list[dict[str, str]],
    prototype_vectors: list[list[float]],
    *,
    model: str,
    dimensions: int,
) -> list[dict[str, Any]]:
    prototype_matrix = _normalize_matrix(np.asarray(prototype_vectors, dtype=float)) if prototype_vectors else np.zeros((0, 0))
    article_matrix = _normalize_matrix(np.asarray(article_vectors, dtype=float)) if article_vectors else np.zeros((0, 0))
    similarities = article_matrix @ prototype_matrix.T if article_matrix.size and prototype_matrix.size else np.zeros((len(articles), len(prototypes)))
    centroid = article_matrix.mean(axis=0) if article_matrix.size else np.zeros(0)
    rows = []
    for index, article in enumerate(articles):
        vector = [float(value) for value in article_vectors[index]] if index < len(article_vectors) else []
        row: dict[str, Any] = {
            **{key: value for key, value in article.items() if key != "text"},
            "embedding": vector,
            "embedding_model": model,
            "embedding_dimensions": int(len(vector) or dimensions),
            "semantic_novelty": _cosine_distance(article_matrix[index], centroid) if article_matrix.size else 0.0,
            "semantic_dispersion": _mean_pairwise_distance(article_matrix, index) if article_matrix.size else 0.0,
            "chapter_16_method": "openai_text_embedding_finance_prototypes",
            "prompt_version": PROMPT_VERSION,
        }
        for prototype_index, prototype in enumerate(prototypes):
            row[f"finance_{prototype['id']}_similarity"] = float(similarities[index, prototype_index]) if similarities.size else 0.0
        rows.append(row)
    return rows


def _extract_vectors(response_data: dict[str, Any]) -> list[list[float]]:
    data = response_data.get("data") or []
    if not isinstance(data, list):
        return []
    ordered = sorted((item for item in data if isinstance(item, dict)), key=lambda item: int(item.get("index", 0)))
    return [[float(value) for value in item.get("embedding", [])] for item in ordered]


def _vectors_from_frame(frame: pd.DataFrame) -> np.ndarray:
    vectors = []
    for value in frame["embedding"]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                continue
        if isinstance(value, (list, tuple)) and value:
            vector = np.asarray(value, dtype=float)
            if np.isfinite(vector).all():
                vectors.append(vector)
    if not vectors:
        return np.zeros((0, 0))
    width = min(len(vector) for vector in vectors)
    return np.vstack([vector[:width] for vector in vectors])


def _response_json(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        data = response.model_dump(mode="json")
        return data if isinstance(data, dict) else {}
    if isinstance(response, dict):
        return response
    return {}


def _load_finance_knowledge_prototypes(path: str | None) -> dict[str, str]:
    prototypes = dict(FINANCE_KNOWLEDGE_PROTOTYPES)
    if not path:
        return prototypes
    try:
        with open(os.path.expanduser(path), encoding="utf-8") as handle:
            parsed = json.load(handle)
    except Exception:
        return prototypes
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            clean_key = _safe_feature_label(str(key))
            clean_value = str(value or "").strip()
            if clean_key and clean_value:
                prototypes[clean_key] = clean_value[:2000]
    return prototypes


def _metadata(request: Chapter16EmbeddingRequest, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "kind": "chapter_16_text_embeddings",
        "status": status,
        "request": request.to_dict(),
        "prompt_version": PROMPT_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        **extra,
    }


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    return matrix / norms


def _cosine_distance(vector: np.ndarray, centroid: np.ndarray) -> float:
    if vector.size == 0 or centroid.size == 0:
        return 0.0
    denominator = float(np.linalg.norm(vector) * np.linalg.norm(centroid))
    if denominator <= 1e-12:
        return 0.0
    return float(1.0 - np.dot(vector, centroid) / denominator)


def _mean_pairwise_distance(matrix: np.ndarray, index: int) -> float:
    if matrix.shape[0] <= 1:
        return 0.0
    vector = matrix[index]
    similarities = matrix @ vector
    distances = 1.0 - similarities
    return float(np.delete(distances, index).mean())


def _finance_knowledge_hash(prototypes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(prototypes, sort_keys=True).encode("utf-8")).hexdigest()


def _article_id(article: dict[str, Any], index: int) -> str:
    raw = "|".join(str(article.get(key) or "") for key in ("published_at", "date", "source", "title", "url"))
    return hashlib.sha256(f"{index}|{raw}".encode("utf-8")).hexdigest()[:16]


def _safe_feature_label(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _read_env_value(path: str | None, key: str) -> str | None:
    if not path:
        return None
    env_path = os.path.expanduser(path)
    if not os.path.exists(env_path):
        return None
    with open(env_path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            candidate, value = line.split("=", 1)
            if candidate.strip() == key:
                return value.strip().strip('"').strip("'")
    return None
