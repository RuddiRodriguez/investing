from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from market_forecasting_engine.chapter_16_text_embeddings import DEFAULT_EMBEDDING_DIMENSIONS, DEFAULT_EMBEDDING_MODEL
from market_forecasting_engine.llm_usage import log_openai_embedding_usage, monotonic_ms, new_llm_call_id


DEFAULT_STRATEGY_CORPUS_DIR = Path("automated_forecasting_engine/strategy_knowledge/corpus")
DEFAULT_STRATEGY_INDEX_PATH = Path("automated_forecasting_engine/strategy_knowledge/indexes/strategy_knowledge.faiss")
STRATEGY_KNOWLEDGE_VERSION = "strategy_knowledge_faiss_v1"
MAX_CHARS_PER_CHUNK = 1800
CHUNK_OVERLAP_CHARS = 250


@dataclass(frozen=True)
class StrategyKnowledgeRequest:
    ticker: str
    corpus_dir: str | Path = DEFAULT_STRATEGY_CORPUS_DIR
    index_path: str | Path = DEFAULT_STRATEGY_INDEX_PATH
    llm_env_file: str | None = None
    embedding_model: str | None = None
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    max_chunks: int = 8
    rebuild_index: bool = False
    timeout_seconds: int = 45

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ticker"] = self.ticker.upper()
        data["corpus_dir"] = str(self.corpus_dir)
        data["index_path"] = str(self.index_path)
        return data


def build_strategy_knowledge_context(report: dict[str, Any], request: StrategyKnowledgeRequest) -> dict[str, Any]:
    corpus_dir = Path(request.corpus_dir).expanduser()
    if not corpus_dir.exists():
        return _context_status(request, "skipped", reason=f"strategy corpus directory does not exist: {corpus_dir}")
    documents = load_strategy_documents(corpus_dir)
    if not documents:
        return _context_status(request, "skipped", reason=f"no supported strategy documents found in {corpus_dir}")
    index_status = ensure_strategy_index(documents, request)
    query = build_strategy_query_from_report(report)
    retrieved = retrieve_strategy_chunks(query, request)
    synthesis = synthesize_strategy_context(query=query, retrieved_chunks=retrieved)
    return {
        "status": "executed" if retrieved else "empty",
        "version": STRATEGY_KNOWLEDGE_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "request": request.to_dict(),
        "query": query,
        "index_status": index_status,
        "retrieved_chunks": retrieved,
        "synthesis": synthesis,
        "decision_policy": {
            "feeds_ceo_llm": True,
            "overrides_model_validation": False,
            "overrides_risk_gates": False,
            "role": "durable strategy context for interpreting ticker setup, entries, exits, and sizing.",
        },
    }


def attach_strategy_knowledge_context(report: dict[str, Any], context: dict[str, Any]) -> None:
    report.setdefault("decision_view", {})["strategy_knowledge_context"] = context
    report.setdefault("final_decision_reasoning", {})["strategy_knowledge_status"] = context.get("status")


def load_strategy_documents(corpus_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(path for path in corpus_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".md", ".txt", ".pdf"})
    documents: list[dict[str, Any]] = []
    for path in paths:
        text, extraction = _extract_document_text(path)
        if not text.strip():
            continue
        documents.append(
            {
                "source_id": _source_id(path),
                "source_path": str(path),
                "source_title": _source_title(path, text),
                "source_type": _source_type(path),
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "extraction": extraction,
            }
        )
    return documents


def ensure_strategy_index(documents: list[dict[str, Any]], request: StrategyKnowledgeRequest) -> dict[str, Any]:
    index_path = Path(request.index_path).expanduser()
    manifest_path = _manifest_path(index_path)
    chunks_path = _chunks_path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_hash = _documents_hash(documents)
    existing_manifest = _read_json(manifest_path)
    if (
        not request.rebuild_index
        and index_path.exists()
        and chunks_path.exists()
        and existing_manifest.get("corpus_hash") == corpus_hash
    ):
        return {
            "status": "current",
            "backend": "faiss",
            "index_path": str(index_path),
            "chunks_path": str(chunks_path),
            "manifest_path": str(manifest_path),
            "corpus_hash": corpus_hash,
            "chunk_count": existing_manifest.get("chunk_count"),
        }
    chunks = _chunk_documents(documents)
    vectors, embedding_status = _embed_chunks(chunks, request)
    if not vectors:
        _write_jsonl(chunks_path, chunks)
        _write_json(
            manifest_path,
            _manifest(request, documents, chunks, corpus_hash, embedding_status, status="lexical_only"),
        )
        return {
            "status": "lexical_only",
            "backend": "faiss",
            "reason": "embedding vectors unavailable; retrieval will use lexical scoring",
            "index_path": str(index_path),
            "chunks_path": str(chunks_path),
            "manifest_path": str(manifest_path),
            "corpus_hash": corpus_hash,
            "chunk_count": len(chunks),
            "embedding_status": embedding_status,
        }
    matrix = _normalize_matrix(np.asarray(vectors, dtype="float32"))
    faiss = _faiss()
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(index_path))
    chunk_rows = []
    for chunk, vector in zip(chunks, vectors, strict=False):
        chunk_rows.append({**chunk, "embedding_model": _embedding_model(request), "embedding_dimensions": len(vector)})
    _write_jsonl(chunks_path, chunk_rows)
    _write_json(manifest_path, _manifest(request, documents, chunk_rows, corpus_hash, embedding_status, status="ready"))
    return {
        "status": "rebuilt",
        "backend": "faiss",
        "index_path": str(index_path),
        "chunks_path": str(chunks_path),
        "manifest_path": str(manifest_path),
        "corpus_hash": corpus_hash,
        "document_count": len(documents),
        "chunk_count": len(chunk_rows),
        "embedding_status": embedding_status,
    }


def retrieve_strategy_chunks(query: str, request: StrategyKnowledgeRequest) -> list[dict[str, Any]]:
    index_path = Path(request.index_path).expanduser()
    chunks = _read_jsonl(_chunks_path(index_path))
    if not chunks:
        return []
    query_vector = _embed_query(query, request)
    lexical = _lexical_rank(query, chunks)
    semantic_hits: dict[int, tuple[float, int]] = {}
    if query_vector and index_path.exists():
        try:
            faiss = _faiss()
            index = faiss.read_index(str(index_path))
            query_matrix = _normalize_matrix(np.asarray([query_vector], dtype="float32"))
            distances, indices = index.search(query_matrix, min(max(1, request.max_chunks * 3), len(chunks)))
            for rank, (idx, score) in enumerate(zip(indices[0].tolist(), distances[0].tolist(), strict=False), start=1):
                if 0 <= idx < len(chunks):
                    semantic_hits[idx] = (float(score), rank)
        except Exception:
            semantic_hits = {}
    scored = []
    for idx, chunk in enumerate(chunks):
        semantic_score = semantic_hits.get(idx, (None, None))[0]
        lexical_score = lexical.get(idx, 0.0)
        score = (float(semantic_score) * 0.80 + lexical_score * 0.20) if semantic_score is not None else lexical_score
        if score > 0:
            scored.append((score, semantic_score, lexical_score, idx, chunk))
    top = sorted(scored, key=lambda item: item[0], reverse=True)[: max(1, int(request.max_chunks))]
    return [_chunk_result(score, semantic, lexical_score, idx, chunk) for score, semantic, lexical_score, idx, chunk in top]


def synthesize_strategy_context(*, query: str, retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    principles: list[str] = []
    entry_rules: list[str] = []
    risk_rules: list[str] = []
    warnings: list[str] = []
    source_titles: list[str] = []
    for chunk in retrieved_chunks:
        text = str(chunk.get("text_excerpt") or "")
        source_titles.append(str(chunk.get("source_title") or chunk.get("source_id") or "unknown"))
        lower = text.lower()
        if any(token in lower for token in ("earnings", "annual", "current")):
            principles.append("Prioritize strong current/annual earnings growth and estimate revisions when judging quality.")
        if any(token in lower for token in ("new product", "new service", "new leadership", "catalyst")):
            principles.append("Look for a new product, service, leadership, market, or catalyst that changes company character.")
        if any(token in lower for token in ("breakout", "base", "pivot", "trendline")):
            entry_rules.append("Prefer disciplined breakouts or early trendline entries from proper bases, not blind chasing.")
        if any(token in lower for token in ("pullback", "10-day", "21-day", "50-day", "support")):
            entry_rules.append("For leaders, pullbacks into institutional support can be safer than extended entries.")
        if any(token in lower for token in ("7%", "8%", "stop", "risk", "position")):
            risk_rules.append("Only enter when price is close enough to a logical stop to keep loss within the defined risk budget.")
        if any(token in lower for token in ("market direction", "follow-through", "confirmed uptrend")):
            warnings.append("Do not get aggressive unless the broad market confirms an uptrend.")
        if any(token in lower for token in ("choppy", "failed", "cash is a position", "chop")):
            warnings.append("Failed breakouts or choppy markets argue for cash, pilot size, or waiting.")
    return {
        "status": "deterministic_synthesis",
        "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "applicable_principles": _dedupe(principles)[:8],
        "entry_rules": _dedupe(entry_rules)[:8],
        "risk_and_sizing_rules": _dedupe(risk_rules)[:8],
        "warnings": _dedupe(warnings)[:8],
        "source_titles": _dedupe(source_titles)[:8],
        "ceo_instruction": (
            "Use this strategy knowledge as durable professional context. Reconcile it with model validation, "
            "current source synthesis, market direction, support/resistance, buy-lower setup, and portfolio risk. "
            "It is not an automatic Buy/Sell override."
        ),
    }


def build_strategy_query_from_report(report: dict[str, Any]) -> str:
    decision = report.get("decision_view", {}) if isinstance(report.get("decision_view"), dict) else {}
    technical = report.get("technical_view", {}) if isinstance(report.get("technical_view"), dict) else {}
    forecasts = report.get("forecasts", []) if isinstance(report.get("forecasts"), list) else []
    long_term = decision.get("long_term_context", {}) if isinstance(decision.get("long_term_context"), dict) else {}
    source_synthesis = long_term.get("llm_source_synthesis", {}) if isinstance(long_term.get("llm_source_synthesis"), dict) else {}
    synthesis = source_synthesis.get("synthesis", {}) if isinstance(source_synthesis.get("synthesis"), dict) else {}
    return "\n".join(
        [
            f"ticker {report.get('ticker')}",
            f"suggested action {report.get('suggested_action')}",
            f"risk level {report.get('risk_level')}",
            f"current price {report.get('current_price')}",
            f"forecasts {json.dumps(_forecast_query_parts(forecasts), sort_keys=True, default=str)}",
            f"trend {json.dumps(technical.get('trend_state', {}), sort_keys=True, default=str)}",
            f"support resistance {json.dumps(technical.get('chapter_13_support_resistance', {}), sort_keys=True, default=str)[:2000]}",
            f"tactical {json.dumps(decision.get('chapter_18_tactical_problem', {}), sort_keys=True, default=str)[:2500]}",
            f"source synthesis {json.dumps({k: synthesis.get(k) for k in ('bullish_evidence','bearish_evidence','decision_implications','analyst_read','valuation_read')}, sort_keys=True, default=str)[:4000]}",
        ]
    )


def _embed_chunks(chunks: list[dict[str, Any]], request: StrategyKnowledgeRequest) -> tuple[list[list[float]], dict[str, Any]]:
    return _embed_texts(
        [chunk["text"] for chunk in chunks],
        request,
        purpose="strategy_knowledge_index",
    )


def _embed_query(query: str, request: StrategyKnowledgeRequest) -> list[float]:
    vectors, _ = _embed_texts([query[:8000]], request, purpose="strategy_knowledge_query")
    return vectors[0] if vectors else []


def _embed_texts(texts: list[str], request: StrategyKnowledgeRequest, *, purpose: str) -> tuple[list[list[float]], dict[str, Any]]:
    api_key = os.environ.get("OPENAI_API_KEY") or _read_env_value(request.llm_env_file, "OPENAI_API_KEY")
    model = _embedding_model(request)
    dimensions = int(request.embedding_dimensions or DEFAULT_EMBEDDING_DIMENSIONS)
    if not api_key:
        return [], {"status": "skipped", "reason": "OPENAI_API_KEY unavailable", "model": model}
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=float(request.timeout_seconds))
        vectors: list[list[float]] = []
        batch_statuses: list[dict[str, Any]] = []
        for batch_index, batch in enumerate(_embedding_batches(texts), start=1):
            payload: dict[str, Any] = {"model": model, "input": batch, "encoding_format": "float"}
            if model.startswith("text-embedding-3"):
                payload["dimensions"] = dimensions
            call_id = new_llm_call_id()
            started_ms = monotonic_ms()
            response = client.embeddings.create(**payload)
            response_data = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
            log_openai_embedding_usage(
                call_id=call_id,
                model=model,
                payload=payload,
                response_data=response_data,
                started_ms=started_ms,
                status="ok",
                context={
                    "purpose": purpose,
                    "ticker": request.ticker.upper(),
                    "process": "forecast_cli",
                    "batch_index": batch_index,
                    "batch_size": len(batch),
                },
                api_key=getattr(client, "api_key", None),
            )
            batch_vectors = _extract_vectors(response_data)
            vectors.extend(batch_vectors)
            usage = response_data.get("usage", {}) if isinstance(response_data.get("usage"), dict) else {}
            batch_statuses.append(
                {
                    "batch_index": batch_index,
                    "input_count": len(batch),
                    "vector_count": len(batch_vectors),
                    "usage": usage,
                }
            )
        return {
            "vectors": vectors,
        }["vectors"], {
            "status": "executed",
            "model": model,
            "dimensions": dimensions,
            "vector_count": len(vectors),
            "batch_count": len(batch_statuses),
            "batches": batch_statuses,
        }
    except Exception as exc:
        log_openai_embedding_usage(
            call_id=new_llm_call_id(),
            model=model,
            payload={"model": model, "input_count": len(texts), "encoding_format": "float"},
            started_ms=monotonic_ms(),
            status="error",
            error=str(exc),
            context={"purpose": purpose, "ticker": request.ticker.upper(), "process": "forecast_cli"},
            api_key=api_key,
        )
        return [], {"status": "failed", "model": model, "reason": f"{type(exc).__name__}: {exc}"}


def _embedding_batches(texts: list[str]) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    max_batch_chars = 200_000
    max_batch_items = 128
    for text in texts:
        clean = str(text or "")
        would_exceed = current and (
            current_chars + len(clean) > max_batch_chars or len(current) >= max_batch_items
        )
        if would_exceed:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(clean)
        current_chars += len(clean)
    if current:
        batches.append(current)
    return batches


def _extract_document_text(path: Path) -> tuple[str, dict[str, Any]]:
    if path.suffix.lower() in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="replace"), {"method": "plain_text", "bytes": path.stat().st_size}
    if path.suffix.lower() == ".pdf":
        try:
            import pdfplumber

            texts = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        texts.append(page_text)
            return "\n\n".join(texts), {"method": "pdfplumber", "pages_with_text": len(texts), "bytes": path.stat().st_size}
        except Exception as exc:
            return "", {"method": "pdfplumber", "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return "", {"method": "unsupported"}


def _chunk_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for document in documents:
        text = _clean_text(document["text"])
        step = max(1, MAX_CHARS_PER_CHUNK - CHUNK_OVERLAP_CHARS)
        for chunk_index, start in enumerate(range(0, max(1, len(text)), step)):
            chunk_text = text[start : start + MAX_CHARS_PER_CHUNK].strip()
            if len(chunk_text) < 80:
                continue
            chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            chunks.append(
                {
                    "chunk_id": f"{document['source_id']}:{chunk_index}:{chunk_hash[:12]}",
                    "source_id": document["source_id"],
                    "source_title": document["source_title"],
                    "source_path": document["source_path"],
                    "source_type": document["source_type"],
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "text_sha256": chunk_hash,
                }
            )
    return chunks


def _lexical_rank(query: str, chunks: list[dict[str, Any]]) -> dict[int, float]:
    query_terms = _terms(query)
    if not query_terms:
        return {}
    return {idx: _lexical_score(query_terms, str(chunk.get("text") or "")) for idx, chunk in enumerate(chunks)}


def _chunk_result(score: float, semantic: float | None, lexical: float, idx: int, chunk: dict[str, Any]) -> dict[str, Any]:
    text = str(chunk.get("text") or "")
    return {
        "rank_index": idx,
        "chunk_id": chunk.get("chunk_id"),
        "source_id": chunk.get("source_id"),
        "source_title": chunk.get("source_title"),
        "source_path": chunk.get("source_path"),
        "source_type": chunk.get("source_type"),
        "chunk_index": chunk.get("chunk_index"),
        "score": round(float(score), 6),
        "semantic_score": round(float(semantic), 6) if semantic is not None else None,
        "lexical_score": round(float(lexical), 6),
        "text_excerpt": text[:1400],
        "text_sha256": chunk.get("text_sha256"),
    }


def _manifest(
    request: StrategyKnowledgeRequest,
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    corpus_hash: str,
    embedding_status: dict[str, Any],
    *,
    status: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "version": STRATEGY_KNOWLEDGE_VERSION,
        "backend": "faiss",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "corpus_hash": corpus_hash,
        "embedding_model": _embedding_model(request),
        "embedding_dimensions": int(request.embedding_dimensions or DEFAULT_EMBEDDING_DIMENSIONS),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "embedding_status": embedding_status,
        "documents": [
            {
                "source_id": doc["source_id"],
                "source_title": doc["source_title"],
                "source_type": doc["source_type"],
                "source_path": doc["source_path"],
                "text_sha256": doc["text_sha256"],
                "extraction": doc.get("extraction", {}),
            }
            for doc in documents
        ],
    }


def _faiss() -> Any:
    try:
        import faiss
    except Exception as exc:
        raise RuntimeError("faiss-cpu is required for strategy knowledge vector indexing.") from exc
    return faiss


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    return matrix / norms


def _extract_vectors(response_data: dict[str, Any]) -> list[list[float]]:
    data = response_data.get("data") or []
    ordered = sorted((item for item in data if isinstance(item, dict)), key=lambda item: int(item.get("index", 0)))
    return [[float(value) for value in item.get("embedding", [])] for item in ordered]


def _forecast_query_parts(forecasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "horizon_days": item.get("horizon_days"),
            "expected_direction": item.get("expected_direction"),
            "expected_return": item.get("expected_return"),
            "directional_confidence": item.get("directional_confidence"),
            "selected_model": item.get("selected_model"),
        }
        for item in forecasts[:5]
        if isinstance(item, dict)
    ]


def _context_status(request: StrategyKnowledgeRequest, status: str, **extra: Any) -> dict[str, Any]:
    return {"status": status, "version": STRATEGY_KNOWLEDGE_VERSION, "created_at_utc": datetime.now(UTC).isoformat(), "request": request.to_dict(), **extra}


def _source_id(path: Path) -> str:
    return hashlib.sha256(str(path.expanduser().resolve()).encode("utf-8")).hexdigest()[:16]


def _source_title(path: Path, text: str) -> str:
    for line in text.splitlines()[:8]:
        clean = line.strip("# ").strip()
        if clean:
            return clean[:160]
    return path.stem[:160]


def _source_type(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "book_pdf"
    if "strategy" in str(path).lower() or "can_slim" in str(path).lower():
        return "strategy_note"
    return "document"


def _documents_hash(documents: list[dict[str, Any]]) -> str:
    payload = [{key: doc.get(key) for key in ("source_id", "source_path", "text_sha256")} for doc in sorted(documents, key=lambda item: item["source_id"])]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _embedding_model(request: StrategyKnowledgeRequest) -> str:
    return request.embedding_model or os.environ.get("OPENAI_EMBEDDING_MODEL") or _read_env_value(request.llm_env_file, "OPENAI_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL


def _manifest_path(index_path: Path) -> Path:
    return index_path.with_suffix(".manifest.json")


def _chunks_path(index_path: Path) -> Path:
    return index_path.with_suffix(".chunks.jsonl")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _cosine(a: list[float], b: list[float]) -> float | None:
    if not a or not b:
        return None
    width = min(len(a), len(b))
    av = np.asarray(a[:width], dtype=float)
    bv = np.asarray(b[:width], dtype=float)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-12:
        return None
    return float(np.dot(av, bv) / denom)


def _terms(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]{3,}", text.lower()) if token not in _STOPWORDS}


def _lexical_score(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    text_terms = _terms(text)
    if not text_terms:
        return 0.0
    return len(query_terms & text_terms) / max(1, len(query_terms))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = _clean_text(value)
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result


def _clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


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


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "have",
    "has",
    "are",
    "was",
    "were",
    "not",
    "but",
    "you",
    "your",
    "price",
    "ticker",
}
