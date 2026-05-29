from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import numpy as np
import pandas as pd

from market_forecasting_engine.data_store import MarketDataStore, frame_sha256, request_key
from market_forecasting_engine.openai_models import DEFAULT_OPENAI_MODEL, DEFAULT_REASONING_EFFORT
from market_forecasting_engine.openai_responses import call_response


POSITIVE_WORDS = {
    "beat",
    "beats",
    "bullish",
    "buy",
    "growth",
    "higher",
    "improve",
    "improved",
    "outperform",
    "positive",
    "profit",
    "rally",
    "record",
    "upgrade",
    "upside",
    "strong",
}
NEGATIVE_WORDS = {
    "bearish",
    "cut",
    "downgrade",
    "fall",
    "falls",
    "fraud",
    "lawsuit",
    "loss",
    "miss",
    "misses",
    "negative",
    "probe",
    "risk",
    "sell",
    "slump",
    "weak",
}


@dataclass(frozen=True)
class AlternativeNewsRequest:
    ticker: str
    provider: str = "yahoo_rss"
    lookback_days: int = 30
    max_items: int = 40
    sentiment_mode: str = "lexicon"
    llm_model: str | None = None
    llm_reasoning_effort: str = DEFAULT_REASONING_EFFORT
    llm_env_file: str | None = None
    llm_timeout_seconds: int = 30

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ticker"] = self.ticker.upper()
        return data


def collect_alternative_news_features(
    request: AlternativeNewsRequest,
    target_index: pd.DatetimeIndex,
    store: MarketDataStore | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download/scrape news and convert it into point-in-time sentiment features."""

    articles, provider_metadata = fetch_news_articles(request)
    scored_articles, sentiment_metadata = score_news_articles(articles, request)
    raw_frame = pd.DataFrame(scored_articles)
    features = aggregate_news_sentiment_features(raw_frame, target_index)
    metadata: dict[str, Any] = {
        "kind": "alternative_news_sentiment",
        "request": request.to_dict(),
        "provider": provider_metadata,
        "sentiment": sentiment_metadata,
        "article_count": int(len(raw_frame)),
        "feature_columns": [str(column) for column in features.columns],
        "raw_data_hash": frame_sha256(raw_frame) if not raw_frame.empty else None,
        "feature_data_hash": frame_sha256(features) if not features.empty else None,
        "registry": alternative_data_registry_entry(request, raw_frame, features, provider_metadata, sentiment_metadata),
    }
    if store is not None:
        key = request_key({"kind": "alternative_news_sentiment", **request.to_dict()})
        artifacts: dict[str, Any] = {}
        if not raw_frame.empty:
            artifacts["raw_articles"] = store.write_frame(
                "raw",
                "alternative_news",
                request.ticker,
                key,
                _storable_articles(raw_frame),
            ).to_dict()
        if not features.empty:
            artifacts["features"] = store.write_frame(
                "features",
                "alternative_news",
                request.ticker,
                key,
                features,
            ).to_dict()
        metadata["artifacts"] = artifacts
        store.write_json("metadata", "alternative_news", request.ticker, f"{key}_metadata", metadata)
    return features, metadata


def fetch_news_articles(request: AlternativeNewsRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    provider = request.provider.lower().replace("-", "_")
    if provider == "yahoo_rss":
        return fetch_yahoo_finance_rss(request)
    if provider in {"yahoo_news", "yfinance_news"}:
        return fetch_yfinance_news(request)
    if provider in {"openai_web", "llm_web"}:
        return fetch_openai_web_news_and_social(request)
    raise ValueError(f"Unsupported alternative news provider `{request.provider}`.")


def fetch_yahoo_finance_rss(request: AlternativeNewsRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ticker = request.ticker.upper()
    encoded = urllib.parse.quote(ticker)
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={encoded}&region=US&lang=en-US"
    cutoff = datetime.now(UTC) - timedelta(days=max(1, request.lookback_days))
    metadata: dict[str, Any] = {
        "name": "yahoo_rss",
        "url": url,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "status": "ok",
    }
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            payload = response.read()
    except Exception as exc:
        fallback_articles, fallback_metadata = fetch_yfinance_news(request)
        metadata.update(
            {
                "status": "fallback",
                "error": str(exc),
                "fallback_provider": fallback_metadata,
                "article_count": int(len(fallback_articles)),
            }
        )
        return fallback_articles, metadata

    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        metadata.update({"status": "failed", "error": f"RSS parse failed: {exc}"})
        return [], metadata

    articles = []
    for item in root.findall(".//item"):
        title = _node_text(item, "title")
        link = _node_text(item, "link")
        description = _node_text(item, "description")
        published_at = _parse_rss_datetime(_node_text(item, "pubDate"))
        if published_at is not None and published_at < cutoff:
            continue
        if not title and not description:
            continue
        articles.append(
            {
                "published_at": published_at.isoformat() if published_at is not None else None,
                "date": str(published_at.date()) if published_at is not None else None,
                "ticker": ticker,
                "source": _source_from_link(link),
                "source_category": "news",
                "title": title,
                "summary": _strip_html(description),
                "url": link,
                "retrieval_method": "rss_scrape",
            }
        )
        if len(articles) >= max(1, request.max_items):
            break
    metadata["article_count"] = int(len(articles))
    return articles, metadata


def fetch_yfinance_news(request: AlternativeNewsRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ticker = request.ticker.upper()
    cutoff = datetime.now(UTC) - timedelta(days=max(1, request.lookback_days))
    metadata: dict[str, Any] = {
        "name": "yahoo_news",
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "status": "ok",
    }
    try:
        import yfinance as yf

        raw_items = yf.Ticker(ticker).news or []
    except Exception as exc:
        metadata.update({"status": "failed", "error": str(exc)})
        return [], metadata

    articles = []
    for raw in raw_items[: max(1, request.max_items)]:
        raw_dict = raw if isinstance(raw, dict) else {}
        content = raw_dict.get("content", raw_dict)
        if not isinstance(content, dict):
            continue
        published_at = _parse_any_datetime(
            content.get("pubDate") or content.get("displayTime") or raw_dict.get("providerPublishTime")
        )
        if published_at is not None and published_at < cutoff:
            continue
        title = str(content.get("title") or raw_dict.get("title") or "").strip()
        summary = _strip_html(str(content.get("summary") or content.get("description") or raw_dict.get("summary") or ""))
        if not title and not summary:
            continue
        provider = content.get("provider", {}) if isinstance(content.get("provider"), dict) else {}
        url = _content_url(content)
        articles.append(
            {
                "published_at": published_at.isoformat() if published_at is not None else None,
                "date": str(published_at.date()) if published_at is not None else None,
                "ticker": ticker,
                "source": str(provider.get("displayName") or provider.get("sourceId") or _source_from_link(url)),
                "source_category": _source_category(content),
                "title": title,
                "summary": summary,
                "url": url,
                "retrieval_method": "yfinance_news",
            }
        )
        if len(articles) >= max(1, request.max_items):
            break
    metadata["article_count"] = int(len(articles))
    return articles, metadata


def fetch_openai_web_news_and_social(request: AlternativeNewsRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    api_key = os.environ.get("OPENAI_API_KEY") or _read_env_value(request.llm_env_file, "OPENAI_API_KEY")
    model = _resolve_openai_model(request.llm_model, request.llm_env_file)
    effective_max_items = max(1, min(int(request.max_items), 12))
    metadata: dict[str, Any] = {
        "name": "openai_web",
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "status": "skipped",
        "model": model,
        "requested_max_items": int(request.max_items),
        "effective_max_items": int(effective_max_items),
    }
    if not api_key:
        metadata["reason"] = "OPENAI_API_KEY is not available."
        return [], metadata
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=float(request.llm_timeout_seconds))
        _, _, parsed = call_response(
            client=client,
            model=model,
            system_message=(
                "Find recent factual market-moving news and public social/forum discussion for a stock. "
                "Return only structured records. Prefer primary or reputable sources. Do not invent URLs or dates."
            ),
            user_message="{{ item.request }}",
            json_schema=_openai_web_news_schema(),
            reasoning_effort=request.llm_reasoning_effort,
            item={
                "request": json.dumps(
                    {
                        "ticker": request.ticker.upper(),
                        "lookback_days": request.lookback_days,
                        "max_items": effective_max_items,
                        "required_fields": [
                            "published_at",
                            "title",
                            "summary",
                            "url",
                            "source",
                            "source_category",
                        ],
                        "allowed_source_categories": ["news", "filing", "forum", "social", "company", "other"],
                    },
                    sort_keys=True,
                )
            },
            tools=[{"type": "web_search", "search_context_size": "medium"}],
        )
        rows = parsed.get("articles", [])
        articles = []
        cutoff = datetime.now(UTC) - timedelta(days=max(1, request.lookback_days))
        for row in rows[:effective_max_items]:
            published_at = _parse_any_datetime(row.get("published_at") or row.get("date"))
            if published_at is not None and published_at < cutoff:
                continue
            title = str(row.get("title") or "").strip()
            summary = str(row.get("summary") or "").strip()
            if not title and not summary:
                continue
            articles.append(
                {
                    "published_at": published_at.isoformat() if published_at is not None else None,
                    "date": str(published_at.date()) if published_at is not None else None,
                    "ticker": request.ticker.upper(),
                    "source": str(row.get("source") or _source_from_link(str(row.get("url") or ""))),
                    "source_category": str(row.get("source_category") or "news").lower(),
                    "title": title,
                    "summary": summary,
                    "url": str(row.get("url") or ""),
                    "retrieval_method": "openai_web_search",
                }
            )
        metadata.update({"status": "ok", "article_count": int(len(articles))})
        return articles, metadata
    except Exception as exc:
        metadata.update({"status": "failed", "error": str(exc)})
        return [], metadata


def _openai_web_news_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "alternative_news_articles",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "articles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "published_at": {"type": "string"},
                            "title": {"type": "string"},
                            "summary": {"type": "string"},
                            "url": {"type": "string"},
                            "source": {"type": "string"},
                            "source_category": {
                                "type": "string",
                                "enum": ["news", "filing", "forum", "social", "company", "other"],
                            },
                        },
                        "required": ["published_at", "title", "summary", "url", "source", "source_category"],
                    },
                }
            },
            "required": ["articles"],
        },
    }


def _loads_json_object(payload: str) -> dict[str, Any]:
    text = payload.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}


def score_news_articles(
    articles: list[dict[str, Any]],
    request: AlternativeNewsRequest,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mode = request.sentiment_mode.lower()
    lexicon_scored = [_score_article_lexicon(article) for article in articles]
    if mode == "lexicon" or not lexicon_scored:
        return lexicon_scored, {"mode": "lexicon", "status": "executed", "llm_status": "not_requested"}
    llm_scored, llm_metadata = _score_articles_with_openai(lexicon_scored, request)
    if mode == "llm":
        return llm_scored if llm_metadata["status"] == "executed" else lexicon_scored, {
            "mode": "llm",
            **llm_metadata,
        }
    if mode == "hybrid":
        return _blend_lexicon_and_llm(lexicon_scored, llm_scored, llm_metadata), {
            "mode": "hybrid",
            "status": "executed",
            "llm_status": llm_metadata["status"],
            "llm_reason": llm_metadata.get("reason"),
            "model": llm_metadata.get("model"),
        }
    raise ValueError("sentiment_mode must be one of: lexicon, llm, hybrid.")


def aggregate_news_sentiment_features(
    articles: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    windows: tuple[int, ...] = (1, 3, 7, 14),
) -> pd.DataFrame:
    index = pd.DatetimeIndex(target_index)
    output = pd.DataFrame(index=index)
    if articles.empty:
        for window in windows:
            output[f"alt_news_sentiment_{window}d"] = 0.0
            output[f"alt_news_volume_{window}d"] = 0.0
            output[f"alt_news_positive_share_{window}d"] = 0.0
            output[f"alt_news_negative_share_{window}d"] = 0.0
        output["alt_news_relevance_7d"] = 0.0
        return output

    frame = articles.copy()
    frame["date"] = pd.to_datetime(frame["date"].fillna(frame["published_at"]), errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date"])
    daily = pd.DataFrame(index=pd.date_range(index.min().normalize(), index.max().normalize(), freq="D"))
    grouped = frame.groupby("date")
    daily["sentiment"] = grouped["sentiment_score"].mean()
    daily["volume"] = grouped.size()
    daily["positive"] = grouped.apply(lambda rows: float((rows["sentiment_score"] > 0.10).mean()), include_groups=False)
    daily["negative"] = grouped.apply(lambda rows: float((rows["sentiment_score"] < -0.10).mean()), include_groups=False)
    daily["relevance"] = grouped["relevance_score"].mean()
    daily = daily.fillna({"sentiment": 0.0, "volume": 0.0, "positive": 0.0, "negative": 0.0, "relevance": 0.0})
    for window in windows:
        output[f"alt_news_sentiment_{window}d"] = daily["sentiment"].rolling(window, min_periods=1).mean().reindex(index.normalize()).to_numpy()
        output[f"alt_news_volume_{window}d"] = daily["volume"].rolling(window, min_periods=1).sum().reindex(index.normalize()).to_numpy()
        output[f"alt_news_positive_share_{window}d"] = daily["positive"].rolling(window, min_periods=1).mean().reindex(index.normalize()).to_numpy()
        output[f"alt_news_negative_share_{window}d"] = daily["negative"].rolling(window, min_periods=1).mean().reindex(index.normalize()).to_numpy()
    output["alt_news_relevance_7d"] = daily["relevance"].rolling(7, min_periods=1).mean().reindex(index.normalize()).to_numpy()
    return output.fillna(0.0)


def alternative_data_registry_entry(
    request: AlternativeNewsRequest,
    articles: pd.DataFrame,
    features: pd.DataFrame,
    provider_metadata: dict[str, Any],
    sentiment_metadata: dict[str, Any],
) -> dict[str, Any]:
    article_dates = pd.to_datetime(articles["date"], errors="coerce") if "date" in articles else pd.Series(dtype="datetime64[ns]")
    return {
        "name": "news_sentiment",
        "source_type": "individuals_and_publishers",
        "provider": request.provider,
        "collection_method": "download_or_scrape",
        "sentiment_method": sentiment_metadata.get("mode"),
        "provider_status": provider_metadata.get("status"),
        "article_count": int(len(articles)),
        "feature_count": int(len(features.columns)),
        "feature_columns": [str(column) for column in features.columns],
        "history_start": str(article_dates.min().date()) if article_dates.notna().any() else None,
        "history_end": str(article_dates.max().date()) if article_dates.notna().any() else None,
        "point_in_time_safe": True,
        "availability_lag": "article_published_at",
        "expected_signal_type": "sentiment_attention_relevance",
        "quality": {
            "history_days": int((article_dates.max() - article_dates.min()).days) + 1 if article_dates.notna().sum() >= 2 else 0,
            "missing_date_rows": int(article_dates.isna().sum()) if len(article_dates) else 0,
            "llm_status": sentiment_metadata.get("llm_status", sentiment_metadata.get("status")),
            "raw_data_hash": frame_sha256(articles) if not articles.empty else None,
        },
    }


def _score_article_lexicon(article: dict[str, Any]) -> dict[str, Any]:
    text = f"{article.get('title', '')} {article.get('summary', '')}".lower()
    tokens = re.findall(r"[a-z][a-z0-9_+-]*", text)
    if not tokens:
        score = 0.0
    else:
        positive = sum(token in POSITIVE_WORDS for token in tokens)
        negative = sum(token in NEGATIVE_WORDS for token in tokens)
        score = (positive - negative) / max(positive + negative, 3)
    ticker = str(article.get("ticker", "")).lower()
    relevance = 1.0 if ticker and ticker in text else 0.60
    output = dict(article)
    output.update(
        {
            "sentiment_score": float(max(-1.0, min(1.0, score))),
            "sentiment_label": _sentiment_label(score),
            "relevance_score": relevance,
            "sentiment_method": "lexicon",
        }
    )
    return output


def _score_articles_with_openai(
    articles: list[dict[str, Any]],
    request: AlternativeNewsRequest,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    api_key = os.environ.get("OPENAI_API_KEY") or _read_env_value(request.llm_env_file, "OPENAI_API_KEY")
    model = _resolve_openai_model(request.llm_model, request.llm_env_file)
    if not api_key:
        return articles, {"status": "skipped", "model": model, "reason": "OPENAI_API_KEY is not available."}
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=float(request.llm_timeout_seconds))
        payload = [
            {
                "id": idx,
                "title": item.get("title"),
                "summary": item.get("summary"),
                "source": item.get("source"),
            }
            for idx, item in enumerate(articles[: request.max_items])
        ]
        _, _, parsed = call_response(
            client=client,
            model=model,
            system_message=(
                "Classify financial-news sentiment for a stock forecast pipeline. "
                "Return structured article scores only."
            ),
            user_message="{{ item.request }}",
            json_schema=_openai_sentiment_schema(),
            reasoning_effort=request.llm_reasoning_effort,
            item={
                "request": json.dumps(
                    {"ticker": request.ticker.upper(), "articles": payload},
                    sort_keys=True,
                )
            },
        )
        scored_by_id = {int(item["id"]): item for item in parsed.get("articles", []) if "id" in item}
        output = []
        for idx, article in enumerate(articles):
            scored = scored_by_id.get(idx, {})
            merged = dict(article)
            if scored:
                merged.update(
                    {
                        "llm_sentiment_score": _clip_score(scored.get("sentiment_score")),
                        "llm_sentiment_label": str(scored.get("sentiment_label") or article.get("sentiment_label")),
                        "llm_relevance_score": _clip_unit(scored.get("relevance_score")),
                        "catalyst_type": scored.get("catalyst_type"),
                        "sentiment_method": "llm",
                    }
                )
                merged["sentiment_score"] = merged["llm_sentiment_score"]
                merged["sentiment_label"] = merged["llm_sentiment_label"]
                merged["relevance_score"] = merged["llm_relevance_score"]
            output.append(merged)
        return output, {"status": "executed", "model": model}
    except Exception as exc:
        return articles, {"status": "failed", "model": model, "reason": str(exc)}


def _openai_sentiment_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "news_sentiment_scores",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "articles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "integer"},
                            "sentiment_score": {"type": "number"},
                            "sentiment_label": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                            "relevance_score": {"type": "number"},
                            "catalyst_type": {"type": "string"},
                        },
                        "required": [
                            "id",
                            "sentiment_score",
                            "sentiment_label",
                            "relevance_score",
                            "catalyst_type",
                        ],
                    },
                }
            },
            "required": ["articles"],
        },
    }


def _resolve_openai_model(model: str | None, env_file: str | None) -> str:
    return model or os.environ.get("OPENAI_MODEL") or _read_env_value(env_file, "OPENAI_MODEL") or DEFAULT_OPENAI_MODEL


def _blend_lexicon_and_llm(
    lexicon_scored: list[dict[str, Any]],
    llm_scored: list[dict[str, Any]],
    llm_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    if llm_metadata.get("status") != "executed":
        return lexicon_scored
    output = []
    for lexical, llm in zip(lexicon_scored, llm_scored, strict=False):
        blended = dict(lexical)
        if "llm_sentiment_score" in llm:
            score = 0.35 * float(lexical.get("sentiment_score", 0.0)) + 0.65 * float(llm["llm_sentiment_score"])
            blended.update(llm)
            blended["sentiment_score"] = float(max(-1.0, min(1.0, score)))
            blended["sentiment_label"] = _sentiment_label(blended["sentiment_score"])
            blended["sentiment_method"] = "hybrid_lexicon_llm"
        output.append(blended)
    return output


def _node_text(item: ET.Element, tag: str) -> str:
    node = item.find(tag)
    return (node.text or "").strip() if node is not None else ""


def _parse_rss_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except Exception:
        return None


def _parse_any_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = pd.to_datetime(text, errors="coerce", utc=True)
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime().astimezone(UTC)
    except Exception:
        return _parse_rss_datetime(text)


def _source_from_link(link: str) -> str:
    try:
        host = urllib.parse.urlparse(link).netloc
        return host.replace("www.", "") or "unknown"
    except Exception:
        return "unknown"


def _content_url(content: dict[str, Any]) -> str:
    for key in ("canonicalUrl", "clickThroughUrl"):
        value = content.get(key)
        if isinstance(value, dict) and value.get("url"):
            return str(value["url"])
    return ""


def _source_category(content: dict[str, Any]) -> str:
    provider = content.get("provider", {}) if isinstance(content.get("provider"), dict) else {}
    source = str(provider.get("displayName") or provider.get("sourceId") or "").lower()
    content_type = str(content.get("contentType") or "").lower()
    if "stocktwits" in source:
        return "social"
    if content_type == "video":
        return "news"
    return "news"


def _strip_html(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value or "")).strip()


def _sentiment_label(score: float) -> str:
    if score > 0.10:
        return "positive"
    if score < -0.10:
        return "negative"
    return "neutral"


def _clip_score(value: Any) -> float:
    try:
        return float(max(-1.0, min(1.0, float(value))))
    except Exception:
        return 0.0


def _clip_unit(value: Any) -> float:
    try:
        return float(max(0.0, min(1.0, float(value))))
    except Exception:
        return 0.5


def _read_env_value(env_file: str | None, key: str) -> str | None:
    if not env_file:
        return None
    try:
        with open(env_file, encoding="utf-8") as handle:
            for line in handle:
                if line.strip().startswith(f"{key}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def _storable_articles(articles: pd.DataFrame) -> pd.DataFrame:
    output = articles.copy()
    if "published_at" in output.columns:
        output.index = pd.to_datetime(output["published_at"], errors="coerce")
    return output
