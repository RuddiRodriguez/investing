from __future__ import annotations

import json
import os
import textwrap
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from market_forecasting_engine.data_store import frame_sha256
from market_forecasting_engine.openai_models import DEFAULT_OPENAI_MODEL, DEFAULT_REASONING_EFFORT
from market_forecasting_engine.openai_responses import call_response


PROMPT_VERSION = "chapter_15_llm_topic_modeling_v1"

FINANCIAL_TOPIC_TAXONOMY: dict[str, str] = {
    "earnings_guidance": "Earnings, revenue, margin, profit warnings, outlook, or management guidance.",
    "regulation_policy": "Regulation, export controls, antitrust, government policy, sanctions, listing rules, or compliance.",
    "product_technology": "Products, AI, chips, software, R&D, platform changes, launches, patents, or technical capability.",
    "macro_rates_fx": "Interest rates, inflation, currencies, commodities, macro cycle, recession, or central-bank policy.",
    "analyst_rating": "Analyst upgrades, downgrades, estimates, target prices, broker commentary, or institutional views.",
    "legal_litigation": "Lawsuits, investigations, settlements, accounting issues, fraud allegations, or legal exposure.",
    "mna_strategy": "Mergers, acquisitions, divestitures, partnerships, restructuring, or strategic repositioning.",
    "supply_chain": "Suppliers, production capacity, inventory, logistics, shortages, order books, or delivery constraints.",
    "competition_market_share": "Competitive positioning, market share, pricing pressure, customers, demand, or industry rivalry.",
    "capital_returns_balance_sheet": "Buybacks, dividends, debt, cash, credit rating, dilution, financing, or balance-sheet strength.",
    "market_technical_flow": "Price action, ETF/index flow, options flow, trading volume, short interest, or market microstructure.",
    "other": "Relevant market-moving topic that does not fit another taxonomy category.",
}

SENTIMENT_LABELS = ("POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED", "INAPPLICABLE", "UNKNOWN")


@dataclass(frozen=True)
class Chapter15TopicRequest:
    ticker: str
    max_articles: int = 24
    max_topics_per_article: int = 3
    llm_model: str | None = None
    llm_reasoning_effort: str = DEFAULT_REASONING_EFFORT
    llm_env_file: str | None = None
    llm_timeout_seconds: int = 30

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ticker"] = self.ticker.upper()
        return data


class Chapter15TopicPrompt:
    MODEL_NAME = DEFAULT_OPENAI_MODEL
    SYSTEM_MESSAGE = textwrap.dedent(
        f"""
        # Role: Financial Topic Modeling Researcher

        # Objective
        Analyze recent financial documents for one tradable ticker. Extract specific document topics, assign each topic
        to exactly one controlled financial taxonomy category, and classify sentiment per assigned topic.

        # Guidelines
        - Base every extracted topic only on the supplied article title, summary, source, and date.
        - Do not infer facts that are absent from the article text.
        - Extract specific topics, not broad themes. A topic should usually be 1 to 5 words.
        - Assign each topic to the single most semantically relevant taxonomy id.
        - If a topic does not fit the taxonomy, assign it to "other".
        - Sentiment is topic-specific and must reflect likely impact for the ticker, not the tone of the writing.
        - Use MIXED when a topic contains both material positive and negative implications.
        - Use INAPPLICABLE when the article has no financial implication for the ticker.
        - Keep topic ids stable by using the provided taxonomy ids exactly.
        - Return structured JSON only.

        # Prompt Version
        {PROMPT_VERSION}
        """
    ).strip()
    USER_MESSAGE = textwrap.dedent(
        """
        Ticker:
        ```
        {ticker}
        ```

        Taxonomy:
        ```
        {taxonomy}
        ```

        Articles:
        ```
        {articles}
        ```
        """
    ).strip()


def extract_llm_topics_for_articles(
    articles: list[dict[str, Any]],
    request: Chapter15TopicRequest,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not articles:
        return [], _metadata(request, "skipped", reason="no_articles")
    api_key = os.environ.get("OPENAI_API_KEY") or _read_env_value(request.llm_env_file, "OPENAI_API_KEY")
    model = request.llm_model or os.environ.get("OPENAI_MODEL") or _read_env_value(request.llm_env_file, "OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    if not api_key:
        return [], _metadata(request, "skipped", model=model, reason="OPENAI_API_KEY is not available.")
    payload_articles = _article_payload(articles[: max(1, int(request.max_articles))])
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=float(request.llm_timeout_seconds))
        _, response_data, parsed = call_response(
            client=client,
            model=model,
            system_message=Chapter15TopicPrompt.SYSTEM_MESSAGE,
            user_message=Chapter15TopicPrompt.USER_MESSAGE,
            json_schema=chapter_15_topic_schema(),
            reasoning_effort=request.llm_reasoning_effort,
            item={
                "ticker": request.ticker.upper(),
                "taxonomy": json.dumps(FINANCIAL_TOPIC_TAXONOMY, sort_keys=True),
                "articles": json.dumps(payload_articles, sort_keys=True),
            },
            usage_context={"purpose": "chapter_15_topic_extraction", "ticker": request.ticker.upper(), "process": "alternative_data"},
        )
        topic_rows = _normalize_topic_response(parsed, articles)
        return topic_rows, _metadata(
            request,
            "executed",
            model=model,
            article_count=len(payload_articles),
            topic_count=len(topic_rows),
            response_id=response_data.get("id"),
        )
    except Exception as exc:
        return [], _metadata(request, "failed", model=model, reason=f"{type(exc).__name__}: {exc}")


def aggregate_topic_features(
    topic_rows: list[dict[str, Any]] | pd.DataFrame,
    target_index: pd.DatetimeIndex,
    *,
    windows: tuple[int, ...] = (1, 3, 7, 14),
) -> pd.DataFrame:
    index = pd.DatetimeIndex(target_index)
    topic_ids = list(FINANCIAL_TOPIC_TAXONOMY)
    initial_columns: dict[str, float] = {}
    for topic_id in topic_ids:
        for window in windows:
            initial_columns[f"alt_topic_{topic_id}_{window}d"] = 0.0
            initial_columns[f"alt_topic_negative_{topic_id}_{window}d"] = 0.0
            initial_columns[f"alt_topic_positive_{topic_id}_{window}d"] = 0.0
    initial_columns["alt_topic_entropy_7d"] = 0.0
    initial_columns["alt_topic_shift_7d"] = 0.0
    output = pd.DataFrame(initial_columns, index=index)
    if isinstance(topic_rows, pd.DataFrame):
        frame = topic_rows.copy()
    else:
        frame = pd.DataFrame(topic_rows)
    if frame.empty:
        return output
    frame["date"] = pd.to_datetime(frame.get("date", frame.get("published_at")), errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        return output
    frame["taxonomy_id"] = frame["taxonomy_id"].map(_safe_topic_id)
    frame["relevance_score"] = pd.to_numeric(frame.get("relevance_score", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    frame["sentiment_label"] = frame.get("sentiment_label", "UNKNOWN").fillna("UNKNOWN").astype(str).str.upper()
    daily_index = pd.date_range(index.min().normalize(), index.max().normalize(), freq="D")
    daily_topic = pd.DataFrame(0.0, index=daily_index, columns=topic_ids)
    daily_negative = pd.DataFrame(0.0, index=daily_index, columns=topic_ids)
    daily_positive = pd.DataFrame(0.0, index=daily_index, columns=topic_ids)
    for (date, topic_id), rows in frame.groupby(["date", "taxonomy_id"]):
        if topic_id not in daily_topic.columns or date not in daily_topic.index:
            continue
        weight = float(rows["relevance_score"].sum())
        daily_topic.loc[date, topic_id] = weight
        daily_negative.loc[date, topic_id] = float(rows.loc[rows["sentiment_label"] == "NEGATIVE", "relevance_score"].sum())
        daily_positive.loc[date, topic_id] = float(rows.loc[rows["sentiment_label"] == "POSITIVE", "relevance_score"].sum())
    for topic_id in topic_ids:
        for window in windows:
            output[f"alt_topic_{topic_id}_{window}d"] = daily_topic[topic_id].rolling(window, min_periods=1).sum().reindex(index.normalize()).to_numpy()
            output[f"alt_topic_negative_{topic_id}_{window}d"] = daily_negative[topic_id].rolling(window, min_periods=1).sum().reindex(index.normalize()).to_numpy()
            output[f"alt_topic_positive_{topic_id}_{window}d"] = daily_positive[topic_id].rolling(window, min_periods=1).sum().reindex(index.normalize()).to_numpy()
    rolling = daily_topic.rolling(7, min_periods=1).sum()
    proportions = rolling.div(rolling.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    entropy = -(proportions.where(proportions > 0.0, 1.0).map(np.log) * proportions).sum(axis=1)
    shift = proportions.diff().abs().sum(axis=1).fillna(0.0)
    output["alt_topic_entropy_7d"] = entropy.reindex(index.normalize()).to_numpy()
    output["alt_topic_shift_7d"] = shift.reindex(index.normalize()).to_numpy()
    return output.fillna(0.0)


def chapter_15_registry_entry(
    request: Chapter15TopicRequest,
    topics: pd.DataFrame,
    features: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    topic_dates = pd.to_datetime(topics["date"], errors="coerce") if "date" in topics else pd.Series(dtype="datetime64[ns]")
    return {
        "name": "chapter_15_llm_topic_modeling",
        "source_type": "financial_text_news_filings_social",
        "chapter": "jansen_chapter_15",
        "prompt_version": PROMPT_VERSION,
        "taxonomy_version": "financial_topic_taxonomy_v1",
        "model": metadata.get("model"),
        "status": metadata.get("status"),
        "topic_count": int(len(topics)),
        "feature_count": int(len(features.columns)),
        "feature_columns": [str(column) for column in features.columns],
        "history_start": str(topic_dates.min().date()) if topic_dates.notna().any() else None,
        "history_end": str(topic_dates.max().date()) if topic_dates.notna().any() else None,
        "point_in_time_safe": True,
        "availability_lag": "article_published_at",
        "expected_signal_type": "topic_attention_topic_sentiment_topic_shift",
        "quality": {
            "raw_topic_hash": frame_sha256(topics) if not topics.empty else None,
            "feature_data_hash": frame_sha256(features) if not features.empty else None,
            "llm_reason": metadata.get("reason"),
        },
    }


def chapter_15_topic_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "chapter_15_financial_topics",
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
                            "classification_reasoning": {"type": "string"},
                            "topics": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "topic": {"type": "string"},
                                        "taxonomy_id": {"type": "string", "enum": list(FINANCIAL_TOPIC_TAXONOMY)},
                                        "sentiment_label": {"type": "string", "enum": list(SENTIMENT_LABELS)},
                                        "sentiment_score": {"type": "number"},
                                        "relevance_score": {"type": "number"},
                                        "impact_horizon": {"type": "string", "enum": ["intraday", "short_term", "medium_term", "long_term", "unknown"]},
                                        "evidence": {"type": "string"},
                                    },
                                    "required": [
                                        "topic",
                                        "taxonomy_id",
                                        "sentiment_label",
                                        "sentiment_score",
                                        "relevance_score",
                                        "impact_horizon",
                                        "evidence",
                                    ],
                                },
                            },
                        },
                        "required": ["id", "classification_reasoning", "topics"],
                    },
                }
            },
            "required": ["articles"],
        },
    }


def _article_payload(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = []
    for idx, article in enumerate(articles):
        payload.append(
            {
                "id": idx,
                "published_at": article.get("published_at") or article.get("date"),
                "title": str(article.get("title") or "")[:500],
                "summary": str(article.get("summary") or "")[:1200],
                "source": str(article.get("source") or ""),
                "source_category": str(article.get("source_category") or "news"),
                "url": str(article.get("url") or ""),
            }
        )
    return payload


def _normalize_topic_response(parsed: dict[str, Any], source_articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for article_result in parsed.get("articles", []):
        article_id = int(article_result.get("id", -1))
        if article_id < 0 or article_id >= len(source_articles):
            continue
        source = source_articles[article_id]
        for idx, topic in enumerate(article_result.get("topics", [])):
            taxonomy_id = _safe_topic_id(topic.get("taxonomy_id"))
            sentiment_label = str(topic.get("sentiment_label") or "UNKNOWN").upper()
            if sentiment_label not in SENTIMENT_LABELS:
                sentiment_label = "UNKNOWN"
            output.append(
                {
                    "article_id": article_id,
                    "topic_index": idx,
                    "ticker": str(source.get("ticker") or "").upper(),
                    "published_at": source.get("published_at"),
                    "date": source.get("date") or source.get("published_at"),
                    "title": source.get("title"),
                    "source": source.get("source"),
                    "source_category": source.get("source_category"),
                    "url": source.get("url"),
                    "topic": str(topic.get("topic") or "").strip()[:120],
                    "taxonomy_id": taxonomy_id,
                    "sentiment_label": sentiment_label,
                    "sentiment_score": _clip(float(topic.get("sentiment_score") or 0.0), -1.0, 1.0),
                    "relevance_score": _clip(float(topic.get("relevance_score") or 0.0), 0.0, 1.0),
                    "impact_horizon": str(topic.get("impact_horizon") or "unknown"),
                    "evidence": str(topic.get("evidence") or "")[:500],
                    "classification_reasoning": str(article_result.get("classification_reasoning") or "")[:500],
                    "topic_method": "llm_controlled_taxonomy",
                    "prompt_version": PROMPT_VERSION,
                }
            )
    return output


def _metadata(request: Chapter15TopicRequest, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "kind": "chapter_15_llm_topic_modeling",
        "status": status,
        "request": request.to_dict(),
        "prompt_version": PROMPT_VERSION,
        "taxonomy": FINANCIAL_TOPIC_TAXONOMY,
        "created_at_utc": datetime.now(UTC).isoformat(),
        **extra,
    }


def _safe_topic_id(value: Any) -> str:
    topic_id = str(value or "other").strip().lower()
    return topic_id if topic_id in FINANCIAL_TOPIC_TAXONOMY else "other"


def _clip(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


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
