from collections import defaultdict
from datetime import datetime, timezone

from aggregation.llm_summary import generate_ticker_signal_summary
from aggregation.repositories import (
    get_sentiment_events_for_aggregation,
    save_ticker_sentiment_signal,
)
from aggregation.schemas import SentimentEventInput, TickerSentimentSignal
from ingestion.cache import build_cache_key, cache_exists, set_cache


def get_signal_label(score: float, positive_count: int, negative_count: int) -> str:
    if positive_count > 0 and negative_count > 0 and abs(score) < 0.25:
        return "mixed"
    if score >= 0.25:
        return "bullish"
    if score <= -0.25:
        return "bearish"
    return "neutral"


def simple_summary(
    ticker: str,
    company_name: str,
    signal_score: float,
    signal_label: str,
    event_count: int,
) -> str:
    return (
        f"{company_name} ({ticker}) has a {signal_label} aggregated news sentiment "
        f"signal with score {signal_score:.2f}, based on {event_count} relevant news events."
    )


def aggregate_ticker_events(
    events: list[SentimentEventInput],
    use_llm_summary: bool = True,
) -> TickerSentimentSignal:
    first = events[0]

    weighted_sum = 0.0
    weight_total = 0.0
    confidence_sum = 0.0
    confidence_weight_total = 0.0

    positive_count = 0
    negative_count = 0
    neutral_count = 0
    mixed_count = 0
    unknown_count = 0

    strongest_positive_event_id = None
    strongest_negative_event_id = None
    strongest_positive_value = None
    strongest_negative_value = None

    for event in events:
        weight = event.magnitude * event.confidence
        weighted_sum += event.sentiment_score * weight
        weight_total += weight

        confidence_sum += event.confidence * weight
        confidence_weight_total += weight

        if event.sentiment_label == "positive":
            positive_count += 1
        elif event.sentiment_label == "negative":
            negative_count += 1
        elif event.sentiment_label == "neutral":
            neutral_count += 1
        elif event.sentiment_label == "mixed":
            mixed_count += 1
        elif event.sentiment_label == "unknown":
            unknown_count += 1

        weighted_impact = event.sentiment_score * weight
        if weighted_impact > 0:
            if strongest_positive_value is None or weighted_impact > strongest_positive_value:
                strongest_positive_value = weighted_impact
                strongest_positive_event_id = event.news_event_id
        if weighted_impact < 0:
            if strongest_negative_value is None or weighted_impact < strongest_negative_value:
                strongest_negative_value = weighted_impact
                strongest_negative_event_id = event.news_event_id

    if weight_total == 0:
        signal_score = 0.0
    else:
        signal_score = weighted_sum / weight_total

    if confidence_weight_total == 0:
        confidence = 0.0
    else:
        confidence = confidence_sum / confidence_weight_total

    signal_label = get_signal_label(
        score=signal_score,
        positive_count=positive_count,
        negative_count=negative_count,
    )

    if use_llm_summary:
        summary = generate_ticker_signal_summary(
            ticker=first.ticker,
            company_name=first.company_name,
            sector=first.sector,
            signal_score=signal_score,
            signal_label=signal_label,
            confidence=confidence,
            events=events,
        )
    else:
        summary = simple_summary(
            ticker=first.ticker,
            company_name=first.company_name,
            signal_score=signal_score,
            signal_label=signal_label,
            event_count=len(events),
        )

    return TickerSentimentSignal(
        sector=first.sector,
        ticker=first.ticker,
        company_name=first.company_name,
        signal_score=signal_score,
        signal_label=signal_label,
        confidence=confidence,
        event_count=len(events),
        positive_count=positive_count,
        negative_count=negative_count,
        neutral_count=neutral_count,
        mixed_count=mixed_count,
        unknown_count=unknown_count,
        strongest_positive_event_id=strongest_positive_event_id,
        strongest_negative_event_id=strongest_negative_event_id,
        summary=summary,
    )


def run_signal_aggregation_agent(
    sector: str,
    limit_per_ticker: int = 20,
    use_llm_summary: bool = True,
    force_refresh: bool = False,
) -> dict:
    aggregation_cache_key = build_cache_key(
        "aggregation",
        sector,
        datetime.now(timezone.utc).date().isoformat(),
        limit_per_ticker,
    )
    if not force_refresh and cache_exists(aggregation_cache_key):
        return {
            "sector": sector,
            "tickers_processed": 0,
            "signals_saved": 0,
            "events_used": 0,
            "llm_summary_skipped": True,
            "cache_key": aggregation_cache_key,
            "cached_run_skipped": True,
        }

    llm_summary_skipped = (
        use_llm_summary
        and not force_refresh
        and cache_exists(aggregation_cache_key)
    )
    use_llm_summary_for_run = use_llm_summary and not llm_summary_skipped

    events = get_sentiment_events_for_aggregation(
        sector=sector,
        limit_per_ticker=limit_per_ticker,
    )

    grouped: dict[tuple[str, str, str], list[SentimentEventInput]] = defaultdict(list)
    for event in events:
        key = (
            event.sector,
            event.ticker,
            event.company_name,
        )
        grouped[key].append(event)

    saved_signals = 0
    for ticker_events in grouped.values():
        signal = aggregate_ticker_events(
            events=ticker_events,
            use_llm_summary=use_llm_summary_for_run,
        )
        save_ticker_sentiment_signal(signal)
        saved_signals += 1

    if use_llm_summary_for_run:
        set_cache(
            cache_key=aggregation_cache_key,
            cache_type="aggregation",
            value=sector,
            ttl_hours=12,
        )
    else:
        set_cache(
            cache_key=aggregation_cache_key,
            cache_type="aggregation",
            value=sector,
            ttl_hours=12,
        )

    return {
        "sector": sector,
        "tickers_processed": len(grouped),
        "signals_saved": saved_signals,
        "events_used": len(events),
        "llm_summary_skipped": llm_summary_skipped,
        "cache_key": aggregation_cache_key,
        "cached_run_skipped": False,
    }
