from sentiment.repositories import (
    get_relevant_unprocessed_news_events,
    save_sentiment_decision,
)
from sentiment.llm_sentiment import evaluate_news_sentiment


def run_news_sentiment_agent(limit: int = 50) -> dict:
    events = get_relevant_unprocessed_news_events(limit=limit)
    processed = 0
    positive = 0
    negative = 0
    neutral = 0
    mixed = 0
    unknown = 0
    for event in events:
        decision = evaluate_news_sentiment(event)
        save_sentiment_decision(
            event=event,
            decision=decision,
        )
        processed += 1
        if decision.sentiment_label == "positive":
            positive += 1
        elif decision.sentiment_label == "negative":
            negative += 1
        elif decision.sentiment_label == "neutral":
            neutral += 1
        elif decision.sentiment_label == "mixed":
            mixed += 1
        elif decision.sentiment_label == "unknown":
            unknown += 1
    return {
        "processed_events": processed,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "mixed": mixed,
        "unknown": unknown,
    }
