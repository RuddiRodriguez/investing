from relevance.repositories import (
    get_unprocessed_news_events,
    save_relevance_decision,
)
from relevance.llm_relevance import evaluate_news_relevance


def run_news_relevance_agent(limit: int = 50) -> dict:
    events = get_unprocessed_news_events(limit=limit)
    processed = 0
    relevant = 0
    not_relevant = 0
    for event in events:
        decision = evaluate_news_relevance(event)
        save_relevance_decision(
            event=event,
            decision=decision,
        )
        processed += 1
        if decision.is_relevant:
            relevant += 1
        else:
            not_relevant += 1
    return {
        "processed_events": processed,
        "relevant_events": relevant,
        "not_relevant_events": not_relevant,
    }
