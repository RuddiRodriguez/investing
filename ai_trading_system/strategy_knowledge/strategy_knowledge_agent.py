from strategy_knowledge.llm_strategy_extractor import (
    extract_strategy_knowledge_with_llm,
)
from strategy_knowledge.repositories import (
    get_strategy_knowledge,
    save_strategy_knowledge,
)


def ingest_strategy_knowledge(
    strategy_name: str,
    source_name: str,
    source_type: str,
    raw_text: str,
) -> dict:
    extracted = extract_strategy_knowledge_with_llm(
        strategy_name=strategy_name,
        source_name=source_name,
        source_type=source_type,
        raw_text=raw_text,
    )
    record_id = save_strategy_knowledge(
        raw_text=raw_text,
        extracted=extracted,
    )
    return {
        "record_id": record_id,
        "strategy_name": extracted.strategy_name,
        "source_name": extracted.source_name,
        "principles": len(extracted.principles),
        "rules": len(extracted.rules),
        "features": len(extracted.features),
        "risk_rules": len(extracted.risk_rules),
        "portfolio_rules": len(extracted.portfolio_rules),
    }


def load_strategy_knowledge_for_agent(strategy_name: str) -> list[dict]:
    return get_strategy_knowledge(strategy_name)
