import json

from ingestion.db import get_connection, utc_now
from strategy_knowledge.schemas import ExtractedStrategyKnowledge


def save_strategy_knowledge(
    raw_text: str,
    extracted: ExtractedStrategyKnowledge,
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO strategy_knowledge (
            strategy_name,
            source_name,
            source_type,
            raw_text,
            principles_json,
            rules_json,
            features_json,
            risk_rules_json,
            portfolio_rules_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            extracted.strategy_name,
            extracted.source_name,
            extracted.source_type,
            raw_text,
            json.dumps([item.model_dump() for item in extracted.principles]),
            json.dumps([item.model_dump() for item in extracted.rules]),
            json.dumps([item.model_dump() for item in extracted.features]),
            json.dumps([item.model_dump() for item in extracted.risk_rules]),
            json.dumps([item.model_dump() for item in extracted.portfolio_rules]),
            utc_now(),
        ),
    )
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def get_strategy_knowledge(strategy_name: str) -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT *
        FROM strategy_knowledge
        WHERE strategy_name = ?
        ORDER BY created_at DESC
        """,
        (strategy_name,),
    )
    rows = cursor.fetchall()
    conn.close()
    result = []
    for row in rows:
        result.append(
            {
                "strategy_name": row["strategy_name"],
                "source_name": row["source_name"],
                "source_type": row["source_type"],
                "principles": json.loads(row["principles_json"]),
                "rules": json.loads(row["rules_json"]),
                "features": json.loads(row["features_json"]),
                "risk_rules": json.loads(row["risk_rules_json"]),
                "portfolio_rules": json.loads(row["portfolio_rules_json"]),
            }
        )
    return result
