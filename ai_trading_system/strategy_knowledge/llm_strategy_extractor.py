import json

from dotenv import load_dotenv
from openai import OpenAI

from strategy_knowledge.schemas import ExtractedStrategyKnowledge

load_dotenv()
client = OpenAI()


def extract_strategy_knowledge_with_llm(
    strategy_name: str,
    source_name: str,
    source_type: str,
    raw_text: str,
) -> ExtractedStrategyKnowledge:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": """
You are a strategy knowledge extraction agent for a trading simulation system.
Your job is to convert trading knowledge text into structured reusable decision knowledge.
Extract only what is supported by the provided text.
Definitions:
- principles: high-level investment beliefs
- rules: decision rules that can influence scoring
- features: measurable market/company data needed to apply the rules
- risk_rules: stop-loss, drawdown, loss control, averaging-down, or protection rules
- portfolio_rules: how the knowledge should influence sizing, frequency, profile behavior, or position preference
Do not make trading recommendations.
Do not invent facts outside the provided text.
Do not mention real-money execution.
""",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "strategy_name": strategy_name,
                        "source_name": source_name,
                        "source_type": source_type,
                        "raw_text": raw_text,
                    },
                    indent=2,
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "extracted_strategy_knowledge",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "strategy_name": {"type": "string"},
                        "source_name": {"type": "string"},
                        "source_type": {"type": "string"},
                        "principles": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "importance": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                    },
                                },
                                "required": ["name", "description", "importance"],
                            },
                        },
                        "rules": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "rule_name": {"type": "string"},
                                    "rule_type": {"type": "string"},
                                    "description": {"type": "string"},
                                    "preferred_condition": {"type": "string"},
                                    "avoid_condition": {
                                        "type": ["string", "null"],
                                    },
                                    "weight": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                    },
                                },
                                "required": [
                                    "rule_name",
                                    "rule_type",
                                    "description",
                                    "preferred_condition",
                                    "avoid_condition",
                                    "weight",
                                ],
                            },
                        },
                        "features": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "feature_name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "calculation_hint": {
                                        "type": ["string", "null"],
                                    },
                                },
                                "required": [
                                    "feature_name",
                                    "description",
                                    "calculation_hint",
                                ],
                            },
                        },
                        "risk_rules": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "rule_name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "threshold": {
                                        "type": ["number", "null"],
                                    },
                                    "action": {"type": "string"},
                                },
                                "required": [
                                    "rule_name",
                                    "description",
                                    "threshold",
                                    "action",
                                ],
                            },
                        },
                        "portfolio_rules": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "rule_name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "profile_effect": {"type": "string"},
                                },
                                "required": [
                                    "rule_name",
                                    "description",
                                    "profile_effect",
                                ],
                            },
                        },
                    },
                    "required": [
                        "strategy_name",
                        "source_name",
                        "source_type",
                        "principles",
                        "rules",
                        "features",
                        "risk_rules",
                        "portfolio_rules",
                    ],
                },
            }
        },
        temperature=0,
    )
    data = json.loads(response.output_text)
    return ExtractedStrategyKnowledge.model_validate(data)
