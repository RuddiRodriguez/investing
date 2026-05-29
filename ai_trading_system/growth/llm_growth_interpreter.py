import json

from dotenv import load_dotenv
from openai import OpenAI

from growth.schemas import (
    GrowthLeadershipLlmInterpretation,
    GrowthLeadershipMetrics,
)
from strategy_knowledge.strategy_knowledge_agent import (
    load_strategy_knowledge_for_agent,
)

load_dotenv()
client = OpenAI()


def interpret_growth_leadership_signal(
    metrics: GrowthLeadershipMetrics,
) -> GrowthLeadershipLlmInterpretation:
    strategy_knowledge = load_strategy_knowledge_for_agent(
        "oneil_growth_leadership"
    )
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": """
You are a growth leadership interpretation agent for a trading simulation.
You receive:
1. Structured strategy knowledge extracted by an LLM from trading notes.
2. Measurable market metrics for one ticker.
Your job:
- interpret whether the ticker matches the stored strategy knowledge
- use only the provided strategy knowledge and metrics
- do not invent facts
- do not make real-money trading recommendations
- do not make final buy/sell decisions
""",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "strategy_knowledge": strategy_knowledge,
                        "metrics": metrics.model_dump(),
                    },
                    indent=2,
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "growth_leadership_interpretation",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "llm_interpretation": {
                            "type": "string"
                        },
                        "main_positive_factors": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "main_risks": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "decision_bias": {
                            "type": "string",
                            "enum": [
                                "supports_long_bias",
                                "neutral",
                                "caution",
                                "avoid_long",
                                "unknown"
                            ]
                        }
                    },
                    "required": [
                        "llm_interpretation",
                        "main_positive_factors",
                        "main_risks",
                        "decision_bias"
                    ]
                }
            }
        },
        temperature=0,
    )
    data = json.loads(response.output_text)
    return GrowthLeadershipLlmInterpretation.model_validate(data)
