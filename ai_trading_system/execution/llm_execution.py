import json

from dotenv import load_dotenv
from openai import OpenAI

from execution.schemas import ExecutionContext, LlmExecutionDecision

load_dotenv()
client = OpenAI()


def evaluate_llm_execution(
    context: ExecutionContext,
) -> LlmExecutionDecision:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": """
You are the simulated execution decision agent for a fake-money trading system.
You receive a trade plan and the current simulated portfolio state.
Your job is to decide whether this simulated order should be:
- filled
- partially filled
- rejected
- skipped
You must use only the provided data.
Important:
- This is fake-money simulation, not real execution.
- Do not recommend real-world trading.
- Do not invent market data.
- Do not decide share quantity or cash accounting.
- You decide execution status and fill ratio only.
- The deterministic engine will calculate quantity, price, cash, holdings.
Rules:
- If trader_status is not running, reject.
- If planned_side is none, skip.
- If max_executable_value <= 0, reject.
- If available_cash_after_reserve <= 0, reject.
- If requested_value > max_executable_value, use partial_fill.
- Aggressive profiles may accept fuller fills.
- Conservative profiles should be more cautious and may use smaller fill ratios.
- For high execution priority and adequate capital, fill is acceptable.
- llm_fill_ratio must be 0 if status is reject or skip.
- llm_fill_ratio must be between 0 and 1.
""",
            },
            {
                "role": "user",
                "content": json.dumps(
                    context.model_dump(),
                    indent=2,
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "llm_execution_decision",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "llm_execution_status": {
                            "type": "string",
                            "enum": ["fill", "partial_fill", "reject", "skip"],
                        },
                        "llm_fill_ratio": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "llm_execution_confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "llm_execution_reason": {
                            "type": "string",
                        },
                        "llm_execution_flags": {
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                        },
                    },
                    "required": [
                        "llm_execution_status",
                        "llm_fill_ratio",
                        "llm_execution_confidence",
                        "llm_execution_reason",
                        "llm_execution_flags",
                    ],
                },
            }
        },
        temperature=0,
    )

    data = json.loads(response.output_text)
    decision = LlmExecutionDecision.model_validate(data)
    if decision.llm_execution_status in ["reject", "skip"]:
        decision.llm_fill_ratio = 0.0
    if decision.llm_execution_status == "fill":
        decision.llm_fill_ratio = 1.0
    return decision
