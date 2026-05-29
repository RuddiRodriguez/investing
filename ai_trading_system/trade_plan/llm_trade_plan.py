import json

from dotenv import load_dotenv
from openai import OpenAI

from trade_plan.schemas import LlmTradePlanDecision, TradePlanInput

load_dotenv()
client = OpenAI()


def evaluate_llm_trade_plan(
    trade_input: TradePlanInput,
) -> LlmTradePlanDecision:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": """
You are the trade planning agent for a simulated trading system.
You receive a portfolio-approved position candidate.
Your job is to convert it into an execution-ready trade plan.
You do not execute trades.
You do not decide dollar amounts.
You do not decide exact share quantity.
You only decide:
- side
- order type
- time in force
- execution priority
- maximum slippage tolerance
- trade plan status
- explanation
Rules:
- Use only the provided data.
- Do not invent external facts.
- If llm_direction is long, planned_side should usually be buy.
- If llm_direction is short, planned_side should usually be sell_short.
- If llm_position_size is 0, trade_plan_status must be skipped.
- If risk is high, prefer limit order and lower slippage.
- If confidence is high and risk is low/medium, market order is allowed.
- Be conservative with short trades.
- max_slippage_pct must be between 0 and 0.05.
- Do not recommend real-money execution.
""",
            },
            {
                "role": "user",
                "content": json.dumps(
                    trade_input.model_dump(),
                    indent=2,
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "llm_trade_plan_decision",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "planned_side": {
                            "type": "string",
                            "enum": ["buy", "sell_short", "none"],
                        },
                        "planned_order_type": {
                            "type": "string",
                            "enum": ["market", "limit"],
                        },
                        "planned_time_in_force": {
                            "type": "string",
                            "enum": ["day", "gtc"],
                        },
                        "execution_priority": {
                            "type": "string",
                            "enum": ["low", "normal", "high"],
                        },
                        "max_slippage_pct": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 0.05,
                        },
                        "trade_plan_status": {
                            "type": "string",
                            "enum": ["planned", "skipped"],
                        },
                        "trade_reason": {
                            "type": "string",
                        },
                        "execution_notes": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "planned_side",
                        "planned_order_type",
                        "planned_time_in_force",
                        "execution_priority",
                        "max_slippage_pct",
                        "trade_plan_status",
                        "trade_reason",
                        "execution_notes",
                    ],
                },
            }
        },
        temperature=0,
    )

    data = json.loads(response.output_text)
    decision = LlmTradePlanDecision.model_validate(data)

    if trade_input.llm_position_size <= 0:
        decision.planned_side = "none"
        decision.trade_plan_status = "skipped"
    if trade_input.llm_direction == "long" and decision.trade_plan_status == "planned":
        decision.planned_side = "buy"
    if trade_input.llm_direction == "short" and decision.trade_plan_status == "planned":
        decision.planned_side = "sell_short"
    if decision.trade_plan_status == "skipped":
        decision.planned_side = "none"

    return decision
