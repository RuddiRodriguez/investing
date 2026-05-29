import json

from dotenv import load_dotenv
from openai import OpenAI

from portfolio.schemas import LlmPortfolioDecision, PortfolioInput, SuggestedPosition
from strategy_knowledge.strategy_knowledge_agent import load_strategy_knowledge_for_agent

load_dotenv()
client = OpenAI()


def evaluate_llm_portfolio_decision(
    portfolio_input: PortfolioInput,
    suggested_position: SuggestedPosition,
    max_position_size: float,
) -> LlmPortfolioDecision:
    strategy_knowledge = load_strategy_knowledge_for_agent(
        "oneil_growth_leadership"
    )

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": """
You are the portfolio construction decision agent for a trading system.
You receive:
- an approved risk decision
- alpha signal data
- risk data
- a deterministic suggested direction and position size
Your job is to make the final portfolio decision.
You may agree or disagree with the deterministic suggestion, but you must use only the provided data.
Rules:
- Do not invent external facts.
- Do not use unstated news.
- Do not recommend dollar amounts or number of shares.
- Position size is a portfolio weight between 0 and max_position_size.
- If action is skip or direction is none, llm_position_size must be 0.
- If risk label is high, position size should be small or zero.
- If alpha confidence is weak, position size should be reduced or zero.
- If alpha score is strong and risk is acceptable, a nonzero position is allowed.
- Be conservative with short positions.
- Do not exceed max_position_size.
Chart confirmation is a hard gate for new long positions.
Rules:
- If chart_decision is not BUY, do not open a long position.
- If chart_decision is WAIT_FOR_BREAKOUT, action should be watch and buy_probability <= 0.49.
- If chart_decision is AVOID, action should be skip and buy_probability <= 0.20.
- If chart_decision is SELL, action should be reduce or skip and buy_probability = 0.
- A long open position requires confirmed breakout, strong volume, and proper entry.
Use buy_trigger, invalid_buy_reason, reason_to_wait, resistance_level, breakout_status, volume_confirmation, and entry_quality when explaining the decision.
Also output buy_probability.
buy_probability means:
- probability that this position should be opened as a long/buy candidate
- 0.0 means no buy interest
- 1.0 means very strong buy conviction inside this simulation
- If llm_direction is not long, buy_probability must be 0.0
- If llm_portfolio_action is skip or watch, buy_probability should usually be below 0.50
- If llm_portfolio_action is open and llm_direction is long, buy_probability should reflect alpha strength, confidence, risk, and strategy alignment
""",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "strategy_knowledge": strategy_knowledge,
                        "max_position_size": max_position_size,
                        "portfolio_input": portfolio_input.model_dump(),
                        "deterministic_suggestion": suggested_position.model_dump(),
                    },
                    indent=2,
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "llm_portfolio_decision",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "llm_direction": {
                            "type": "string",
                            "enum": ["long", "short", "none"],
                        },
                        "llm_portfolio_action": {
                            "type": "string",
                            "enum": ["open", "skip", "watch", "reduce"],
                        },
                        "llm_position_size": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": max_position_size,
                        },
                        "llm_confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "buy_probability": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "portfolio_reason": {
                            "type": "string",
                        },
                        "portfolio_flags": {
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                        },
                    },
                    "required": [
                        "llm_direction",
                        "llm_portfolio_action",
                        "llm_position_size",
                        "llm_confidence",
                        "buy_probability",
                        "portfolio_reason",
                        "portfolio_flags",
                    ],
                },
            }
        },
        temperature=0,
    )

    data = json.loads(response.output_text)
    decision = LlmPortfolioDecision.model_validate(data)

    chart_decision = portfolio_input.chart_decision
    if chart_decision != "BUY":
        if chart_decision == "SELL":
            decision.llm_portfolio_action = "reduce"
            if "chart_sell_signal" not in decision.portfolio_flags:
                decision.portfolio_flags.append("chart_sell_signal")
        elif chart_decision == "AVOID":
            decision.llm_portfolio_action = "skip"
            if "chart_avoid_signal" not in decision.portfolio_flags:
                decision.portfolio_flags.append("chart_avoid_signal")
        else:
            decision.llm_portfolio_action = "watch"
            if "wait_for_chart_confirmation" not in decision.portfolio_flags:
                decision.portfolio_flags.append("wait_for_chart_confirmation")
        decision.llm_direction = "none"
        decision.llm_position_size = 0.0
        if chart_decision == "WAIT_FOR_BREAKOUT":
            decision.buy_probability = min(decision.buy_probability, 0.49)
        elif chart_decision == "AVOID":
            decision.buy_probability = min(decision.buy_probability, 0.20)
        else:
            decision.buy_probability = 0.0
    if chart_decision == "BUY":
        if decision.llm_direction == "long" and decision.llm_portfolio_action == "open":
            decision.buy_probability = max(decision.buy_probability, 0.60)

    if decision.llm_direction != "long":
        decision.buy_probability = 0.0
    if decision.llm_portfolio_action in ["skip", "watch"]:
        decision.llm_position_size = 0.0
        decision.buy_probability = min(decision.buy_probability, 0.49)
    if decision.llm_direction == "none":
        decision.llm_position_size = 0.0
        decision.buy_probability = 0.0
    if decision.llm_position_size > max_position_size:
        decision.llm_position_size = max_position_size
    return decision
