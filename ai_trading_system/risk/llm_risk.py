import json

from dotenv import load_dotenv
from openai import OpenAI

from risk.schemas import LlmRiskDecision, RiskInput, RuleBasedRiskResult
from strategy_knowledge.strategy_knowledge_agent import load_strategy_knowledge_for_agent

load_dotenv()
client = OpenAI()


def _signal_disagreement(risk_input: RiskInput) -> str:
    sentiment = risk_input.sentiment_score
    technical = risk_input.technical_score
    if sentiment is None or technical is None:
        return "unknown"
    if sentiment > 0.25 and technical < -0.25:
        return "conflict"
    if sentiment < -0.25 and technical > 0.25:
        return "conflict"
    return "aligned"


def evaluate_llm_risk(
    risk_input: RiskInput,
    rule_result: RuleBasedRiskResult,
) -> LlmRiskDecision:
    strategy_knowledge = load_strategy_knowledge_for_agent(
        "oneil_growth_leadership"
    )

    payload = {
        "strategy_knowledge": strategy_knowledge,
        "risk_input": {
            **risk_input.model_dump(),
            "signal_disagreement": _signal_disagreement(risk_input),
        },
        "rule_based_result": rule_result.model_dump(),
    }

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": """
You are the final risk decision agent for a trading system.
You receive deterministic risk metrics and rule-based risk output.
Your job is to make the final risk decision.
You are allowed to agree or disagree with the rule-based decision, but you must justify it using only the provided data.
You must not invent external facts.
You must not use unstated news.
You must not recommend exact shares or dollar amounts.
You must not ignore severe risk metrics.
Your decision must be conservative when:
- alpha confidence is low
- alpha score is weak
- volatility is high
- drawdown is severe
- sentiment and technical signals conflict
- technical signal is negative while alpha is positive
- data is incomplete
Output:
- llm_trade_allowed
- llm_risk_score
- llm_risk_label
- position_size_multiplier
- risk_flags
- risk_summary
- decision_reason
Position size multiplier:
- 0.00 means do not trade
- 0.25 means very small position
- 0.50 means reduced position
- 0.75 means moderate position
- 1.00 means full allowed position
Use 0.00 if llm_trade_allowed is false.
""",
            },
            {
                "role": "user",
                "content": json.dumps(payload, indent=2),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "llm_risk_decision",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "llm_risk_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "llm_risk_label": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "extreme", "unknown"],
                        },
                        "llm_trade_allowed": {
                            "type": "boolean",
                        },
                        "position_size_multiplier": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "risk_flags": {
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                        },
                        "risk_summary": {
                            "type": "string",
                        },
                        "decision_reason": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "llm_risk_score",
                        "llm_risk_label",
                        "llm_trade_allowed",
                        "position_size_multiplier",
                        "risk_flags",
                        "risk_summary",
                        "decision_reason",
                    ],
                },
            }
        },
        temperature=0,
    )

    data = json.loads(response.output_text)
    decision = LlmRiskDecision.model_validate(data)
    if not decision.llm_trade_allowed:
        decision.position_size_multiplier = 0.0
    return decision
