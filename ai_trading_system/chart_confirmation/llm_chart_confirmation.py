import json

from dotenv import load_dotenv
from openai import OpenAI

from chart_confirmation.schemas import ChartMetrics, LlmChartDecision
from strategy_knowledge.strategy_knowledge_agent import (
    load_strategy_knowledge_for_agent,
)

load_dotenv()
client = OpenAI()


def evaluate_chart_confirmation_with_llm(
    metrics: ChartMetrics,
) -> LlmChartDecision:
    strategy_knowledge = load_strategy_knowledge_for_agent(
        "oneil_growth_leadership"
    )
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": """
You are a strict chart confirmation agent for a fake-money trading simulation.
You receive:
1. Strategy knowledge extracted from the user's trading notes.
2. Deterministic chart metrics calculated from OHLCV data.
Your job is to produce a chart-valid decision.
Allowed decisions:
- BUY
- WAIT_FOR_BREAKOUT
- AVOID
- SELL
- HOLD
Decision rules:
- BUY only if trend_reading is strong_upward or weak_upward,
  breakout_status is confirmed_breakout,
  volume_confirmation is strong_volume,
  and entry_quality is proper_entry.
- WAIT_FOR_BREAKOUT if price is near resistance, trend is not broken,
  but breakout is not confirmed yet.
- AVOID if trend is weak downward, breakout is not confirmed,
  volume is weak, or entry quality is avoid / too_early / too_late / overextended.
- SELL if sell_signal is failed_breakout, heavy_distribution,
  stop_loss_triggered, or overextended_reversal.
- HOLD only for an existing position when the chart is not a fresh buy
  but also not a sell.
Do not guess.
Do not call BUY before the graph confirms strength.
Do not ignore weak volume.
Do not ignore resistance.
Do not make real-money recommendations.
Use only the provided metrics and strategy knowledge.
""",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "strategy_knowledge": strategy_knowledge,
                        "chart_metrics": metrics.model_dump(),
                    },
                    indent=2,
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "llm_chart_confirmation_decision",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "chart_decision": {
                            "type": "string",
                            "enum": [
                                "BUY",
                                "WAIT_FOR_BREAKOUT",
                                "AVOID",
                                "SELL",
                                "HOLD",
                            ],
                        },
                        "chart_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "chart_confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "llm_chart_reason": {
                            "type": "string",
                        },
                        "chart_flags": {
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                        },
                    },
                    "required": [
                        "chart_decision",
                        "chart_score",
                        "chart_confidence",
                        "llm_chart_reason",
                        "chart_flags",
                    ],
                },
            }
        },
        temperature=0,
    )
    data = json.loads(response.output_text)
    decision = LlmChartDecision.model_validate(data)
    # Hard deterministic overrides from the notes.
    if metrics.sell_signal != "none":
        decision.chart_decision = "SELL"
        decision.chart_score = min(decision.chart_score, 0.20)
        if metrics.sell_signal not in decision.chart_flags:
            decision.chart_flags.append(metrics.sell_signal)
    elif metrics.breakout_status != "confirmed_breakout":
        if decision.chart_decision == "BUY":
            decision.chart_decision = "WAIT_FOR_BREAKOUT"
            decision.chart_score = min(decision.chart_score, 0.55)
            decision.chart_flags.append("buy_blocked_no_confirmed_breakout")
    elif metrics.volume_confirmation != "strong_volume":
        if decision.chart_decision == "BUY":
            decision.chart_decision = "WAIT_FOR_BREAKOUT"
            decision.chart_score = min(decision.chart_score, 0.55)
            decision.chart_flags.append("buy_blocked_no_volume_confirmation")
    elif metrics.entry_quality != "proper_entry":
        if decision.chart_decision == "BUY":
            decision.chart_decision = "AVOID"
            decision.chart_score = min(decision.chart_score, 0.35)
            decision.chart_flags.append("buy_blocked_bad_entry_quality")
    elif metrics.trend_reading not in ["strong_upward", "weak_upward"]:
        if decision.chart_decision == "BUY":
            decision.chart_decision = "AVOID"
            decision.chart_score = min(decision.chart_score, 0.35)
            decision.chart_flags.append("buy_blocked_bad_trend")
    return decision
