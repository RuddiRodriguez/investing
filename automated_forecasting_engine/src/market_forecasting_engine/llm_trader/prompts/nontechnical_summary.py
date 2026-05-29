system_message = """
# Role: Beginner Trading Summary Writer

You summarize an autonomous trader decision for a non-technical person.

The input is already the output from a trading LLM plus the compact technical packet. Do not make a new trading decision. Do not add new market facts. Your job is to translate the decision into plain language while keeping important numbers.

Write for a beginner:
- avoid jargon when possible
- explain technical terms in simple words
- keep the exact decision, confidence, risk, buy levels, sell levels, stop loss, and take-profit levels
- show all price levels in dollars first and euros next in parentheses when the currency context has a USD to EUR rate
- format price text like $1,505.22 (€1,385.00)
- if the EUR conversion rate is unavailable, keep the dollar amount and explicitly say the EUR conversion is unavailable
- make clear what to do if the person does not own the ticker
- make clear what to do if the person already owns the ticker
- make every recheck trigger goal-oriented: say whether the trigger is for considering Buy, Sell, Hold, reducing risk, taking profit, or continuing to wait
- do not write vague trigger text like "recheck if price changes" unless it also says what decision the user is rechecking for
- explain why the decision was made in everyday language
- preserve the warnings and change triggers

Return exactly one JSON object matching the schema.
""".strip()


user_message = """
Ticker:
{{ item.ticker }}

Trader Name:
{{ item.trader_name }}

Trader Profile:
{{ item.trader_profile_json }}

Portfolio Context:
{{ item.portfolio_context_json }}

Currency Context:
{{ item.currency_context_json }}

Trader Decision JSON:
{{ item.trader_decision_json }}

Technical Packet JSON:
{{ item.technical_packet_json }}

Task:
Create a beginner-friendly summary for a non-technical person. Keep the numbers, levels, and practical details. For every price level, show dollars first and euros next in parentheses using the currency context.

Make the output decision-oriented. A beginner should understand:
- what to do now
- what price or event would change the decision
- whether that future trigger is about buying, selling, holding, reducing risk, or taking profit
- what changes if the user already owns the asset versus does not own it
""".strip()


json_schema = {
    "type": "json_schema",
    "name": "nontechnical_trader_summary",
    "description": "Beginner-friendly summary of an autonomous trader decision.",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "headline": {
                "type": "string",
                "description": "One short plain-language headline.",
            },
            "decision": {
                "type": "string",
                "enum": ["Buy", "Hold", "Sell"],
            },
            "confidence": {
                "type": "number",
            },
            "risk_level": {
                "type": "string",
            },
            "plain_language_summary": {
                "type": "string",
                "description": "A short beginner-friendly explanation of the decision.",
            },
            "what_to_do_now": {
                "type": "string",
                "description": "Simple action instruction for today.",
            },
            "if_not_owned": {
                "type": "string",
                "description": "Plain-language instruction if the user does not own the ticker.",
            },
            "if_owned": {
                "type": "string",
                "description": "Plain-language instruction if the user already owns the ticker.",
            },
            "important_prices": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "price_usd": {"type": ["number", "null"]},
                        "price_eur": {"type": ["number", "null"]},
                        "display": {
                            "type": "string",
                            "description": "Price formatted as dollars first and euros in parentheses, for example $1,505.22 (€1,385.00).",
                        },
                        "plain_meaning": {"type": "string"},
                    },
                    "required": ["label", "price_usd", "price_eur", "display", "plain_meaning"],
                    "additionalProperties": False,
                },
            },
            "currency_note": {
                "type": "string",
                "description": "Plain-language note describing the USD to EUR conversion rate or explaining that EUR conversion was unavailable.",
            },
            "why_this_decision": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Simple bullet-style reasons.",
            },
            "main_risks": {
                "type": "array",
                "items": {"type": "string"},
            },
            "when_to_recheck": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "Plain-language trigger that includes the decision purpose, for example: Recheck for a possible BUY if price closes above $X and holds; otherwise continue HOLD/wait.",
                },
            },
            "decision_triggers": {
                "type": "array",
                "description": "Goal-oriented triggers that tell a beginner what decision to consider when a level/event happens.",
                "items": {
                    "type": "object",
                    "properties": {
                        "trigger": {
                            "type": "string",
                            "description": "The price/event condition to watch.",
                        },
                        "decision_goal": {
                            "type": "string",
                            "enum": ["consider_buy", "consider_sell", "consider_hold", "reduce_risk", "take_profit", "keep_waiting", "manual_review"],
                            "description": "The decision being rechecked.",
                        },
                        "if_not_owned_action": {
                            "type": "string",
                            "description": "Concrete beginner instruction if the user does not own the ticker.",
                        },
                        "if_owned_action": {
                            "type": "string",
                            "description": "Concrete beginner instruction if the user already owns the ticker.",
                        },
                        "plain_reason": {
                            "type": "string",
                            "description": "Simple reason why this trigger matters.",
                        },
                    },
                    "required": [
                        "trigger",
                        "decision_goal",
                        "if_not_owned_action",
                        "if_owned_action",
                        "plain_reason",
                    ],
                    "additionalProperties": False,
                },
            },
            "beginner_notes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short explanations of important concepts like Hold, stop loss, breakout, or take profit.",
            },
        },
        "required": [
            "headline",
            "decision",
            "confidence",
            "risk_level",
            "plain_language_summary",
            "what_to_do_now",
            "if_not_owned",
            "if_owned",
            "important_prices",
            "currency_note",
            "why_this_decision",
            "main_risks",
            "when_to_recheck",
            "decision_triggers",
            "beginner_notes",
        ],
        "additionalProperties": False,
    },
}
