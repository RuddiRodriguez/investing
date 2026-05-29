system_message = """
# Role: Autonomous Stock Trader

You are an autonomous trader agent with deep technical-analysis, market-structure, sentiment, and risk-management knowledge.

Your job is to make one ticker-level trading decision from:
- the completed rule-based forecast and technical report
- the trader profile
- current market/news/sentiment/forum context from web search when available
- optional portfolio context supplied by the user

You are not allowed to place orders. You produce a governed trading plan.

## Decision Style

Anchor first on the rule-based forecast engine. The engine has already computed models, validation, technical chapters, trade/risk controls, portfolio capital/risk controls, and discipline/governance checks.

Use fresh market context to explain, downgrade, refine entry/exit levels, or require waiting. Do not ignore hard rule blocks from validation, trade/risk, portfolio capital, or discipline governance.

Respect the trader profile:
- aggressive can accept earlier entries and higher volatility, but cannot bypass stops or hard blocks
- medium balances technical evidence, sentiment, and execution quality
- conservative requires stronger alignment and should prefer waiting when evidence conflicts

## Market Context Work

If web search is available, search recent sources about the ticker:
- company news and filings
- earnings, guidance, analyst changes, product/regulatory events
- market/sector context
- sentiment in investor discussions such as Reddit, forums, or social/news discussions when visible

Separate hard facts from sentiment. Treat forums as weak context, not evidence by itself.

## Output Rules

Return exactly one JSON object matching the schema.
Use numeric price levels when the technical report gives enough support, resistance, stop, target, or forecast levels.
If a price is not justified, use null and explain what to wait for.
Never recommend Buy/Sell if the report says new capital is blocked. In that case use Hold and explain the block.
""".strip()


user_message = """
Today:
{{ item.today }}

Ticker:
{{ item.ticker }}

Trader:
{{ item.trader_name }}

Trader Profile:
{{ item.trader_profile_json }}

Portfolio Context:
{{ item.portfolio_context_json }}

Technical Forecast Packet:
{{ item.technical_packet_json }}

Task:
Use the technical packet and fresh market/sentiment context to decide Buy, Hold, or Sell.
Give:
- the final decision
- price to wait for before buying when not owned
- price to consider selling/taking profit when already owned
- stop loss or invalidation level
- take profit level
- what would make you change the decision
- key risks
- questions you would ask the user next if this becomes an interactive trader
""".strip()


json_schema = {
    "type": "json_schema",
    "name": "autonomous_trader_decision",
    "description": "Structured autonomous trader decision for one ticker.",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["Buy", "Hold", "Sell"],
                "description": "Final governed ticker decision.",
            },
            "confidence": {
                "type": "number",
                "description": "Decision confidence from 0 to 1.",
            },
            "trader_profile_used": {
                "type": "string",
                "enum": ["aggressive", "medium", "conservative"],
            },
            "timeframe": {
                "type": "string",
                "description": "Expected holding/review timeframe.",
            },
            "technical_read": {
                "type": "string",
                "description": "Short interpretation of the rule-based technical and forecast packet.",
            },
            "market_context_read": {
                "type": "string",
                "description": "Short interpretation of news, fundamentals, sector, and sentiment context.",
            },
            "sentiment": {
                "type": "string",
                "enum": ["bullish", "mixed", "bearish", "insufficient"],
            },
            "decision_reasoning": {
                "type": "string",
                "description": "Brief reasoning joining technical evidence, market context, and the trader profile.",
            },
            "entry_plan": {
                "type": "object",
                "properties": {
                    "entry_style": {
                        "type": "string",
                        "enum": [
                            "buy_now",
                            "wait_for_pullback",
                            "wait_for_breakout",
                            "do_not_enter",
                            "reduce_or_exit",
                            "hold_existing_only",
                        ],
                    },
                    "buy_near": {"type": ["number", "null"]},
                    "buy_above": {"type": ["number", "null"]},
                    "sell_near": {"type": ["number", "null"]},
                    "stop_loss": {"type": ["number", "null"]},
                    "take_profit": {"type": ["number", "null"]},
                    "invalidation": {"type": "string"},
                },
                "required": [
                    "entry_style",
                    "buy_near",
                    "buy_above",
                    "sell_near",
                    "stop_loss",
                    "take_profit",
                    "invalidation",
                ],
                "additionalProperties": False,
            },
            "portfolio_plan": {
                "type": "object",
                "properties": {
                    "if_not_owned": {"type": "string"},
                    "if_owned": {"type": "string"},
                    "position_size_comment": {"type": "string"},
                },
                "required": ["if_not_owned", "if_owned", "position_size_comment"],
                "additionalProperties": False,
            },
            "rule_blocks": {
                "type": "array",
                "items": {"type": "string"},
            },
            "risks": {
                "type": "array",
                "items": {"type": "string"},
            },
            "change_triggers": {
                "type": "array",
                "items": {"type": "string"},
            },
            "questions_for_user": {
                "type": "array",
                "items": {"type": "string"},
            },
            "sources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "source_type": {
                            "type": "string",
                            "enum": ["news", "company", "filing", "forum", "market_data", "other"],
                        },
                        "relevance": {"type": "string"},
                    },
                    "required": ["title", "url", "source_type", "relevance"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "decision",
            "confidence",
            "trader_profile_used",
            "timeframe",
            "technical_read",
            "market_context_read",
            "sentiment",
            "decision_reasoning",
            "entry_plan",
            "portfolio_plan",
            "rule_blocks",
            "risks",
            "change_triggers",
            "questions_for_user",
            "sources",
        ],
        "additionalProperties": False,
    },
}
