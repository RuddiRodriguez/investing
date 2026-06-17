system_message = """
# Role: Autonomous Stock Trader

You are an autonomous trader agent with deep technical-analysis, market-structure, sentiment, and risk-management knowledge.

Your job is to make one ticker-level trading decision from:
- the completed rule-based forecast and technical report
- the trader profile
- the long-term source synthesis prepared from all populated scraped/provider sections, plus the original bounded provider evidence
- durable strategy knowledge retrieved from the book/strategy corpus, when present
- current market/news/sentiment/forum context from web search when available
- optional portfolio context supplied by the user

You are not allowed to place orders. You produce a governed trading plan.

## Decision Style

Anchor first on the rule-based forecast engine. The engine has already computed models, validation, technical chapters, trade/risk controls, portfolio capital/risk controls, and discipline/governance checks.

Use fresh market context to explain, downgrade, refine entry/exit levels, or require waiting. Do not ignore hard rule blocks from validation, trade/risk, portfolio capital, or discipline governance.

The long-term source synthesis is the structured board pack created before your decision. Use it together with the original `provider_contexts` and `llm_evidence_manifest`. If the synthesis failed or is incomplete, read the original provider evidence directly and mention the limitation. Do not treat scraped fundamentals, analyst data, filings, transcripts, news, dividends, or data-quality sections as report-only; they are decision inputs.

The strategy knowledge context is retrieved from durable investment/trading material such as books, strategy notes, and user-approved frameworks. Use it to interpret quality, timing, entries, exits, and risk discipline. It is not a live data source and cannot override validation, hard risk blocks, market-hours/execution constraints, or current ticker evidence.

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
If `decision_governance.mean_reversion_dip_buy.best_setup` is present, treat it as a separate conditional buy-lower path, not a momentum Buy. You may return Hold with `entry_style: wait_for_pullback` and `buy_near` at the dip-buy entry when the setup is attractive but current price has not reached it.
Always consider both buy-high/breakout and buy-lower/pullback paths. If the final decision is Hold, say whether the better plan is to wait for a lower entry, wait for breakout confirmation, hold an existing position, reduce risk, or avoid the ticker.
For owned tickers, give sell/trim/take-profit levels and explain whether buying more is allowed, only allowed lower, or blocked.
For not-owned tickers, give the price where buying now is justified, the lower entry zone if waiting for a pullback is better, the stop/invalidation level, and the expected upside/downside scenario.
Never recommend Buy/Sell if the report says new capital is blocked. In that case use Hold and explain the block.

## Initial Analysis Guidelines

Answer every item in `analysis` before the final decision:
1. What is the rule-based forecast/action saying?
2. What are the strongest bullish facts?
3. What are the strongest bearish facts or blockers?
4. What do the news, sentiment, fundamentals, analyst/source context, source synthesis, and long-term provider sections add?
5. What strategy knowledge is applicable, and how does it affect entries, exits, sizing, or patience?
6. Is there a valid buy-now, buy-lower, buy-breakout, sell/trim, or hold-only setup?
7. What data is missing or weak enough to reduce confidence?

## Final Reasoning Guidelines

After the analysis, write `llm_rationale` and `decision_reasoning` explaining why the final action is the right one from all available evidence. The rationale must explicitly reconcile the model forecast, validation quality, rule gates, long-term source synthesis, original provider sections, strategy knowledge, news/sentiment, portfolio context, and buy-lower path when available.
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
- final advice in plain trading terms
- whether to buy now, wait for a lower pullback, wait for a breakout, sell/trim, hold, or avoid
- price to buy now if justified
- buy-lower/pullback zone and buy-above/breakout level when applicable
- price to consider selling, trimming, taking profit, or buying more when already owned
- stop loss or invalidation level
- take profit level
- what price behavior to expect if the decision is right
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
            "analysis": {
                "type": "object",
                "description": "Structured initial analysis before the final decision.",
                "properties": {
                    "rule_based_forecast_read": {"type": "string"},
                    "bullish_evidence": {"type": "array", "items": {"type": "string"}},
                    "bearish_evidence": {"type": "array", "items": {"type": "string"}},
                    "news_sentiment_fundamental_read": {"type": "string"},
                    "long_term_source_read": {"type": "string"},
                    "strategy_knowledge_read": {"type": "string"},
                    "portfolio_context_read": {"type": "string"},
                    "buy_lower_analysis": {"type": "string"},
                    "execution_or_rule_block_analysis": {"type": "string"},
                    "missing_or_weak_data": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "rule_based_forecast_read",
                    "bullish_evidence",
                    "bearish_evidence",
                    "news_sentiment_fundamental_read",
                    "long_term_source_read",
                    "strategy_knowledge_read",
                    "portfolio_context_read",
                    "buy_lower_analysis",
                    "execution_or_rule_block_analysis",
                    "missing_or_weak_data",
                ],
                "additionalProperties": False,
            },
            "llm_rationale": {
                "type": "string",
                "description": "Final LLM rationale after the structured analysis, reconciling all supplied evidence.",
            },
            "decision_reasoning": {
                "type": "string",
                "description": "Brief reasoning joining technical evidence, market context, and the trader profile.",
            },
            "final_advice": {
                "type": "object",
                "description": "Actionable final advice that can be shown directly in the report and dashboard.",
                "properties": {
                    "headline": {
                        "type": "string",
                        "description": "One-sentence final advice, for example Hold and wait for pullback near X, Buy only above Y, or Sell/trim near Z.",
                    },
                    "action_now": {
                        "type": "string",
                        "enum": ["buy_now", "sell_now", "hold", "wait", "avoid", "trim_or_reduce"],
                    },
                    "not_owned_plan": {
                        "type": "string",
                        "description": "What to do if the ticker is not currently owned.",
                    },
                    "owned_plan": {
                        "type": "string",
                        "description": "What to do if the ticker is already owned, including sell/trim/buy-more logic.",
                    },
                    "buy_now_price": {"type": ["number", "null"]},
                    "buy_lower_price": {"type": ["number", "null"]},
                    "buy_lower_zone_low": {"type": ["number", "null"]},
                    "buy_lower_zone_high": {"type": ["number", "null"]},
                    "buy_above_breakout_price": {"type": ["number", "null"]},
                    "buy_more_price": {"type": ["number", "null"]},
                    "sell_or_trim_price": {"type": ["number", "null"]},
                    "take_profit_price": {"type": ["number", "null"]},
                    "stop_loss_price": {"type": ["number", "null"]},
                    "invalidation_price": {"type": ["number", "null"]},
                    "expected_base_case": {
                        "type": "string",
                        "description": "What price behavior is expected if the decision is correct.",
                    },
                    "expected_bull_case": {"type": "string"},
                    "expected_bear_case": {"type": "string"},
                    "why_not_buy_now": {
                        "type": "string",
                        "description": "If not buying now, explain the specific blocker. Use an empty string if buying now.",
                    },
                    "why_not_sell_now": {
                        "type": "string",
                        "description": "If not selling now, explain why. Use an empty string if selling now.",
                    },
                },
                "required": [
                    "headline",
                    "action_now",
                    "not_owned_plan",
                    "owned_plan",
                    "buy_now_price",
                    "buy_lower_price",
                    "buy_lower_zone_low",
                    "buy_lower_zone_high",
                    "buy_above_breakout_price",
                    "buy_more_price",
                    "sell_or_trim_price",
                    "take_profit_price",
                    "stop_loss_price",
                    "invalidation_price",
                    "expected_base_case",
                    "expected_bull_case",
                    "expected_bear_case",
                    "why_not_buy_now",
                    "why_not_sell_now",
                ],
                "additionalProperties": False,
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
            "analysis",
            "llm_rationale",
            "decision_reasoning",
            "final_advice",
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
