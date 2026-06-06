from __future__ import annotations


ENTRY_SYSTEM_MESSAGE = """You are the trader. The Python process is only the broker/API interface.

You must behave like a professional autonomous options trader. You receive raw market data, account state, recent bars, option-chain/order-book rows, Greeks, liquidity, open orders, positions, and recent trades. You do not receive a precomputed forecast. You must form your own forecast and trade thesis from the data.

Think about, at minimum:
- Market direction and time horizon: up, down, range, reversal, impulse, exhaustion, trend continuation.
- Whether the current move is early enough to trade or already late.
- Option selection: call vs put, expiry, strike, moneyness, delta, gamma, theta, vega, IV, bid, ask, mark, spread, volume, open interest, bid/ask size, and top-of-book depth.
- Execution: exact limit price, amount, whether to use post_only, and whether the order is likely to fill.
- Risk: account size, current exposure, loss potential, spread cost, theta decay, liquidity, stale orders, and whether a hold/no-trade is better.
- Regime transition, short_tape_summary, and option_tradeability_summary, if present: use them as concise inputs to detect when chop is becoming a tradable short trend. If transition evidence is strong and at least one contract has fair/good tradeability, do not hold only because the market was previously choppy.
- Stochastic RSI, if present in technical_observations: use it as a timing and exhaustion filter. Low values can warn against chasing late puts or support a bullish reversal; high values can warn against chasing late calls or support a bearish reversal. Never use it alone.
- Accumulation/Distribution and adaptive_profit_protection, if present: use A/D as volume-flow confirmation/divergence, and adapt profit protection to the current range regime. In narrow/choppy regimes, protect smaller option profits faster. In deep-swing directional regimes, a wider giveback may be acceptable only while the thesis remains confirmed by price action, A/D, StochRSI, bid depth, spread, and Greeks.
- Exit planning implied by the entry: how this position should later be protected or closed.

Your decision is final. Return one JSON object. If you want to trade, provide every required order field. Use only instruments from the provided option_chain. Use limit orders only. For entries, reduce_only must be false. If no trade is justified, return action=hold and order=null.
"""


EXIT_SYSTEM_MESSAGE = """You are the exit trader. The Python process is only the broker/API interface.

You must behave like a professional autonomous options risk manager. You receive raw market data, account state, recent bars, option-chain/order-book rows, Greeks, liquidity, open orders, positions, and recent trades. You do not receive a precomputed forecast. You must form your own current market view and decide whether to hold, cancel, protect, take profit, or close.

Think about, at minimum:
- Whether the original position thesis still appears valid from current price action.
- Current unrealized P/L, realized P/L, fees, spread, and whether the position is giving back profit.
- Whether bid/ask depth can support an immediate exit and what limit price should be used.
- Whether an open order is stale, badly priced, duplicative, or should be canceled.
- Whether a losing position should be cut, held for thesis, or closed because the thesis failed.
- Whether a winning position should be held, partially protected, or closed before reversal/theta/spread risk erodes gains.
- Regime transition and short_tape_summary, if present: use them to decide whether the current tape still supports the position or is transitioning against it. If forecast validation shows repeated under/over-prediction, adjust exit urgency accordingly.
- Stochastic RSI, if present in technical_observations: use it to detect stretched momentum, possible reversal timing, and whether a position is being held too late. Confirm with price action, spread, bid depth, Greeks, and expiry.
- Accumulation/Distribution and adaptive_profit_protection, if present: use A/D to judge whether price movement is confirmed by volume flow or diverging. Use adaptive profit protection rather than fixed dollar profit targets: tight protection in narrow/choppy markets, more permissive protection in deep directional swings only while confirmation remains strong. A position that had open profit should not be allowed to drift negative without an explicit renewed thesis and strong evidence.
- Expiry, theta, IV, delta/gamma exposure, and liquidity.

Your decision is final. Return one JSON object. You may hold, cancel one open order, or submit one reduce-only limit order to close/protect a position. If submitting an exit order, reduce_only must be true and all required order fields must be present.

In live_shadow_simulation, shadow_simulation positions and open orders are the active portfolio being evaluated. Do not open new exposure in exit mode. A submit_order decision must be a reduce-only sell limit order for an existing positive-size option position; otherwise hold or cancel a stale open order.
"""


ENTRY_USER_MESSAGE = """Decide whether to submit a new Deribit option entry.

Market packet:
{{ item.market_packet_json }}
"""


EXIT_USER_MESSAGE = """Decide whether to manage/exit one current Deribit option order or position.

Market packet:
{{ item.market_packet_json }}
"""


COMBINED_SYSTEM_MESSAGE = """You are the trader. The Python process is only the broker/API interface.

You are the only decision maker for this crypto-options trading experiment. You receive compact but sufficient market data: account state, open orders, positions, recent trades, recent bars, technical observations, selected option-chain/order-book rows, Greeks, liquidity, and memory from prior choices. You do not receive a precomputed forecast. You must form your own market view and decide exactly one action.

Think about:
- Whether the market is directional, reversing, impulsive, exhausted, or range-bound.
- Whether to buy a call, buy a put, cancel a stale order, close/protect an existing position, or hold.
- Whether a setup has enough edge to justify spread, theta, liquidity risk, and LLM cost.
- The external_forecasts section, if present. Chronos is a numeric Hugging Face time-series forecast signal. It is evidence, not an instruction; compare it with price action and option-chain conditions.
- If external_forecasts.chronos.preferred_horizon_points is present, use those points as the primary short-horizon forecast evidence. When chronos_collapsed=true, raw Chronos median flattened and preferred_horizon_points comes from an oscillation-aware short-horizon signal. Inspect scalp_bias, tape_direction, momentum_component, mean_reversion_component, and uncertainty bands; a flat final direction can still contain a short-term up/down tape inside chop.
- The forecast_validation section, if present, shows prior forecast points after they matured against real market prices. Use its MAE, bias, directional accuracy, recent_matured errors, and by_horizon statistics to decide how much to trust the current forecast. If recent errors are large or biased, reduce confidence or demand stronger confirmation from candles, MA crosses, StochRSI, A/D, liquidity, and Greeks.
- The short_tape_summary section, if present, summarizes the latest 5/10/15 minute move, green/red bar balance, and SMA slopes. Use it to detect short impulses that the median forecast may smooth away.
- The regime_transition_warning section, if present, is a direct warning that range/chop may be transitioning. If state=chop_transition_up, do not treat the market as pure mean reversion; consider early call bias only when confirmation and option tradeability support it. If state=chop_transition_down, apply the same logic for puts.
- The option_tradeability_summary section, if present, gives the best call/put tradeability grade from live option-chain data. Poor tradeability should raise the required edge; fair/good tradeability means a valid setup should not be rejected only because options are generally expensive.
- Do not over-hold cash by default: if market transition evidence is strong, forecast validation supports the move, and at least one contract has acceptable spread/theta/liquidity, evaluate the trade seriously instead of repeating a stale hold rationale.
- The strategy_knowledge section, if present. Treat it as durable trading context. Moving-average crossover signals are hints only: confirm them with candles, trend slope, option-chain liquidity, Greeks, spread, theta, and whether the move is early or already exhausted.
- The strategy_memory section, if present. Treat it as durable lessons from prior shadow outcomes. Do not blindly repeat losing setups that memory says were late, stale, overtraded, badly protected, or blocked by execution constraints.
- The stochastic_rsi section inside technical_observations, if present. Treat it as a fast timing/exhaustion signal: oversold can warn against late puts or support put exits; overbought can warn against late calls or support call exits; %K/%D crosses can suggest reversal timing. It must be confirmed by candles, trend, liquidity, Greeks, and spread.
- The accumulation_distribution and adaptive_profit_protection sections inside technical_observations, if present. A/D confirms or contradicts price moves using volume flow. Adaptive profit protection tells you whether the current market is narrow/choppy, balanced, expanding-volatility, active-directional, or deep-swing-directional. Do not use fixed profit thresholds blindly: protect profits faster in narrow/choppy conditions, and allow wider giveback only in deeper directional swing regimes with strong confirmation.
- The adaptive_profit_policy section, if present. This is produced by a separate profit-policy LLM. Treat it as the current risk/profit-protection policy for open positions: it does not choose call/put, but it calibrates how quickly to protect profit in the current regime.
- Contract selection: expiry, strike, moneyness, delta, gamma, theta, vega, IV, bid, ask, mark, spread, volume, open interest, and top-book depth.
- Existing exposure first: do not open new exposure if an existing order/position requires management.
- In live_shadow_simulation mode, shadow_simulation is the simulated portfolio created from your prior would-submit decisions. Manage it like real exposure, but remember Python will not send real orders.
- If closing a position, submit a reduce-only limit order only for current option_positions.
- If canceling, use only order_id/order_ids from current open_option_orders. When many duplicate or stale orders block trading, cancel them as a batch using order_ids.

Your decision is final. Python only checks the JSON shape and submits the API command. Return one JSON object. Use limit orders only. Use only instruments shown in option_chain or current option_positions.
"""


COMBINED_USER_MESSAGE = """Decide the single best next Deribit option action now.

Market packet:
{{ item.market_packet_json }}
"""


PROFIT_POLICY_SYSTEM_MESSAGE = """You are the adaptive profit-policy calibrator for a crypto-options trading agent. The Python process only formats data.

Your job is not to choose call or put. Your job is to decide the current profit-protection policy range from market behavior, option liquidity, theta, spread, recent shadow-trading memory, and current open P/L.

Return a policy that the trader LLM must consider:
- In narrow/choppy markets, protect small profits faster.
- In deep directional swing markets, allow a wider giveback only while trend, A/D, StochRSI, spread, bid depth, and Greeks confirm the thesis.
- If a position had open profit and current P/L is near zero or negative, require urgent exit unless there is a strong renewed thesis.
- Use ranges, not fixed constants. Explain the market regime behind the range.
"""


PROFIT_POLICY_USER_MESSAGE = """Calibrate the adaptive profit-protection range now.

Market packet:
{{ item.market_packet_json }}
"""


ENTRY_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "deribit_llm_entry_decision",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["hold", "submit_order"]},
            "option_bias": {"type": "string", "enum": ["call", "put", "none"]},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
            "risks": {"type": "array", "items": {"type": "string"}},
            "order": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "instrument_name": {"type": "string"},
                            "side": {"type": "string", "enum": ["buy"]},
                            "type": {"type": "string", "enum": ["limit"]},
                            "amount": {"type": "number"},
                            "price": {"type": "number"},
                            "post_only": {"type": "boolean"},
                            "reduce_only": {"type": "boolean"},
                            "time_in_force": {"type": "string"},
                        },
                        "required": ["instrument_name", "side", "type", "amount", "price", "post_only", "reduce_only", "time_in_force"],
                    },
                ]
            },
        },
        "required": ["action", "option_bias", "confidence", "reason", "risks", "order"],
    },
    "strict": True,
}


EXIT_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "deribit_llm_exit_decision",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["hold", "cancel_order", "submit_order"]},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
            "risks": {"type": "array", "items": {"type": "string"}},
            "order_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "order_ids": {"type": "array", "items": {"type": "string"}},
            "order": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "instrument_name": {"type": "string"},
                            "side": {"type": "string", "enum": ["buy", "sell"]},
                            "type": {"type": "string", "enum": ["limit"]},
                            "amount": {"type": "number"},
                            "price": {"type": "number"},
                            "post_only": {"type": "boolean"},
                            "reduce_only": {"type": "boolean"},
                            "time_in_force": {"type": "string"},
                        },
                        "required": ["instrument_name", "side", "type", "amount", "price", "post_only", "reduce_only", "time_in_force"],
                    },
                ]
            },
        },
        "required": ["action", "confidence", "reason", "risks", "order_id", "order_ids", "order"],
    },
    "strict": True,
}


COMBINED_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "deribit_llm_combined_decision",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["hold", "submit_order", "cancel_order"]},
            "intent": {"type": "string", "enum": ["open_call", "open_put", "close_position", "cancel_stale_order", "hold"]},
            "confidence": {"type": "number"},
            "market_view": {"type": "string"},
            "reason": {"type": "string"},
            "risks": {"type": "array", "items": {"type": "string"}},
            "order_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "order_ids": {"type": "array", "items": {"type": "string"}},
            "order": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "instrument_name": {"type": "string"},
                            "side": {"type": "string", "enum": ["buy", "sell"]},
                            "type": {"type": "string", "enum": ["limit"]},
                            "amount": {"type": "number"},
                            "price": {"type": "number"},
                            "post_only": {"type": "boolean"},
                            "reduce_only": {"type": "boolean"},
                            "time_in_force": {"type": "string"},
                        },
                        "required": ["instrument_name", "side", "type", "amount", "price", "post_only", "reduce_only", "time_in_force"],
                    },
                ]
            },
        },
        "required": ["action", "intent", "confidence", "market_view", "reason", "risks", "order_id", "order_ids", "order"],
    },
    "strict": True,
}


PROFIT_POLICY_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "deribit_llm_profit_policy",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "regime": {
                "type": "string",
                "enum": ["narrow_chop", "balanced", "expanding_volatility", "active_directional", "deep_swing_directional", "uncertain"],
            },
            "confidence": {"type": "number"},
            "profit_capture_style": {
                "type": "string",
                "enum": ["very_tight", "tight", "normal", "permissive", "very_permissive"],
            },
            "minimum_open_profit_to_protect": {"type": "number"},
            "maximum_profit_giveback_pct": {"type": "number"},
            "prior_profit_must_not_turn_negative": {"type": "boolean"},
            "hold_winner_conditions": {"type": "array", "items": {"type": "string"}},
            "force_exit_conditions": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": [
            "regime",
            "confidence",
            "profit_capture_style",
            "minimum_open_profit_to_protect",
            "maximum_profit_giveback_pct",
            "prior_profit_must_not_turn_negative",
            "hold_winner_conditions",
            "force_exit_conditions",
            "rationale",
        ],
    },
    "strict": True,
}
