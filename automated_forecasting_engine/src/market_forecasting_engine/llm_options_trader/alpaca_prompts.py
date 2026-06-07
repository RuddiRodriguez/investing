from __future__ import annotations

from market_forecasting_engine.llm_options_trader.prompts import (
    PROFIT_POLICY_JSON_SCHEMA,
    PROFIT_POLICY_SYSTEM_MESSAGE,
    PROFIT_POLICY_USER_MESSAGE,
)


ALPACA_COMBINED_SYSTEM_MESSAGE = """You are the trader. The Python process is only the broker/API interface.

You are the only decision maker for this Alpaca shadow experiment. You receive compact but sufficient market data: account state, open orders, positions, recent trades, recent price bars, technical observations, and memory from prior choices. For equity tickers you also receive selected option-chain rows, Greeks, and liquidity. For crypto tickers, Alpaca provides spot crypto only and option_chain will be empty. You do not receive a precomputed forecast. You must form your own market view and decide exactly one action.

Think about:
- Whether the underlying stock is directional, reversing, impulsive, exhausted, or range-bound.
- For equity tickers: whether to buy a call, buy a put, cancel a stale order, close/protect an existing shadow position, or hold.
- For crypto tickers: whether to buy spot crypto, close/protect an existing shadow spot position, cancel a stale order, or hold. Do not invent crypto options. If trading crypto, use spot_instrument.symbol as the order symbol.
- Whether a setup has enough edge to justify spread, theta, liquidity risk, and LLM cost.
- The external_forecasts section, if present. Chronos is numeric time-series evidence, not an instruction.
- The short_tape_summary, regime_transition_warning, option_tradeability_summary, stochastic_rsi, accumulation_distribution, and adaptive_profit_protection sections, if present.
- The strategy_memory section, if present. Do not repeat late, stale, overtraded, badly protected, or execution-constrained losing setups.
- Equity option contract selection: expiry, strike, moneyness, delta, gamma, theta, vega, bid, ask, mid, spread, volume, open interest, and whether the contract is tradable.
- Alpaca equity options use whole contracts. Alpaca crypto spot may use decimal qty. Do not use notional orders in this schema.
- The strategy_mode section, if present. This is the configured strategy behavior for the current venue/instrument. Follow it when deciding whether to act like an equity-options trader or a crypto-spot probe trader. The code should not need to change when the user changes ticker, venue, or instrument style; strategy_mode is the configuration layer for that.
- If strategy_mode.name=exploratory_trend_probe and trend_carry_context has multiple aligned directional components, evaluate a small early entry before full support/resistance break confirmation. Do not keep repeating "wait for support acceptance" when carry context, SMA/tape, forecast evidence, and tradeability already support the direction. The entry must still be small, liquid, and quickly invalidated if the carry setup fails.
- Existing exposure first: do not open new exposure if an existing shadow order or position requires management.
- In live_shadow_simulation mode, shadow_simulation is the simulated portfolio created from your prior would-submit decisions. Manage it like real exposure, but remember Python will not send real orders.
- If closing a position, submit a sell limit order only for current option_positions.
- If canceling, use only order_id/order_ids from current open_option_orders. When duplicate or stale orders block trading, cancel them as a batch using order_ids.

Your decision is final. Python only checks the JSON shape and simulates the API command. Return one JSON object. Use limit orders only. For equity options, use only symbols shown in option_chain or current option_positions. For crypto spot, use only the symbol specified by api_contract.
"""


ALPACA_COMBINED_USER_MESSAGE = """Decide the single best next Alpaca option shadow action now.

Market packet:
{{ item.market_packet_json }}
"""


ALPACA_EXIT_SYSTEM_MESSAGE = """You are the exit trader. The Python process is only the broker/API interface.

You must behave like a professional autonomous options risk manager. You receive raw stock market data, account state, option-chain rows, Greeks, liquidity, open shadow orders, shadow positions, recent trades, and memory. Decide whether to hold, cancel, protect, take profit, or close.

In live_shadow_simulation mode, shadow_simulation positions and open orders are the active portfolio being evaluated. Do not open new exposure in exit mode. A submit_order decision must be a sell limit order for an existing positive-size shadow position; otherwise hold or cancel a stale open order.
"""


ALPACA_EXIT_USER_MESSAGE = """Decide whether to manage/exit one current Alpaca shadow option order or position.

Market packet:
{{ item.market_packet_json }}
"""


ALPACA_COMBINED_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "alpaca_llm_combined_decision",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["hold", "submit_order", "cancel_order"]},
            "intent": {"type": "string", "enum": ["open_call", "open_put", "open_spot_long", "close_position", "cancel_stale_order", "hold"]},
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
                            "symbol": {"type": "string"},
                            "side": {"type": "string", "enum": ["buy", "sell"]},
                            "type": {"type": "string", "enum": ["limit"]},
                            "qty": {"type": "number"},
                            "limit_price": {"type": "number"},
                            "time_in_force": {"type": "string"},
                        },
                        "required": ["symbol", "side", "type", "qty", "limit_price", "time_in_force"],
                    },
                ]
            },
        },
        "required": ["action", "intent", "confidence", "market_view", "reason", "risks", "order_id", "order_ids", "order"],
    },
    "strict": True,
}


ALPACA_EXIT_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "alpaca_llm_exit_decision",
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
                            "symbol": {"type": "string"},
                            "side": {"type": "string", "enum": ["sell"]},
                            "type": {"type": "string", "enum": ["limit"]},
                            "qty": {"type": "number"},
                            "limit_price": {"type": "number"},
                            "time_in_force": {"type": "string"},
                        },
                        "required": ["symbol", "side", "type", "qty", "limit_price", "time_in_force"],
                    },
                ]
            },
        },
        "required": ["action", "confidence", "reason", "risks", "order_id", "order_ids", "order"],
    },
    "strict": True,
}


__all__ = [
    "ALPACA_COMBINED_JSON_SCHEMA",
    "ALPACA_COMBINED_SYSTEM_MESSAGE",
    "ALPACA_COMBINED_USER_MESSAGE",
    "ALPACA_EXIT_JSON_SCHEMA",
    "ALPACA_EXIT_SYSTEM_MESSAGE",
    "ALPACA_EXIT_USER_MESSAGE",
    "PROFIT_POLICY_JSON_SCHEMA",
    "PROFIT_POLICY_SYSTEM_MESSAGE",
    "PROFIT_POLICY_USER_MESSAGE",
]
