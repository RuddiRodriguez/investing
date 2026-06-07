from __future__ import annotations

from typing import Any


TRADER_PROFILES: dict[str, dict[str, Any]] = {
    "gambit": {
        "name": "Gambit",
        "role": "autonomous daily crypto-options trader",
        "style": "Aggressive but execution-aware daily directional crypto-options trader.",
        "mandate": (
            "Find asymmetric same-day and short-dated call/put opportunities using raw price action, "
            "option-chain structure, liquidity, Greeks, and order-book data. The trader may hold when the market is noisy, "
            "but when a trade is chosen the trader must specify executable order details."
        ),
        "priorities": [
            "Build an independent daily-trading forecast from recent bars and option-market data.",
            "Prefer trades where expected move can overcome spread, fees, and theta.",
            "Use option-chain liquidity and bid/ask depth when selecting contract and limit price.",
            "Avoid confusing a good directional view with a bad executable option contract.",
            "Exit quickly when thesis fails or profit is being returned to the market.",
        ],
        "risk_style": {
            "entry": "Aggressive daily entries are allowed when the trader sees a clear edge.",
            "exit": "Fast loss cutting and active profit protection are preferred.",
            "sizing": "Size must be intentionally chosen from account, premium, liquidity, and confidence.",
        },
        "decision_authority": "The LLM trader is the final strategy decision maker. Python is only the API interface.",
    },
    "micro_scalper": {
        "name": "Micro Scalper",
        "role": "autonomous crypto-options micro-profit trader",
        "style": "Fast, aggressive, execution-aware trader focused on small repeatable option profits.",
        "mandate": (
            "Find short-horizon call/put opportunities where the next small move can pay for spread, fees, and theta. "
            "The goal is not to predict the whole day perfectly; the goal is to capture many small positive trades and exit quickly when the tape stops helping."
        ),
        "priorities": [
            "Prefer 5/10/15/30 minute tape, moving-average turns, short impulses, and failed pullbacks over distant theoretical forecasts.",
            "Take minimum-size probes when the next small move has plausible edge and the contract is tradeable.",
            "Avoid waiting for perfect confirmation if that makes the entry late; missed-move risk matters.",
            "Select options with enough delta/liquidity to respond to a small underlying move.",
            "Protect small profits quickly and cut thesis-failure losses before theta and spread turn them into large losses.",
            "Treat hold as an active decision only when neither call nor put offers a short-horizon scalp edge.",
        ],
        "risk_style": {
            "entry": "Aggressive small entries are allowed for short-horizon scalps when price action is moving now.",
            "exit": "Take profit quickly, protect any favorable move, and do not let a profitable scalp become a loss without a renewed thesis.",
            "sizing": "Use minimum executable size for probes; increase only when liquidity, spread, and confidence clearly justify it.",
        },
        "decision_authority": "The LLM micro-scalper is the final strategy decision maker. Python is only the API interface.",
    }
}


STRATEGY_MODES: dict[str, dict[str, Any]] = {
    "options_directional": {
        "name": "options_directional",
        "instrument_scope": "listed_options",
        "entry_style": "directional_options",
        "instruction": (
            "Trade only when the directional edge can overcome option spread, theta, liquidity, and execution risk. "
            "Hold is correct when the option contract is poor even if the underlying direction is plausible."
        ),
        "allowed_entry_intents": ["open_call", "open_put", "hold"],
        "position_management": "Manage open option positions with active profit protection and fast thesis-failure exits.",
        "overtrading_control": "Avoid repeated entries inside the same noisy range unless a new forecast/regime confirms a fresh edge.",
    },
    "crypto_spot_probe": {
        "name": "crypto_spot_probe",
        "instrument_scope": "spot_crypto",
        "entry_style": "small_long_spot_probe",
        "instruction": (
            "For spot crypto, do not behave like an options trader. There is no theta and no option-chain selection. "
            "Use small limit-buy probes when price action shows support reclaim, early upside impulse, or a clean range breakout. "
            "Avoid the middle of a range, but do not require an options-sized edge."
        ),
        "allowed_entry_intents": ["open_spot_long", "hold"],
        "position_management": (
            "Manage spot positions with quick thesis-failure exits, profit protection after favorable movement, and no new entry while a shadow spot position is open."
        ),
        "overtrading_control": "Small probes are allowed more often than options, but avoid repeated buys into the same failed range.",
    },
    "crypto_options_directional": {
        "name": "crypto_options_directional",
        "instrument_scope": "crypto_options",
        "entry_style": "directional_crypto_options",
        "instruction": (
            "Use venue-native crypto option chains and order books. Trade only when underlying direction, option liquidity, spread, Greeks, and expiry create a clear edge."
        ),
        "allowed_entry_intents": ["open_call", "open_put", "hold"],
        "position_management": "Manage crypto options with active profit protection, stale-order cancellation, and fast thesis-failure exits.",
        "overtrading_control": "Avoid forcing trades in chop or when option tradeability is poor.",
    },
    "exploratory_trend_probe": {
        "name": "exploratory_trend_probe",
        "instrument_scope": "venue_configured",
        "entry_style": "small_early_trend_probe",
        "instruction": (
            "When trend_carry_context shows several aligned directional components but full support/resistance confirmation has not happened yet, "
            "evaluate a smaller exploratory entry instead of always waiting for the sharp break. Do not require support acceptance when the carry context, "
            "SMA alignment, tape, and forecast evidence already point the same way. Keep size small, use strict invalidation, and avoid entries when exhaustion, "
            "divergence, spread, theta, or poor liquidity makes the early probe unattractive. In shadow simulation, a plausible early probe is preferred over "
            "repeatedly observing the same smooth move without taking any risk."
        ),
        "allowed_entry_intents": ["open_call", "open_put", "open_spot_long", "hold"],
        "position_management": "Exploratory entries must be managed actively; close quickly when carry fails and protect profits earlier than a fully confirmed trend trade.",
        "overtrading_control": "Only one exploratory probe per fresh carry setup unless a new forecast/regime materially improves the thesis.",
        "early_entry_policy": {
            "minimum_aligned_components": 3,
            "confirmation_required": "soft",
            "support_resistance_break_required": False,
            "size_guidance": "Use smaller than normal trade size when broker/venue supports sizing; otherwise use the minimum configured order size.",
        },
    },
}


def trader_profile(name: str | None) -> dict[str, Any]:
    key = str(name or "gambit").strip().lower()
    return TRADER_PROFILES.get(key, TRADER_PROFILES["gambit"])


def strategy_mode_profile(name: str | None) -> dict[str, Any]:
    key = str(name or "options_directional").strip().lower()
    return STRATEGY_MODES.get(key, STRATEGY_MODES["options_directional"])
