trader_profiles = {
    "aggressive": {
        "name": "aggressive",
        "style": "Fast, opportunistic, accepts volatility, but still respects hard risk gates.",
        "risk_budget": "higher",
        "entry_bias": "can act earlier when technical evidence and fresh market context agree",
        "stop_bias": "uses wider stops only when the chart structure supports it",
        "hold_bias": "does not default to Hold when upside/downside asymmetry is strong",
    },
    "medium": {
        "name": "medium",
        "style": "Balanced trader, combines rule-based technical evidence with fresh market context.",
        "risk_budget": "moderate",
        "entry_bias": "waits for either a fair entry level or confirmation near technical levels",
        "stop_bias": "uses the existing technical stop unless market context clearly argues for tighter risk",
        "hold_bias": "uses Hold when the edge is unclear or execution levels are poor",
    },
    "conservative": {
        "name": "conservative",
        "style": "Capital preservation first, requires stronger alignment and avoids crowded or fragile setups.",
        "risk_budget": "lower",
        "entry_bias": "prefers waiting for pullbacks, breakouts with confirmation, or cleaner risk/reward",
        "stop_bias": "uses tighter invalidation and avoids widening stops",
        "hold_bias": "defaults to Hold when technical, validation, sentiment, or governance evidence conflict",
    },
}
