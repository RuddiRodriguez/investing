from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RiskProfile:
    name: str
    minimum_directional_confidence: float
    minimum_edge_fraction: float
    validation_mae_edge_multiplier: float
    maximum_risk_score: float
    options_min_edge_pct: float
    options_max_spread_pct: float
    options_min_probability_breakeven: float
    risk_budget_pct: float
    description: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


RISK_PROFILES: dict[str, RiskProfile] = {
    "conservative": RiskProfile(
        name="conservative",
        minimum_directional_confidence=0.60,
        minimum_edge_fraction=0.0025,
        validation_mae_edge_multiplier=0.35,
        maximum_risk_score=0.70,
        options_min_edge_pct=0.15,
        options_max_spread_pct=0.10,
        options_min_probability_breakeven=0.58,
        risk_budget_pct=0.0025,
        description="Fewest trades; requires stronger confidence, cleaner validation, and cheaper/liquid options.",
    ),
    "medium": RiskProfile(
        name="medium",
        minimum_directional_confidence=0.55,
        minimum_edge_fraction=0.0015,
        validation_mae_edge_multiplier=0.20,
        maximum_risk_score=0.85,
        options_min_edge_pct=0.08,
        options_max_spread_pct=0.18,
        options_min_probability_breakeven=0.52,
        risk_budget_pct=0.0050,
        description="Balanced default; still requires positive edge after validation and option cost.",
    ),
    "aggressive": RiskProfile(
        name="aggressive",
        minimum_directional_confidence=0.52,
        minimum_edge_fraction=0.0010,
        validation_mae_edge_multiplier=0.12,
        maximum_risk_score=0.92,
        options_min_edge_pct=0.03,
        options_max_spread_pct=0.30,
        options_min_probability_breakeven=0.51,
        risk_budget_pct=0.0075,
        description="More speculative paper-trading profile; accepts smaller edges but still requires non-negative risk-adjusted EV.",
    ),
}


def risk_profile_for_name(name: str | None) -> RiskProfile:
    normalized = (name or "medium").strip().lower()
    if normalized not in RISK_PROFILES:
        raise ValueError(f"Unknown risk profile `{name}`. Choose one of: {', '.join(sorted(RISK_PROFILES))}.")
    return RISK_PROFILES[normalized]
