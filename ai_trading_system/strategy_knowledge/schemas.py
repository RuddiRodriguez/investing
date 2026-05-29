from pydantic import BaseModel, Field


class StrategyPrinciple(BaseModel):
    name: str
    description: str
    importance: float = Field(ge=0, le=1)


class StrategyRule(BaseModel):
    rule_name: str
    rule_type: str
    description: str
    preferred_condition: str
    avoid_condition: str | None = None
    weight: float = Field(ge=0, le=1)


class StrategyFeature(BaseModel):
    feature_name: str
    description: str
    calculation_hint: str | None = None


class StrategyRiskRule(BaseModel):
    rule_name: str
    description: str
    threshold: float | None = None
    action: str


class StrategyPortfolioRule(BaseModel):
    rule_name: str
    description: str
    profile_effect: str


class ExtractedStrategyKnowledge(BaseModel):
    strategy_name: str
    source_name: str
    source_type: str
    principles: list[StrategyPrinciple]
    rules: list[StrategyRule]
    features: list[StrategyFeature]
    risk_rules: list[StrategyRiskRule]
    portfolio_rules: list[StrategyPortfolioRule]
