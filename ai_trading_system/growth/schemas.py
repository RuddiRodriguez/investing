from pydantic import BaseModel


class GrowthLeadershipMetrics(BaseModel):
    ticker: str
    signal_date: str
    quarterly_earnings_growth: float | None = None
    annual_earnings_growth: float | None = None
    sales_growth: float | None = None
    relative_strength: float | None = None
    price_near_high: float | None = None
    breakout_volume_ratio: float | None = None
    base_pattern_quality: float | None = None


class GrowthLeadershipLlmInterpretation(BaseModel):
    llm_interpretation: str
    main_positive_factors: list[str]
    main_risks: list[str]
    decision_bias: str
