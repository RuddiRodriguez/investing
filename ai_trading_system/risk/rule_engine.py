from risk.schemas import RiskInput, RuleBasedRiskResult


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(value, maximum))


def risk_label(score: float) -> str:
    if score >= 0.80:
        return "extreme"
    if score >= 0.60:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def calculate_rule_based_risk(risk_input: RiskInput) -> RuleBasedRiskResult:
    volatility = risk_input.volatility_20d
    drawdown = risk_input.max_drawdown_60d

    volatility_risk = 0.50
    if volatility is not None:
        volatility_risk = clamp(volatility / 0.80)

    drawdown_risk = 0.50
    if drawdown is not None:
        drawdown_risk = clamp(abs(drawdown) / 0.30)

    confidence_risk = 1.0 - clamp(risk_input.alpha_confidence)
    signal_strength_risk = 1.0 - clamp(abs(risk_input.alpha_score))

    rule_based_risk_score = (
        0.35 * volatility_risk
        + 0.30 * drawdown_risk
        + 0.20 * confidence_risk
        + 0.15 * signal_strength_risk
    )
    rule_based_risk_score = clamp(rule_based_risk_score)

    label = risk_label(rule_based_risk_score)

    rejection_reason = None
    allowed = True
    if risk_input.alpha_confidence < 0.45:
        allowed = False
        rejection_reason = "Alpha confidence is below the minimum threshold."
    elif abs(risk_input.alpha_score) < 0.25:
        allowed = False
        rejection_reason = "Alpha score is too weak."
    elif volatility is not None and volatility > 0.90:
        allowed = False
        rejection_reason = "20-day annualized volatility is too high."
    elif drawdown is not None and drawdown < -0.35:
        allowed = False
        rejection_reason = "60-day drawdown is too severe."

    return RuleBasedRiskResult(
        rule_based_risk_score=rule_based_risk_score,
        rule_based_risk_label=label,
        rule_based_trade_allowed=allowed,
        rule_based_rejection_reason=rejection_reason,
    )
