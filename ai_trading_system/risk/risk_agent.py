from risk.llm_risk import evaluate_llm_risk
from risk.repositories import get_risk_inputs, save_risk_decision
from risk.rule_engine import calculate_rule_based_risk
from risk.schemas import LlmRiskDecision, RiskDecision, RiskInput, RuleBasedRiskResult


def build_risk_decision(
    risk_input: RiskInput,
    rule_result: RuleBasedRiskResult,
    llm_decision: LlmRiskDecision,
) -> RiskDecision:
    return RiskDecision(
        sector=risk_input.sector,
        ticker=risk_input.ticker,
        company_name=risk_input.company_name,
        signal_date=risk_input.signal_date,
        alpha_score=risk_input.alpha_score,
        alpha_label=risk_input.alpha_label,
        alpha_confidence=risk_input.alpha_confidence,
        sentiment_score=risk_input.sentiment_score,
        sentiment_label=risk_input.sentiment_label,
        sentiment_confidence=risk_input.sentiment_confidence,
        technical_score=risk_input.technical_score,
        technical_label=risk_input.technical_label,
        technical_confidence=risk_input.technical_confidence,
        volatility_20d=risk_input.volatility_20d,
        max_drawdown_60d=risk_input.max_drawdown_60d,
        return_20d=risk_input.return_20d,
        return_60d=risk_input.return_60d,
        price_vs_sma_20=risk_input.price_vs_sma_20,
        price_vs_sma_50=risk_input.price_vs_sma_50,
        rule_based_risk_score=rule_result.rule_based_risk_score,
        rule_based_risk_label=rule_result.rule_based_risk_label,
        rule_based_trade_allowed=rule_result.rule_based_trade_allowed,
        rule_based_rejection_reason=rule_result.rule_based_rejection_reason,
        llm_risk_score=llm_decision.llm_risk_score,
        llm_risk_label=llm_decision.llm_risk_label,
        llm_trade_allowed=llm_decision.llm_trade_allowed,
        position_size_multiplier=llm_decision.position_size_multiplier,
        risk_flags=llm_decision.risk_flags,
        risk_summary=llm_decision.risk_summary,
        decision_reason=llm_decision.decision_reason,
    )


def run_risk_agent(sector: str) -> dict:
    inputs = get_risk_inputs(sector=sector)

    processed = 0
    allowed = 0
    rejected = 0
    low = 0
    medium = 0
    high = 0
    extreme = 0
    unknown = 0

    for risk_input in inputs:
        rule_result = calculate_rule_based_risk(risk_input)
        llm_decision = evaluate_llm_risk(
            risk_input=risk_input,
            rule_result=rule_result,
        )
        final_decision = build_risk_decision(
            risk_input=risk_input,
            rule_result=rule_result,
            llm_decision=llm_decision,
        )
        save_risk_decision(final_decision)
        processed += 1

        if final_decision.llm_trade_allowed:
            allowed += 1
        else:
            rejected += 1

        if final_decision.llm_risk_label == "low":
            low += 1
        elif final_decision.llm_risk_label == "medium":
            medium += 1
        elif final_decision.llm_risk_label == "high":
            high += 1
        elif final_decision.llm_risk_label == "extreme":
            extreme += 1
        else:
            unknown += 1

    return {
        "sector": sector,
        "processed": processed,
        "allowed": allowed,
        "rejected": rejected,
        "low": low,
        "medium": medium,
        "high": high,
        "extreme": extreme,
        "unknown": unknown,
    }
