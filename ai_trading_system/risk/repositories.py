import json

from ingestion.db import get_connection, utc_now
from risk.schemas import RiskInput, RiskDecision


def get_risk_inputs(sector: str) -> list[RiskInput]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            cas.sector,
            cas.ticker,
            cas.company_name,
            cas.signal_date,
            cas.alpha_score,
            cas.alpha_label,
            cas.confidence AS alpha_confidence,
            cas.sentiment_score,
            cas.sentiment_label,
            cas.sentiment_confidence,
            cas.technical_score,
            cas.technical_label,
            cas.technical_confidence,
            ts.volatility_20d,
            ts.max_drawdown_60d,
            ts.return_20d,
            ts.return_60d,
            ts.price_vs_sma_20,
            ts.price_vs_sma_50
        FROM combined_alpha_signals cas
        JOIN technical_signals ts
            ON cas.ticker = ts.ticker
           AND cas.signal_date = ts.signal_date
           AND lower(cas.sector) = lower(ts.sector)
        LEFT JOIN risk_decisions rd
            ON cas.ticker = rd.ticker
           AND cas.signal_date = rd.signal_date
           AND lower(cas.sector) = lower(rd.sector)
        WHERE lower(cas.sector) = lower(?)
          AND rd.id IS NULL
        ORDER BY ABS(cas.alpha_score) DESC
        """,
        (sector,),
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        RiskInput(
            sector=row["sector"],
            ticker=row["ticker"],
            company_name=row["company_name"],
            signal_date=row["signal_date"],
            alpha_score=row["alpha_score"],
            alpha_label=row["alpha_label"],
            alpha_confidence=row["alpha_confidence"],
            sentiment_score=row["sentiment_score"],
            sentiment_label=row["sentiment_label"],
            sentiment_confidence=row["sentiment_confidence"],
            technical_score=row["technical_score"],
            technical_label=row["technical_label"],
            technical_confidence=row["technical_confidence"],
            volatility_20d=row["volatility_20d"],
            max_drawdown_60d=row["max_drawdown_60d"],
            return_20d=row["return_20d"],
            return_60d=row["return_60d"],
            price_vs_sma_20=row["price_vs_sma_20"],
            price_vs_sma_50=row["price_vs_sma_50"],
        )
        for row in rows
    ]


def save_risk_decision(decision: RiskDecision) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO risk_decisions (
            sector,
            ticker,
            company_name,
            signal_date,
            alpha_score,
            alpha_label,
            alpha_confidence,
            sentiment_score,
            sentiment_label,
            sentiment_confidence,
            technical_score,
            technical_label,
            technical_confidence,
            volatility_20d,
            max_drawdown_60d,
            return_20d,
            return_60d,
            price_vs_sma_20,
            price_vs_sma_50,
            rule_based_risk_score,
            rule_based_risk_label,
            rule_based_trade_allowed,
            rule_based_rejection_reason,
            llm_risk_score,
            llm_risk_label,
            llm_trade_allowed,
            position_size_multiplier,
            risk_flags,
            risk_summary,
            decision_reason,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            decision.sector,
            decision.ticker,
            decision.company_name,
            decision.signal_date,
            decision.alpha_score,
            decision.alpha_label,
            decision.alpha_confidence,
            decision.sentiment_score,
            decision.sentiment_label,
            decision.sentiment_confidence,
            decision.technical_score,
            decision.technical_label,
            decision.technical_confidence,
            decision.volatility_20d,
            decision.max_drawdown_60d,
            decision.return_20d,
            decision.return_60d,
            decision.price_vs_sma_20,
            decision.price_vs_sma_50,
            decision.rule_based_risk_score,
            decision.rule_based_risk_label,
            1 if decision.rule_based_trade_allowed else 0,
            decision.rule_based_rejection_reason,
            decision.llm_risk_score,
            decision.llm_risk_label,
            1 if decision.llm_trade_allowed else 0,
            decision.position_size_multiplier,
            json.dumps(decision.risk_flags),
            decision.risk_summary,
            decision.decision_reason,
            utc_now(),
        ),
    )
    risk_decision_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return risk_decision_id
