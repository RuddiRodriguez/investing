import json

from ingestion.db import get_connection, utc_now
from portfolio.schemas import PortfolioInput, PortfolioPosition


def get_portfolio_inputs(sector: str) -> list[PortfolioInput]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            rd.sector,
            rd.ticker,
            rd.company_name,
            rd.signal_date,
            rd.alpha_score,
            rd.alpha_label,
            rd.alpha_confidence,
            rd.llm_risk_score,
            rd.llm_risk_label,
            rd.position_size_multiplier,
            rd.risk_summary,
            rd.decision_reason,
            cas.chart_decision,
            cas.chart_score,
            cas.chart_confidence,
            cas.trend_reading,
            cas.breakout_status,
            cas.volume_confirmation,
            cas.entry_quality,
            cas.support_level,
            cas.resistance_level,
            cas.buy_trigger,
            cas.invalid_buy_reason,
            cas.reason_to_wait,
            cas.current_price_stop_7_pct,
            cas.current_price_stop_8_pct,
            cas.breakout_entry_stop_7_pct,
            cas.breakout_entry_stop_8_pct,
            cas.stop_loss_7_pct,
            cas.stop_loss_8_pct,
            cas.danger_level
        FROM risk_decisions rd
        LEFT JOIN combined_alpha_signals cas
            ON rd.ticker = cas.ticker
           AND rd.signal_date = cas.signal_date
           AND lower(rd.sector) = lower(cas.sector)
        LEFT JOIN portfolio_positions pp
            ON rd.ticker = pp.ticker
           AND rd.signal_date = pp.signal_date
           AND lower(rd.sector) = lower(pp.sector)
        WHERE lower(rd.sector) = lower(?)
          AND rd.llm_trade_allowed = 1
          AND pp.id IS NULL
        ORDER BY ABS(rd.alpha_score) DESC
        """,
        (sector,),
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        PortfolioInput(
            sector=row["sector"],
            ticker=row["ticker"],
            company_name=row["company_name"],
            signal_date=row["signal_date"],
            alpha_score=row["alpha_score"],
            alpha_label=row["alpha_label"],
            alpha_confidence=row["alpha_confidence"],
            llm_risk_score=row["llm_risk_score"],
            llm_risk_label=row["llm_risk_label"],
            position_size_multiplier=row["position_size_multiplier"],
            risk_summary=row["risk_summary"],
            decision_reason=row["decision_reason"],
            chart_decision=row["chart_decision"],
            chart_score=row["chart_score"],
            chart_confidence=row["chart_confidence"],
            trend_reading=row["trend_reading"],
            breakout_status=row["breakout_status"],
            volume_confirmation=row["volume_confirmation"],
            entry_quality=row["entry_quality"],
            support_level=row["support_level"],
            resistance_level=row["resistance_level"],
            buy_trigger=row["buy_trigger"],
            invalid_buy_reason=row["invalid_buy_reason"],
            reason_to_wait=row["reason_to_wait"],
            current_price_stop_7_pct=row["current_price_stop_7_pct"],
            current_price_stop_8_pct=row["current_price_stop_8_pct"],
            breakout_entry_stop_7_pct=row["breakout_entry_stop_7_pct"],
            breakout_entry_stop_8_pct=row["breakout_entry_stop_8_pct"],
            stop_loss_7_pct=row["stop_loss_7_pct"],
            stop_loss_8_pct=row["stop_loss_8_pct"],
            danger_level=row["danger_level"],
        )
        for row in rows
    ]


def save_portfolio_position(position: PortfolioPosition) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO portfolio_positions (
            sector,
            ticker,
            company_name,
            signal_date,
            alpha_score,
            alpha_label,
            alpha_confidence,
            llm_risk_score,
            llm_risk_label,
            position_size_multiplier,
            suggested_direction,
            suggested_position_size,
            llm_direction,
            llm_portfolio_action,
            llm_position_size,
            llm_confidence,
            buy_probability,
            portfolio_reason,
            portfolio_flags,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            position.sector,
            position.ticker,
            position.company_name,
            position.signal_date,
            position.alpha_score,
            position.alpha_label,
            position.alpha_confidence,
            position.llm_risk_score,
            position.llm_risk_label,
            position.position_size_multiplier,
            position.suggested_direction,
            position.suggested_position_size,
            position.llm_direction,
            position.llm_portfolio_action,
            position.llm_position_size,
            position.llm_confidence,
            position.buy_probability,
            position.portfolio_reason,
            json.dumps(position.portfolio_flags),
            utc_now(),
        ),
    )
    position_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return position_id
