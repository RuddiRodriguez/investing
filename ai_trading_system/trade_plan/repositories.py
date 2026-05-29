import json

from ingestion.db import get_connection, utc_now
from trade_plan.schemas import TradePlanInput, TradePlan


def _parse_json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
        return []
    except json.JSONDecodeError:
        return []


def get_trade_plan_inputs(sector: str) -> list[TradePlanInput]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            pp.sector,
            pp.ticker,
            pp.company_name,
            pp.signal_date,
            pp.alpha_score,
            pp.alpha_label,
            pp.alpha_confidence,
            pp.llm_risk_score,
            pp.llm_risk_label,
            pp.position_size_multiplier,
            pp.suggested_direction,
            pp.suggested_position_size,
            pp.llm_direction,
            pp.llm_portfolio_action,
            pp.llm_position_size,
            pp.llm_confidence,
            pp.buy_probability,
            pp.portfolio_reason,
            pp.portfolio_flags
        FROM portfolio_positions pp
                JOIN combined_alpha_signals cas
                        ON pp.ticker = cas.ticker
                     AND pp.signal_date = cas.signal_date
                     AND lower(pp.sector) = lower(cas.sector)
        LEFT JOIN trade_plans tp
            ON pp.ticker = tp.ticker
           AND pp.signal_date = tp.signal_date
           AND lower(pp.sector) = lower(tp.sector)
        WHERE lower(pp.sector) = lower(?)
          AND pp.llm_portfolio_action = 'open'
                    AND pp.llm_direction = 'long'
          AND pp.llm_position_size > 0
                    AND pp.buy_probability >= 0.60
                    AND cas.chart_decision = 'BUY'
                    AND cas.breakout_status = 'confirmed_breakout'
                    AND cas.volume_confirmation = 'strong_volume'
                    AND cas.entry_quality = 'proper_entry'
          AND tp.id IS NULL
        ORDER BY pp.llm_position_size DESC
        """,
        (sector,),
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        TradePlanInput(
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
            suggested_direction=row["suggested_direction"],
            suggested_position_size=row["suggested_position_size"],
            llm_direction=row["llm_direction"],
            llm_portfolio_action=row["llm_portfolio_action"],
            llm_position_size=row["llm_position_size"],
            llm_confidence=row["llm_confidence"],
            buy_probability=row["buy_probability"],
            portfolio_reason=row["portfolio_reason"],
            portfolio_flags=_parse_json_list(row["portfolio_flags"]),
        )
        for row in rows
    ]


def save_trade_plan(trade_plan: TradePlan) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO trade_plans (
            sector,
            ticker,
            company_name,
            signal_date,
            llm_direction,
            llm_portfolio_action,
            llm_position_size,
            buy_probability,
            planned_side,
            planned_order_type,
            planned_time_in_force,
            execution_priority,
            max_slippage_pct,
            trade_plan_status,
            trade_reason,
            execution_notes,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trade_plan.sector,
            trade_plan.ticker,
            trade_plan.company_name,
            trade_plan.signal_date,
            trade_plan.llm_direction,
            trade_plan.llm_portfolio_action,
            trade_plan.llm_position_size,
            trade_plan.buy_probability,
            trade_plan.planned_side,
            trade_plan.planned_order_type,
            trade_plan.planned_time_in_force,
            trade_plan.execution_priority,
            trade_plan.max_slippage_pct,
            trade_plan.trade_plan_status,
            trade_plan.trade_reason,
            trade_plan.execution_notes,
            utc_now(),
        ),
    )
    trade_plan_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return trade_plan_id
