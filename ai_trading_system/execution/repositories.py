import json

from execution.schemas import ExecutionInput, LatestPrice, TradeExecution
from ingestion.db import get_connection, utc_now
from trader.schemas import PortfolioState, TraderProfile


def get_trade_plans_for_execution(
    sector: str,
    trader_name: str,
) -> list[ExecutionInput]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            tp.id AS trade_plan_id,
            tp.sector,
            tp.ticker,
            tp.company_name,
            tp.signal_date,
            tp.planned_side,
            tp.planned_order_type,
            tp.planned_time_in_force,
            tp.execution_priority,
            tp.max_slippage_pct,
            tp.llm_position_size
        FROM trade_plans tp
        LEFT JOIN trade_executions te
            ON tp.id = te.trade_plan_id
           AND te.trader_name = ?
        WHERE lower(tp.sector) = lower(?)
          AND tp.trade_plan_status = 'planned'
          AND tp.planned_side IN ('buy', 'sell_short')
          AND tp.llm_position_size > 0
          AND te.id IS NULL
        ORDER BY tp.execution_priority DESC, tp.llm_position_size DESC
        """,
        (trader_name, sector),
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        ExecutionInput(
            trade_plan_id=row["trade_plan_id"],
            sector=row["sector"],
            ticker=row["ticker"],
            company_name=row["company_name"],
            signal_date=row["signal_date"],
            planned_side=row["planned_side"],
            planned_order_type=row["planned_order_type"],
            planned_time_in_force=row["planned_time_in_force"],
            execution_priority=row["execution_priority"],
            max_slippage_pct=row["max_slippage_pct"],
            llm_position_size=row["llm_position_size"],
        )
        for row in rows
    ]


def get_latest_price(ticker: str) -> LatestPrice | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT ticker, close, timestamp
        FROM price_bars
        WHERE ticker = ?
          AND close IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        (ticker,),
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return LatestPrice(
        ticker=row["ticker"],
        price=row["close"],
        timestamp=row["timestamp"],
    )


def get_profile_for_execution(trader_name: str) -> TraderProfile | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT *
        FROM trader_profiles
        WHERE trader_name = ?
        """,
        (trader_name,),
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return TraderProfile(
        trader_name=row["trader_name"],
        profile_type=row["profile_type"],
        initial_cash=row["initial_cash"],
        current_cash=row["current_cash"],
        max_position_size=row["max_position_size"],
        max_portfolio_exposure=row["max_portfolio_exposure"],
        min_cash_reserve=row["min_cash_reserve"],
        trade_frequency=row["trade_frequency"],
        risk_tolerance=row["risk_tolerance"],
        status=row["status"],
    )


def get_state_for_execution(trader_name: str) -> PortfolioState | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT *
        FROM portfolio_state
        WHERE trader_name = ?
        """,
        (trader_name,),
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return PortfolioState(
        trader_name=row["trader_name"],
        cash=row["cash"],
        invested_value=row["invested_value"],
        total_portfolio_value=row["total_portfolio_value"],
        open_positions_count=row["open_positions_count"],
    )


def save_trade_execution(execution: TradeExecution) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO trade_executions (
            trader_name,
            trade_plan_id,
            sector,
            ticker,
            company_name,
            signal_date,
            side,
            order_type,
            time_in_force,
            requested_position_size,
            execution_price,
            quantity,
            gross_value,
            simulated_slippage_pct,
            commission,
            llm_execution_status,
            llm_fill_ratio,
            llm_execution_confidence,
            llm_execution_reason,
            llm_execution_flags,
            execution_status,
            execution_reason,
            executed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            execution.trader_name,
            execution.trade_plan_id,
            execution.sector,
            execution.ticker,
            execution.company_name,
            execution.signal_date,
            execution.side,
            execution.order_type,
            execution.time_in_force,
            execution.requested_position_size,
            execution.execution_price,
            execution.quantity,
            execution.gross_value,
            execution.simulated_slippage_pct,
            execution.commission,
            execution.llm_execution_status,
            execution.llm_fill_ratio,
            execution.llm_execution_confidence,
            execution.llm_execution_reason,
            json.dumps(execution.llm_execution_flags),
            execution.execution_status,
            execution.execution_reason,
            utc_now(),
        ),
    )
    execution_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return execution_id


def upsert_holding_after_execution(
    trader_name: str,
    ticker: str,
    company_name: str,
    direction: str,
    quantity: float,
    execution_price: float,
    position_size: float,
) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT *
        FROM portfolio_holdings
        WHERE trader_name = ?
          AND ticker = ?
          AND direction = ?
        """,
        (trader_name, ticker, direction),
    )
    existing = cursor.fetchone()

    gross_value = quantity * execution_price

    if existing is None:
        cursor.execute(
            """
            INSERT INTO portfolio_holdings (
                trader_name,
                ticker,
                company_name,
                direction,
                quantity,
                average_entry_price,
                current_price,
                market_value,
                unrealized_pnl,
                unrealized_pnl_pct,
                position_size,
                opened_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trader_name,
                ticker,
                company_name,
                direction,
                quantity,
                execution_price,
                execution_price,
                gross_value,
                0.0,
                0.0,
                position_size,
                utc_now(),
                utc_now(),
            ),
        )
    else:
        old_quantity = existing["quantity"]
        old_avg = existing["average_entry_price"]
        new_quantity = old_quantity + quantity
        new_avg = (old_quantity * old_avg + quantity * execution_price) / new_quantity
        new_market_value = new_quantity * execution_price
        cursor.execute(
            """
            UPDATE portfolio_holdings
            SET quantity = ?,
                average_entry_price = ?,
                current_price = ?,
                market_value = ?,
                unrealized_pnl = ?,
                unrealized_pnl_pct = ?,
                position_size = ?,
                updated_at = ?
            WHERE trader_name = ?
              AND ticker = ?
              AND direction = ?
            """,
            (
                new_quantity,
                new_avg,
                execution_price,
                new_market_value,
                0.0,
                0.0,
                position_size,
                utc_now(),
                trader_name,
                ticker,
                direction,
            ),
        )
    conn.commit()
    conn.close()


def update_portfolio_state_after_execution(
    trader_name: str,
    cash_change: float,
) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT cash
        FROM portfolio_state
        WHERE trader_name = ?
        """,
        (trader_name,),
    )
    row = cursor.fetchone()
    if row is None:
        conn.close()
        return

    new_cash = row["cash"] + cash_change

    cursor.execute(
        """
        SELECT
            COALESCE(SUM(market_value), 0) AS invested_value,
            COUNT(*) AS open_positions_count
        FROM portfolio_holdings
        WHERE trader_name = ?
        """,
        (trader_name,),
    )
    holdings_row = cursor.fetchone()
    invested_value = holdings_row["invested_value"]
    open_positions_count = holdings_row["open_positions_count"]
    total_portfolio_value = new_cash + invested_value

    cursor.execute(
        """
        UPDATE portfolio_state
        SET cash = ?,
            invested_value = ?,
            total_portfolio_value = ?,
            open_positions_count = ?,
            last_updated_at = ?
        WHERE trader_name = ?
        """,
        (
            new_cash,
            invested_value,
            total_portfolio_value,
            open_positions_count,
            utc_now(),
            trader_name,
        ),
    )
    cursor.execute(
        """
        UPDATE trader_profiles
        SET current_cash = ?,
            updated_at = ?
        WHERE trader_name = ?
        """,
        (
            new_cash,
            utc_now(),
            trader_name,
        ),
    )
    conn.commit()
    conn.close()
