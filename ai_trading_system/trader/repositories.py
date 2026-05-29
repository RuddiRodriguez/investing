import json

from ingestion.db import get_connection, utc_now
from trader.schemas import (
    AggregatePortfolioSnapshot,
    PortfolioState,
    TraderProfile,
    TraderRunLog,
)


def trader_exists(trader_name: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id
        FROM trader_profiles
        WHERE trader_name = ?
        """,
        (trader_name,),
    )
    row = cursor.fetchone()
    conn.close()
    return row is not None


def save_trader_profile(profile: TraderProfile) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO trader_profiles (
            trader_name,
            profile_type,
            initial_cash,
            current_cash,
            max_position_size,
            max_portfolio_exposure,
            min_cash_reserve,
            trade_frequency,
            risk_tolerance,
            status,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            profile.trader_name,
            profile.profile_type,
            profile.initial_cash,
            profile.current_cash,
            profile.max_position_size,
            profile.max_portfolio_exposure,
            profile.min_cash_reserve,
            profile.trade_frequency,
            profile.risk_tolerance,
            profile.status,
            utc_now(),
            utc_now(),
        ),
    )
    conn.commit()
    conn.close()


def get_trader_profile(trader_name: str) -> TraderProfile | None:
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


def create_initial_portfolio_state(profile: TraderProfile) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO portfolio_state (
            trader_name,
            cash,
            invested_value,
            total_portfolio_value,
            open_positions_count,
            last_updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            profile.trader_name,
            profile.initial_cash,
            0.0,
            profile.initial_cash,
            0,
            utc_now(),
        ),
    )
    conn.commit()
    conn.close()


def get_portfolio_state(trader_name: str) -> PortfolioState | None:
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


def set_trader_status(
    trader_name: str,
    status: str,
) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE trader_profiles
        SET status = ?,
            updated_at = ?
        WHERE trader_name = ?
        """,
        (
            status,
            utc_now(),
            trader_name,
        ),
    )
    conn.commit()
    conn.close()


def save_trader_log(log: TraderRunLog) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO trader_run_log (
            trader_name,
            event_type,
            message,
            metadata,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            log.trader_name,
            log.event_type,
            log.message,
            json.dumps(log.metadata) if log.metadata else None,
            utc_now(),
        ),
    )
    conn.commit()
    conn.close()


def build_aggregate_portfolio_snapshot() -> AggregatePortfolioSnapshot:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            COUNT(*) AS total_traders,
            SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) AS active_traders,
            COALESCE(SUM(initial_cash), 0) AS total_initial_cash,
            COALESCE(SUM(current_cash), 0) AS total_current_cash
        FROM trader_profiles
        """
    )
    trader_row = cursor.fetchone()

    cursor.execute(
        """
        SELECT
            COALESCE(SUM(invested_value), 0) AS total_invested_value,
            COALESCE(SUM(total_portfolio_value), 0) AS total_portfolio_value
        FROM portfolio_state
        """
    )
    portfolio_row = cursor.fetchone()
    conn.close()

    return AggregatePortfolioSnapshot(
        total_traders=trader_row["total_traders"],
        active_traders=trader_row["active_traders"] or 0,
        total_initial_cash=trader_row["total_initial_cash"],
        total_current_cash=trader_row["total_current_cash"],
        total_invested_value=portfolio_row["total_invested_value"],
        total_portfolio_value=portfolio_row["total_portfolio_value"],
    )


def save_aggregate_portfolio_snapshot(
    snapshot: AggregatePortfolioSnapshot,
) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO aggregate_portfolio_log (
            total_traders,
            active_traders,
            total_initial_cash,
            total_current_cash,
            total_invested_value,
            total_portfolio_value,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            snapshot.total_traders,
            snapshot.active_traders,
            snapshot.total_initial_cash,
            snapshot.total_current_cash,
            snapshot.total_invested_value,
            snapshot.total_portfolio_value,
            utc_now(),
        ),
    )
    conn.commit()
    conn.close()
