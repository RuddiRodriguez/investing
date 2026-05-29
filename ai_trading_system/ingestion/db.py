import sqlite3
from pathlib import Path
from datetime import datetime, timezone

# Keep one stable DB location regardless of terminal cwd.
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "trading.db"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def add_column_if_missing(cursor, table_name: str, column_name: str, column_definition: str) -> None:
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row["name"] for row in cursor.fetchall()]
    if column_name not in columns:
        cursor.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
        )


def init_db() -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ticker_universe (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sector TEXT NOT NULL,
        rank INTEGER NOT NULL,
        ticker TEXT NOT NULL,
        company_name TEXT NOT NULL,
        exchange TEXT,
        country TEXT,
        industry TEXT,
        relevance_reason TEXT,
        source_url TEXT,
        discovered_at TEXT NOT NULL,
        UNIQUE(sector, ticker)
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS raw_news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sector TEXT NOT NULL,
        ticker TEXT NOT NULL,
        company_name TEXT NOT NULL,
        source TEXT,
        title TEXT NOT NULL,
        url TEXT NOT NULL,
        published_at TEXT,
        raw_summary TEXT,
        fetched_at TEXT NOT NULL,
        UNIQUE(ticker, url)
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS news_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        raw_news_id INTEGER NOT NULL,
        sector TEXT NOT NULL,
        ticker TEXT NOT NULL,
        company_name TEXT NOT NULL,
        title TEXT NOT NULL,
        summary TEXT NOT NULL,
        event_type TEXT NOT NULL,
        affected_direction TEXT NOT NULL,
        relevance_score REAL NOT NULL,
        confidence REAL NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(raw_news_id) REFERENCES raw_news(id)
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS news_relevance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        news_event_id INTEGER NOT NULL,
        ticker TEXT NOT NULL,
        company_name TEXT NOT NULL,
        sector TEXT NOT NULL,
        is_relevant INTEGER NOT NULL,
        relevance_score REAL NOT NULL,
        impact_horizon TEXT NOT NULL,
        affected_scope TEXT NOT NULL,
        reason TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(news_event_id) REFERENCES news_events(id),
        UNIQUE(news_event_id)
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS news_sentiment (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        news_event_id INTEGER NOT NULL,
        ticker TEXT NOT NULL,
        company_name TEXT NOT NULL,
        sector TEXT NOT NULL,
        sentiment_score REAL NOT NULL,
        sentiment_label TEXT NOT NULL,
        magnitude REAL NOT NULL,
        confidence REAL NOT NULL,
        time_horizon TEXT NOT NULL,
        main_driver TEXT NOT NULL,
        risk_flags TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(news_event_id) REFERENCES news_events(id),
        UNIQUE(news_event_id)
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ticker_sentiment_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sector TEXT NOT NULL,
        ticker TEXT NOT NULL,
        company_name TEXT NOT NULL,
        signal_score REAL NOT NULL,
        signal_label TEXT NOT NULL,
        confidence REAL NOT NULL,
        event_count INTEGER NOT NULL,
        positive_count INTEGER NOT NULL,
        negative_count INTEGER NOT NULL,
        neutral_count INTEGER NOT NULL,
        mixed_count INTEGER NOT NULL,
        unknown_count INTEGER NOT NULL,
        strongest_positive_event_id INTEGER,
        strongest_negative_event_id INTEGER,
        summary TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS price_bars (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sector TEXT NOT NULL,
        ticker TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        adjusted_close REAL,
        volume INTEGER,
        source TEXT NOT NULL,
        fetched_at TEXT NOT NULL,
        UNIQUE(ticker, timestamp, timeframe)
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ingestion_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cache_key TEXT UNIQUE NOT NULL,
        cache_type TEXT NOT NULL,
        value TEXT,
        created_at TEXT NOT NULL,
        expires_at TEXT
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS technical_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sector TEXT NOT NULL,
        ticker TEXT NOT NULL,
        signal_date TEXT NOT NULL,
        close REAL NOT NULL,
        return_5d REAL,
        return_20d REAL,
        return_60d REAL,
        volatility_20d REAL,
        sma_20 REAL,
        sma_50 REAL,
        price_vs_sma_20 REAL,
        price_vs_sma_50 REAL,
        max_drawdown_60d REAL,
        technical_score REAL NOT NULL,
        technical_label TEXT NOT NULL,
        confidence REAL NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(ticker, signal_date)
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS combined_alpha_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sector TEXT NOT NULL,
        ticker TEXT NOT NULL,
        company_name TEXT NOT NULL,
        signal_date TEXT NOT NULL,
        sentiment_score REAL,
        sentiment_label TEXT,
        sentiment_confidence REAL,
        technical_score REAL,
        technical_label TEXT,
        technical_confidence REAL,
        oneil_score REAL,
        oneil_label TEXT,
        growth_decision_bias TEXT,
        chart_decision TEXT,
        chart_score REAL,
        chart_confidence REAL,
        breakout_status TEXT,
        volume_confirmation TEXT,
        entry_quality TEXT,
        resistance_level REAL,
        stop_loss_7_pct REAL,
        stop_loss_8_pct REAL,
        danger_level REAL,
        alpha_score REAL NOT NULL,
        alpha_label TEXT NOT NULL,
        confidence REAL NOT NULL,
        main_driver TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(ticker, signal_date)
    )
    """)
    add_column_if_missing(cursor, "combined_alpha_signals", "oneil_score", "REAL")
    add_column_if_missing(cursor, "combined_alpha_signals", "oneil_label", "TEXT")
    add_column_if_missing(cursor, "combined_alpha_signals", "growth_decision_bias", "TEXT")
    add_column_if_missing(cursor, "combined_alpha_signals", "chart_decision", "TEXT")
    add_column_if_missing(cursor, "combined_alpha_signals", "chart_score", "REAL")
    add_column_if_missing(cursor, "combined_alpha_signals", "chart_confidence", "REAL")
    add_column_if_missing(cursor, "combined_alpha_signals", "breakout_status", "TEXT")
    add_column_if_missing(cursor, "combined_alpha_signals", "volume_confirmation", "TEXT")
    add_column_if_missing(cursor, "combined_alpha_signals", "entry_quality", "TEXT")
    add_column_if_missing(cursor, "combined_alpha_signals", "resistance_level", "REAL")
    add_column_if_missing(cursor, "combined_alpha_signals", "stop_loss_7_pct", "REAL")
    add_column_if_missing(cursor, "combined_alpha_signals", "stop_loss_8_pct", "REAL")
    add_column_if_missing(cursor, "combined_alpha_signals", "danger_level", "REAL")
    add_column_if_missing(cursor, "combined_alpha_signals", "trend_reading", "TEXT")
    add_column_if_missing(cursor, "combined_alpha_signals", "support_level", "REAL")
    add_column_if_missing(cursor, "combined_alpha_signals", "buy_trigger", "TEXT")
    add_column_if_missing(cursor, "combined_alpha_signals", "invalid_buy_reason", "TEXT")
    add_column_if_missing(cursor, "combined_alpha_signals", "reason_to_wait", "TEXT")
    add_column_if_missing(cursor, "combined_alpha_signals", "current_price_stop_7_pct", "REAL")
    add_column_if_missing(cursor, "combined_alpha_signals", "current_price_stop_8_pct", "REAL")
    add_column_if_missing(cursor, "combined_alpha_signals", "breakout_entry_stop_7_pct", "REAL")
    add_column_if_missing(cursor, "combined_alpha_signals", "breakout_entry_stop_8_pct", "REAL")
    add_column_if_missing(cursor, "combined_alpha_signals", "stop_loss_7_pct", "REAL")
    add_column_if_missing(cursor, "combined_alpha_signals", "stop_loss_8_pct", "REAL")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chart_confirmation_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sector TEXT NOT NULL,
        ticker TEXT NOT NULL,
        company_name TEXT NOT NULL,
        signal_date TEXT NOT NULL,
        current_price REAL NOT NULL,
        open_price REAL,
        high_price REAL,
        low_price REAL,
        close_price REAL,
        latest_volume REAL,
        trend_status TEXT NOT NULL,
        trend_reading TEXT NOT NULL,
        base_status TEXT NOT NULL,
        support_level REAL,
        resistance_level REAL,
        breakout_status TEXT NOT NULL,
        breakout_price REAL,
        volume_confirmation TEXT NOT NULL,
        volume_ratio REAL,
        entry_quality TEXT NOT NULL,
        extension_pct REAL,
        buy_trigger TEXT NOT NULL,
        invalid_buy_reason TEXT NOT NULL,
        reason_to_wait TEXT NOT NULL,
        current_price_stop_7_pct REAL,
        current_price_stop_8_pct REAL,
        breakout_entry_stop_7_pct REAL,
        breakout_entry_stop_8_pct REAL,
        danger_level REAL,
        sell_signal TEXT NOT NULL,
        chart_decision TEXT NOT NULL,
        chart_score REAL NOT NULL,
        chart_confidence REAL NOT NULL,
        llm_chart_reason TEXT NOT NULL,
        chart_flags TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(ticker, signal_date)
    )
    """)
    add_column_if_missing(cursor, "chart_confirmation_signals", "open_price", "REAL")
    add_column_if_missing(cursor, "chart_confirmation_signals", "high_price", "REAL")
    add_column_if_missing(cursor, "chart_confirmation_signals", "low_price", "REAL")
    add_column_if_missing(cursor, "chart_confirmation_signals", "close_price", "REAL")
    add_column_if_missing(cursor, "chart_confirmation_signals", "latest_volume", "REAL")
    add_column_if_missing(cursor, "chart_confirmation_signals", "trend_reading", "TEXT NOT NULL DEFAULT 'unknown'")
    add_column_if_missing(cursor, "chart_confirmation_signals", "support_level", "REAL")
    add_column_if_missing(cursor, "chart_confirmation_signals", "buy_trigger", "TEXT NOT NULL DEFAULT ''")
    add_column_if_missing(cursor, "chart_confirmation_signals", "invalid_buy_reason", "TEXT NOT NULL DEFAULT ''")
    add_column_if_missing(cursor, "chart_confirmation_signals", "reason_to_wait", "TEXT NOT NULL DEFAULT ''")
    add_column_if_missing(cursor, "chart_confirmation_signals", "current_price_stop_7_pct", "REAL")
    add_column_if_missing(cursor, "chart_confirmation_signals", "current_price_stop_8_pct", "REAL")
    add_column_if_missing(cursor, "chart_confirmation_signals", "breakout_entry_stop_7_pct", "REAL")
    add_column_if_missing(cursor, "chart_confirmation_signals", "breakout_entry_stop_8_pct", "REAL")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS risk_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sector TEXT NOT NULL,
        ticker TEXT NOT NULL,
        company_name TEXT NOT NULL,
        signal_date TEXT NOT NULL,
        alpha_score REAL NOT NULL,
        alpha_label TEXT NOT NULL,
        alpha_confidence REAL NOT NULL,
        sentiment_score REAL,
        sentiment_label TEXT,
        sentiment_confidence REAL,
        technical_score REAL,
        technical_label TEXT,
        technical_confidence REAL,
        volatility_20d REAL,
        max_drawdown_60d REAL,
        return_20d REAL,
        return_60d REAL,
        price_vs_sma_20 REAL,
        price_vs_sma_50 REAL,
        rule_based_risk_score REAL NOT NULL,
        rule_based_risk_label TEXT NOT NULL,
        rule_based_trade_allowed INTEGER NOT NULL,
        rule_based_rejection_reason TEXT,
        llm_risk_score REAL NOT NULL,
        llm_risk_label TEXT NOT NULL,
        llm_trade_allowed INTEGER NOT NULL,
        position_size_multiplier REAL NOT NULL,
        risk_flags TEXT NOT NULL,
        risk_summary TEXT NOT NULL,
        decision_reason TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(ticker, signal_date)
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS portfolio_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sector TEXT NOT NULL,
        ticker TEXT NOT NULL,
        company_name TEXT NOT NULL,
        signal_date TEXT NOT NULL,
        alpha_score REAL NOT NULL,
        alpha_label TEXT NOT NULL,
        alpha_confidence REAL NOT NULL,
        llm_risk_score REAL NOT NULL,
        llm_risk_label TEXT NOT NULL,
        position_size_multiplier REAL NOT NULL,
        suggested_direction TEXT NOT NULL,
        suggested_position_size REAL NOT NULL,
        llm_direction TEXT NOT NULL,
        llm_portfolio_action TEXT NOT NULL,
        llm_position_size REAL NOT NULL,
        llm_confidence REAL NOT NULL,
        buy_probability REAL NOT NULL,
        portfolio_reason TEXT NOT NULL,
        portfolio_flags TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(ticker, signal_date)
    )
    """)
    add_column_if_missing(
        cursor,
        "portfolio_positions",
        "buy_probability",
        "REAL NOT NULL DEFAULT 0.0",
    )
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trade_plans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sector TEXT NOT NULL,
        ticker TEXT NOT NULL,
        company_name TEXT NOT NULL,
        signal_date TEXT NOT NULL,
        llm_direction TEXT NOT NULL,
        llm_portfolio_action TEXT NOT NULL,
        llm_position_size REAL NOT NULL,
        buy_probability REAL NOT NULL,
        planned_side TEXT NOT NULL,
        planned_order_type TEXT NOT NULL,
        planned_time_in_force TEXT NOT NULL,
        execution_priority TEXT NOT NULL,
        max_slippage_pct REAL NOT NULL,
        trade_plan_status TEXT NOT NULL,
        trade_reason TEXT NOT NULL,
        execution_notes TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(ticker, signal_date)
    )
    """)
    add_column_if_missing(
        cursor,
        "trade_plans",
        "buy_probability",
        "REAL NOT NULL DEFAULT 0.0",
    )
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trader_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trader_name TEXT UNIQUE NOT NULL,
        profile_type TEXT NOT NULL,
        initial_cash REAL NOT NULL,
        current_cash REAL NOT NULL,
        max_position_size REAL NOT NULL,
        max_portfolio_exposure REAL NOT NULL,
        min_cash_reserve REAL NOT NULL,
        trade_frequency TEXT NOT NULL,
        risk_tolerance REAL NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS portfolio_state (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trader_name TEXT NOT NULL,
        cash REAL NOT NULL,
        invested_value REAL NOT NULL,
        total_portfolio_value REAL NOT NULL,
        open_positions_count INTEGER NOT NULL,
        last_updated_at TEXT NOT NULL,
        UNIQUE(trader_name),
        FOREIGN KEY(trader_name) REFERENCES trader_profiles(trader_name)
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS portfolio_holdings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trader_name TEXT NOT NULL,
        ticker TEXT NOT NULL,
        company_name TEXT NOT NULL,
        direction TEXT NOT NULL,
        quantity REAL NOT NULL,
        average_entry_price REAL NOT NULL,
        current_price REAL NOT NULL,
        market_value REAL NOT NULL,
        unrealized_pnl REAL NOT NULL,
        unrealized_pnl_pct REAL NOT NULL,
        position_size REAL NOT NULL,
        opened_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(trader_name, ticker, direction),
        FOREIGN KEY(trader_name) REFERENCES trader_profiles(trader_name)
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trader_run_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trader_name TEXT NOT NULL,
        event_type TEXT NOT NULL,
        message TEXT NOT NULL,
        metadata TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(trader_name) REFERENCES trader_profiles(trader_name)
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS aggregate_portfolio_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        total_traders INTEGER NOT NULL,
        active_traders INTEGER NOT NULL,
        total_initial_cash REAL NOT NULL,
        total_current_cash REAL NOT NULL,
        total_invested_value REAL NOT NULL,
        total_portfolio_value REAL NOT NULL,
        created_at TEXT NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trade_executions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trader_name TEXT NOT NULL,
        trade_plan_id INTEGER NOT NULL,
        sector TEXT NOT NULL,
        ticker TEXT NOT NULL,
        company_name TEXT NOT NULL,
        signal_date TEXT NOT NULL,
        side TEXT NOT NULL,
        order_type TEXT NOT NULL,
        time_in_force TEXT NOT NULL,
        requested_position_size REAL NOT NULL,
        execution_price REAL NOT NULL,
        quantity REAL NOT NULL,
        gross_value REAL NOT NULL,
        simulated_slippage_pct REAL NOT NULL,
        commission REAL NOT NULL,
        llm_execution_status TEXT NOT NULL,
        llm_fill_ratio REAL NOT NULL,
        llm_execution_confidence REAL NOT NULL,
        llm_execution_reason TEXT NOT NULL,
        llm_execution_flags TEXT NOT NULL,
        execution_status TEXT NOT NULL,
        execution_reason TEXT NOT NULL,
        executed_at TEXT NOT NULL,
        FOREIGN KEY(trader_name) REFERENCES trader_profiles(trader_name),
        FOREIGN KEY(trade_plan_id) REFERENCES trade_plans(id),
        UNIQUE(trader_name, trade_plan_id)
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS portfolio_valuation_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trader_name TEXT NOT NULL,
        cash REAL NOT NULL,
        invested_value REAL NOT NULL,
        total_portfolio_value REAL NOT NULL,
        unrealized_pnl REAL NOT NULL,
        unrealized_pnl_pct REAL NOT NULL,
        open_positions_count INTEGER NOT NULL,
        valuation_reason TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(trader_name) REFERENCES trader_profiles(trader_name)
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS strategy_knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy_name TEXT NOT NULL,
        source_name TEXT NOT NULL,
        source_type TEXT NOT NULL,
        raw_text TEXT NOT NULL,
        principles_json TEXT NOT NULL,
        rules_json TEXT NOT NULL,
        features_json TEXT NOT NULL,
        risk_rules_json TEXT NOT NULL,
        portfolio_rules_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(strategy_name, source_name)
    )
    """)
    conn.commit()
    conn.close()
