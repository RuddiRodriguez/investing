from __future__ import annotations

import json
import pytest

import pandas as pd

from scripts.advanced_pipeline.config import PipelineConfig
from scripts.advanced_pipeline.paper_agent import (
    DEFAULT_PORTFOLIO_SYMBOL_MAP,
    build_initial_state_from_transactions,
    choose_start_index,
    initialize_state,
    make_run_directory,
    normalize_target_weights,
    portfolio_value,
    rebalance_state,
    run_resumable_agent,
    load_json_file,
    simulate_month,
)


def _prices(periods: int = 320) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=periods)
    return pd.DataFrame(
        {
            "AAA": [100 + step for step in range(periods)],
            "BBB": [100 + step * 0.5 for step in range(periods)],
        },
        index=index,
    )


def _transaction_export(tmp_path) -> str:
    path = tmp_path / "transactions.csv"
    path.write_text(
        "\n".join(
            [
                '"datetime","date","account_type","category","type","asset_class","name","symbol","shares","price","amount","fee","tax","currency","original_amount","original_currency","fx_rate","description","transaction_id","counterparty_name","counterparty_iban","payment_reference","mcc_code"',
                '"2026-05-01T08:00:00Z","2026-05-01","DEFAULT","CASH","TRANSFER_INSTANT_INBOUND","","Funding","","","","100.00","","","EUR","","","","Funding","1","","","",""',
                '"2026-05-02T08:00:00Z","2026-05-02","DEFAULT","TRADING","BUY","STOCK","Alpha Corp","ISINAAA","5","10.0","-50.00","-1.00","","EUR","","","","Buy Alpha","2","","","",""',
                '"2026-05-03T08:00:00Z","2026-05-03","DEFAULT","TRADING","BUY","STOCK","Beta Corp","ISINBBB","2","10.0","-20.00","-1.00","","EUR","","","","Buy Beta","3","","","",""',
                '"2026-05-03T12:00:00Z","2026-05-03","DEFAULT","CASH","CARD_TRANSACTION","","Groceries","","","","-10.00","","","EUR","","","","Groceries","4","","","",""',
                '"2026-05-04T08:00:00Z","2026-05-04","DEFAULT","CASH","DIVIDEND","STOCK","Alpha Corp","ISINAAA","5","","2.00","","","EUR","","","","Dividend","5","","","",""',
                '"2026-05-05T08:00:00Z","2026-05-05","DEFAULT","TRADING","BUY","STOCK","Ignored Corp","ISINZZZ","1","30.0","-30.00","-1.00","","EUR","","","","Ignored","6","","","",""',
            ]
        ),
        encoding="utf-8",
    )
    return str(path)


def _signal_fn(history_prices, history_benchmark, config, fundamentals, news, sectors):
    signal_date = history_prices.index[-1]
    tickers = list(history_prices.columns)
    index = pd.MultiIndex.from_product([[signal_date], tickers], names=["date", "ticker"])
    decisions = pd.DataFrame(
        {
            "decision": ["BUY", "HOLD"],
            "confidence": [0.75, 0.55],
            "expected_excess_return": [0.04, 0.0],
            "lower_bound": [0.01, -0.02],
            "upper_bound": [0.08, 0.02],
            "risk_score": [0.30, 0.20],
            "alpha_score": [0.02, 0.0],
            "main_drivers": [["learned alpha"], ["no strong positive driver"]],
            "main_risks": [["normal model and market risk"], ["normal model and market risk"]],
            "reason_codes": [["expected excess return above buy threshold"], ["signal inside hold zone"]],
        },
        index=index,
    )
    records = [
        {
            "ticker": "AAA",
            "decision": "BUY",
            "confidence": 0.75,
            "expected_excess_return": 0.04,
            "lower_bound": 0.01,
            "upper_bound": 0.08,
            "risk_score": 0.30,
            "alpha_score": 0.02,
            "main_positive_drivers": ["learned alpha"],
            "main_risks": ["normal model and market risk"],
            "reason_codes": ["expected excess return above buy threshold"],
        },
        {
            "ticker": "BBB",
            "decision": "HOLD",
            "confidence": 0.55,
            "expected_excess_return": 0.0,
            "lower_bound": -0.02,
            "upper_bound": 0.02,
            "risk_score": 0.20,
            "alpha_score": 0.0,
            "main_positive_drivers": ["no strong positive driver"],
            "main_risks": ["normal model and market risk"],
            "reason_codes": ["signal inside hold zone"],
        },
    ]
    return {
        "as_of_date": signal_date,
        "regime": "neutral",
        "decisions": decisions,
        "target_weights": pd.Series({"AAA": 0.6, "BBB": 0.0, "CASH": 0.4}),
        "records": records,
        "diagnostics": {},
    }


def test_choose_start_index_respects_warmup_and_horizon() -> None:
    prices = _prices()
    selected = choose_start_index(prices.index, min_history_days=260, simulation_days=21, seed=7)

    assert 260 <= selected <= len(prices.index) - 22


def test_choose_start_index_falls_back_to_latest_available_trading_day() -> None:
    prices = _prices()

    selected = choose_start_index(
        prices.index,
        min_history_days=260,
        simulation_days=21,
        seed=7,
        start_date="2025-01-11",
    )

    assert prices.index[selected] == pd.Timestamp("2025-01-10")


def test_choose_start_index_allows_latest_available_day_for_continuous_mode() -> None:
    prices = _prices()

    selected = choose_start_index(
        prices.index,
        min_history_days=260,
        simulation_days=21,
        seed=7,
        start_date="2026-05-14",
        require_full_horizon=False,
    )

    assert prices.index[selected] == prices.index.max()


def test_choose_start_index_reports_latest_valid_date_for_fixed_horizon() -> None:
    prices = _prices()

    with pytest.raises(ValueError, match="Latest valid start date for a 21-day historical run is"):
        choose_start_index(
            prices.index,
            min_history_days=260,
            simulation_days=21,
            seed=7,
            start_date="2026-05-14",
        )


def test_build_initial_state_from_transactions_uses_mapped_holdings_and_cash(tmp_path) -> None:
    transactions_path = _transaction_export(tmp_path)

    state, portfolio_as_of, metadata = build_initial_state_from_transactions(
        transactions_path=transactions_path,
        tickers=["AAA", "BBB"],
        as_of_date="2026-05-04",
        symbol_map={"ISINAAA": "AAA", "ISINBBB": "BBB"},
    )

    assert str(portfolio_as_of.date()) == "2026-05-04"
    assert state["cash"] == pytest.approx(20.0)
    assert state["units"]["AAA"] == pytest.approx(5.0)
    assert state["units"]["BBB"] == pytest.approx(2.0)
    assert state["avg_cost"]["AAA"] == pytest.approx(10.2)
    assert state["avg_cost"]["BBB"] == pytest.approx(10.5)
    assert metadata["portfolio_seed_positions"]["AAA"]["units"] == pytest.approx(5.0)
    assert metadata["excluded_portfolio_assets"] == []


def test_build_initial_state_from_transactions_records_excluded_assets(tmp_path) -> None:
    transactions_path = _transaction_export(tmp_path)

    _, _, metadata = build_initial_state_from_transactions(
        transactions_path=transactions_path,
        tickers=["AAA", "BBB"],
        as_of_date="2026-05-05",
        symbol_map={"ISINAAA": "AAA", "ISINBBB": "BBB"},
    )

    assert len(metadata["excluded_portfolio_assets"]) == 1
    assert metadata["excluded_portfolio_assets"][0]["source_symbol"] == "ISINZZZ"
    snapshot = metadata["portfolio_seed_snapshot"]
    assert any(row["asset_scope"] == "cash" for row in snapshot)
    assert any(row["asset_scope"] == "tracked" and row["simulation_ticker"] == "AAA" for row in snapshot)
    assert any(row["asset_scope"] == "excluded" and row["source_symbol"] == "ISINZZZ" for row in snapshot)


def test_default_portfolio_symbol_map_includes_etf_holdings(tmp_path) -> None:
    transactions_path = _transaction_export(tmp_path)

    state, _, metadata = build_initial_state_from_transactions(
        transactions_path=transactions_path,
        tickers=["AAA", "BBB", "VWCE.DE"],
        as_of_date="2026-05-05",
        symbol_map={**DEFAULT_PORTFOLIO_SYMBOL_MAP, "ISINAAA": "AAA", "ISINBBB": "BBB", "ISINZZZ": "VWCE.DE"},
    )

    assert state["units"]["VWCE.DE"] == pytest.approx(1.0)
    assert not any(row["source_symbol"] == "ISINZZZ" and row["asset_scope"] == "excluded" for row in metadata["portfolio_seed_snapshot"])


def test_rebalance_state_buys_target_weight() -> None:
    state = initialize_state(["AAA", "BBB"], initial_cash=1000.0)
    weights = normalize_target_weights(pd.Series({"AAA": 0.5, "CASH": 0.5}), ["AAA", "BBB"])
    prices_today = pd.Series({"AAA": 100.0, "BBB": 50.0})

    updated, trades = rebalance_state(state, weights, prices_today, pd.Timestamp("2026-01-02"))

    assert updated["units"]["AAA"] == 5.0
    assert updated["cash"] == 500.0
    assert (trades["action"] == "BUY").any()


def test_simulate_month_writes_artifacts(tmp_path) -> None:
    prices = _prices()
    benchmark = prices["AAA"]
    config = PipelineConfig(tickers=("AAA", "BBB"), benchmark="AAA", cache_dir=tmp_path)
    run_dir = make_run_directory(tmp_path, "demo_run")

    summary = simulate_month(
        config=config,
        market_inputs={
            "asset_prices": prices,
            "benchmark": benchmark,
            "fundamentals": None,
            "news": None,
            "sectors": None,
        },
        run_dir=run_dir,
        initial_cash=10_000.0,
        simulation_days=5,
        seed=7,
        start_date=str(prices.index[260].date()),
        step_seconds=0.0,
        signal_fn=_signal_fn,
    )

    assert (run_dir / "daily_summary.csv").exists()
    assert (run_dir / "trades.csv").exists()
    assert (run_dir / "positions.csv").exists()
    assert (run_dir / "decision_snapshots.csv").exists()
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert summary["completed_steps"] == 5
    daily = pd.read_csv(run_dir / "daily_summary.csv")
    assert len(daily) == 5


def test_run_resumable_agent_can_resume_existing_run(tmp_path) -> None:
    prices = _prices()
    benchmark = prices["AAA"]
    config = PipelineConfig(tickers=("AAA", "BBB"), benchmark="AAA", cache_dir=tmp_path)
    run_dir = make_run_directory(tmp_path, "resume_run")

    first = run_resumable_agent(
        config=config,
        market_inputs={
            "asset_prices": prices,
            "benchmark": benchmark,
            "fundamentals": None,
            "news": None,
            "sectors": None,
        },
        run_dir=run_dir,
        initial_cash=10_000.0,
        simulation_days=2,
        seed=7,
        start_date=str(prices.index[260].date()),
        step_seconds=0.0,
        continuous=False,
        poll_seconds=0.0,
        signal_fn=_signal_fn,
    )

    second = run_resumable_agent(
        config=config,
        market_inputs={
            "asset_prices": prices,
            "benchmark": benchmark,
            "fundamentals": None,
            "news": None,
            "sectors": None,
        },
        run_dir=run_dir,
        initial_cash=10_000.0,
        simulation_days=4,
        seed=7,
        start_date=str(prices.index[260].date()),
        step_seconds=0.0,
        continuous=False,
        poll_seconds=0.0,
        signal_fn=_signal_fn,
    )

    metadata = load_json_file(run_dir / "metadata.json")
    daily = pd.read_csv(run_dir / "daily_summary.csv")

    assert first["completed_steps"] == 2
    assert second["completed_steps"] == 4
    assert metadata["status"] == "completed"
    assert len(daily) == 4