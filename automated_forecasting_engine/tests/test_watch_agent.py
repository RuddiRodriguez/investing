import json
from argparse import Namespace
from pathlib import Path

from market_forecasting_engine.watch_agent.cli import (
    apply_portfolio_context_to_args,
    append_decision_log,
    asset_class_for_ticker,
    decide_action,
    find_decision_file,
    latest_alpaca_price_snapshot,
    load_portfolio_context,
    load_decision,
    memory_file,
    memory_needs_refresh,
    resolve_live_price_provider,
    resolve_decision_file,
    should_print_action,
    startup_output_dir,
    update_last_check_memory,
    write_memory,
)
from market_forecasting_engine.watch_agent.dashboard import read_watch_logs
from market_forecasting_engine import data_providers


def _decision():
    return {
        "decision": "Buy",
        "confidence": 0.86,
        "entry_plan": {
            "entry_style": "do_not_enter",
            "buy_near": 1505.22,
            "buy_above": 1653.53,
            "sell_near": 1653.53,
            "stop_loss": 1245.39,
            "take_profit": 1762.87,
        },
    }


def _hold_decision():
    decision = _decision()
    decision["decision"] = "Hold"
    return decision


def test_watch_agent_alerts_buy_when_not_owned_and_buy_near_reached() -> None:
    action, reason = decide_action(_decision(), price=1500.0, holding_status="not_owned")

    assert action == "BUY"
    assert reason == "BUY_NEAR_REACHED"


def test_watch_agent_does_not_buy_when_llm_holds_even_if_buy_near_reached() -> None:
    action, reason = decide_action(_hold_decision(), price=1500.0, holding_status="not_owned")

    assert action == "HOLD"
    assert reason == "BUY_NEAR_REACHED_BUT_LLM_NOT_BUY"


def test_watch_agent_does_not_buy_when_llm_holds_even_if_breakout_reached() -> None:
    action, reason = decide_action(_hold_decision(), price=1660.0, holding_status="not_owned")

    assert action == "HOLD"
    assert reason == "BUY_ABOVE_REACHED_BUT_LLM_NOT_BUY"


def test_watch_agent_holds_stock_alerts_when_market_is_closed() -> None:
    action, reason = decide_action(
        _decision(),
        price=1500.0,
        holding_status="not_owned",
        market_status={"asset_class": "stock", "market_status": "market_closed", "is_tradable": False},
    )

    assert action == "HOLD"
    assert reason == "MARKET_CLOSED"


def test_watch_agent_allows_crypto_alerts_when_market_is_24_7() -> None:
    action, reason = decide_action(
        _decision(),
        price=1500.0,
        holding_status="not_owned",
        market_status={"asset_class": "crypto", "market_status": "crypto_24_7", "is_tradable": True},
    )

    assert action == "BUY"
    assert reason == "BUY_NEAR_REACHED"


def test_watch_agent_classifies_crypto_tickers() -> None:
    assert asset_class_for_ticker("ETH-USD") == "crypto"
    assert asset_class_for_ticker("BTC-USD") == "crypto"
    assert asset_class_for_ticker("ASML") == "stock"


def test_watch_agent_auto_live_provider_prefers_alpaca_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key-id")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")

    assert resolve_live_price_provider("auto") == "alpaca"


def test_watch_agent_auto_live_provider_falls_back_to_yahoo_without_alpaca(monkeypatch) -> None:
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)

    assert resolve_live_price_provider("auto") == "yahoo"


def test_watch_agent_alpaca_live_snapshot_uses_provider_layer(monkeypatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key-id")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")
    monkeypatch.setattr("market_forecasting_engine.watch_agent.cli._stock_market_is_open", lambda calendar="XNYS": (True, "open"))

    def fake_get_json(url: str, headers=None):
        assert "/v2/stocks/ASML/bars?" in url
        assert "timeframe=1Min" in url
        return {
            "bars": [
                {"t": "2026-05-29T13:30:00Z", "o": 100.0, "h": 101.0, "l": 99.5, "c": 100.5, "v": 12345},
                {"t": "2026-05-29T13:31:00Z", "o": 100.5, "h": 102.0, "l": 100.0, "c": 101.5, "v": 23456},
            ]
        }

    monkeypatch.setattr(data_providers, "_get_json", fake_get_json)

    snapshot = latest_alpaca_price_snapshot("ASML", interval="1m")

    assert snapshot["price"] == 101.5
    assert snapshot["price_provider"] == "alpaca"
    assert snapshot["asset_class"] == "stock"
    assert snapshot["market_status"] == "market_open"


def test_watch_dashboard_reads_latest_record_per_ticker_profile(tmp_path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "ASML_medium_20260528.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ticker": "ASML",
                        "profile": "medium",
                        "checked_at": "2026-05-28T10:00:00+00:00",
                        "action": "HOLD",
                        "price": 1600,
                    }
                ),
                json.dumps(
                    {
                        "ticker": "ASML",
                        "profile": "medium",
                        "checked_at": "2026-05-28T11:00:00+00:00",
                        "action": "BUY",
                        "price": 1610,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    (log_dir / "ETH-USD_medium_20260528.jsonl").write_text(
        json.dumps(
            {
                "ticker": "ETH-USD",
                "profile": "medium",
                "checked_at": "2026-05-28T10:30:00+00:00",
                "action": "SELL",
                "price": 2000,
            }
        ),
        encoding="utf-8",
    )

    state = read_watch_logs(log_dir)
    latest = {row["ticker"]: row for row in state["latest"]}

    assert latest["ASML"]["action"] == "BUY"
    assert latest["ASML"]["price"] == 1610
    assert latest["ETH-USD"]["action"] == "SELL"
    assert len(state["histories"]["ASML_medium"]) == 2


def test_watch_dashboard_shows_portfolio_placeholders_before_first_log(tmp_path) -> None:
    log_dir = tmp_path / "logs"
    context_dir = tmp_path / "portfolio_contexts"
    log_dir.mkdir()
    context_dir.mkdir()
    (context_dir / "nvd.de_medium.json").write_text(
        json.dumps(
            {
                "broker": "trade_republic",
                "name": "NVIDIA",
                "isin": "US67066G1040",
                "ticker": "NVD.DE",
                "listing": {"price_provider": "yahoo"},
                "position": {
                    "holding_status": "owned",
                    "quantity": 0.059796,
                    "avg_cost": 195.8304,
                    "current_price": 195.04,
                    "current_value": 11.66,
                    "unrealized_pl": -0.049,
                    "unrealized_pl_pct": -0.42,
                },
            }
        ),
        encoding="utf-8",
    )

    state = read_watch_logs(log_dir)

    assert len(state["latest"]) == 1
    row = state["latest"][0]
    assert row["ticker"] == "NVD.DE"
    assert row["profile"] == "medium"
    assert row["action"] == "STARTING"
    assert row["portfolio_name"] == "NVIDIA"
    assert row["portfolio_entry_price"] == 195.8304


def test_watch_agent_alerts_buy_when_not_owned_and_breakout_reached() -> None:
    action, reason = decide_action(_decision(), price=1660.0, holding_status="not_owned")

    assert action == "BUY"
    assert reason == "BUY_ABOVE_REACHED"


def test_watch_agent_holds_when_not_owned_and_stop_is_broken() -> None:
    action, reason = decide_action(_decision(), price=1200.0, holding_status="not_owned")

    assert action == "HOLD"
    assert reason == "DO_NOT_ENTER_STOP_OR_INVALIDATION_BROKEN"


def test_watch_agent_alerts_sell_when_owned_and_stop_is_reached() -> None:
    action, reason = decide_action(_decision(), price=1240.0, holding_status="owned")

    assert action == "SELL"
    assert reason == "STOP_LOSS_REACHED"


def test_watch_agent_alerts_sell_when_owned_and_take_profit_is_reached() -> None:
    action, reason = decide_action(_decision(), price=1800.0, holding_status="owned")

    assert action == "SELL"
    assert reason == "TAKE_PROFIT_REACHED"


def test_watch_agent_loads_full_trader_decision_file(tmp_path) -> None:
    run_dir = tmp_path / "asml_llm_trader"
    run_dir.mkdir()
    decision_file = run_dir / "trader_decision.json"
    decision_file.write_text(json.dumps({"llm_decision": _decision()}), encoding="utf-8")

    found = find_decision_file("ASML", run_dir=run_dir)
    loaded = load_decision(found)

    assert found == decision_file
    assert loaded["entry_plan"]["buy_near"] == 1505.22


def test_watch_agent_reuses_memory_without_refresh(tmp_path) -> None:
    run_dir = tmp_path / "stored_run"
    run_dir.mkdir()
    decision_file = run_dir / "trader_decision.json"
    decision_file.write_text(json.dumps({"llm_decision": _decision()}), encoding="utf-8")
    args = _args(tmp_path)
    state_path = memory_file(args.state_dir, args.ticker, args.profile)
    write_memory(state_path, args, decision_file, "startup_forecast_llm_trader")

    resolved, memory = resolve_decision_file(args)

    assert resolved == decision_file
    assert memory["source"] == "startup_forecast_llm_trader"


def test_watch_agent_runs_startup_trader_when_memory_missing(tmp_path, monkeypatch) -> None:
    args = _args(tmp_path)
    args.trader_output_dir = str(tmp_path / "fresh_run")

    def fake_run_autonomous_trader(fake_args):
        output_dir = fake_args.output_dir
        folder = Path(output_dir)
        folder.mkdir(parents=True)
        (folder / "trader_decision.json").write_text(json.dumps({"llm_decision": _decision()}), encoding="utf-8")
        return {"llm_decision": _decision()}

    monkeypatch.setattr("market_forecasting_engine.watch_agent.cli.run_autonomous_trader", fake_run_autonomous_trader)

    resolved, memory = resolve_decision_file(args)

    assert resolved == tmp_path / "fresh_run" / "trader_decision.json"
    assert memory["source"] == "startup_forecast_llm_trader"
    assert memory_file(args.state_dir, args.ticker, args.profile).exists()


def test_watch_agent_force_refresh_ignores_memory(tmp_path, monkeypatch) -> None:
    old_run = tmp_path / "old_run"
    old_run.mkdir()
    old_decision = old_run / "trader_decision.json"
    old_decision.write_text(json.dumps({"llm_decision": _decision()}), encoding="utf-8")
    args = _args(tmp_path)
    write_memory(memory_file(args.state_dir, args.ticker, args.profile), args, old_decision, "startup_forecast_llm_trader")
    args.force_refresh = True
    args.trader_output_dir = str(tmp_path / "forced_run")

    def fake_run_autonomous_trader(fake_args):
        folder = Path(fake_args.output_dir)
        folder.mkdir(parents=True)
        (folder / "trader_decision.json").write_text(json.dumps({"llm_decision": _decision()}), encoding="utf-8")
        return {"llm_decision": _decision()}

    monkeypatch.setattr("market_forecasting_engine.watch_agent.cli.run_autonomous_trader", fake_run_autonomous_trader)

    resolved, memory = resolve_decision_file(args)

    assert resolved == tmp_path / "forced_run" / "trader_decision.json"
    assert memory["force_refresh"] is True


def test_watch_agent_memory_refreshes_after_ttl(tmp_path, monkeypatch) -> None:
    old_run = tmp_path / "old_run"
    old_run.mkdir()
    old_decision = old_run / "trader_decision.json"
    old_decision.write_text(json.dumps({"llm_decision": _decision()}), encoding="utf-8")
    args = _args(tmp_path)
    args.refresh_after_hours = 24.0
    stale_memory = write_memory(memory_file(args.state_dir, args.ticker, args.profile), args, old_decision, "startup_forecast_llm_trader")
    stale_memory["created_at_utc"] = "2020-01-01T00:00:00+00:00"
    memory_file(args.state_dir, args.ticker, args.profile).write_text(json.dumps(stale_memory), encoding="utf-8")
    args.trader_output_dir = str(tmp_path / "refreshed_run")

    def fake_run_autonomous_trader(fake_args):
        folder = Path(fake_args.output_dir)
        folder.mkdir(parents=True)
        (folder / "trader_decision.json").write_text(json.dumps({"llm_decision": _decision()}), encoding="utf-8")
        return {"llm_decision": _decision()}

    monkeypatch.setattr("market_forecasting_engine.watch_agent.cli.run_autonomous_trader", fake_run_autonomous_trader)

    resolved, memory = resolve_decision_file(args)

    assert resolved == tmp_path / "refreshed_run" / "trader_decision.json"
    assert memory["source"] == "scheduled_refresh_forecast_llm_trader"


def test_watch_agent_default_startup_output_dir_is_stable(tmp_path) -> None:
    args = _args(tmp_path)

    output_dir = startup_output_dir(args)

    assert output_dir == Path(args.state_dir) / "llm_run" / "ASML_medium"


def test_watch_agent_appends_daily_decision_log(tmp_path) -> None:
    args = _args(tmp_path)
    args.log_dir = str(tmp_path / "logs")
    memory = {
        "source": "startup_forecast_llm_trader",
        "decision_file": str(tmp_path / "trader_decision.json"),
    }

    append_decision_log(args, memory, _decision(), 1500.0, "BUY", "BUY_NEAR_REACHED", True)

    logs = list((tmp_path / "logs").glob("ASML_medium_*.jsonl"))
    assert len(logs) == 1
    record = json.loads(logs[0].read_text().splitlines()[0])
    assert record["action"] == "BUY"
    assert record["reason"] == "BUY_NEAR_REACHED"
    assert record["price"] == 1500.0
    assert record["printed"] is True


def test_watch_agent_appends_portfolio_context_to_decision_log(tmp_path) -> None:
    args = _args(tmp_path)
    args.log_dir = str(tmp_path / "logs")
    args.portfolio_context = {
        "broker": "trade_republic",
        "name": "ASML",
        "isin": "NL0010273215",
        "position": {
            "quantity": 0.25,
            "avg_cost": 650.0,
            "current_value": 170.0,
            "unrealized_pl": 7.5,
            "unrealized_pl_pct": 4.6,
        },
    }
    memory = {
        "source": "startup_forecast_llm_trader",
        "decision_file": str(tmp_path / "trader_decision.json"),
    }

    append_decision_log(args, memory, _decision(), 700.0, "HOLD", "OWNED_NO_SELL_TRIGGER", True)

    logs = list((tmp_path / "logs").glob("ASML_medium_*.jsonl"))
    record = json.loads(logs[0].read_text().splitlines()[0])
    assert record["portfolio_broker"] == "trade_republic"
    assert record["portfolio_name"] == "ASML"
    assert record["portfolio_isin"] == "NL0010273215"
    assert record["portfolio_quantity"] == 0.25
    assert record["portfolio_entry_price"] == 650.0
    assert record["portfolio_position_value"] == 170.0
    assert record["portfolio_unrealized_pl"] == 7.5
    assert record["portfolio_unrealized_pl_pct"] == 4.6


def test_watch_agent_loads_portfolio_context_and_applies_position_defaults(tmp_path) -> None:
    context_path = tmp_path / "portfolio_context.json"
    context_path.write_text(
        json.dumps(
            {
                "broker": "trade_republic",
                "name": "NVIDIA",
                "isin": "US67066G1040",
                "ticker": "NVD.DE",
                "position": {"quantity": 0.059796, "avg_cost": 195.8304, "current_value": 11.66},
                "portfolio_summary": {"total_current_value": 108.45},
            }
        ),
        encoding="utf-8",
    )
    args = _args(tmp_path)
    args.portfolio_context_file = str(context_path)

    args.portfolio_context = load_portfolio_context(args)
    apply_portfolio_context_to_args(args)

    assert args.quantity == 0.059796
    assert args.entry_price == 195.8304
    assert args.position_value == 11.66
    assert args.account_equity == 108.45
    assert "trade_republic" in args.portfolio_notes


def test_watch_agent_quiet_unchanged_suppresses_repeated_hold(tmp_path) -> None:
    args = _args(tmp_path)
    args.quiet_unchanged = True
    memory = {
        "_refreshed_this_run": False,
        "last_check": {
            "action": "HOLD",
            "reason": "WAITING_FOR_ADVICE_LEVEL",
        },
    }

    assert should_print_action(args, memory, "HOLD", "WAITING_FOR_ADVICE_LEVEL") is False


def test_watch_agent_quiet_unchanged_prints_changed_hold(tmp_path) -> None:
    args = _args(tmp_path)
    args.quiet_unchanged = True
    memory = {
        "_refreshed_this_run": False,
        "last_check": {
            "action": "HOLD",
            "reason": "WAITING_FOR_ADVICE_LEVEL",
        },
    }

    assert should_print_action(args, memory, "HOLD", "DO_NOT_ENTER_STOP_OR_INVALIDATION_BROKEN") is True


def test_watch_agent_quiet_unchanged_prints_buy_and_refreshed_forecast(tmp_path) -> None:
    args = _args(tmp_path)
    args.quiet_unchanged = True
    memory = {
        "_refreshed_this_run": False,
        "last_check": {
            "action": "BUY",
            "reason": "BUY_NEAR_REACHED",
        },
    }

    assert should_print_action(args, memory, "BUY", "BUY_NEAR_REACHED") is True
    assert should_print_action(args, {"_refreshed_this_run": True, "last_check": memory["last_check"]}, "HOLD", "WAITING_FOR_ADVICE_LEVEL") is True


def test_watch_agent_updates_last_check_without_resetting_forecast_timestamp(tmp_path) -> None:
    args = _args(tmp_path)
    run_dir = tmp_path / "stored_run"
    run_dir.mkdir()
    decision_file = run_dir / "trader_decision.json"
    decision_file.write_text(json.dumps({"llm_decision": _decision()}), encoding="utf-8")
    state_path = memory_file(args.state_dir, args.ticker, args.profile)
    memory = write_memory(state_path, args, decision_file, "startup_forecast_llm_trader")
    created_at = memory["created_at_utc"]

    update_last_check_memory(args, memory, 1500.0, "BUY", "BUY_NEAR_REACHED", True, tmp_path / "log.jsonl")

    saved = json.loads(state_path.read_text())
    assert saved["created_at_utc"] == created_at
    assert saved["last_check"]["action"] == "BUY"
    assert saved["last_check"]["reason"] == "BUY_NEAR_REACHED"


def test_watch_agent_memory_needs_refresh_for_stale_timestamp() -> None:
    assert memory_needs_refresh({"created_at_utc": "2020-01-01T00:00:00+00:00"}, 24.0) is True


def _args(tmp_path):
    return Namespace(
        ticker="ASML",
        profile="medium",
        run_dir=None,
        decision_file=None,
        state_dir=str(tmp_path / "watch_state"),
        trader_output_dir=None,
        force_refresh=False,
        refresh_after_hours=24.0,
        log_dir=None,
        holding_status="not_owned",
        interval_seconds=3600,
        once=True,
        price=1500.0,
        csv=None,
        provider=None,
        live_price_provider="auto",
        live_price_interval="1m",
        start="2020-01-01",
        end=None,
        interval="1d",
        adjustment_policy="auto_adjust",
        target_column="close",
        horizons="1,5,30",
        selection_metric="mae",
        confidence_level=0.80,
        calendar="XNYS",
        chart_scale="log",
        no_lightgbm=True,
        no_statistical_models=True,
        include_lstm=False,
        trader_name="watch_agent_startup_trader",
        entry_price=None,
        quantity=None,
        position_value=None,
        account_equity=None,
        portfolio_notes="",
        portfolio_context_json=None,
        portfolio_context_file=None,
        write_plots=False,
        dry_run=True,
        llm_model=None,
        reasoning_effort="low",
        summary_model=None,
        summary_reasoning_effort="low",
        llm_timeout=120,
        llm_env_file=None,
        usd_eur_rate=None,
        no_web_search=True,
        no_summary=True,
        search_context_size="medium",
        no_progress=True,
        quiet_unchanged=False,
        prompt="automated_forecasting_engine/src/market_forecasting_engine/llm_trader/prompts/autonomous_trader.py",
        summary_prompt="automated_forecasting_engine/src/market_forecasting_engine/llm_trader/prompts/nontechnical_summary.py",
    )
