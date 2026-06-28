from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from market_forecasting_engine.alpaca_broker import AlpacaPaperBroker
from market_forecasting_engine.long_term_sources import DEFAULT_LONG_TERM_SOURCE_PROVIDERS
from market_forecasting_engine.risk_profiles import risk_profile_for_name
from market_forecasting_engine.virtual_trader_enrichment import (
    VirtualTraderEnrichmentConfig,
    load_selected_candidates,
    run_virtual_trader_enrichment,
)
from market_forecasting_engine.virtual_trader_memory import VirtualTraderMemory, memory_summary_for_prompt


@dataclass(frozen=True)
class VirtualTraderPipelineConfig:
    output_dir: str | Path
    selected_candidates_path: str | Path | None = None
    enrichment_board_path: str | Path | None = None
    memory_path: str | Path = "automated_forecasting_engine/runs/virtual_trader/memory.json"
    max_candidates: int = 3
    provider: str = "yahoo"
    start: str = "2020-01-01"
    interval: str = "1d"
    horizons: str = "5,10,20"
    forecast_backend: str = "full"
    risk_profile: str = "medium"
    trader_profile: str = "medium"
    llm_env_file: str | Path | None = None
    llm_provider: str = "llm_studio"
    llm_model: str | None = None
    ceo_llm_provider: str = "openai"
    ceo_llm_model: str | None = None
    llm_reasoning_effort: str = "none"
    llm_timeout_seconds: int = 120
    llm_search_context_size: str = "medium"
    enable_bayesian_heavy: bool = False
    include_lstm: bool = False
    deep_learning_profile: str = "off"
    tune: str = "fixed"
    optuna_trials: int = 25
    validation_workers: int = 0
    max_notional_per_trade: float = 100.0
    max_position_pct_equity: float = 0.025
    max_total_new_exposure_pct_equity: float = 0.08
    entry_limit_offset_bps: float = 10.0
    sell_limit_offset_bps: float = 10.0
    execute_paper_orders: bool = True
    place_buy_lower_limit_orders: bool = True
    allow_market_closed_orders: bool = False
    allow_repeated_symbol_orders: bool = False
    disable_alpaca: bool = False
    source_synthesis_dry_run: bool = False
    skip_enrichment: bool = False
    progress: bool = True


def run_virtual_trader_pipeline(config: VirtualTraderPipelineConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    _load_env_file(config.llm_env_file)
    memory = VirtualTraderMemory.load(config.memory_path)
    started_at = datetime.now(UTC)
    _progress(config, f"virtual trader pipeline started output={output_dir}")

    broker_state = _load_broker_state(config)
    order_lifecycle_events: list[dict[str, Any]] = []
    if broker_state.get("status") == "ok":
        order_lifecycle_events = memory.broker_snapshot(
            account=broker_state.get("account"),
            positions=broker_state.get("positions", []),
            orders=broker_state.get("open_orders", []),
            recent_orders=broker_state.get("recent_orders", []),
        )
        memory.save()

    enrichment_board = _ensure_enrichment(config, output_dir)
    candidates = _candidate_rows_from_enrichment(enrichment_board, max_candidates=config.max_candidates)
    _progress(config, f"forecasting enriched candidates={','.join(row.get('ticker', '') for row in candidates) or 'none'}")

    forecast_rows = []
    order_rows = []
    for position, candidate in enumerate(candidates, start=1):
        ticker = str(candidate.get("ticker") or "").upper()
        _progress(config, f"forecast {position}/{len(candidates)} {ticker}")
        forecast_result = _run_forecast_for_candidate(candidate, config, output_dir, memory)
        forecast_rows.append(forecast_result)
        order_plan = build_alpaca_paper_order_plan(
            report=forecast_result.get("report", {}),
            candidate=candidate,
            broker_state=broker_state,
            memory=memory,
            config=config,
        )
        order_result = maybe_submit_alpaca_order(order_plan, config)
        order_plan["order_result"] = order_result
        order_rows.append(order_plan)
        memory.record_decision(_decision_memory_record(forecast_result, order_plan))
        memory.record_order(_order_memory_record(order_plan))
        memory.save()

    final_selection = _portfolio_selection(forecast_rows, order_rows, memory)
    board = {
        "run_type": "virtual_trader_active_paper_pipeline",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "config": _config_for_report(config),
        "policy": {
            "autonomous_trader": True,
            "paper_account_default": True,
            "real_money_ready_policy": (
                "This path is built as a paper autonomous trader first. Live-money transfer requires explicit broker mode, "
                "execution flags, risk envelope review, and broker constraint validation."
            ),
            "default_execution": "paper_orders_enabled" if config.execute_paper_orders else "dry_run",
            "independent_paths": "Discovery, ranking, enrichment, forecast CLI, and watch agents can run without virtual trader memory.",
        },
        "broker_state": _broker_state_for_board(broker_state),
        "order_lifecycle_events": order_lifecycle_events,
        "memory": {
            "path": str(Path(config.memory_path).expanduser()),
            "portfolio_context": memory.portfolio_context(),
        },
        "enrichment_board": enrichment_board,
        "forecasts": [_compact_forecast_row(row) for row in forecast_rows],
        "order_plans": order_rows,
        "portfolio_selection": final_selection,
        "artifact_paths": {},
    }
    board["artifact_paths"] = {
        "virtual_trader_board": _write_json(output_dir / "virtual_trader_board.json", board),
        "virtual_trader_board_markdown": str(output_dir / "virtual_trader_board.md"),
    }
    _write_markdown_board(output_dir / "virtual_trader_board.md", board)
    _write_json(output_dir / "virtual_trader_board.json", board)
    _progress(config, "virtual trader pipeline complete")
    return board


def build_alpaca_paper_order_plan(
    *,
    report: dict[str, Any],
    candidate: dict[str, Any],
    broker_state: dict[str, Any],
    memory: VirtualTraderMemory,
    config: VirtualTraderPipelineConfig,
) -> dict[str, Any]:
    ticker = str(report.get("ticker") or candidate.get("ticker") or "").upper()
    decision = report.get("llm_final_decision") if isinstance(report.get("llm_final_decision"), dict) else {}
    final_advice = report.get("final_advice") if isinstance(report.get("final_advice"), dict) else {}
    if not final_advice and isinstance(decision.get("final_advice"), dict):
        final_advice = decision["final_advice"]
    final_advice = dict(final_advice)
    action = str(decision.get("decision") or report.get("suggested_action") or "Hold").lower()
    current_price = _to_float(report.get("current_price"))
    watch_conditions = _watch_conditions(final_advice, current_price)
    final_advice["watch_condition_status"] = watch_conditions
    account = broker_state.get("account") if isinstance(broker_state.get("account"), dict) else {}
    equity = _to_float(account.get("equity") or account.get("portfolio_value")) or 0.0
    buying_power = _to_float(account.get("buying_power")) or 0.0
    position = _position_for_symbol(broker_state.get("positions", []), ticker)
    open_orders = [order for order in broker_state.get("open_orders", []) if str(order.get("symbol") or "").upper() == ticker]
    execution_gate = report.get("decision_view", {}).get("autonomous_execution_gate", {}) if isinstance(report.get("decision_view"), dict) else {}
    blocks = list(execution_gate.get("execution_blocks", []) or []) if isinstance(execution_gate, dict) else []
    warnings = list(execution_gate.get("warnings", []) or []) if isinstance(execution_gate, dict) else []
    if broker_state.get("status") != "ok":
        blocks.append(f"broker_state_unavailable: {broker_state.get('error')}")
    clock = broker_state.get("clock") if isinstance(broker_state.get("clock"), dict) else {}
    if not config.allow_market_closed_orders and clock and not bool(clock.get("is_open")):
        blocks.append("alpaca_market_closed")
    if open_orders and not config.allow_repeated_symbol_orders:
        blocks.append("existing_open_order_for_symbol")
    if action == "hold":
        breakout_status = watch_conditions.get("breakout_buy", {}).get("status")
        buy_lower_status = watch_conditions.get("buy_lower", {}).get("status")
        if breakout_status == "trigger_crossed":
            warnings.append("breakout_price_already_crossed_ceo_requires_confirmation_or_retest")
        if buy_lower_status in {"trigger_crossed", "inside_zone"}:
            warnings.append("buy_lower_zone_already_reached_ceo_still_hold")

    order_payload: dict[str, Any] | None = None
    intent = "no_trade"
    reason = "ceo_action_not_executable"
    if action == "buy":
        intent = "buy_now"
        limit_price = _buy_now_limit_price(final_advice, current_price, config.entry_limit_offset_bps)
        notional = _trade_notional(equity, buying_power, config)
        if position:
            warnings.append("position_already_exists; order sized as add only if not blocked by policy")
        if notional <= 0:
            blocks.append("insufficient_buying_power_or_equity")
        if limit_price is None:
            blocks.append("missing_buy_limit_reference")
        if not blocks and limit_price is not None:
            order_payload = {
                "symbol": ticker,
                "side": "buy",
                "order_type": "limit",
                "notional": round(notional, 2),
                "limit_price": round(limit_price, 2),
                "time_in_force": "day",
                "client_order_id": _client_order_id("vt_buy", ticker),
            }
            reason = "ceo_buy_with_limit_price"
    elif action == "sell":
        intent = "sell_or_trim"
        qty = _position_qty(position)
        limit_price = _sell_limit_price(final_advice, current_price, config.sell_limit_offset_bps)
        if qty <= 0:
            blocks.append("no_owned_position_to_sell")
        if limit_price is None:
            blocks.append("missing_sell_limit_reference")
        if not blocks and limit_price is not None:
            order_payload = {
                "symbol": ticker,
                "side": "sell",
                "order_type": "limit",
                "qty": round(qty, 6),
                "limit_price": round(limit_price, 2),
                "time_in_force": "day",
                "client_order_id": _client_order_id("vt_sell", ticker),
            }
            reason = "ceo_sell_with_limit_price"
    elif config.place_buy_lower_limit_orders:
        buy_lower = _buy_lower_limit_price(final_advice)
        if buy_lower is not None and current_price is not None and buy_lower < current_price:
            intent = "buy_lower_limit_watch"
            notional = _trade_notional(equity, buying_power, config)
            if notional <= 0:
                blocks.append("insufficient_buying_power_or_equity")
            if not blocks:
                order_payload = {
                    "symbol": ticker,
                    "side": "buy",
                    "order_type": "limit",
                    "notional": round(notional, 2),
                    "limit_price": round(buy_lower, 2),
                    "time_in_force": "day",
                    "client_order_id": _client_order_id("vt_dip", ticker),
                }
                reason = _buy_lower_order_reason(watch_conditions)

    if order_payload is None and action == "hold":
        reason = _hold_watch_reason(watch_conditions)
    memory_context = memory.context_for_ticker(ticker)
    return {
        "ticker": ticker,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "action": action,
        "intent": intent,
        "reason": reason,
        "current_price": current_price,
        "candidate_rank": candidate.get("rank"),
        "candidate_ranking_score": candidate.get("ranking_score"),
        "final_advice": final_advice,
        "watch_conditions": watch_conditions,
        "execution_allowed": bool(order_payload and not blocks),
        "execution_blocks": blocks,
        "warnings": warnings,
        "order_payload": order_payload,
        "memory_context": memory_context,
        "dry_run": not config.execute_paper_orders,
        "policy": {
            "no_market_orders_for_entries": True,
            "paper_order_submission_requires_execute_flag": True,
            "buy_lower_limits_require_place_buy_lower_limit_orders": True,
        },
    }


def maybe_submit_alpaca_order(order_plan: dict[str, Any], config: VirtualTraderPipelineConfig) -> dict[str, Any]:
    if not order_plan.get("order_payload"):
        return {"submitted": False, "reason": "no_order_payload"}
    if not order_plan.get("execution_allowed"):
        return {"submitted": False, "reason": "execution_blocked", "blocks": order_plan.get("execution_blocks", [])}
    if not config.execute_paper_orders:
        return {"submitted": False, "reason": "dry_run", "paper_order_payload": order_plan["order_payload"]}
    broker = AlpacaPaperBroker()
    payload = order_plan["order_payload"]
    try:
        response = broker.submit_order(
            symbol=payload["symbol"],
            side=payload["side"],
            order_type=payload["order_type"],
            notional=payload.get("notional"),
            qty=payload.get("qty"),
            limit_price=payload.get("limit_price"),
            time_in_force=payload.get("time_in_force", "day"),
            client_order_id=payload.get("client_order_id"),
        )
    except Exception as exc:
        return {"submitted": False, "reason": "broker_submit_failed", "error": _safe_error(exc), "paper_order_payload": payload}
    return {"submitted": True, "broker_response": response, "paper_order_payload": payload}


def _ensure_enrichment(config: VirtualTraderPipelineConfig, output_dir: Path) -> dict[str, Any]:
    if config.enrichment_board_path:
        return _read_json(Path(config.enrichment_board_path).expanduser())
    if config.selected_candidates_path is None:
        raise ValueError("Pass --selected-candidates or --enrichment-board.")
    if config.skip_enrichment:
        selected = load_selected_candidates(config.selected_candidates_path, max_candidates=config.max_candidates)
        return {
            "run_type": "virtual_trader_selected_candidates_without_enrichment",
            "candidates": [{"ticker": row.get("ticker"), "candidate": row, "artifacts": {}} for row in selected],
            "policy": {"enrichment_skipped": True},
        }
    return run_virtual_trader_enrichment(
        VirtualTraderEnrichmentConfig(
            selected_candidates_path=config.selected_candidates_path,
            output_dir=output_dir / "enrichment",
            max_candidates=config.max_candidates,
            providers=DEFAULT_LONG_TERM_SOURCE_PROVIDERS,
            env_file=config.llm_env_file,
            source_synthesis_dry_run=config.source_synthesis_dry_run,
            llm_model=config.llm_model,
            llm_reasoning_effort=config.llm_reasoning_effort,
            llm_timeout_seconds=float(config.llm_timeout_seconds),
            progress=config.progress,
        )
    )


def _candidate_rows_from_enrichment(enrichment_board: dict[str, Any], *, max_candidates: int) -> list[dict[str, Any]]:
    rows = enrichment_board.get("candidates", []) if isinstance(enrichment_board.get("candidates"), list) else []
    candidates = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else dict(row)
        candidate.setdefault("ticker", row.get("ticker"))
        candidate.setdefault("enrichment_artifacts", row.get("artifacts", {}))
        candidate.setdefault("source_synthesis", row.get("source_synthesis"))
        candidate.setdefault("strategy_knowledge_context", row.get("strategy_knowledge_context"))
        candidates.append(candidate)
    candidates = sorted(candidates, key=lambda item: int(item.get("rank") or 999999))
    return candidates[: max(0, int(max_candidates))]


def _run_forecast_for_candidate(
    candidate: dict[str, Any],
    config: VirtualTraderPipelineConfig,
    output_dir: Path,
    memory: VirtualTraderMemory,
) -> dict[str, Any]:
    ticker = str(candidate.get("ticker") or "").upper()
    ticker_dir = output_dir / "forecasts" / _safe_symbol(ticker)
    ticker_dir.mkdir(parents=True, exist_ok=True)
    portfolio_notes = "\n".join(
        [
            "Virtual trader candidate context:",
            json.dumps(candidate, indent=2, sort_keys=True, default=str)[:5000],
            "Virtual trader memory:",
            memory_summary_for_prompt(memory, ticker)[:5000],
        ]
    )
    if config.forecast_backend == "pure_llm":
        return _run_pure_llm_forecast_for_candidate(
            ticker=ticker,
            ticker_dir=ticker_dir,
            candidate=candidate,
            config=config,
            memory=memory,
            portfolio_notes=portfolio_notes,
        )
    if config.forecast_backend != "full":
        raise ValueError(f"Unsupported forecast_backend={config.forecast_backend!r}. Use 'full' or 'pure_llm'.")
    command = [
        sys.executable,
        "-m",
        "market_forecasting_engine.cli",
        "--ticker",
        ticker,
        "--provider",
        config.provider,
        "--start",
        config.start,
        "--interval",
        config.interval,
        "--horizons",
        config.horizons,
        "--output-dir",
        str(ticker_dir),
        "--trader-profile",
        config.trader_profile,
        "--llm-timeout",
        str(config.llm_timeout_seconds),
        "--llm-search-context-size",
        config.llm_search_context_size,
        "--portfolio-notes",
        portfolio_notes,
        "--validation-workers",
        str(config.validation_workers),
        "--tune",
        config.tune,
        "--optuna-trials",
        str(config.optuna_trials),
    ]
    if config.llm_env_file:
        command.extend(["--llm-env-file", str(config.llm_env_file), "--long-term-source-env-file", str(config.llm_env_file)])
    if config.llm_model:
        command.extend(["--llm-model", config.llm_model])
    command.extend(["--llm-reasoning-effort", config.llm_reasoning_effort])
    if config.enable_bayesian_heavy:
        command.append("--enable-bayesian-heavy")
    if config.include_lstm:
        command.append("--include-lstm")
    if config.deep_learning_profile != "off":
        command.extend(["--deep-learning-profile", config.deep_learning_profile])
    started = datetime.now(UTC)
    result = subprocess.run(command, cwd=Path.cwd(), text=True, capture_output=True, check=False)
    (ticker_dir / "forecast_command.json").write_text(json.dumps({"command": command}, indent=2) + "\n", encoding="utf-8")
    (ticker_dir / "forecast_stdout.txt").write_text(result.stdout or "", encoding="utf-8")
    (ticker_dir / "forecast_stderr.txt").write_text(result.stderr or "", encoding="utf-8")
    report_path = ticker_dir / "forecast_report.json"
    report = _read_json(report_path) if report_path.exists() else {}
    return {
        "ticker": ticker,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "returncode": result.returncode,
        "forecast_output_dir": str(ticker_dir),
        "forecast_report_path": str(report_path) if report_path.exists() else None,
        "command_path": str(ticker_dir / "forecast_command.json"),
        "stdout_path": str(ticker_dir / "forecast_stdout.txt"),
        "stderr_path": str(ticker_dir / "forecast_stderr.txt"),
        "report": report,
        "error": None if result.returncode == 0 else (result.stderr or result.stdout)[-3000:],
    }


def _run_pure_llm_forecast_for_candidate(
    *,
    ticker: str,
    ticker_dir: Path,
    candidate: dict[str, Any],
    config: VirtualTraderPipelineConfig,
    memory: VirtualTraderMemory,
    portfolio_notes: str,
) -> dict[str, Any]:
    ticker_memory = memory.context_for_ticker(ticker)
    position = ticker_memory.get("position") if isinstance(ticker_memory.get("position"), dict) else None
    holding_status = "owned" if position else "not_owned"
    command = [
        sys.executable,
        "-m",
        "market_forecasting_engine.pure_llm_stock_forecaster",
        "--ticker",
        ticker,
        "--company",
        str(candidate.get("company") or candidate.get("name") or ticker),
        "--provider",
        config.provider,
        "--start",
        config.start,
        "--interval",
        config.interval,
        "--output-dir",
        str(ticker_dir),
        "--llm-provider",
        config.llm_provider,
        "--trader-profile",
        config.trader_profile,
        "--holding-status",
        holding_status,
        "--portfolio-notes",
        portfolio_notes,
        "--llm-timeout",
        str(config.llm_timeout_seconds),
        "--reasoning-effort",
        config.llm_reasoning_effort,
        "--search-context-size",
        config.llm_search_context_size,
        "--ceo-llm-provider",
        config.ceo_llm_provider,
    ]
    if config.llm_model:
        command.extend(["--llm-model", config.llm_model])
    if config.ceo_llm_model:
        command.extend(["--ceo-llm-model", config.ceo_llm_model])
    if config.llm_env_file:
        command.extend(["--llm-env-file", str(config.llm_env_file)])
    if position:
        _append_position_args(command, position)
    started = datetime.now(UTC)
    result = subprocess.run(command, cwd=Path.cwd(), text=True, capture_output=True, check=False)
    (ticker_dir / "forecast_command.json").write_text(json.dumps({"command": command}, indent=2) + "\n", encoding="utf-8")
    (ticker_dir / "forecast_stdout.txt").write_text(result.stdout or "", encoding="utf-8")
    (ticker_dir / "forecast_stderr.txt").write_text(result.stderr or "", encoding="utf-8")
    pure_report_path = ticker_dir / f"{_pure_llm_safe_name(ticker)}_pure_llm_stock_forecast.json"
    pure_report = _read_json(pure_report_path) if pure_report_path.exists() else {}
    report = _pure_llm_report_for_order_planner(pure_report, ticker=ticker)
    report_path = ticker_dir / "forecast_report.json"
    if report:
        _write_json(report_path, report)
    return {
        "ticker": ticker,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "returncode": result.returncode,
        "forecast_output_dir": str(ticker_dir),
        "forecast_report_path": str(report_path) if report_path.exists() else None,
        "pure_llm_report_path": str(pure_report_path) if pure_report_path.exists() else None,
        "command_path": str(ticker_dir / "forecast_command.json"),
        "stdout_path": str(ticker_dir / "forecast_stdout.txt"),
        "stderr_path": str(ticker_dir / "forecast_stderr.txt"),
        "report": report,
        "error": None if result.returncode == 0 else (result.stderr or result.stdout)[-3000:],
    }


def _append_position_args(command: list[str], position: dict[str, Any]) -> None:
    values = {
        "--entry-price": position.get("avg_entry_price"),
        "--quantity": position.get("qty"),
        "--position-value": position.get("market_value"),
    }
    for flag, value in values.items():
        parsed = _to_float(value)
        if parsed is not None:
            command.extend([flag, str(parsed)])


def _pure_llm_report_for_order_planner(pure_report: dict[str, Any], *, ticker: str) -> dict[str, Any]:
    if not pure_report:
        return {}
    advice = pure_report.get("advice") if isinstance(pure_report.get("advice"), dict) else {}
    forecast = pure_report.get("forecast") if isinstance(pure_report.get("forecast"), dict) else {}
    final_advice = advice.get("final_advice") if isinstance(advice.get("final_advice"), dict) else {}
    return {
        "ticker": str(advice.get("ticker") or forecast.get("ticker") or ticker).upper(),
        "current_price": forecast.get("current_price"),
        "suggested_action": advice.get("decision") or "Hold",
        "llm_final_decision": advice,
        "final_advice": final_advice,
        "autonomous_llm_trader": {
            "status": "executed",
            "provider": pure_report.get("ceo_provider"),
            "model": pure_report.get("ceo_model"),
            "source_path": "pure_llm_stock_forecaster",
        },
        "decision_view": {
            "autonomous_execution_gate": {
                "execution_blocks": [],
                "warnings": [
                    "pure_llm_forecast_not_walk_forward_validated",
                    "standalone_pure_llm_script_is_advice_only_pipeline_provides_execution_gate",
                ],
            }
        },
        "pure_llm_stock_forecast": forecast,
        "pure_llm_artifact": pure_report,
    }


def _pure_llm_safe_name(value: str) -> str:
    return str(value).upper().replace("/", "_").replace(" ", "_")


def _load_broker_state(config: VirtualTraderPipelineConfig) -> dict[str, Any]:
    if config.disable_alpaca:
        return {"status": "disabled", "reason": "Alpaca broker disabled by config."}
    try:
        broker = AlpacaPaperBroker()
        return {
            "status": "ok",
            "account": broker.account(),
            "clock": broker.clock(),
            "positions": broker.positions(),
            "open_orders": broker.orders(status="open", limit=100),
            "recent_orders": broker.orders(status="all", limit=100, direction="desc"),
        }
    except Exception as exc:
        return {"status": "error", "error": _safe_error(exc), "positions": [], "open_orders": [], "recent_orders": []}


def _portfolio_selection(forecast_rows: list[dict[str, Any]], order_rows: list[dict[str, Any]], memory: VirtualTraderMemory) -> dict[str, Any]:
    candidates = []
    for forecast_row, order_row in zip(forecast_rows, order_rows, strict=False):
        report = forecast_row.get("report", {})
        decision = report.get("llm_final_decision") if isinstance(report.get("llm_final_decision"), dict) else {}
        confidence = _to_float(decision.get("confidence")) or 0.0
        action = str(order_row.get("action") or "").lower()
        score = confidence
        if action == "buy" and order_row.get("execution_allowed"):
            score += 0.25
        elif action == "hold" and order_row.get("final_advice", {}).get("buy_lower_price") is not None:
            score += 0.08
        if order_row.get("execution_blocks"):
            score -= 0.20
        candidates.append(
            {
                "ticker": order_row.get("ticker"),
                "action": action,
                "score": round(max(0.0, min(1.0, score)), 4),
                "execution_allowed": order_row.get("execution_allowed"),
                "intent": order_row.get("intent"),
                "reason": order_row.get("reason"),
            }
        )
    ranked = sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)
    return {
        "ranked": ranked,
        "act_now": [row for row in ranked if row.get("execution_allowed") and row.get("action") in {"buy", "sell"}],
        "watch": [row for row in ranked if row.get("intent") in {"buy_lower_limit_watch", "no_trade"} or row.get("action") == "hold"],
        "memory_considered": True,
        "active_watch_plans": memory.portfolio_context().get("active_watch_plans", {}),
    }


def _decision_memory_record(forecast_result: dict[str, Any], order_plan: dict[str, Any]) -> dict[str, Any]:
    report = forecast_result.get("report", {})
    decision = report.get("llm_final_decision") if isinstance(report.get("llm_final_decision"), dict) else {}
    return {
        "ticker": order_plan.get("ticker"),
        "action": order_plan.get("action"),
        "intent": order_plan.get("intent"),
        "reason": order_plan.get("reason"),
        "forecast_output_dir": forecast_result.get("forecast_output_dir"),
        "forecast_returncode": forecast_result.get("returncode"),
        "confidence": decision.get("confidence"),
        "final_advice": order_plan.get("final_advice"),
        "change_triggers": decision.get("change_triggers") if isinstance(decision.get("change_triggers"), list) else [],
        "execution_allowed": order_plan.get("execution_allowed"),
        "execution_blocks": order_plan.get("execution_blocks"),
        "order_payload": order_plan.get("order_payload"),
    }


def _order_memory_record(order_plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": order_plan.get("ticker"),
        "symbol": (order_plan.get("order_payload") or {}).get("symbol") or order_plan.get("ticker"),
        "action": order_plan.get("action"),
        "intent": order_plan.get("intent"),
        "execution_allowed": order_plan.get("execution_allowed"),
        "order_payload": order_plan.get("order_payload"),
        "order_result": order_plan.get("order_result"),
    }


def _compact_forecast_row(row: dict[str, Any]) -> dict[str, Any]:
    report = row.get("report", {})
    decision = report.get("llm_final_decision") if isinstance(report.get("llm_final_decision"), dict) else {}
    final_advice = report.get("final_advice") if isinstance(report.get("final_advice"), dict) else {}
    return {
        "ticker": row.get("ticker"),
        "returncode": row.get("returncode"),
        "forecast_output_dir": row.get("forecast_output_dir"),
        "forecast_report_path": row.get("forecast_report_path"),
        "suggested_action": report.get("suggested_action"),
        "current_price": report.get("current_price"),
        "ceo_status": report.get("autonomous_llm_trader", {}).get("status") if isinstance(report.get("autonomous_llm_trader"), dict) else None,
        "ceo_decision": decision.get("decision"),
        "ceo_confidence": decision.get("confidence"),
        "headline": final_advice.get("headline"),
        "buy_now_price": final_advice.get("buy_now_price"),
        "buy_lower_price": final_advice.get("buy_lower_price"),
        "buy_lower_zone_low": final_advice.get("buy_lower_zone_low"),
        "buy_lower_zone_high": final_advice.get("buy_lower_zone_high"),
        "buy_above_breakout_price": final_advice.get("buy_above_breakout_price"),
        "sell_or_trim_price": final_advice.get("sell_or_trim_price"),
        "stop_loss_price": final_advice.get("stop_loss_price"),
        "take_profit_price": final_advice.get("take_profit_price"),
        "error": row.get("error"),
    }


def _broker_state_for_board(broker_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": broker_state.get("status"),
        "error": broker_state.get("error"),
        "clock": broker_state.get("clock"),
        "account": broker_state.get("account"),
        "positions_count": len(broker_state.get("positions", []) or []),
        "open_orders_count": len(broker_state.get("open_orders", []) or []),
        "recent_orders_count": len(broker_state.get("recent_orders", []) or []),
    }


def _write_markdown_board(path: Path, board: dict[str, Any]) -> None:
    lines = [
        "# Virtual Trader Active Paper Board",
        "",
        f"Generated: {board.get('generated_at_utc')}",
        f"Execution mode: {board.get('policy', {}).get('default_execution')}",
        "",
        "## Decisions",
        "",
    ]
    for row in board.get("forecasts", []):
        lines.append(
            f"- {row.get('ticker')}: CEO={row.get('ceo_decision')} confidence={row.get('ceo_confidence')} "
            f"price={row.get('current_price')} headline={row.get('headline')}"
        )
    lines.extend(["", "## Order Plans", ""])
    for row in board.get("order_plans", []):
        lines.append(
            f"- {row.get('ticker')}: intent={row.get('intent')} allowed={row.get('execution_allowed')} "
            f"submitted={row.get('order_result', {}).get('submitted')} reason={row.get('reason')} "
            f"blocks={'; '.join(row.get('execution_blocks', []) or [])}"
        )
    lines.extend(["", "## Order Lifecycle", ""])
    lifecycle_events = board.get("order_lifecycle_events", []) if isinstance(board.get("order_lifecycle_events"), list) else []
    if not lifecycle_events:
        lines.append("- No broker order lifecycle changes detected in this run.")
    for event in lifecycle_events:
        lines.append(
            f"- {event.get('symbol')}: {event.get('previous_status')} -> {event.get('status')} "
            f"state={event.get('lifecycle_state')} reason={event.get('reason')} client_id={event.get('client_order_id')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _config_for_report(config: VirtualTraderPipelineConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def _trade_notional(equity: float, buying_power: float, config: VirtualTraderPipelineConfig) -> float:
    profile = risk_profile_for_name(config.risk_profile)
    risk_notional = max(0.0, equity * float(profile.risk_budget_pct) * 10.0)
    cap_notional = max(0.0, equity * float(config.max_position_pct_equity))
    return float(max(0.0, min(config.max_notional_per_trade, cap_notional, risk_notional, buying_power * 0.95)))


def _buy_now_limit_price(final_advice: dict[str, Any], current_price: float | None, offset_bps: float) -> float | None:
    reference = _to_float(final_advice.get("buy_now_price")) or current_price
    if reference is None:
        return None
    return reference * (1.0 + float(offset_bps) / 10_000.0)


def _buy_lower_limit_price(final_advice: dict[str, Any]) -> float | None:
    return (
        _to_float(final_advice.get("buy_lower_price"))
        or _to_float(final_advice.get("buy_lower_zone_high"))
        or _to_float(final_advice.get("buy_lower_zone_low"))
    )


def _sell_limit_price(final_advice: dict[str, Any], current_price: float | None, offset_bps: float) -> float | None:
    reference = _to_float(final_advice.get("sell_or_trim_price")) or current_price
    if reference is None:
        return None
    return reference * (1.0 - float(offset_bps) / 10_000.0)


def _watch_conditions(final_advice: dict[str, Any], current_price: float | None) -> dict[str, Any]:
    buy_lower_low = _to_float(final_advice.get("buy_lower_zone_low"))
    buy_lower_high = _to_float(final_advice.get("buy_lower_zone_high"))
    buy_lower_price = _to_float(final_advice.get("buy_lower_price"))
    breakout_price = _to_float(final_advice.get("buy_above_breakout_price"))
    stop_loss_price = _to_float(final_advice.get("stop_loss_price"))
    invalidation_price = _to_float(final_advice.get("invalidation_price"))
    take_profit_price = _to_float(final_advice.get("take_profit_price"))
    return {
        "current_price": current_price,
        "buy_lower": {
            "zone_low": buy_lower_low,
            "zone_high": buy_lower_high,
            "preferred_price": buy_lower_price,
            "status": _buy_lower_status(current_price, buy_lower_low, buy_lower_high, buy_lower_price),
        },
        "breakout_buy": {
            "trigger_price": breakout_price,
            "status": _price_trigger_status(
                current_price=current_price,
                trigger_price=breakout_price,
                direction="above",
            ),
        },
        "stop_loss": {
            "trigger_price": stop_loss_price,
            "status": _price_trigger_status(
                current_price=current_price,
                trigger_price=stop_loss_price,
                direction="below",
            ),
        },
        "invalidation": {
            "trigger_price": invalidation_price,
            "status": _price_trigger_status(
                current_price=current_price,
                trigger_price=invalidation_price,
                direction="below",
            ),
        },
        "take_profit": {
            "trigger_price": take_profit_price,
            "status": _price_trigger_status(
                current_price=current_price,
                trigger_price=take_profit_price,
                direction="above",
            ),
        },
    }


def _buy_lower_status(
    current_price: float | None,
    zone_low: float | None,
    zone_high: float | None,
    preferred_price: float | None,
) -> str:
    if current_price is None:
        return "unknown_current_price"
    high = zone_high or preferred_price
    low = zone_low or preferred_price or high
    if high is None:
        return "not_defined"
    if current_price <= high and (low is None or current_price >= low):
        return "inside_zone"
    if current_price < (low or high):
        return "trigger_crossed"
    return "waiting_for_pullback"


def _price_trigger_status(*, current_price: float | None, trigger_price: float | None, direction: str) -> str:
    if trigger_price is None:
        return "not_defined"
    if current_price is None:
        return "unknown_current_price"
    if direction == "above":
        return "trigger_crossed" if current_price >= trigger_price else "waiting_for_breakout"
    if direction == "below":
        return "trigger_crossed" if current_price <= trigger_price else "not_triggered"
    return "unknown_direction"


def _hold_watch_reason(watch_conditions: dict[str, Any]) -> str:
    breakout_status = watch_conditions.get("breakout_buy", {}).get("status")
    buy_lower_status = watch_conditions.get("buy_lower", {}).get("status")
    if breakout_status == "trigger_crossed":
        return "ceo_hold_after_breakout_crossed_needs_confirmation_or_retest"
    if buy_lower_status in {"trigger_crossed", "inside_zone"}:
        return "ceo_hold_after_buy_lower_zone_reached_needs_confirmation"
    return "ceo_hold_watch_plan_only"


def _buy_lower_order_reason(watch_conditions: dict[str, Any]) -> str:
    breakout_status = watch_conditions.get("breakout_buy", {}).get("status")
    if breakout_status == "trigger_crossed":
        return "hold_after_breakout_crossed_buy_lower_retest_limit_enabled"
    return "hold_with_buy_lower_limit_enabled"


def _position_for_symbol(positions: Any, ticker: str) -> dict[str, Any] | None:
    if not isinstance(positions, list):
        return None
    for position in positions:
        if isinstance(position, dict) and str(position.get("symbol") or "").upper() == ticker.upper():
            return position
    return None


def _position_qty(position: dict[str, Any] | None) -> float:
    if not position:
        return 0.0
    return max(0.0, _to_float(position.get("qty")) or 0.0)


def _client_order_id(prefix: str, ticker: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    safe = _safe_symbol(ticker)[:12]
    return f"{prefix}_{safe}_{stamp}"[:48]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return str(path)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(str(value).replace(",", ""))
    except Exception:
        return None
    return parsed if parsed == parsed else None


def _safe_symbol(ticker: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(ticker)).strip("_") or "unknown"


def _safe_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _load_env_file(path: str | Path | None) -> None:
    if path is None:
        return
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _progress(config: VirtualTraderPipelineConfig, message: str) -> None:
    if config.progress:
        print(f"[virtual-trader] {datetime.now().strftime('%H:%M:%S')} {message}", flush=True)
