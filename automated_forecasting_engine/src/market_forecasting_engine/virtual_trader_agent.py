from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import plistlib
import sys
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from market_forecasting_engine.alpaca_broker import AlpacaPaperBroker
from market_forecasting_engine.llm_usage import usage_log_path
from market_forecasting_engine.virtual_trader_memory import VirtualTraderMemory
from market_forecasting_engine.virtual_trader_pipeline import VirtualTraderPipelineConfig, run_virtual_trader_pipeline
from market_forecasting_engine.virtual_trader_planner import run_virtual_trader_planner
from market_forecasting_engine.virtual_trader_scout import ScoutConfig, run_virtual_trader_scout


LAUNCH_AGENT_LABEL = "com.marketforecasting.virtualtrader.agent"


@dataclass(frozen=True)
class VirtualTraderAgentConfig:
    project_dir: str | Path = "/Users/ruddigarcia/Projects/invest"
    output_root: str | Path = "automated_forecasting_engine/runs/virtual_trader_agent"
    memory_path: str | Path = "automated_forecasting_engine/runs/virtual_trader/memory.json"
    env_file: str | Path | None = "/Users/ruddigarcia/Projects/invest/.env"
    loop_interval_seconds: int = 14_400
    min_loop_interval_seconds: int = 900
    max_loop_interval_seconds: int = 21_600
    market_intelligence_min_refresh_seconds: int = 3_600
    market_intelligence_max_refresh_seconds: int = 21_600
    market_intelligence_search_context_size: str = "low"
    planner_dry_run: bool = False
    once: bool = False
    max_universe_tickers: int = 350
    scout_final_candidates: int = 8
    max_managed_candidates: int = 5
    analyst_pages: int = 12
    scout_start: str = "2025-01-01"
    forecast_start: str = "2020-01-01"
    provider: str = "yahoo"
    interval: str = "1d"
    horizons: str = "5,10,20"
    risk_profile: str = "medium"
    trader_profile: str = "medium"
    llm_model: str | None = None
    llm_reasoning_effort: str = "none"
    llm_timeout_seconds: int = 120
    llm_search_context_size: str = "medium"
    max_notional_per_trade: float = 100.0
    max_position_pct_equity: float = 0.025
    execute_paper_orders: bool = True
    allow_market_closed_orders: bool = False
    allow_repeated_symbol_orders: bool = False
    source_synthesis_dry_run: bool = False
    progress: bool = True


def run_virtual_trader_agent(config: VirtualTraderAgentConfig) -> None:
    project_dir = Path(config.project_dir).expanduser()
    os.chdir(project_dir)
    _load_env_file(config.env_file)
    cycle = 0
    _progress(config, f"agent started project_dir={project_dir} memory={config.memory_path}")
    while True:
        cycle += 1
        cycle_started = datetime.now(UTC)
        try:
            result = run_agent_cycle(config, cycle=cycle)
            _progress(
                config,
                f"cycle {cycle} complete mode={result.get('mode')} "
                f"selected={','.join(result.get('managed_tickers', [])) or 'none'} "
                f"orders={len(result.get('order_plans', []) or [])} "
                f"next_check={result.get('next_check_seconds')}s",
            )
        except Exception as exc:
            _progress(config, f"cycle {cycle} failed error={type(exc).__name__}: {exc}")
            _write_cycle_error(config, cycle, cycle_started, exc)
        if config.once:
            break
        sleep_seconds = int((result or {}).get("next_check_seconds") or config.loop_interval_seconds)
        _progress(config, f"sleeping {sleep_seconds} seconds")
        time.sleep(max(30, sleep_seconds))


def run_agent_cycle(config: VirtualTraderAgentConfig, *, cycle: int) -> dict[str, Any]:
    output_root = Path(config.output_root).expanduser()
    cycle_dir = output_root / datetime.now(UTC).strftime("cycle_%Y%m%d_%H%M%S")
    cycle_dir.mkdir(parents=True, exist_ok=True)
    memory = VirtualTraderMemory.load(config.memory_path)
    broker_state = _broker_state()
    if broker_state.get("status") == "ok":
        memory.broker_snapshot(
            account=broker_state.get("account"),
            positions=broker_state.get("positions", []),
            orders=broker_state.get("open_orders", []),
        )
        memory.save()
    portfolio_state = build_portfolio_state(memory=memory, broker_state=broker_state, config=config)
    intelligence_decision = market_intelligence_refresh_decision(memory=memory, portfolio_state=portfolio_state, config=config)
    if intelligence_decision["refresh"]:
        _progress(config, f"refreshing market intelligence reason={intelligence_decision.get('reason')}")
        intelligence = collect_market_intelligence(config=config, portfolio_state=portfolio_state)
        memory.record_market_intelligence(intelligence)
        memory.save()
    else:
        intelligence = memory.portfolio_context().get("market_intelligence", {})
        _progress(config, f"market intelligence reused reason={intelligence_decision.get('reason')}")
    mode = _agent_mode(memory, broker_state)
    _progress(
        config,
        f"cycle {cycle} mode={mode} broker={broker_state.get('status')} "
        f"positions={len(broker_state.get('positions', []) or [])} "
        f"open_orders={len(broker_state.get('open_orders', []) or [])} "
        f"portfolio_state={portfolio_state.get('state')} diversity={portfolio_state.get('diversity', {}).get('status')}",
    )
    _progress(config, "running portfolio planner")
    planner_result = run_virtual_trader_planner(
        portfolio_state=portfolio_state,
        market_intelligence=intelligence,
        memory_context=memory.portfolio_context(),
        config=_planner_config(config),
        llm_env_file=str(config.env_file) if config.env_file else None,
        llm_model=config.llm_model,
        llm_reasoning_effort=config.llm_reasoning_effort,
        timeout_seconds=float(config.llm_timeout_seconds),
        dry_run=config.planner_dry_run,
    )
    plan = planner_result.get("plan", {})
    memory.record_active_plan(
        {
            "planner_status": planner_result.get("status"),
            "planner_model": planner_result.get("model"),
            "plan": plan,
            "reason": planner_result.get("reason"),
        }
    )
    memory.save()
    _write_json(cycle_dir / "portfolio_plan.json", planner_result)
    _progress(
        config,
        f"planner status={planner_result.get('status')} mode={plan.get('cycle_mode')} "
        f"scan={plan.get('should_scan_new_candidates')} forecast_tickers={','.join(plan.get('forecast_tickers', []) or [])}",
    )

    scout_dir = cycle_dir / "scout"
    selected_path = scout_dir / "selected_candidates.json"
    scout_summary: dict[str, Any] = {"status": "skipped", "reason": "planner_did_not_request_scan"}
    if bool(plan.get("should_scan_new_candidates")) or _plan_has_task(plan, "scan_new_candidates"):
        _progress(config, "planner requested discovery/ranking scout")
        scout_summary = run_virtual_trader_scout(
            ScoutConfig(
                output_dir=scout_dir,
                start=config.scout_start,
                env_file=config.env_file,
                max_universe_tickers=config.max_universe_tickers,
                final_candidates=config.scout_final_candidates,
                analyst_pages=config.analyst_pages,
                progress=config.progress,
            )
        )
    else:
        scout_dir.mkdir(parents=True, exist_ok=True)
        _write_json(selected_path, [])
    managed_candidates = _managed_candidates(
        selected_path=selected_path,
        memory=memory,
        broker_state=broker_state,
        max_candidates=config.max_managed_candidates,
        forecast_tickers=plan.get("forecast_tickers", []),
    )
    max_from_plan = int(plan.get("max_candidates_to_forecast") or config.max_managed_candidates)
    managed_candidates = managed_candidates[: max(1, min(config.max_managed_candidates, max_from_plan))]
    managed_path = cycle_dir / "managed_candidates.json"
    _write_json(managed_path, managed_candidates)
    _progress(config, f"managed candidate set={','.join(row.get('ticker', '') for row in managed_candidates)}")

    if managed_candidates and _plan_should_forecast(plan):
        _progress(config, "planner requested active paper trader pipeline")
        board = run_virtual_trader_pipeline(
            VirtualTraderPipelineConfig(
                output_dir=cycle_dir / "active_pipeline",
                selected_candidates_path=managed_path,
                memory_path=config.memory_path,
                max_candidates=len(managed_candidates),
                provider=config.provider,
                start=config.forecast_start,
                interval=config.interval,
                horizons=config.horizons,
                risk_profile=config.risk_profile,
                trader_profile=config.trader_profile,
                llm_env_file=config.env_file,
                llm_model=config.llm_model,
                llm_reasoning_effort=config.llm_reasoning_effort,
                llm_timeout_seconds=config.llm_timeout_seconds,
                llm_search_context_size=config.llm_search_context_size,
                max_notional_per_trade=config.max_notional_per_trade,
                max_position_pct_equity=config.max_position_pct_equity,
                execute_paper_orders=config.execute_paper_orders,
                allow_market_closed_orders=config.allow_market_closed_orders,
                allow_repeated_symbol_orders=config.allow_repeated_symbol_orders,
                source_synthesis_dry_run=config.source_synthesis_dry_run,
                progress=config.progress,
            )
        )
    else:
        _progress(config, "planner did not request forecasts this cycle")
        board = {
            "artifact_paths": {},
            "order_plans": [],
            "portfolio_selection": {},
            "policy": {"planner_skipped_forecasts": True},
        }
    memory = VirtualTraderMemory.load(config.memory_path)
    cycle_record = {
        "cycle": cycle,
        "mode": mode,
        "cycle_dir": str(cycle_dir),
        "scout_dir": str(scout_dir),
        "llm_usage_audit": {
            "planner_and_agent_usage_log": str(usage_log_path(process_name="virtual_trader_agent")),
            "forecast_ceo_usage_logs": "Forecast subprocesses use the same OpenAI usage logger with their own process/purpose context.",
        },
        "portfolio_state": portfolio_state,
        "market_intelligence": {
            "status": intelligence.get("status"),
            "fetched_at_utc": intelligence.get("fetched_at_utc"),
            "summary": intelligence.get("summary"),
            "refresh_decision": intelligence_decision,
        },
        "portfolio_plan": {
            "status": planner_result.get("status"),
            "model": planner_result.get("model"),
            "plan": plan,
            "reason": planner_result.get("reason"),
        },
        "next_check_seconds": plan.get("next_wakeup_seconds") or portfolio_state.get("next_check_seconds"),
        "managed_tickers": [row.get("ticker") for row in managed_candidates],
        "pipeline_board": board.get("artifact_paths", {}).get("virtual_trader_board"),
        "order_summary": [
            {
                "ticker": row.get("ticker"),
                "intent": row.get("intent"),
                "execution_allowed": row.get("execution_allowed"),
                "submitted": row.get("order_result", {}).get("submitted") if isinstance(row.get("order_result"), dict) else None,
                "reason": row.get("reason"),
                "blocks": row.get("execution_blocks", []),
            }
            for row in board.get("order_plans", [])
        ],
    }
    memory.record_cycle(cycle_record)
    memory.save()
    result = {
        **cycle_record,
        "scout_summary": scout_summary,
        "managed_candidates_path": str(managed_path),
        "order_plans": board.get("order_plans", []),
    }
    _write_json(cycle_dir / "agent_cycle.json", result)
    _write_latest_pointer(output_root, cycle_dir)
    return result


def market_intelligence_refresh_decision(
    *,
    memory: VirtualTraderMemory,
    portfolio_state: dict[str, Any],
    config: VirtualTraderAgentConfig,
) -> dict[str, Any]:
    intelligence = memory.portfolio_context().get("market_intelligence", {})
    fetched_at = _parse_time(intelligence.get("fetched_at_utc")) if isinstance(intelligence, dict) else None
    age_seconds = (datetime.now(UTC) - fetched_at).total_seconds() if fetched_at else None
    market_open = bool(portfolio_state.get("market_clock", {}).get("is_open")) if isinstance(portfolio_state.get("market_clock"), dict) else False
    near_trigger = any(row.get("near_trigger") for row in portfolio_state.get("trigger_proximity", []) or [])
    min_refresh = int(config.market_intelligence_min_refresh_seconds)
    max_refresh = int(config.market_intelligence_max_refresh_seconds)
    if fetched_at is None:
        return {"refresh": True, "reason": "no_prior_market_intelligence", "age_seconds": None}
    if near_trigger and age_seconds >= min_refresh:
        return {"refresh": True, "reason": "near_trade_or_risk_trigger", "age_seconds": round(age_seconds, 2)}
    if market_open and age_seconds >= min_refresh:
        return {"refresh": True, "reason": "market_open_refresh_due", "age_seconds": round(age_seconds, 2)}
    if age_seconds >= max_refresh:
        return {"refresh": True, "reason": "max_refresh_age_exceeded", "age_seconds": round(age_seconds, 2)}
    return {"refresh": False, "reason": "fresh_enough", "age_seconds": round(age_seconds, 2)}


def collect_market_intelligence(
    *,
    config: VirtualTraderAgentConfig,
    portfolio_state: dict[str, Any],
) -> dict[str, Any]:
    tickers = [row.get("symbol") for row in portfolio_state.get("positions", []) if row.get("symbol")]
    topics = {
        "market_news": f"stock market news today {' '.join(tickers[:8])}",
        "economic_news": "US Europe economic news today inflation rates jobs central bank market impact",
        "political_regulatory_news": "political regulatory geopolitical news today stock market impact",
        "earnings_events": f"earnings corporate events analyst upgrades downgrades today {' '.join(tickers[:8])}",
    }
    items = []
    for topic, query in topics.items():
        items.extend(_fetch_yahoo_rss_items(topic=topic, query=query, limit=6))
    summary = _market_intelligence_summary(items)
    return {
        "status": "ok" if items else "empty",
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "source": "yahoo_news_rss_queries",
        "topics": list(topics),
        "portfolio_tickers": tickers,
        "summary": summary,
        "items": items[:40],
        "decision_policy": {
            "feeds_virtual_trader_context": True,
            "overrides_ceo_decision": False,
            "role": "Global market, macro, political/regulatory, and event context for scan cadence, caution, and CEO portfolio notes.",
        },
    }


def _fetch_yahoo_rss_items(*, topic: str, query: str, limit: int) -> list[dict[str, Any]]:
    try:
        url = "https://news.search.yahoo.com/rss?" + urlencode({"p": query})
        request = Request(url, headers={"User-Agent": "Mozilla/5.0 market-forecasting-engine/virtual-trader-agent"})
        with urlopen(request, timeout=20) as response:
            text = response.read().decode("utf-8", errors="ignore")
        import xml.etree.ElementTree as ET

        root = ET.fromstring(text)
        rows = []
        for item in root.findall(".//item")[:limit]:
            rows.append(
                {
                    "topic": topic,
                    "title": (item.findtext("title") or "").strip(),
                    "link": (item.findtext("link") or "").strip(),
                    "published": (item.findtext("pubDate") or "").strip(),
                    "source": "yahoo_news_rss",
                }
            )
        return [row for row in rows if row["title"]]
    except Exception as exc:
        return [{"topic": topic, "source": "yahoo_news_rss", "status": "error", "error": f"{type(exc).__name__}: {exc}"}]


def _market_intelligence_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in items if item.get("title")]
    by_topic: dict[str, int] = {}
    for item in valid:
        topic = str(item.get("topic") or "unknown")
        by_topic[topic] = by_topic.get(topic, 0) + 1
    titles = [item["title"] for item in valid[:12]]
    return {
        "item_count": len(valid),
        "topic_counts": by_topic,
        "top_headlines": titles,
        "risk_keywords_detected": sorted(
            {
                keyword
                for title in titles
                for keyword in ("inflation", "rates", "tariff", "war", "fed", "ecb", "recession", "earnings", "guidance")
                if keyword in title.lower()
            }
        ),
    }


def build_portfolio_state(
    *,
    memory: VirtualTraderMemory,
    broker_state: dict[str, Any],
    config: VirtualTraderAgentConfig,
) -> dict[str, Any]:
    portfolio = memory.portfolio_context()
    account = broker_state.get("account") if isinstance(broker_state.get("account"), dict) else portfolio.get("account", {})
    positions = broker_state.get("positions") if isinstance(broker_state.get("positions"), list) else portfolio.get("positions", [])
    open_orders = broker_state.get("open_orders") if isinstance(broker_state.get("open_orders"), list) else portfolio.get("open_orders", [])
    clock = broker_state.get("clock") if isinstance(broker_state.get("clock"), dict) else {}
    equity = _to_float(account.get("equity") or account.get("portfolio_value")) or 0.0
    cash = _to_float(account.get("cash")) or 0.0
    buying_power = _to_float(account.get("buying_power")) or 0.0
    position_rows = [_position_state(row, equity=equity) for row in positions if isinstance(row, dict)]
    total_market_value = sum(abs(float(row.get("market_value") or 0.0)) for row in position_rows)
    concentration = max((float(row.get("portfolio_weight") or 0.0) for row in position_rows), default=0.0)
    asset_classes: dict[str, int] = {}
    for row in position_rows:
        asset_class = str(row.get("asset_class") or "unknown")
        asset_classes[asset_class] = asset_classes.get(asset_class, 0) + 1
    watch_plans = portfolio.get("active_watch_plans", {}) if isinstance(portfolio.get("active_watch_plans"), dict) else {}
    trigger_proximity = _watch_trigger_proximity(position_rows, watch_plans)
    diversity = _diversity_state(position_rows, concentration)
    state = "bootstrap_empty_portfolio"
    if position_rows:
        state = "manage_invested_portfolio"
    elif open_orders or watch_plans:
        state = "manage_pending_portfolio"
    next_check_seconds, cadence_reason = _adaptive_next_check_seconds(
        config=config,
        market_is_open=bool(clock.get("is_open")) if clock else False,
        position_rows=position_rows,
        open_orders=open_orders,
        trigger_proximity=trigger_proximity,
        diversity=diversity,
    )
    return {
        "state": state,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "market_clock": clock,
        "account": {
            "equity": equity,
            "cash": cash,
            "buying_power": buying_power,
            "cash_pct_equity": round(cash / equity, 4) if equity else None,
            "buying_power_pct_equity": round(buying_power / equity, 4) if equity else None,
        },
        "positions": position_rows,
        "open_orders_count": len(open_orders or []),
        "active_watch_plan_count": len(watch_plans),
        "total_market_value": round(total_market_value, 2),
        "gross_exposure_pct_equity": round(total_market_value / equity, 4) if equity else None,
        "largest_position_pct_equity": round(concentration, 4),
        "asset_class_counts": asset_classes,
        "diversity": diversity,
        "trigger_proximity": trigger_proximity,
        "next_check_seconds": next_check_seconds,
        "cadence_reason": cadence_reason,
    }


def _position_state(position: dict[str, Any], *, equity: float) -> dict[str, Any]:
    market_value = _to_float(position.get("market_value")) or 0.0
    current_price = _to_float(position.get("current_price"))
    avg_entry = _to_float(position.get("avg_entry_price"))
    unrealized_plpc = _to_float(position.get("unrealized_plpc"))
    return {
        "symbol": str(position.get("symbol") or "").upper(),
        "asset_class": position.get("asset_class"),
        "qty": _to_float(position.get("qty")),
        "avg_entry_price": avg_entry,
        "current_price": current_price,
        "market_value": round(market_value, 2),
        "cost_basis": _to_float(position.get("cost_basis")),
        "unrealized_pl": _to_float(position.get("unrealized_pl")),
        "unrealized_plpc": unrealized_plpc,
        "portfolio_weight": round(abs(market_value) / equity, 4) if equity else None,
        "above_entry": current_price is not None and avg_entry is not None and current_price > avg_entry,
    }


def _diversity_state(position_rows: list[dict[str, Any]], concentration: float) -> dict[str, Any]:
    symbols = [row.get("symbol") for row in position_rows if row.get("symbol")]
    asset_classes = sorted({str(row.get("asset_class") or "unknown") for row in position_rows})
    if not symbols:
        status = "empty"
        warnings = ["no_positions_yet"]
    else:
        warnings = []
        if len(symbols) < 3:
            warnings.append("too_few_positions_for_diversification")
        if concentration > 0.35:
            warnings.append("single_position_concentration_above_35pct")
        if len(asset_classes) <= 1 and len(symbols) > 1:
            warnings.append("single_asset_class_concentration")
        status = "needs_diversification" if warnings else "acceptable"
    return {
        "status": status,
        "position_count": len(symbols),
        "asset_classes": asset_classes,
        "warnings": warnings,
        "policy": "Diversity affects scan priority, sizing caution, and whether the agent should prefer new sectors over adding to existing concentration.",
    }


def _watch_trigger_proximity(position_rows: list[dict[str, Any]], watch_plans: dict[str, Any]) -> list[dict[str, Any]]:
    by_symbol = {str(row.get("symbol") or "").upper(): row for row in position_rows}
    proximity = []
    for ticker, plan in watch_plans.items():
        symbol = str(ticker or "").upper()
        current = _to_float((by_symbol.get(symbol) or {}).get("current_price"))
        if current is None:
            continue
        for key in ("buy_lower_price", "buy_above_breakout_price", "sell_or_trim_price", "stop_loss_price"):
            level = _to_float(plan.get(key)) if isinstance(plan, dict) else None
            if level is None:
                continue
            distance = abs(current / level - 1.0) if level else None
            proximity.append(
                {
                    "ticker": symbol,
                    "level_type": key,
                    "current_price": current,
                    "level": level,
                    "distance_pct": round(distance, 4) if distance is not None else None,
                    "near_trigger": distance is not None and distance <= 0.025,
                }
            )
    return sorted(proximity, key=lambda row: row.get("distance_pct") if row.get("distance_pct") is not None else 999)[:20]


def _adaptive_next_check_seconds(
    *,
    config: VirtualTraderAgentConfig,
    market_is_open: bool,
    position_rows: list[dict[str, Any]],
    open_orders: list[dict[str, Any]],
    trigger_proximity: list[dict[str, Any]],
    diversity: dict[str, Any],
) -> tuple[int, str]:
    base = int(config.loop_interval_seconds)
    if open_orders:
        target, reason = int(config.min_loop_interval_seconds), "open_orders_need_monitoring"
    elif any(row.get("near_trigger") for row in trigger_proximity):
        target, reason = int(config.min_loop_interval_seconds), "price_near_watch_or_risk_trigger"
    elif market_is_open and position_rows:
        target, reason = min(base, 3600), "market_open_with_positions"
    elif market_is_open and diversity.get("status") in {"empty", "needs_diversification"}:
        target, reason = min(base, 7200), "market_open_portfolio_building"
    elif not market_is_open and position_rows:
        target, reason = min(base, 14_400), "market_closed_with_positions"
    else:
        target, reason = base, "default_cadence"
    bounded = max(int(config.min_loop_interval_seconds), min(int(config.max_loop_interval_seconds), int(target)))
    return bounded, reason


def install_launch_agent(config: VirtualTraderAgentConfig, *, python_executable: str | None = None) -> Path:
    project_dir = Path(config.project_dir).expanduser()
    launch_dir = Path.home() / "Library" / "LaunchAgents"
    log_dir = Path(config.output_root).expanduser() / "launchd_logs"
    launch_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    plist_path = launch_dir / f"{LAUNCH_AGENT_LABEL}.plist"
    python_path = python_executable or sys.executable
    program_arguments = [
        python_path,
        "-m",
        "market_forecasting_engine.virtual_trader_agent_cli",
        "--project-dir",
        str(project_dir),
        "--output-root",
        str(config.output_root),
        "--memory-path",
        str(config.memory_path),
        "--loop-interval-seconds",
        str(config.loop_interval_seconds),
        "--min-loop-interval-seconds",
        str(config.min_loop_interval_seconds),
        "--max-loop-interval-seconds",
        str(config.max_loop_interval_seconds),
        "--market-intelligence-min-refresh-seconds",
        str(config.market_intelligence_min_refresh_seconds),
        "--market-intelligence-max-refresh-seconds",
        str(config.market_intelligence_max_refresh_seconds),
        "--market-intelligence-search-context-size",
        config.market_intelligence_search_context_size,
        "--max-universe-tickers",
        str(config.max_universe_tickers),
        "--scout-final-candidates",
        str(config.scout_final_candidates),
        "--max-managed-candidates",
        str(config.max_managed_candidates),
        "--analyst-pages",
        str(config.analyst_pages),
        "--horizons",
        config.horizons,
        "--risk-profile",
        config.risk_profile,
        "--trader-profile",
        config.trader_profile,
    ]
    if config.env_file:
        program_arguments.extend(["--env-file", str(config.env_file)])
    if not config.execute_paper_orders:
        program_arguments.append("--dry-run")
    if config.planner_dry_run:
        program_arguments.append("--planner-dry-run")
    if config.allow_market_closed_orders:
        program_arguments.append("--allow-market-closed-orders")
    if config.allow_repeated_symbol_orders:
        program_arguments.append("--allow-repeated-symbol-orders")
    plist = {
        "Label": LAUNCH_AGENT_LABEL,
        "ProgramArguments": program_arguments,
        "WorkingDirectory": str(project_dir),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(log_dir / "virtual_trader_agent.out.log"),
        "StandardErrorPath": str(log_dir / "virtual_trader_agent.err.log"),
        "EnvironmentVariables": {
            "PYTHONPATH": str(project_dir / "automated_forecasting_engine" / "src"),
        },
    }
    with plist_path.open("wb") as handle:
        plistlib.dump(plist, handle)
    return plist_path


def _agent_mode(memory: VirtualTraderMemory, broker_state: dict[str, Any]) -> str:
    positions = broker_state.get("positions", []) if isinstance(broker_state.get("positions"), list) else []
    open_orders = broker_state.get("open_orders", []) if isinstance(broker_state.get("open_orders"), list) else []
    if not memory.has_trading_history() and not positions and not open_orders:
        return "bootstrap_portfolio_from_scratch"
    return "manage_existing_virtual_portfolio"


def _managed_candidates(
    *,
    selected_path: Path,
    memory: VirtualTraderMemory,
    broker_state: dict[str, Any],
    max_candidates: int,
    forecast_tickers: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    selected = json.loads(selected_path.read_text(encoding="utf-8")) if selected_path.exists() else []
    rows = selected if isinstance(selected, list) else []
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    forecast_set = [str(ticker).upper() for ticker in (forecast_tickers or []) if str(ticker).strip()]
    broker_positions = broker_state.get("positions", []) if isinstance(broker_state.get("positions"), list) else []
    portfolio = memory.portfolio_context()
    memory_positions = portfolio.get("positions", []) if isinstance(portfolio.get("positions"), list) else []
    watch_plans = portfolio.get("active_watch_plans", {}) if isinstance(portfolio.get("active_watch_plans"), dict) else {}
    for ticker in forecast_set:
        if ticker in seen:
            continue
        matched = next((row for row in rows if isinstance(row, dict) and str(row.get("ticker") or "").upper() == ticker), None)
        if matched is not None:
            output.append(matched)
        else:
            metadata = (
                next((row for row in broker_positions if str(row.get("symbol") or "").upper() == ticker), None)
                or next((row for row in memory_positions if str(row.get("symbol") or "").upper() == ticker), None)
                or watch_plans.get(ticker)
                or {}
            )
            output.append(_memory_candidate(ticker, reason="planner_requested_forecast", metadata=metadata))
        seen.add(ticker)
        if len(output) >= max_candidates:
            return output
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").upper()
        if ticker and ticker not in seen:
            output.append(row)
            seen.add(ticker)
        if len(output) >= max_candidates:
            break
    for position in broker_positions or memory_positions:
        ticker = str(position.get("symbol") or "").upper()
        if ticker and ticker not in seen:
            output.append(_memory_candidate(ticker, reason="existing_alpaca_paper_position", metadata=position))
            seen.add(ticker)
    for ticker, plan in (watch_plans or {}).items():
        normalized = str(ticker or "").upper()
        if normalized and normalized not in seen:
            output.append(_memory_candidate(normalized, reason="active_virtual_trader_watch_plan", metadata=plan))
            seen.add(normalized)
    return output[: max_candidates]


def _memory_candidate(ticker: str, *, reason: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "rank": 999,
        "ranking_score": 0.5,
        "score": 0.5,
        "selected": True,
        "eligible": True,
        "reasons": [reason],
        "discovery_sources": ["virtual_trader_memory"],
        "source_counts": {"virtual_trader_memory": 1},
        "latest_records": [{"ticker": ticker, "source": "virtual_trader_memory", "reason": reason, "metadata": metadata}],
    }


def _broker_state() -> dict[str, Any]:
    try:
        broker = AlpacaPaperBroker()
        return {
            "status": "ok",
            "account": broker.account(),
            "clock": broker.clock(),
            "positions": broker.positions(),
            "open_orders": broker.orders(status="open", limit=100),
        }
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "positions": [], "open_orders": []}


def _write_cycle_error(config: VirtualTraderAgentConfig, cycle: int, started_at: datetime, exc: Exception) -> None:
    output_root = Path(config.output_root).expanduser()
    error_dir = output_root / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        error_dir / f"cycle_{cycle}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json",
        {
            "cycle": cycle,
            "started_at_utc": started_at.isoformat(),
            "failed_at_utc": datetime.now(UTC).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        },
    )


def _write_latest_pointer(output_root: Path, cycle_dir: Path) -> None:
    _write_json(
        output_root / "latest_cycle.json",
        {
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "cycle_dir": str(cycle_dir),
            "agent_cycle": str(cycle_dir / "agent_cycle.json"),
            "virtual_trader_board": str(cycle_dir / "active_pipeline" / "virtual_trader_board.json"),
        },
    )


def _planner_config(config: VirtualTraderAgentConfig) -> dict[str, Any]:
    return {
        "loop_interval_seconds": config.loop_interval_seconds,
        "min_loop_interval_seconds": config.min_loop_interval_seconds,
        "max_loop_interval_seconds": config.max_loop_interval_seconds,
        "market_intelligence_min_refresh_seconds": config.market_intelligence_min_refresh_seconds,
        "market_intelligence_max_refresh_seconds": config.market_intelligence_max_refresh_seconds,
        "market_intelligence_search_context_size": config.market_intelligence_search_context_size,
        "max_universe_tickers": config.max_universe_tickers,
        "scout_final_candidates": config.scout_final_candidates,
        "max_managed_candidates": config.max_managed_candidates,
        "horizons": config.horizons,
        "risk_profile": config.risk_profile,
        "trader_profile": config.trader_profile,
        "max_notional_per_trade": config.max_notional_per_trade,
        "max_position_pct_equity": config.max_position_pct_equity,
        "execute_paper_orders": config.execute_paper_orders,
        "allow_market_closed_orders": config.allow_market_closed_orders,
        "allow_repeated_symbol_orders": config.allow_repeated_symbol_orders,
    }


def _plan_has_task(plan: dict[str, Any], task_type: str) -> bool:
    tasks = plan.get("tasks", []) if isinstance(plan.get("tasks"), list) else []
    return any(isinstance(task, dict) and task.get("task_type") == task_type for task in tasks)


def _plan_should_forecast(plan: dict[str, Any]) -> bool:
    if bool(plan.get("should_scan_new_candidates")) or _plan_has_task(plan, "scan_new_candidates"):
        return True
    if plan.get("forecast_tickers"):
        return True
    tasks = plan.get("tasks", []) if isinstance(plan.get("tasks"), list) else []
    return any(isinstance(task, dict) and task.get("task_type") == "refresh_forecast" for task in tasks)


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(str(value).replace(",", ""))
    except Exception:
        return None
    return parsed if parsed == parsed else None


def _load_env_file(path: str | Path | None) -> None:
    if not path:
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


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return str(path)


def _progress(config: VirtualTraderAgentConfig, message: str) -> None:
    if config.progress:
        print(f"[virtual-agent] {datetime.now().strftime('%H:%M:%S')} {message}", flush=True)
