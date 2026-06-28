from __future__ import annotations

import argparse
import json
import math
import os
import selectors
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, time as local_time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from market_forecasting_engine.deribit_broker import DeribitLiveSpotBroker
from market_forecasting_engine.live_trading.deribit_spot_agent import (
    account_subset,
    agent_round_amount,
    agent_round_price,
    analyze_existing_protection,
    book_quote,
    dedupe_orders,
    round_down_to_step,
    round_to_tick,
    spread_fraction,
    strict_json,
)


DEFAULT_OUTPUT_DIR = "automated_forecasting_engine/runs/live_deribit_eth_usdc_daily_agent"


@dataclass(frozen=True)
class ForecastDecision:
    report_path: Path
    report: dict[str, Any]
    ceo_decision: dict[str, Any]
    final_advice: dict[str, Any]
    created_at_utc: str | None


def main() -> None:
    args = build_parser().parse_args()
    if args.execute_live_orders and not args.confirm_live_deribit_eth_usdc_orders:
        raise SystemExit("Live execution requires --confirm-live-deribit-eth-usdc-orders.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    broker = DeribitLiveSpotBroker()
    log_progress(
        "starting ETH/USDC daily agent "
        f"instrument={args.instrument.upper()} live_execution={bool(args.execute_live_orders)} "
        f"check_interval_seconds={int(args.check_interval_seconds)}"
    )
    while True:
        log_progress("cycle started")
        record = run_cycle(args=args, broker=broker)
        report_path = write_agent_report(record, output_dir)
        append_agent_log(record, output_dir)
        print(json.dumps(_cycle_summary(record, report_path), indent=2, default=str), flush=True)
        log_progress(
            "cycle complete "
            f"latest_price={(record.get('market') or {}).get('latest_price')} "
            f"action={(record.get('decision') or {}).get('action')} "
            f"reason={(record.get('decision') or {}).get('reason')} "
            f"orders={len(record.get('order_results') or [])}"
        )
        if args.once:
            break
        sleep_seconds = max(60, int(args.check_interval_seconds))
        log_progress(f"sleeping {sleep_seconds}s until next live-price check")
        time.sleep(sleep_seconds)


def log_progress(message: str) -> None:
    print(f"[eth-agent] {datetime.now().strftime('%H:%M:%S')} {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dedicated autonomous ETH/USDC Deribit live agent. It runs one full forecast/CEO report "
            "per local day, then monitors live Deribit prices and account state against that cached decision."
        )
    )
    parser.add_argument("--instrument", default="ETH_USDC")
    parser.add_argument("--ticker", default="ETH-USDC")
    parser.add_argument("--base-currency", default="ETH")
    parser.add_argument("--quote-currency", default="USDC")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--llm-env-file", default=None)
    parser.add_argument("--active-timezone", default="Europe/Amsterdam")
    parser.add_argument("--daily-forecast-local-time", default="07:00")
    parser.add_argument("--check-interval-seconds", type=int, default=3600)
    parser.add_argument("--forecast-lookback-days", type=int, default=90)
    parser.add_argument("--forecast-provider", default="deribit")
    parser.add_argument("--forecast-interval", default="1h")
    parser.add_argument("--forecast-horizons", default="1,2,3,6,8,12")
    parser.add_argument("--forecast-validation-window", type=int, default=48)
    parser.add_argument("--forecast-step-size", type=int, default=12)
    parser.add_argument("--forecast-max-splits", type=int, default=5)
    parser.add_argument("--forecast-min-training-rows", type=int, default=240)
    parser.add_argument("--forecast-validation-workers", type=int, default=0)
    parser.add_argument("--forecast-llm-timeout", type=int, default=240)
    parser.add_argument("--forecast-timeout-seconds", type=int, default=3600)
    parser.add_argument("--forecast-retry-after-seconds", type=int, default=14400)
    parser.add_argument("--force-forecast-refresh", action="store_true")
    parser.add_argument("--max-notional-usdc", type=float, default=100.0)
    parser.add_argument("--max-base-position", type=float, default=0.25)
    parser.add_argument("--min-order-base-amount", type=float, default=0.0001)
    parser.add_argument("--max-spread-pct", type=float, default=0.003)
    parser.add_argument("--entry-price-tolerance-pct", type=float, default=0.0015)
    parser.add_argument("--stop-protection-coverage-ratio", type=float, default=0.95)
    parser.add_argument("--inventory-scope", choices=("codex_only", "allow_manual"), default="codex_only")
    parser.add_argument("--managed-base-balance", type=float, default=None)
    parser.add_argument("--replace-protection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--execute-live-orders", action="store_true")
    parser.add_argument("--confirm-live-deribit-eth-usdc-orders", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser


def run_cycle(*, args: argparse.Namespace, broker: DeribitLiveSpotBroker) -> dict[str, Any]:
    now = datetime.now(UTC)
    output_dir = Path(args.output_dir)
    log_progress("loading state and daily forecast/CEO decision")
    state = read_state(output_dir, args.instrument)
    forecast_record = ensure_daily_forecast(args=args, state=state, now=now)
    state.update(
        {
            "last_forecast_report_path": str(forecast_record.report_path),
            "last_forecast_created_at_utc": forecast_record.created_at_utc,
        }
    )
    if "last_forecast_refresh_error" not in state:
        state["last_forecast_date"] = _local_date(now, args.active_timezone)
    log_progress("reading live Deribit market, balances, and open orders")
    market_packet = read_live_market(args=args, broker=broker)
    log_progress("reconciling open sell-order exposure")
    sell_order_reconciliation = reconcile_open_sell_orders(
        args=args,
        broker=broker,
        state=state,
        market_packet=market_packet,
    )
    if sell_order_reconciliation.get("cancelled_count"):
        log_progress(
            "sell-order reconciliation cancelled "
            f"{sell_order_reconciliation.get('cancelled_count')} excess agent-managed sell order(s); refreshing market"
        )
        market_packet = read_live_market(args=args, broker=broker)
    elif sell_order_reconciliation.get("status") == "manual_review_required":
        log_progress("sell-order reconciliation found unmanaged over-exposure; no manual orders were cancelled")
    log_progress("building live action from cached CEO decision")
    plan = decide_from_cached_ceo(args=args, state=state, forecast_record=forecast_record, market_packet=market_packet)
    order_results: list[dict[str, Any]] = []
    if args.execute_live_orders and plan.get("execution_allowed"):
        label_base = f"codex-eth-usdc-daily-{now.strftime('%Y%m%d%H%M%S')}"
        log_progress(f"submitting live plan action={plan.get('action')} reason={plan.get('reason')}")
        order_results = execute_live_plan(broker=broker, args=args, plan=plan, label_base=label_base)
        update_state_from_order_results(state=state, instrument=args.instrument.upper(), results=order_results)
    else:
        plan["order_submission"] = {
            "submitted": False,
            "reason": "dry_run" if not args.execute_live_orders else "execution_not_allowed_by_plan",
        }
    state["last_cycle_at_utc"] = now.isoformat()
    write_state(output_dir, args.instrument, state)
    return {
        "checked_at_utc": now.isoformat(),
        "mode": "deribit_eth_usdc_daily_forecast_agent",
        "venue": "deribit_live",
        "instrument": args.instrument.upper(),
        "ticker": args.ticker.upper(),
        "safety": {
            "execute_live_orders": bool(args.execute_live_orders),
            "confirmation_required": True,
            "confirmation_provided": bool(args.confirm_live_deribit_eth_usdc_orders),
            "market_orders_allowed": False,
            "policy": "Dry-run by default. Live spot orders require both explicit execution flags. Only limit/stop-limit style protection is created by this agent.",
        },
        "forecast": {
            "report_path": str(forecast_record.report_path),
            "created_at_utc": forecast_record.created_at_utc,
            "ceo_decision": forecast_record.ceo_decision,
            "final_advice": forecast_record.final_advice,
            "top_level_action": forecast_record.report.get("suggested_action"),
            "risk_level": forecast_record.report.get("risk_level"),
            "forecasts": forecast_record.report.get("forecasts", []),
        },
        "market": market_packet,
        "sell_order_reconciliation": sell_order_reconciliation,
        "decision": plan,
        "order_results": order_results,
        "agent_state": state,
    }


def ensure_daily_forecast(*, args: argparse.Namespace, state: dict[str, Any], now: datetime) -> ForecastDecision:
    output_dir = Path(args.output_dir)
    forecast_due = is_forecast_due(args=args, state=state, now=now)
    report_path = Path(str(state.get("last_forecast_report_path") or ""))
    if forecast_due or not report_path.exists() or bool(args.force_forecast_refresh):
        log_progress("daily forecast refresh is due; launching full forecast pipeline")
        try:
            report_path = run_morning_forecast(args=args, now=now)
            state.pop("last_forecast_refresh_error", None)
            state.pop("last_forecast_refresh_failed_at_utc", None)
            log_progress(f"daily forecast refresh completed report={report_path}")
        except Exception as exc:
            fallback_path = report_path if report_path.exists() else _latest_valid_forecast_report(output_dir)
            state["last_forecast_refresh_failed_at_utc"] = datetime.now(UTC).isoformat()
            state["last_forecast_refresh_error"] = {
                "error": str(exc),
                "fallback_report_path": str(fallback_path) if fallback_path else None,
                "policy": (
                    "Forecast refresh failed, but the agent stays alive and continues hourly live-price "
                    "monitoring with the last valid CEO decision when one exists."
                ),
            }
            if fallback_path is None:
                raise RuntimeError("Forecast refresh failed and no previous valid forecast report is available.") from exc
            report_path = fallback_path
            log_progress(f"daily forecast refresh failed; using fallback report={report_path}")
    else:
        log_progress(f"using cached forecast report={report_path}")
    return load_forecast_decision(report_path)


def is_forecast_due(*, args: argparse.Namespace, state: dict[str, Any], now: datetime) -> bool:
    local_now = now.astimezone(ZoneInfo(args.active_timezone))
    last_date = str(state.get("last_forecast_date") or "")
    today = local_now.date().isoformat()
    if last_date != today:
        failed_at = _parse_utc_datetime(state.get("last_forecast_refresh_failed_at_utc"))
        if failed_at is not None and (now - failed_at).total_seconds() < max(0, int(args.forecast_retry_after_seconds)):
            return False
        return local_now.time() >= _parse_local_time(args.daily_forecast_local_time)
    return False


def run_morning_forecast(*, args: argparse.Namespace, now: datetime) -> Path:
    project_dir = Path(args.project_dir).resolve()
    output_dir = Path(args.output_dir)
    run_date = _local_date(now, args.active_timezone)
    forecast_root = output_dir / "forecasts"
    forecast_root.mkdir(parents=True, exist_ok=True)
    final_output = forecast_root / run_date
    attempt_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    forecast_output = forecast_root / f".{run_date}.tmp.{attempt_stamp}.{os.getpid()}"
    failed_output = forecast_root / f"{run_date}.failed.{attempt_stamp}"
    forecast_output.mkdir(parents=True, exist_ok=True)
    start = (now - timedelta(days=int(args.forecast_lookback_days))).date().isoformat()
    cmd = [
        sys.executable,
        "-m",
        "market_forecasting_engine.cli",
        "--ticker",
        args.ticker,
        "--provider",
        args.forecast_provider,
        "--start",
        start,
        "--interval",
        args.forecast_interval,
        "--horizons",
        args.forecast_horizons,
        "--min-training-rows",
        str(args.forecast_min_training_rows),
        "--validation-window",
        str(args.forecast_validation_window),
        "--step-size",
        str(args.forecast_step_size),
        "--max-splits",
        str(args.forecast_max_splits),
        "--validation-workers",
        str(args.forecast_validation_workers),
        "--selection-metric",
        "mae",
        "--tactical-profile",
        "short_term",
        "--calendar",
        "24/7",
        "--trader-profile",
        "aggressive",
        "--holding-status",
        "not_owned",
        "--output-dir",
        str(forecast_output),
        "--llm-timeout",
        str(args.forecast_llm_timeout),
    ]
    if args.llm_env_file:
        cmd.extend(["--llm-env-file", args.llm_env_file])
    env = os.environ.copy()
    src = project_dir / "automated_forecasting_engine" / "src"
    env["PYTHONPATH"] = str(src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    started = datetime.now(UTC)
    try:
        result = _run_streaming_forecast_command(
            cmd=cmd,
            cwd=project_dir,
            env=env,
            timeout_seconds=max(300, int(args.forecast_timeout_seconds)),
        )
    except subprocess.TimeoutExpired as exc:
        command_report = {
            "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "returncode": None,
            "timeout_seconds": int(args.forecast_timeout_seconds),
            "stdout_tail": (exc.stdout or exc.output or "")[-8000:] if isinstance(exc.stdout or exc.output, str) else "",
            "stderr_tail": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "",
            "forecast_output_dir": str(forecast_output),
            "status": "timeout",
            "policy": "Timed-out forecast attempts remain in a failed directory and are never promoted as executable reports.",
        }
        _write_command_result_and_archive(forecast_output=forecast_output, failed_output=failed_output, command_report=command_report)
        raise RuntimeError(f"Morning forecast timed out before mandatory CEO decision. See {failed_output / 'forecast_command_result.json'}") from exc
    command_report = {
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-8000:],
        "stderr_tail": result.stderr[-8000:],
        "forecast_output_dir": str(forecast_output),
        "status": "completed" if result.returncode == 0 else "failed",
    }
    (forecast_output / "forecast_command_result.json").write_text(json.dumps(command_report, indent=2), encoding="utf-8")
    if result.returncode != 0:
        _archive_attempt(forecast_output, failed_output)
        raise RuntimeError(f"Morning forecast failed. See {failed_output / 'forecast_command_result.json'}")
    report_path = forecast_output / "forecast_report.json"
    if not report_path.exists():
        _archive_attempt(forecast_output, failed_output)
        raise RuntimeError(f"Morning forecast completed but did not write {failed_output / 'forecast_report.json'}.")
    try:
        load_forecast_decision(report_path)
    except Exception as exc:
        command_report["status"] = "failed_missing_mandatory_llm_final_decision"
        command_report["mandatory_field"] = "llm_final_decision.decision"
        command_report["validation_error"] = str(exc)
        (forecast_output / "forecast_command_result.json").write_text(json.dumps(command_report, indent=2), encoding="utf-8")
        _archive_attempt(forecast_output, failed_output)
        raise RuntimeError(
            f"Morning forecast completed without mandatory llm_final_decision.decision. "
            f"Attempt archived at {failed_output} and was not promoted."
        ) from exc
    _promote_forecast_attempt(forecast_output=forecast_output, final_output=final_output)
    return final_output / "forecast_report.json"


def _run_streaming_forecast_command(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    forecast_env = env.copy()
    forecast_env.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=forecast_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    started_monotonic = time.monotonic()
    chunks: list[str] = []
    while True:
        elapsed = time.monotonic() - started_monotonic
        if elapsed > timeout_seconds:
            process.kill()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            output = "".join(chunks)
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_seconds, output=output)
        events = selector.select(timeout=0.2)
        for key, _ in events:
            raw = os.read(key.fileobj.fileno(), 4096)
            if not raw:
                try:
                    selector.unregister(key.fileobj)
                except KeyError:
                    pass
                continue
            text = raw.decode("utf-8", errors="replace")
            chunks.append(text)
            sys.stdout.write(text)
            sys.stdout.flush()
        if process.poll() is not None:
            for key, _ in selector.select(timeout=0):
                raw = os.read(key.fileobj.fileno(), 4096)
                if raw:
                    text = raw.decode("utf-8", errors="replace")
                    chunks.append(text)
                    sys.stdout.write(text)
                    sys.stdout.flush()
            break
    selector.close()
    output = "".join(chunks)
    return subprocess.CompletedProcess(args=cmd, returncode=int(process.returncode or 0), stdout=output, stderr="")


def _write_command_result_and_archive(*, forecast_output: Path, failed_output: Path, command_report: dict[str, Any]) -> None:
    forecast_output.mkdir(parents=True, exist_ok=True)
    (forecast_output / "forecast_command_result.json").write_text(json.dumps(command_report, indent=2), encoding="utf-8")
    _archive_attempt(forecast_output, failed_output)


def _archive_attempt(forecast_output: Path, failed_output: Path) -> None:
    if failed_output.exists():
        shutil.rmtree(failed_output)
    if forecast_output.exists():
        shutil.move(str(forecast_output), str(failed_output))


def _promote_forecast_attempt(*, forecast_output: Path, final_output: Path) -> None:
    archive = final_output.with_name(f"{final_output.name}.previous.{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}")
    if final_output.exists():
        shutil.move(str(final_output), str(archive))
    shutil.move(str(forecast_output), str(final_output))


def load_forecast_decision(report_path: Path) -> ForecastDecision:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    ceo = _extract_ceo_decision(report)
    final_advice = _extract_final_advice(report, ceo)
    return ForecastDecision(
        report_path=report_path,
        report=report,
        ceo_decision=ceo,
        final_advice=final_advice,
        created_at_utc=str(report.get("generated_at_utc") or report.get("as_of_timestamp") or "") or None,
    )


def _extract_ceo_decision(report: dict[str, Any]) -> dict[str, Any]:
    direct = report.get("llm_final_decision")
    if isinstance(direct, dict) and direct.get("decision"):
        return direct
    raise RuntimeError(
        "Forecast report is not executable for the ETH/USDC daily agent: "
        "mandatory llm_final_decision.decision is missing."
    )


def _extract_final_advice(report: dict[str, Any], ceo: dict[str, Any]) -> dict[str, Any]:
    advice = ceo.get("final_advice") if isinstance(ceo, dict) else None
    if isinstance(advice, dict):
        return advice
    reasoning = report.get("final_decision_reasoning")
    if isinstance(reasoning, dict) and isinstance(reasoning.get("final_advice"), dict):
        return reasoning["final_advice"]
    return {}


def read_live_market(*, args: argparse.Namespace, broker: DeribitLiveSpotBroker) -> dict[str, Any]:
    instrument = args.instrument.upper()
    base = args.base_currency.upper()
    quote_currency = args.quote_currency.upper()
    instrument_details = broker.public_get("get_instrument", {"instrument_name": instrument})
    order_book = broker.order_book(instrument, depth=10)
    ticker = broker.ticker(instrument)
    quote = book_quote(order_book)
    latest_price = quote["mid"] or quote["ask"] or quote["bid"] or _float(ticker.get("last_price"))
    base_account = broker.account_summary(currency=base)
    quote_account = broker.account_summary(currency=quote_currency)
    open_orders = dedupe_orders(
        [
            *broker.open_orders(currency=base, kind="spot"),
            *broker.open_orders(currency=quote_currency, kind="spot"),
        ]
    )
    return {
        "instrument": instrument,
        "instrument_details": instrument_details,
        "ticker": ticker,
        "order_book": order_book,
        "quote": quote,
        "latest_price": latest_price,
        "spread_pct": spread_fraction(quote),
        "account": {
            base: account_subset(base_account),
            quote_currency: account_subset(quote_account),
        },
        "open_orders": open_orders,
    }


def reconcile_open_sell_orders(
    *,
    args: argparse.Namespace,
    broker: DeribitLiveSpotBroker,
    state: dict[str, Any],
    market_packet: dict[str, Any],
) -> dict[str, Any]:
    instrument = args.instrument.upper()
    base = args.base_currency.upper()
    base_account = ((market_packet.get("account") or {}).get(base) or {})
    base_balance = _float(base_account.get("balance"))
    managed_base_balance = managed_balance(args=args, state=state, base_balance=base_balance)
    open_orders = market_packet.get("open_orders") or []
    sell_orders = [_summarize_sell_order(order) for order in open_orders if _is_active_sell_order(order, instrument)]
    sell_orders = [order for order in sell_orders if order["remaining_amount"] > 0]
    total_open_sell_amount = sum(float(order["remaining_amount"]) for order in sell_orders)
    exposure_limit = max(0.0, managed_base_balance)
    tolerance = max(1e-9, float(args.min_order_base_amount) * 0.1)
    base_payload: dict[str, Any] = {
        "status": "ok",
        "policy": (
            "Deribit spot does not provide a native OCO guarantee in this agent path. "
            "The agent caps active sell exposure to the managed ETH balance. It may cancel only "
            "agent-labelled excess sell orders; unmanaged/manual sell orders require visible review."
        ),
        "instrument": instrument,
        "base_balance": base_balance,
        "managed_base_balance": managed_base_balance,
        "total_open_sell_amount": total_open_sell_amount,
        "excess_sell_amount": max(0.0, total_open_sell_amount - exposure_limit),
        "sell_order_count": len(sell_orders),
        "cancelled_count": 0,
        "cancelled_orders": [],
        "would_cancel_orders": [],
        "kept_orders": [],
        "unmanaged_sell_orders": [],
    }
    if total_open_sell_amount <= exposure_limit + tolerance:
        base_payload["kept_orders"] = sell_orders
        return base_payload
    managed_orders = [order for order in sell_orders if _is_agent_managed_label(order.get("label"))]
    unmanaged_orders = [order for order in sell_orders if not _is_agent_managed_label(order.get("label"))]
    unmanaged_amount = sum(float(order["remaining_amount"]) for order in unmanaged_orders)
    remaining_capacity = max(0.0, exposure_limit - unmanaged_amount)
    keep: list[dict[str, Any]] = []
    cancel: list[dict[str, Any]] = []
    for order in sorted(managed_orders, key=_sell_order_keep_priority):
        remaining_amount = float(order["remaining_amount"])
        if remaining_amount <= remaining_capacity + tolerance:
            keep.append(order)
            remaining_capacity = max(0.0, remaining_capacity - remaining_amount)
        else:
            cancel.append(order)
    base_payload["kept_orders"] = [*unmanaged_orders, *keep]
    base_payload["unmanaged_sell_orders"] = unmanaged_orders
    if unmanaged_orders:
        base_payload["status"] = "manual_review_required" if unmanaged_amount > exposure_limit + tolerance else "reconciled_agent_orders_only"
        base_payload["manual_review_reason"] = (
            "Unmanaged/manual active sell orders contribute to sell exposure. "
            "The agent will not cancel them automatically."
        )
    else:
        base_payload["status"] = "reconciled_needed"
    if not cancel:
        return base_payload
    if not bool(args.execute_live_orders):
        base_payload["status"] = "dry_run_would_cancel_excess_agent_sell_orders"
        base_payload["would_cancel_orders"] = cancel
        return base_payload
    cancelled: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for order in cancel:
        order_id = str(order.get("order_id") or "")
        if not order_id:
            failed.append({"order": order, "error": "missing_order_id"})
            continue
        try:
            result = broker.cancel_order(order_id)
        except Exception as exc:
            failed.append({"order": order, "error": str(exc)})
            continue
        cancelled.append({"order": order, "result": result})
    base_payload["cancelled_orders"] = cancelled
    base_payload["cancelled_count"] = len(cancelled)
    if failed:
        base_payload["status"] = "cancel_failed_manual_review_required"
        base_payload["cancel_failures"] = failed
    elif unmanaged_orders and unmanaged_amount > exposure_limit + tolerance:
        base_payload["status"] = "manual_review_required"
    else:
        base_payload["status"] = "reconciled"
    return base_payload


def _is_active_sell_order(order: dict[str, Any], instrument: str) -> bool:
    if str(order.get("instrument_name") or "").upper() != instrument.upper():
        return False
    if str(order.get("direction") or "").lower() != "sell":
        return False
    state = str(order.get("order_state") or "").lower()
    return state in {"open", "untriggered"}


def _summarize_sell_order(order: dict[str, Any]) -> dict[str, Any]:
    amount = max(0.0, _float(order.get("amount")) - _float(order.get("filled_amount")))
    order_type = str(order.get("order_type") or order.get("type") or "").lower()
    trigger_price = _float(order.get("trigger_price"))
    return {
        "order_id": order.get("order_id"),
        "label": order.get("label"),
        "instrument_name": order.get("instrument_name"),
        "direction": order.get("direction"),
        "order_state": order.get("order_state"),
        "order_type": order_type,
        "remaining_amount": amount,
        "price": _float(order.get("price")) or None,
        "trigger_price": trigger_price or None,
        "is_stop": _is_stop_order_summary(order_type=order_type, trigger_price=trigger_price),
    }


def _is_stop_order_summary(*, order_type: str, trigger_price: float) -> bool:
    return "stop" in order_type or trigger_price > 0


def _is_agent_managed_label(label: Any) -> bool:
    return str(label or "").startswith("codex-eth-usdc-daily-")


def _sell_order_keep_priority(order: dict[str, Any]) -> tuple[int, float, float, str]:
    is_stop = bool(order.get("is_stop"))
    remaining_amount = float(order.get("remaining_amount") or 0.0)
    price = float(order.get("trigger_price") or order.get("price") or 0.0)
    return (0 if is_stop else 1, remaining_amount, price, str(order.get("order_id") or ""))


def decide_from_cached_ceo(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    forecast_record: ForecastDecision,
    market_packet: dict[str, Any],
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    ceo = forecast_record.ceo_decision
    advice = forecast_record.final_advice
    quote = market_packet.get("quote") or {}
    latest_price = _float(market_packet.get("latest_price"))
    spread_pct = market_packet.get("spread_pct")
    instrument_details = market_packet.get("instrument_details") or {}
    args.price_tick_size = _float(instrument_details.get("tick_size")) or 0.01
    args.base_amount_step = _float(instrument_details.get("contract_size")) or 0.000001
    args.min_order_base_amount = max(float(args.min_order_base_amount), _float(instrument_details.get("min_trade_amount")) or 0.0)
    base_account = ((market_packet.get("account") or {}).get(args.base_currency.upper()) or {})
    quote_account = ((market_packet.get("account") or {}).get(args.quote_currency.upper()) or {})
    base_balance = _float(base_account.get("balance"))
    quote_available = _float(quote_account.get("available_funds"))
    managed_base_balance = managed_balance(args=args, state=state, base_balance=base_balance)
    open_orders = market_packet.get("open_orders") or []
    base = _base_decision_payload(
        args=args,
        latest_price=latest_price,
        spread_pct=spread_pct,
        base_balance=base_balance,
        managed_base_balance=managed_base_balance,
        quote_available=quote_available,
        ceo=ceo,
        advice=advice,
    )
    if not ceo:
        return {**base, "action": "hold", "reason": "missing_ceo_final_decision", "execution_allowed": False}
    if latest_price <= 0:
        return {**base, "action": "hold", "reason": "missing_live_price", "execution_allowed": False}
    if spread_pct is not None and float(spread_pct) > float(args.max_spread_pct):
        return {**base, "action": "hold", "reason": "spread_too_wide", "execution_allowed": False}
    if has_open_entry(open_orders=open_orders, instrument=args.instrument):
        return {**base, "action": "hold", "reason": "open_entry_order_exists", "execution_allowed": False}
    protection = maybe_protection_plan(args=args, latest_price=latest_price, managed_base_balance=managed_base_balance, open_orders=open_orders, advice=advice)
    if protection is not None:
        return {**base, **protection, "created_at_utc": now}
    action = str(advice.get("action_now") or ceo.get("decision") or "").lower()
    buy_plan = maybe_buy_plan(
        args=args,
        state=state,
        latest_price=latest_price,
        quote=quote,
        quote_available=quote_available,
        base_balance=base_balance,
        managed_base_balance=managed_base_balance,
        advice=advice,
        action=action,
    )
    if buy_plan is not None:
        return {**base, **buy_plan, "created_at_utc": now}
    sell_plan = maybe_sell_plan(
        args=args,
        latest_price=latest_price,
        managed_base_balance=managed_base_balance,
        quote=quote,
        open_orders=open_orders,
        advice=advice,
        action=action,
    )
    if sell_plan is not None:
        return {**base, **sell_plan, "created_at_utc": now}
    return {**base, "action": "hold", "reason": "cached_ceo_decision_no_live_trigger", "execution_allowed": False, "created_at_utc": now}


def maybe_buy_plan(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    latest_price: float,
    quote: dict[str, Any],
    quote_available: float,
    base_balance: float,
    managed_base_balance: float,
    advice: dict[str, Any],
    action: str,
) -> dict[str, Any] | None:
    if managed_base_balance >= float(args.max_base_position):
        return None
    invalidation = _float_or_none(advice.get("invalidation_price")) or _float_or_none(advice.get("stop_loss_price"))
    if invalidation and latest_price <= invalidation:
        return {
            "action": "hold",
            "reason": "buy_blocked_below_ceo_invalidation",
            "execution_allowed": False,
            "invalidation_price": invalidation,
        }
    triggers: list[tuple[str, float | None]] = [
        ("buy_now", _float_or_none(advice.get("buy_now_price"))),
        ("buy_lower", _float_or_none(advice.get("buy_lower_price")) or _float_or_none(advice.get("buy_lower_zone_high"))),
        ("breakout", _float_or_none(advice.get("buy_above_breakout_price"))),
    ]
    selected: tuple[str, float | None] | None = None
    tolerance = float(args.entry_price_tolerance_pct)
    for name, level in triggers:
        if name == "buy_now" and (action == "buy" or level):
            if level is None or latest_price <= level * (1.0 + tolerance):
                selected = (name, level)
                break
        if name == "buy_lower" and level and latest_price <= level * (1.0 + tolerance):
            selected = (name, level)
            break
        if name == "breakout" and level and latest_price >= level:
            selected = (name, level)
            break
    if selected is None:
        return None
    trigger_name, trigger_level = selected
    signal_key = f"{_local_date(datetime.now(UTC), args.active_timezone)}:{trigger_name}:{trigger_level or 'market'}"
    if signal_key in set(state.get("submitted_signal_keys") or []):
        return {"action": "hold", "reason": "signal_already_submitted_today", "execution_allowed": False, "signal_key": signal_key}
    ask = _float(quote.get("ask")) or latest_price
    remaining_base_capacity = max(0.0, float(args.max_base_position) - base_balance)
    max_notional = min(float(args.max_notional_usdc), quote_available)
    amount = agent_round_amount(args, min(remaining_base_capacity, max_notional / ask))
    if amount < float(args.min_order_base_amount):
        return {"action": "hold", "reason": "not_enough_quote_or_capacity_for_min_order", "execution_allowed": False}
    limit_price = agent_round_price(args, ask * (1.0 + tolerance))
    return {
        "action": "buy_spot",
        "reason": f"cached_ceo_{trigger_name}_trigger_reached",
        "execution_allowed": True,
        "signal_key": signal_key,
        "entry_order": {"side": "buy", "type": "limit", "amount": amount, "price": limit_price},
        "post_fill_protection_plan": protection_orders_from_advice(args=args, amount=amount, entry_reference=limit_price, advice=advice),
    }


def maybe_sell_plan(
    *,
    args: argparse.Namespace,
    latest_price: float,
    managed_base_balance: float,
    quote: dict[str, Any],
    open_orders: list[dict[str, Any]],
    advice: dict[str, Any],
    action: str,
) -> dict[str, Any] | None:
    if managed_base_balance < float(args.min_order_base_amount):
        return None
    stop = _float_or_none(advice.get("stop_loss_price")) or _float_or_none(advice.get("invalidation_price"))
    trim = _float_or_none(advice.get("sell_or_trim_price"))
    take_profit = _float_or_none(advice.get("take_profit_price"))
    trigger = None
    if action == "sell":
        trigger = "ceo_sell_now"
    elif stop and latest_price <= stop:
        trigger = "stop_loss_reached"
    elif trim and latest_price >= trim:
        trigger = "sell_or_trim_price_reached"
    elif take_profit and latest_price >= take_profit:
        trigger = "take_profit_price_reached"
    if trigger is None:
        return None
    existing_protection = analyze_existing_protection(
        open_orders=open_orders,
        instrument=args.instrument.upper(),
        base_balance=managed_base_balance,
        latest_price=latest_price,
    )
    if existing_protection.get("sell_coverage_ratio", 0.0) >= float(args.stop_protection_coverage_ratio):
        return {
            "action": "hold",
            "reason": "existing_sell_orders_already_cover_position",
            "execution_allowed": False,
            "trigger_seen": trigger,
            "existing_protection": existing_protection,
            "policy": (
                "Deribit spot orders are not treated as native OCO here. If active sell orders already "
                "reserve the managed ETH balance, the agent does not submit another sell."
            ),
        }
    bid = _float(quote.get("bid")) or latest_price
    amount = agent_round_amount(args, min(managed_base_balance, float(args.max_base_position)))
    if amount < float(args.min_order_base_amount):
        return None
    return {
        "action": "sell_spot",
        "reason": trigger,
        "execution_allowed": True,
        "entry_order": {"side": "sell", "type": "limit", "amount": amount, "price": agent_round_price(args, bid * 0.9985)},
    }


def maybe_protection_plan(
    *,
    args: argparse.Namespace,
    latest_price: float,
    managed_base_balance: float,
    open_orders: list[dict[str, Any]],
    advice: dict[str, Any],
) -> dict[str, Any] | None:
    if managed_base_balance < float(args.min_order_base_amount):
        return None
    protection = analyze_existing_protection(
        open_orders=open_orders,
        instrument=args.instrument.upper(),
        base_balance=managed_base_balance,
        latest_price=latest_price,
    )
    if protection.get("sell_coverage_ratio", 0.0) >= float(args.stop_protection_coverage_ratio) and not bool(args.replace_protection):
        return None
    amount = agent_round_amount(args, managed_base_balance)
    orders = protection_orders_from_advice(args=args, amount=amount, entry_reference=latest_price, advice=advice)
    if not orders:
        return None
    return {
        "action": "protect_existing_position",
        "reason": "existing_position_missing_or_insufficient_protection",
        "execution_allowed": True,
        "entry_order": None,
        "protection": orders,
        "existing_protection": protection,
    }


def protection_orders_from_advice(*, args: argparse.Namespace, amount: float, entry_reference: float, advice: dict[str, Any]) -> dict[str, Any]:
    orders: dict[str, Any] = {}
    take_profit = _float_or_none(advice.get("take_profit_price")) or _float_or_none(advice.get("sell_or_trim_price"))
    stop = _float_or_none(advice.get("stop_loss_price")) or _float_or_none(advice.get("invalidation_price"))
    if take_profit and take_profit > entry_reference:
        orders["take_profit"] = {"side": "sell", "type": "limit", "amount": amount, "price": agent_round_price(args, take_profit)}
    if stop and stop < entry_reference:
        trigger = agent_round_price(args, stop)
        orders["stop_loss"] = {
            "side": "sell",
            "type": "stop_limit",
            "amount": amount,
            "price": agent_round_price(args, trigger * 0.998),
            "trigger_price": trigger,
            "trigger": "index_price",
        }
    return orders


def execute_live_plan(*, broker: DeribitLiveSpotBroker, args: argparse.Namespace, plan: dict[str, Any], label_base: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    instrument = args.instrument.upper()
    entry = plan.get("entry_order")
    entry_result: dict[str, Any] | None = None
    if entry:
        signal_key = plan.get("signal_key")
        entry_result = _submit_spot_order_safely(
            broker=broker,
            side=entry["side"],
            instrument_name=instrument,
            amount=float(entry["amount"]),
            order_type=entry["type"],
            price=entry.get("price"),
            label=f"{label_base}-entry",
        )
        results.append({"action": "submit_entry_order", "payload": entry, "signal_key": signal_key, "result": entry_result})
    for name, order in (plan.get("protection") or {}).items():
        results.append(
            {
                "action": f"submit_{name}_order",
                "payload": order,
                "result": _submit_spot_order_safely(
                    broker=broker,
                    side=order["side"],
                    instrument_name=instrument,
                    amount=float(order["amount"]),
                    order_type=order["type"],
                    price=order.get("price"),
                    trigger_price=order.get("trigger_price"),
                    trigger=order.get("trigger"),
                    label=f"{label_base}-{name}",
                ),
            }
        )
    if entry and entry.get("side") == "buy" and plan.get("post_fill_protection_plan"):
        filled_amount = _filled_amount(entry_result)
        if filled_amount >= float(args.min_order_base_amount):
            for name, order in (plan.get("post_fill_protection_plan") or {}).items():
                adjusted = {**order, "amount": agent_round_amount(args, min(float(order["amount"]), filled_amount))}
                if adjusted["amount"] < float(args.min_order_base_amount):
                    continue
                results.append(
                    {
                        "action": f"submit_post_fill_{name}_order",
                        "payload": adjusted,
                        "result": _submit_spot_order_safely(
                            broker=broker,
                            side=adjusted["side"],
                            instrument_name=instrument,
                            amount=float(adjusted["amount"]),
                            order_type=adjusted["type"],
                            price=adjusted.get("price"),
                            trigger_price=adjusted.get("trigger_price"),
                            trigger=adjusted.get("trigger"),
                            label=f"{label_base}-post-fill-{name}",
                        ),
                    }
                )
        else:
            results.append(
                {
                    "action": "post_fill_protection_pending",
                    "reason": "entry_not_filled_yet",
                    "protection_plan": plan.get("post_fill_protection_plan"),
                }
            )
    return results


def _submit_spot_order_safely(
    *,
    broker: DeribitLiveSpotBroker,
    side: str,
    instrument_name: str,
    amount: float,
    order_type: str,
    price: float | None = None,
    trigger_price: float | None = None,
    trigger: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    try:
        result = broker.submit_spot_order(
            side=side,
            instrument_name=instrument_name,
            amount=amount,
            order_type=order_type,
            price=price,
            trigger_price=trigger_price,
            trigger=trigger,
            label=label,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "blocked_or_rejected": True,
            "policy": "Broker-side order rejection is recorded and does not crash the live monitoring loop.",
        }
    if isinstance(result, dict):
        return {"ok": True, **result}
    return {"ok": True, "result": result}


def _filled_amount(order_result: dict[str, Any] | None) -> float:
    if not isinstance(order_result, dict):
        return 0.0
    if order_result.get("ok") is False:
        return 0.0
    order = order_result.get("order") if isinstance(order_result.get("order"), dict) else {}
    filled = _float(order.get("filled_amount"))
    if filled > 0:
        return filled
    trades = order_result.get("trades") if isinstance(order_result.get("trades"), list) else []
    return sum(_float(trade.get("amount")) for trade in trades if isinstance(trade, dict))


def _base_decision_payload(
    *,
    args: argparse.Namespace,
    latest_price: float,
    spread_pct: float | None,
    base_balance: float,
    managed_base_balance: float,
    quote_available: float,
    ceo: dict[str, Any],
    advice: dict[str, Any],
) -> dict[str, Any]:
    return {
        "instrument": args.instrument.upper(),
        "latest_price": latest_price,
        "spread_pct": spread_pct,
        "base_balance": base_balance,
        "managed_base_balance": managed_base_balance,
        "quote_available": quote_available,
        "ceo_action": ceo.get("decision"),
        "ceo_confidence": ceo.get("confidence"),
        "final_advice": advice,
        "limits": {
            "max_notional_usdc": float(args.max_notional_usdc),
            "max_base_position": float(args.max_base_position),
            "min_order_base_amount": float(args.min_order_base_amount),
            "max_spread_pct": float(args.max_spread_pct),
            "inventory_scope": args.inventory_scope,
        },
    }


def has_open_entry(*, open_orders: list[dict[str, Any]], instrument: str) -> bool:
    for order in open_orders:
        if str(order.get("instrument_name") or "").upper() != instrument.upper():
            continue
        if str(order.get("direction") or "").lower() == "buy" and str(order.get("order_state") or "").lower() in {"open", "untriggered"}:
            return True
    return False


def managed_balance(*, args: argparse.Namespace, state: dict[str, Any], base_balance: float) -> float:
    if args.inventory_scope == "allow_manual":
        return base_balance
    if args.managed_base_balance is not None:
        return min(base_balance, max(0.0, float(args.managed_base_balance)))
    return min(base_balance, max(0.0, _float(state.get("managed_base_balance"))))


def update_state_from_order_results(*, state: dict[str, Any], instrument: str, results: list[dict[str, Any]]) -> None:
    submitted = set(state.get("submitted_signal_keys") or [])
    for result in results:
        if result.get("action") == "submit_entry_order":
            signal_key = result.get("signal_key")
            if signal_key:
                submitted.add(str(signal_key))
    state["submitted_signal_keys"] = sorted(submitted)[-100:]
    state.setdefault("order_history", []).extend(results)
    state["order_history"] = state["order_history"][-200:]
    state["instrument"] = instrument


def read_state(output_dir: Path, instrument: str) -> dict[str, Any]:
    path = state_path(output_dir, instrument)
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"instrument": instrument.upper(), "managed_base_balance": 0.0, "submitted_signal_keys": []}
    return parsed if isinstance(parsed, dict) else {"instrument": instrument.upper(), "managed_base_balance": 0.0, "submitted_signal_keys": []}


def write_state(output_dir: Path, instrument: str, state: dict[str, Any]) -> Path:
    path = state_path(output_dir, instrument)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(strict_json(state), indent=2, allow_nan=False, default=str), encoding="utf-8")
    return path


def state_path(output_dir: Path, instrument: str) -> Path:
    return output_dir / "state" / f"{instrument.upper()}_daily_agent_state.json"


def write_agent_report(record: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{record['instrument']}_daily_agent_report.json"
    payload = json.dumps(strict_json(record), indent=2, allow_nan=False, default=str)
    path.write_text(payload, encoding="utf-8")
    snapshots = output_dir / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    stamp = str(record.get("checked_at_utc") or datetime.now(UTC).isoformat()).replace(":", "").replace("-", "").replace("+", "Z").replace(".", "_")
    (snapshots / f"{record['instrument']}_daily_agent_report_{stamp}.json").write_text(payload, encoding="utf-8")
    return path


def append_agent_log(record: dict[str, Any], output_dir: Path) -> Path:
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    date = str(record.get("checked_at_utc") or datetime.now(UTC).isoformat())[:10]
    path = logs / f"{record['instrument']}_{date}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(strict_json(record), allow_nan=False, default=str) + "\n")
    return path


def _cycle_summary(record: dict[str, Any], report_path: Path) -> dict[str, Any]:
    decision = record.get("decision") or {}
    forecast = record.get("forecast") or {}
    return {
        "report": str(report_path),
        "instrument": record.get("instrument"),
        "latest_price": (record.get("market") or {}).get("latest_price"),
        "ceo_decision": (forecast.get("ceo_decision") or {}).get("decision"),
        "action": decision.get("action"),
        "reason": decision.get("reason"),
        "execution_allowed": decision.get("execution_allowed"),
        "execute_live_orders": (record.get("safety") or {}).get("execute_live_orders"),
        "order_result_count": len(record.get("order_results") or []),
    }


def _parse_local_time(value: str) -> local_time:
    hour, minute = [int(part) for part in value.split(":", 1)]
    return local_time(hour=hour, minute=minute)


def _parse_utc_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _latest_valid_forecast_report(output_dir: Path) -> Path | None:
    candidates = []
    forecast_root = output_dir / "forecasts"
    if forecast_root.exists():
        candidates.extend(sorted(forecast_root.glob("*/forecast_report.json"), key=lambda path: path.stat().st_mtime, reverse=True))
    candidates.append(output_dir / "forecast_report.json")
    for path in candidates:
        if not path.exists():
            continue
        try:
            load_forecast_decision(path)
        except Exception:
            continue
        return path
    return None


def _local_date(now: datetime, timezone: str) -> str:
    return now.astimezone(ZoneInfo(timezone)).date().isoformat()


def _float_or_none(value: Any) -> float | None:
    number = _float(value)
    return number if number > 0 else None


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


if __name__ == "__main__":
    main()
