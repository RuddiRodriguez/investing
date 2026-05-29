import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from market_forecasting_engine.llm_trader.run import run_autonomous_trader
from market_forecasting_engine.openai_models import DEFAULT_OPENAI_MODEL, DEFAULT_REASONING_EFFORT


def build_parser():
    parser = argparse.ArgumentParser(description="Watch one ticker against a remembered LLM trader decision.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--profile", choices=("aggressive", "medium", "conservative"), default="medium")
    parser.add_argument("--run-dir", default=None, help="Explicit folder containing trader_decision.json or trader_decision_only.json.")
    parser.add_argument("--decision-file", default=None, help="Explicit trader_decision.json or trader_decision_only.json path.")
    parser.add_argument("--state-dir", default="automated_forecasting_engine/runs/watch_agent_state")
    parser.add_argument("--trader-output-dir", default=None, help="Where to save the startup LLM trader run when memory is missing or refreshed.")
    parser.add_argument("--force-refresh", action="store_true", help="Force a new forecast + LLM trader run at startup.")
    parser.add_argument("--refresh-after-hours", type=float, default=12.0, help="Refresh the forecast + LLM trader memory after this many hours.")
    parser.add_argument("--log-dir", default=None, help="Where to append hourly watch decision logs. Defaults to state-dir/logs.")
    parser.add_argument("--holding-status", choices=("not_owned", "owned"), default="not_owned")
    parser.add_argument("--interval-seconds", type=int, default=3600)
    parser.add_argument("--once", action="store_true", help="Check once and exit.")
    parser.add_argument("--price", type=float, default=None, help="Manual price for testing or offline checks.")
    parser.add_argument("--csv")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--adjustment-policy", default="auto_adjust")
    parser.add_argument("--target-column", default="close")
    parser.add_argument("--horizons", default="1,5,30")
    parser.add_argument("--selection-metric", default="mae")
    parser.add_argument("--confidence-level", type=float, default=0.80)
    parser.add_argument("--calendar", default="XNYS")
    parser.add_argument("--chart-scale", choices=("log", "linear"), default="log")
    parser.add_argument("--no-lightgbm", action="store_true")
    parser.add_argument("--no-statistical-models", action="store_true")
    parser.add_argument("--include-lstm", action="store_true")
    parser.add_argument("--trader-name", default="watch_agent_startup_trader")
    parser.add_argument("--entry-price", type=float, default=None)
    parser.add_argument("--quantity", type=float, default=None)
    parser.add_argument("--position-value", type=float, default=None)
    parser.add_argument("--account-equity", type=float, default=None)
    parser.add_argument("--portfolio-notes", default="")
    parser.add_argument("--write-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--llm-model", default=None, help=f"Defaults to OPENAI_MODEL or {DEFAULT_OPENAI_MODEL}.")
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--summary-model", default=None)
    parser.add_argument("--summary-reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument("--llm-env-file", default=None)
    parser.add_argument("--usd-eur-rate", type=float, default=None)
    parser.add_argument("--no-web-search", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--search-context-size", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress messages and only print watch actions.")
    parser.add_argument(
        "--quiet-unchanged",
        action="store_true",
        help="Only print when the watch action changes, BUY/SELL triggers, or a fresh forecast/LLM refresh ran.",
    )
    parser.add_argument("--prompt", default=str(Path(__file__).parents[1] / "llm_trader" / "prompts" / "autonomous_trader.py"))
    parser.add_argument("--summary-prompt", default=str(Path(__file__).parents[1] / "llm_trader" / "prompts" / "nontechnical_summary.py"))
    return parser


def main():
    args = build_parser().parse_args()
    progress(
        args,
        "START",
        "watch agent wake-up",
        ticker=args.ticker.upper(),
        profile=args.profile,
        holding_status=args.holding_status,
        once=args.once,
    )
    decision_path, memory = resolve_decision_file(args)
    progress(args, "DECISION", "loading remembered trader decision", decision_file=decision_path)
    decision = load_decision(decision_path)
    while True:
        progress(args, "PRICE", "checking current price", mode="manual" if args.price is not None else "live_yahoo")
        price = args.price if args.price is not None else latest_price(args.ticker)
        progress(args, "PRICE", "price resolved", price=f"{price:.2f}")
        progress(args, "ACTION", "evaluating watch levels")
        action, reason = decide_action(decision, price, args.holding_status)
        should_print = should_print_action(args, memory, action, reason)
        if should_print:
            print_loaded_advice(args, memory, decision_path, decision)
            print_alert(args.ticker, args.profile, price, action, reason, decision, args.holding_status)
        log_path = append_decision_log(args, memory, decision, price, action, reason, should_print)
        update_last_check_memory(args, memory, price, action, reason, should_print, log_path)
        progress(args, "LOG", "decision appended", log_file=log_path)
        if args.once:
            progress(args, "DONE", "one-shot wake-up complete")
            break
        progress(args, "SLEEP", "waiting for next check", seconds=max(1, int(args.interval_seconds)))
        time.sleep(max(1, int(args.interval_seconds)))


def resolve_decision_file(args):
    state_path = memory_file(args.state_dir, args.ticker, args.profile)
    progress(args, "MEMORY", "checking watcher memory", state_file=state_path)
    if not args.force_refresh and (args.decision_file or args.run_dir):
        decision_path = find_decision_file(args.ticker, args.run_dir, args.decision_file)
        progress(args, "MEMORY", "using explicit saved decision", decision_file=decision_path)
        memory = write_memory(state_path, args, decision_path, "explicit_saved_decision")
        memory["_refreshed_this_run"] = False
        return decision_path, memory
    if not args.force_refresh and state_path.exists():
        memory = json.loads(state_path.read_text())
        memory["_refreshed_this_run"] = False
        decision_path = Path(memory["decision_file"])
        age_hours = memory_age_hours(memory)
        progress(
            args,
            "MEMORY",
            "found saved watcher memory",
            decision_file=decision_path,
            age_hours=f"{age_hours:.2f}" if age_hours is not None else None,
            refresh_after_hours=args.refresh_after_hours,
        )
        if decision_path.exists() and not memory_needs_refresh(memory, args.refresh_after_hours):
            remaining = next_refresh_hours(memory, args.refresh_after_hours)
            progress(
                args,
                "MEMORY",
                "reusing saved forecast and LLM decision",
                next_refresh_in_hours=f"{remaining:.2f}" if remaining is not None else None,
            )
            return decision_path, memory
        if not decision_path.exists():
            progress(args, "MEMORY", "saved decision file is missing; full refresh required", decision_file=decision_path)
        else:
            progress(args, "MEMORY", "saved decision is stale; full refresh required")
    elif args.force_refresh:
        progress(args, "MEMORY", "force refresh requested")
    else:
        progress(args, "MEMORY", "no saved memory found; full forecast and LLM run required")
    source = "forced_refresh_forecast_llm_trader" if args.force_refresh else "startup_forecast_llm_trader"
    if state_path.exists() and not args.force_refresh:
        source = "scheduled_refresh_forecast_llm_trader"
    return run_startup_llm_forecast(args, state_path, source)


def run_startup_llm_forecast(args, state_path, source):
    output_dir = startup_output_dir(args)
    args.output_dir = str(output_dir)
    args.progress_active = True
    args.progress_logger = lambda stage, message, **fields: progress(args, stage, message, **fields)
    progress(args, "REFRESH", "starting full forecast and LLM trader run", source=source, output_dir=output_dir)
    result = run_autonomous_trader(args)
    decision_path = output_dir / "trader_decision.json"
    if not decision_path.exists():
        decision_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    progress(args, "REFRESH", "full forecast and LLM trader run finished", decision_file=decision_path)
    memory = write_memory(state_path, args, decision_path, source)
    memory["_refreshed_this_run"] = True
    progress(args, "MEMORY", "watcher memory updated", state_file=state_path)
    return decision_path, memory


def find_decision_file(ticker, run_dir=None, decision_file=None):
    if decision_file:
        path = Path(decision_file)
        if path.exists():
            return path
        raise FileNotFoundError(f"Decision file not found: {path}")
    if run_dir:
        folder = Path(run_dir)
        for name in ("trader_decision.json", "trader_decision_only.json"):
            path = folder / name
            if path.exists():
                return path
        raise FileNotFoundError(f"No trader decision file found in: {folder}")
    raise FileNotFoundError("Pass --run-dir or --decision-file, or let the watcher create its own startup LLM trader run.")


def load_decision(path):
    data = json.loads(Path(path).read_text())
    if "llm_decision" in data:
        return data["llm_decision"]
    return data


def memory_file(state_dir, ticker, profile):
    safe = f"{ticker.upper()}_{profile}".replace("/", "_").replace(" ", "_")
    return Path(state_dir) / f"{safe}.json"


def startup_output_dir(args):
    if args.trader_output_dir:
        return Path(args.trader_output_dir)
    safe = f"{args.ticker.upper()}_{args.profile}".replace("/", "_").replace(" ", "_")
    return Path(args.state_dir) / "llm_run" / safe


def write_memory(state_path, args, decision_path, source):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if state_path.exists():
        try:
            existing = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    memory = {
        "ticker": args.ticker.upper(),
        "profile": args.profile,
        "source": source,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_file": str(Path(decision_path)),
        "run_dir": str(Path(decision_path).parent),
        "force_refresh": bool(args.force_refresh),
    }
    if existing.get("ticker") == memory["ticker"] and existing.get("profile") == memory["profile"] and existing.get("last_check"):
        memory["last_check"] = existing["last_check"]
    state_path.write_text(json.dumps(memory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return memory


def memory_needs_refresh(memory, refresh_after_hours):
    if refresh_after_hours is None or float(refresh_after_hours) <= 0:
        return False
    created = memory.get("created_at_utc")
    if not created:
        return True
    created_at = datetime.fromisoformat(created.replace("Z", "+00:00"))
    age = datetime.now(timezone.utc) - created_at
    return age.total_seconds() >= float(refresh_after_hours) * 3600


def append_decision_log(args, memory, decision, price, action, reason, printed=None):
    log_dir = Path(args.log_dir) if args.log_dir else Path(args.state_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    safe = f"{args.ticker.upper()}_{args.profile}_{today}".replace("/", "_").replace(" ", "_")
    log_path = log_dir / f"{safe}.jsonl"
    entry = decision.get("entry_plan", {}) or {}
    record = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "ticker": args.ticker.upper(),
        "profile": args.profile,
        "holding_status": args.holding_status,
        "price": float(price),
        "action": action,
        "reason": reason,
        "llm_decision": decision.get("decision"),
        "llm_confidence": decision.get("confidence"),
        "memory_source": memory.get("source"),
        "forecast_refreshed_this_run": bool(memory.get("_refreshed_this_run")),
        "decision_file": memory.get("decision_file"),
        "buy_near": entry.get("buy_near"),
        "buy_above": entry.get("buy_above"),
        "sell_near": entry.get("sell_near"),
        "stop_loss": entry.get("stop_loss"),
        "take_profit": entry.get("take_profit"),
    }
    if printed is not None:
        record["printed"] = bool(printed)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return log_path


def latest_price(ticker):
    import yfinance as yf

    frame = yf.download(ticker, period="5d", interval="1h", auto_adjust=True, progress=False)
    if frame.empty:
        raise RuntimeError(f"No live price data returned for {ticker}.")
    close = frame["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        raise RuntimeError(f"No valid close price returned for {ticker}.")
    return float(close.iloc[-1])


def decide_action(decision, price, holding_status):
    entry = decision.get("entry_plan", {}) or {}
    decision_action = str(decision.get("decision") or "Hold")
    entry_style = str(entry.get("entry_style") or "")
    buy_near = number(entry.get("buy_near"))
    buy_above = number(entry.get("buy_above"))
    sell_near = number(entry.get("sell_near"))
    stop_loss = number(entry.get("stop_loss"))
    take_profit = number(entry.get("take_profit"))
    if holding_status == "owned":
        if stop_loss is not None and price <= stop_loss:
            return "SELL", "STOP_LOSS_REACHED"
        if take_profit is not None and price >= take_profit:
            return "SELL", "TAKE_PROFIT_REACHED"
        if sell_near is not None and price >= sell_near:
            return "SELL", "SELL_LEVEL_REACHED"
        if decision_action == "Sell":
            return "SELL", "LLM_DECISION_SELL"
        return "HOLD", "OWNED_NO_SELL_TRIGGER"
    if stop_loss is not None and price <= stop_loss:
        return "HOLD", "DO_NOT_ENTER_STOP_OR_INVALIDATION_BROKEN"
    if entry_style == "buy_now":
        return "BUY", "LLM_ENTRY_STYLE_BUY_NOW"
    if buy_above is not None and price >= buy_above:
        return "BUY", "BUY_ABOVE_REACHED"
    if buy_near is not None and price <= buy_near:
        return "BUY", "BUY_NEAR_REACHED"
    if decision_action == "Buy" and buy_near is None and buy_above is None:
        return "BUY", "LLM_DECISION_BUY_NO_LEVEL"
    return "HOLD", "WAITING_FOR_ADVICE_LEVEL"


def print_alert(ticker, profile, price, action, reason, decision, holding_status):
    now = datetime.now().isoformat(timespec="seconds")
    print(
        f"{now} | {ticker.upper()} | profile={profile} | holding={holding_status} | price={price:.2f} | ACTION: {action} | reason={reason}",
        flush=True,
    )
    if action in {"BUY", "SELL"}:
        print(f"*** {action} {ticker.upper()} ***", flush=True)
    elif action == "HOLD":
        print(f"*** HOLD {ticker.upper()} ***", flush=True)
    print(format_levels(decision), flush=True)


def print_loaded_advice(args, memory, decision_path, decision):
    print(
        f"WATCH AGENT STARTED ticker={args.ticker.upper()} profile={args.profile} source={memory.get('source')} decision_file={decision_path}",
        flush=True,
    )
    print(f"LOADED LLM ADVICE decision={decision.get('decision')} confidence={decision.get('confidence')}", flush=True)
    print(format_levels(decision), flush=True)


def should_print_action(args, memory, action, reason):
    if not getattr(args, "quiet_unchanged", False):
        return True
    if memory.get("_refreshed_this_run"):
        return True
    if action in {"BUY", "SELL"}:
        return True
    last = memory.get("last_check") or {}
    if not last:
        return True
    return last.get("action") != action or last.get("reason") != reason


def update_last_check_memory(args, memory, price, action, reason, printed, log_path):
    state_path = memory_file(args.state_dir, args.ticker, args.profile)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    updated = {key: value for key, value in memory.items() if not key.startswith("_")}
    updated["last_check"] = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "holding_status": args.holding_status,
        "price": float(price),
        "action": action,
        "reason": reason,
        "printed": bool(printed),
        "log_file": str(log_path),
    }
    state_path.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    memory.clear()
    memory.update(updated)


def format_levels(decision):
    entry = decision.get("entry_plan", {}) or {}
    return (
        "WATCH LEVELS "
        f"buy_near={entry.get('buy_near')} "
        f"buy_above={entry.get('buy_above')} "
        f"sell_near={entry.get('sell_near')} "
        f"stop_loss={entry.get('stop_loss')} "
        f"take_profit={entry.get('take_profit')}"
    )


def progress(args, stage, message, **fields):
    if getattr(args, "no_progress", False):
        return
    if getattr(args, "quiet_unchanged", False) and not getattr(args, "progress_active", False):
        return
    now = datetime.now().isoformat(timespec="seconds")
    clean_fields = []
    for key, value in fields.items():
        if value is None:
            continue
        clean_fields.append(f"{key}={value}")
    suffix = f" | {' '.join(clean_fields)}" if clean_fields else ""
    print(f"{now} | PROGRESS | {stage} | {message}{suffix}", flush=True)


def memory_age_hours(memory):
    created = memory.get("created_at_utc")
    if not created:
        return None
    try:
        created_at = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError:
        return None
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - created_at
    return age.total_seconds() / 3600


def next_refresh_hours(memory, refresh_after_hours):
    if refresh_after_hours is None or float(refresh_after_hours) <= 0:
        return None
    age = memory_age_hours(memory)
    if age is None:
        return None
    return max(0.0, float(refresh_after_hours) - age)


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
