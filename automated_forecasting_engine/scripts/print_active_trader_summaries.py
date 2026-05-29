#!/usr/bin/env python3
import argparse
import json
import os
import plistlib
import subprocess
from pathlib import Path


def main():
    args = build_parser().parse_args()
    agents = discover_agents(args)
    if not agents:
        print("No active watch-agent traders found.")
        return 0

    printed = 0
    print(f"Active trader summaries: {len(agents)}")
    for agent in agents:
        summary_path, summary = find_summary(agent, args)
        print_agent_summary(agent, summary_path, summary, args.compact)
        printed += 1
    return 0 if printed else 1


def build_parser():
    parser = argparse.ArgumentParser(description="Print the latest plain-language summary for active watch-agent traders.")
    parser.add_argument("--project-dir", default="/Users/ruddigarcia/Projects/invest")
    parser.add_argument("--state-dir", default="automated_forecasting_engine/runs/watch_agent_state")
    parser.add_argument("--launch-agents-dir", default="~/Library/LaunchAgents")
    parser.add_argument("--include-inactive", action="store_true", help="Include matching plist files even if launchd has not loaded them.")
    parser.add_argument("--compact", action="store_true", help="Print only the short summary, action, and recheck triggers.")
    parser.add_argument("--details", action="store_true", help="Deprecated; full detail is printed by default.")
    return parser


def discover_agents(args):
    launch_agents_dir = Path(args.launch_agents_dir).expanduser()
    plists = sorted(launch_agents_dir.glob("com.marketforecasting.watchagent*.plist"))
    agents = []
    seen_labels = set()
    for plist_path in plists:
        data = plistlib.loads(plist_path.read_bytes())
        label = data.get("Label")
        if not label:
            continue
        loaded, loaded_path = launchctl_status(label)
        if not loaded and not args.include_inactive:
            continue
        if loaded_path and loaded_path.exists() and plist_path.resolve() != loaded_path.resolve():
            continue
        if label in seen_labels:
            continue
        seen_labels.add(label)
        env = data.get("EnvironmentVariables") or {}
        project_dir = Path(env.get("PROJECT_DIR") or data.get("WorkingDirectory") or args.project_dir).expanduser()
        state_dir = Path(env.get("STATE_DIR") or args.state_dir).expanduser()
        if not state_dir.is_absolute():
            state_dir = project_dir / state_dir
        ticker = env.get("TICKER") or infer_ticker_from_label(label)
        profile = env.get("PROFILE") or infer_profile_from_label(label)
        agents.append(
            {
                "label": label,
                "plist": plist_path,
                "loaded": loaded,
                "ticker": ticker,
                "profile": profile,
                "holding_status": env.get("HOLDING_STATUS", "unknown"),
                "project_dir": project_dir,
                "state_dir": state_dir,
            }
        )
    return agents


def launchctl_status(label):
    service = f"gui/{os.getuid()}/{label}"
    result = subprocess.run(
        ["launchctl", "print", service],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False, None
    loaded_path = None
    for line in result.stdout.splitlines():
        clean = line.strip()
        if clean.startswith("path = "):
            loaded_path = Path(clean.removeprefix("path = ").strip())
            break
    return True, loaded_path


def infer_ticker_from_label(label):
    parts = label.split(".")
    if len(parts) >= 2:
        return parts[-2].upper()
    return "UNKNOWN"


def infer_profile_from_label(label):
    parts = label.split(".")
    if parts:
        return parts[-1].lower()
    return "unknown"


def find_summary(agent, args):
    candidates = []
    memory = load_memory(agent)
    if memory:
        run_dir = memory.get("run_dir")
        decision_file = memory.get("decision_file")
        if run_dir:
            candidates.append(Path(run_dir) / "trader_summary.json")
        if decision_file:
            candidates.append(Path(decision_file).parent / "trader_summary.json")

    candidates.append(agent["state_dir"] / "llm_run" / safe_trader_name(agent["ticker"], agent["profile"]) / "trader_summary.json")
    candidates.extend(find_matching_summaries(agent))

    seen = set()
    existing = []
    for path in candidates:
        path = absolutize(path, agent["project_dir"])
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            existing.append(path)

    if not existing:
        return None, None

    summary_path = max(existing, key=lambda path: path.stat().st_mtime)
    summary = json.loads(summary_path.read_text())
    return summary_path, summary


def load_memory(agent):
    state_file = agent["state_dir"] / f"{safe_trader_name(agent['ticker'], agent['profile'])}.json"
    if not state_file.exists():
        return {}
    return json.loads(state_file.read_text())


def find_matching_summaries(agent):
    runs_root = agent["project_dir"] / "automated_forecasting_engine" / "runs"
    if not runs_root.exists():
        return []
    matches = []
    for summary_path in runs_root.rglob("trader_summary.json"):
        decision_path = summary_path.parent / "trader_decision.json"
        if not decision_path.exists():
            continue
        try:
            decision = json.loads(decision_path.read_text())
        except json.JSONDecodeError:
            continue
        profile = decision.get("trader_profile") or {}
        if str(decision.get("ticker", "")).upper() != agent["ticker"].upper():
            continue
        if str(profile.get("name", "")).lower() != agent["profile"].lower():
            continue
        matches.append(summary_path)
    return matches


def safe_trader_name(ticker, profile):
    return f"{ticker.upper()}_{profile}".replace("/", "_").replace(" ", "_")


def absolutize(path, project_dir):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return project_dir / path


def print_agent_summary(agent, summary_path, summary, compact):
    print()
    print("=" * 80)
    print(f"{agent['ticker']} | profile={agent['profile']} | holding={agent['holding_status']}")
    print(f"Label: {agent['label']}")
    if not summary:
        expected = agent["state_dir"] / "llm_run" / safe_trader_name(agent["ticker"], agent["profile"]) / "trader_summary.json"
        print("No trader_summary.json found yet for this active trader.")
        print(f"Expected: {expected}")
        return

    print(f"Decision: {summary.get('decision', 'unknown')} | Risk: {summary.get('risk_level', 'unknown')} | Confidence: {summary.get('confidence', 'unknown')}")
    if summary.get("headline"):
        print(f"Headline: {summary['headline']}")
    print()
    print_section("Plain summary", summary.get("plain_language_summary") or "plain_language_summary is missing in trader_summary.json.")
    print_section("What to do now", summary.get("what_to_do_now"))
    print_section("If not owned", summary.get("if_not_owned"))
    print_section("If owned", summary.get("if_owned"))
    print_decision_triggers(summary)
    if not compact:
        print_prices(summary.get("important_prices") or [])
        print_list("Why this decision", summary.get("why_this_decision") or [])
        print_list("Main risks", summary.get("main_risks") or [])
        print_list("Beginner notes", summary.get("beginner_notes") or [])
        print_section("Currency note", summary.get("currency_note"))
    print(f"Source: {summary_path}")


def print_section(label, value):
    if value:
        print(f"{label}:")
        print(f"  {value}")


def print_list(label, items):
    if not items:
        return
    print(f"{label}:")
    for item in items:
        print(f"- {item}")


def print_prices(items):
    if not items:
        return
    print("Important prices:")
    for item in items:
        label = item.get("label", "Price")
        display = item.get("display")
        if not display:
            display = format_price_pair(item.get("price_usd"), item.get("price_eur"))
        meaning = item.get("plain_meaning")
        print(f"- {label}: {display}")
        if meaning:
            print(f"  Meaning: {meaning}")


def print_decision_triggers(summary):
    triggers = summary.get("decision_triggers") or []
    if triggers:
        print("Decision triggers:")
        for trigger in triggers:
            print(f"- Watch: {trigger.get('trigger')}")
            print(f"  Goal: {human_goal(trigger.get('decision_goal'))}")
            print(f"  If not owned: {trigger.get('if_not_owned_action')}")
            print(f"  If owned: {trigger.get('if_owned_action')}")
            print(f"  Why: {trigger.get('plain_reason')}")
        return
    rechecks = summary.get("when_to_recheck") or []
    if rechecks:
        print("When to recheck:")
        for item in rechecks:
            print(f"- {item}")


def human_goal(value):
    labels = {
        "consider_buy": "consider buying",
        "consider_sell": "consider selling",
        "consider_hold": "consider holding",
        "reduce_risk": "reduce risk",
        "take_profit": "take profit",
        "keep_waiting": "keep waiting",
        "manual_review": "manual review",
    }
    return labels.get(value, value or "review")


def format_price_pair(price_usd, price_eur):
    if price_usd is None and price_eur is None:
        return "not available"
    if price_usd is None:
        return f"EUR {price_eur}"
    if price_eur is None:
        return f"${price_usd:,.2f}"
    return f"${price_usd:,.2f} (€{price_eur:,.2f})"


if __name__ == "__main__":
    raise SystemExit(main())
