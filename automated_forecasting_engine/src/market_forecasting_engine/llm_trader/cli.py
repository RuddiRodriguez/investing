import argparse
import json
from pathlib import Path

from market_forecasting_engine.llm_trader.run import run_autonomous_trader
from market_forecasting_engine.openai_models import DEFAULT_OPENAI_MODEL, DEFAULT_REASONING_EFFORT


def build_parser():
    parser = argparse.ArgumentParser(description="Run the autonomous LLM trader for one ticker.")
    parser.add_argument("--ticker", required=True)
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
    parser.add_argument("--profile", choices=("aggressive", "medium", "conservative"), default="medium")
    parser.add_argument("--trader-name", default="autonomous_trader_1")
    parser.add_argument("--holding-status", choices=("not_owned", "owned"), default="not_owned")
    parser.add_argument("--entry-price", type=float, default=None)
    parser.add_argument("--quantity", type=float, default=None)
    parser.add_argument("--position-value", type=float, default=None)
    parser.add_argument("--account-equity", type=float, default=None)
    parser.add_argument("--portfolio-notes", default="")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--write-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--llm-model", default=None, help=f"Defaults to OPENAI_MODEL or {DEFAULT_OPENAI_MODEL}.")
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--summary-model", default=None)
    parser.add_argument("--summary-reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument("--llm-env-file", default=None)
    parser.add_argument("--usd-eur-rate", type=float, default=None, help="Optional manual USD to EUR conversion rate for the beginner summary.")
    parser.add_argument("--no-web-search", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--search-context-size", choices=("low", "medium", "high"), default="medium")
    parser.add_argument(
        "--prompt",
        default=str(Path(__file__).parent / "prompts" / "autonomous_trader.py"),
    )
    parser.add_argument(
        "--summary-prompt",
        default=str(Path(__file__).parent / "prompts" / "nontechnical_summary.py"),
    )
    return parser


def main():
    args = build_parser().parse_args()
    result = run_autonomous_trader(args)
    decision = result["llm_decision"]
    trader_summary = result["nontechnical_summary"]
    print(summary(result))
    print(json.dumps(decision, indent=2, sort_keys=True, default=str))
    print(json.dumps(trader_summary, indent=2, sort_keys=True, default=str))


def summary(result):
    report = result["forecast_report"]
    decision = result["llm_decision"]
    lines = [
        f"Ticker: {result['ticker']}",
        f"Trader: {result['trader_name']} ({result['trader_profile']['name']})",
        f"Forecast Action: {report.get('suggested_action')}",
        f"Risk Level: {report.get('risk_level')}",
        f"LLM Decision: {decision.get('decision')}",
        f"Confidence: {decision.get('confidence')}",
    ]
    trader_summary = result.get("nontechnical_summary", {})
    if trader_summary.get("headline"):
        lines.append(f"Beginner Summary: {trader_summary.get('headline')}")
    entry = decision.get("entry_plan", {})
    if entry:
        lines.extend(
            [
                f"Entry Style: {entry.get('entry_style')}",
                f"Buy Near: {entry.get('buy_near')}",
                f"Buy Above: {entry.get('buy_above')}",
                f"Sell Near: {entry.get('sell_near')}",
                f"Stop Loss: {entry.get('stop_loss')}",
                f"Take Profit: {entry.get('take_profit')}",
            ]
        )
    artifacts = report.get("artifacts", {})
    if artifacts.get("forecast_report"):
        lines.append(f"Forecast Report: {artifacts['forecast_report']}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
