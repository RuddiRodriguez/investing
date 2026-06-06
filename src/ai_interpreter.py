"""Optional LLM-based interpretation for strategy results.

This module is intentionally isolated so the app still works without LLM credentials.
"""

from __future__ import annotations

import json
from typing import Any
from datetime import date

import pandas as pd
from dotenv import load_dotenv

from src.llm_handler import LLMRequest, call_llm, normalize_provider_name, resolve_llm_model


load_dotenv()


def _series_to_pct_map(series: pd.Series) -> dict[str, float]:
    return {str(k): round(float(v) * 100, 2) for k, v in series.items()}


def build_analysis_payload(results: dict[str, Any], user_note: str) -> dict[str, Any]:
    """Reduce the backtest output to a compact payload for the model."""

    latest_signals = results["latest_signal_frame"].copy()
    latest_signals["momentum"] = latest_signals["momentum"].fillna(0.0)
    latest_signals["above_trend"] = latest_signals["above_trend"].fillna(False)

    latest_signals = latest_signals.reset_index().rename(columns={"index": "ticker"})
    latest_signals = latest_signals.to_dict(orient="records")

    equity_curve = results["equity_curve"]
    drawdown = results["drawdown"]
    metrics = results["metrics"]
    metric_map = {
        "Total Return": round(float(metrics["Total Return"]) * 100, 2),
        "Annualized Return": round(float(metrics["Annualized Return"]) * 100, 2),
        "Annualized Volatility": round(float(metrics["Annualized Volatility"]) * 100, 2),
        "Sharpe (rf=0)": round(float(metrics["Sharpe (rf=0)"]), 2),
        "Max Drawdown": round(float(metrics["Max Drawdown"]) * 100, 2),
        "Win Rate": round(float(metrics["Win Rate"]) * 100, 2),
    }

    return {
        "portfolio_metrics": metric_map,
        "latest_allocation_pct": _series_to_pct_map(results["latest_allocation"]),
        "latest_signals": latest_signals,
        "last_portfolio_value": round(float(equity_curve.iloc[-1]), 4),
        "latest_drawdown_pct": round(float(drawdown.iloc[-1]) * 100, 2),
        "user_note": user_note.strip(),
    }


def interpret_results_with_openai(
    results: dict[str, Any],
    user_note: str = "",
    model: str = "gpt-4.1",
    use_web_search: bool = True,
    provider: str = "openai",
) -> str:
    """Ask the selected LLM provider for a structured interpretation."""

    provider_name = normalize_provider_name(provider)
    selected_model = resolve_llm_model(model, provider_name)
    payload = build_analysis_payload(results, user_note)

    tools = [{"type": "web_search"}] if use_web_search and provider_name in {"openai", "bedrock"} else []
    tool_choice: str | dict[str, Any] = "auto"
    if tools:
        tool_choice = "required"

    today = date.today().isoformat()
    instructions = (
        "You are a cautious investment research assistant. "
        "Interpret a medium-term ETF allocation model for a beginner retail investor. "
        "Use the provided quantitative results first, then optionally blend in current macro context, "
        "market sentiment, geopolitical risks, inflation, rates, and notable recent market news. "
        "Do not present certainty. Do not claim guaranteed profits. "
        "Use plain English and avoid technical jargon unless you explain it immediately. "
        "Keep sentences short and concrete. "
        f"Today is {today}. "
        "If web search is enabled, use current sources and anchor your comments to the current date. "
        "Do not present old news as if it happened this week or this month. "
        "If you cannot find reliable current information, say that clearly. "
        "Return concise markdown with these sections: "
        "1) What The Model Sees "
        "2) What Is Happening In The World "
        "3) Main Risks "
        "4) Simple Action Guide. "
        "The suggested stance must be one of: Buy, Hold, Reduce, or Stay In Cash. "
        "Base it on the model and current conditions, and explain the reasoning in beginner-friendly language. "
        "Treat this as educational guidance, not financial advice. "
        "When you mention news or macro events, include the month and year."
    )

    result = call_llm(
        LLMRequest(
            provider=provider_name,
            model=selected_model,
            payload={
                "instructions": instructions,
                "input": (
                    "Interpret this ETF rotation model output and provide a cautious stance.\n\n"
                    f"{json.dumps(payload, indent=2)}"
                ),
                "tools": tools,
                "tool_choice": tool_choice,
                "include": ["web_search_call.action.sources"] if tools else [],
            },
        )
    )

    output_text = result.output_text.strip()
    if not tools:
        return output_text

    sources = []
    for item in getattr(result.response, "output", []):
        if getattr(item, "type", None) != "web_search_call":
            continue
        action = getattr(item, "action", None)
        if not action:
            continue
        for source in getattr(action, "sources", []) or []:
            title = getattr(source, "title", "Source")
            url = getattr(source, "url", "")
            if url:
                sources.append(f"- [{title}]({url})")

    if sources:
        output_text = f"{output_text}\n\n**Sources**\n" + "\n".join(dict.fromkeys(sources))

    return output_text
