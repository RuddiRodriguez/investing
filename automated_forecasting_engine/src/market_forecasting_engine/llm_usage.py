from __future__ import annotations

import inspect
import json
import os
import hashlib
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


DEFAULT_OPENAI_PRICE_PER_1M = {
    "gpt-5.5": {"input_per_1m": 5.00, "cached_input_per_1m": 0.50, "output_per_1m": 30.00},
    "gpt-5.5-pro": {"input_per_1m": 30.00, "output_per_1m": 180.00},
    "gpt-5.4": {"input_per_1m": 2.50, "cached_input_per_1m": 0.25, "output_per_1m": 15.00},
    "gpt-5.4-mini": {"input_per_1m": 0.75, "cached_input_per_1m": 0.075, "output_per_1m": 4.50},
    "gpt-5.4-nano": {"input_per_1m": 0.20, "cached_input_per_1m": 0.02, "output_per_1m": 1.25},
    "gpt-5.4-pro": {"input_per_1m": 30.00, "output_per_1m": 180.00},
    "gpt-4o-mini": {"input_per_1m": 0.15, "cached_input_per_1m": 0.075, "output_per_1m": 0.60},
    "gpt-4o": {"input_per_1m": 2.50, "cached_input_per_1m": 1.25, "output_per_1m": 10.00},
    "text-embedding-3-small": {"input_per_1m": 0.02, "output_per_1m": 0.00},
    "text-embedding-3-large": {"input_per_1m": 0.13, "output_per_1m": 0.00},
    "text-embedding-ada-002": {"input_per_1m": 0.10, "output_per_1m": 0.00},
}
DEFAULT_OPENAI_TOOL_PRICE_PER_1K_CALLS = {
    "web_search": 10.00,
    "web_search_preview_reasoning": 10.00,
    "web_search_preview_non_reasoning": 25.00,
    "file_search": 2.50,
}
PRICING_SOURCE_URL = "https://developers.openai.com/api/docs/pricing"


def new_llm_call_id() -> str:
    return uuid4().hex


def usage_log_path(branch_name: str | None = None, process_name: str | None = None) -> Path:
    configured = os.getenv("OPENAI_USAGE_LOG_FILE") or os.getenv("LLM_USAGE_LOG_FILE")
    if configured:
        return Path(configured)
    today = datetime.now(UTC).strftime("%Y%m%d")
    branch = _safe_label(branch_name or _usage_branch_name())
    process = _safe_label(process_name or _usage_process_name({}))
    return Path("automated_forecasting_engine/runs/openai_usage") / branch / f"openai_usage_{branch}_{process}_{today}.jsonl"


def monotonic_ms() -> float:
    return time.perf_counter() * 1000.0


def log_openai_usage(
    *,
    call_id: str,
    model: str,
    payload: dict[str, Any],
    response_data: dict[str, Any] | None = None,
    started_ms: float | None = None,
    status: str = "ok",
    error: str | None = None,
    context: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> None:
    routing = _usage_routing(context or {})
    path = usage_log_path(branch_name=routing["branch"], process_name=routing["process"])
    usage = _extract_usage(response_data or {})
    request = _request_summary(payload)
    cost = _estimated_cost_usd(model, usage, request)
    record = {
        "logged_at_utc": datetime.now(UTC).isoformat(),
        "call_id": call_id,
        "status": status,
        "model": model,
        "response_id": (response_data or {}).get("id"),
        "routing": routing,
        "api_key": _api_key_report(api_key or os.getenv("OPENAI_API_KEY")),
        "usage": usage,
        "estimated_cost_usd": cost["estimated_cost_usd"],
        "cost_breakdown": cost,
        "pricing_source": PRICING_SOURCE_URL,
        "request": request,
        "duration_ms": round(monotonic_ms() - started_ms, 2) if started_ms is not None else None,
        "caller": _caller_context(),
        "context": context or {},
    }
    if error:
        record["error"] = str(error)[:500]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    except Exception:
        # Usage logging must never break a trading or LLM decision path.
        return


def log_openai_embedding_usage(
    *,
    call_id: str,
    model: str,
    payload: dict[str, Any],
    response_data: dict[str, Any] | None = None,
    started_ms: float | None = None,
    status: str = "ok",
    error: str | None = None,
    context: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> None:
    routing = _usage_routing(context or {})
    path = usage_log_path(branch_name=routing["branch"], process_name=routing["process"])
    usage = _extract_embedding_usage(response_data or {})
    request = _embedding_request_summary(payload)
    cost = _estimated_cost_usd(model, usage, request)
    record = {
        "logged_at_utc": datetime.now(UTC).isoformat(),
        "call_id": call_id,
        "status": status,
        "model": model,
        "response_id": (response_data or {}).get("id"),
        "routing": routing,
        "api_key": _api_key_report(api_key or os.getenv("OPENAI_API_KEY")),
        "usage": usage,
        "estimated_cost_usd": cost["estimated_cost_usd"],
        "cost_breakdown": cost,
        "pricing_source": PRICING_SOURCE_URL,
        "request": request,
        "duration_ms": round(monotonic_ms() - started_ms, 2) if started_ms is not None else None,
        "caller": _caller_context(),
        "context": context or {},
    }
    if error:
        record["error"] = str(error)[:500]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    except Exception:
        return


def _extract_usage(response_data: dict[str, Any]) -> dict[str, Any]:
    usage = response_data.get("usage") or {}
    if not isinstance(usage, dict):
        return {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "input_tokens_details": usage.get("input_tokens_details"),
        "output_tokens_details": usage.get("output_tokens_details"),
        "cached_input_tokens": _nested_number(usage, ("input_tokens_details", "cached_tokens")),
    }


def _extract_embedding_usage(response_data: dict[str, Any]) -> dict[str, Any]:
    usage = response_data.get("usage") or {}
    if not isinstance(usage, dict):
        return {}
    prompt_tokens = usage.get("prompt_tokens")
    total_tokens = usage.get("total_tokens")
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": 0,
        "total_tokens": total_tokens if total_tokens is not None else prompt_tokens,
        "prompt_tokens": prompt_tokens,
    }


def _request_summary(payload: dict[str, Any]) -> dict[str, Any]:
    tools = payload.get("tools") or []
    text = payload.get("text") or {}
    return {
        "api": "responses",
        "tool_count": len(tools) if isinstance(tools, list) else 0,
        "tools": [tool.get("type") for tool in tools if isinstance(tool, dict)],
        "text_format": (text.get("format") or {}).get("name") if isinstance(text, dict) else None,
        "has_reasoning": "reasoning" in payload,
        "store": payload.get("store"),
    }


def _embedding_request_summary(payload: dict[str, Any]) -> dict[str, Any]:
    inputs = payload.get("input") or []
    return {
        "api": "embeddings",
        "input_count": len(inputs) if isinstance(inputs, list) else 1,
        "dimensions": payload.get("dimensions"),
        "encoding_format": payload.get("encoding_format", "float"),
        "store": False,
        "tool_count": 0,
        "tools": [],
    }


def _estimated_cost_usd(model: str, usage: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    rates = _price_rates()
    model_rates = rates.get(model) or rates.get(_base_model_name(model))
    if not model_rates:
        return {
            "estimated_cost_usd": None,
            "reason": "no_price_for_model",
            "model_price_key": _base_model_name(model),
        }
    try:
        input_tokens = float(usage.get("input_tokens") or 0.0)
        cached_input_tokens = float(usage.get("cached_input_tokens") or 0.0)
        output_tokens = float(usage.get("output_tokens") or 0.0)
        input_rate = float(model_rates.get("input_per_1m") or model_rates.get("input") or 0.0)
        cached_input_rate = float(model_rates.get("cached_input_per_1m") or model_rates.get("cached_input") or input_rate)
        output_rate = float(model_rates.get("output_per_1m") or model_rates.get("output") or 0.0)
    except (TypeError, ValueError):
        return {"estimated_cost_usd": None, "reason": "invalid_usage_or_price"}
    uncached_input_tokens = max(0.0, input_tokens - cached_input_tokens)
    input_cost = uncached_input_tokens / 1_000_000.0 * input_rate
    cached_input_cost = cached_input_tokens / 1_000_000.0 * cached_input_rate
    output_cost = output_tokens / 1_000_000.0 * output_rate
    tool_cost = _tool_cost_usd(request)
    total = input_cost + cached_input_cost + output_cost + tool_cost
    return {
        "estimated_cost_usd": round(total, 8),
        "input_cost_usd": round(input_cost, 8),
        "cached_input_cost_usd": round(cached_input_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "tool_cost_usd": round(tool_cost, 8),
        "input_rate_per_1m": input_rate,
        "cached_input_rate_per_1m": cached_input_rate,
        "output_rate_per_1m": output_rate,
        "model_price_key": model if model in rates else _base_model_name(model),
        "tool_price_per_1k_calls": DEFAULT_OPENAI_TOOL_PRICE_PER_1K_CALLS,
    }


def _price_rates() -> dict[str, Any]:
    raw = os.getenv("OPENAI_USAGE_PRICE_PER_1M_JSON") or os.getenv("LLM_USAGE_PRICE_PER_1M_JSON")
    if not raw:
        return DEFAULT_OPENAI_PRICE_PER_1M
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return DEFAULT_OPENAI_PRICE_PER_1M
    if not isinstance(parsed, dict):
        return DEFAULT_OPENAI_PRICE_PER_1M
    return {**DEFAULT_OPENAI_PRICE_PER_1M, **parsed}


def _tool_cost_usd(request: dict[str, Any]) -> float:
    tools = request.get("tools") or []
    total = 0.0
    for tool in tools:
        key = str(tool or "")
        price = DEFAULT_OPENAI_TOOL_PRICE_PER_1K_CALLS.get(key)
        if price is not None:
            total += price / 1_000.0
    return total


def _nested_number(data: dict[str, Any], keys: tuple[str, ...]) -> int | float | None:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if isinstance(current, (int, float)) else None


def _api_key_report(api_key: str | None) -> dict[str, Any]:
    if not api_key:
        return {"configured": False}
    clean = str(api_key).strip().strip('"').strip("'")
    return {
        "configured": True,
        "masked": _mask_secret(clean),
        "sha256_12": hashlib.sha256(clean.encode("utf-8")).hexdigest()[:12],
        "prefix": clean.split("-", 2)[0] if "-" in clean else clean[:2],
    }


def _mask_secret(value: str) -> str:
    if len(value) <= 12:
        return "***"
    return f"{value[:7]}...{value[-4:]}"


def _base_model_name(model: str) -> str:
    parts = str(model or "").split("-")
    if len(parts) >= 4 and all(part.isdigit() for part in parts[-3:]):
        return "-".join(parts[:-3])
    return str(model or "")


def _usage_routing(context: dict[str, Any]) -> dict[str, str]:
    return {
        "branch": _usage_branch_name(),
        "process": _usage_process_name(context),
    }


def _usage_branch_name() -> str:
    configured = os.getenv("OPENAI_USAGE_BRANCH") or os.getenv("LLM_USAGE_BRANCH")
    if configured:
        return configured
    try:
        completed = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=Path.cwd(),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return "unknown_branch"
    branch = completed.stdout.strip()
    return branch or "unknown_branch"


def _usage_process_name(context: dict[str, Any]) -> str:
    configured = os.getenv("OPENAI_USAGE_PROCESS_NAME") or os.getenv("LLM_USAGE_PROCESS_NAME")
    if configured:
        return configured
    explicit = context.get("process") or context.get("process_name")
    if explicit:
        return str(explicit)
    purpose = str(context.get("purpose") or "")
    if purpose.startswith("daily_trade"):
        return "daily_trade"
    if purpose.startswith("alternative_news"):
        return "alternative_data"
    if purpose.startswith("chapter_18"):
        return "pipeline_tactical_review"
    if purpose in {"autonomous_trader_decision", "nontechnical_trader_summary"}:
        return "autonomous_trader"
    if purpose:
        return purpose
    return "default"


def _safe_label(value: str) -> str:
    label = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value or "").strip())
    label = label.strip("._-")
    return label or "unknown"


def _caller_context() -> dict[str, Any]:
    for frame in inspect.stack()[2:12]:
        module = frame.frame.f_globals.get("__name__", "")
        if module in {__name__, "market_forecasting_engine.openai_responses", "market_forecasting_engine.llm_trader.responses_api"}:
            continue
        return {"module": module, "function": frame.function, "file": frame.filename, "line": frame.lineno}
    return {}
