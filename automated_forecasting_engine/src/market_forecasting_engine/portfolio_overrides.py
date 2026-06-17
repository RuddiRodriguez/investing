from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


WATCH_STATE_OVERRIDE_FILE = Path("automated_forecasting_engine/runs/watch_agent_state/portfolio_overrides.json")
TRADE_REPUBLIC_OVERRIDE_FILE = Path("automated_forecasting_engine/trade_republic_exports/portfolio_overrides.json")


def load_portfolio_overrides(*paths: Path) -> dict[str, Any]:
    for path in paths:
        if not path.exists():
            continue
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def sold_tickers(overrides: dict[str, Any]) -> set[str]:
    sold: set[str] = set()
    for ticker, value in _iter_override_items(overrides):
        if _is_sold_override(value):
            sold.add(_normalize_ticker(ticker))
    return sold


def is_sold_ticker(ticker: str | None, overrides: dict[str, Any]) -> bool:
    normalized = _normalize_ticker(ticker)
    return bool(normalized and normalized in sold_tickers(overrides))


def apply_open_position_overrides(report: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    sold = sold_tickers(overrides)
    if not sold:
        return report
    adjusted = deepcopy(report)
    holdings = adjusted.get("holdings") or []
    kept = [row for row in holdings if not is_sold_ticker(_holding_ticker(row), overrides)]
    removed = [row for row in holdings if is_sold_ticker(_holding_ticker(row), overrides)]
    adjusted["holdings"] = kept
    summary = adjusted.setdefault("summary", {})
    if isinstance(summary, dict):
        summary["holding_count"] = len(kept)
        for key in ("total_current_value", "total_open_cost_basis", "total_unrealized_pl"):
            if key in summary:
                summary[key] = _subtract(summary.get(key), sum(_number(row.get(_holding_total_key(key))) or 0.0 for row in removed))
        current_value = _number(summary.get("total_current_value"))
        open_cost = _number(summary.get("total_open_cost_basis"))
        if current_value is not None and open_cost not in (None, 0):
            summary["total_unrealized_pl_pct"] = ((current_value - open_cost) / open_cost) * 100.0
        summary["manual_position_overrides_applied"] = sorted(sold)
        summary["manual_position_override_note"] = (
            "Sold holdings are removed from open-position views. Sale proceeds are not estimated locally; "
            "the next successful broker refresh should replace this override."
        )
    adjusted["manual_position_overrides"] = {ticker: _override_for_ticker(overrides, ticker) for ticker in sorted(sold)}
    return adjusted


def default_override_paths(project_dir: Path | None = None) -> list[Path]:
    base = project_dir or Path.cwd()
    return [base / WATCH_STATE_OVERRIDE_FILE, base / TRADE_REPUBLIC_OVERRIDE_FILE]


def _iter_override_items(overrides: dict[str, Any]) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for section in ("sold", "overrides", "positions"):
        value = overrides.get(section)
        if isinstance(value, dict):
            items.extend((str(ticker), details) for ticker, details in value.items())
    return items


def _is_sold_override(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "sold"
    if not isinstance(value, dict):
        return False
    status = str(value.get("status") or value.get("holding_status") or value.get("state") or "").strip().lower()
    return status in {"sold", "closed", "removed"}


def _override_for_ticker(overrides: dict[str, Any], ticker: str) -> Any:
    normalized = _normalize_ticker(ticker)
    for candidate, details in _iter_override_items(overrides):
        if _normalize_ticker(candidate) == normalized:
            return details
    return None


def _holding_ticker(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    return str(row.get("ticker") or row.get("alpaca_ticker") or "").strip()


def _normalize_ticker(ticker: str | None) -> str:
    return str(ticker or "").strip().upper()


def _holding_total_key(summary_key: str) -> str:
    return {
        "total_current_value": "current_value",
        "total_open_cost_basis": "open_cost_basis",
        "total_unrealized_pl": "unrealized_pl",
    }[summary_key]


def _subtract(value: Any, amount: float) -> float | Any:
    number = _number(value)
    if number is None:
        return value
    return number - amount


def _number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
