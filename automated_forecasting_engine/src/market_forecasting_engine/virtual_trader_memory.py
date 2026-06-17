from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


MEMORY_VERSION = "virtual_trader_memory_v1"


@dataclass(frozen=True)
class VirtualTraderMemory:
    path: Path
    state: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "VirtualTraderMemory":
        memory_path = Path(path).expanduser()
        if memory_path.exists():
            state = json.loads(memory_path.read_text(encoding="utf-8"))
            if isinstance(state, dict):
                state.setdefault("version", MEMORY_VERSION)
                state.setdefault("created_at_utc", datetime.now(UTC).isoformat())
                state.setdefault("updated_at_utc", datetime.now(UTC).isoformat())
                state.setdefault("positions", {})
                state.setdefault("watch_plans", {})
                state.setdefault("decisions", [])
                state.setdefault("orders", [])
                state.setdefault("rejections", [])
                state.setdefault("notes", [])
                state.setdefault("market_intelligence", {})
                return cls(memory_path, state)
        now = datetime.now(UTC).isoformat()
        return cls(
            memory_path,
            {
                "version": MEMORY_VERSION,
                "created_at_utc": now,
                "updated_at_utc": now,
                "positions": {},
                "watch_plans": {},
                "decisions": [],
                "orders": [],
                "rejections": [],
                "notes": [],
                "market_intelligence": {},
            },
        )

    def save(self) -> None:
        self.state["updated_at_utc"] = datetime.now(UTC).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    def broker_snapshot(self, *, account: dict[str, Any] | None, positions: list[dict[str, Any]], orders: list[dict[str, Any]]) -> None:
        self.state["last_broker_snapshot"] = {
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "account": _compact_account(account or {}),
            "positions": [_compact_position(position) for position in positions],
            "open_orders": [_compact_order(order) for order in orders],
        }
        self.state["positions"] = {str(item.get("symbol") or "").upper(): item for item in self.state["last_broker_snapshot"]["positions"]}

    def record_decision(self, decision: dict[str, Any]) -> None:
        entry = {"recorded_at_utc": datetime.now(UTC).isoformat(), **decision}
        self.state.setdefault("decisions", []).append(entry)
        self.state["decisions"] = self.state["decisions"][-500:]
        ticker = str(decision.get("ticker") or "").upper()
        action = str(decision.get("action") or "").lower()
        final_advice = decision.get("final_advice", {}) if isinstance(decision.get("final_advice"), dict) else {}
        if ticker and action in {"hold", "buy"}:
            watch_plan = _watch_plan_from_decision(ticker, decision, final_advice)
            if watch_plan:
                self.state.setdefault("watch_plans", {})[ticker] = watch_plan
        if ticker and action in {"reject", "blocked", "no_trade"}:
            self.state.setdefault("rejections", []).append(
                {
                    "recorded_at_utc": entry["recorded_at_utc"],
                    "ticker": ticker,
                    "reason": decision.get("reason") or decision.get("block_reason"),
                    "decision": decision,
                }
            )
            self.state["rejections"] = self.state["rejections"][-300:]

    def record_order(self, order: dict[str, Any]) -> None:
        self.state.setdefault("orders", []).append({"recorded_at_utc": datetime.now(UTC).isoformat(), **order})
        self.state["orders"] = self.state["orders"][-500:]

    def record_cycle(self, cycle: dict[str, Any]) -> None:
        self.state.setdefault("cycles", []).append({"recorded_at_utc": datetime.now(UTC).isoformat(), **cycle})
        self.state["cycles"] = self.state["cycles"][-300:]

    def record_market_intelligence(self, intelligence: dict[str, Any]) -> None:
        self.state["market_intelligence"] = intelligence

    def record_active_plan(self, plan: dict[str, Any]) -> None:
        entry = {"recorded_at_utc": datetime.now(UTC).isoformat(), **plan}
        self.state["active_plan"] = entry
        self.state.setdefault("plan_history", []).append(entry)
        self.state["plan_history"] = self.state["plan_history"][-300:]

    def has_trading_history(self) -> bool:
        return bool(self.state.get("decisions") or self.state.get("orders") or self.state.get("positions") or self.state.get("watch_plans"))

    def context_for_ticker(self, ticker: str) -> dict[str, Any]:
        symbol = str(ticker or "").upper()
        decisions = [item for item in self.state.get("decisions", []) if str(item.get("ticker") or "").upper() == symbol]
        orders = [item for item in self.state.get("orders", []) if str(item.get("ticker") or item.get("symbol") or "").upper() == symbol]
        rejections = [item for item in self.state.get("rejections", []) if str(item.get("ticker") or "").upper() == symbol]
        return {
            "ticker": symbol,
            "position": self.state.get("positions", {}).get(symbol),
            "active_watch_plan": self.state.get("watch_plans", {}).get(symbol),
            "recent_decisions": decisions[-8:],
            "recent_orders": orders[-8:],
            "recent_rejections": rejections[-5:],
        }

    def portfolio_context(self) -> dict[str, Any]:
        snapshot = self.state.get("last_broker_snapshot", {}) if isinstance(self.state.get("last_broker_snapshot"), dict) else {}
        return {
            "memory_version": self.state.get("version"),
            "updated_at_utc": self.state.get("updated_at_utc"),
            "account": snapshot.get("account", {}),
            "positions": list(self.state.get("positions", {}).values()),
            "open_orders": snapshot.get("open_orders", []),
            "active_watch_plans": self.state.get("watch_plans", {}),
            "recent_decisions": self.state.get("decisions", [])[-20:],
            "recent_orders": self.state.get("orders", [])[-20:],
            "recent_rejections": self.state.get("rejections", [])[-20:],
            "recent_cycles": self.state.get("cycles", [])[-10:],
            "market_intelligence": self.state.get("market_intelligence", {}),
            "active_plan": self.state.get("active_plan", {}),
            "recent_plans": self.state.get("plan_history", [])[-10:],
        }


def memory_summary_for_prompt(memory: VirtualTraderMemory, ticker: str) -> str:
    context = memory.context_for_ticker(ticker)
    portfolio = memory.portfolio_context()
    return json.dumps(
        {
            "ticker_memory": context,
            "portfolio_memory": {
                "account": portfolio.get("account"),
                "positions": portfolio.get("positions"),
                "open_orders": portfolio.get("open_orders"),
                "active_watch_plan_count": len(portfolio.get("active_watch_plans", {}) or {}),
            },
            "instruction": (
                "Use this memory to avoid repeating stale recommendations, respect existing positions and open orders, "
                "and update prior watch plans when the new evidence changes the trade setup."
            ),
        },
        indent=2,
        sort_keys=True,
        default=str,
    )


def _watch_plan_from_decision(ticker: str, decision: dict[str, Any], final_advice: dict[str, Any]) -> dict[str, Any] | None:
    buy_lower = final_advice.get("buy_lower_price")
    zone_low = final_advice.get("buy_lower_zone_low")
    zone_high = final_advice.get("buy_lower_zone_high")
    breakout = final_advice.get("buy_above_breakout_price")
    sell_trim = final_advice.get("sell_or_trim_price")
    stop = final_advice.get("stop_loss_price")
    if not any(value is not None for value in (buy_lower, zone_low, zone_high, breakout, sell_trim, stop)):
        return None
    return {
        "ticker": ticker,
        "created_or_updated_at_utc": datetime.now(UTC).isoformat(),
        "source_run": decision.get("forecast_output_dir"),
        "action": decision.get("action"),
        "buy_lower_price": buy_lower,
        "buy_lower_zone_low": zone_low,
        "buy_lower_zone_high": zone_high,
        "buy_above_breakout_price": breakout,
        "sell_or_trim_price": sell_trim,
        "stop_loss_price": stop,
        "take_profit_price": final_advice.get("take_profit_price"),
        "headline": final_advice.get("headline"),
        "what_would_change": decision.get("change_triggers") or [],
    }


def _compact_account(account: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "status",
        "currency",
        "buying_power",
        "cash",
        "portfolio_value",
        "equity",
        "last_equity",
        "pattern_day_trader",
        "trading_blocked",
        "transfers_blocked",
        "account_blocked",
    )
    return {key: account.get(key) for key in keys if key in account}


def _compact_position(position: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "asset_id",
        "symbol",
        "exchange",
        "asset_class",
        "qty",
        "avg_entry_price",
        "side",
        "market_value",
        "cost_basis",
        "unrealized_pl",
        "unrealized_plpc",
        "current_price",
        "lastday_price",
        "change_today",
    )
    return {key: position.get(key) for key in keys if key in position}


def _compact_order(order: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "client_order_id",
        "created_at",
        "updated_at",
        "submitted_at",
        "symbol",
        "asset_class",
        "qty",
        "notional",
        "filled_qty",
        "side",
        "type",
        "time_in_force",
        "limit_price",
        "stop_price",
        "status",
    )
    return {key: order.get(key) for key in keys if key in order}
