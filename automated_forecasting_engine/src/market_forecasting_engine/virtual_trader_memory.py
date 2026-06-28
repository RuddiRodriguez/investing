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

    def broker_snapshot(
        self,
        *,
        account: dict[str, Any] | None,
        positions: list[dict[str, Any]],
        orders: list[dict[str, Any]],
        recent_orders: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        lifecycle_events = self.reconcile_order_lifecycle(open_orders=orders, recent_orders=recent_orders or [])
        self.state["last_broker_snapshot"] = {
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "account": _compact_account(account or {}),
            "positions": [_compact_position(position) for position in positions],
            "open_orders": [_compact_order(order) for order in orders],
            "recent_orders": [_compact_order(order) for order in (recent_orders or [])],
            "order_lifecycle_events": lifecycle_events,
        }
        self.state["positions"] = {str(item.get("symbol") or "").upper(): item for item in self.state["last_broker_snapshot"]["positions"]}
        return lifecycle_events

    def reconcile_order_lifecycle(
        self,
        *,
        open_orders: list[dict[str, Any]],
        recent_orders: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        now = datetime.now(UTC).isoformat()
        existing = self.state.setdefault("order_lifecycle", {})
        previous_open = self.state.get("last_broker_snapshot", {}).get("open_orders", [])
        broker_rows = [_compact_order(row) for row in [*(recent_orders or []), *(open_orders or [])] if isinstance(row, dict)]
        by_order_id = {str(row.get("id")): row for row in broker_rows if row.get("id")}
        by_client_id = {str(row.get("client_order_id")): row for row in broker_rows if row.get("client_order_id")}
        previous_open_by_client_id = {str(row.get("client_order_id")): row for row in previous_open if row.get("client_order_id")}
        current_open_client_ids = {str(row.get("client_order_id")) for row in open_orders if isinstance(row, dict) and row.get("client_order_id")}
        current_open_order_ids = {str(row.get("id")) for row in open_orders if isinstance(row, dict) and row.get("id")}
        events: list[dict[str, Any]] = []
        for memory_order in self.state.get("orders", []):
            if not isinstance(memory_order, dict):
                continue
            order_result = memory_order.get("order_result") if isinstance(memory_order.get("order_result"), dict) else {}
            if not order_result.get("submitted"):
                continue
            broker_response = order_result.get("broker_response") if isinstance(order_result.get("broker_response"), dict) else {}
            order_payload = memory_order.get("order_payload") if isinstance(memory_order.get("order_payload"), dict) else {}
            broker_id = str(broker_response.get("id") or "").strip()
            client_id = str(broker_response.get("client_order_id") or order_payload.get("client_order_id") or "").strip()
            if not broker_id and not client_id:
                continue
            key = broker_id or client_id
            matched = by_order_id.get(broker_id) if broker_id else None
            if matched is None and client_id:
                matched = by_client_id.get(client_id)
            if matched is None and client_id in previous_open_by_client_id and client_id not in current_open_client_ids:
                previous = previous_open_by_client_id[client_id]
                matched = {**previous, "status": "not_open_missing_from_recent_orders"}
            if matched is None:
                continue
            status = _normalized_order_status(matched.get("status"))
            if broker_id and broker_id in current_open_order_ids:
                lifecycle_state = "open"
            elif client_id and client_id in current_open_client_ids:
                lifecycle_state = "open"
            else:
                lifecycle_state = _lifecycle_state_for_status(status)
            prior = existing.get(key, {}) if isinstance(existing.get(key), dict) else {}
            previous_status = prior.get("status")
            entry = {
                "broker_order_id": broker_id or matched.get("id"),
                "client_order_id": client_id or matched.get("client_order_id"),
                "symbol": str(matched.get("symbol") or memory_order.get("symbol") or memory_order.get("ticker") or "").upper(),
                "side": matched.get("side") or order_payload.get("side"),
                "type": matched.get("type") or matched.get("order_type") or order_payload.get("order_type"),
                "time_in_force": matched.get("time_in_force") or order_payload.get("time_in_force"),
                "limit_price": matched.get("limit_price") or order_payload.get("limit_price"),
                "notional": matched.get("notional") or order_payload.get("notional"),
                "qty": matched.get("qty") or order_payload.get("qty"),
                "filled_qty": matched.get("filled_qty"),
                "filled_avg_price": matched.get("filled_avg_price"),
                "status": status,
                "lifecycle_state": lifecycle_state,
                "created_at": matched.get("created_at") or broker_response.get("created_at"),
                "submitted_at": matched.get("submitted_at") or broker_response.get("submitted_at"),
                "updated_at": matched.get("updated_at") or broker_response.get("updated_at"),
                "expires_at": matched.get("expires_at") or broker_response.get("expires_at"),
                "expired_at": matched.get("expired_at") or broker_response.get("expired_at"),
                "filled_at": matched.get("filled_at") or broker_response.get("filled_at"),
                "canceled_at": matched.get("canceled_at") or broker_response.get("canceled_at"),
                "failed_at": matched.get("failed_at") or broker_response.get("failed_at"),
                "last_reconciled_at_utc": now,
                "source": "alpaca_recent_orders" if matched.get("status") != "not_open_missing_from_recent_orders" else "previous_open_order_absent_from_broker",
            }
            existing[key] = entry
            memory_order["order_lifecycle"] = entry
            if previous_status != status:
                event = {
                    "event": "order_lifecycle_changed",
                    "recorded_at_utc": now,
                    "broker_order_id": entry.get("broker_order_id"),
                    "client_order_id": entry.get("client_order_id"),
                    "symbol": entry.get("symbol"),
                    "previous_status": previous_status,
                    "status": status,
                    "lifecycle_state": lifecycle_state,
                    "reason": _lifecycle_event_reason(previous_status, status, lifecycle_state),
                    "order": entry,
                }
                events.append(event)
        self.state["order_lifecycle"] = existing
        if events:
            self.state.setdefault("order_lifecycle_events", []).extend(events)
            self.state["order_lifecycle_events"] = self.state["order_lifecycle_events"][-500:]
        self.state["recent_order_lifecycle_events"] = events
        return events

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
            "broker_recent_orders": snapshot.get("recent_orders", []),
            "order_lifecycle": self.state.get("order_lifecycle", {}),
            "recent_order_lifecycle_events": self.state.get("recent_order_lifecycle_events", []),
            "order_lifecycle_events": self.state.get("order_lifecycle_events", [])[-50:],
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
        "expires_at",
        "expired_at",
        "filled_at",
        "canceled_at",
        "failed_at",
        "symbol",
        "asset_class",
        "qty",
        "notional",
        "filled_qty",
        "filled_avg_price",
        "side",
        "type",
        "order_type",
        "time_in_force",
        "limit_price",
        "stop_price",
        "status",
    )
    return {key: order.get(key) for key in keys if key in order}


def _normalized_order_status(value: Any) -> str:
    status = str(value or "unknown").strip().lower()
    return status or "unknown"


def _lifecycle_state_for_status(status: str) -> str:
    if status in {"new", "accepted", "pending_new", "partially_filled", "held", "pending_replace", "accepted_for_bidding"}:
        return "open"
    if status == "filled":
        return "filled"
    if status in {"canceled", "expired", "rejected", "stopped", "suspended", "calculated", "done_for_day"}:
        return "terminal"
    if status == "not_open_missing_from_recent_orders":
        return "unknown_not_open"
    return "unknown"


def _lifecycle_event_reason(previous_status: Any, status: str, lifecycle_state: str) -> str:
    if status == "expired":
        return "broker_reported_order_expired"
    if status == "filled":
        return "broker_reported_order_filled"
    if status == "canceled":
        return "broker_reported_order_canceled"
    if status == "rejected":
        return "broker_reported_order_rejected"
    if previous_status is None:
        return f"broker_order_first_seen_{status}"
    if lifecycle_state == "unknown_not_open":
        return "order_absent_from_open_orders_and_recent_orders"
    return "broker_order_status_changed"
