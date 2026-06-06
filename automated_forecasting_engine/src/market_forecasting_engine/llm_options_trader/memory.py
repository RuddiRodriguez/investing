from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def memory_path(output_dir: Path, currency: str) -> Path:
    return output_dir / "memory" / f"{currency.upper()}_llm_trader_memory.jsonl"


def strategy_memory_path(output_dir: Path, currency: str) -> Path:
    return output_dir / "memory" / f"{currency.upper()}_strategy_memory.json"


def load_recent_memory(output_dir: Path, currency: str, *, limit: int = 40) -> dict[str, Any]:
    path = memory_path(output_dir, currency)
    if not path.exists():
        return {"status": "empty", "events": [], "summary": "No prior LLM trader memory is available yet."}
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    recent = rows[-max(1, int(limit)) :]
    return {
        "status": "ok",
        "event_count": len(rows),
        "events_returned": len(recent),
        "summary": summarize_memory(recent),
        "events": recent,
    }


def load_strategy_memory(output_dir: Path, currency: str, *, max_lessons: int = 12) -> dict[str, Any]:
    path = strategy_memory_path(output_dir, currency)
    if not path.exists():
        return {
            "status": "empty",
            "lesson_count": 0,
            "summary": "No persistent strategy lessons have been learned yet.",
            "lessons": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "status": "unreadable",
            "lesson_count": 0,
            "summary": "Strategy memory exists but could not be parsed.",
            "lessons": [],
        }
    lessons = payload.get("lessons") if isinstance(payload.get("lessons"), list) else []
    lessons = sorted(
        lessons,
        key=lambda item: (float(item.get("confidence") or 0.0), int(item.get("occurrences") or 0), str(item.get("last_seen_utc") or "")),
        reverse=True,
    )[: max(1, int(max_lessons))]
    return {
        "status": "ok",
        "updated_at_utc": payload.get("updated_at_utc"),
        "lesson_count": len(payload.get("lessons") or []),
        "summary": summarize_strategy_memory(lessons),
        "lessons": lessons,
    }


def append_memory_event(output_dir: Path, currency: str, event: dict[str, Any]) -> None:
    path = memory_path(output_dir, currency)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"remembered_at_utc": datetime.now(UTC).isoformat(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def update_strategy_memory_from_record(output_dir: Path, currency: str, record: dict[str, Any], *, max_lessons: int = 24) -> dict[str, Any]:
    path = strategy_memory_path(output_dir, currency)
    path.parent.mkdir(parents=True, exist_ok=True)
    current: dict[str, Any] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    existing = current.get("lessons") if isinstance(current.get("lessons"), list) else []
    by_id = {str(item.get("id")): dict(item) for item in existing if item.get("id")}
    for lesson in infer_strategy_lessons(record):
        lesson_id = str(lesson["id"])
        prior = by_id.get(lesson_id, {})
        occurrences = int(prior.get("occurrences") or 0) + 1
        confidence = min(0.95, max(float(prior.get("confidence") or 0.0), float(lesson.get("confidence") or 0.0)) + min(0.10, occurrences * 0.01))
        by_id[lesson_id] = {
            **prior,
            **lesson,
            "occurrences": occurrences,
            "confidence": round(confidence, 4),
            "first_seen_utc": prior.get("first_seen_utc") or lesson.get("last_seen_utc"),
            "last_seen_utc": lesson.get("last_seen_utc"),
        }
    lessons = sorted(
        by_id.values(),
        key=lambda item: (float(item.get("confidence") or 0.0), int(item.get("occurrences") or 0), str(item.get("last_seen_utc") or "")),
        reverse=True,
    )[: max(1, int(max_lessons))]
    payload = {
        "status": "ok",
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "currency": currency.upper(),
        "lesson_count": len(lessons),
        "summary": summarize_strategy_memory(lessons),
        "lessons": lessons,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return payload


def compact_decision_event(*, process: str, record: dict[str, Any]) -> dict[str, Any]:
    decision = record.get("llm_decision") if isinstance(record.get("llm_decision"), dict) else {}
    order_result = record.get("order_result") if isinstance(record.get("order_result"), dict) else {}
    packet = record.get("market_packet") if isinstance(record.get("market_packet"), dict) else {}
    account = packet.get("account") if isinstance(packet.get("account"), dict) else {}
    positions = packet.get("option_positions") if isinstance(packet.get("option_positions"), list) else []
    orders = packet.get("open_option_orders") if isinstance(packet.get("open_option_orders"), list) else []
    shadow = packet.get("shadow_simulation") if isinstance(packet.get("shadow_simulation"), dict) else {}
    price_summary = packet.get("price_summary") if isinstance(packet.get("price_summary"), dict) else {}
    technical = packet.get("technical_observations") if isinstance(packet.get("technical_observations"), dict) else {}
    return {
        "process": process,
        "checked_at_utc": record.get("checked_at_utc"),
        "latest_underlying_price": packet.get("latest_underlying_price"),
        "account": {
            "currency": account.get("currency"),
            "equity": account.get("equity"),
            "available_funds": account.get("available_funds"),
            "options_session_rpl": account.get("options_session_rpl"),
            "options_session_upl": account.get("options_session_upl"),
        },
        "open_order_count": len(orders),
        "position_count": len(positions),
        "shadow": {
            "open_order_count": shadow.get("open_order_count"),
            "position_count": shadow.get("position_count"),
            "realized_pnl": shadow.get("realized_pnl"),
            "unrealized_pnl": shadow.get("unrealized_pnl"),
            "total_pnl": shadow.get("total_pnl"),
            "profit_protection_audit": shadow.get("profit_protection_audit"),
        },
        "market_context": {
            "price_summary": {
                "latest_close": price_summary.get("latest_close"),
                "return_1h": price_summary.get("return_1h"),
                "return_4h": price_summary.get("return_4h"),
                "realized_volatility": price_summary.get("realized_volatility"),
            },
            "technical_observations": technical,
        },
        "decision": {
            "action": decision.get("action"),
            "option_bias": decision.get("option_bias"),
            "confidence": decision.get("confidence"),
            "reason": decision.get("reason"),
            "risks": decision.get("risks"),
            "order": decision.get("order"),
            "order_id": decision.get("order_id"),
        },
        "order_result": {
            "submitted": order_result.get("submitted"),
            "reason": order_result.get("reason"),
            "blocks": order_result.get("blocks"),
            "validated_order": order_result.get("validated_order"),
        },
    }


def infer_strategy_lessons(record: dict[str, Any]) -> list[dict[str, Any]]:
    packet = record.get("market_packet") if isinstance(record.get("market_packet"), dict) else {}
    shadow = packet.get("shadow_simulation") if isinstance(packet.get("shadow_simulation"), dict) else {}
    decision = record.get("llm_decision") if isinstance(record.get("llm_decision"), dict) else {}
    order_result = record.get("order_result") if isinstance(record.get("order_result"), dict) else {}
    technical = packet.get("technical_observations") if isinstance(packet.get("technical_observations"), dict) else {}
    price_summary = packet.get("price_summary") if isinstance(packet.get("price_summary"), dict) else {}
    now = str(record.get("checked_at_utc") or datetime.now(UTC).isoformat())
    lessons: list[dict[str, Any]] = []

    for position in shadow_positions(shadow):
        realized = _as_float(position.get("realized_pnl"))
        unrealized = _as_float(position.get("unrealized_pnl"))
        instrument = str(position.get("instrument_name") or "")
        option_side = "put" if instrument.endswith("-P") else "call" if instrument.endswith("-C") else "option"
        if realized is not None and realized < -0.01:
            lessons.append(
                _lesson(
                    lesson_id=f"closed_loss_{option_side}",
                    lesson=f"Recent simulated {option_side} positions closed at a loss. Require fresher confirmation before repeating that side.",
                    evidence=f"{instrument} realized P/L {realized:.4f}; LLM reason: {decision.get('reason')}",
                    confidence=0.58,
                    tags=["loss", option_side, "entry_filter"],
                    now=now,
                    record=record,
                    price_summary=price_summary,
                    technical=technical,
                )
            )
        if unrealized is not None and unrealized < -0.01 and decision.get("action") == "hold":
            lessons.append(
                _lesson(
                    lesson_id=f"held_losing_{option_side}",
                    lesson=f"When an open simulated {option_side} is losing, the exit review must explicitly justify hold versus close.",
                    evidence=f"{instrument} unrealized P/L {unrealized:.4f}; decision held.",
                    confidence=0.54,
                    tags=["risk", option_side, "exit_management"],
                    now=now,
                    record=record,
                    price_summary=price_summary,
                    technical=technical,
                )
            )

    audit = shadow.get("profit_protection_audit") if isinstance(shadow.get("profit_protection_audit"), dict) else {}
    if int(audit.get("prior_profit_turned_loss_count") or 0) > 0:
        lessons.append(
            _lesson(
                lesson_id="profit_turned_loss_guard",
                lesson="Do not let a previously profitable shadow option become a losing trade; tighten exit once open P/L has been positive.",
                evidence=f"Profit audit: {audit}",
                confidence=0.72,
                tags=["profit_protection", "exit_management"],
                now=now,
                record=record,
                price_summary=price_summary,
                technical=technical,
            )
        )
    latest_close = audit.get("latest_close") if isinstance(audit.get("latest_close"), dict) else {}
    if latest_close and str(latest_close.get("side") or "").lower() == "sell":
        lessons.append(
            _lesson(
                lesson_id="recent_close_wait_for_new_setup",
                lesson="After a simulated close, wait for a fresh setup instead of immediately reopening the same stale idea.",
                evidence=f"Latest simulated close: {latest_close}",
                confidence=0.50,
                tags=["cooldown", "reentry"],
                now=now,
                record=record,
                price_summary=price_summary,
                technical=technical,
            )
        )
    if order_result.get("blocks"):
        lessons.append(
            _lesson(
                lesson_id="format_or_execution_blocks",
                lesson="When an order is blocked by format or execution constraints, produce a simpler valid limit order or hold.",
                evidence=f"Blocks: {order_result.get('blocks')}",
                confidence=0.62,
                tags=["execution", "format"],
                now=now,
                record=record,
                price_summary=price_summary,
                technical=technical,
            )
        )
    return lessons


def shadow_positions(shadow: dict[str, Any]) -> list[dict[str, Any]]:
    positions = shadow.get("positions")
    if isinstance(positions, list):
        return [item for item in positions if isinstance(item, dict)]
    if isinstance(positions, dict):
        return [item for item in positions.values() if isinstance(item, dict)]
    return []


def summarize_memory(events: list[dict[str, Any]]) -> str:
    if not events:
        return "No prior LLM trader memory is available yet."
    submitted = sum(1 for event in events if ((event.get("order_result") or {}).get("submitted") is True))
    blocked = sum(1 for event in events if (event.get("order_result") or {}).get("blocks"))
    holds = sum(1 for event in events if ((event.get("decision") or {}).get("action") == "hold"))
    recent_reasons = [
        str((event.get("decision") or {}).get("reason") or "")[:140]
        for event in events[-5:]
        if (event.get("decision") or {}).get("reason")
    ]
    return (
        f"Recent memory events: {len(events)}. Submitted orders: {submitted}. Holds: {holds}. "
        f"Format/API blocks: {blocked}. Recent reasons: {recent_reasons}."
    )


def summarize_strategy_memory(lessons: list[dict[str, Any]]) -> str:
    if not lessons:
        return "No persistent strategy lessons have been learned yet."
    top = [
        f"{item.get('id')}: {item.get('lesson')} (confidence {item.get('confidence')}, seen {item.get('occurrences')})"
        for item in lessons[:5]
    ]
    return "Persistent lessons: " + " | ".join(top)


def _lesson(
    *,
    lesson_id: str,
    lesson: str,
    evidence: str,
    confidence: float,
    tags: list[str],
    now: str,
    record: dict[str, Any],
    price_summary: dict[str, Any],
    technical: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": lesson_id,
        "lesson": lesson,
        "evidence": evidence,
        "confidence": confidence,
        "tags": tags,
        "last_seen_utc": now,
        "last_decision": {
            "action": ((record.get("llm_decision") or {}) if isinstance(record.get("llm_decision"), dict) else {}).get("action"),
            "intent": ((record.get("llm_decision") or {}) if isinstance(record.get("llm_decision"), dict) else {}).get("intent"),
            "reason": ((record.get("llm_decision") or {}) if isinstance(record.get("llm_decision"), dict) else {}).get("reason"),
        },
        "market_snapshot": {
            "latest_close": price_summary.get("latest_close"),
            "return_1h": price_summary.get("return_1h"),
            "return_4h": price_summary.get("return_4h"),
            "technical_observations": technical,
        },
    }


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None
