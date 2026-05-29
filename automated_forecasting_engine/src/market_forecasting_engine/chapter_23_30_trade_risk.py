from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def analyze_chapter_23_30_trade_risk_plan(
    report: dict[str, Any],
    prices: pd.DataFrame | None = None,
    target_column: str = "close",
) -> dict[str, Any]:
    """Build a combined Chapters 23-30 trade/risk planning packet.

    These chapters are practical trade management, not a fresh forecast. The
    output converts the completed tactical and selection flow into explicit
    controls for risk, sizing, stops, pivots, trendlines, and support/resistance.
    """

    target = target_column.lower()
    action = str(report.get("suggested_action") or "Hold")
    current_price = _finite_or_none(report.get("current_price"))
    chapter_18 = report.get("decision_view", {}).get("chapter_18_tactical_problem", {})
    tactical_plan = chapter_18.get("trade_plan", {})
    chapter_19 = report.get("operations_view", {}).get("chapter_19_validation", {})
    chapter_20 = report.get("selection_view", {}).get("chapter_20_ticker_suitability", {})
    chapter_21 = report.get("selection_view", {}).get("chapter_21_chart_selection", {})
    technical = report.get("technical_view", {})
    habit = _habit_profile(report, prices=prices, target_column=target)

    high_risk_controls = _chapter_23_high_risk_controls(report, habit=habit, chapter_20=chapter_20)
    probable_moves = _chapter_24_probable_moves(report, habit=habit, prices=prices, target_column=target)
    margin_short_policy = _chapter_25_margin_short_policy(report, action=action, high_risk_controls=high_risk_controls)
    position_sizing = _chapter_26_position_sizing(
        report=report,
        action=action,
        current_price=current_price,
        tactical_plan=tactical_plan,
        high_risk_controls=high_risk_controls,
    )
    stop_order_plan = _chapter_27_stop_order_plan(
        action=action,
        current_price=current_price,
        tactical_plan=tactical_plan,
        probable_moves=probable_moves,
    )
    pivot_confirmation = _chapter_28_top_bottom_confirmation(technical)
    trendline_execution = _chapter_29_trendline_execution(
        action=action,
        technical=technical,
        stop_order_plan=stop_order_plan,
    )
    support_resistance_execution = _chapter_30_support_resistance_execution(
        action=action,
        current_price=current_price,
        technical=technical,
        stop_order_plan=stop_order_plan,
    )
    commitment = _commitment_type(
        action=action,
        chapter_19=chapter_19,
        chapter_21=chapter_21,
        high_risk_controls=high_risk_controls,
        margin_short_policy=margin_short_policy,
        stop_order_plan=stop_order_plan,
    )
    risk_notes = _risk_notes(
        commitment=commitment,
        high_risk_controls=high_risk_controls,
        margin_short_policy=margin_short_policy,
        position_sizing=position_sizing,
        stop_order_plan=stop_order_plan,
        pivot_confirmation=pivot_confirmation,
        trendline_execution=trendline_execution,
        support_resistance_execution=support_resistance_execution,
    )

    return {
        "principle": (
            "Chapters 23-30 convert a selected ticker into a controlled trade/risk plan: "
            "speculative controls, probable movement, leverage policy, position size, stops, pivot confirmation, "
            "trendline execution, and support/resistance execution."
        ),
        "state": commitment["state"],
        "status": commitment["status"],
        "decision_policy": {
            "mode": "trade_risk_plan_report_only",
            "influences_final_action": False,
            "intended_consumer": "human_or_llm_trade_risk_reviewer",
            "reason": "This layer plans execution and risk controls but does not override Chapter 18 action, Chapter 19 validation, or Chapter 21 selection.",
        },
        "commitment": commitment,
        "chapter_23_high_risk_controls": high_risk_controls,
        "chapter_24_probable_moves": probable_moves,
        "chapter_25_margin_short_policy": margin_short_policy,
        "chapter_26_position_sizing": position_sizing,
        "chapter_27_stop_order_plan": stop_order_plan,
        "chapter_28_top_bottom_confirmation": pivot_confirmation,
        "chapter_29_trendline_execution": trendline_execution,
        "chapter_30_support_resistance_execution": support_resistance_execution,
        "execution_summary": {
            "commitment_type": commitment["commitment_type"],
            "entry_plan": commitment["entry_plan"],
            "initial_stop": stop_order_plan.get("initial_stop"),
            "trailing_rule": stop_order_plan.get("trailing_rule"),
            "position_size_policy": position_sizing.get("position_size_policy"),
            "target_zone": support_resistance_execution.get("target_zone"),
            "risk_budget_pct": position_sizing.get("risk_budget_pct"),
        },
        "risk_notes": risk_notes,
        "llm_integration": {
            "status": "planned",
            "note": (
                "A second-version LLM trade/risk reviewer should consume this packet together with Chapters 18-21. "
                "It can explain, compare, and downgrade plans, but it must not override hard risk gates, stops, sizing limits, or operational validation."
            ),
        },
        "technical_method_card": chapter_23_30_trade_risk_method_card(target_column=target),
    }


def apply_chapter_23_30_trade_risk_plan(
    report: dict[str, Any],
    prices: pd.DataFrame | None = None,
    target_column: str = "close",
) -> dict[str, Any]:
    trade_risk = analyze_chapter_23_30_trade_risk_plan(
        report,
        prices=prices,
        target_column=target_column,
    )
    report.setdefault("trade_risk_view", {})["chapter_23_30_trade_risk_plan"] = trade_risk
    report.setdefault("technical_view", {})["chapter_23_30_trade_risk_plan"] = trade_risk
    report.setdefault("diagnostics", {})["chapter_23_30_trade_risk_plan"] = trade_risk
    report.setdefault("governance", {}).setdefault("trade_risk_method_cards", {})[
        "chapter_23_30_trade_risk_plan"
    ] = trade_risk["technical_method_card"]
    return trade_risk


def chapter_23_30_trade_risk_method_card(target_column: str = "close") -> dict[str, Any]:
    return {
        "name": "edwards_magee_chapters_23_30_trade_risk_plan",
        "version": "chapter_23_30_trade_risk_v1",
        "target_column": target_column.lower(),
        "decision_policy": "report_only_trade_risk_planning_no_action_override",
        "implemented_controls": [
            "chapter_23_high_risk_and_speculative_controls",
            "chapter_24_probable_move_profile",
            "chapter_25_margin_and_short_policy",
            "chapter_26_risk_budget_position_sizing",
            "chapter_27_protective_and_progressive_stop_plan",
            "chapter_28_confirmed_top_bottom_stop_movement",
            "chapter_29_trendline_execution_rules",
            "chapter_30_support_resistance_execution_rules",
        ],
        "chapter_alignment": [
            "do_not_treat_speculative_stocks_as_normal_positions",
            "estimate_probable_movement_before_sizing",
            "separate_margin_and_short_selling_from ordinary long exposure",
            "size_from_risk_budget_and_stop_distance",
            "always define protective and progressive stops",
            "move stops only after confirmed pivots",
            "use trendlines and support_resistance_for_execution",
        ],
        "future_llm_integration": (
            "A second-version LLM reviewer may synthesize this trade/risk packet with Chapters 18-21, "
            "but it must remain subordinate to hard operational, stop, and sizing gates."
        ),
    }


def _chapter_23_high_risk_controls(
    report: dict[str, Any],
    habit: dict[str, Any],
    chapter_20: dict[str, Any],
) -> dict[str, Any]:
    profile = chapter_20.get("profile_fit", {}).get("primary_profile")
    risk_level = str(report.get("risk_level") or "Unknown")
    vol = _finite_or_none(habit.get("realized_volatility_20d"))
    atr = _finite_or_none(habit.get("atr_pct_20d"))
    drawdown = abs(_finite_or_none(habit.get("max_drawdown_252d")) or 0.0)
    speculative = profile == "speculative_satellite" or risk_level == "High" or (vol is not None and vol >= 0.70) or drawdown >= 0.45
    mania = (vol is not None and vol >= 1.20) or (atr is not None and atr >= 0.08)
    size_multiplier = 0.50 if speculative else 1.0
    if mania:
        size_multiplier = 0.25
    return {
        "status": "SpeculativeControls" if speculative else "NormalControls",
        "speculative": bool(speculative),
        "mania_warning": bool(mania),
        "profile": profile,
        "risk_level": risk_level,
        "realized_volatility_20d": _round(vol),
        "atr_pct_20d": _round(atr),
        "max_drawdown_252d": _round(habit.get("max_drawdown_252d")),
        "position_size_multiplier": _round(size_multiplier),
        "policy": (
            "Treat as a speculative satellite with reduced size and no margin."
            if speculative
            else "Treat as a normal candidate subject to stops and Chapter 21 selection."
        ),
    }


def _chapter_24_probable_moves(
    report: dict[str, Any],
    habit: dict[str, Any],
    prices: pd.DataFrame | None,
    target_column: str,
) -> dict[str, Any]:
    current_price = _finite_or_none(report.get("current_price"))
    atr_pct = _finite_or_none(habit.get("atr_pct_20d"))
    realized_vol = _finite_or_none(habit.get("realized_volatility_20d"))
    if atr_pct is None and realized_vol is not None:
        atr_pct = realized_vol / np.sqrt(252)
    if atr_pct is None:
        atr_pct = _fallback_atr_pct(prices, target_column=target_column)
    day_move = atr_pct if atr_pct is not None else None
    five_day_move = day_move * np.sqrt(5) if day_move is not None else None
    thirty_day_move = day_move * np.sqrt(30) if day_move is not None else None
    return {
        "status": "Measured" if day_move is not None else "Unavailable",
        "basis": "20-session ATR percent or realized-volatility fallback",
        "one_day_normal_move_pct": _round(day_move),
        "five_day_normal_move_pct": _round(five_day_move),
        "thirty_day_normal_move_pct": _round(thirty_day_move),
        "one_day_price_band": _price_band(current_price, day_move),
        "five_day_price_band": _price_band(current_price, five_day_move),
        "thirty_day_price_band": _price_band(current_price, thirty_day_move),
        "interpretation": "Use these as normal movement bands for stop distance, entry patience, and target sanity checks.",
    }


def _chapter_25_margin_short_policy(
    report: dict[str, Any],
    action: str,
    high_risk_controls: dict[str, Any],
) -> dict[str, Any]:
    short_allowed = False
    margin_allowed = False
    sell_interpretation = "risk_reduction_or_exit_only"
    if action == "Sell":
        sell_interpretation = "short_selling_requires_explicit_permission"
    if high_risk_controls.get("speculative"):
        margin_reason = "Speculative/high-risk candidates should not use margin in this rule-based plan."
    else:
        margin_reason = "Margin is disabled by default because account-level leverage constraints were not supplied."
    return {
        "status": "Restricted",
        "margin_allowed": margin_allowed,
        "short_allowed": short_allowed,
        "sell_interpretation": sell_interpretation,
        "requires_user_profile": True,
        "policy": margin_reason,
        "warning": (
            "A Sell action is treated as exit/risk reduction unless a separate short-selling profile and borrow/liquidity checks are supplied."
            if action == "Sell"
            else "No margin is assumed for long entries."
        ),
    }


def _chapter_26_position_sizing(
    report: dict[str, Any],
    action: str,
    current_price: float | None,
    tactical_plan: dict[str, Any],
    high_risk_controls: dict[str, Any],
) -> dict[str, Any]:
    stop_plan = tactical_plan.get("stop_plan", {})
    stop = _finite_or_none(stop_plan.get("level"))
    stop_distance_pct = _finite_or_none(stop_plan.get("distance_pct") or tactical_plan.get("max_loss_pct"))
    base_risk_budget = {"Low": 0.010, "Medium": 0.0075, "High": 0.0035}.get(str(report.get("risk_level")), 0.005)
    risk_budget = base_risk_budget * float(high_risk_controls.get("position_size_multiplier") or 1.0)
    risk_per_share = abs(current_price - stop) if current_price is not None and stop is not None else None
    units_per_100k = None
    notional_per_100k = None
    if action in {"Buy", "Sell"} and risk_per_share is not None and risk_per_share > 0:
        units_per_100k = (100_000.0 * risk_budget) / risk_per_share
        notional_per_100k = units_per_100k * float(current_price or 0.0)
    policy = "No new position sizing while commitment is not directional."
    if action in {"Buy", "Sell"}:
        policy = "Size from account risk budget divided by per-share risk to the selected stop."
    return {
        "status": "Planned" if units_per_100k is not None else "FormulaOnly",
        "position_size_policy": policy,
        "risk_budget_pct": _round(risk_budget),
        "account_equity": None,
        "account_equity_status": "not_supplied",
        "initial_stop": _round(stop),
        "stop_distance_pct": _round(stop_distance_pct),
        "risk_per_share": _round(risk_per_share),
        "units_formula": "shares = account_equity * risk_budget_pct / abs(entry_price - stop_price)",
        "example_units_per_100k_account": _round(units_per_100k),
        "example_notional_per_100k_account": _round(notional_per_100k),
        "risk_budget_basis": "Risk budget is reduced for High risk or speculative/mania controls.",
    }


def _chapter_27_stop_order_plan(
    action: str,
    current_price: float | None,
    tactical_plan: dict[str, Any],
    probable_moves: dict[str, Any],
) -> dict[str, Any]:
    stop_plan = tactical_plan.get("stop_plan", {})
    initial_stop = _finite_or_none(stop_plan.get("level"))
    normal_move = _finite_or_none(probable_moves.get("one_day_normal_move_pct"))
    stop_distance = _finite_or_none(stop_plan.get("distance_pct") or tactical_plan.get("max_loss_pct"))
    too_tight = bool(normal_move is not None and stop_distance is not None and stop_distance < normal_move * 1.2)
    if action not in {"Buy", "Sell"}:
        return {
            "status": "NoActiveStop",
            "initial_stop": None,
            "protective_stop_type": "none",
            "trailing_rule": "No progressive stop while there is no new commitment.",
            "warning": "Keep watching existing chart levels, but no new stop order is planned.",
        }
    return {
        "status": "Planned" if initial_stop is not None else "Missing",
        "initial_stop": _round(initial_stop),
        "stop_source": stop_plan.get("source"),
        "protective_stop_type": "protective_stop_order_or_manual_alert",
        "stop_distance_pct": _round(stop_distance),
        "too_tight_for_normal_move": too_tight,
        "trailing_rule": (
            "For longs, raise the stop only after a confirmed higher low; for shorts, lower it only after a confirmed lower high."
        ),
        "progressive_stop_policy": "Use Chapter 28 pivot confirmation before moving stops; do not widen stops after entry.",
        "warning": "Stop may trigger before reversing; the rule prefers a controlled exit over unmanaged loss.",
    }


def _chapter_28_top_bottom_confirmation(technical: dict[str, Any]) -> dict[str, Any]:
    magee = technical.get("magee_basing_points", {}).get("preferred", {})
    return {
        "status": "Measured" if magee else "Unavailable",
        "trend_state": magee.get("trend_state"),
        "active_basing_stop": _round(magee.get("active_basing_stop")),
        "stop_status": magee.get("stop_status"),
        "confirmation_rule": (
            "Treat minor tops/bottoms as operational only after confirmation. "
            "Move long stops after confirmed higher lows and move short stops after confirmed lower highs."
        ),
        "policy": "Confirmed pivots control stop movement; unconfirmed lows/highs are watch points, not stop-ratchet points.",
    }


def _chapter_29_trendline_execution(
    action: str,
    technical: dict[str, Any],
    stop_order_plan: dict[str, Any],
) -> dict[str, Any]:
    chapter_14 = technical.get("chapter_14_trendlines", {})
    trendline = chapter_14.get("trendlines", {}).get("preferred", {})
    kind = trendline.get("kind")
    status = trendline.get("status")
    if action == "Buy":
        entry_rule = "Prefer pullbacks toward valid rising support or retests after upside breaks; avoid buying after decisive uptrend-line failure."
    elif action == "Sell":
        entry_rule = "Treat rallies toward valid falling resistance as risk-reduction or short-review zones; shorting still requires explicit permission."
    else:
        entry_rule = "No new trendline entry while final action is Hold; use trendline state for review cadence."
    return {
        "status": status or "Unavailable",
        "trendline_kind": kind,
        "effective_decisive_break": bool(trendline.get("effective_decisive_break")),
        "entry_rule": entry_rule,
        "exit_rule": "Exit or reduce if price decisively violates the controlling trendline and the stop plan agrees.",
        "stop_alignment": stop_order_plan.get("initial_stop"),
    }


def _chapter_30_support_resistance_execution(
    action: str,
    current_price: float | None,
    technical: dict[str, Any],
    stop_order_plan: dict[str, Any],
) -> dict[str, Any]:
    chapter_13 = technical.get("chapter_13_support_resistance", {})
    support = chapter_13.get("support_zones", {}).get("nearest", {})
    resistance = chapter_13.get("resistance_zones", {}).get("nearest", {})
    support_level = _first_finite(support.get("center"), support.get("lower"), support.get("upper"))
    resistance_level = _first_finite(resistance.get("center"), resistance.get("upper"), resistance.get("lower"))
    if action == "Buy":
        entry_zone = _zone(support.get("lower"), support.get("upper"), support_level)
        target_zone = _zone(resistance.get("lower"), resistance.get("upper"), resistance_level)
        rule = "Prefer entries near support or after resistance retest; avoid buying directly into nearby strong resistance."
    elif action == "Sell":
        entry_zone = _zone(resistance.get("lower"), resistance.get("upper"), resistance_level)
        target_zone = _zone(support.get("lower"), support.get("upper"), support_level)
        rule = "Prefer exits or short reviews near resistance; avoid pressing shorts into nearby strong support."
    else:
        entry_zone = None
        target_zone = _zone(resistance.get("lower"), resistance.get("upper"), resistance_level) or _zone(support.get("lower"), support.get("upper"), support_level)
        rule = "Use support/resistance for watchlist alerts while no new commitment is active."
    return {
        "status": "Measured" if support or resistance else "Unavailable",
        "nearest_support": support,
        "nearest_resistance": resistance,
        "entry_zone": entry_zone,
        "target_zone": target_zone,
        "invalidation_zone": stop_order_plan.get("initial_stop"),
        "execution_rule": rule,
        "current_price": _round(current_price),
    }


def _commitment_type(
    action: str,
    chapter_19: dict[str, Any],
    chapter_21: dict[str, Any],
    high_risk_controls: dict[str, Any],
    margin_short_policy: dict[str, Any],
    stop_order_plan: dict[str, Any],
) -> dict[str, Any]:
    chart_selection = chapter_21.get("chart_selection", {})
    bucket = chart_selection.get("chart_book_bucket")
    if chapter_19.get("status") == "fail" or chapter_19.get("action_gate", {}).get("hard_block_new_commitments"):
        return _commitment_payload("Blocked", "blocked", "no_new_commitment", "NoEntry", "Chapter 19 validation blocks new commitments.")
    if bucket in {"excluded", None}:
        return _commitment_payload("NoCommitment", "excluded", "no_new_commitment", "NoEntry", "Chapter 21 does not keep this ticker in the active chart book.")
    if action == "Buy" and bucket == "trade_candidates" and stop_order_plan.get("status") == "Planned":
        return _commitment_payload("Planned", "long_candidate", "candidate_long_commitment", "PlanLongEntry", "Long candidate is allowed only with the stop and sizing plan.")
    if action == "Sell" and bucket == "trade_candidates":
        if not margin_short_policy.get("short_allowed"):
            return _commitment_payload("Restricted", "risk_reduction_only", "risk_reduction_or_exit", "ReduceOrExitOnly", "Sell is treated as risk reduction because short selling is not enabled.")
        return _commitment_payload("Planned", "short_candidate", "candidate_short_commitment", "PlanShortEntry", "Short candidate is allowed only with explicit short permissions and stop plan.")
    if bucket == "active_review":
        return _commitment_payload("ReviewOnly", "active_review", "active_review_no_new_commitment", "WatchActively", "Ticker is in active chart review but not a fresh trade candidate.")
    if bucket == "watchlist":
        return _commitment_payload("WatchOnly", "watchlist", "watchlist_no_new_commitment", "KeepInWatchlist", "Ticker remains on watchlist only.")
    return _commitment_payload("MonitorOnly", "monitor_only", "monitor_only", "MonitorOnly", "Ticker is monitor-only until evidence improves.")


def _commitment_payload(state: str, status: str, commitment_type: str, entry_plan: str, reason: str) -> dict[str, Any]:
    return {
        "state": state,
        "status": status,
        "commitment_type": commitment_type,
        "entry_plan": entry_plan,
        "reason": reason,
    }


def _risk_notes(**sections: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    commitment = sections["commitment"]
    if commitment.get("commitment_type") in {"no_new_commitment", "active_review_no_new_commitment", "watchlist_no_new_commitment", "monitor_only"}:
        notes.append("No new capital commitment is planned by Chapters 23-30.")
    high_risk = sections["high_risk_controls"]
    if high_risk.get("speculative"):
        notes.append("High-risk/speculative controls reduce position size and prohibit margin by default.")
    margin = sections["margin_short_policy"]
    if not margin.get("margin_allowed"):
        notes.append("Margin is disabled unless an explicit account-level profile is supplied.")
    if not margin.get("short_allowed"):
        notes.append("Sell signals are treated as exit/risk-reduction, not automatic short entries.")
    sizing = sections["position_sizing"]
    if sizing.get("account_equity_status") == "not_supplied":
        notes.append("Position size is formula-based because account equity was not supplied.")
    stop = sections["stop_order_plan"]
    if stop.get("too_tight_for_normal_move"):
        notes.append("The selected stop may be tight relative to normal daily movement.")
    pivot = sections["pivot_confirmation"]
    if pivot.get("status") == "Measured":
        notes.append("Move stops only after confirmed pivot highs/lows; do not ratchet on unconfirmed noise.")
    return notes


def _habit_profile(report: dict[str, Any], prices: pd.DataFrame | None, target_column: str) -> dict[str, Any]:
    habit = (
        report.get("selection_view", {})
        .get("chapter_20_ticker_suitability", {})
        .get("instrument_habit_profile", {})
    )
    if habit and habit.get("status") == "measured":
        return habit
    return _price_habit_profile(prices, target_column=target_column)


def _price_habit_profile(prices: pd.DataFrame | None, target_column: str) -> dict[str, Any]:
    if prices is None or prices.empty:
        return {"status": "not_supplied"}
    frame = prices.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    if target_column not in frame:
        return {"status": "missing_target"}
    close = pd.to_numeric(frame[target_column], errors="coerce").dropna()
    returns = np.log(close.replace(0, np.nan)).diff().replace([np.inf, -np.inf], np.nan).dropna()
    realized = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 5 else None
    recent = close.tail(252)
    drawdown = recent / recent.cummax() - 1.0 if not recent.empty else pd.Series(dtype=float)
    return {
        "status": "measured",
        "realized_volatility_20d": _round(realized),
        "atr_pct_20d": _round(_fallback_atr_pct(prices, target_column=target_column)),
        "max_drawdown_252d": _round(float(drawdown.min())) if not drawdown.empty else None,
    }


def _fallback_atr_pct(prices: pd.DataFrame | None, target_column: str) -> float | None:
    if prices is None or prices.empty:
        return None
    frame = prices.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    if {"high", "low", target_column}.issubset(set(frame.columns)):
        high = pd.to_numeric(frame["high"], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce")
        close = pd.to_numeric(frame[target_column], errors="coerce")
        previous = close.shift(1)
        true_range = pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)
        latest = close.dropna().iloc[-1] if close.notna().any() else np.nan
        atr = true_range.tail(20).mean()
        if np.isfinite(atr) and np.isfinite(latest) and latest > 0:
            return float(atr / latest)
    if target_column in frame:
        close = pd.to_numeric(frame[target_column], errors="coerce").dropna()
        returns = close.pct_change().abs().dropna()
        if not returns.empty:
            return float(returns.tail(20).mean())
    return None


def _price_band(current_price: float | None, move_pct: float | None) -> dict[str, float | None]:
    if current_price is None or move_pct is None:
        return {"lower": None, "upper": None}
    return {
        "lower": _round(current_price * (1.0 - move_pct)),
        "upper": _round(current_price * (1.0 + move_pct)),
    }


def _zone(lower: Any, upper: Any, center: Any) -> dict[str, float | None] | None:
    center_value = _finite_or_none(center)
    lower_value = _finite_or_none(lower)
    upper_value = _finite_or_none(upper)
    if center_value is None and lower_value is None and upper_value is None:
        return None
    if center_value is None:
        finite = [value for value in (lower_value, upper_value) if value is not None]
        center_value = float(np.mean(finite)) if finite else None
    return {
        "lower": _round(lower_value if lower_value is not None else center_value),
        "center": _round(center_value),
        "upper": _round(upper_value if upper_value is not None else center_value),
    }


def _first_finite(*values: Any) -> float | None:
    for value in values:
        finite = _finite_or_none(value)
        if finite is not None:
            return finite
    return None


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _round(value: Any, digits: int = 4) -> float | None:
    numeric = _finite_or_none(value)
    if numeric is None:
        return None
    return round(float(numeric), digits)
