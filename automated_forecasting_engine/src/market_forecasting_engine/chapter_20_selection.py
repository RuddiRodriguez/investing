from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PROFILE_WEIGHTS = {
    "short_term_trader": {
        "operational": 0.15,
        "validation": 0.15,
        "technical": 0.15,
        "liquidity": 0.20,
        "movement": 0.25,
        "risk": 0.10,
    },
    "intermediate_trader": {
        "operational": 0.15,
        "validation": 0.20,
        "technical": 0.20,
        "liquidity": 0.15,
        "movement": 0.15,
        "risk": 0.15,
    },
    "long_term_investor": {
        "operational": 0.20,
        "validation": 0.15,
        "technical": 0.25,
        "liquidity": 0.10,
        "movement": 0.20,
        "risk": 0.10,
    },
    "speculative_satellite": {
        "operational": 0.10,
        "validation": 0.10,
        "technical": 0.10,
        "liquidity": 0.20,
        "movement": 0.35,
        "risk": 0.15,
    },
    "index_or_diversifier": {
        "operational": 0.20,
        "validation": 0.10,
        "technical": 0.20,
        "liquidity": 0.15,
        "movement": 0.20,
        "risk": 0.15,
    },
}


def analyze_chapter_20_ticker_suitability(
    report: dict[str, Any],
    prices: pd.DataFrame | None = None,
    target_column: str = "close",
) -> dict[str, Any]:
    """Score whether a completed ticker report is worth selecting for review.

    Chapter 20 is not a new forecast and not a trade override. It classifies the
    type of instrument we are looking at after the Chapter 18 tactical plan and
    Chapter 19 operational validation are available.
    """

    target = target_column.lower()
    instrument = _instrument_habit_profile(prices, target_column=target)
    operational = _operational_fit(report)
    validation = _validation_fit(report)
    technical = _technical_fit(report)
    liquidity = _liquidity_fit(instrument)
    movement = _movement_fit(instrument)
    risk = _risk_fit(report, instrument)
    tactical = _tactical_fit(report)
    asset = _asset_context(report)
    component_scores = {
        "operational": operational,
        "validation": validation,
        "technical": technical,
        "liquidity": liquidity,
        "movement": movement,
        "risk": risk,
        "tactical_readiness": tactical,
    }
    profile_scores = _profile_scores(
        component_scores=component_scores,
        movement_scores=movement,
        asset_context=asset,
    )
    primary_profile = max(profile_scores, key=lambda key: float(profile_scores[key]["score"]))
    best_score = float(profile_scores[primary_profile]["score"])
    classification = _classification(
        report=report,
        primary_profile=primary_profile,
        best_score=best_score,
        operational_score=float(operational["score"]),
        tactical_score=float(tactical["score"]),
    )

    return {
        "principle": (
            "Chapter 20 asks whether this is the kind of instrument worth following or trading for a given profile. "
            "It is a suitability and selection layer above the single-ticker forecast, not a stronger Buy/Sell signal."
        ),
        "state": classification["state"],
        "status": classification["status"],
        "decision_policy": {
            "mode": "selection_suitability_report_only",
            "influences_final_action": False,
            "intended_consumer": "chapter_21_chart_selection_and_human_or_llm_reviewer",
            "reason": "Chapter 20 ranks ticker suitability but does not override Chapter 18 tactical action or Chapter 19 validation.",
        },
        "profile_fit": {
            "primary_profile": primary_profile,
            "classification": classification["classification"],
            "suitability_score": _round(best_score),
            "selection_hint": classification["selection_hint"],
            "trade_candidate_eligible": bool(classification["trade_candidate_eligible"]),
            "active_review_eligible": bool(classification["active_review_eligible"]),
            "watchlist_eligible": bool(classification["watchlist_eligible"]),
            "reason": classification["reason"],
        },
        "profile_scores": profile_scores,
        "component_scores": component_scores,
        "instrument_habit_profile": instrument,
        "asset_context": asset,
        "rule_interpretation": _rule_interpretation(
            report=report,
            primary_profile=primary_profile,
            classification=classification,
            component_scores=component_scores,
        ),
        "chapter_21_readiness": {
            "status": classification["selection_hint"],
            "message": _chapter_21_message(classification["selection_hint"]),
        },
        "chapter_22_diversification_readiness": {
            "status": "requires_universe_context",
            "message": "Single-ticker suitability is ready; sector, correlation, and concentration checks require multiple reports.",
        },
        "llm_integration": {
            "status": "planned",
            "note": (
                "This rule-based suitability layer is designed to be combined later with an LLM reviewer. "
                "The LLM should explain and rank completed reports, but it should not bypass Chapter 18 or Chapter 19 gates."
            ),
        },
        "technical_method_card": chapter_20_ticker_suitability_method_card(target_column=target),
    }


def apply_chapter_20_ticker_suitability(
    report: dict[str, Any],
    prices: pd.DataFrame | None = None,
    target_column: str = "close",
) -> dict[str, Any]:
    suitability = analyze_chapter_20_ticker_suitability(
        report,
        prices=prices,
        target_column=target_column,
    )
    report.setdefault("selection_view", {})["chapter_20_ticker_suitability"] = suitability
    report.setdefault("technical_view", {})["chapter_20_ticker_suitability"] = suitability
    report.setdefault("diagnostics", {})["chapter_20_ticker_suitability"] = suitability
    report.setdefault("governance", {}).setdefault("selection_method_cards", {})[
        "chapter_20_ticker_suitability"
    ] = suitability["technical_method_card"]
    return suitability


def chapter_20_ticker_suitability_method_card(target_column: str = "close") -> dict[str, Any]:
    return {
        "name": "edwards_magee_chapter_20_ticker_suitability",
        "version": "chapter_20_selection_profile_v1",
        "target_column": target_column.lower(),
        "decision_policy": "report_only_selection_suitability_no_trade_override",
        "implemented_controls": [
            "operational_readiness_from_chapter_19",
            "forecast_validation_fit",
            "technical_clarity_and_conflict_fit",
            "liquidity_and_dollar_volume_fit",
            "movement_personality_profile",
            "risk_fit",
            "tactical_readiness_from_chapter_18",
            "profile_scores_for_short_term_intermediate_long_term_speculative_and_index_roles",
        ],
        "chapter_20_alignment": [
            "separate_kind_of_instrument_from_trade_timing",
            "match_instrument_to_speculator_or_investor_profile",
            "prefer_liquid_and_technically_readable_instruments",
            "treat_speculative_names_as_special_satellites",
            "prepare_selection_inputs_for_chapters_21_and_22",
        ],
        "future_llm_integration": (
            "A later LLM layer may review the rule-based suitability packet and rank multiple completed reports, "
            "but it must remain subordinate to Chapter 18 tactical gates and Chapter 19 validation."
        ),
    }


def _profile_scores(
    component_scores: dict[str, dict[str, Any]],
    movement_scores: dict[str, Any],
    asset_context: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    for profile, weights in PROFILE_WEIGHTS.items():
        movement_key = {
            "short_term_trader": "short_term_score",
            "intermediate_trader": "intermediate_score",
            "long_term_investor": "long_term_score",
            "speculative_satellite": "speculative_score",
            "index_or_diversifier": "diversifier_score",
        }[profile]
        values = {
            "operational": float(component_scores["operational"]["score"]),
            "validation": float(component_scores["validation"]["score"]),
            "technical": float(component_scores["technical"]["score"]),
            "liquidity": float(component_scores["liquidity"]["score"]),
            "movement": float(movement_scores.get(movement_key, 0.5)),
            "risk": float(component_scores["risk"]["score"]),
        }
        score = sum(values[name] * weight for name, weight in weights.items())
        if profile == "index_or_diversifier" and asset_context.get("is_diversified_vehicle"):
            score = min(1.0, score + 0.08)
        scores[profile] = {
            "score": _round(score),
            "inputs": {name: _round(value) for name, value in values.items()},
            "weights": weights,
        }
    return scores


def _classification(
    report: dict[str, Any],
    primary_profile: str,
    best_score: float,
    operational_score: float,
    tactical_score: float,
) -> dict[str, Any]:
    action = str(report.get("suggested_action") or "Hold")
    risk_level = str(report.get("risk_level") or "Unknown")
    if operational_score <= 0.05:
        return {
            "state": "Avoid",
            "status": "avoid",
            "classification": "avoid_operationally_incomplete",
            "selection_hint": "avoid_for_now",
            "trade_candidate_eligible": False,
            "active_review_eligible": False,
            "watchlist_eligible": False,
            "reason": "Chapter 19 validation did not pass enough checks for a reliable selection decision.",
        }
    if primary_profile == "speculative_satellite" and best_score >= 0.55:
        return {
            "state": "Speculative",
            "status": "watchlist",
            "classification": "speculative_watchlist",
            "selection_hint": "keep_watching",
            "trade_candidate_eligible": bool(action in {"Buy", "Sell"} and tactical_score >= 0.65),
            "active_review_eligible": True,
            "watchlist_eligible": True,
            "reason": "The instrument has enough movement to monitor, but it should be treated as a speculative satellite.",
        }
    if action in {"Buy", "Sell"} and best_score >= 0.65 and tactical_score >= 0.55:
        return {
            "state": "Suitable",
            "status": "trade_candidate",
            "classification": "suitable_trade_candidate",
            "selection_hint": "active_review",
            "trade_candidate_eligible": True,
            "active_review_eligible": True,
            "watchlist_eligible": True,
            "reason": "Suitability, tactical readiness, and final action are aligned enough for Chapter 21 review.",
        }
    if best_score >= 0.55:
        return {
            "state": "Suitable",
            "status": "watchlist",
            "classification": "suitable_watchlist",
            "selection_hint": "keep_watching",
            "trade_candidate_eligible": False,
            "active_review_eligible": True,
            "watchlist_eligible": True,
            "reason": "The ticker is suitable to follow, but the completed tactical decision is not a fresh trade candidate.",
        }
    if best_score >= 0.40 and risk_level != "High":
        return {
            "state": "Marginal",
            "status": "monitor",
            "classification": "monitor_only",
            "selection_hint": "monitor_only",
            "trade_candidate_eligible": False,
            "active_review_eligible": False,
            "watchlist_eligible": True,
            "reason": "The ticker has some usable traits, but the profile fit is not strong.",
        }
    return {
        "state": "Avoid",
        "status": "avoid",
        "classification": "avoid_for_now",
        "selection_hint": "avoid_for_now",
        "trade_candidate_eligible": False,
        "active_review_eligible": False,
        "watchlist_eligible": False,
        "reason": "Profile fit is weak or risk dominates the suitability evidence.",
    }


def _operational_fit(report: dict[str, Any]) -> dict[str, Any]:
    chapter_19 = report.get("operations_view", {}).get("chapter_19_validation", {})
    status = chapter_19.get("status")
    score = {"pass": 1.0, "warn": 0.75, "fail": 0.0}.get(str(status), 0.50)
    return {
        "score": _round(score),
        "status": status or "unknown",
        "reason": f"Chapter 19 validation status is {status or 'unknown'}.",
    }


def _validation_fit(report: dict[str, Any]) -> dict[str, Any]:
    forecasts = [item for item in report.get("forecasts", []) if isinstance(item, dict)]
    if not forecasts:
        return {"score": 0.0, "status": "missing", "reason": "No forecasts are available."}
    confidences = [_finite_or_none(item.get("directional_confidence")) for item in forecasts]
    confidences = [value for value in confidences if value is not None]
    avg_confidence = float(np.mean(confidences)) if confidences else 0.50
    confidence_score = _clip((avg_confidence - 0.50) / 0.25)
    decision = report.get("technical_view", {}).get("decision_diagnostics", {})
    edge_ratio = _finite_or_none(decision.get("edge_to_error_ratio"))
    edge_score = _clip((edge_ratio or 0.0) / 2.0)
    holdout_scores = []
    for forecast in forecasts:
        metrics = forecast.get("validation_metrics", {})
        mae = _finite_or_none(metrics.get("mae"))
        holdout = _finite_or_none(metrics.get("holdout_mae"))
        if mae is not None and holdout is not None and mae > 0:
            holdout_scores.append(_clip(1.0 - max(0.0, holdout / mae - 1.0)))
    holdout_score = float(np.mean(holdout_scores)) if holdout_scores else 0.50
    score = 0.45 * confidence_score + 0.35 * edge_score + 0.20 * holdout_score
    return {
        "score": _round(score),
        "status": "measured",
        "average_directional_confidence": _round(avg_confidence),
        "edge_to_error_ratio": _round(edge_ratio),
        "holdout_stability_score": _round(holdout_score),
        "reason": "Validation fit combines directional confidence, edge/error, and holdout stability.",
    }


def _technical_fit(report: dict[str, Any]) -> dict[str, Any]:
    technical = report.get("technical_view", {})
    trend_state = technical.get("trend_state", {}).get("state")
    chapter_17 = technical.get("chapter_17_governance_context", {})
    humility = chapter_17.get("computer_humility", {})
    fragility = humility.get("decision_fragility", {}).get("level")
    conflict = humility.get("method_conflict_score", {}).get("level")
    trend_score = {"Bullish": 0.75, "Bearish": 0.75, "Neutral": 0.45}.get(str(trend_state), 0.50)
    fragility_score = _level_score(fragility)
    conflict_score = _level_score(conflict)
    score = 0.35 * trend_score + 0.35 * fragility_score + 0.30 * conflict_score
    return {
        "score": _round(score),
        "status": "measured",
        "trend_state": trend_state or "unknown",
        "decision_fragility": fragility or "unknown",
        "method_conflict": conflict or "unknown",
        "reason": "Technical fit rewards readable trend evidence and penalizes fragile or conflicting method stacks.",
    }


def _risk_fit(report: dict[str, Any], instrument: dict[str, Any]) -> dict[str, Any]:
    risk_level = str(report.get("risk_level") or "Unknown")
    risk_score = {"Low": 1.0, "Medium": 0.65, "High": 0.30}.get(risk_level, 0.50)
    drawdown = abs(float(instrument.get("max_drawdown_252d") or 0.0))
    drawdown_score = _clip(1.0 - drawdown / 0.60)
    score = 0.70 * risk_score + 0.30 * drawdown_score
    return {
        "score": _round(score),
        "status": "measured",
        "risk_level": risk_level,
        "max_drawdown_252d": _round(instrument.get("max_drawdown_252d")),
        "reason": "Risk fit combines model risk level and recent drawdown.",
    }


def _tactical_fit(report: dict[str, Any]) -> dict[str, Any]:
    action = str(report.get("suggested_action") or "Hold")
    chapter_18 = report.get("decision_view", {}).get("chapter_18_tactical_problem", {})
    plan = chapter_18.get("trade_plan", {})
    gate = chapter_18.get("rule_gate", {})
    rr = _finite_or_none(plan.get("reward_to_risk") or plan.get("target_plan", {}).get("reward_to_risk"))
    rr_score = _clip((rr or 0.0) / 2.5)
    action_score = 0.75 if action in {"Buy", "Sell"} else 0.40
    hard_block_count = len(gate.get("hard_blockers", []) or gate.get("hard_block_reasons", []) or [])
    block_score = 0.0 if hard_block_count else 1.0
    score = 0.45 * action_score + 0.35 * rr_score + 0.20 * block_score
    return {
        "score": _round(score),
        "status": "measured",
        "final_action": action,
        "reward_to_risk": _round(rr),
        "hard_block_count": int(hard_block_count),
        "reason": "Tactical readiness uses the final governed action, reward/risk, and hard blockers.",
    }


def _liquidity_fit(instrument: dict[str, Any]) -> dict[str, Any]:
    dollar_volume = _finite_or_none(instrument.get("average_dollar_volume_20d"))
    if dollar_volume is None:
        score = 0.50
        band = "unknown"
    elif dollar_volume >= 20_000_000:
        score = 1.0
        band = "high"
    elif dollar_volume >= 2_000_000:
        score = 0.75
        band = "medium"
    elif dollar_volume >= 500_000:
        score = 0.45
        band = "thin"
    else:
        score = 0.20
        band = "illiquid"
    return {
        "score": _round(score),
        "status": band,
        "average_dollar_volume_20d": _round(dollar_volume),
        "reason": "Liquidity fit uses recent average dollar volume when volume is available.",
    }


def _movement_fit(instrument: dict[str, Any]) -> dict[str, Any]:
    atr_pct = _finite_or_none(instrument.get("atr_pct_20d"))
    realized_vol = _finite_or_none(instrument.get("realized_volatility_20d"))
    efficiency = _finite_or_none(instrument.get("trend_efficiency_60d"))
    atr = atr_pct if atr_pct is not None else (realized_vol or 0.20) / np.sqrt(252)
    short_term = _band_score(atr, 0.004, 0.012, 0.050, 0.090)
    intermediate = _band_score(atr, 0.003, 0.008, 0.045, 0.080)
    long_term = _band_score(atr, 0.001, 0.004, 0.030, 0.065)
    speculative = _band_score(atr, 0.015, 0.035, 0.110, 0.200)
    diversifier = 0.70 * long_term + 0.30 * _clip(1.0 - abs((efficiency or 0.35) - 0.35))
    return {
        "status": "measured" if atr_pct is not None or realized_vol is not None else "estimated",
        "atr_pct_20d": _round(atr_pct),
        "realized_volatility_20d": _round(realized_vol),
        "trend_efficiency_60d": _round(efficiency),
        "short_term_score": _round(short_term),
        "intermediate_score": _round(intermediate),
        "long_term_score": _round(long_term),
        "speculative_score": _round(speculative),
        "diversifier_score": _round(diversifier),
        "reason": "Movement fit maps recent volatility and path behavior to different trading/investing profiles.",
    }


def _instrument_habit_profile(prices: pd.DataFrame | None, target_column: str) -> dict[str, Any]:
    if prices is None or prices.empty:
        return {
            "status": "not_supplied",
            "reason": "Price frame was not supplied to Chapter 20.",
        }
    frame = prices.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    if target_column not in frame:
        return {
            "status": "missing_target",
            "reason": f"Target column `{target_column}` was not found.",
        }
    close = pd.to_numeric(frame[target_column], errors="coerce").dropna()
    if close.empty:
        return {
            "status": "missing_prices",
            "reason": "No finite target prices were available.",
        }
    returns = np.log(close.replace(0, np.nan)).diff().replace([np.inf, -np.inf], np.nan).dropna()
    realized_vol_20 = float(returns.tail(20).std() * np.sqrt(252)) if len(returns.tail(20)) >= 5 else None
    realized_vol_60 = float(returns.tail(60).std() * np.sqrt(252)) if len(returns.tail(60)) >= 10 else None
    recent = close.tail(252)
    running_max = recent.cummax()
    drawdown = recent / running_max - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else None
    trend_efficiency = _trend_efficiency(close.tail(60))
    atr_pct = _atr_pct(frame, target_column=target_column)
    average_volume_20 = _series_mean_tail(frame.get("volume"), 20)
    average_dollar_volume_20 = (
        float(average_volume_20 * close.tail(20).mean()) if average_volume_20 is not None and not close.tail(20).empty else None
    )
    return {
        "status": "measured",
        "rows": int(len(close)),
        "latest_price": _round(close.iloc[-1]),
        "return_20d": _round(close.iloc[-1] / close.iloc[-20] - 1.0) if len(close) >= 20 and close.iloc[-20] else None,
        "return_60d": _round(close.iloc[-1] / close.iloc[-60] - 1.0) if len(close) >= 60 and close.iloc[-60] else None,
        "realized_volatility_20d": _round(realized_vol_20),
        "realized_volatility_60d": _round(realized_vol_60),
        "atr_pct_20d": _round(atr_pct),
        "trend_efficiency_60d": _round(trend_efficiency),
        "max_drawdown_252d": _round(max_drawdown),
        "average_volume_20d": _round(average_volume_20),
        "average_dollar_volume_20d": _round(average_dollar_volume_20),
        "reason": "Instrument habit profile summarizes movement, liquidity, and drawdown behavior from recent prices.",
    }


def _asset_context(report: dict[str, Any]) -> dict[str, Any]:
    metadata = report.get("governance", {}).get("security_metadata", {})
    ticker = str(report.get("ticker") or "").upper()
    asset_class = str(metadata.get("asset_class") or metadata.get("quote_type") or "unknown").lower()
    instrument_type = str(metadata.get("instrument_type") or metadata.get("type") or "unknown").lower()
    is_diversified = any(
        token in " ".join([ticker, asset_class, instrument_type])
        for token in ("ETF", "FUND", "INDEX", "TRUST")
    )
    return {
        "ticker": ticker,
        "asset_class": asset_class,
        "instrument_type": instrument_type,
        "sector": metadata.get("sector"),
        "industry": metadata.get("industry"),
        "is_diversified_vehicle": bool(is_diversified),
    }


def _rule_interpretation(
    report: dict[str, Any],
    primary_profile: str,
    classification: dict[str, Any],
    component_scores: dict[str, dict[str, Any]],
) -> list[str]:
    notes = [
        f"Primary Chapter 20 profile is {primary_profile}.",
        f"Classification is {classification['classification']}.",
    ]
    action = report.get("suggested_action")
    if action == "Hold":
        notes.append("A Hold action can still be a valid watchlist result, but it is not a trade candidate.")
    weak = [
        (name, payload)
        for name, payload in component_scores.items()
        if name != "movement" and float(payload.get("score", 0.0)) < 0.45
    ]
    if weak:
        name, payload = min(weak, key=lambda item: float(item[1].get("score", 0.0)))
        notes.append(f"Weakest suitability component is {name}: {payload.get('reason')}")
    return notes


def _chapter_21_message(selection_hint: str) -> str:
    if selection_hint == "active_review":
        return "Eligible for Chapter 21 active chart-list selection."
    if selection_hint == "keep_watching":
        return "Suitable for a watchlist; Chapter 21 can decide whether it stays in active review."
    if selection_hint == "monitor_only":
        return "Monitor only unless later reports improve profile fit."
    return "Do not promote to active chart selection without improved evidence."


def _level_score(level: object) -> float:
    return {"Low": 1.0, "Medium": 0.60, "High": 0.25}.get(str(level), 0.50)


def _trend_efficiency(close: pd.Series) -> float | None:
    if len(close) < 5:
        return None
    net = abs(float(close.iloc[-1] / close.iloc[0] - 1.0))
    path = close.pct_change().abs().sum()
    if not np.isfinite(path) or path <= 0:
        return None
    return float(_clip(net / float(path)))


def _atr_pct(frame: pd.DataFrame, target_column: str) -> float | None:
    if not {"high", "low", target_column}.issubset(set(frame.columns)):
        return None
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame[target_column], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.tail(20).mean()
    latest = close.dropna().iloc[-1] if close.notna().any() else np.nan
    if not np.isfinite(atr) or not np.isfinite(latest) or latest <= 0:
        return None
    return float(atr / latest)


def _series_mean_tail(series: pd.Series | None, window: int) -> float | None:
    if series is None:
        return None
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    value = float(numeric.tail(window).mean())
    return value if np.isfinite(value) else None


def _band_score(value: float | None, floor: float, ideal_low: float, ideal_high: float, ceiling: float) -> float:
    if value is None or not np.isfinite(value):
        return 0.50
    if ideal_low <= value <= ideal_high:
        return 1.0
    if value < floor or value > ceiling:
        return 0.15
    if value < ideal_low:
        return _clip((value - floor) / (ideal_low - floor))
    return _clip((ceiling - value) / (ceiling - ideal_high))


def _clip(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


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
