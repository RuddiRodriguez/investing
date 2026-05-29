from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PIVOT_LEFT_BARS = 3
PIVOT_RIGHT_BARS = 3
MIN_PATTERN_BARS = 60
ZONE_TOLERANCE_PCT = {"daily": 0.018, "weekly": 0.025, "monthly": 0.035, "chart": 0.018}
LOOKBACK_BARS = {"daily": 504, "weekly": 260, "monthly": 180, "chart": 504}
BREAK_CONFIRM_PCT = 0.006
NEAR_ZONE_PCT = 0.035


def analyze_chapter_13_support_resistance(prices: pd.DataFrame, target_column: str = "close") -> dict[str, Any]:
    """Analyze Edwards/Magee Chapter 13 support and resistance zones."""

    target = target_column.lower()
    frames = {
        "daily": prices.copy(),
        "weekly": _resample_ohlcv(prices, "W-FRI"),
        "monthly": _resample_ohlcv(prices, "ME"),
    }
    timeframes = {
        name: _analyze_timeframe(frame=frame, timeframe=name, target_column=target)
        for name, frame in frames.items()
    }
    return {
        "principle": (
            "Chapter 13 treats support and resistance as historical supply/demand zones, not single recent highs or lows. "
            "Old tops can become support and old bottoms can become resistance."
        ),
        "primary_timeframe": "weekly",
        "secondary_timeframe": "monthly",
        "decision_rule": "Use zones as governance: reward/risk, support failure, resistance breakout, and role-reversal context.",
        "preferred": _choose_preferred(timeframes),
        "support_zones": {
            "nearest": _choose_nearest(timeframes, "nearest_support"),
            "strongest": _collect_strongest(timeframes, "support_zones"),
        },
        "resistance_zones": {
            "nearest": _choose_nearest(timeframes, "nearest_resistance"),
            "strongest": _collect_strongest(timeframes, "resistance_zones"),
        },
        "round_number_zones": {
            "nearest_support": _choose_nearest(timeframes, "round_number_support"),
            "nearest_resistance": _choose_nearest(timeframes, "round_number_resistance"),
        },
        "timeframes": timeframes,
        "technical_method_card": chapter_13_support_resistance_method_card(target_column=target),
    }


def latest_chapter_13_support_resistance(
    prices: pd.DataFrame,
    target_column: str = "close",
    timeframe: str = "chart",
) -> dict[str, Any]:
    """Return Chapter 13 support/resistance diagnostics for a plotted bar set."""

    return _analyze_timeframe(
        frame=prices,
        timeframe=timeframe,
        target_column=target_column.lower(),
    )


def chapter_13_support_resistance_method_card(target_column: str = "close") -> dict[str, Any]:
    return {
        "name": "edwards_magee_chapter_13_support_resistance",
        "version": "chapter_13_alignment_v1",
        "target_column": target_column.lower(),
        "implemented_controls": [
            "volume_weighted_zones",
            "old_top_as_support",
            "old_bottom_as_resistance",
            "zone_strength_score",
            "attack_count_consumption",
            "round_number_zones",
            "support_failure_and_resistance_breakout",
        ],
        "zone_inputs": {
            "pivots": "confirmed highs/lows only after right-side bars elapse",
            "volume": "volume around pivots weights potential supply/demand",
            "distance": "zones gain importance when price moved materially away afterward",
            "age": "older zones decay slowly but remain eligible",
            "attacks": "repeated tests consume remaining supply/demand",
        },
        "decision_use": (
            "Strong nearby resistance can block fresh Buy actions; strong nearby support can block fresh Sell actions; "
            "volume-confirmed support failure or resistance breakout can support matching model direction."
        ),
    }


def _analyze_timeframe(frame: pd.DataFrame, timeframe: str, target_column: str) -> dict[str, Any]:
    clean = frame.copy().sort_index()
    if clean.empty or target_column not in clean.columns:
        return _empty_timeframe(timeframe, "InsufficientData", "missing target column", len(clean))
    clean.index = pd.DatetimeIndex(clean.index)
    clean = clean.loc[pd.to_numeric(clean[target_column], errors="coerce").notna()]
    if len(clean) < MIN_PATTERN_BARS:
        return _empty_timeframe(timeframe, "InsufficientData", "not enough bars for Chapter 13 zone analysis", len(clean))

    close = pd.to_numeric(clean[target_column], errors="coerce")
    high = pd.to_numeric(clean["high"], errors="coerce") if "high" in clean.columns else close
    low = pd.to_numeric(clean["low"], errors="coerce") if "low" in clean.columns else close
    volume = pd.to_numeric(clean["volume"], errors="coerce") if "volume" in clean.columns else pd.Series(np.nan, index=clean.index)
    lookback = min(LOOKBACK_BARS.get(timeframe, LOOKBACK_BARS["chart"]), len(clean))
    scope = clean.tail(lookback)
    zones = _build_zones(scope, timeframe=timeframe, target_column=target_column)
    latest_close = float(close.iloc[-1])
    support_zones = [zone for zone in zones if zone.get("role") == "support"]
    resistance_zones = [zone for zone in zones if zone.get("role") == "resistance"]
    support_zones = sorted(support_zones, key=lambda zone: (float(zone.get("distance_to_zone_pct") or 999), -float(zone.get("remaining_strength") or 0)))
    resistance_zones = sorted(resistance_zones, key=lambda zone: (float(zone.get("distance_to_zone_pct") or 999), -float(zone.get("remaining_strength") or 0)))
    round_support, round_resistance = _round_number_zones(clean, timeframe=timeframe, target_column=target_column)
    active_state = _active_state(
        latest_close=latest_close,
        latest_volume=volume.iloc[-1],
        volume_baseline=volume.shift(1).rolling(20, min_periods=5).mean().iloc[-1],
        nearest_support=support_zones[0] if support_zones else {},
        nearest_resistance=resistance_zones[0] if resistance_zones else {},
    )
    preferred = _preferred_timeframe_zone(
        nearest_support=support_zones[0] if support_zones else {},
        nearest_resistance=resistance_zones[0] if resistance_zones else {},
        active_state=active_state,
        timeframe=timeframe,
    )
    return {
        "state": "Measured",
        "timeframe": timeframe,
        "rows": int(len(clean)),
        "lookback_bars": int(lookback),
        "start_date": _date_string(clean.index[0]),
        "end_date": _date_string(clean.index[-1]),
        "latest_close": _finite_or_none(latest_close),
        "support_zones": [_preferred_payload(zone) for zone in support_zones[:5]],
        "resistance_zones": [_preferred_payload(zone) for zone in resistance_zones[:5]],
        "nearest_support": _preferred_payload(support_zones[0]) if support_zones else _empty_zone("NoSupportZone", timeframe, "NoPattern", "no support zone below price"),
        "nearest_resistance": _preferred_payload(resistance_zones[0]) if resistance_zones else _empty_zone("NoResistanceZone", timeframe, "NoPattern", "no resistance zone above price"),
        "round_number_support": round_support,
        "round_number_resistance": round_resistance,
        "active_state": active_state,
        "preferred": preferred,
    }


def _build_zones(frame: pd.DataFrame, timeframe: str, target_column: str) -> list[dict[str, Any]]:
    close = pd.to_numeric(frame[target_column], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce") if "high" in frame.columns else close
    low = pd.to_numeric(frame["low"], errors="coerce") if "low" in frame.columns else close
    volume = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame.columns else pd.Series(np.nan, index=frame.index)
    pivot_high, pivot_low = _confirmed_pivots(high, low)
    events = []
    for date, value in pivot_high.dropna().items():
        events.append(_zone_event(frame, volume, date, value, "top"))
    for date, value in pivot_low.dropna().items():
        events.append(_zone_event(frame, volume, date, value, "bottom"))
    events = [event for event in events if event is not None]
    if not events:
        return []
    tolerance = _zone_tolerance(close, high, low, timeframe)
    clusters = _cluster_events(events, tolerance)
    return [
        _zone_payload(cluster=cluster, frame=frame, close=close, high=high, low=low, volume=volume, tolerance=tolerance, timeframe=timeframe)
        for cluster in clusters
    ]


def _zone_event(frame: pd.DataFrame, volume: pd.Series, date: Any, value: Any, kind: str) -> dict[str, Any] | None:
    price = _finite_or_none(value)
    if price is None or price <= 0:
        return None
    position = int(frame.index.get_loc(date))
    return {
        "date": _date_string(date),
        "timestamp": pd.Timestamp(date),
        "position": position,
        "price": price,
        "kind": kind,
        "volume": _finite_or_none(volume.iloc[position]),
    }


def _cluster_events(events: list[dict[str, Any]], tolerance: float) -> list[list[dict[str, Any]]]:
    ordered = sorted(events, key=lambda item: float(item["price"]))
    clusters: list[list[dict[str, Any]]] = []
    for event in ordered:
        if not clusters:
            clusters.append([event])
            continue
        center = float(np.mean([item["price"] for item in clusters[-1]]))
        if abs(float(event["price"]) - center) <= tolerance:
            clusters[-1].append(event)
        else:
            clusters.append([event])
    return [cluster for cluster in clusters if len(cluster) >= 1]


def _zone_payload(
    cluster: list[dict[str, Any]],
    frame: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    tolerance: float,
    timeframe: str,
) -> dict[str, Any]:
    prices = np.array([float(item["price"]) for item in cluster], dtype=float)
    volumes = np.array([float(item["volume"]) for item in cluster if item.get("volume") is not None], dtype=float)
    center = float(np.average(prices, weights=np.resize(volumes, len(prices)))) if len(volumes) == len(prices) and np.sum(volumes) > 0 else float(np.mean(prices))
    lower = float(max(0.01, min(prices) - tolerance))
    upper = float(max(prices) + tolerance)
    latest_close = float(close.iloc[-1])
    top_count = sum(1 for item in cluster if item["kind"] == "top")
    bottom_count = sum(1 for item in cluster if item["kind"] == "bottom")
    if latest_close > upper * (1.0 + BREAK_CONFIRM_PCT):
        role = "support"
        role_reversal = "OldTopAsSupport" if top_count >= bottom_count else "BottomSupport"
        distance = (latest_close - upper) / latest_close
    elif latest_close < lower * (1.0 - BREAK_CONFIRM_PCT):
        role = "resistance"
        role_reversal = "OldBottomAsResistance" if bottom_count >= top_count else "TopResistance"
        distance = (lower - latest_close) / latest_close
    else:
        role = "active_zone"
        role_reversal = "InsideZone"
        distance = 0.0
    first_position = min(int(item["position"]) for item in cluster)
    last_position = max(int(item["position"]) for item in cluster)
    attack_count = _attack_count(high, low, close, lower=lower, upper=upper, start_position=last_position + 1)
    max_excursion = _max_excursion(close, center=center, start_position=last_position)
    total_volume = float(np.nansum(volumes)) if len(volumes) else np.nan
    volume_baseline = volume.rolling(20, min_periods=5).mean().median()
    volume_ratio = total_volume / (float(volume_baseline) * max(len(cluster), 1)) if pd.notna(volume_baseline) and float(volume_baseline) > 0 else np.nan
    strength = _zone_strength(
        touch_count=len(cluster),
        volume_ratio=volume_ratio,
        max_excursion=max_excursion,
        age_bars=len(frame) - 1 - last_position,
        lookback_bars=len(frame),
        attack_count=attack_count,
    )
    return {
        "pattern": "SupportResistanceZone",
        "status": "Active" if role != "active_zone" else "InsideZone",
        "role": role,
        "role_reversal": role_reversal,
        "timeframe": timeframe,
        "score": _finite_or_none(strength),
        "remaining_strength": _finite_or_none(strength),
        "center": _finite_or_none(center),
        "lower": _finite_or_none(lower),
        "upper": _finite_or_none(upper),
        "width_pct": _finite_or_none((upper - lower) / center if center else None),
        "distance_to_zone_pct": _finite_or_none(max(distance, 0.0)),
        "latest_close": _finite_or_none(latest_close),
        "touch_count": int(len(cluster)),
        "top_pivot_count": int(top_count),
        "bottom_pivot_count": int(bottom_count),
        "attack_count": int(attack_count),
        "max_excursion_pct": _finite_or_none(max_excursion),
        "volume_ratio": _finite_or_none(volume_ratio),
        "start_date": _date_string(frame.index[first_position]),
        "latest_touch_date": _date_string(frame.index[last_position]),
        "age_bars": int(len(frame) - 1 - last_position),
        "events": [
            {
                "date": item["date"],
                "kind": item["kind"],
                "price": _finite_or_none(item["price"]),
                "volume": _finite_or_none(item.get("volume")),
            }
            for item in sorted(cluster, key=lambda item: item["position"])[-6:]
        ],
        "reliability_notes": _zone_notes(role_reversal, attack_count, max_excursion, volume_ratio),
    }


def _active_state(
    latest_close: float,
    latest_volume: Any,
    volume_baseline: Any,
    nearest_support: dict[str, Any],
    nearest_resistance: dict[str, Any],
) -> dict[str, Any]:
    volume_ratio = None
    if pd.notna(latest_volume) and pd.notna(volume_baseline) and float(volume_baseline) > 0:
        volume_ratio = float(latest_volume) / float(volume_baseline)
    volume_confirmed = volume_ratio is not None and volume_ratio >= 1.25
    support_failure = False
    resistance_breakout = False
    if nearest_support and nearest_support.get("upper") is not None:
        support_failure = latest_close < float(nearest_support["lower"]) * (1.0 - BREAK_CONFIRM_PCT)
    if nearest_resistance and nearest_resistance.get("lower") is not None:
        resistance_breakout = latest_close > float(nearest_resistance["upper"]) * (1.0 + BREAK_CONFIRM_PCT)
    return {
        "support_failure": bool(support_failure),
        "resistance_breakout": bool(resistance_breakout),
        "volume_confirmed": bool(volume_confirmed),
        "latest_volume_ratio": _finite_or_none(volume_ratio),
        "near_support": bool(nearest_support and float(nearest_support.get("distance_to_zone_pct") or 999) <= NEAR_ZONE_PCT),
        "near_resistance": bool(nearest_resistance and float(nearest_resistance.get("distance_to_zone_pct") or 999) <= NEAR_ZONE_PCT),
    }


def _preferred_timeframe_zone(
    nearest_support: dict[str, Any],
    nearest_resistance: dict[str, Any],
    active_state: dict[str, Any],
    timeframe: str,
) -> dict[str, Any]:
    candidates = [zone for zone in (nearest_support, nearest_resistance) if zone and zone.get("pattern") == "SupportResistanceZone"]
    if active_state.get("support_failure") and nearest_support:
        output = dict(nearest_support)
        output["status"] = "SupportFailure"
        output["direction"] = "bearish"
        return _preferred_payload(output)
    if active_state.get("resistance_breakout") and nearest_resistance:
        output = dict(nearest_resistance)
        output["status"] = "ResistanceBreakout"
        output["direction"] = "bullish"
        return _preferred_payload(output)
    if candidates:
        return _preferred_payload(min(candidates, key=lambda zone: float(zone.get("distance_to_zone_pct") or 999)))
    return _empty_zone("NoSupportResistanceZone", timeframe, "NoPattern", "no active Chapter 13 zone")


def _round_number_zones(frame: pd.DataFrame, timeframe: str, target_column: str) -> tuple[dict[str, Any], dict[str, Any]]:
    close = pd.to_numeric(frame[target_column], errors="coerce")
    latest_close = float(close.iloc[-1])
    step = _round_step(latest_close)
    lower_level = np.floor(latest_close / step) * step
    upper_level = np.ceil(latest_close / step) * step
    if upper_level <= latest_close:
        upper_level += step
    if lower_level >= latest_close:
        lower_level -= step
    width = max(latest_close * 0.006, step * 0.04)
    support = _round_zone_payload(lower_level, width, latest_close, "support", timeframe)
    resistance = _round_zone_payload(upper_level, width, latest_close, "resistance", timeframe)
    return support, resistance


def _round_zone_payload(level: float, width: float, latest_close: float, role: str, timeframe: str) -> dict[str, Any]:
    if level <= 0:
        return _empty_zone("RoundNumberZone", timeframe, "NoPattern", "round-number level unavailable")
    distance = (latest_close - (level + width)) / latest_close if role == "support" else ((level - width) - latest_close) / latest_close
    return {
        "pattern": "RoundNumberZone",
        "status": "Active",
        "role": role,
        "role_reversal": "PsychologicalRoundNumber",
        "timeframe": timeframe,
        "score": 0.35,
        "remaining_strength": 0.35,
        "center": _finite_or_none(level),
        "lower": _finite_or_none(max(0.01, level - width)),
        "upper": _finite_or_none(level + width),
        "distance_to_zone_pct": _finite_or_none(max(distance, 0.0)),
        "latest_close": _finite_or_none(latest_close),
        "touch_count": 0,
        "attack_count": 0,
        "reliability_notes": ["Round-number level is psychological context and weaker than a volume-built historical zone."],
    }


def _choose_preferred(timeframes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for name in ("weekly", "monthly", "daily"):
        preferred = timeframes.get(name, {}).get("preferred", {})
        if preferred.get("status") not in {None, "NoPattern", "InsufficientData"}:
            candidates.append(preferred)
    if candidates:
        return _preferred_payload(max(candidates, key=_zone_rank))
    return {"pattern": "NoChapter13Zone", "status": "NoPattern", "reason": "no Chapter 13 support/resistance zone detected"}


def _choose_nearest(timeframes: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    candidates = []
    for name in ("weekly", "monthly", "daily"):
        zone = timeframes.get(name, {}).get(key, {})
        if zone.get("status") not in {None, "NoPattern", "InsufficientData"}:
            candidates.append(zone)
    if candidates:
        return _preferred_payload(min(candidates, key=lambda zone: (float(zone.get("distance_to_zone_pct") or 999), -float(zone.get("remaining_strength") or 0))))
    return {"pattern": "NoChapter13Zone", "status": "NoPattern", "reason": f"{key} unavailable"}


def _collect_strongest(timeframes: dict[str, dict[str, Any]], key: str) -> list[dict[str, Any]]:
    zones: list[dict[str, Any]] = []
    for name in ("monthly", "weekly", "daily"):
        zones.extend(timeframes.get(name, {}).get(key, [])[:3])
    zones = [zone for zone in zones if zone.get("status") not in {None, "NoPattern", "InsufficientData"}]
    return [_preferred_payload(zone) for zone in sorted(zones, key=_zone_rank, reverse=True)[:6]]


def _zone_rank(zone: dict[str, Any]) -> tuple[float, float, str]:
    status_bonus = {"SupportFailure": 1.0, "ResistanceBreakout": 1.0, "Active": 0.6, "InsideZone": 0.3}.get(str(zone.get("status")), 0.0)
    timeframe_bonus = {"monthly": 0.25, "weekly": 0.20, "daily": 0.10, "chart": 0.05}.get(str(zone.get("timeframe")), 0.0)
    role_bonus = {"OldTopAsSupport": 0.20, "OldBottomAsResistance": 0.20}.get(str(zone.get("role_reversal")), 0.0)
    strength = float(zone.get("remaining_strength") or zone.get("score") or 0.0)
    distance_penalty = min(float(zone.get("distance_to_zone_pct") or 0.0), 0.20)
    date = zone.get("latest_touch_date") or ""
    return (strength + status_bonus + timeframe_bonus + role_bonus - distance_penalty, strength, str(date))


def _confirmed_pivots(high: pd.Series, low: pd.Series) -> tuple[pd.Series, pd.Series]:
    window = PIVOT_LEFT_BARS + PIVOT_RIGHT_BARS + 1
    raw_high = high == high.rolling(window=window, center=True).max()
    raw_low = low == low.rolling(window=window, center=True).min()
    confirmed_high = high.shift(PIVOT_RIGHT_BARS).where(raw_high.shift(PIVOT_RIGHT_BARS, fill_value=False), np.nan)
    confirmed_low = low.shift(PIVOT_RIGHT_BARS).where(raw_low.shift(PIVOT_RIGHT_BARS, fill_value=False), np.nan)
    return confirmed_high, confirmed_low


def _zone_tolerance(close: pd.Series, high: pd.Series, low: pd.Series, timeframe: str) -> float:
    median_close = float(close.dropna().median())
    atr = _true_range(high, low, close).rolling(20, min_periods=5).mean().median()
    pct_width = median_close * ZONE_TOLERANCE_PCT.get(timeframe, ZONE_TOLERANCE_PCT["chart"])
    atr_width = float(atr) * 0.75 if pd.notna(atr) else 0.0
    return float(max(pct_width, atr_width, median_close * 0.008))


def _zone_strength(
    touch_count: int,
    volume_ratio: float,
    max_excursion: float,
    age_bars: int,
    lookback_bars: int,
    attack_count: int,
) -> float:
    touch_component = 0.25 * _clip01(touch_count / 5.0)
    volume_component = 0.25 * _clip01(float(volume_ratio) / 2.0) if np.isfinite(volume_ratio) else 0.10
    excursion_component = 0.25 * _clip01(max_excursion / 0.20)
    age_component = 0.15 * (0.45 + 0.55 * np.exp(-age_bars / max(lookback_bars, 1)))
    base = 0.10 + touch_component + volume_component + excursion_component + age_component
    remaining = base * max(0.45, 1.0 - 0.12 * max(0, attack_count - 1))
    return float(min(max(remaining, 0.0), 0.99))


def _attack_count(high: pd.Series, low: pd.Series, close: pd.Series, lower: float, upper: float, start_position: int) -> int:
    attacks = 0
    in_attack = False
    for position in range(max(0, start_position), len(close)):
        touched = float(high.iloc[position]) >= lower and float(low.iloc[position]) <= upper
        if touched and not in_attack:
            attacks += 1
            in_attack = True
        elif not touched:
            in_attack = False
    return attacks


def _max_excursion(close: pd.Series, center: float, start_position: int) -> float:
    scope = close.iloc[start_position:]
    if scope.empty or center <= 0:
        return 0.0
    return float(max(abs(float(scope.max()) / center - 1.0), abs(float(scope.min()) / center - 1.0)))


def _zone_notes(role_reversal: str, attack_count: int, max_excursion: float, volume_ratio: float) -> list[str]:
    notes: list[str] = []
    if role_reversal == "OldTopAsSupport":
        notes.append("Former top is now acting as potential support.")
    if role_reversal == "OldBottomAsResistance":
        notes.append("Former bottom is now acting as potential resistance.")
    if attack_count >= 2:
        notes.append("Repeated attacks have consumed part of the zone's remaining supply/demand.")
    if max_excursion >= 0.10:
        notes.append("Price moved far enough away to create meaningful vested-interest pressure.")
    if np.isfinite(volume_ratio) and volume_ratio >= 1.4:
        notes.append("Zone has above-baseline volume concentration.")
    return notes


def _round_step(price: float) -> float:
    if price < 10:
        return 1.0
    if price < 50:
        return 5.0
    if price < 100:
        return 10.0
    if price < 250:
        return 25.0
    if price < 500:
        return 50.0
    return 100.0


def _preferred_payload(zone: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "pattern",
        "status",
        "role",
        "role_reversal",
        "direction",
        "timeframe",
        "score",
        "remaining_strength",
        "center",
        "lower",
        "upper",
        "width_pct",
        "distance_to_zone_pct",
        "latest_close",
        "touch_count",
        "top_pivot_count",
        "bottom_pivot_count",
        "attack_count",
        "max_excursion_pct",
        "volume_ratio",
        "start_date",
        "latest_touch_date",
        "age_bars",
        "events",
        "reliability_notes",
        "reason",
    ]
    return {key: zone[key] for key in keys if key in zone}


def _empty_timeframe(timeframe: str, status: str, reason: str, rows: int) -> dict[str, Any]:
    return {
        "state": status,
        "timeframe": timeframe,
        "rows": int(rows),
        "support_zones": [],
        "resistance_zones": [],
        "nearest_support": _empty_zone("NoSupportZone", timeframe, status, reason),
        "nearest_resistance": _empty_zone("NoResistanceZone", timeframe, status, reason),
        "round_number_support": _empty_zone("RoundNumberZone", timeframe, status, reason),
        "round_number_resistance": _empty_zone("RoundNumberZone", timeframe, status, reason),
        "active_state": {},
        "preferred": _empty_zone("NoSupportResistanceZone", timeframe, status, reason),
    }


def _empty_zone(pattern: str, timeframe: str, status: str, reason: str) -> dict[str, Any]:
    return {"pattern": pattern, "status": status, "timeframe": timeframe, "reason": reason}


def _resample_ohlcv(prices: pd.DataFrame, rule: str) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()
    aggregations: dict[str, str] = {}
    if "open" in prices.columns:
        aggregations["open"] = "first"
    if "high" in prices.columns:
        aggregations["high"] = "max"
    if "low" in prices.columns:
        aggregations["low"] = "min"
    if "close" in prices.columns:
        aggregations["close"] = "last"
    if "volume" in prices.columns:
        aggregations["volume"] = "sum"
    for optional in ("dividends", "stock_splits"):
        if optional in prices.columns:
            aggregations[optional] = "sum"
    frame = prices.resample(rule).agg(aggregations)
    return frame.dropna(subset=["close"]) if "close" in frame.columns else frame.dropna(how="all")


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prior_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prior_close).abs(),
            (low - prior_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def _date_string(value: Any) -> str:
    return str(pd.Timestamp(value).date())


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _finite_or_none(value: Any) -> float | None:
    try:
        output = float(value)
    except Exception:
        return None
    return output if np.isfinite(output) else None
