from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from market_forecasting_engine.daily_trade import build_intraday_chart_confirmation
from market_forecasting_engine.data import normalize_price_frame
from market_forecasting_engine.risk_profiles import risk_profile_for_name


def annotate_mean_reversion_dip_buy(
    report: dict[str, Any],
    prices: pd.DataFrame,
    target_column: str,
    risk_profile_name: str = "medium",
) -> None:
    """Add conditional buy-lower setups as a separate view from momentum forecasts."""

    risk_profile = risk_profile_for_name(risk_profile_name)
    chart = report.get("decision_view", {}).get("production_gate", {}).get("chart_confirmation") or build_intraday_chart_confirmation(
        prices, target_column=target_column
    )
    current_price = float(report.get("current_price") or prices[target_column].iloc[-1])
    if current_price <= 0:
        return

    candidates = []
    for forecast in report.get("forecasts", []):
        predicted = _float_or_none(forecast.get("predicted_price"))
        lower = _float_or_none(forecast.get("lower_price"))
        upper = _float_or_none(forecast.get("upper_price"))
        horizon_bars = _horizon_bars(forecast)
        if predicted is None or lower is None or upper is None or horizon_bars <= 0:
            continue
        if predicted >= current_price * (1.0 - risk_profile.minimum_edge_fraction):
            continue

        entry = float(predicted)
        stop = min(float(lower), entry * 0.995)
        target = min(float(chart.get("resistance_level") or current_price), current_price)
        if target <= entry or stop >= entry:
            continue

        reward = target - entry
        risk = entry - stop
        reward_risk = reward / max(risk, 1e-9)
        stats = historical_reversal_stats(
            prices=prices,
            target_column=target_column,
            current_price=current_price,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            horizon_bars=horizon_bars,
        )
        sigma = max((upper - lower) / 3.92, current_price * 0.001)
        terminal_reclaim_probability = 1.0 - _normal_cdf((target - predicted) / sigma)
        reversal_probability = stats["success_probability"] if stats["event_count"] >= 8 else terminal_reclaim_probability
        reasons = []
        if reward_risk < 1.5:
            reasons.append("reward_risk_too_low")
        if reversal_probability < max(0.50, risk_profile.minimum_directional_confidence - 0.02):
            reasons.append("reversal_probability_too_low")
        if stats["event_count"] < 8:
            reasons.append("limited_historical_reversal_events")

        candidates.append(
            {
                "setup": "conditional_dip_buy",
                "horizon_days": forecast.get("horizon_days"),
                "horizon_hours": forecast.get("horizon_hours"),
                "entry_price": round(entry, 2),
                "stop_price": round(stop, 2),
                "target_price": round(target, 2),
                "reward_risk": round(float(reward_risk), 3),
                "reversal_probability": round(float(reversal_probability), 4),
                "historical_event_count": stats["event_count"],
                "historical_success_probability": stats["success_probability"],
                "terminal_reclaim_probability": round(float(terminal_reclaim_probability), 4),
                "source_forecast_direction": forecast.get("expected_direction"),
                "source_forecast_allowed": forecast.get("trade_allowed"),
                "source_model": forecast.get("selected_model"),
                "allowed": not reasons,
                "reasons": reasons,
                "order_template": {
                    "symbol": _alpaca_order_symbol(report.get("ticker", "")),
                    "side": "buy",
                    "type": "limit",
                    "limit_price": round(entry, 2),
                    "time_in_force": "gtc",
                },
                "policy": (
                    "This is a conditional mean-reversion setup: buy lower only if price reaches the entry zone "
                    "and the stop/target math clears the reversal gate."
                ),
            }
        )

    candidates = sorted(candidates, key=lambda item: (item["allowed"], item["reward_risk"], item["reversal_probability"]), reverse=True)
    report.setdefault("decision_view", {})["mean_reversion_dip_buy"] = {
        "best_setup": candidates[0] if candidates else None,
        "setups": candidates,
        "policy": (
            "Momentum forecasts can be bearish while a lower conditional buy setup is attractive. "
            "This view evaluates dip-buy entries separately using forecast downside zones, support/resistance, "
            "reward/risk, and historical reversal behavior."
        ),
    }


def historical_reversal_stats(
    *,
    prices: pd.DataFrame,
    target_column: str,
    current_price: float,
    entry_price: float,
    stop_price: float,
    target_price: float,
    horizon_bars: int,
) -> dict[str, Any]:
    close = normalize_price_frame(prices, target_column=target_column)[target_column.lower()].astype(float).dropna()
    if len(close) < horizon_bars + 20 or current_price <= 0 or entry_price <= 0:
        return {"event_count": 0, "success_count": 0, "success_probability": 0.0}
    entry_ratio = entry_price / current_price
    stop_ratio = stop_price / entry_price
    target_ratio = target_price / entry_price
    events = 0
    successes = 0
    max_start = max(0, len(close) - horizon_bars - 1)
    for index in range(max_start):
        anchor = float(close.iloc[index])
        entry_level = anchor * entry_ratio
        stop_level = entry_level * stop_ratio
        target_level = entry_level * target_ratio
        future = close.iloc[index + 1 : index + horizon_bars + 1]
        hit_positions = np.where(future.to_numpy(dtype=float) <= entry_level)[0]
        if len(hit_positions) == 0:
            continue
        events += 1
        after_hit = future.iloc[int(hit_positions[0]) :]
        for price in after_hit:
            value = float(price)
            if value <= stop_level:
                break
            if value >= target_level:
                successes += 1
                break
    probability = successes / events if events else 0.0
    return {"event_count": int(events), "success_count": int(successes), "success_probability": round(float(probability), 4)}


def _horizon_bars(forecast: dict[str, Any]) -> int:
    value = forecast.get("horizon_bars") or forecast.get("horizon_days") or forecast.get("horizon_hours")
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def _float_or_none(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _alpaca_order_symbol(ticker: object) -> object:
    symbol = str(ticker or "")
    return symbol.replace("-", "/") if symbol.endswith("-USD") else ticker
