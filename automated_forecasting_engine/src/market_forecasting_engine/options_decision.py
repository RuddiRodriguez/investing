from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OptionsDecisionConfig:
    ticker: str
    chain_csv: str | None = None
    risk_profile: str = "medium"
    strike_pct_range: float = 0.04
    strike_count: int = 17
    risk_free_rate: float = 0.04
    iv_multiplier: float = 1.25
    min_edge_pct: float = 0.08
    max_spread_pct: float = 0.18
    min_probability_above_breakeven: float = 0.52
    max_candidates: int = 8


def build_options_decision(
    report: dict[str, Any],
    prices: pd.DataFrame,
    *,
    target_column: str = "close",
    config: OptionsDecisionConfig,
) -> dict[str, Any]:
    current_price = float(report.get("current_price") or prices[target_column].astype(float).iloc[-1])
    as_of = pd.Timestamp(report.get("as_of_timestamp") or prices.index[-1])
    volatility = _annualized_intraday_volatility(prices[target_column].astype(float))
    chain = _load_or_build_chain(
        config=config,
        current_price=current_price,
        as_of=as_of,
        forecasts=report.get("forecasts", []),
        annualized_volatility=volatility,
    )
    candidates = _score_chain(
        chain,
        forecasts=report.get("forecasts", []),
        current_price=current_price,
        as_of=as_of,
        annualized_volatility=volatility,
        config=config,
    )
    ranked = candidates.sort_values(["decision_rank", "expected_value_pct"], ascending=[True, False])
    top = ranked.head(config.max_candidates).copy()
    best_trade = _best_trade(top, config)
    summary = {
        "ticker": config.ticker.upper(),
        "as_of": as_of.isoformat(),
        "current_price": current_price,
        "mode": "real_chain_csv" if config.chain_csv else "synthetic_chain_for_research",
        "risk_profile": config.risk_profile,
        "annualized_volatility_estimate": volatility,
        "policy": {
            "min_edge_pct": config.min_edge_pct,
            "max_spread_pct": config.max_spread_pct,
            "min_probability_above_breakeven": config.min_probability_above_breakeven,
            "note": "Options scoring compares forecast distributions to option breakeven, premium, spread, and validation error. Synthetic chains are for research only.",
        },
        "best_trade": best_trade,
        "top_candidates": top.replace([np.inf, -np.inf], np.nan).to_dict(orient="records"),
    }
    return summary


def write_options_artifacts(decision: dict[str, Any], output_dir: Path) -> dict[str, str]:
    path = output_dir / "options_decision.json"
    path.write_text(json.dumps(decision, indent=2, default=str) + "\n", encoding="utf-8")
    candidates_path = output_dir / "options_candidates.csv"
    pd.DataFrame(decision.get("top_candidates", [])).to_csv(candidates_path, index=False)
    return {"options_decision": str(path), "options_candidates": str(candidates_path)}


def _load_or_build_chain(
    *,
    config: OptionsDecisionConfig,
    current_price: float,
    as_of: pd.Timestamp,
    forecasts: list[dict[str, Any]],
    annualized_volatility: float,
) -> pd.DataFrame:
    if config.chain_csv:
        chain = pd.read_csv(config.chain_csv)
        return _normalize_chain(chain, as_of=as_of, current_price=current_price, risk_free_rate=config.risk_free_rate)
    return _synthetic_chain(
        current_price=current_price,
        as_of=as_of,
        forecasts=forecasts,
        annualized_volatility=annualized_volatility * config.iv_multiplier,
        risk_free_rate=config.risk_free_rate,
        strike_pct_range=config.strike_pct_range,
        strike_count=config.strike_count,
    )


def _normalize_chain(chain: pd.DataFrame, *, as_of: pd.Timestamp, current_price: float, risk_free_rate: float) -> pd.DataFrame:
    output = chain.copy()
    output.columns = [str(column).strip().lower().replace(" ", "_") for column in output.columns]
    if "option_type" not in output and "type" in output:
        output["option_type"] = output["type"]
    required = {"strike", "option_type"}
    missing = required - set(output.columns)
    if missing:
        raise ValueError(f"Options chain CSV is missing required columns: {sorted(missing)}")
    output["option_type"] = output["option_type"].astype(str).str.lower().str[0].map({"c": "call", "p": "put"}).fillna(output["option_type"])
    if "expiry" not in output:
        if "expiration" in output:
            output["expiry"] = output["expiration"]
        elif "expiry_hours" in output:
            output["expiry"] = as_of + pd.to_timedelta(pd.to_numeric(output["expiry_hours"], errors="coerce"), unit="h")
        else:
            raise ValueError("Options chain CSV must include expiry, expiration, or expiry_hours.")
    output["expiry"] = pd.to_datetime(output["expiry"], errors="coerce")
    output["time_to_expiry_years"] = (output["expiry"] - as_of).dt.total_seconds() / (365.0 * 24.0 * 3600.0)
    output["time_to_expiry_years"] = output["time_to_expiry_years"].clip(lower=1.0 / (365.0 * 24.0))
    if "mid" not in output:
        if {"bid", "ask"}.issubset(output.columns):
            output["mid"] = (pd.to_numeric(output["bid"], errors="coerce") + pd.to_numeric(output["ask"], errors="coerce")) / 2.0
        elif "mark" in output:
            output["mid"] = pd.to_numeric(output["mark"], errors="coerce")
        elif "last" in output:
            output["mid"] = pd.to_numeric(output["last"], errors="coerce")
    if "bid" not in output:
        output["bid"] = output["mid"] * 0.97
    if "ask" not in output:
        output["ask"] = output["mid"] * 1.03
    if "iv" not in output:
        output["iv"] = np.nan
    output["iv"] = pd.to_numeric(output["iv"], errors="coerce")
    output["risk_free_rate"] = risk_free_rate
    output["synthetic"] = False
    return output.dropna(subset=["strike", "mid", "expiry", "time_to_expiry_years"])


def _synthetic_chain(
    *,
    current_price: float,
    as_of: pd.Timestamp,
    forecasts: list[dict[str, Any]],
    annualized_volatility: float,
    risk_free_rate: float,
    strike_pct_range: float,
    strike_count: int,
) -> pd.DataFrame:
    strikes = np.linspace(current_price * (1.0 - strike_pct_range), current_price * (1.0 + strike_pct_range), strike_count)
    expiry_hours = sorted({float(item.get("horizon_hours", 1.0) or 1.0) for item in forecasts}) or [1.0, 2.0, 4.0]
    rows = []
    iv = max(float(annualized_volatility), 0.20)
    for hours in expiry_hours:
        expiry = as_of + pd.Timedelta(hours=hours)
        t = max(hours / (365.0 * 24.0), 1.0 / (365.0 * 24.0))
        for strike in strikes:
            for option_type in ("call", "put"):
                mid = _black_scholes_price(current_price, float(strike), t, risk_free_rate, iv, option_type)
                spread = max(0.01, mid * 0.08)
                rows.append(
                    {
                        "symbol": f"SYN-{option_type.upper()}-{expiry.strftime('%Y%m%d%H%M')}-{strike:.2f}",
                        "option_type": option_type,
                        "strike": float(strike),
                        "expiry": expiry,
                        "expiry_hours": float(hours),
                        "time_to_expiry_years": t,
                        "bid": max(0.0, mid - spread / 2.0),
                        "ask": mid + spread / 2.0,
                        "mid": mid,
                        "iv": iv,
                        "delta": _black_scholes_delta(current_price, float(strike), t, risk_free_rate, iv, option_type),
                        "open_interest": np.nan,
                        "volume": np.nan,
                        "risk_free_rate": risk_free_rate,
                        "synthetic": True,
                    }
                )
    return pd.DataFrame(rows)


def _score_chain(
    chain: pd.DataFrame,
    *,
    forecasts: list[dict[str, Any]],
    current_price: float,
    as_of: pd.Timestamp,
    annualized_volatility: float,
    config: OptionsDecisionConfig,
) -> pd.DataFrame:
    forecast_frame = pd.DataFrame(forecasts)
    if forecast_frame.empty:
        return pd.DataFrame()
    forecast_frame["forecast_dt"] = pd.to_datetime(forecast_frame["forecast_date"], errors="coerce")
    rows = []
    for _, option in chain.iterrows():
        expiry = pd.Timestamp(option["expiry"])
        forecast = _nearest_forecast(forecast_frame, expiry)
        if forecast is None:
            continue
        t = float(option.get("time_to_expiry_years", max((expiry - as_of).total_seconds() / (365.0 * 24.0 * 3600.0), 1e-6)))
        strike = float(option["strike"])
        option_type = str(option["option_type"]).lower()
        mid = float(option["mid"])
        bid = float(option.get("bid", mid))
        ask = float(option.get("ask", mid))
        spread_pct = (ask - bid) / max(mid, 1e-9)
        forecast_price = float(forecast.get("predicted_price", current_price))
        validation_mae = _validation_mae(forecast)
        sigma = _forecast_price_sigma(current_price, forecast_price, validation_mae, annualized_volatility, t)
        probability_itm = _probability_itm(forecast_price, sigma, strike, option_type)
        breakeven = strike + ask if option_type == "call" else strike - ask
        probability_above_breakeven = _probability_breakeven(forecast_price, sigma, breakeven, option_type)
        expected_payoff = _expected_option_payoff(forecast_price, sigma, strike, option_type)
        expected_value = expected_payoff - ask
        expected_value_pct = expected_value / max(ask, 1e-9)
        liquidity_penalty = min(0.25, spread_pct * 0.50)
        validation_penalty = min(0.35, validation_mae * 8.0)
        risk_adjusted_ev_pct = expected_value_pct - liquidity_penalty - validation_penalty
        decision = _candidate_decision(
            expected_value_pct=expected_value_pct,
            risk_adjusted_ev_pct=risk_adjusted_ev_pct,
            probability_above_breakeven=probability_above_breakeven,
            spread_pct=spread_pct,
            config=config,
        )
        rows.append(
            {
                "symbol": option.get("symbol", ""),
                "ticker": config.ticker.upper(),
                "option_type": option_type,
                "strike": strike,
                "expiry": expiry.isoformat(),
                "expiry_hours": round((expiry - as_of).total_seconds() / 3600.0, 4),
                "matched_horizon_hours": forecast.get("horizon_hours"),
                "forecast_price": forecast_price,
                "forecast_direction": forecast.get("expected_direction"),
                "directional_confidence": forecast.get("directional_confidence"),
                "lower_price": forecast.get("lower_price"),
                "upper_price": forecast.get("upper_price"),
                "validation_mae": validation_mae,
                "spot": current_price,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_pct": spread_pct,
                "iv": float(option.get("iv", np.nan)),
                "delta": option.get("delta", np.nan),
                "breakeven": breakeven,
                "probability_itm": probability_itm,
                "probability_above_breakeven": probability_above_breakeven,
                "expected_payoff": expected_payoff,
                "expected_value": expected_value,
                "expected_value_pct": expected_value_pct,
                "risk_adjusted_expected_value_pct": risk_adjusted_ev_pct,
                "decision": decision["decision"],
                "decision_reasons": ";".join(decision["reasons"]),
                "decision_rank": decision["rank"],
                "synthetic": bool(option.get("synthetic", False)),
            }
        )
    return pd.DataFrame(rows)


def _best_trade(candidates: pd.DataFrame, config: OptionsDecisionConfig) -> dict[str, Any]:
    if candidates.empty:
        return {"action": "no_trade", "reason": "No options candidates were available."}
    tradable = candidates[candidates["decision"] == "candidate"]
    if tradable.empty:
        best = candidates.sort_values("risk_adjusted_expected_value_pct", ascending=False).iloc[0]
        return {
            "action": "no_trade",
            "reason": str(best.get("decision_reasons", "No candidate cleared the options gates.")),
            "best_rejected": _row_to_dict(best),
        }
    best = tradable.sort_values("risk_adjusted_expected_value_pct", ascending=False).iloc[0]
    return {
        "action": f"buy_{best['option_type']}",
        "symbol": best.get("symbol", ""),
        "strike": float(best["strike"]),
        "expiry": best["expiry"],
        "ask": float(best["ask"]),
        "breakeven": float(best["breakeven"]),
        "probability_above_breakeven": float(best["probability_above_breakeven"]),
        "expected_value_pct": float(best["expected_value_pct"]),
        "risk_adjusted_expected_value_pct": float(best["risk_adjusted_expected_value_pct"]),
        "synthetic": bool(best.get("synthetic", False)),
        "policy": "Candidate cleared expected-value, breakeven-probability, and spread gates.",
    }


def _nearest_forecast(forecast_frame: pd.DataFrame, expiry: pd.Timestamp) -> dict[str, Any] | None:
    valid = forecast_frame.dropna(subset=["forecast_dt"])
    if valid.empty:
        return None
    distances = (valid["forecast_dt"] - expiry).abs()
    return valid.loc[distances.idxmin()].to_dict()


def _validation_mae(forecast: dict[str, Any]) -> float:
    metrics = forecast.get("validation_metrics", {}) or {}
    values = [
        float(metrics.get("mae", 0.0) or 0.0),
        float(metrics.get("holdout_mae", 0.0) or 0.0),
    ]
    return max(values)


def _forecast_price_sigma(
    current_price: float,
    forecast_price: float,
    validation_mae: float,
    annualized_volatility: float,
    time_to_expiry: float,
) -> float:
    validation_sigma = max(validation_mae * current_price, current_price * 0.001)
    market_sigma = current_price * max(annualized_volatility, 0.05) * math.sqrt(max(time_to_expiry, 1e-9))
    drift_sigma = abs(forecast_price - current_price) * 0.35
    return max(validation_sigma, market_sigma, drift_sigma, current_price * 0.001)


def _probability_itm(mean: float, sigma: float, strike: float, option_type: str) -> float:
    if option_type == "call":
        return 1.0 - _normal_cdf((strike - mean) / sigma)
    return _normal_cdf((strike - mean) / sigma)


def _probability_breakeven(mean: float, sigma: float, breakeven: float, option_type: str) -> float:
    if option_type == "call":
        return 1.0 - _normal_cdf((breakeven - mean) / sigma)
    return _normal_cdf((breakeven - mean) / sigma)


def _expected_option_payoff(mean: float, sigma: float, strike: float, option_type: str) -> float:
    if sigma <= 0:
        return max(0.0, mean - strike) if option_type == "call" else max(0.0, strike - mean)
    d = (mean - strike) / sigma
    pdf = _normal_pdf(d)
    if option_type == "call":
        return (mean - strike) * _normal_cdf(d) + sigma * pdf
    return (strike - mean) * _normal_cdf(-d) + sigma * pdf


def _candidate_decision(
    *,
    expected_value_pct: float,
    risk_adjusted_ev_pct: float,
    probability_above_breakeven: float,
    spread_pct: float,
    config: OptionsDecisionConfig,
) -> dict[str, Any]:
    reasons = []
    if expected_value_pct < config.min_edge_pct:
        reasons.append("expected_value_below_threshold")
    if risk_adjusted_ev_pct < 0.0:
        reasons.append("risk_adjusted_ev_negative")
    if probability_above_breakeven < config.min_probability_above_breakeven:
        reasons.append("low_probability_above_breakeven")
    if spread_pct > config.max_spread_pct:
        reasons.append("spread_too_wide")
    if reasons:
        return {"decision": "reject", "reasons": reasons, "rank": 1}
    return {"decision": "candidate", "reasons": ["passed_options_gates"], "rank": 0}


def _annualized_intraday_volatility(close: pd.Series) -> float:
    returns = np.log(close.replace(0, np.nan)).diff().replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return 0.50
    interval_minutes = _infer_interval_minutes(close.index)
    periods_per_year = 365.0 * 24.0 * 60.0 / max(interval_minutes, 1.0)
    return float(returns.tail(2000).std() * math.sqrt(periods_per_year))


def _infer_interval_minutes(index: pd.Index) -> float:
    if len(index) < 2:
        return 5.0
    deltas = pd.Series(pd.DatetimeIndex(index)).diff().dropna().dt.total_seconds() / 60.0
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return 5.0
    return float(deltas.median())


def _black_scholes_price(spot: float, strike: float, t: float, r: float, sigma: float, option_type: str) -> float:
    d1, d2 = _d1_d2(spot, strike, t, r, sigma)
    if option_type == "call":
        return max(0.0, spot * _normal_cdf(d1) - strike * math.exp(-r * t) * _normal_cdf(d2))
    return max(0.0, strike * math.exp(-r * t) * _normal_cdf(-d2) - spot * _normal_cdf(-d1))


def _black_scholes_delta(spot: float, strike: float, t: float, r: float, sigma: float, option_type: str) -> float:
    d1, _ = _d1_d2(spot, strike, t, r, sigma)
    if option_type == "call":
        return _normal_cdf(d1)
    return _normal_cdf(d1) - 1.0


def _d1_d2(spot: float, strike: float, t: float, r: float, sigma: float) -> tuple[float, float]:
    sigma = max(sigma, 1e-6)
    t = max(t, 1e-9)
    d1 = (math.log(max(spot, 1e-9) / max(strike, 1e-9)) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return d1, d2


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    output = row.replace([np.inf, -np.inf], np.nan).to_dict()
    return {key: (None if pd.isna(value) else value) for key, value in output.items()}
