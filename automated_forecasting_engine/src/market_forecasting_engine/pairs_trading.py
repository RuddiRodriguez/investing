from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


def analyze_pair(
    prices_a: pd.Series,
    prices_b: pd.Series,
    *,
    symbol_a: str = "asset_a",
    symbol_b: str = "asset_b",
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.5,
    min_rows: int = 120,
) -> dict[str, Any]:
    """Return cointegration and spread diagnostics for one candidate pair.

    This helper is deliberately standalone. It does not fetch data, place trades,
    or influence the single-ticker forecasting action gate.
    """

    joined = pd.concat(
        [
            pd.to_numeric(prices_a, errors="coerce").rename("a"),
            pd.to_numeric(prices_b, errors="coerce").rename("b"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    joined = joined[(joined["a"] > 0) & (joined["b"] > 0)]
    if len(joined) < min_rows:
        return {
            "status": "insufficient_data",
            "symbol_a": symbol_a,
            "symbol_b": symbol_b,
            "rows": int(len(joined)),
            "minimum_rows": int(min_rows),
        }

    log_a = np.log(joined["a"])
    log_b = np.log(joined["b"])
    hedge_ratio = _hedge_ratio(log_a, log_b)
    spread = log_a - hedge_ratio * log_b
    zscore = _last_zscore(spread)
    coint = _cointegration_test(log_a, log_b)
    half_life = _half_life_days(spread)
    signal = _spread_signal(zscore, entry_zscore=entry_zscore, exit_zscore=exit_zscore)
    status = "candidate" if coint["p_value"] < 0.05 and np.isfinite(half_life) else "not_cointegrated"
    return {
        "status": status,
        "symbol_a": symbol_a,
        "symbol_b": symbol_b,
        "rows": int(len(joined)),
        "hedge_ratio": _finite(hedge_ratio),
        "spread_zscore": _finite(zscore),
        "half_life_days": _finite(half_life),
        "cointegration": coint,
        "signal": signal,
        "entry_zscore": float(entry_zscore),
        "exit_zscore": float(exit_zscore),
        "decision_policy": {
            "influences_single_ticker_forecast": False,
            "mode": "research_only",
        },
    }


def rank_cointegrated_pairs(
    price_frame: pd.DataFrame,
    *,
    max_pairs: int = 20,
    min_rows: int = 120,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.5,
) -> list[dict[str, Any]]:
    """Rank pair diagnostics for a wide price frame with one column per ticker."""

    numeric = price_frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    results = []
    for left, right in combinations(numeric.columns, 2):
        result = analyze_pair(
            numeric[left],
            numeric[right],
            symbol_a=str(left),
            symbol_b=str(right),
            min_rows=min_rows,
            entry_zscore=entry_zscore,
            exit_zscore=exit_zscore,
        )
        if result["status"] != "insufficient_data":
            results.append(result)
    results.sort(key=_pair_rank_key)
    return results[: max(int(max_pairs), 0)]


def analyze_pairs_from_panel(
    panel: pd.DataFrame,
    *,
    ticker_column: str = "ticker",
    price_column: str = "close",
    max_pairs: int = 20,
    min_rows: int = 120,
) -> list[dict[str, Any]]:
    """Rank pairs from a long panel containing ticker and price columns."""

    if ticker_column not in panel or price_column not in panel:
        raise ValueError(f"Panel must contain `{ticker_column}` and `{price_column}` columns.")
    wide = panel.pivot_table(values=price_column, index=panel.index, columns=ticker_column, aggfunc="last")
    return rank_cointegrated_pairs(wide, max_pairs=max_pairs, min_rows=min_rows)


def _pair_rank_key(result: dict[str, Any]) -> tuple[int, float, float]:
    status_rank = 0 if result.get("status") == "candidate" else 1
    p_value = _number(result.get("cointegration", {}).get("p_value"))
    zscore = abs(_number(result.get("spread_zscore")))
    return status_rank, p_value, -zscore


def _hedge_ratio(log_a: pd.Series, log_b: pd.Series) -> float:
    variance = float(np.var(log_b.to_numpy(dtype=float)))
    if not np.isfinite(variance) or variance <= 0:
        return 1.0
    covariance = float(np.cov(log_a.to_numpy(dtype=float), log_b.to_numpy(dtype=float))[0, 1])
    ratio = covariance / variance
    return ratio if np.isfinite(ratio) else 1.0


def _cointegration_test(log_a: pd.Series, log_b: pd.Series) -> dict[str, Any]:
    try:
        from statsmodels.tsa.stattools import coint

        statistic, p_value, critical_values = coint(log_a, log_b)
        return {
            "method": "engle_granger",
            "statistic": _finite(float(statistic)),
            "p_value": _finite(float(p_value)),
            "critical_values": [_finite(float(value)) for value in critical_values],
        }
    except Exception:
        spread = log_a - _hedge_ratio(log_a, log_b) * log_b
        acf1 = _number(spread.autocorr(lag=1))
        return {
            "method": "acf_fallback",
            "statistic": _finite(acf1),
            "p_value": _finite(max(0.0, min(1.0, abs(acf1)))),
            "critical_values": [],
        }


def _half_life_days(spread: pd.Series) -> float:
    clean = pd.to_numeric(spread, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 20:
        return float("inf")
    lagged = clean.shift(1).dropna()
    delta = clean.diff().dropna().reindex(lagged.index)
    frame = pd.concat([lagged.rename("lagged"), delta.rename("delta")], axis=1).dropna()
    if frame["lagged"].var() <= 0:
        return float("inf")
    beta = float(np.polyfit(frame["lagged"], frame["delta"], 1)[0])
    if not np.isfinite(beta) or beta >= 0:
        return float("inf")
    return float(-np.log(2) / beta)


def _last_zscore(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return 0.0
    std = clean.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return 0.0
    return float((clean.iloc[-1] - clean.mean()) / std)


def _spread_signal(zscore: float, *, entry_zscore: float, exit_zscore: float) -> str:
    if zscore >= entry_zscore:
        return "short_spread"
    if zscore <= -entry_zscore:
        return "long_spread"
    if abs(zscore) <= exit_zscore:
        return "neutral_exit_zone"
    return "watch"


def _number(value: Any) -> float:
    try:
        number = float(value if value is not None else 0.0)
    except Exception:
        return 0.0
    return number if np.isfinite(number) else 0.0


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None
