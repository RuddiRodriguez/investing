from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def evaluate_factors(
    features: pd.DataFrame,
    supervised: pd.DataFrame,
    horizons: tuple[int, ...],
    top_n: int = 25,
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate feature predictive quality with IC, quantile spread, and turnover."""

    result: dict[str, list[dict[str, Any]]] = {}
    for horizon in horizons:
        target_name = f"target_log_return_{horizon}d"
        if target_name not in supervised.columns:
            continue
        target = supervised[target_name]
        rows = []
        for column in features.columns:
            series = features[column]
            pair = pd.concat([series.rename("factor"), target.rename("target")], axis=1)
            pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 40 or pair["factor"].nunique(dropna=True) < 5 or pair["target"].nunique(dropna=True) < 3:
                continue

            rank_ic = _safe_corr(pair["factor"].rank(), pair["target"].rank())
            pearson_ic = _safe_corr(pair["factor"], pair["target"])
            quantile_spread = _quantile_spread(pair)
            turnover = _factor_turnover(pair["factor"])
            stability = _rank_ic_stability(pair)
            rows.append(
                {
                    "feature": column,
                    "rows": int(len(pair)),
                    "rank_ic": _finite(rank_ic),
                    "pearson_ic": _finite(pearson_ic),
                    "quantile_spread_return": _finite(quantile_spread),
                    "top_quantile_turnover": _finite(turnover),
                    "rank_ic_stability": stability,
                    "score": _finite(abs(rank_ic) * math.sqrt(len(pair)) + abs(quantile_spread) * 10),
                }
            )
        result[str(horizon)] = sorted(rows, key=lambda row: row["score"], reverse=True)[:top_n]
    return result


def _quantile_spread(pair: pd.DataFrame) -> float:
    try:
        quantiles = pd.qcut(pair["factor"], q=5, labels=False, duplicates="drop")
    except ValueError:
        return 0.0
    frame = pair.assign(quantile=quantiles).dropna()
    if frame["quantile"].nunique() < 2:
        return 0.0
    low = frame.loc[frame["quantile"] == frame["quantile"].min(), "target"].mean()
    high = frame.loc[frame["quantile"] == frame["quantile"].max(), "target"].mean()
    return float(high - low)


def _factor_turnover(factor: pd.Series) -> float:
    try:
        quantiles = pd.qcut(factor, q=5, labels=False, duplicates="drop")
    except ValueError:
        return 0.0
    top = (quantiles == quantiles.max()).astype(float)
    if len(top) <= 1:
        return 0.0
    return float(top.diff().abs().fillna(0.0).mean())


def _rank_ic_stability(pair: pd.DataFrame, chunks: int = 4) -> dict[str, float]:
    if len(pair) < chunks * 20:
        return {"mean": 0.0, "std": 0.0, "positive_share": 0.0}
    values = []
    chunk_size = int(math.ceil(len(pair) / chunks))
    for start in range(0, len(pair), chunk_size):
        chunk = pair.iloc[start : start + chunk_size]
        if len(chunk) < 10 or chunk["factor"].nunique(dropna=True) < 3 or chunk["target"].nunique(dropna=True) < 3:
            continue
        value = _safe_corr(chunk["factor"].rank(), chunk["target"].rank())
        if np.isfinite(value):
            values.append(value)
    if not values:
        return {"mean": 0.0, "std": 0.0, "positive_share": 0.0}
    values_array = np.asarray(values, dtype=float)
    return {
        "mean": float(values_array.mean()),
        "std": float(values_array.std(ddof=1)) if len(values_array) > 1 else 0.0,
        "positive_share": float((values_array > 0).mean()),
    }


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left.rename("left"), right.rename("right")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["left"].nunique(dropna=True) < 2 or pair["right"].nunique(dropna=True) < 2:
        return 0.0
    value = pair["left"].corr(pair["right"])
    return float(value) if np.isfinite(value) else 0.0
