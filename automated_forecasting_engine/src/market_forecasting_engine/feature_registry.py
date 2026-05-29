from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


def build_feature_registry(
    features: pd.DataFrame,
    factor_evaluation: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Create auditable metadata for engineered features."""

    diagnostics_by_feature = _factor_diagnostics_by_feature(factor_evaluation or {})
    entries = []
    for column in features.columns:
        series = features[column]
        entries.append(
            {
                "name": str(column),
                "family": classify_feature_family(str(column)),
                "source": infer_feature_source(str(column)),
                "lookback_days": infer_lookback_days(str(column)),
                "availability": infer_feature_availability(str(column)),
                "missing_pct": float(series.isna().mean()) if len(series) else 0.0,
                "latest_is_available": bool(series.notna().iloc[-1]) if len(series) else False,
                "finite_count": int(np.isfinite(pd.to_numeric(series, errors="coerce")).sum()),
                "factor_diagnostics": diagnostics_by_feature.get(str(column), {}),
            }
        )

    family_counts = dict(Counter(entry["family"] for entry in entries))
    source_counts = dict(Counter(entry["source"] for entry in entries))
    return {
        "feature_count": int(len(entries)),
        "family_counts": family_counts,
        "source_counts": source_counts,
        "entries": entries,
    }


def classify_feature_family(name: str) -> str:
    clean = name.lower()
    if clean.startswith("relative_"):
        return "relative_strength"
    if clean.startswith("cs_") or clean.startswith("panel_cs_") or clean.startswith("exo_panel_cs_"):
        return "cross_sectional"
    if clean.startswith("structure_"):
        if "gap" in clean:
            return "chart_gap"
        if "pivot" in clean or "support" in clean or "resistance" in clean or "breakout" in clean or "breakdown" in clean:
            return "chart_structure"
        if "channel" in clean or "trend" in clean or "rectangle" in clean:
            return "trend_structure"
        return "chart_structure"
    if clean.startswith("volatility_") or "volatility" in clean or "atr" in clean or "drawdown" in clean:
        return "volatility_risk"
    if "dollar_volume" in clean or "illiquidity" in clean or "money_flow" in clean or "on_balance" in clean:
        return "liquidity_volume"
    if clean.startswith("volume_") or "volume" in clean:
        return "liquidity_volume"
    if clean.startswith("log_return") or clean.startswith("lagged_log_return"):
        return "returns"
    if clean.startswith("momentum") or clean in {"rsi_14", "macd", "macd_signal", "macd_hist"}:
        return "momentum"
    if "sma" in clean or "ema" in clean or "bollinger" in clean:
        return "trend_overlap"
    if clean.startswith("exo_"):
        return "external_context"
    if clean in {"day_of_week", "month", "is_month_end", "is_quarter_end"}:
        return "calendar"
    return "other"


def infer_feature_source(name: str) -> str:
    clean = name.lower()
    if clean.startswith(("relative_", "structure_", "volume_", "dollar_volume", "atr", "true_range")):
        return "ohlcv"
    if clean.startswith(("cs_", "panel_cs_", "exo_panel_cs_")):
        return "panel_universe"
    if clean.startswith("exo_"):
        return "external_context"
    if clean in {"day_of_week", "month", "is_month_end", "is_quarter_end"}:
        return "calendar"
    return "price"


def infer_feature_availability(name: str) -> str:
    clean = name.lower()
    if clean.startswith("exo_"):
        return "as_of_external_data_availability"
    if clean.startswith(("cs_", "panel_cs_", "exo_panel_cs_")):
        return "after_close_with_full_universe_bar"
    if clean in {"day_of_week", "month", "is_month_end", "is_quarter_end"}:
        return "session_calendar_known_before_open"
    return "after_close"


def infer_lookback_days(name: str) -> int | None:
    clean = name.lower()
    matches = re.findall(r"_(\d+)(?:d)?(?:_|$)", clean)
    if matches:
        return max(int(match) for match in matches)
    for token in ("52w", "252"):
        if token in clean:
            return 252
    return None


def _factor_diagnostics_by_feature(factor_evaluation: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    by_feature: dict[str, dict[str, Any]] = {}
    for horizon, rows in factor_evaluation.items():
        for row in rows:
            feature = str(row.get("feature"))
            by_feature.setdefault(feature, {})[str(horizon)] = {
                "rank_ic": row.get("rank_ic"),
                "pearson_ic": row.get("pearson_ic"),
                "quantile_spread_return": row.get("quantile_spread_return"),
                "top_quantile_turnover": row.get("top_quantile_turnover"),
                "rank_ic_stability": row.get("rank_ic_stability"),
                "score": row.get("score"),
            }
    return by_feature
