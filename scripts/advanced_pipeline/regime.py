"""Market regime classification."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import equal_weight_benchmark, validate_prices


class RegimeDetector:
    """Classify the current market environment from live price history."""

    def detect(self, prices: pd.DataFrame, benchmark: pd.Series | None = None) -> str:
        clean = validate_prices(prices)
        series = benchmark.reindex(clean.index).ffill() if benchmark is not None else equal_weight_benchmark(clean)
        returns = series.pct_change()
        return_20d = series.pct_change(20).iloc[-1]
        return_60d = series.pct_change(60).iloc[-1]
        volatility_20d = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        volatility_252d = returns.rolling(252).std().iloc[-1] * np.sqrt(252)
        breadth = clean.pct_change(20).iloc[-1].gt(0).mean()
        dispersion = clean.pct_change(20).iloc[-1].std()

        if pd.isna(return_60d) or pd.isna(volatility_20d):
            return "insufficient_history"
        if volatility_20d > max(0.30, volatility_252d * 1.5) and return_20d < 0:
            return "high_volatility_crash"
        if return_60d > 0.04 and breadth >= 0.60 and volatility_20d < 0.28:
            return "risk_on_growth"
        if return_60d < -0.03 or breadth <= 0.35:
            return "risk_off_defensive"
        if dispersion > 0.12 and 0.35 < breadth < 0.70:
            return "sector_rotation"
        return "neutral"

    def score(self, regime: str) -> float:
        scores = {
            "risk_on_growth": 0.75,
            "sector_rotation": 0.25,
            "neutral": 0.0,
            "risk_off_defensive": -0.45,
            "high_volatility_crash": -0.85,
            "insufficient_history": 0.0,
        }
        return scores.get(regime, 0.0)

    def expert_weights(self, regime: str) -> dict[str, float]:
        weights = {
            "risk_on_growth": {
                "tabular": 0.34,
                "technical": 0.30,
                "fundamental": 0.12,
                "news": 0.09,
                "graph": 0.10,
                "regime": 0.05,
            },
            "risk_off_defensive": {
                "tabular": 0.28,
                "technical": 0.14,
                "fundamental": 0.24,
                "news": 0.10,
                "graph": 0.08,
                "regime": 0.16,
            },
            "high_volatility_crash": {
                "tabular": 0.20,
                "technical": 0.10,
                "fundamental": 0.18,
                "news": 0.08,
                "graph": 0.06,
                "regime": 0.38,
            },
            "sector_rotation": {
                "tabular": 0.30,
                "technical": 0.22,
                "fundamental": 0.12,
                "news": 0.08,
                "graph": 0.23,
                "regime": 0.05,
            },
        }
        selected = weights.get(
            regime,
            {
                "tabular": 0.35,
                "technical": 0.25,
                "fundamental": 0.15,
                "news": 0.10,
                "graph": 0.10,
                "regime": 0.05,
            },
        )
        total = sum(selected.values())
        return {name: value / total for name, value in selected.items()}
