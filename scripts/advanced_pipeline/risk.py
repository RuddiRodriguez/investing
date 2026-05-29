"""Risk and uncertainty estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig


class RiskEngine:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def score(self, features: pd.DataFrame, alpha: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=features.index)
        volatility = features.get("volatility_60d", pd.Series(0.20, index=features.index)).fillna(0.20)
        downside = features.get("downside_volatility_60d", volatility).fillna(volatility)
        drawdown = features.get("drawdown_252d", pd.Series(0.0, index=features.index)).abs().fillna(0.0)
        var = features.get("var_95_252d", pd.Series(-0.02, index=features.index)).abs().fillna(0.02)
        es = features.get("expected_shortfall_95_252d", pd.Series(-0.03, index=features.index)).abs().fillna(var)
        horizon_scale = np.sqrt(self.config.primary_horizon / 252)

        output["expected_volatility"] = volatility * horizon_scale
        output["tail_loss"] = es
        output["drawdown_risk"] = drawdown
        output["uncertainty"] = (
            output["expected_volatility"] * 0.65
            + output["tail_loss"] * 0.25
            + alpha["model_confidence"].rsub(1.0).clip(0.0, 1.0) * 0.10
        ).clip(lower=0.0)
        output["risk_score"] = (
            (volatility / 0.60) * 0.35
            + (downside / 0.45) * 0.25
            + (drawdown / 0.50) * 0.20
            + (es / 0.08) * 0.20
        ).clip(0.0, 1.0)
        output["risk_penalty"] = output["risk_score"] * 0.04 + output["uncertainty"] * 0.35
        return output
