"""Constrained portfolio construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig


BUY = "BUY"
HOLD = "HOLD"
SELL = "SELL"
CASH = "CASH"


class PortfolioOptimizer:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def optimize(self, decisions: pd.DataFrame, sectors: dict[str, str] | None = None) -> pd.Series:
        candidates = decisions[decisions["decision"] == BUY].copy()
        tickers = sorted(set(decisions.index.get_level_values("ticker")).union({CASH}))
        weights = pd.Series(0.0, index=tickers)
        if candidates.empty:
            weights[CASH] = 1.0
            return weights

        risk = candidates["expected_volatility"].replace(0.0, np.nan).fillna(candidates["expected_volatility"].median())
        score = candidates["alpha_score"].clip(lower=0.0) + candidates["confidence"].clip(lower=0.0) * 0.05
        raw = (score / risk).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        if raw.sum() <= 0:
            weights[CASH] = 1.0
            return weights

        target = raw / raw.sum()
        target = self._cap_positions(target)
        target = self._cap_sectors(target, sectors)
        target = target[target >= 0.01]
        if target.empty or target.sum() <= 0:
            weights[CASH] = 1.0
            return weights
        annualized_vol = candidates.loc[target.index, "expected_volatility"].mean() * np.sqrt(
            252 / self.config.primary_horizon
        )
        scale = min(1.0, self.config.volatility_target / annualized_vol) if annualized_vol > 0 else 1.0
        target = target * scale

        for index, value in target.items():
            weights[index[1]] = float(value)
        weights[CASH] = max(0.0, 1.0 - weights.drop(CASH).sum())
        return weights.sort_values(ascending=False)

    def _cap_positions(self, weights: pd.Series) -> pd.Series:
        capped = weights.copy()
        for _ in range(10):
            over = capped > self.config.max_position_size
            if not over.any():
                break
            excess = (capped[over] - self.config.max_position_size).sum()
            capped[over] = self.config.max_position_size
            under = ~over
            if capped[under].sum() <= 0:
                break
            capped[under] += capped[under] / capped[under].sum() * excess
        return capped.clip(upper=self.config.max_position_size)

    def _cap_sectors(self, weights: pd.Series, sectors: dict[str, str] | None) -> pd.Series:
        if not sectors:
            return weights
        normalized = {ticker.upper(): sector for ticker, sector in sectors.items()}
        capped = weights.copy()
        for _ in range(10):
            sector_totals: dict[str, float] = {}
            for index, value in capped.items():
                sector = normalized.get(index[1], "Unknown")
                sector_totals[sector] = sector_totals.get(sector, 0.0) + float(value)
            over_sectors = {sector: value for sector, value in sector_totals.items() if value > self.config.max_sector_weight}
            if not over_sectors:
                break
            for sector, value in over_sectors.items():
                members = [index for index in capped.index if normalized.get(index[1], "Unknown") == sector]
                if capped.loc[members].sum() > 0:
                    capped.loc[members] *= self.config.max_sector_weight / value
        return capped
