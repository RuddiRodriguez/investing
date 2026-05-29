"""Tabular alpha model for forward excess-return forecasts."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


class TabularAlphaModel:
    """Leakage-safe cross-sectional model for multi-horizon excess returns."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.models: dict[int, Any] = {}
        self.errors: dict[int, float] = {}
        self.calibrators: dict[int, IsotonicRegression] = {}
        self.feature_columns: list[str] = []

    def fit(self, frame: pd.DataFrame) -> "TabularAlphaModel":
        target_columns = [f"target_excess_return_{horizon}d" for horizon in self.config.horizons]
        self.feature_columns = [
            column
            for column in frame.columns
            if not str(column).startswith("target_") and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if not self.feature_columns:
            raise ValueError("No numeric features were available for model training.")

        features = self._clean_features(frame[self.feature_columns])
        for horizon in self.config.horizons:
            target_name = f"target_excess_return_{horizon}d"
            target = pd.to_numeric(frame[target_name], errors="coerce")
            valid = target.notna()
            if valid.sum() < self.config.min_model_rows:
                self.models[horizon] = None
                self.errors[horizon] = float(target[valid].abs().median() if valid.any() else 0.03)
                continue
            x = features.loc[valid]
            y = target.loc[valid]
            if len(x) > self.config.max_model_rows:
                x = x.tail(self.config.max_model_rows)
                y = y.tail(self.config.max_model_rows)
            model = self._build_model(len(x))
            model.fit(x, y)
            fitted = pd.Series(model.predict(x), index=x.index)
            self.models[horizon] = model
            self.errors[horizon] = float(max(mean_absolute_error(y, fitted), y.abs().median() * 0.25, 0.005))
            self.calibrators[horizon] = self._fit_calibrator(fitted, y)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        features = self._clean_features(frame[self.feature_columns])
        output = pd.DataFrame(index=frame.index)
        confidence_parts = []
        for horizon in self.config.horizons:
            model = self.models.get(horizon)
            if model is None:
                prediction = pd.Series(0.0, index=frame.index)
                probability = pd.Series(0.5, index=frame.index)
            else:
                prediction = pd.Series(model.predict(features), index=frame.index)
                probability = self._probability(horizon, prediction)
            output[f"tabular_alpha_{horizon}d"] = prediction
            output[f"outperform_probability_{horizon}d"] = probability
            confidence_parts.append((probability - 0.5).abs() * 2.0)

        primary = self.config.primary_horizon
        output["expected_model_return"] = output[f"tabular_alpha_{primary}d"]
        output["outperform_probability"] = output[f"outperform_probability_{primary}d"]
        scale = max(self.errors.get(primary, 0.02), 0.005)
        output["tabular_signal"] = np.tanh(output["expected_model_return"] / scale)
        output["model_confidence"] = pd.concat(confidence_parts, axis=1).mean(axis=1).clip(0.0, 1.0)
        return output

    def _build_model(self, rows: int):
        if rows < 450:
            return make_pipeline(StandardScaler(), Ridge(alpha=5.0))
        return HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.04,
            max_leaf_nodes=24,
            l2_regularization=0.05,
            random_state=self.config.random_state,
        )

    def _clean_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        clean = frame.apply(pd.to_numeric, errors="coerce")
        clean = clean.replace([np.inf, -np.inf], np.nan)
        return clean.fillna(clean.median()).fillna(0.0)

    def _fit_calibrator(self, prediction: pd.Series, target: pd.Series) -> IsotonicRegression:
        if target.gt(0).nunique() < 2:
            return IsotonicRegression(out_of_bounds="clip").fit([0.0, 1.0], [0.5, 0.5])
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
            prediction,
            target.gt(0).astype(float),
        )

    def _probability(self, horizon: int, prediction: pd.Series) -> pd.Series:
        calibrator = self.calibrators.get(horizon)
        if calibrator is not None:
            return pd.Series(calibrator.predict(prediction), index=prediction.index).clip(0.0, 1.0)
        scale = max(self.errors.get(horizon, 0.02), 0.005)
        return pd.Series(1.0 / (1.0 + np.exp(-prediction / scale)), index=prediction.index)
