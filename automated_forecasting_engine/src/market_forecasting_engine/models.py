from __future__ import annotations

import copy
import importlib.util
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


class ForecastCandidate:
    name: str
    family: str

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "ForecastCandidate":
        raise NotImplementedError

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def clone(self) -> "ForecastCandidate":
        return copy.deepcopy(self)

    def parameter_count(self, n_features: int) -> int:
        return 1

    def parameters(self) -> dict[str, Any]:
        return {}

    def model_diagnostics(self) -> dict[str, Any]:
        return {}


class DropAllNaNColumns(BaseEstimator, TransformerMixin):
    def fit(self, X: Any, y: Any = None) -> "DropAllNaNColumns":
        frame = pd.DataFrame(X)
        self.input_columns_ = list(frame.columns)
        self.keep_columns_ = [column for column in frame.columns if frame[column].notna().any()]
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        frame = _frame_with_columns(X, getattr(self, "input_columns_", None))
        if not getattr(self, "keep_columns_", None):
            return pd.DataFrame(index=frame.index)
        return frame.reindex(columns=self.keep_columns_)


class DataFrameSimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy: str = "median") -> None:
        self.strategy = strategy

    def fit(self, X: Any, y: Any = None) -> "DataFrameSimpleImputer":
        frame = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)
        self.columns_ = list(frame.columns)
        self.imputer_ = SimpleImputer(strategy=self.strategy)
        self.imputer_.fit(frame)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        if not hasattr(self, "imputer_"):
            raise RuntimeError("DataFrameSimpleImputer has not been fitted.")
        frame = _frame_with_columns(X, self.columns_).replace([np.inf, -np.inf], np.nan)
        values = self.imputer_.transform(frame)
        columns = self.columns_
        if values.shape[1] != len(columns):
            columns = list(self.imputer_.get_feature_names_out(self.columns_))
        return pd.DataFrame(values, columns=columns, index=frame.index)


def _frame_with_columns(X: Any, columns: list[Any] | None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    if columns is not None:
        return pd.DataFrame(X, columns=columns)
    return pd.DataFrame(X)


def _pipeline_feature_names(estimator: Pipeline) -> list[Any]:
    for step_name in ("imputer", "drop_all_nan"):
        if step_name not in estimator.named_steps:
            continue
        step = estimator.named_steps[step_name]
        if hasattr(step, "get_feature_names_out"):
            try:
                return list(step.get_feature_names_out())
            except Exception:
                pass
        if hasattr(step, "columns_"):
            return list(step.columns_)
        if hasattr(step, "keep_columns_"):
            return list(step.keep_columns_)
    return []


@dataclass
class HistoricalMeanReturn(ForecastCandidate):
    name: str = "historical_mean_return"
    family: str = "statistical_baseline"
    mean_: float = 0.0

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "HistoricalMeanReturn":
        self.mean_ = float(target.mean())
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.mean_, dtype=float)

    def parameters(self) -> dict[str, Any]:
        return {"mean_return": self.mean_}


@dataclass
class RecentMeanReturn(ForecastCandidate):
    window: int = 20
    name: str = "recent_mean_return"
    family: str = "statistical_baseline"
    recent_mean_: float = 0.0

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "RecentMeanReturn":
        self.recent_mean_ = float(target.tail(self.window).mean())
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.recent_mean_, dtype=float)

    def parameters(self) -> dict[str, Any]:
        return {"window": self.window, "recent_mean_return": self.recent_mean_}


@dataclass
class ExponentialSmoothingReturn(ForecastCandidate):
    alpha: float = 0.20
    name: str = "exponential_smoothing_return"
    family: str = "state_space_baseline"
    smoothed_return_: float = 0.0

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "ExponentialSmoothingReturn":
        self.smoothed_return_ = float(target.ewm(alpha=self.alpha, adjust=False).mean().iloc[-1])
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.smoothed_return_, dtype=float)

    def parameters(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "smoothed_return": self.smoothed_return_}


class SklearnRegressorCandidate(ForecastCandidate):
    def __init__(
        self,
        name: str,
        family: str,
        estimator_factory: Callable[[], Any],
        parameter_count_hint: int = 20,
    ) -> None:
        self.name = name
        self.family = family
        self.estimator_factory = estimator_factory
        self.parameter_count_hint = parameter_count_hint
        self.estimator_: Any | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "SklearnRegressorCandidate":
        self.estimator_ = self.estimator_factory()
        self.estimator_.fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.estimator_ is None:
            raise RuntimeError(f"{self.name} has not been fitted.")
        return np.asarray(self.estimator_.predict(features), dtype=float)

    def parameter_count(self, n_features: int) -> int:
        if self.name.startswith(("ridge_", "lasso_", "elastic_net_")):
            return n_features + 1
        return self.parameter_count_hint

    def parameters(self) -> dict[str, Any]:
        estimator = self.estimator_
        if estimator is None:
            return {}
        final_step = estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
        params = final_step.get_params(deep=False)
        return {key: _json_safe(value) for key, value in params.items()}

    def model_diagnostics(self) -> dict[str, Any]:
        estimator = self.estimator_
        if estimator is None or not isinstance(estimator, Pipeline):
            return {}
        final_step = estimator.steps[-1][1]
        tree_diagnostics = _tree_model_diagnostics(estimator, final_step)
        if tree_diagnostics:
            return tree_diagnostics
        coefficients = getattr(final_step, "coef_", None)
        if coefficients is None:
            return {}
        coef = np.asarray(coefficients, dtype=float).reshape(-1)
        columns = _pipeline_feature_names(estimator)
        if len(columns) != len(coef):
            columns = [f"feature_{idx}" for idx in range(len(coef))]
        ranked = sorted(
            (
                {
                    "feature": str(column),
                    "coefficient": _json_safe(value),
                    "abs_coefficient": _json_safe(abs(value)),
                }
                for column, value in zip(columns, coef)
            ),
            key=lambda row: float(row["abs_coefficient"]),
            reverse=True,
        )
        nonzero = int(np.sum(np.abs(coef) > 1e-10))
        return {
            "linear_coefficients": {
                "coefficient_count": int(len(coef)),
                "nonzero_coefficient_count": nonzero,
                "sparsity_ratio": float(1.0 - nonzero / max(len(coef), 1)),
                "top_positive": sorted(ranked, key=lambda row: float(row["coefficient"]), reverse=True)[:10],
                "top_negative": sorted(ranked, key=lambda row: float(row["coefficient"]))[:10],
                "top_absolute": ranked[:15],
            }
        }


class OptunaTunedRegressorCandidate(ForecastCandidate):
    def __init__(
        self,
        model_kind: str,
        n_trials: int = 25,
        timeout_seconds: int | None = None,
        inner_splits: int = 3,
        random_state: int = 42,
    ) -> None:
        self.model_kind = model_kind
        self.name = f"optuna_{model_kind}"
        self.family = "machine_learning_tuned"
        self.n_trials = n_trials
        self.timeout_seconds = timeout_seconds
        self.inner_splits = inner_splits
        self.random_state = random_state
        self.estimator_: Any | None = None
        self.best_params_: dict[str, Any] = {}
        self.best_value_: float | None = None
        self.trial_count_: int = 0

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "OptunaTunedRegressorCandidate":
        optuna = _import_optuna()
        splits = _inner_time_series_splits(len(target), max_splits=self.inner_splits)
        if not splits:
            raise ValueError(f"{self.name} requires enough rows for inner time-series validation.")

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(trial: Any) -> float:
            params = _suggest_optuna_params(trial, self.model_kind)
            fold_scores = []
            for train_idx, validation_idx in splits:
                model = _optuna_estimator_pipeline(self.model_kind, params, random_state=self.random_state)
                train_features = features.iloc[train_idx]
                train_target = target.iloc[train_idx]
                validation_features = features.iloc[validation_idx]
                validation_target = target.iloc[validation_idx]
                try:
                    model.fit(train_features, train_target)
                    predicted = np.asarray(model.predict(validation_features), dtype=float)
                except Exception:
                    return float("inf")
                if len(predicted) != len(validation_target) or not np.isfinite(predicted).all():
                    return float("inf")
                fold_scores.append(float(np.mean(np.abs(validation_target.to_numpy(dtype=float) - predicted))))
            return float(np.mean(fold_scores)) if fold_scores else float("inf")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(objective, n_trials=max(1, self.n_trials), timeout=self.timeout_seconds, show_progress_bar=False)

        completed = [trial for trial in study.trials if trial.value is not None and np.isfinite(float(trial.value))]
        if not completed:
            raise ValueError(f"{self.name} did not complete any finite Optuna trials.")

        best = min(completed, key=lambda trial: float(trial.value))
        self.best_params_ = dict(best.params)
        self.best_value_ = float(best.value)
        self.trial_count_ = len(completed)
        self.estimator_ = _optuna_estimator_pipeline(self.model_kind, self.best_params_, random_state=self.random_state)
        self.estimator_.fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.estimator_ is None:
            raise RuntimeError(f"{self.name} has not been fitted.")
        return np.asarray(self.estimator_.predict(features), dtype=float)

    def parameter_count(self, n_features: int) -> int:
        if self.model_kind == "elastic_net":
            return n_features + 1
        n_estimators = self.best_params_.get("n_estimators")
        if n_estimators is not None:
            return int(n_estimators)
        return max(1, len(self.best_params_))

    def parameters(self) -> dict[str, Any]:
        return {
            "tuning": "optuna",
            "model_kind": self.model_kind,
            "n_trials_requested": self.n_trials,
            "trials_completed": self.trial_count_,
            "inner_splits": self.inner_splits,
            "inner_objective": "walk_forward_mae",
            "best_inner_mae": self.best_value_,
            "best_params": {key: _json_safe(value) for key, value in self.best_params_.items()},
        }

    def model_diagnostics(self) -> dict[str, Any]:
        return {
            "tuning": "optuna",
            "model_kind": self.model_kind,
            "trials_completed": self.trial_count_,
            "inner_splits": self.inner_splits,
            "best_inner_mae": self.best_value_,
            "holdout_policy": "Optuna objectives run only inside each fit slice; final holdout is never used for tuning.",
        }


@dataclass
class ARIMAReturnCandidate(ForecastCandidate):
    order: tuple[int, int, int] = (1, 0, 1)
    name: str = "arima"
    family: str = "classical_forecasting"
    result_: Any | None = None
    fallback_: float = 0.0

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "ARIMAReturnCandidate":
        from statsmodels.tsa.arima.model import ARIMA

        series = _target_series(target)
        self.fallback_ = float(series.tail(20).mean())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result_ = ARIMA(
                series,
                order=self.order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.result_ is None:
            return np.full(len(features), self.fallback_, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.result_.forecast(steps=len(features))
        return _clean_predictions(forecast, len(features), fallback=self.fallback_)

    def parameter_count(self, n_features: int) -> int:
        return sum(self.order) + 1

    def parameters(self) -> dict[str, Any]:
        return {"order": list(self.order), "fallback_return": self.fallback_}


@dataclass
class SARIMAReturnCandidate(ARIMAReturnCandidate):
    order: tuple[int, int, int] = (1, 0, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 5)
    name: str = "sarima"
    family: str = "classical_forecasting"

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "SARIMAReturnCandidate":
        from statsmodels.tsa.arima.model import ARIMA

        series = _target_series(target)
        self.fallback_ = float(series.tail(20).mean())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result_ = ARIMA(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()
        return self

    def parameter_count(self, n_features: int) -> int:
        return sum(self.order) + sum(self.seasonal_order[:3]) + 1

    def parameters(self) -> dict[str, Any]:
        return {
            "order": list(self.order),
            "seasonal_order": list(self.seasonal_order),
            "fallback_return": self.fallback_,
        }


@dataclass
class GARCHReturnCandidate(ForecastCandidate):
    p: int = 1
    q: int = 1
    mean_lags: int = 1
    name: str = "garch"
    family: str = "volatility_model"
    result_: Any | None = None
    fallback_: float = 0.0
    one_step_volatility_: float = 0.0
    annualized_volatility_: float = 0.0

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "GARCHReturnCandidate":
        from arch import arch_model

        series = _target_series(target) * 100.0
        self.fallback_ = float(series.tail(20).mean() / 100.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(
                series,
                mean="AR",
                lags=self.mean_lags,
                vol="GARCH",
                p=self.p,
                q=self.q,
                rescale=False,
            )
            self.result_ = model.fit(disp="off", show_warning=False)
            forecast = self.result_.forecast(horizon=1, reindex=False)
            variance = float(forecast.variance.iloc[-1, 0])
            self.one_step_volatility_ = float(np.sqrt(max(variance, 0.0)) / 100.0)
            self.annualized_volatility_ = float(self.one_step_volatility_ * np.sqrt(252))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.result_ is None:
            return np.full(len(features), self.fallback_, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.result_.forecast(horizon=len(features), reindex=False)
        predicted = np.asarray(forecast.mean.iloc[-1], dtype=float) / 100.0
        return _clean_predictions(predicted, len(features), fallback=self.fallback_)

    def parameter_count(self, n_features: int) -> int:
        return self.p + self.q + self.mean_lags + 2

    def parameters(self) -> dict[str, Any]:
        return {
            "p": self.p,
            "q": self.q,
            "mean_lags": self.mean_lags,
            "fallback_return": self.fallback_,
            "one_step_volatility": self.one_step_volatility_,
            "annualized_volatility": self.annualized_volatility_,
        }

    def model_diagnostics(self) -> dict[str, Any]:
        return {
            "volatility_model": {
                "one_step_volatility": self.one_step_volatility_,
                "annualized_volatility": self.annualized_volatility_,
            }
        }


@dataclass
class VARReturnCandidate(ForecastCandidate):
    maxlags: int = 2
    feature_limit: int = 3
    name: str = "var"
    family: str = "multivariate_time_series"
    result_: Any | None = None
    lag_order_: int = 1
    fallback_: float = 0.0
    selected_feature_columns_: list[str] | None = None
    last_values_: np.ndarray | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "VARReturnCandidate":
        from statsmodels.tsa.api import VAR

        frame = self._training_frame(features, target)
        self.fallback_ = float(frame["target"].tail(20).mean())
        if len(frame) < max(20, self.maxlags * 8):
            raise ValueError("Not enough rows for VAR candidate.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result_ = VAR(frame).fit(maxlags=self.maxlags, ic=None, trend="c")
        self.lag_order_ = int(max(getattr(self.result_, "k_ar", 1), 1))
        self.last_values_ = frame.tail(self.lag_order_).to_numpy(dtype=float)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.result_ is None or self.last_values_ is None:
            return np.full(len(features), self.fallback_, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.result_.forecast(self.last_values_, steps=len(features))
        return _clean_predictions(forecast[:, 0], len(features), fallback=self.fallback_)

    def parameter_count(self, n_features: int) -> int:
        series_count = 1 + len(self.selected_feature_columns_ or [])
        return series_count * series_count * self.maxlags + series_count

    def parameters(self) -> dict[str, Any]:
        return {
            "maxlags": self.maxlags,
            "lag_order": self.lag_order_,
            "selected_feature_columns": self.selected_feature_columns_ or [],
            "fallback_return": self.fallback_,
        }

    def _training_frame(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        preferred_columns = [
            column
            for column in (
                "log_return_1d",
                "momentum_5d",
                "volatility_20d",
                "rsi_14",
                "volume_change_1d",
            )
            if column in features.columns
        ]
        external_columns = [column for column in features.columns if column.startswith("exo_")]
        candidate_columns = preferred_columns + external_columns
        correlations = {}
        for column in candidate_columns:
            pair = pd.concat([target.rename("target"), features[column]], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) >= 20:
                correlations[column] = abs(float(pair["target"].corr(pair[column])))
        usable_columns = [
            column
            for column, _ in sorted(correlations.items(), key=lambda item: (np.nan_to_num(item[1]), item[0]), reverse=True)
            if np.isfinite(correlations[column])
        ][: self.feature_limit]
        if not usable_columns:
            usable_columns = preferred_columns[: self.feature_limit]
        self.selected_feature_columns_ = usable_columns
        frame = pd.concat([target.rename("target"), features[usable_columns]], axis=1)
        frame = frame.replace([np.inf, -np.inf], np.nan).ffill().dropna()
        return frame


@dataclass
class KalmanFilterReturnCandidate(ForecastCandidate):
    process_noise_scale: float = 0.03
    name: str = "kalman_filter"
    family: str = "state_space_model"
    level_: float = 0.0
    slope_: float = 0.0
    observation_variance_: float = 1e-4
    process_variance_: float = 1e-5

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "KalmanFilterReturnCandidate":
        series = _target_series(target)
        values = series.to_numpy(dtype=float)
        if len(values) == 0:
            self.level_ = 0.0
            self.slope_ = 0.0
            return self

        diffs = np.diff(values)
        obs_var = float(np.var(values, ddof=1)) if len(values) > 1 else 1e-4
        diff_var = float(np.var(diffs, ddof=1)) if len(diffs) > 1 else obs_var * 0.1
        self.observation_variance_ = max(obs_var, 1e-8)
        self.process_variance_ = max(diff_var * self.process_noise_scale, 1e-10)

        transition = np.array([[1.0, 1.0], [0.0, 1.0]])
        observation = np.array([[1.0, 0.0]])
        process_noise = np.array(
            [[self.process_variance_, 0.0], [0.0, self.process_variance_ * 0.25]],
            dtype=float,
        )
        observation_noise = np.array([[self.observation_variance_]], dtype=float)
        state = np.array([values[0], 0.0], dtype=float)
        covariance = np.eye(2, dtype=float) * self.observation_variance_

        for value in values:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process_noise
            residual = np.array([[value]], dtype=float) - observation @ state.reshape(-1, 1)
            residual_covariance = observation @ covariance @ observation.T + observation_noise
            kalman_gain = covariance @ observation.T @ np.linalg.inv(residual_covariance)
            state = state + (kalman_gain @ residual).ravel()
            covariance = (np.eye(2) - kalman_gain @ observation) @ covariance

        self.level_ = float(state[0])
        self.slope_ = float(state[1])
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        predictions = []
        level = self.level_
        slope = self.slope_
        for _ in range(len(features)):
            level += slope
            predictions.append(level)
        return np.asarray(predictions, dtype=float)

    def parameter_count(self, n_features: int) -> int:
        return 4

    def parameters(self) -> dict[str, Any]:
        return {
            "process_noise_scale": self.process_noise_scale,
            "level": self.level_,
            "slope": self.slope_,
            "observation_variance": self.observation_variance_,
            "process_variance": self.process_variance_,
        }


@dataclass
class LSTMReturnCandidate(ForecastCandidate):
    sequence_length: int = 20
    hidden_size: int = 12
    max_epochs: int = 25
    learning_rate: float = 0.01
    min_rows: int = 120
    device: str = "auto"
    random_state: int = 42
    name: str = "lstm"
    family: str = "deep_learning"
    model_: Any | None = None
    target_mean_: float = 0.0
    target_std_: float = 1.0
    prediction_clip_low_: float = -0.25
    prediction_clip_high_: float = 0.25
    last_sequence_: np.ndarray | None = None
    fallback_: float = 0.0
    device_used_: str = "unavailable"
    epochs_run_: int = 0

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "LSTMReturnCandidate":
        import torch
        from torch import nn

        torch.manual_seed(self.random_state)
        torch.set_num_threads(1)

        values = _target_series(target).to_numpy(dtype=np.float32)
        self.fallback_ = float(np.mean(values[-20:])) if len(values) else 0.0
        if len(values) < self.min_rows:
            raise ValueError(f"LSTM candidate requires at least {self.min_rows} rows.")
        if len(values) <= self.sequence_length + 5:
            self.last_sequence_ = values[-self.sequence_length :]
            return self

        self.target_mean_ = float(np.mean(values))
        self.target_std_ = float(max(np.std(values), 1e-6))
        quantile_low = float(np.quantile(values, 0.01))
        quantile_high = float(np.quantile(values, 0.99))
        clip_width = max(5.0 * self.target_std_, 0.02)
        self.prediction_clip_low_ = float(max(quantile_low - clip_width, -0.50))
        self.prediction_clip_high_ = float(min(quantile_high + clip_width, 0.50))
        scaled = ((values - self.target_mean_) / self.target_std_).astype(np.float32)
        x_train = []
        y_train = []
        for index in range(self.sequence_length, len(scaled)):
            x_train.append(scaled[index - self.sequence_length : index])
            y_train.append(scaled[index])

        device = _torch_device(self.device)
        self.device_used_ = str(device)
        x_tensor = torch.tensor(np.asarray(x_train), dtype=torch.float32, device=device).unsqueeze(-1)
        y_tensor = torch.tensor(np.asarray(y_train), dtype=torch.float32, device=device).unsqueeze(-1)

        class _TinyLSTM(nn.Module):
            def __init__(self, hidden_size: int) -> None:
                super().__init__()
                self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
                self.output = nn.Linear(hidden_size, 1)

            def forward(self, data: Any) -> Any:
                encoded, _ = self.lstm(data)
                return self.output(encoded[:, -1, :])

        self.model_ = _TinyLSTM(self.hidden_size).to(device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_function = nn.MSELoss()
        self.model_.train()
        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            prediction = self.model_(x_tensor)
            loss = loss_function(prediction, y_tensor)
            loss.backward()
            optimizer.step()
            self.epochs_run_ = epoch + 1

        self.last_sequence_ = scaled[-self.sequence_length :]
        self.model_.to("cpu")
        self.device_used_ = f"{self.device_used_}->cpu_for_prediction"
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.last_sequence_ is None or len(self.last_sequence_) < self.sequence_length:
            return np.full(len(features), self.fallback_, dtype=float)

        import torch

        sequence = self.last_sequence_.astype(np.float32).copy()
        predictions = []
        self.model_.eval()
        with torch.no_grad():
            for _ in range(len(features)):
                x_tensor = torch.tensor(sequence[-self.sequence_length :], dtype=torch.float32).reshape(1, -1, 1)
                scaled_prediction = float(self.model_(x_tensor).item())
                prediction = scaled_prediction * self.target_std_ + self.target_mean_
                predictions.append(float(np.clip(prediction, self.prediction_clip_low_, self.prediction_clip_high_)))
                sequence = np.append(sequence, scaled_prediction).astype(np.float32)
        return _clean_predictions(predictions, len(features), fallback=self.fallback_)

    def parameter_count(self, n_features: int) -> int:
        return self.hidden_size * 4 + self.hidden_size + 1

    def parameters(self) -> dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "min_rows": self.min_rows,
            "device": self.device,
            "device_used": self.device_used_,
            "epochs_run": self.epochs_run_,
            "target_mean": self.target_mean_,
            "target_std": self.target_std_,
            "prediction_clip_low": self.prediction_clip_low_,
            "prediction_clip_high": self.prediction_clip_high_,
            "fallback_return": self.fallback_,
        }

    def model_diagnostics(self) -> dict[str, Any]:
        return {
            "chapter_17_deep_learning": {
                "architecture": "target_sequence_lstm",
                "device_used": self.device_used_,
                "epochs_run": int(self.epochs_run_),
            }
        }


@dataclass
class TabularMLPReturnCandidate(ForecastCandidate):
    hidden_size: int = 24
    max_epochs: int = 30
    learning_rate: float = 0.003
    weight_decay: float = 1e-4
    dropout: float = 0.15
    validation_fraction: float = 0.20
    patience: int = 5
    min_rows: int = 140
    time_budget_seconds: float = 12.0
    device: str = "auto"
    random_state: int = 42
    name: str = "tabular_mlp_fast"
    family: str = "deep_learning"
    model_: Any | None = None
    feature_columns_: list[Any] | None = None
    feature_median_: pd.Series | None = None
    feature_mean_: pd.Series | None = None
    feature_std_: pd.Series | None = None
    target_mean_: float = 0.0
    target_std_: float = 1.0
    prediction_clip_low_: float = -0.25
    prediction_clip_high_: float = 0.25
    fallback_: float = 0.0
    best_validation_loss_: float | None = None
    final_train_loss_: float | None = None
    epochs_run_: int = 0
    stopped_reason_: str = "not_fitted"
    device_used_: str = "unavailable"

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "TabularMLPReturnCandidate":
        import torch
        from torch import nn

        torch.manual_seed(self.random_state)
        torch.set_num_threads(1)

        clean_target = _target_series(target).astype(float)
        if len(clean_target) < self.min_rows:
            raise ValueError(f"{self.name} requires at least {self.min_rows} rows.")
        self.fallback_ = float(clean_target.tail(20).mean()) if len(clean_target) else 0.0
        self.feature_columns_ = list(features.columns)
        frame = pd.DataFrame(features).replace([np.inf, -np.inf], np.nan)
        self.feature_median_ = frame.median(numeric_only=True).reindex(frame.columns).fillna(0.0)
        imputed = frame.fillna(self.feature_median_)
        self.feature_mean_ = imputed.mean(numeric_only=True).reindex(frame.columns).fillna(0.0)
        self.feature_std_ = imputed.std(numeric_only=True).reindex(frame.columns).replace(0.0, 1.0).fillna(1.0)
        x_values = ((imputed - self.feature_mean_) / self.feature_std_).to_numpy(dtype=np.float32)
        y_values = clean_target.to_numpy(dtype=np.float32)
        self.target_mean_ = float(np.mean(y_values))
        self.target_std_ = float(max(np.std(y_values), 1e-6))
        quantile_low = float(np.quantile(y_values, 0.01))
        quantile_high = float(np.quantile(y_values, 0.99))
        clip_width = max(5.0 * self.target_std_, 0.02)
        self.prediction_clip_low_ = float(max(quantile_low - clip_width, -0.50))
        self.prediction_clip_high_ = float(min(quantile_high + clip_width, 0.50))
        y_scaled = ((y_values - self.target_mean_) / self.target_std_).astype(np.float32)

        validation_rows = max(10, int(round(len(y_scaled) * self.validation_fraction)))
        validation_rows = min(validation_rows, max(10, len(y_scaled) // 3))
        train_stop = len(y_scaled) - validation_rows
        if train_stop < 50:
            raise ValueError(f"{self.name} requires at least 50 training rows after validation split.")

        device = _torch_device(self.device)
        self.device_used_ = str(device)
        x_train = torch.tensor(x_values[:train_stop], dtype=torch.float32, device=device)
        y_train = torch.tensor(y_scaled[:train_stop], dtype=torch.float32, device=device).unsqueeze(-1)
        x_validation = torch.tensor(x_values[train_stop:], dtype=torch.float32, device=device)
        y_validation = torch.tensor(y_scaled[train_stop:], dtype=torch.float32, device=device).unsqueeze(-1)

        class _TabularMLP(nn.Module):
            def __init__(self, n_features: int, hidden_size: int, dropout: float) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_features, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, max(4, hidden_size // 2)),
                    nn.ReLU(),
                    nn.Linear(max(4, hidden_size // 2), 1),
                )

            def forward(self, data: Any) -> Any:
                return self.net(data)

        self.model_ = _TabularMLP(x_train.shape[1], self.hidden_size, self.dropout).to(device)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_function = nn.MSELoss()
        best_state = copy.deepcopy(self.model_.state_dict())
        best_loss = float("inf")
        stale_epochs = 0
        started = time.monotonic()
        self.model_.train()
        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            train_prediction = self.model_(x_train)
            train_loss = loss_function(train_prediction, y_train)
            train_loss.backward()
            optimizer.step()
            self.final_train_loss_ = float(train_loss.detach().cpu().item())

            self.model_.eval()
            with torch.no_grad():
                validation_loss = float(loss_function(self.model_(x_validation), y_validation).detach().cpu().item())
            self.model_.train()
            self.epochs_run_ = epoch + 1
            if validation_loss + 1e-7 < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.patience:
                self.stopped_reason_ = "early_stopping"
                break
            if time.monotonic() - started >= self.time_budget_seconds:
                self.stopped_reason_ = "time_budget"
                break
        else:
            self.stopped_reason_ = "max_epochs"

        self.model_.load_state_dict(best_state)
        self.best_validation_loss_ = float(best_loss) if np.isfinite(best_loss) else None
        self.model_.to("cpu")
        self.device_used_ = f"{self.device_used_}->cpu_for_prediction"
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.feature_columns_ is None:
            return np.full(len(features), self.fallback_, dtype=float)
        import torch

        frame = pd.DataFrame(features).reindex(columns=self.feature_columns_).replace([np.inf, -np.inf], np.nan)
        imputed = frame.fillna(self.feature_median_)
        scaled = ((imputed - self.feature_mean_) / self.feature_std_).to_numpy(dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            raw = self.model_(torch.tensor(scaled, dtype=torch.float32)).cpu().numpy().reshape(-1)
        predictions = raw * self.target_std_ + self.target_mean_
        predictions = np.clip(predictions, self.prediction_clip_low_, self.prediction_clip_high_)
        return _clean_predictions(predictions, len(features), fallback=self.fallback_)

    def parameter_count(self, n_features: int) -> int:
        hidden_2 = max(4, self.hidden_size // 2)
        return int((n_features + 1) * self.hidden_size + (self.hidden_size + 1) * hidden_2 + hidden_2 + 1)

    def parameters(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "validation_fraction": self.validation_fraction,
            "patience": self.patience,
            "min_rows": self.min_rows,
            "time_budget_seconds": self.time_budget_seconds,
            "device": self.device,
            "device_used": self.device_used_,
            "epochs_run": self.epochs_run_,
            "stopped_reason": self.stopped_reason_,
            "best_validation_loss": self.best_validation_loss_,
            "final_train_loss": self.final_train_loss_,
            "target_mean": self.target_mean_,
            "target_std": self.target_std_,
            "prediction_clip_low": self.prediction_clip_low_,
            "prediction_clip_high": self.prediction_clip_high_,
            "fallback_return": self.fallback_,
        }

    def model_diagnostics(self) -> dict[str, Any]:
        return {
            "chapter_17_deep_learning": {
                "architecture": "bounded_tabular_mlp",
                "device_used": self.device_used_,
                "epochs_run": int(self.epochs_run_),
                "stopped_reason": self.stopped_reason_,
                "best_validation_loss": self.best_validation_loss_,
                "final_train_loss": self.final_train_loss_,
            }
        }


@dataclass
class TemporalCNNReturnCandidate(ForecastCandidate):
    lookback: int = 30
    channels: int = 16
    kernel_size: int = 5
    max_epochs: int = 45
    learning_rate: float = 0.002
    weight_decay: float = 1e-4
    dropout: float = 0.15
    validation_fraction: float = 0.20
    patience: int = 6
    min_rows: int = 220
    time_budget_seconds: float = 30.0
    device: str = "auto"
    random_state: int = 42
    name: str = "temporal_cnn_research"
    family: str = "deep_learning"
    model_: Any | None = None
    feature_columns_: list[Any] | None = None
    feature_median_: pd.Series | None = None
    feature_mean_: pd.Series | None = None
    feature_std_: pd.Series | None = None
    last_feature_window_: np.ndarray | None = None
    target_mean_: float = 0.0
    target_std_: float = 1.0
    prediction_clip_low_: float = -0.25
    prediction_clip_high_: float = 0.25
    fallback_: float = 0.0
    best_validation_loss_: float | None = None
    final_train_loss_: float | None = None
    epochs_run_: int = 0
    stopped_reason_: str = "not_fitted"
    device_used_: str = "unavailable"

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "TemporalCNNReturnCandidate":
        import torch
        from torch import nn

        torch.manual_seed(self.random_state)
        torch.set_num_threads(1)

        clean_target = _target_series(target).astype(float)
        if len(clean_target) < self.min_rows:
            raise ValueError(f"{self.name} requires at least {self.min_rows} rows.")
        if len(clean_target) <= self.lookback + 20:
            raise ValueError(f"{self.name} requires enough rows for {self.lookback}-bar windows.")

        self.fallback_ = float(clean_target.tail(20).mean()) if len(clean_target) else 0.0
        self.feature_columns_ = list(features.columns)
        frame = pd.DataFrame(features).replace([np.inf, -np.inf], np.nan)
        self.feature_median_ = frame.median(numeric_only=True).reindex(frame.columns).fillna(0.0)
        imputed = frame.fillna(self.feature_median_)
        self.feature_mean_ = imputed.mean(numeric_only=True).reindex(frame.columns).fillna(0.0)
        self.feature_std_ = imputed.std(numeric_only=True).reindex(frame.columns).replace(0.0, 1.0).fillna(1.0)
        x_values = ((imputed - self.feature_mean_) / self.feature_std_).to_numpy(dtype=np.float32)
        y_values = clean_target.to_numpy(dtype=np.float32)
        self.target_mean_ = float(np.mean(y_values))
        self.target_std_ = float(max(np.std(y_values), 1e-6))
        quantile_low = float(np.quantile(y_values, 0.01))
        quantile_high = float(np.quantile(y_values, 0.99))
        clip_width = max(5.0 * self.target_std_, 0.02)
        self.prediction_clip_low_ = float(max(quantile_low - clip_width, -0.50))
        self.prediction_clip_high_ = float(min(quantile_high + clip_width, 0.50))
        y_scaled = ((y_values - self.target_mean_) / self.target_std_).astype(np.float32)

        windows = []
        labels = []
        for idx in range(self.lookback, len(x_values)):
            windows.append(x_values[idx - self.lookback : idx].T)
            labels.append(y_scaled[idx])
        x_window = np.asarray(windows, dtype=np.float32)
        y_window = np.asarray(labels, dtype=np.float32)
        validation_rows = max(10, int(round(len(y_window) * self.validation_fraction)))
        validation_rows = min(validation_rows, max(10, len(y_window) // 3))
        train_stop = len(y_window) - validation_rows
        if train_stop < 50:
            raise ValueError(f"{self.name} requires at least 50 training windows after validation split.")

        device = _torch_device(self.device)
        self.device_used_ = str(device)
        x_train = torch.tensor(x_window[:train_stop], dtype=torch.float32, device=device)
        y_train = torch.tensor(y_window[:train_stop], dtype=torch.float32, device=device).unsqueeze(-1)
        x_validation = torch.tensor(x_window[train_stop:], dtype=torch.float32, device=device)
        y_validation = torch.tensor(y_window[train_stop:], dtype=torch.float32, device=device).unsqueeze(-1)

        class _TemporalCNN(nn.Module):
            def __init__(self, n_features: int, channels: int, kernel_size: int, dropout: float) -> None:
                super().__init__()
                padding = max(kernel_size // 2, 0)
                self.net = nn.Sequential(
                    nn.Conv1d(n_features, channels, kernel_size=kernel_size, padding=padding),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(channels, 1),
                )

            def forward(self, data: Any) -> Any:
                return self.net(data)

        self.model_ = _TemporalCNN(x_train.shape[1], self.channels, self.kernel_size, self.dropout).to(device)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_function = nn.MSELoss()
        best_state = copy.deepcopy(self.model_.state_dict())
        best_loss = float("inf")
        stale_epochs = 0
        started = time.monotonic()
        self.model_.train()
        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            train_prediction = self.model_(x_train)
            train_loss = loss_function(train_prediction, y_train)
            train_loss.backward()
            optimizer.step()
            self.final_train_loss_ = float(train_loss.detach().cpu().item())

            self.model_.eval()
            with torch.no_grad():
                validation_loss = float(loss_function(self.model_(x_validation), y_validation).detach().cpu().item())
            self.model_.train()
            self.epochs_run_ = epoch + 1
            if validation_loss + 1e-7 < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.patience:
                self.stopped_reason_ = "early_stopping"
                break
            if time.monotonic() - started >= self.time_budget_seconds:
                self.stopped_reason_ = "time_budget"
                break
        else:
            self.stopped_reason_ = "max_epochs"

        self.model_.load_state_dict(best_state)
        self.best_validation_loss_ = float(best_loss) if np.isfinite(best_loss) else None
        self.last_feature_window_ = x_values[-self.lookback :].copy()
        self.model_.to("cpu")
        self.device_used_ = f"{self.device_used_}->cpu_for_prediction"
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.feature_columns_ is None:
            return np.full(len(features), self.fallback_, dtype=float)
        import torch

        frame = pd.DataFrame(features).reindex(columns=self.feature_columns_).replace([np.inf, -np.inf], np.nan)
        imputed = frame.fillna(self.feature_median_)
        scaled = ((imputed - self.feature_mean_) / self.feature_std_).to_numpy(dtype=np.float32)
        windows = self._prediction_windows(scaled)
        self.model_.eval()
        with torch.no_grad():
            raw = self.model_(torch.tensor(windows, dtype=torch.float32)).cpu().numpy().reshape(-1)
        predictions = raw * self.target_std_ + self.target_mean_
        predictions = np.clip(predictions, self.prediction_clip_low_, self.prediction_clip_high_)
        return _clean_predictions(predictions, len(features), fallback=self.fallback_)

    def _prediction_windows(self, scaled: np.ndarray) -> np.ndarray:
        if len(scaled) == 0:
            return np.empty((0, 0, self.lookback), dtype=np.float32)
        history = self.last_feature_window_
        if history is None or len(history) < self.lookback:
            pad = np.repeat(scaled[[0]], self.lookback, axis=0)
            history = pad.astype(np.float32)
        combined = np.vstack([history, scaled]).astype(np.float32)
        windows = []
        for idx in range(len(scaled)):
            end = len(history) + idx
            windows.append(combined[end - self.lookback : end].T)
        return np.asarray(windows, dtype=np.float32)

    def parameter_count(self, n_features: int) -> int:
        first = (n_features * self.kernel_size + 1) * self.channels
        second = (self.channels * 3 + 1) * self.channels
        dense = self.channels + 1
        return int(first + second + dense)

    def parameters(self) -> dict[str, Any]:
        return {
            "lookback": self.lookback,
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "validation_fraction": self.validation_fraction,
            "patience": self.patience,
            "min_rows": self.min_rows,
            "time_budget_seconds": self.time_budget_seconds,
            "device": self.device,
            "device_used": self.device_used_,
            "epochs_run": self.epochs_run_,
            "stopped_reason": self.stopped_reason_,
            "best_validation_loss": self.best_validation_loss_,
            "final_train_loss": self.final_train_loss_,
            "prediction_clip_low": self.prediction_clip_low_,
            "prediction_clip_high": self.prediction_clip_high_,
            "fallback_return": self.fallback_,
        }

    def model_diagnostics(self) -> dict[str, Any]:
        return {
            "chapter_18_cnn": {
                "architecture": "temporal_conv1d",
                "device_used": self.device_used_,
                "epochs_run": int(self.epochs_run_),
                "stopped_reason": self.stopped_reason_,
                "best_validation_loss": self.best_validation_loss_,
                "final_train_loss": self.final_train_loss_,
                "trading_gate": False,
            }
        }


def default_candidates(
    random_state: int = 42,
    include_lightgbm: bool = True,
    include_statistical_models: bool = True,
    include_lstm: bool = False,
    deep_learning_profile: str = "off",
    search_level: str = "fast",
    tuning_mode: str = "fixed",
    optuna_trials: int = 25,
    optuna_timeout_seconds: int | None = None,
    optuna_inner_splits: int = 3,
    optuna_families: tuple[str, ...] = ("lightgbm", "xgboost", "elastic_net", "random_forest", "extra_trees", "gradient_boosting"),
) -> list[ForecastCandidate]:
    if tuning_mode not in {"fixed", "optuna"}:
        raise ValueError("tuning_mode must be `fixed` or `optuna`.")
    if deep_learning_profile not in {"off", "fast", "research"}:
        raise ValueError("deep_learning_profile must be `off`, `fast`, or `research`.")
    if search_level not in {"realtime", "fast", "expanded", "medium", "broad"}:
        raise ValueError("search_level must be `realtime`, `fast`, `expanded`, `medium`, or `broad`.")
    if search_level in {"medium", "broad"}:
        search_level = "expanded"
    realtime = search_level == "realtime"
    expanded = search_level == "expanded"
    candidates: list[ForecastCandidate] = [
        HistoricalMeanReturn(),
        RecentMeanReturn(window=10, name="recent_mean_return_10d"),
        RecentMeanReturn(window=20),
        RecentMeanReturn(window=60, name="recent_mean_return_60d"),
        ExponentialSmoothingReturn(alpha=0.05, name="exponential_smoothing_return_alpha_0_05"),
        ExponentialSmoothingReturn(alpha=0.15),
        ExponentialSmoothingReturn(alpha=0.35, name="exponential_smoothing_return_alpha_0_35"),
        KalmanFilterReturnCandidate(),
    ]
    candidates.extend(
        [
            _ridge_candidate("ridge_alpha_1_0", alpha=1.0),
            _lasso_candidate("lasso_alpha_0_0001", alpha=0.0001, random_state=random_state),
            _elastic_net_candidate("elastic_net_alpha_0_0005_l1_0_25", alpha=0.0005, l1_ratio=0.25, random_state=random_state),
            _decision_tree_candidate(
                "decision_tree_shallow",
                max_depth=3,
                min_samples_leaf=8,
                random_state=random_state,
            ),
        ]
    )
    if realtime:
        if include_statistical_models:
            candidates.extend(_optional_statistical_candidates(search_level=search_level))
        return candidates

    candidates.extend(
        [
            _random_forest_candidate(
                "random_forest_balanced",
                n_estimators=180,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=random_state,
            ),
            _extra_trees_candidate(
                "extra_trees_balanced",
                n_estimators=180,
                min_samples_leaf=6,
                max_features="sqrt",
                random_state=random_state,
            ),
            _gradient_boosting_candidate(
                "gradient_boosting_conservative",
                n_estimators=120,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=6,
                random_state=random_state,
            ),
            _hist_gradient_boosting_candidate(
                "hist_gradient_boosting_conservative",
                max_iter=120,
                learning_rate=0.03,
                max_leaf_nodes=15,
                min_samples_leaf=12,
                random_state=random_state,
            ),
        ]
    )
    if expanded:
        candidates.extend(
            [
                _ridge_candidate("ridge_alpha_10_0", alpha=10.0),
                _ridge_candidate("ridge_alpha_100_0", alpha=100.0),
                _lasso_candidate("lasso_alpha_0_00001", alpha=0.00001, random_state=random_state),
                _lasso_candidate("lasso_alpha_0_001", alpha=0.001, random_state=random_state),
                _elastic_net_candidate("elastic_net_alpha_0_0001_l1_0_10", alpha=0.0001, l1_ratio=0.10, random_state=random_state),
                _elastic_net_candidate("elastic_net_alpha_0_001_l1_0_60", alpha=0.001, l1_ratio=0.60, random_state=random_state),
                _decision_tree_candidate(
                    "decision_tree_medium",
                    max_depth=5,
                    min_samples_leaf=12,
                    random_state=random_state,
                ),
                _random_forest_candidate(
                    "random_forest_smoother",
                    n_estimators=220,
                    min_samples_leaf=10,
                    max_features=0.65,
                    random_state=random_state,
                ),
                _extra_trees_candidate(
                    "extra_trees_smoother",
                    n_estimators=220,
                    min_samples_leaf=10,
                    max_features=0.65,
                    random_state=random_state,
                ),
                _gradient_boosting_candidate(
                    "gradient_boosting_deeper",
                    n_estimators=180,
                    learning_rate=0.02,
                    max_depth=3,
                    min_samples_leaf=8,
                    random_state=random_state,
                ),
                _hist_gradient_boosting_candidate(
                    "hist_gradient_boosting_regularized",
                    max_iter=180,
                    learning_rate=0.02,
                    max_leaf_nodes=31,
                    min_samples_leaf=18,
                    random_state=random_state,
                ),
            ]
        )

    if include_statistical_models:
        candidates.extend(_optional_statistical_candidates(search_level=search_level))

    if include_lightgbm:
        candidates.extend(_lightgbm_candidates(random_state=random_state, expanded=expanded))
    candidates.extend(_xgboost_candidates(random_state=random_state, expanded=expanded))

    if tuning_mode == "optuna":
        candidates.extend(
            _optuna_candidates(
                families=optuna_families,
                include_lightgbm=include_lightgbm,
                random_state=random_state,
                n_trials=optuna_trials,
                timeout_seconds=optuna_timeout_seconds,
                inner_splits=optuna_inner_splits,
            )
        )

    if deep_learning_profile in {"fast", "research"}:
        mlp = _tabular_mlp_candidate(random_state=random_state, profile=deep_learning_profile)
        if mlp is not None:
            candidates.append(mlp)

    if include_lstm or deep_learning_profile == "research":
        lstm = _lstm_candidate(random_state=random_state)
        if lstm is not None:
            candidates.append(lstm)
    if deep_learning_profile == "research":
        cnn = _temporal_cnn_candidate(random_state=random_state)
        if cnn is not None:
            candidates.append(cnn)

    return candidates


def _optional_statistical_candidates(search_level: str) -> list[ForecastCandidate]:
    candidates: list[ForecastCandidate] = []
    realtime = search_level == "realtime"
    expanded = search_level == "expanded"
    if importlib.util.find_spec("statsmodels") is not None:
        arima_orders = [(1, 0, 1)]
        sarima_orders = [((1, 0, 1), (1, 0, 1, 5))]
        var_lags = [2]
        if realtime:
            arima_orders = []
            sarima_orders = []
            var_lags = [1, 2]
        if expanded:
            arima_orders.extend([(1, 0, 0), (0, 0, 1), (2, 0, 1)])
            sarima_orders.extend([((1, 0, 0), (1, 0, 0, 5)), ((0, 0, 1), (0, 0, 1, 5))])
            var_lags.extend([1, 3])
        candidates.extend(
            ARIMAReturnCandidate(order=order, name=f"arima_{_order_name(order)}")
            for order in arima_orders
        )
        candidates.extend(
            SARIMAReturnCandidate(
                order=order,
                seasonal_order=seasonal_order,
                name=f"sarima_{_order_name(order)}_{_order_name(seasonal_order)}",
            )
            for order, seasonal_order in sarima_orders
        )
        candidates.extend(VARReturnCandidate(maxlags=maxlags, name=f"var_lag_{maxlags}") for maxlags in var_lags)
    if not realtime and importlib.util.find_spec("arch") is not None:
        garch_orders = [(1, 1)]
        if expanded:
            garch_orders.extend([(1, 2), (2, 1)])
        candidates.extend(
            GARCHReturnCandidate(p=p, q=q, name=f"garch_{p}_{q}") for p, q in garch_orders
        )
    return candidates


def _optuna_candidates(
    families: tuple[str, ...],
    include_lightgbm: bool,
    random_state: int,
    n_trials: int,
    timeout_seconds: int | None,
    inner_splits: int,
) -> list[ForecastCandidate]:
    if importlib.util.find_spec("optuna") is None:
        return []
    normalized = [family.strip().lower().replace("-", "_") for family in families if family.strip()]
    candidates: list[ForecastCandidate] = []
    for family in normalized:
        if family == "lightgbm" and (not include_lightgbm or importlib.util.find_spec("lightgbm") is None):
            continue
        if family == "xgboost" and importlib.util.find_spec("xgboost") is None:
            continue
        if family not in {"elastic_net", "random_forest", "extra_trees", "gradient_boosting", "lightgbm", "xgboost"}:
            continue
        candidates.append(
            OptunaTunedRegressorCandidate(
                model_kind=family,
                n_trials=n_trials,
                timeout_seconds=timeout_seconds,
                inner_splits=inner_splits,
                random_state=random_state,
            )
        )
    return candidates


def _import_optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna tuning requires the optional `optuna` dependency.") from exc
    return optuna


def _inner_time_series_splits(n_rows: int, max_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_rows < 45:
        return []
    split_count = max(1, min(max_splits, 5))
    validation_window = max(10, min(45, n_rows // 5))
    min_train = max(25, int(n_rows * 0.45))
    starts = list(range(min_train, n_rows - validation_window + 1, validation_window))
    starts = starts[-split_count:]
    splits = []
    for start in starts:
        stop = min(start + validation_window, n_rows)
        train_idx = np.arange(0, start)
        validation_idx = np.arange(start, stop)
        if len(train_idx) >= 25 and len(validation_idx) >= 5:
            splits.append((train_idx, validation_idx))
    return splits


def _suggest_optuna_params(trial: Any, model_kind: str) -> dict[str, Any]:
    if model_kind == "elastic_net":
        return {
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
        }
    if model_kind == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 320),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 30),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
        }
    if model_kind == "extra_trees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 320),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 30),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
        }
    if model_kind == "gradient_boosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 60, 260),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 4),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 30),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
        }
    if model_kind == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 320),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 8, 40),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        }
    if model_kind == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 320),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 4),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 20),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
    raise ValueError(f"Unsupported Optuna model family `{model_kind}`.")


def _optuna_estimator_pipeline(model_kind: str, params: dict[str, Any], random_state: int) -> Pipeline:
    if model_kind == "elastic_net":
        model = ElasticNet(
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=20_000,
            random_state=random_state,
        )
        return Pipeline(
            [
                ("drop_all_nan", DropAllNaNColumns()),
                ("imputer", DataFrameSimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
    if model_kind == "random_forest":
        model = RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            max_depth=int(params["max_depth"]),
            random_state=random_state,
            n_jobs=1,
        )
    elif model_kind == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=int(params["n_estimators"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            max_depth=int(params["max_depth"]),
            random_state=random_state,
            n_jobs=1,
        )
    elif model_kind == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            subsample=float(params["subsample"]),
            random_state=random_state,
        )
    elif model_kind == "lightgbm":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            num_leaves=int(params["num_leaves"]),
            min_child_samples=int(params["min_child_samples"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            random_state=random_state,
            n_jobs=1,
            verbosity=-1,
        )
    elif model_kind == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            min_child_weight=int(params["min_child_weight"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]),
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
        )
    else:
        raise ValueError(f"Unsupported Optuna model family `{model_kind}`.")
    return Pipeline(
        [
            ("drop_all_nan", DropAllNaNColumns()),
            ("imputer", DataFrameSimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def _ridge_candidate(name: str, alpha: float) -> ForecastCandidate:
    return SklearnRegressorCandidate(
        name=name,
        family="linear_regularized",
        estimator_factory=lambda: Pipeline(
            [
                ("drop_all_nan", DropAllNaNColumns()),
                ("imputer", DataFrameSimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=alpha)),
            ]
        ),
    )


def _lasso_candidate(name: str, alpha: float, random_state: int) -> ForecastCandidate:
    return SklearnRegressorCandidate(
        name=name,
        family="linear_regularized",
        estimator_factory=lambda: Pipeline(
            [
                ("drop_all_nan", DropAllNaNColumns()),
                ("imputer", DataFrameSimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    Lasso(alpha=alpha, max_iter=20_000, random_state=random_state, selection="random"),
                ),
            ]
        ),
    )


def _elastic_net_candidate(name: str, alpha: float, l1_ratio: float, random_state: int) -> ForecastCandidate:
    return SklearnRegressorCandidate(
        name=name,
        family="linear_regularized",
        estimator_factory=lambda: Pipeline(
            [
                ("drop_all_nan", DropAllNaNColumns()),
                ("imputer", DataFrameSimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20_000, random_state=random_state),
                ),
            ]
        ),
    )


def _random_forest_candidate(
    name: str,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: str | float,
    random_state: int,
) -> ForecastCandidate:
    return SklearnRegressorCandidate(
        name=name,
        family="tree_ensemble",
        estimator_factory=lambda: Pipeline(
            [
                ("drop_all_nan", DropAllNaNColumns()),
                ("imputer", DataFrameSimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=n_estimators,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        parameter_count_hint=n_estimators,
    )


def _decision_tree_candidate(
    name: str,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
) -> ForecastCandidate:
    return SklearnRegressorCandidate(
        name=name,
        family="tree_model",
        estimator_factory=lambda: Pipeline(
            [
                ("drop_all_nan", DropAllNaNColumns()),
                ("imputer", DataFrameSimpleImputer(strategy="median")),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        parameter_count_hint=max(1, 2 ** max_depth),
    )


def _extra_trees_candidate(
    name: str,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: str | float,
    random_state: int,
) -> ForecastCandidate:
    return SklearnRegressorCandidate(
        name=name,
        family="tree_ensemble",
        estimator_factory=lambda: Pipeline(
            [
                ("drop_all_nan", DropAllNaNColumns()),
                ("imputer", DataFrameSimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=n_estimators,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        parameter_count_hint=n_estimators,
    )


def _gradient_boosting_candidate(
    name: str,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
) -> ForecastCandidate:
    return SklearnRegressorCandidate(
        name=name,
        family="boosting_model",
        estimator_factory=lambda: Pipeline(
            [
                ("drop_all_nan", DropAllNaNColumns()),
                ("imputer", DataFrameSimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        parameter_count_hint=n_estimators,
    )


def _hist_gradient_boosting_candidate(
    name: str,
    max_iter: int,
    learning_rate: float,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    random_state: int,
) -> ForecastCandidate:
    return SklearnRegressorCandidate(
        name=name,
        family="boosting_model",
        estimator_factory=lambda: Pipeline(
            [
                ("drop_all_nan", DropAllNaNColumns()),
                ("imputer", DataFrameSimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=max_iter,
                        learning_rate=learning_rate,
                        max_leaf_nodes=max_leaf_nodes,
                        min_samples_leaf=min_samples_leaf,
                        l2_regularization=0.01,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        parameter_count_hint=max_iter,
    )


def _lightgbm_candidates(random_state: int, expanded: bool) -> list[ForecastCandidate]:
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        return []

    configs = [
        ("lightgbm_balanced", 180, 0.03, 15, 12),
    ]
    if expanded:
        configs.extend(
            [
                ("lightgbm_small_leaves", 220, 0.02, 7, 18),
                ("lightgbm_more_leaves", 180, 0.025, 31, 12),
            ]
        )

    return [
        SklearnRegressorCandidate(
            name=name,
            family="boosting_model",
            estimator_factory=lambda n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, min_child_samples=min_child_samples: Pipeline(
                [
                    ("drop_all_nan", DropAllNaNColumns()),
                    ("imputer", DataFrameSimpleImputer(strategy="median")),
                    (
                        "model",
                        LGBMRegressor(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            num_leaves=num_leaves,
                            min_child_samples=min_child_samples,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            random_state=random_state,
                            n_jobs=1,
                            verbosity=-1,
                        ),
                    ),
                ]
            ),
            parameter_count_hint=n_estimators,
        )
        for name, n_estimators, learning_rate, num_leaves, min_child_samples in configs
    ]


def _xgboost_candidates(random_state: int, expanded: bool) -> list[ForecastCandidate]:
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return []

    configs = [
        ("xgboost_conservative", 160, 0.025, 2, 8),
    ]
    if expanded:
        configs.extend(
            [
                ("xgboost_regularized", 220, 0.02, 3, 12),
            ]
        )
    return [
        SklearnRegressorCandidate(
            name=name,
            family="boosting_model",
            estimator_factory=lambda n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight: Pipeline(
                [
                    ("drop_all_nan", DropAllNaNColumns()),
                    ("imputer", DataFrameSimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            min_child_weight=min_child_weight,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            reg_lambda=1.0,
                            objective="reg:squarederror",
                            random_state=random_state,
                            n_jobs=1,
                            verbosity=0,
                        ),
                    ),
                ]
            ),
            parameter_count_hint=n_estimators,
        )
        for name, n_estimators, learning_rate, max_depth, min_child_weight in configs
    ]


def _lstm_candidate(random_state: int) -> ForecastCandidate | None:
    if importlib.util.find_spec("torch") is None:
        return None
    return LSTMReturnCandidate(random_state=random_state)


def _tabular_mlp_candidate(random_state: int, profile: str) -> ForecastCandidate | None:
    if importlib.util.find_spec("torch") is None:
        return None
    if profile == "research":
        return TabularMLPReturnCandidate(
            name="tabular_mlp_research",
            hidden_size=48,
            max_epochs=60,
            patience=8,
            min_rows=180,
            time_budget_seconds=30.0,
            random_state=random_state,
        )
    return TabularMLPReturnCandidate(random_state=random_state)


def _temporal_cnn_candidate(random_state: int) -> ForecastCandidate | None:
    if importlib.util.find_spec("torch") is None:
        return None
    return TemporalCNNReturnCandidate(random_state=random_state)


def _torch_device(preferred: str) -> Any:
    import torch

    requested = preferred.lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "mps":
        return torch.device("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _order_name(order: tuple[int, ...]) -> str:
    return "_".join(str(value) for value in order)


def _target_series(target: pd.Series) -> pd.Series:
    series = pd.to_numeric(target, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        raise ValueError("Target series is empty after cleaning.")
    return series.astype(float)


def _clean_predictions(predictions: Any, length: int, fallback: float) -> np.ndarray:
    values = np.asarray(predictions, dtype=float).reshape(-1)
    if len(values) < length:
        values = np.pad(values, (0, length - len(values)), constant_values=fallback)
    values = values[:length]
    values = np.where(np.isfinite(values), values, fallback)
    return values.astype(float)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def _tree_model_diagnostics(estimator: Pipeline, final_step: Any) -> dict[str, Any]:
    importances = getattr(final_step, "feature_importances_", None)
    if importances is None:
        return {}
    values = np.asarray(importances, dtype=float).reshape(-1)
    columns = _pipeline_feature_names(estimator)
    if len(columns) != len(values):
        columns = [f"feature_{idx}" for idx in range(len(values))]
    total = float(np.sum(values))
    normalized = values / total if total > 0 else np.zeros_like(values)
    ranked = sorted(
        (
            {
                "feature": str(column),
                "importance": _json_safe(value),
                "normalized_importance": _json_safe(norm),
            }
            for column, value, norm in zip(columns, values, normalized)
        ),
        key=lambda row: float(row["importance"]),
        reverse=True,
    )
    top_share = float(sum(float(row["normalized_importance"]) for row in ranked[:5]))
    estimators = getattr(final_step, "estimators_", None)
    tree_count = int(len(estimators)) if estimators is not None else 0
    if tree_count == 0 and hasattr(final_step, "tree_"):
        tree_count = 1
    depth = getattr(final_step, "get_depth", lambda: None)()
    leaf_count = getattr(final_step, "get_n_leaves", lambda: None)()
    return {
        "tree_model": {
            "tree_count": tree_count,
            "max_depth": _json_safe(depth),
            "leaf_count": _json_safe(leaf_count),
            "feature_importance_count": int(len(ranked)),
            "top_5_importance_share": _json_safe(top_share),
            "top_importances": ranked[:20],
            "interpretation": "Tree importances explain fitted split usage; they are model diagnostics, not causal proof.",
        }
    }
