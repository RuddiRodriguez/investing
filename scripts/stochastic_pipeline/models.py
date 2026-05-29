"""Stochastic process models for price and volatility forecasting."""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Callable

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


TRADING_DAYS_PER_YEAR = 252
SQRT_TWO_OVER_PI = float(np.sqrt(2.0 / np.pi))
MIN_OBSERVATIONS = 80
MIN_VARIANCE = 1e-10
ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class VolatilityModelResult:
    model_name: str
    mean_return: float
    parameters: dict[str, float]
    conditional_variance: pd.Series
    volatility_forecast: pd.DataFrame
    price_cone: pd.DataFrame
    log_likelihood: float


@dataclass(frozen=True)
class RegimeModelResult:
    model_name: str
    current_regime: str
    current_state: int
    state_labels: dict[int, str]
    state_summary: pd.DataFrame
    transition_matrix: pd.DataFrame
    filtered_state_probabilities: pd.DataFrame
    forecast_state_probabilities: pd.DataFrame
    price_cone: pd.DataFrame
    parameters: dict[str, float]
    annualized_volatility: float
    log_likelihood: float



def default_progress_callback(step: int, total: int, message: str) -> None:
    return None



def _progress(progress_callback: ProgressCallback | None, step: int, total: int, message: str) -> None:
    callback = progress_callback or default_progress_callback
    callback(step, total, message)



def _coerce_close_series(prices: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(prices, pd.DataFrame):
        if prices.empty:
            raise ValueError("No prices available for stochastic analysis.")
        series = prices.iloc[:, 0]
    else:
        series = prices.copy()
    close = pd.to_numeric(series, errors="coerce").dropna()
    close = close[close > 0.0]
    if len(close) < MIN_OBSERVATIONS:
        raise ValueError(f"Need at least {MIN_OBSERVATIONS} positive price observations. Found {len(close)}.")
    if not isinstance(close.index, pd.DatetimeIndex):
        close.index = pd.RangeIndex(len(close))
    return close.sort_index()



def _prepare_history(close: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    log_returns = np.log(close / close.shift(1)).dropna()
    history = pd.DataFrame({"date": close.index, "price": close.to_numpy()})
    return history, log_returns


def _annualized_realized_volatility(price_path: pd.Series) -> float:
    log_returns = np.log(price_path / price_path.shift(1)).dropna()
    if len(log_returns) < 2:
        return float("nan")
    return float(log_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _build_regime_feature_frame(log_returns: pd.Series) -> pd.DataFrame:
    features = pd.DataFrame(
        {
            "log_return": log_returns,
            "abs_log_return": log_returns.abs(),
            "rolling_vol_20d": log_returns.rolling(20).std(),
            "rolling_mean_5d": log_returns.rolling(5).mean(),
            "rolling_mean_20d": log_returns.rolling(20).mean(),
        }
    ).dropna()
    if len(features) < 60:
        raise ValueError("Need at least 60 feature rows to fit the stochastic regime model.")
    return features


def _label_regime_states(hidden_states: np.ndarray, features: pd.DataFrame) -> tuple[dict[int, str], pd.DataFrame]:
    state_summary = (
        features.assign(state=hidden_states)
        .groupby("state", as_index=True)
        .agg(
            mean_log_return=("log_return", "mean"),
            daily_volatility=("log_return", lambda values: float(np.std(values, ddof=1))),
            mean_abs_log_return=("abs_log_return", "mean"),
            rolling_vol_20d=("rolling_vol_20d", "mean"),
            rolling_mean_5d=("rolling_mean_5d", "mean"),
            rolling_mean_20d=("rolling_mean_20d", "mean"),
            observations=("log_return", "count"),
        )
        .sort_index()
    )
    remaining = list(state_summary.index)
    crash_state = int(state_summary["mean_log_return"].idxmin())
    remaining.remove(crash_state)
    explosive_state = int(state_summary.loc[remaining, "mean_log_return"].idxmax())
    remaining.remove(explosive_state)
    low_vol_state = int(state_summary.loc[remaining, "daily_volatility"].idxmin())
    remaining.remove(low_vol_state)
    trend_state = int(remaining[0])
    labels = {
        low_vol_state: "low_volatility_normal",
        trend_state: "high_volatility_trend",
        crash_state: "crash_stress",
        explosive_state: "explosive_momentum",
    }
    state_summary = state_summary.assign(regime_label=state_summary.index.map(labels))
    state_summary["annualized_volatility"] = state_summary["daily_volatility"] * np.sqrt(TRADING_DAYS_PER_YEAR)
    return labels, state_summary


def _simulate_regime_conditioned_paths(
    *,
    hmm_model: GaussianHMM,
    scaler: StandardScaler,
    current_state: int,
    last_price: float,
    horizon_days: int,
    num_paths: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    paths = np.empty((horizon_days + 1, num_paths), dtype=float)
    returns = np.empty((horizon_days, num_paths), dtype=float)
    paths[0, :] = last_price
    for path_index in range(num_paths):
        state = int(current_state)
        for step in range(horizon_days):
            sample_scaled = rng.multivariate_normal(hmm_model.means_[state], hmm_model.covars_[state])
            sample = scaler.inverse_transform(sample_scaled.reshape(1, -1))[0]
            log_return = float(sample[0])
            returns[step, path_index] = log_return
            paths[step + 1, path_index] = paths[step, path_index] * np.exp(log_return)
            state = int(rng.choice(hmm_model.n_components, p=hmm_model.transmat_[state]))
    return paths, returns


def fit_regime_hmm(close: pd.Series, horizon_days: int, num_paths: int, seed: int) -> RegimeModelResult:
    _, log_returns = _prepare_history(close)
    features = _build_regime_feature_frame(log_returns)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    hmm_model: GaussianHMM | None = None
    hidden_states: np.ndarray | None = None
    filtered_probabilities: np.ndarray | None = None
    best_score = -np.inf
    for seed_offset in range(4):
        candidate = GaussianHMM(
            n_components=4,
            covariance_type="full",
            n_iter=300,
            random_state=seed + seed_offset,
            min_covar=1e-5,
        )
        with contextlib.redirect_stderr(io.StringIO()):
            candidate.fit(scaled_features)
        candidate_score = float(candidate.score(scaled_features))
        if candidate_score > best_score:
            hmm_model = candidate
            hidden_states = candidate.predict(scaled_features)
            filtered_probabilities = candidate.predict_proba(scaled_features)
            best_score = candidate_score
    if hmm_model is None or hidden_states is None or filtered_probabilities is None:
        raise ValueError("Unable to fit the stochastic regime HMM.")
    state_labels, state_summary = _label_regime_states(hidden_states, features)
    current_state = int(hidden_states[-1])
    current_regime = state_labels[current_state]
    current_probabilities = filtered_probabilities[-1]

    forecast_probabilities: list[np.ndarray] = []
    state_probability = current_probabilities.copy()
    for _ in range(horizon_days):
        forecast_probabilities.append(state_probability)
        state_probability = state_probability @ hmm_model.transmat_

    paths, simulated_returns = _simulate_regime_conditioned_paths(
        hmm_model=hmm_model,
        scaler=scaler,
        current_state=current_state,
        last_price=float(close.iloc[-1]),
        horizon_days=horizon_days,
        num_paths=num_paths,
        seed=seed,
    )
    index = _future_index(close.index, horizon_days)
    price_cone = _percentile_frame_from_paths(paths, index)
    annualized_volatility = float(np.std(simulated_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    filtered_probability_frame = pd.DataFrame(
        filtered_probabilities,
        index=features.index,
        columns=[state_labels[index] for index in range(hmm_model.n_components)],
    )
    forecast_probability_frame = pd.DataFrame(
        forecast_probabilities,
        index=index[1:],
        columns=[state_labels[index] for index in range(hmm_model.n_components)],
    )
    transition_matrix = pd.DataFrame(
        hmm_model.transmat_,
        index=[state_labels[index] for index in range(hmm_model.n_components)],
        columns=[state_labels[index] for index in range(hmm_model.n_components)],
    )
    return RegimeModelResult(
        model_name="Regime HMM",
        current_regime=current_regime,
        current_state=current_state,
        state_labels=state_labels,
        state_summary=state_summary,
        transition_matrix=transition_matrix,
        filtered_state_probabilities=filtered_probability_frame,
        forecast_state_probabilities=forecast_probability_frame,
        price_cone=price_cone,
        parameters={"n_components": 4.0, "feature_rows": float(len(features)), "fit_restarts": 4.0},
        annualized_volatility=annualized_volatility,
        log_likelihood=best_score,
    )



def _future_index(index: pd.Index, horizon_days: int) -> pd.Index:
    if isinstance(index, pd.DatetimeIndex):
        last = pd.Timestamp(index[-1])
        future = pd.bdate_range(last + pd.offsets.BDay(1), periods=horizon_days)
        return pd.Index([last, *future])
    return pd.RangeIndex(horizon_days + 1)



def _percentile_frame_from_paths(paths: np.ndarray, index: pd.Index) -> pd.DataFrame:
    percentiles = {
        "p10": np.quantile(paths, 0.10, axis=1),
        "p25": np.quantile(paths, 0.25, axis=1),
        "p50": np.quantile(paths, 0.50, axis=1),
        "p75": np.quantile(paths, 0.75, axis=1),
        "p90": np.quantile(paths, 0.90, axis=1),
        "mean": np.mean(paths, axis=1),
    }
    return pd.DataFrame(percentiles, index=index)



def _price_cone_from_variance(
    last_price: float,
    mean_return: float,
    variance_forecast: np.ndarray,
    index: pd.Index,
) -> pd.DataFrame:
    horizon = len(variance_forecast)
    cumulative_variance = np.cumsum(variance_forecast)
    days = np.arange(1, horizon + 1, dtype=float)
    median = last_price * np.exp(mean_return * days)
    percentiles: dict[str, np.ndarray] = {"p50": median}
    normal = NormalDist()
    for label, probability in (("p10", 0.10), ("p25", 0.25), ("p75", 0.75), ("p90", 0.90)):
        z_value = normal.inv_cdf(probability)
        percentiles[label] = last_price * np.exp(mean_return * days + z_value * np.sqrt(cumulative_variance))
    cone = pd.DataFrame(percentiles, index=index[1:])
    origin = pd.DataFrame({column: [last_price] for column in cone.columns}, index=index[:1])
    return pd.concat([origin, cone], axis=0)



def estimate_gbm(close: pd.Series, horizon_days: int, num_paths: int, seed: int) -> dict[str, Any]:
    _, log_returns = _prepare_history(close)
    mean_return = float(log_returns.mean())
    volatility = float(log_returns.std(ddof=1))
    rng = np.random.default_rng(seed)
    increments = rng.normal(loc=mean_return, scale=max(volatility, np.sqrt(MIN_VARIANCE)), size=(horizon_days, num_paths))
    cumulative = np.vstack([np.zeros(num_paths), np.cumsum(increments, axis=0)])
    paths = float(close.iloc[-1]) * np.exp(cumulative)
    index = _future_index(close.index, horizon_days)
    percentile_frame = _percentile_frame_from_paths(paths, index)
    return {
        "model_name": "GBM",
        "parameters": {
            "mean_daily_log_return": mean_return,
            "daily_volatility": volatility,
            "annualized_drift": mean_return * TRADING_DAYS_PER_YEAR,
            "annualized_volatility": volatility * np.sqrt(TRADING_DAYS_PER_YEAR),
        },
        "price_cone": percentile_frame,
    }


def estimate_jump_diffusion(close: pd.Series, horizon_days: int, num_paths: int, seed: int) -> dict[str, Any]:
    _, log_returns = _prepare_history(close)
    mean_return = float(log_returns.mean())
    volatility = float(log_returns.std(ddof=1))
    centered_returns = log_returns - mean_return
    jump_threshold = max(2.0 * volatility, np.quantile(np.abs(centered_returns), 0.90))
    jump_mask = np.abs(centered_returns) >= jump_threshold
    jump_returns = centered_returns.loc[jump_mask]
    jump_intensity = float(jump_mask.mean())
    jump_mean = float(jump_returns.mean()) if not jump_returns.empty else 0.0
    jump_volatility = float(jump_returns.std(ddof=1)) if len(jump_returns) > 1 else max(volatility, np.sqrt(MIN_VARIANCE))
    adjusted_drift = mean_return - jump_intensity * jump_mean

    rng = np.random.default_rng(seed)
    diffusion_increments = rng.normal(
        loc=adjusted_drift,
        scale=max(volatility, np.sqrt(MIN_VARIANCE)),
        size=(horizon_days, num_paths),
    )
    jump_counts = rng.poisson(jump_intensity, size=(horizon_days, num_paths))
    jump_components = np.zeros((horizon_days, num_paths), dtype=float)
    positive_jump_positions = np.argwhere(jump_counts > 0)
    for row_index, col_index in positive_jump_positions:
        count = int(jump_counts[row_index, col_index])
        jump_components[row_index, col_index] = float(rng.normal(jump_mean, jump_volatility, size=count).sum())
    increments = diffusion_increments + jump_components
    cumulative = np.vstack([np.zeros(num_paths), np.cumsum(increments, axis=0)])
    paths = float(close.iloc[-1]) * np.exp(cumulative)
    index = _future_index(close.index, horizon_days)
    percentile_frame = _percentile_frame_from_paths(paths, index)
    return {
        "model_name": "Jump Diffusion",
        "parameters": {
            "mean_daily_log_return": mean_return,
            "daily_diffusion_volatility": volatility,
            "jump_intensity_daily": jump_intensity,
            "jump_mean": jump_mean,
            "jump_volatility": jump_volatility,
            "annualized_drift": mean_return * TRADING_DAYS_PER_YEAR,
            "annualized_volatility": float(np.std(increments, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)),
            "jump_observations": float(jump_mask.sum()),
        },
        "price_cone": percentile_frame,
    }



def _garch_log_likelihood(residuals: np.ndarray, omega: float, alpha: float, beta: float) -> tuple[float, np.ndarray]:
    variance = np.empty(len(residuals), dtype=float)
    variance[0] = max(np.var(residuals, ddof=1), MIN_VARIANCE)
    for index in range(1, len(residuals)):
        variance[index] = omega + alpha * residuals[index - 1] ** 2 + beta * variance[index - 1]
        variance[index] = max(variance[index], MIN_VARIANCE)
    log_likelihood = -0.5 * float(np.sum(np.log(2.0 * np.pi) + np.log(variance) + (residuals**2) / variance))
    return log_likelihood, variance



def fit_garch11(log_returns: pd.Series, horizon_days: int, close: pd.Series) -> VolatilityModelResult:
    mean_return = float(log_returns.mean())
    residuals = (log_returns - mean_return).to_numpy()
    base_variance = max(float(np.var(residuals, ddof=1)), MIN_VARIANCE)
    best: tuple[float, dict[str, float], np.ndarray] | None = None
    for alpha in np.linspace(0.03, 0.22, 6):
        for beta in np.linspace(0.72, 0.96, 7):
            if alpha + beta >= 0.995:
                continue
            for scale in (0.75, 1.0, 1.25):
                omega = base_variance * (1.0 - alpha - beta) * scale
                if omega <= 0.0:
                    continue
                log_likelihood, variance = _garch_log_likelihood(residuals, omega, alpha, beta)
                if best is None or log_likelihood > best[0]:
                    best = (log_likelihood, {"omega": omega, "alpha": alpha, "beta": beta}, variance)
    if best is None:
        raise ValueError("Unable to fit GARCH(1,1) parameters.")

    log_likelihood, parameters, variance = best
    forecast = np.empty(horizon_days, dtype=float)
    forecast[0] = parameters["omega"] + parameters["alpha"] * residuals[-1] ** 2 + parameters["beta"] * variance[-1]
    forecast[0] = max(forecast[0], MIN_VARIANCE)
    for index in range(1, horizon_days):
        forecast[index] = parameters["omega"] + (parameters["alpha"] + parameters["beta"]) * forecast[index - 1]
        forecast[index] = max(forecast[index], MIN_VARIANCE)

    vol_index = _future_index(close.index, horizon_days)[1:]
    volatility_forecast = pd.DataFrame(
        {
            "variance": forecast,
            "daily_volatility": np.sqrt(forecast),
            "annualized_volatility": np.sqrt(forecast * TRADING_DAYS_PER_YEAR),
        },
        index=vol_index,
    )
    return VolatilityModelResult(
        model_name="GARCH(1,1)",
        mean_return=mean_return,
        parameters={**parameters, "persistence": parameters["alpha"] + parameters["beta"]},
        conditional_variance=pd.Series(variance, index=log_returns.index),
        volatility_forecast=volatility_forecast,
        price_cone=_price_cone_from_variance(float(close.iloc[-1]), mean_return, forecast, _future_index(close.index, horizon_days)),
        log_likelihood=log_likelihood,
    )



def _egarch_log_likelihood(
    residuals: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[float, np.ndarray]:
    log_variance = np.empty(len(residuals), dtype=float)
    base_variance = max(float(np.var(residuals, ddof=1)), MIN_VARIANCE)
    log_variance[0] = np.log(base_variance)
    for index in range(1, len(residuals)):
        standardized = residuals[index - 1] / np.sqrt(np.exp(log_variance[index - 1]))
        shock = alpha * (abs(standardized) - SQRT_TWO_OVER_PI) + gamma * standardized
        log_variance[index] = omega + beta * log_variance[index - 1] + shock
        log_variance[index] = float(np.clip(log_variance[index], np.log(MIN_VARIANCE), np.log(10.0)))
    variance = np.exp(log_variance)
    log_likelihood = -0.5 * float(np.sum(np.log(2.0 * np.pi) + log_variance + (residuals**2) / variance))
    return log_likelihood, variance



def fit_egarch11(log_returns: pd.Series, horizon_days: int, close: pd.Series) -> VolatilityModelResult:
    mean_return = float(log_returns.mean())
    residuals = (log_returns - mean_return).to_numpy()
    base_variance = max(float(np.var(residuals, ddof=1)), MIN_VARIANCE)
    best: tuple[float, dict[str, float], np.ndarray] | None = None
    base_log_variance = float(np.log(base_variance))
    for alpha in np.linspace(0.04, 0.20, 5):
        for beta in np.linspace(0.75, 0.98, 6):
            for gamma in np.linspace(-0.20, 0.20, 5):
                for offset in (-0.25, 0.0, 0.25):
                    omega = (1.0 - beta) * base_log_variance + offset
                    log_likelihood, variance = _egarch_log_likelihood(residuals, omega, alpha, beta, gamma)
                    if best is None or log_likelihood > best[0]:
                        best = (log_likelihood, {"omega": omega, "alpha": alpha, "beta": beta, "gamma": gamma}, variance)
    if best is None:
        raise ValueError("Unable to fit EGARCH(1,1) parameters.")

    log_likelihood, parameters, variance = best
    forecast_log_variance = np.empty(horizon_days, dtype=float)
    last_log_variance = float(np.log(max(variance[-1], MIN_VARIANCE)))
    last_standardized = residuals[-1] / np.sqrt(max(variance[-1], MIN_VARIANCE))
    forecast_log_variance[0] = (
        parameters["omega"]
        + parameters["beta"] * last_log_variance
        + parameters["alpha"] * (abs(last_standardized) - SQRT_TWO_OVER_PI)
        + parameters["gamma"] * last_standardized
    )
    for index in range(1, horizon_days):
        forecast_log_variance[index] = parameters["omega"] + parameters["beta"] * forecast_log_variance[index - 1]
    forecast_log_variance = np.clip(forecast_log_variance, np.log(MIN_VARIANCE), np.log(10.0))
    forecast_variance = np.exp(forecast_log_variance)

    vol_index = _future_index(close.index, horizon_days)[1:]
    volatility_forecast = pd.DataFrame(
        {
            "variance": forecast_variance,
            "daily_volatility": np.sqrt(forecast_variance),
            "annualized_volatility": np.sqrt(forecast_variance * TRADING_DAYS_PER_YEAR),
        },
        index=vol_index,
    )
    return VolatilityModelResult(
        model_name="EGARCH(1,1)",
        mean_return=mean_return,
        parameters={**parameters, "persistence": parameters["beta"]},
        conditional_variance=pd.Series(variance, index=log_returns.index),
        volatility_forecast=volatility_forecast,
        price_cone=_price_cone_from_variance(float(close.iloc[-1]), mean_return, forecast_variance, _future_index(close.index, horizon_days)),
        log_likelihood=log_likelihood,
    )



def run_stochastic_analysis(
    prices: pd.DataFrame | pd.Series,
    horizon_days: int = 30,
    num_paths: int = 1000,
    seed: int = 42,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    close = _coerce_close_series(prices)
    history, log_returns = _prepare_history(close)
    total_steps = 6

    _progress(progress_callback, 1, total_steps, "Preparing stochastic price history")
    gbm_result = estimate_gbm(close, horizon_days=horizon_days, num_paths=num_paths, seed=seed)

    _progress(progress_callback, 2, total_steps, "Estimating jump diffusion model")
    jump_result = estimate_jump_diffusion(close, horizon_days=horizon_days, num_paths=num_paths, seed=seed)

    _progress(progress_callback, 3, total_steps, "Fitting GARCH(1,1) volatility model")
    garch_result = fit_garch11(log_returns, horizon_days=horizon_days, close=close)

    _progress(progress_callback, 4, total_steps, "Fitting EGARCH(1,1) volatility model")
    egarch_result = fit_egarch11(log_returns, horizon_days=horizon_days, close=close)

    _progress(progress_callback, 5, total_steps, "Fitting four-state HMM regime model")
    regime_result = fit_regime_hmm(close, horizon_days=horizon_days, num_paths=num_paths, seed=seed)

    _progress(progress_callback, 6, total_steps, "Building stochastic model summaries")
    comparison = pd.DataFrame(
        [
            {
                "model": gbm_result["model_name"],
                "annualized_volatility": gbm_result["parameters"]["annualized_volatility"],
                "terminal_median_price": gbm_result["price_cone"].iloc[-1]["p50"],
                "terminal_p10_price": gbm_result["price_cone"].iloc[-1]["p10"],
                "terminal_p90_price": gbm_result["price_cone"].iloc[-1]["p90"],
                "log_likelihood": np.nan,
            },
            {
                "model": jump_result["model_name"],
                "annualized_volatility": jump_result["parameters"]["annualized_volatility"],
                "terminal_median_price": jump_result["price_cone"].iloc[-1]["p50"],
                "terminal_p10_price": jump_result["price_cone"].iloc[-1]["p10"],
                "terminal_p90_price": jump_result["price_cone"].iloc[-1]["p90"],
                "log_likelihood": np.nan,
            },
            {
                "model": garch_result.model_name,
                "annualized_volatility": garch_result.volatility_forecast.iloc[-1]["annualized_volatility"],
                "terminal_median_price": garch_result.price_cone.iloc[-1]["p50"],
                "terminal_p10_price": garch_result.price_cone.iloc[-1]["p10"],
                "terminal_p90_price": garch_result.price_cone.iloc[-1]["p90"],
                "log_likelihood": garch_result.log_likelihood,
            },
            {
                "model": egarch_result.model_name,
                "annualized_volatility": egarch_result.volatility_forecast.iloc[-1]["annualized_volatility"],
                "terminal_median_price": egarch_result.price_cone.iloc[-1]["p50"],
                "terminal_p10_price": egarch_result.price_cone.iloc[-1]["p10"],
                "terminal_p90_price": egarch_result.price_cone.iloc[-1]["p90"],
                "log_likelihood": egarch_result.log_likelihood,
            },
            {
                "model": regime_result.model_name,
                "annualized_volatility": regime_result.annualized_volatility,
                "terminal_median_price": regime_result.price_cone.iloc[-1]["p50"],
                "terminal_p10_price": regime_result.price_cone.iloc[-1]["p10"],
                "terminal_p90_price": regime_result.price_cone.iloc[-1]["p90"],
                "log_likelihood": regime_result.log_likelihood,
            },
        ]
    )

    return {
        "history": history,
        "log_returns": log_returns,
        "gbm": gbm_result,
        "jump": jump_result,
        "garch": garch_result,
        "egarch": egarch_result,
        "regime": regime_result,
        "comparison": comparison,
        "ticker": str(close.name or "PRICE"),
        "last_price": float(close.iloc[-1]),
    }


def run_stochastic_backtest(
    prices: pd.DataFrame | pd.Series,
    horizon_days: int = 30,
    evaluation_windows: int = 12,
    step_days: int = 5,
    num_paths: int = 1000,
    seed: int = 42,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    close = _coerce_close_series(prices)
    minimum_required_rows = MIN_OBSERVATIONS + horizon_days + 1
    if len(close) < minimum_required_rows:
        raise ValueError(
            f"Need at least {minimum_required_rows} rows to run a rolling stochastic backtest. Found {len(close)}."
        )

    eligible_positions = list(range(MIN_OBSERVATIONS - 1, len(close) - horizon_days, max(1, step_days)))
    if not eligible_positions:
        raise ValueError("No eligible start dates available for rolling stochastic backtest.")
    selected_positions = eligible_positions[-max(1, evaluation_windows) :]

    rows: list[dict[str, Any]] = []
    total_steps = len(selected_positions)
    for step, position in enumerate(selected_positions, start=1):
        start_date = pd.Timestamp(close.index[position])
        _progress(progress_callback, step, total_steps, f"Backtesting stochastic models from {start_date.date()}")
        training_close = close.iloc[: position + 1].copy()
        actual_future = close.iloc[position + 1 : position + 1 + horizon_days].copy()
        realized_path = pd.concat([training_close.iloc[-1:], actual_future])
        realized_annualized_volatility = _annualized_realized_volatility(realized_path)
        analysis = run_stochastic_analysis(
            training_close,
            horizon_days=horizon_days,
            num_paths=num_paths,
            seed=seed,
            progress_callback=None,
        )
        actual_terminal_price = float(actual_future.iloc[-1])
        for label in ("gbm", "jump", "garch", "egarch", "regime"):
            model_result = analysis[label]
            price_cone = model_result["price_cone"] if label in {"gbm", "jump"} else model_result.price_cone
            model_name = model_result["model_name"] if label in {"gbm", "jump"} else model_result.model_name
            terminal_row = price_cone.iloc[-1]
            terminal_median_price = float(terminal_row["p50"])
            terminal_p10_price = float(terminal_row["p10"])
            terminal_p90_price = float(terminal_row["p90"])
            forecast_annualized_volatility = float(
                analysis["comparison"].loc[
                    analysis["comparison"]["model"] == model_name,
                    "annualized_volatility",
                ].iloc[0]
            )
            upper_breach = actual_terminal_price > terminal_p90_price
            lower_breach = actual_terminal_price < terminal_p10_price
            rows.append(
                {
                    "simulation_start_date": start_date,
                    "actual_terminal_date": pd.Timestamp(actual_future.index[-1]),
                    "model": model_name,
                    "start_price": float(training_close.iloc[-1]),
                    "actual_terminal_price": actual_terminal_price,
                    "terminal_median_price": terminal_median_price,
                    "terminal_error": terminal_median_price - actual_terminal_price,
                    "terminal_abs_error": abs(terminal_median_price - actual_terminal_price),
                    "terminal_p10_price": terminal_p10_price,
                    "terminal_p90_price": terminal_p90_price,
                    "terminal_coverage_10_90": terminal_p10_price <= actual_terminal_price <= terminal_p90_price,
                    "upper_cone_breach": upper_breach,
                    "lower_cone_breach": lower_breach,
                    "cone_breach_label": "upper" if upper_breach else "lower" if lower_breach else "inside",
                    "annualized_volatility": forecast_annualized_volatility,
                    "realized_annualized_volatility": realized_annualized_volatility,
                    "volatility_forecast_error": forecast_annualized_volatility - realized_annualized_volatility,
                    "volatility_forecast_abs_error": abs(forecast_annualized_volatility - realized_annualized_volatility),
                }
            )

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("model", as_index=False)
        .agg(
            windows=("simulation_start_date", "count"),
            median_terminal_error=("terminal_error", "median"),
            mean_terminal_abs_error=("terminal_abs_error", "mean"),
            median_terminal_abs_error=("terminal_abs_error", "median"),
            rmse_terminal_error=("terminal_error", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            coverage_10_90=("terminal_coverage_10_90", "mean"),
            upper_breach_rate=("upper_cone_breach", "mean"),
            lower_breach_rate=("lower_cone_breach", "mean"),
            mean_annualized_volatility=("annualized_volatility", "mean"),
            mean_realized_annualized_volatility=("realized_annualized_volatility", "mean"),
            mean_volatility_forecast_error=("volatility_forecast_error", "mean"),
            median_volatility_forecast_error=("volatility_forecast_error", "median"),
            mean_volatility_forecast_abs_error=("volatility_forecast_abs_error", "mean"),
        )
        .sort_values("mean_terminal_abs_error")
        .reset_index(drop=True)
    )
    return {
        "detail": detail,
        "summary": summary,
        "horizon_days": horizon_days,
        "evaluation_windows": len(selected_positions),
        "step_days": max(1, step_days),
        "ticker": str(close.name or "PRICE"),
    }
