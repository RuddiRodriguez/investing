from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from market_forecasting_engine.daily_trade import build_intraday_feature_frame
from market_forecasting_engine.data import normalize_price_frame
from market_forecasting_engine.data_providers import DataRequest, load_prices_with_provider
from market_forecasting_engine.llm_options_trader.chronos_forecast import build_chronos_forecast


DEFAULT_HORIZON_MINUTES = (5, 10, 15, 30)


@dataclass(frozen=True)
class CandidateResult:
    rank: int
    model_name: str
    model_family: str
    horizon_minutes: int
    mae_bps: float
    rmse_bps: float
    directional_accuracy: float
    up_precision: float | None
    down_precision: float | None
    score: float


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_step("loading prices")
    prices, metadata = load_spy_prices(args)
    log_step(f"loaded {len(prices)} price rows")
    supervised = build_supervised_frame(prices, DEFAULT_HORIZON_MINUTES)
    target_col = f"target_return_{args.primary_horizon_minutes}m"
    data = supervised.dropna(subset=[target_col]).copy()
    feature_cols = sorted(
        col
        for col in data.columns
        if col.startswith("intraday_") or col.startswith("price_") or col.startswith("technical_")
    )
    feature_cols = [col for col in feature_cols if data[col].notna().mean() >= 0.75]
    if len(data) < args.min_rows:
        raise RuntimeError(f"Need at least {args.min_rows} supervised rows; got {len(data)}.")
    if not feature_cols:
        raise RuntimeError("No usable feature columns were produced.")

    train_idx, validation_idx, test_idx = chronological_splits(len(data), args.train_fraction, args.validation_fraction)
    log_step("evaluating candidate models")
    candidate_results, candidate_models = evaluate_candidates(
        data=data,
        feature_cols=feature_cols,
        target_col=target_col,
        train_idx=train_idx,
        validation_idx=validation_idx,
        primary_horizon=args.primary_horizon_minutes,
        random_state=args.random_state,
    )
    best_three = candidate_results[:3]
    best_family = next(
        item.model_family
        for item in candidate_results
        if not item.model_family.endswith("_baseline")
    )

    log_step(f"tuning best trainable family: {best_family}")
    study, tuned_model = tune_best_family(
        family=best_family,
        data=data,
        feature_cols=feature_cols,
        target_col=target_col,
        train_idx=train_idx,
        validation_idx=validation_idx,
        trials=args.trials,
        random_state=args.random_state,
    )

    final_train_idx = np.r_[train_idx, validation_idx]
    log_step("fitting tuned model and building validation tables")
    tuned_model.fit(data.iloc[final_train_idx][feature_cols], data.iloc[final_train_idx][target_col])
    validation_tables = build_horizon_validation_tables(
        model=tuned_model,
        data=data,
        feature_cols=feature_cols,
        test_idx=test_idx,
        horizons=DEFAULT_HORIZON_MINUTES,
        prices=prices,
    )
    tuned_primary_metrics = metrics_for_predictions(
        data.iloc[test_idx][target_col].to_numpy(dtype=float),
        tuned_model.predict(data.iloc[test_idx][feature_cols]),
    )
    log_step("running chronos comparison")
    chronos_comparison = build_chronos_comparison(
        args=args,
        prices=prices,
        data=data,
        validation_idx=validation_idx,
        test_idx=test_idx,
        horizons=DEFAULT_HORIZON_MINUTES,
    )
    log_step(f"chronos comparison status: {chronos_comparison.get('status')}")

    log_step("writing outputs")
    write_outputs(
        output_dir=output_dir,
        args=args,
        prices=prices,
        data=data,
        metadata=metadata,
        feature_cols=feature_cols,
        candidate_results=candidate_results,
        best_three=best_three,
        best_family=best_family,
        study=study,
        tuned_primary_metrics=tuned_primary_metrics,
        validation_tables=validation_tables,
        chronos_comparison=chronos_comparison,
        tuned_model=tuned_model,
    )
    print(json.dumps({"report": str(output_dir / "report.md"), "output_dir": str(output_dir)}, indent=2))


def log_step(message: str) -> None:
    print(f"[spy-optuna] {pd.Timestamp.utcnow().isoformat()} {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent SPY intraday model-selection and Optuna tuning experiment.")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--provider", default="alpaca", choices=["alpaca", "yahoo", "polygon", "csv"])
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--lookback-days", type=int, default=45)
    parser.add_argument("--primary-horizon-minutes", type=int, default=15)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--enable-chronos-compare", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chronos-model", default="amazon/chronos-t5-tiny")
    parser.add_argument("--chronos-context-rows", type=int, default=512)
    parser.add_argument("--chronos-num-samples", type=int, default=24)
    parser.add_argument("--chronos-validation-stride", type=int, default=30)
    parser.add_argument("--chronos-max-evals", type=int, default=12)
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--min-rows", type=int, default=1200)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        default="experiments/spy_optuna_forecaster/runs/latest",
    )
    return parser.parse_args()


def load_spy_prices(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=int(args.lookback_days))
    request = DataRequest(
        ticker=args.ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        interval=args.interval,
        target_column="close",
    )
    result = load_prices_with_provider(args.provider, request, store=None, use_cache=False)
    frame = normalize_price_frame(result.frame, target_column="close").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame, result.metadata


def build_supervised_frame(prices: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    frame = normalize_price_frame(prices, target_column="close").sort_index()
    features = build_intraday_feature_frame(frame, target_column="close")
    close = frame["close"].astype(float)
    output = features.copy()
    output["price_close"] = close
    output["price_volume"] = frame["volume"].astype(float) if "volume" in frame.columns else 1.0
    output["price_range_pct"] = (frame["high"].astype(float) - frame["low"].astype(float)) / close
    for horizon in horizons:
        bars = max(1, int(round(horizon / infer_interval_minutes(frame.index))))
        future = close.shift(-bars)
        output[f"target_return_{horizon}m"] = np.log(future.replace(0, np.nan)) - np.log(close.replace(0, np.nan))
        output[f"target_price_{horizon}m"] = future
    return output.replace([np.inf, -np.inf], np.nan)


def infer_interval_minutes(index: pd.Index) -> float:
    timestamps = pd.DatetimeIndex(index).sort_values()
    if len(timestamps) < 3:
        return 1.0
    diffs = timestamps.to_series().diff().dropna().dt.total_seconds() / 60.0
    intraday = diffs[(diffs > 0) & (diffs <= 120)]
    if intraday.empty:
        return 1.0
    return float(intraday.median())


def chronological_splits(n_rows: int, train_fraction: float, validation_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = max(1, int(n_rows * train_fraction))
    validation_end = max(train_end + 1, int(n_rows * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, n_rows - 1)
    return np.arange(0, train_end), np.arange(train_end, validation_end), np.arange(validation_end, n_rows)


def evaluate_candidates(
    *,
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    primary_horizon: int,
    random_state: int,
) -> tuple[list[CandidateResult], dict[str, Any]]:
    models = base_candidate_models(random_state)
    results: list[CandidateResult] = []
    fitted: dict[str, Any] = {}
    x_train = data.iloc[train_idx][feature_cols]
    y_train = data.iloc[train_idx][target_col].to_numpy(dtype=float)
    x_val = data.iloc[validation_idx][feature_cols]
    y_val = data.iloc[validation_idx][target_col].to_numpy(dtype=float)
    for family, pred in {
        "zero_return_baseline": np.zeros_like(y_val),
        "recent_momentum_baseline": recent_momentum_prediction(x_val),
    }.items():
        metrics = metrics_for_predictions(y_val, pred)
        results.append(
            CandidateResult(
                rank=0,
                model_name=family,
                model_family=family,
                horizon_minutes=primary_horizon,
                mae_bps=metrics["mae_bps"],
                rmse_bps=metrics["rmse_bps"],
                directional_accuracy=metrics["directional_accuracy"],
                up_precision=metrics["up_precision"],
                down_precision=metrics["down_precision"],
                score=metrics["score"],
            )
        )
    for family, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        metrics = metrics_for_predictions(y_val, pred)
        results.append(
            CandidateResult(
                rank=0,
                model_name=f"{family}_baseline",
                model_family=family,
                horizon_minutes=primary_horizon,
                mae_bps=metrics["mae_bps"],
                rmse_bps=metrics["rmse_bps"],
                directional_accuracy=metrics["directional_accuracy"],
                up_precision=metrics["up_precision"],
                down_precision=metrics["down_precision"],
                score=metrics["score"],
            )
        )
        fitted[family] = model
    results = sorted(results, key=lambda item: item.score)
    results = [CandidateResult(**{**asdict(item), "rank": rank}) for rank, item in enumerate(results, start=1)]
    return results, fitted


def recent_momentum_prediction(features: pd.DataFrame) -> np.ndarray:
    for column in (
        "intraday_log_return_12_bars",
        "intraday_log_return_6_bars",
        "intraday_log_return_3_bars",
        "intraday_log_return_1_bars",
    ):
        if column in features.columns:
            values = features[column].to_numpy(dtype=float)
            return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.zeros(len(features), dtype=float)


def base_candidate_models(random_state: int) -> dict[str, Any]:
    return {
        "ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=random_state)),
            ]
        ),
        "elastic_net": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.0005, l1_ratio=0.25, max_iter=5000, random_state=random_state)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=180, min_samples_leaf=10, max_features=0.65, n_jobs=-1, random_state=random_state)),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesRegressor(n_estimators=220, min_samples_leaf=8, max_features=0.75, n_jobs=-1, random_state=random_state)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(max_iter=220, learning_rate=0.035, max_leaf_nodes=21, l2_regularization=0.05, random_state=random_state)),
            ]
        ),
        "lightgbm": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LGBMRegressor(n_estimators=250, learning_rate=0.035, num_leaves=24, min_child_samples=40, subsample=0.8, colsample_bytree=0.8, random_state=random_state, verbosity=-1)),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", XGBRegressor(n_estimators=220, learning_rate=0.035, max_depth=3, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=random_state, n_jobs=-1)),
            ]
        ),
    }


def tune_best_family(
    *,
    family: str,
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    trials: int,
    random_state: int,
) -> tuple[optuna.Study, Any]:
    x_train = data.iloc[train_idx][feature_cols]
    y_train = data.iloc[train_idx][target_col].to_numpy(dtype=float)
    x_val = data.iloc[validation_idx][feature_cols]
    y_val = data.iloc[validation_idx][target_col].to_numpy(dtype=float)

    def objective(trial: optuna.Trial) -> float:
        model = model_for_trial(family, trial, random_state)
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        return metrics_for_predictions(y_val, pred)["score"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=max(1, int(trials)), show_progress_bar=False)
    return study, model_from_params(family, study.best_params, random_state)


def model_for_trial(family: str, trial: optuna.Trial, random_state: int) -> Any:
    if family == "ridge":
        return model_from_params(family, {"alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True)}, random_state)
    if family == "elastic_net":
        return model_from_params(
            family,
            {
                "alpha": trial.suggest_float("alpha", 1e-5, 0.05, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 0.95),
            },
            random_state,
        )
    if family == "random_forest":
        return model_from_params(
            family,
            {
                "n_estimators": trial.suggest_int("n_estimators", 120, 420),
                "max_features": trial.suggest_float("max_features", 0.35, 0.95),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 30),
                "max_depth": trial.suggest_int("max_depth", 3, 18),
            },
            random_state,
        )
    if family == "extra_trees":
        return model_from_params(
            family,
            {
                "n_estimators": trial.suggest_int("n_estimators", 160, 500),
                "max_features": trial.suggest_float("max_features", 0.35, 1.0),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 25),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
            },
            random_state,
        )
    if family == "hist_gradient_boosting":
        return model_from_params(
            family,
            {
                "max_iter": trial.suggest_int("max_iter", 80, 420),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.12, log=True),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 45),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-5, 2.0, log=True),
            },
            random_state,
        )
    if family == "lightgbm":
        return model_from_params(
            family,
            {
                "n_estimators": trial.suggest_int("n_estimators", 80, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.12, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 8, 64),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.55, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            },
            random_state,
        )
    if family == "xgboost":
        return model_from_params(
            family,
            {
                "n_estimators": trial.suggest_int("n_estimators", 80, 450),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.12, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 7),
                "subsample": trial.suggest_float("subsample", 0.55, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            },
            random_state,
        )
    raise ValueError(f"Unsupported model family {family}.")


def model_from_params(family: str, params: dict[str, Any], random_state: int) -> Any:
    if family == "ridge":
        model = Ridge(alpha=float(params.get("alpha", 1.0)), random_state=random_state)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)])
    if family == "elastic_net":
        model = ElasticNet(
            alpha=float(params.get("alpha", 0.0005)),
            l1_ratio=float(params.get("l1_ratio", 0.25)),
            max_iter=8000,
            random_state=random_state,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)])
    if family == "random_forest":
        model = RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 180)),
            max_features=float(params.get("max_features", 0.65)),
            min_samples_leaf=int(params.get("min_samples_leaf", 10)),
            max_depth=int(params["max_depth"]) if "max_depth" in params else None,
            n_jobs=-1,
            random_state=random_state,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    if family == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=int(params.get("n_estimators", 220)),
            max_features=float(params.get("max_features", 0.75)),
            min_samples_leaf=int(params.get("min_samples_leaf", 8)),
            max_depth=int(params["max_depth"]) if "max_depth" in params else None,
            n_jobs=-1,
            random_state=random_state,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    if family == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            max_iter=int(params.get("max_iter", 220)),
            learning_rate=float(params.get("learning_rate", 0.035)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 21)),
            l2_regularization=float(params.get("l2_regularization", 0.05)),
            random_state=random_state,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    if family == "lightgbm":
        model = LGBMRegressor(
            n_estimators=int(params.get("n_estimators", 250)),
            learning_rate=float(params.get("learning_rate", 0.035)),
            num_leaves=int(params.get("num_leaves", 24)),
            min_child_samples=int(params.get("min_child_samples", 40)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            reg_lambda=float(params.get("reg_lambda", 0.0)),
            random_state=random_state,
            verbosity=-1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    if family == "xgboost":
        model = XGBRegressor(
            n_estimators=int(params.get("n_estimators", 220)),
            learning_rate=float(params.get("learning_rate", 0.035)),
            max_depth=int(params.get("max_depth", 3)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    raise ValueError(f"Unsupported model family {family}.")


def metrics_for_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"mae_bps": math.inf, "rmse_bps": math.inf, "directional_accuracy": 0.0, "up_precision": None, "down_precision": None, "score": math.inf}
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    nonzero = true_direction != 0
    directional_accuracy = float((true_direction[nonzero] == pred_direction[nonzero]).mean()) if nonzero.any() else 0.0
    pred_up = pred_direction > 0
    pred_down = pred_direction < 0
    up_precision = float((true_direction[pred_up] > 0).mean()) if pred_up.any() else None
    down_precision = float((true_direction[pred_down] < 0).mean()) if pred_down.any() else None
    mae_bps = float(mean_absolute_error(y_true, y_pred) * 10000.0)
    rmse_bps = float(math.sqrt(mean_squared_error(y_true, y_pred)) * 10000.0)
    score = mae_bps - directional_accuracy * 4.0
    return {
        "mae_bps": mae_bps,
        "rmse_bps": rmse_bps,
        "directional_accuracy": directional_accuracy,
        "up_precision": up_precision,
        "down_precision": down_precision,
        "score": score,
    }


def build_horizon_validation_tables(
    *,
    model: Any,
    data: pd.DataFrame,
    feature_cols: list[str],
    test_idx: np.ndarray,
    horizons: tuple[int, ...],
    prices: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    x_test = data.iloc[test_idx][feature_cols]
    predicted_primary = model.predict(x_test)
    latest_close = data.iloc[test_idx]["price_close"].to_numpy(dtype=float)
    for horizon in horizons:
        target_col = f"target_return_{horizon}m"
        target_price_col = f"target_price_{horizon}m"
        if target_col not in data.columns:
            continue
        actual_return = data.iloc[test_idx][target_col].to_numpy(dtype=float)
        actual_price = data.iloc[test_idx][target_price_col].to_numpy(dtype=float)
        if horizon == 15:
            pred_return = predicted_primary
        else:
            scale = max(1.0, horizon / 15.0)
            pred_return = predicted_primary * scale
        predicted_price = latest_close * np.exp(pred_return)
        table = pd.DataFrame(
            {
                "as_of": data.iloc[test_idx].index.astype(str),
                "horizon_minutes": horizon,
                "as_of_price": latest_close,
                "predicted_return_bps": pred_return * 10000.0,
                "actual_return_bps": actual_return * 10000.0,
                "predicted_price": predicted_price,
                "actual_price": actual_price,
                "direction_correct": np.sign(pred_return) == np.sign(actual_return),
                "absolute_error_bps": np.abs((predicted_price - actual_price) / latest_close) * 10000.0,
            }
        )
        tables[str(horizon)] = table.replace([np.inf, -np.inf], np.nan).dropna()
    return tables


def build_chronos_comparison(
    *,
    args: argparse.Namespace,
    prices: pd.DataFrame,
    data: pd.DataFrame,
    validation_idx: np.ndarray,
    test_idx: np.ndarray,
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    if not bool(args.enable_chronos_compare):
        return {"enabled": False, "status": "disabled"}
    close = prices["close"].astype(float)
    price_positions = {pd.Timestamp(timestamp): position for position, timestamp in enumerate(prices.index)}
    stride = max(1, int(args.chronos_validation_stride))
    candidate_indices = np.r_[validation_idx[::stride], test_idx[::stride]]
    max_evals = max(1, int(args.chronos_max_evals))
    if len(candidate_indices) > max_evals:
        candidate_indices = candidate_indices[-max_evals:]
    forecast_hours = tuple(float(minutes) / 60.0 for minutes in horizons)
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    interval_minutes = infer_interval_minutes(prices.index)
    for row_idx in candidate_indices:
        as_of = pd.Timestamp(data.index[int(row_idx)])
        price_pos = price_positions.get(as_of)
        if price_pos is None or price_pos < 30:
            continue
        start_pos = max(0, price_pos - int(args.chronos_context_rows) + 1)
        context = prices.iloc[start_pos : price_pos + 1]
        log_step(f"chronos anchor as_of={as_of.isoformat()} context_rows={len(context)}")
        forecast = build_chronos_forecast(
            price_bars=price_bars_for_chronos(context),
            forecast_hours=forecast_hours,
            data_interval=args.interval,
            model_name=args.chronos_model,
            context_rows=int(args.chronos_context_rows),
            num_samples=int(args.chronos_num_samples),
            refresh_seconds=0,
            enabled=True,
        )
        if forecast.get("status") != "ok":
            reason = str(forecast.get("reason") or forecast.get("status") or "unknown")
            if reason not in failures:
                failures.append(reason)
            continue
        log_step(f"chronos anchor complete status={forecast.get('status')}")
        points = {
            int(round(float(point.get("horizon_hours", 0.0)) * 60.0)): point
            for point in forecast.get("horizon_points", [])
            if isinstance(point, dict)
        }
        as_of_price = float(close.iloc[price_pos])
        for horizon in horizons:
            bars_ahead = max(1, int(round(float(horizon) / interval_minutes)))
            future_pos = price_pos + bars_ahead
            point = points.get(int(horizon))
            if future_pos >= len(close) or not point:
                continue
            predicted_price = _safe_float(point.get("predicted_price"))
            if predicted_price is None:
                continue
            actual_price = float(close.iloc[future_pos])
            predicted_return = math.log(predicted_price / as_of_price) if predicted_price > 0 and as_of_price > 0 else np.nan
            actual_return = math.log(actual_price / as_of_price) if actual_price > 0 and as_of_price > 0 else np.nan
            records.append(
                {
                    "as_of": as_of.isoformat(),
                    "horizon_minutes": int(horizon),
                    "as_of_price": as_of_price,
                    "predicted_return_bps": predicted_return * 10000.0,
                    "actual_return_bps": actual_return * 10000.0,
                    "predicted_price": predicted_price,
                    "actual_price": actual_price,
                    "direction_correct": bool(np.sign(predicted_return) == np.sign(actual_return)),
                    "absolute_error_bps": abs((predicted_price - actual_price) / as_of_price) * 10000.0,
                }
            )
    predictions = pd.DataFrame(records)
    if predictions.empty:
        return {
            "enabled": True,
            "status": "failed",
            "model": args.chronos_model,
            "rows": 0,
            "failures": failures,
        }
    summary = (
        predictions.groupby("horizon_minutes", as_index=False)
        .agg(
            rows=("absolute_error_bps", "size"),
            mae_bps=("absolute_error_bps", "mean"),
            median_abs_error_bps=("absolute_error_bps", "median"),
            directional_accuracy=("direction_correct", "mean"),
            mean_predicted_return_bps=("predicted_return_bps", "mean"),
            mean_actual_return_bps=("actual_return_bps", "mean"),
        )
        .sort_values("horizon_minutes")
    )
    return {
        "enabled": True,
        "status": "ok",
        "model": args.chronos_model,
        "context_rows": int(args.chronos_context_rows),
        "num_samples": int(args.chronos_num_samples),
        "validation_stride": stride,
        "rows": int(len(predictions)),
        "predictions": predictions,
        "summary": summary,
        "failures": failures,
    }


def price_bars_for_chronos(frame: pd.DataFrame) -> list[dict[str, Any]]:
    normalized = normalize_price_frame(frame, target_column="close").sort_index()
    rows: list[dict[str, Any]] = []
    for timestamp, row in normalized.iterrows():
        close = float(row["close"])
        rows.append(
            {
                "timestamp": pd.Timestamp(timestamp).isoformat(),
                "open": float(row.get("open", close)),
                "high": float(row.get("high", close)),
                "low": float(row.get("low", close)),
                "close": close,
                "volume": float(row.get("volume", 0.0)),
            }
        )
    return rows


def _safe_float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def write_outputs(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    prices: pd.DataFrame,
    data: pd.DataFrame,
    metadata: dict[str, Any],
    feature_cols: list[str],
    candidate_results: list[CandidateResult],
    best_three: list[CandidateResult],
    best_family: str,
    study: optuna.Study,
    tuned_primary_metrics: dict[str, float | None],
    validation_tables: dict[str, pd.DataFrame],
    chronos_comparison: dict[str, Any],
    tuned_model: Any,
) -> None:
    candidates_df = pd.DataFrame([asdict(row) for row in candidate_results])
    best_three_df = pd.DataFrame([asdict(row) for row in best_three])
    candidates_df.to_csv(output_dir / "candidate_model_results.csv", index=False)
    best_three_df.to_csv(output_dir / "best_3_model_candidates.csv", index=False)
    validation_summary = summarize_validation_tables(validation_tables)
    validation_summary.to_csv(output_dir / "validation_summary_by_horizon.csv", index=False)
    chronos_summary = pd.DataFrame()
    if chronos_comparison.get("status") == "ok":
        chronos_predictions = chronos_comparison["predictions"]
        chronos_summary = chronos_comparison["summary"]
        chronos_predictions.to_csv(output_dir / "chronos_validation_predictions.csv", index=False)
        chronos_summary.to_csv(output_dir / "chronos_validation_summary_by_horizon.csv", index=False)
    for horizon, table in validation_tables.items():
        table.to_csv(output_dir / f"validation_predictions_{horizon}m.csv", index=False)
    joblib.dump(tuned_model, output_dir / "tuned_spy_forecaster.joblib")

    plot_paths = write_validation_plots(output_dir, prices, validation_tables)
    payload = {
        "ticker": args.ticker.upper(),
        "provider_metadata": metadata,
        "rows": {"prices": int(len(prices)), "supervised": int(len(data))},
        "feature_count": len(feature_cols),
        "primary_horizon_minutes": args.primary_horizon_minutes,
        "candidate_results": [asdict(row) for row in candidate_results],
        "best_three": [asdict(row) for row in best_three],
        "tuned_model": {
            "family": best_family,
            "selection_note": "Optuna tunes the best trainable model family from the selection split; baseline rows are references only.",
            "optuna_best_value": float(study.best_value),
            "optuna_best_params": study.best_params,
            "holdout_primary_metrics": tuned_primary_metrics,
        },
        "validation_summary": validation_summary.to_dict(orient="records"),
        "chronos_comparison": chronos_payload_for_json(chronos_comparison),
        "plots": plot_paths,
    }
    (output_dir / "report.json").write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    write_markdown_report(output_dir, payload, best_three_df, validation_summary, chronos_summary, plot_paths)


def chronos_payload_for_json(chronos_comparison: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in chronos_comparison.items() if key not in {"predictions", "summary"}}
    if isinstance(chronos_comparison.get("summary"), pd.DataFrame):
        payload["summary"] = chronos_comparison["summary"].to_dict(orient="records")
    return payload


def summarize_validation_tables(validation_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon, table in validation_tables.items():
        if table.empty:
            continue
        rows.append(
            {
                "horizon_minutes": int(horizon),
                "rows": int(len(table)),
                "mae_bps": float(table["absolute_error_bps"].mean()),
                "median_abs_error_bps": float(table["absolute_error_bps"].median()),
                "directional_accuracy": float(table["direction_correct"].mean()),
                "mean_predicted_return_bps": float(table["predicted_return_bps"].mean()),
                "mean_actual_return_bps": float(table["actual_return_bps"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("horizon_minutes")


def write_validation_plots(output_dir: Path, prices: pd.DataFrame, validation_tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    close = prices["close"].astype(float)
    validation_times: list[pd.Timestamp] = []
    for table in validation_tables.values():
        if table.empty:
            continue
        validation_times.extend(pd.to_datetime(table["as_of"], errors="coerce").dropna().tolist())
    if validation_times:
        start_time = min(validation_times) - pd.Timedelta(minutes=90)
        end_time = max(validation_times) + pd.Timedelta(minutes=90)
        history = close[(close.index >= start_time) & (close.index <= end_time)]
        if history.empty:
            history = close.tail(900)
    else:
        history = close.tail(900)

    fig, ax = plt.subplots(figsize=(14, 7))
    history = history.sort_index()
    x_positions = np.arange(len(history))
    ax.plot(x_positions, history.values, color="#1f2937", linewidth=1.4, label="Actual SPY close")
    colors = {5: "#2563eb", 10: "#16a34a", 15: "#dc2626", 30: "#7c3aed"}
    for horizon, table in validation_tables.items():
        if table.empty:
            continue
        h = int(horizon)
        sample = table.tail(120).copy()
        sample["as_of"] = pd.to_datetime(sample["as_of"], errors="coerce")
        sample = sample.dropna(subset=["as_of", "predicted_price"])
        sample_positions = np.searchsorted(history.index.to_numpy(), sample["as_of"].to_numpy())
        sample_positions = np.clip(sample_positions, 0, max(len(history) - 1, 0))
        ax.scatter(sample_positions, sample["predicted_price"], s=20, color=colors.get(h, "#64748b"), alpha=0.75, label=f"Predicted {h}m")
    ax.set_title("SPY Validation Predictions vs Actual Price")
    ax.set_xlabel("Trading-bar sequence; closed-market gaps compressed")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    if len(history) > 0:
        tick_count = min(8, len(history))
        tick_positions = np.linspace(0, len(history) - 1, tick_count, dtype=int)
        tick_labels = [pd.Timestamp(history.index[pos]).strftime("%m-%d %H:%M") for pos in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.legend(loc="best")
    fig.tight_layout()
    png = plot_dir / "validation_predictions_vs_actual.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    paths["validation_predictions_vs_actual"] = str(png)

    summary_rows = []
    for horizon, table in validation_tables.items():
        if table.empty:
            continue
        summary_rows.append((int(horizon), table["absolute_error_bps"].dropna().to_numpy()))
    if summary_rows:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot([values for _, values in summary_rows], tick_labels=[f"{h}m" for h, _ in summary_rows], showfliers=False)
        ax.set_title("SPY Validation Absolute Error Distribution")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Absolute error, bps")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        png = plot_dir / "validation_error_distribution.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        paths["validation_error_distribution"] = str(png)
    return paths


def write_markdown_report(
    output_dir: Path,
    payload: dict[str, Any],
    best_three: pd.DataFrame,
    validation_summary: pd.DataFrame,
    chronos_summary: pd.DataFrame,
    plot_paths: dict[str, str],
) -> None:
    tuned = payload["tuned_model"]
    lines = [
        "# SPY Optuna Forecaster Report",
        "",
        f"Ticker: `{payload['ticker']}`",
        f"Primary tuning horizon: `{payload['primary_horizon_minutes']} minutes`",
        f"Rows: prices `{payload['rows']['prices']}`, supervised `{payload['rows']['supervised']}`",
        f"Feature count: `{payload['feature_count']}`",
        "",
        "## Best 3 Model Candidates",
        "",
        best_three.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Final Tuned Model",
        "",
        "The final row is evaluated on a later chronological holdout. It is not the same split used to rank the candidate table.",
        "",
        pd.DataFrame(
            [
                {
                    "model_family": tuned["family"],
                    "optuna_best_value": tuned["optuna_best_value"],
                    **{f"param_{key}": value for key, value in tuned["optuna_best_params"].items()},
                    **{f"holdout_{key}": value for key, value in tuned["holdout_primary_metrics"].items()},
                }
            ]
        ).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Validation Table",
        "",
        validation_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Chronos Comparison",
        "",
    ]
    chronos = payload.get("chronos_comparison") or {}
    if chronos.get("status") == "ok" and not chronos_summary.empty:
        lines.extend(
            [
                f"Model: `{chronos.get('model')}`",
                f"Walk-forward stride: every `{chronos.get('validation_stride')}` validation bars",
                "",
                chronos_summary.to_markdown(index=False, floatfmt=".4f"),
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"Chronos status: `{chronos.get('status', 'not_run')}`",
                f"Reason/failures: `{chronos.get('failures') or chronos.get('reason') or ''}`",
                "",
            ]
        )
    lines.extend(
        [
        "## Validation Plots",
        "",
        ]
    )
    for name, path in plot_paths.items():
        lines.append(f"### {name.replace('_', ' ').title()}")
        lines.append("")
        lines.append(f"![{name}]({Path(path).resolve()})")
        lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
