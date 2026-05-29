"""Predict 10-day stock moves with a three-class LightGBM model."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit


CSV_FILE = "stock_prices.csv"
PRICE_COLUMN = "Close"
FUTURE_DAYS = 10
UP_RETURN_THRESHOLD = 0.02
DOWN_RETURN_THRESHOLD = -0.02
UP_PROBABILITY = 0.55
DOWN_PROBABILITY = 0.55
OPTUNA_TRIALS = 25
HOLD_SIGNAL = "HOLD"
UP_SIGNAL = "UP"
DOWN_SIGNAL = "DOWN"
MODEL_FILE = "lightgbm_stock_model.pkl"
SETTINGS_FILE = "lightgbm_stock_model_settings.json"
CLASS_BY_SIGNAL = {DOWN_SIGNAL: -1, HOLD_SIGNAL: 0, UP_SIGNAL: 1}
FEATURE_COLUMNS = [
    "Return 1",
    "Return 5",
    "Return 10",
    "Return 20",
    "Distance 50",
    "Distance 200",
    "Volatility 20",
    "RSI",
    "Volume Change 20",
    "Distance High 20",
    "Distance Low 20",
    "Drawdown 60",
]


def read_prices(csv_file):
    return pd.read_csv(csv_file)


def add_rsi(prices):
    price_change = prices[PRICE_COLUMN].diff()
    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.rolling(14).mean()
    average_loss = losses.rolling(14).mean()

    prices["RSI"] = 100 - (100 / (1 + average_gain / average_loss))
    return prices


def add_features(prices):
    prices["Return 1"] = prices[PRICE_COLUMN].pct_change()
    prices["Return 5"] = prices[PRICE_COLUMN].pct_change(5)
    prices["Return 10"] = prices[PRICE_COLUMN].pct_change(10)
    prices["Return 20"] = prices[PRICE_COLUMN].pct_change(20)

    prices["Moving Average 50"] = prices[PRICE_COLUMN].rolling(50).mean()
    prices["Moving Average 200"] = prices[PRICE_COLUMN].rolling(200).mean()
    prices["Distance 50"] = prices[PRICE_COLUMN] / prices["Moving Average 50"] - 1
    prices["Distance 200"] = prices[PRICE_COLUMN] / prices["Moving Average 200"] - 1

    prices["Volatility 20"] = prices["Return 1"].rolling(20).std()
    prices["Volume Change 20"] = prices["Volume"].pct_change(20)

    prices["High 20"] = prices[PRICE_COLUMN].rolling(20).max()
    prices["Low 20"] = prices[PRICE_COLUMN].rolling(20).min()
    prices["High 60"] = prices[PRICE_COLUMN].rolling(60).max()
    prices["Distance High 20"] = prices[PRICE_COLUMN] / prices["High 20"] - 1
    prices["Distance Low 20"] = prices[PRICE_COLUMN] / prices["Low 20"] - 1
    prices["Drawdown 60"] = prices[PRICE_COLUMN] / prices["High 60"] - 1

    prices = add_rsi(prices)
    return prices


def add_target(prices):
    prices["Future Close"] = prices[PRICE_COLUMN].shift(-FUTURE_DAYS)
    prices["Future Return"] = prices["Future Close"] / prices[PRICE_COLUMN] - 1
    prices["Target"] = np.select(
        [
            prices["Future Return"] < DOWN_RETURN_THRESHOLD,
            prices["Future Return"] > UP_RETURN_THRESHOLD,
        ],
        [CLASS_BY_SIGNAL[DOWN_SIGNAL], CLASS_BY_SIGNAL[UP_SIGNAL]],
        default=CLASS_BY_SIGNAL[HOLD_SIGNAL],
    )
    return prices


def build_model(settings):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        **settings,
        random_state=1,
        verbosity=-1,
        n_jobs=1,
        force_col_wise=True,
    )


def fit_model(prices, settings):
    model = build_model(settings)
    model.fit(prices[FEATURE_COLUMNS], prices["Target"])
    return model


def tune_model_settings(prices):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    split = TimeSeriesSplit(n_splits=5)
    features = prices[FEATURE_COLUMNS]
    target = prices["Target"]

    def objective(trial):
        settings = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 3, 5, 8]),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

        scores = []
        for train_index, test_index in split.split(features):
            train_features = features.iloc[train_index]
            test_features = features.iloc[test_index]
            train_target = target.iloc[train_index]
            test_target = target.iloc[test_index]

            model = build_model(settings)
            model.fit(train_features, train_target)
            prediction = model.predict(test_features)
            score = balanced_accuracy_score(test_target, prediction)
            scores.append(score)

        return sum(scores) / len(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)

    print("Best settings:")
    print(study.best_params)

    return study.best_params


def train_model(prices):
    settings = tune_model_settings(prices)
    return fit_model(prices, settings)


def train_model_with_settings(prices):
    settings = tune_model_settings(prices)
    model = fit_model(prices, settings)
    return model, settings


def save_model_artifacts(model, settings, model_file=MODEL_FILE, settings_file=SETTINGS_FILE):
    joblib.dump(model, model_file)
    with open(settings_file, "w", encoding="utf-8") as file_handle:
        json.dump(settings, file_handle, indent=2, sort_keys=True)


def load_model(model_file=MODEL_FILE):
    return joblib.load(model_file)


def get_class_probabilities(model, features):
    probability_values = model.predict_proba(features)[0]
    probability_by_target = {
        int(target): probability
        for target, probability in zip(model.classes_, probability_values)
    }
    return {
        DOWN_SIGNAL: probability_by_target.get(CLASS_BY_SIGNAL[DOWN_SIGNAL], 0.0),
        HOLD_SIGNAL: probability_by_target.get(CLASS_BY_SIGNAL[HOLD_SIGNAL], 0.0),
        UP_SIGNAL: probability_by_target.get(CLASS_BY_SIGNAL[UP_SIGNAL], 0.0),
    }


def get_signal(probability_up, probability_down):
    if probability_up >= UP_PROBABILITY and probability_up > probability_down:
        return UP_SIGNAL

    if probability_down >= DOWN_PROBABILITY and probability_down > probability_up:
        return DOWN_SIGNAL

    return HOLD_SIGNAL


def get_trade_return(signal, future_return, transaction_cost=0.0):
    if signal == UP_SIGNAL:
        return future_return - transaction_cost

    return 0.0


def predict_direction(prices):
    prices = add_features(prices)

    training_prices = add_target(prices.copy())
    training_prices = training_prices.dropna()

    model = train_model(training_prices)

    latest_prices = prices.dropna().tail(1)
    probabilities = get_class_probabilities(model, latest_prices[FEATURE_COLUMNS])

    print(
        f"Probability of more than {UP_RETURN_THRESHOLD:.0%} gain in {FUTURE_DAYS} days: "
        f"{probabilities[UP_SIGNAL]:.2%}"
    )
    print(
        f"Probability of more than {abs(DOWN_RETURN_THRESHOLD):.0%} loss in {FUTURE_DAYS} days: "
        f"{probabilities[DOWN_SIGNAL]:.2%}"
    )
    return get_signal(probabilities[UP_SIGNAL], probabilities[DOWN_SIGNAL])


if __name__ == "__main__":
    prices = read_prices(CSV_FILE)
    direction = predict_direction(prices)
    print(direction)
