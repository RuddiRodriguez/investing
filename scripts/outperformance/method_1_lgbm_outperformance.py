"""Predict whether a stock will outperform a benchmark with LightGBM."""

import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import yfinance as yf
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit


STOCK_FILE = "stock_prices.csv"
BENCHMARK_FILE = "benchmark_prices.csv"
BENCHMARK_TICKER = "SPY"
PRICE_COLUMN = "Close"
VOLUME_COLUMN = "Volume"
FUTURE_DAYS = 10
OUTPERFORM_THRESHOLD = 0.02
UNDERPERFORM_THRESHOLD = -0.02
OUTPERFORM_PROBABILITY = 0.55
UNDERPERFORM_PROBABILITY = 0.55
OPTUNA_TRIALS = 25
MIN_TRAIN_SIZE = 252
TEST_RATIO = 0.25
TRANSACTION_COST = 0.001

OUTPERFORM_SIGNAL = "OUTPERFORM"
HOLD_SIGNAL = "HOLD"
UNDERPERFORM_SIGNAL = "UNDERPERFORM"
CLASS_BY_SIGNAL = {
    UNDERPERFORM_SIGNAL: -1,
    HOLD_SIGNAL: 0,
    OUTPERFORM_SIGNAL: 1,
}

FEATURE_COLUMNS = [
    "Stock Return 5",
    "Stock Return 20",
    "Benchmark Return 5",
    "Benchmark Return 20",
    "Relative Return 5",
    "Relative Return 20",
    "Relative Strength Return 20",
    "Stock Distance 50",
    "Stock Distance 200",
    "Benchmark Distance 50",
    "Benchmark Distance 200",
    "Stock Volatility 20",
    "Benchmark Volatility 20",
    "Relative Volatility 20",
    "Stock RSI",
    "Benchmark RSI",
    "Stock Drawdown 60",
    "Benchmark Drawdown 60",
    "Stock Volume Change 20",
    "Benchmark Volume Change 20",
]


def read_prices(csv_file):
    prices = pd.read_csv(csv_file)
    prices["Date"] = pd.to_datetime(prices["Date"], utc=True).dt.date
    return prices


def download_benchmark_prices(start_date):
    benchmark = yf.Ticker(BENCHMARK_TICKER)
    benchmark_prices = benchmark.history(
        start=start_date,
        auto_adjust=True,
    )
    benchmark_prices.to_csv(BENCHMARK_FILE)
    return read_prices(BENCHMARK_FILE)


def prepare_data(stock_prices, benchmark_prices):
    stock = stock_prices[["Date", PRICE_COLUMN, VOLUME_COLUMN]].copy()
    benchmark = benchmark_prices[["Date", PRICE_COLUMN, VOLUME_COLUMN]].copy()

    stock = stock.rename(
        columns={
            PRICE_COLUMN: "Stock Close",
            VOLUME_COLUMN: "Stock Volume",
        }
    )
    benchmark = benchmark.rename(
        columns={
            PRICE_COLUMN: "Benchmark Close",
            VOLUME_COLUMN: "Benchmark Volume",
        }
    )

    prices = pd.merge(stock, benchmark, on="Date", how="inner")
    return prices.sort_values("Date").reset_index(drop=True)


def add_rsi(prices, close_column, rsi_column):
    price_change = prices[close_column].diff()
    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.rolling(14).mean()
    average_loss = losses.rolling(14).mean()

    prices[rsi_column] = 100 - (100 / (1 + average_gain / average_loss))
    return prices


def add_single_asset_features(prices, close_column, volume_column, prefix):
    prices[f"{prefix} Return 1"] = prices[close_column].pct_change()
    prices[f"{prefix} Return 5"] = prices[close_column].pct_change(5)
    prices[f"{prefix} Return 20"] = prices[close_column].pct_change(20)

    prices[f"{prefix} Moving Average 50"] = prices[close_column].rolling(50).mean()
    prices[f"{prefix} Moving Average 200"] = prices[close_column].rolling(200).mean()
    prices[f"{prefix} Distance 50"] = prices[close_column] / prices[f"{prefix} Moving Average 50"] - 1
    prices[f"{prefix} Distance 200"] = prices[close_column] / prices[f"{prefix} Moving Average 200"] - 1

    prices[f"{prefix} Volatility 20"] = prices[f"{prefix} Return 1"].rolling(20).std()
    prices[f"{prefix} Volume Change 20"] = prices[volume_column].pct_change(20)

    prices[f"{prefix} High 60"] = prices[close_column].rolling(60).max()
    prices[f"{prefix} Drawdown 60"] = prices[close_column] / prices[f"{prefix} High 60"] - 1

    prices = add_rsi(prices, close_column, f"{prefix} RSI")
    return prices


def add_features(prices):
    prices = add_single_asset_features(prices, "Stock Close", "Stock Volume", "Stock")
    prices = add_single_asset_features(prices, "Benchmark Close", "Benchmark Volume", "Benchmark")

    prices["Relative Return 5"] = prices["Stock Return 5"] - prices["Benchmark Return 5"]
    prices["Relative Return 20"] = prices["Stock Return 20"] - prices["Benchmark Return 20"]
    prices["Relative Volatility 20"] = prices["Stock Volatility 20"] - prices["Benchmark Volatility 20"]

    prices["Relative Strength"] = prices["Stock Close"] / prices["Benchmark Close"]
    prices["Relative Strength Return 20"] = prices["Relative Strength"].pct_change(20)
    return prices


def add_target(prices):
    prices["Stock Future Close"] = prices["Stock Close"].shift(-FUTURE_DAYS)
    prices["Benchmark Future Close"] = prices["Benchmark Close"].shift(-FUTURE_DAYS)

    prices["Stock Future Return"] = prices["Stock Future Close"] / prices["Stock Close"] - 1
    prices["Benchmark Future Return"] = prices["Benchmark Future Close"] / prices["Benchmark Close"] - 1
    prices["Future Relative Return"] = prices["Stock Future Return"] - prices["Benchmark Future Return"]

    prices["Target"] = np.select(
        [
            prices["Future Relative Return"] < UNDERPERFORM_THRESHOLD,
            prices["Future Relative Return"] > OUTPERFORM_THRESHOLD,
        ],
        [CLASS_BY_SIGNAL[UNDERPERFORM_SIGNAL], CLASS_BY_SIGNAL[OUTPERFORM_SIGNAL]],
        default=CLASS_BY_SIGNAL[HOLD_SIGNAL],
    )
    return prices


def build_model(settings):
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


def get_class_probabilities(model, features):
    probability_values = model.predict_proba(features)[0]
    probability_by_target = {
        int(target): probability
        for target, probability in zip(model.classes_, probability_values)
    }
    return {
        UNDERPERFORM_SIGNAL: probability_by_target.get(CLASS_BY_SIGNAL[UNDERPERFORM_SIGNAL], 0.0),
        HOLD_SIGNAL: probability_by_target.get(CLASS_BY_SIGNAL[HOLD_SIGNAL], 0.0),
        OUTPERFORM_SIGNAL: probability_by_target.get(CLASS_BY_SIGNAL[OUTPERFORM_SIGNAL], 0.0),
    }


def get_signal(probability_outperform, probability_underperform):
    if probability_outperform >= OUTPERFORM_PROBABILITY and probability_outperform > probability_underperform:
        return OUTPERFORM_SIGNAL

    if probability_underperform >= UNDERPERFORM_PROBABILITY and probability_underperform > probability_outperform:
        return UNDERPERFORM_SIGNAL

    return HOLD_SIGNAL


def get_strategy_return(signal, stock_return, benchmark_return):
    if signal == OUTPERFORM_SIGNAL:
        return stock_return - TRANSACTION_COST

    return benchmark_return


def prepare_training_data(stock_prices, benchmark_prices):
    prices = prepare_data(stock_prices, benchmark_prices)
    prices = add_features(prices)
    prices = add_target(prices)
    return prices.dropna().reset_index(drop=True)


def predict_outperformance(stock_prices, benchmark_prices):
    prices = prepare_training_data(stock_prices, benchmark_prices)
    model = fit_model(prices, tune_model_settings(prices))

    latest_row = prices.tail(1)
    probabilities = get_class_probabilities(model, latest_row[FEATURE_COLUMNS])
    signal = get_signal(probabilities[OUTPERFORM_SIGNAL], probabilities[UNDERPERFORM_SIGNAL])

    print(f"Probability of outperforming by more than {OUTPERFORM_THRESHOLD:.0%}: {probabilities[OUTPERFORM_SIGNAL]:.2%}")
    print(f"Probability of underperforming by more than {abs(UNDERPERFORM_THRESHOLD):.0%}: {probabilities[UNDERPERFORM_SIGNAL]:.2%}")
    return signal


def check_predictions(prices):
    test_start = max(int(len(prices) * (1 - TEST_RATIO)), MIN_TRAIN_SIZE)
    if test_start >= len(prices):
        raise ValueError("Not enough data to reserve an untouched test period.")

    best_settings = tune_model_settings(prices.iloc[:test_start])

    results = []
    for current_index in range(test_start, len(prices), FUTURE_DAYS):
        train_prices = prices.iloc[:current_index]
        if train_prices["Target"].nunique() < 2:
            continue

        current_row = prices.iloc[[current_index]]
        model = fit_model(train_prices, best_settings)
        probabilities = get_class_probabilities(model, current_row[FEATURE_COLUMNS])
        signal = get_signal(probabilities[OUTPERFORM_SIGNAL], probabilities[UNDERPERFORM_SIGNAL])

        stock_return = current_row["Stock Future Return"].iat[0]
        benchmark_return = current_row["Benchmark Future Return"].iat[0]
        strategy_return = get_strategy_return(signal, stock_return, benchmark_return)

        results.append(
            {
                "Date": current_row["Date"].iat[0],
                "Stock Close": current_row["Stock Close"].iat[0],
                "Benchmark Close": current_row["Benchmark Close"].iat[0],
                "Probability Outperform": probabilities[OUTPERFORM_SIGNAL],
                "Probability Hold": probabilities[HOLD_SIGNAL],
                "Probability Underperform": probabilities[UNDERPERFORM_SIGNAL],
                "Signal": signal,
                "Prediction": CLASS_BY_SIGNAL[signal],
                "Target": current_row["Target"].iat[0],
                "Correct": CLASS_BY_SIGNAL[signal] == current_row["Target"].iat[0],
                "Stock Future Return": stock_return,
                "Benchmark Future Return": benchmark_return,
                "Future Relative Return": current_row["Future Relative Return"].iat[0],
                "Strategy Return": strategy_return,
            }
        )

    results = pd.DataFrame(results)
    results["Strategy Equity"] = (1 + results["Strategy Return"]).cumprod()
    results["Stock Equity"] = (1 + results["Stock Future Return"]).cumprod()
    results["Benchmark Equity"] = (1 + results["Benchmark Future Return"]).cumprod()
    results["Relative Equity"] = results["Strategy Equity"] / results["Benchmark Equity"]
    return results


def get_max_drawdown(equity):
    drawdown = equity / equity.cummax() - 1
    return drawdown.min()


def print_results(results):
    outperform_signals = results[results["Signal"] == OUTPERFORM_SIGNAL]
    underperform_signals = results[results["Signal"] == UNDERPERFORM_SIGNAL]
    hold_signals = results[results["Signal"] == HOLD_SIGNAL]
    strong_signals = results[results["Signal"] != HOLD_SIGNAL]

    accuracy = results["Correct"].mean()
    strong_accuracy = strong_signals["Correct"].mean()
    average_relative_return = results["Future Relative Return"].mean()
    outperform_average_relative_return = outperform_signals["Future Relative Return"].mean()
    underperform_average_relative_return = underperform_signals["Future Relative Return"].mean()

    strategy_total_return = results["Strategy Equity"].iloc[-1] - 1
    stock_total_return = results["Stock Equity"].iloc[-1] - 1
    benchmark_total_return = results["Benchmark Equity"].iloc[-1] - 1
    relative_total_return = results["Relative Equity"].iloc[-1] - 1

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Strong signal accuracy: {strong_accuracy:.2%}")
    print(f"OUTPERFORM signals: {len(outperform_signals)}")
    print(f"HOLD signals: {len(hold_signals)}")
    print(f"UNDERPERFORM signals: {len(underperform_signals)}")
    print(f"Average relative return: {average_relative_return:.2%}")
    print(f"Average relative return after OUTPERFORM signals: {outperform_average_relative_return:.2%}")
    print(f"Average relative return after UNDERPERFORM signals: {underperform_average_relative_return:.2%}")
    print(f"Strategy total return: {strategy_total_return:.2%}")
    print(f"Stock total return: {stock_total_return:.2%}")
    print(f"Benchmark total return: {benchmark_total_return:.2%}")
    print(f"Strategy return vs benchmark: {relative_total_return:.2%}")
    print(f"Strategy max drawdown: {get_max_drawdown(results['Strategy Equity']):.2%}")
    print(f"Benchmark max drawdown: {get_max_drawdown(results['Benchmark Equity']):.2%}")


def plot_results(results):
    fig = px.scatter(
        results,
        x="Date",
        y="Stock Close",
        color="Signal",
        color_discrete_map={
            OUTPERFORM_SIGNAL: "green",
            HOLD_SIGNAL: "gray",
            UNDERPERFORM_SIGNAL: "red",
        },
        title="Outperformance Signals",
    )
    fig.show()

    fig = px.line(
        results,
        x="Date",
        y=["Strategy Equity", "Stock Equity", "Benchmark Equity"],
        title="Outperformance Strategy vs Stock vs Benchmark",
    )
    fig.show()

    fig = px.line(
        results,
        x="Date",
        y="Relative Equity",
        title="Strategy Equity Relative To Benchmark",
    )
    fig.show()


if __name__ == "__main__":
    stock_prices = read_prices(STOCK_FILE)
    start_date = stock_prices["Date"].min().isoformat()
    benchmark_prices = download_benchmark_prices(start_date)

    signal = predict_outperformance(stock_prices, benchmark_prices)
    print(signal)

    prices = prepare_training_data(stock_prices, benchmark_prices)
    results = check_predictions(prices)
    print_results(results)
    plot_results(results)
