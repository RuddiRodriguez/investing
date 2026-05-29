"""Run a frozen-model live forward test for the LightGBM direction model."""

import argparse
import sys
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError
from pandas.tseries.offsets import BDay


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.direction.method_5_lgbm import (
    CSV_FILE,
    FEATURE_COLUMNS,
    FUTURE_DAYS,
    HOLD_SIGNAL,
    MODEL_FILE,
    PRICE_COLUMN,
    SETTINGS_FILE,
    UP_RETURN_THRESHOLD,
    UP_SIGNAL,
    add_features,
    add_target,
    get_class_probabilities,
    get_signal,
    load_model,
    read_prices,
    save_model_artifacts,
    train_model_with_settings,
)


LIVE_PREDICTIONS_FILE = "live_predictions.csv"
LIVE_PREDICTION_COLUMNS = [
    "prediction_date",
    "close",
    "probability_up",
    "probability_down",
    "signal",
    "evaluation_date",
    "future_close",
    "future_return",
    "target_hit",
    "correct",
]


def normalize_prices(prices):
    normalized = prices.copy()
    normalized["Date"] = pd.to_datetime(normalized["Date"], utc=True).dt.tz_localize(None)
    return normalized.sort_values("Date").reset_index(drop=True)


def prepare_training_prices(prices):
    prepared = add_features(prices.copy())
    prepared = add_target(prepared)
    return prepared.dropna().reset_index(drop=True)


def build_prediction_record(prices, model):
    featured_prices = add_features(prices.copy())
    latest_prices = featured_prices.dropna().tail(1)
    if latest_prices.empty:
        raise ValueError("Not enough price history to build features for a live prediction.")

    probabilities = get_class_probabilities(model, latest_prices[FEATURE_COLUMNS])
    prediction_date = pd.to_datetime(latest_prices["Date"].iat[0])
    signal = get_signal(probabilities[UP_SIGNAL], probabilities["DOWN"])

    return {
        "prediction_date": prediction_date,
        "close": latest_prices[PRICE_COLUMN].iat[0],
        "probability_up": probabilities[UP_SIGNAL],
        "probability_down": probabilities["DOWN"],
        "signal": signal,
        "evaluation_date": prediction_date + BDay(FUTURE_DAYS),
        "future_close": pd.NA,
        "future_return": pd.NA,
        "target_hit": pd.NA,
        "correct": pd.NA,
    }


def load_live_predictions(log_file=LIVE_PREDICTIONS_FILE):
    log_path = Path(log_file)
    if not log_path.exists():
        return pd.DataFrame(columns=LIVE_PREDICTION_COLUMNS)

    try:
        predictions = pd.read_csv(log_path)
    except EmptyDataError:
        return pd.DataFrame(columns=LIVE_PREDICTION_COLUMNS)

    if predictions.empty:
        return pd.DataFrame(columns=LIVE_PREDICTION_COLUMNS)

    for column in ["prediction_date", "evaluation_date"]:
        predictions[column] = pd.to_datetime(predictions[column], errors="coerce")

    return predictions


def save_live_predictions(predictions, log_file=LIVE_PREDICTIONS_FILE):
    ordered = predictions.copy()
    ordered = ordered[LIVE_PREDICTION_COLUMNS]
    ordered = ordered.sort_values("prediction_date").reset_index(drop=True)
    ordered.to_csv(log_file, index=False, date_format="%Y-%m-%d")


def append_prediction(predictions, record):
    updated = predictions.copy()
    prediction_date = pd.to_datetime(record["prediction_date"])
    existing_mask = pd.to_datetime(updated.get("prediction_date", pd.Series(dtype="datetime64[ns]"))) == prediction_date

    record_frame = pd.DataFrame([record], columns=LIVE_PREDICTION_COLUMNS)
    if not updated.empty and existing_mask.any():
        updated.loc[existing_mask, LIVE_PREDICTION_COLUMNS] = record_frame.iloc[0].values
        return updated

    return pd.concat([updated, record_frame], ignore_index=True)


def update_completed_predictions(predictions, prices):
    updated = predictions.copy()
    if updated.empty:
        return updated

    dated_prices = normalize_prices(prices)
    price_index_by_date = {
        pd.Timestamp(date_value): index
        for index, date_value in enumerate(dated_prices["Date"])
    }

    for row_index, row in updated.iterrows():
        if pd.notna(row["future_close"]):
            continue

        prediction_date = pd.to_datetime(row["prediction_date"])
        if prediction_date not in price_index_by_date:
            continue

        prediction_position = price_index_by_date[prediction_date]
        future_position = prediction_position + FUTURE_DAYS
        if future_position >= len(dated_prices):
            continue

        future_row = dated_prices.iloc[future_position]
        future_close = future_row[PRICE_COLUMN]
        future_return = future_close / row["close"] - 1
        target_hit = future_return > UP_RETURN_THRESHOLD

        updated.at[row_index, "evaluation_date"] = future_row["Date"]
        updated.at[row_index, "future_close"] = future_close
        updated.at[row_index, "future_return"] = future_return
        updated.at[row_index, "target_hit"] = target_hit
        if row["signal"] == UP_SIGNAL:
            updated.at[row_index, "correct"] = target_hit

    return updated


def train_and_freeze_model(csv_file=CSV_FILE, model_file=MODEL_FILE, settings_file=SETTINGS_FILE):
    prices = normalize_prices(read_prices(csv_file))
    training_prices = prepare_training_prices(prices)
    model, settings = train_model_with_settings(training_prices)
    save_model_artifacts(model, settings, model_file=model_file, settings_file=settings_file)
    print(f"Saved model to {model_file}")
    print(f"Saved settings to {settings_file}")


def predict_and_log(
    csv_file=CSV_FILE,
    model_file=MODEL_FILE,
    log_file=LIVE_PREDICTIONS_FILE,
):
    prices = normalize_prices(read_prices(csv_file))
    model = load_model(model_file)

    predictions = load_live_predictions(log_file)
    predictions = update_completed_predictions(predictions, prices)

    record = build_prediction_record(prices, model)
    predictions = append_prediction(predictions, record)
    save_live_predictions(predictions, log_file)

    print(f"Prediction date: {record['prediction_date'].date()}")
    print(f"Close: {record['close']:.2f}")
    print(f"Probability UP: {record['probability_up']:.2%}")
    print(f"Probability DOWN: {record['probability_down']:.2%}")
    print(f"Signal: {record['signal']}")
    print(f"Evaluation date: {record['evaluation_date'].date()}")
    print(f"Updated log: {log_file}")


def update_log_only(csv_file=CSV_FILE, log_file=LIVE_PREDICTIONS_FILE):
    prices = normalize_prices(read_prices(csv_file))
    predictions = load_live_predictions(log_file)
    predictions = update_completed_predictions(predictions, prices)
    save_live_predictions(predictions, log_file)

    completed = predictions["future_close"].notna().sum()
    pending = predictions["future_close"].isna().sum()
    print(f"Completed predictions: {completed}")
    print(f"Pending predictions: {pending}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["train", "predict", "update"],
        help="train freezes a model, predict logs the latest signal, update backfills matured predictions",
    )
    parser.add_argument("--csv-file", default=CSV_FILE)
    parser.add_argument("--model-file", default=MODEL_FILE)
    parser.add_argument("--settings-file", default=SETTINGS_FILE)
    parser.add_argument("--log-file", default=LIVE_PREDICTIONS_FILE)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "train":
        train_and_freeze_model(args.csv_file, args.model_file, args.settings_file)
        return

    if args.command == "predict":
        predict_and_log(args.csv_file, args.model_file, args.log_file)
        return

    update_log_only(args.csv_file, args.log_file)


if __name__ == "__main__":
    main()