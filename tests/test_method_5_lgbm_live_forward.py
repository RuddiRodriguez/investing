import pandas as pd

from scripts.direction.method_5_lgbm import HOLD_SIGNAL, UP_SIGNAL
from scripts.direction.method_5_lgbm_live_forward import (
    append_prediction,
    build_prediction_record,
    load_live_predictions,
    save_live_predictions,
    update_completed_predictions,
)


class StubModel:
    classes_ = [-1, 0, 1]

    def predict_proba(self, features):
        return [[0.10, 0.20, 0.70]]


def _make_prices(periods=260):
    dates = pd.bdate_range("2026-01-02", periods=periods)
    close = pd.Series(range(100, 100 + periods), dtype=float)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000_000,
        }
    )


def test_build_prediction_record_uses_latest_row_and_up_probability() -> None:
    record = build_prediction_record(_make_prices(), StubModel())

    assert record["signal"] == UP_SIGNAL
    assert record["probability_up"] == 0.70
    assert record["probability_down"] == 0.10
    assert pd.isna(record["future_close"])


def test_append_prediction_replaces_same_day_record() -> None:
    prediction_date = pd.Timestamp("2026-05-15")
    predictions = pd.DataFrame(
        [
            {
                "prediction_date": prediction_date,
                "close": 100.0,
                "probability_up": 0.55,
                "probability_down": 0.20,
                "signal": HOLD_SIGNAL,
                "evaluation_date": prediction_date,
                "future_close": pd.NA,
                "future_return": pd.NA,
                "target_hit": pd.NA,
                "correct": pd.NA,
            }
        ]
    )

    updated = append_prediction(
        predictions,
        {
            "prediction_date": prediction_date,
            "close": 101.0,
            "probability_up": 0.70,
            "probability_down": 0.10,
            "signal": UP_SIGNAL,
            "evaluation_date": prediction_date,
            "future_close": pd.NA,
            "future_return": pd.NA,
            "target_hit": pd.NA,
            "correct": pd.NA,
        },
    )

    assert len(updated) == 1
    assert updated.loc[0, "close"] == 101.0
    assert updated.loc[0, "signal"] == UP_SIGNAL


def test_update_completed_predictions_fills_matured_up_prediction() -> None:
    prices = _make_prices()
    prediction_date = prices.loc[5, "Date"]
    evaluation_date = prices.loc[15, "Date"]
    predictions = pd.DataFrame(
        [
            {
                "prediction_date": prediction_date,
                "close": prices.loc[5, "Close"],
                "probability_up": 0.70,
                "probability_down": 0.10,
                "signal": UP_SIGNAL,
                "evaluation_date": prediction_date,
                "future_close": pd.NA,
                "future_return": pd.NA,
                "target_hit": pd.NA,
                "correct": pd.NA,
            }
        ]
    )

    updated = update_completed_predictions(predictions, prices)

    assert updated.loc[0, "evaluation_date"] == evaluation_date
    assert updated.loc[0, "future_close"] == prices.loc[15, "Close"]
    assert updated.loc[0, "future_return"] == (prices.loc[15, "Close"] / prices.loc[5, "Close"]) - 1
    assert updated.loc[0, "target_hit"] == (updated.loc[0, "future_return"] > 0.02)
    assert updated.loc[0, "correct"] == updated.loc[0, "target_hit"]


def test_save_and_load_live_predictions_round_trip(tmp_path) -> None:
    log_file = tmp_path / "live_predictions.csv"
    predictions = pd.DataFrame(
        [
            {
                "prediction_date": pd.Timestamp("2026-05-15"),
                "close": 100.0,
                "probability_up": 0.70,
                "probability_down": 0.10,
                "signal": UP_SIGNAL,
                "evaluation_date": pd.Timestamp("2026-05-29"),
                "future_close": pd.NA,
                "future_return": pd.NA,
                "target_hit": pd.NA,
                "correct": pd.NA,
            }
        ]
    )

    save_live_predictions(predictions, log_file)
    loaded = load_live_predictions(log_file)

    assert loaded.loc[0, "prediction_date"] == pd.Timestamp("2026-05-15")
    assert loaded.loc[0, "signal"] == UP_SIGNAL