from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.advanced_pipeline.forecast import (
    FORECAST_FEATURE_COLUMNS,
    PeerTrainingBundle,
    add_forecast_features,
    add_forecast_targets,
    aggregate_historical_news_features,
    augment_classifier_training_set,
    generate_forecast,
    load_forecast_artifacts,
    model_file_for_ticker,
    next_trading_dates,
    resolve_prediction_start_date,
    save_forecast_artifacts,
    train_forecast_artifacts,
)


def _market_prices(periods: int = 420) -> pd.DataFrame:
    index = pd.bdate_range("2022-01-03", periods=periods)
    return pd.DataFrame(
        {
            "AAA": [100 + 0.2 * step + (step % 7) * 0.3 for step in range(periods)],
            "SPY": [400 + 0.1 * step for step in range(periods)],
            "QQQ": [300 + 0.12 * step for step in range(periods)],
            "^VIX": [20 + ((step % 20) - 10) * 0.15 for step in range(periods)],
        },
        index=index,
    )


def test_add_forecast_features_builds_expected_columns() -> None:
    features = add_forecast_features(_market_prices())
    assert set(FORECAST_FEATURE_COLUMNS).issubset(features.columns)


def test_add_forecast_features_uses_only_prior_information() -> None:
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 110.0, 121.0, 133.1],
            "SPY": [200.0, 202.0, 204.0, 206.0],
            "QQQ": [300.0, 303.0, 306.0, 309.0],
            "^VIX": [20.0, 21.0, 22.0, 23.0],
            "VOLUME_AAA": [10.0, 20.0, 30.0, 40.0],
        },
        index=pd.bdate_range("2024-01-01", periods=4),
    )

    features = add_forecast_features(prices)

    assert pd.isna(features.iloc[1]["log_return_1d"])
    expected_prior_return = float(pd.Series([100.0, 110.0]).pipe(lambda series: np.log(series.iloc[1] / series.iloc[0])))
    assert features.iloc[2]["log_return_1d"] == expected_prior_return


def test_aggregate_historical_news_features_builds_rolling_columns() -> None:
    index = pd.bdate_range("2024-01-01", periods=5)
    news = pd.DataFrame(
        {
            "date": [index[0], index[0], index[1]],
            "title": ["great growth upgrade", "profit record", "lawsuit warning risk"],
            "sentiment_score": [0.8, 0.6, -0.7],
        }
    )

    features = aggregate_historical_news_features(news, index)

    assert {"news_sentiment_7d", "news_volume_3d", "news_positive_share_7d"}.issubset(features.columns)
    assert features.loc[index[0], "news_volume_1d"] == 2
    assert features.loc[index[1], "news_negative_share_7d"] > 0


def test_add_forecast_targets_adds_horizon_targets() -> None:
    prices = _market_prices()
    features = add_forecast_features(prices)
    labeled = add_forecast_targets(features, prices, max_horizon_days=3)
    assert "target_log_return_1d" in labeled.columns
    assert "target_log_return_3d" in labeled.columns
    assert "target_excess_log_return_3d" in labeled.columns


def test_train_save_load_and_generate_forecast(monkeypatch) -> None:
    prices = _market_prices()

    def fake_news(*args, **kwargs):
        return (
            pd.DataFrame(
                {
                    "date": [prices.index[200], prices.index[201]],
                    "title": ["profit growth", "warning downgrade"],
                    "sentiment_score": [0.5, -0.5],
                }
            ),
            "keyword",
        )

    monkeypatch.setattr("scripts.advanced_pipeline.forecast.fetch_historical_news_features", fake_news)
    artifacts = train_forecast_artifacts(
        ticker="AAA",
        prices=prices,
        prediction_start_date="2023-06-01",
        max_horizon_days=3,
    )
    save_forecast_artifacts(artifacts)
    loaded = load_forecast_artifacts("AAA")

    forecast = generate_forecast(
        ticker="AAA",
        prices=prices,
        prediction_start_date="2023-06-01",
        forecast_horizon_days=3,
        artifacts=loaded,
    )

    assert len(forecast["forecast"]) == 3
    assert {"predicted_price", "lower_price", "upper_price", "probability_threshold_hit", "predicted_excess_log_return", "trade_probability", "trade_decision"}.issubset(
        forecast["forecast"].columns
    )
    assert model_file_for_ticker("AAA").exists()
    assert forecast["metadata"]["news_feature_mode"] == "keyword"
    assert forecast["metadata"]["benchmark_symbol"] in {"QQQ", "SPY", "AAA"}
    assert forecast["metadata"]["signal_benchmark_symbol"] in {"SOXX", "QQQ", "SPY", "AAA"}
    assert forecast["metadata"]["decision_horizon"] in {1, 2, 3}
    assert "deployment_gate" in forecast["metadata"]
    horizon_metrics = forecast["metadata"]["validation_metrics"][1]
    assert horizon_metrics["train_rows"] > 0
    assert horizon_metrics["validation_rows"] > 0
    assert "model_validation_mae_log_return" in horizon_metrics
    assert "dummy_validation_mae_log_return" in horizon_metrics
    assert "relative_momentum_validation_mae_log_return" in horizon_metrics
    assert "validation_probability_calibration_gap" in horizon_metrics
    assert loaded.interval_adjustments[1] >= 0.0
    assert horizon_metrics["validation_end_date"] < forecast["prediction_start_date"]


def test_train_forecast_artifacts_defaults_to_short_decision_horizons(monkeypatch) -> None:
    prices = _market_prices()

    def fake_news(*args, **kwargs):
        return (pd.DataFrame(columns=["date", "title", "sentiment_score"]), "none")

    monkeypatch.setattr("scripts.advanced_pipeline.forecast.fetch_historical_news_features", fake_news)
    artifacts = train_forecast_artifacts(
        ticker="AAA",
        prices=prices,
        prediction_start_date="2023-06-01",
        max_horizon_days=5,
        use_openai_news=False,
    )

    assert artifacts.max_horizon_days == 3
    assert sorted(artifacts.validation_metrics) == [1, 2, 3]


def test_augment_classifier_training_set_prefers_signal_relative_peer_targets() -> None:
    base_index = pd.bdate_range("2024-01-01", periods=45)
    base_features = pd.DataFrame(0.0, index=base_index, columns=FORECAST_FEATURE_COLUMNS)
    base_target = pd.Series(0, index=base_index, dtype=int)

    peer_training = pd.DataFrame(0.0, index=base_index, columns=FORECAST_FEATURE_COLUMNS)
    peer_training["target_log_return_1d"] = np.log1p(0.03)
    peer_training["target_signal_excess_log_return_1d"] = np.log1p(-0.03)

    features, target, peer_rows = augment_classifier_training_set(
        base_features,
        base_target,
        peer_training_bundles=(PeerTrainingBundle(ticker="BBB", training=peer_training),),
        horizon=1,
        threshold_return=0.0,
        peer_target_column="target_signal_excess_log_return_1d",
        as_of_date=base_index[-1],
    )

    assert peer_rows == len(base_index)
    assert len(features) == len(base_index) * 2
    assert int(target.iloc[-1]) == 0


def test_next_trading_dates_extends_when_actual_future_is_missing() -> None:
    index = pd.bdate_range("2024-01-01", periods=3)
    future = next_trading_dates(index, pd.Timestamp("2024-01-03"), 2)
    assert len(future) == 2
    assert future[0] > pd.Timestamp("2024-01-03")


def test_resolve_prediction_start_date_uses_latest_available_trading_date() -> None:
    index = pd.bdate_range("2026-05-04", "2026-05-14")

    assert resolve_prediction_start_date(index, "2026-05-10") == pd.Timestamp("2026-05-08")
    assert resolve_prediction_start_date(index, "2026-05-14") == pd.Timestamp("2026-05-14")
