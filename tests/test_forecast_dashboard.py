from __future__ import annotations

import pandas as pd

from advanced_pipeline_dashboard import build_actual_price_series, build_forecast_price_chart


def _forecast_result() -> dict:
    return {
        "prediction_start_date": "2026-05-06",
        "anchor_price": 101.0,
        "history": pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-05-04", "2026-05-05", "2026-05-06"]),
                "price": [100.0, 100.5, 101.0],
            }
        ),
        "forecast": pd.DataFrame(
            {
                "forecast_date": pd.to_datetime(["2026-05-07", "2026-05-08", "2026-05-11", "2026-05-12"]),
                "predicted_price": [102.0, 103.0, 104.0, 105.0],
                "lower_price": [100.0, 101.0, 102.0, 103.0],
                "upper_price": [104.0, 105.0, 106.0, 107.0],
                "actual_price": [102.5, 103.5, 104.5, float("nan")],
                "residual": [0.5, 0.5, 0.5, float("nan")],
            }
        ),
    }


def test_actual_price_series_continues_through_known_real_future_prices() -> None:
    actual = build_actual_price_series(_forecast_result())

    assert actual["date"].tolist() == pd.to_datetime(
        ["2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08", "2026-05-11"]
    ).tolist()
    assert actual["price"].tolist()[-1] == 104.5


def test_forecast_chart_labels_real_prices_after_start_without_future_wording() -> None:
    figure = build_forecast_price_chart(_forecast_result())
    trace_names = [trace.name for trace in figure.data]

    assert "Actual price" in trace_names
    assert "Real price after forecast start" in trace_names
    assert "Actual future" not in trace_names
