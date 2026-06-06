from __future__ import annotations

import pandas as pd

from market_forecasting_engine.curve_shape import build_curve_shape_analysis


def test_curve_shape_labels_breakdown_after_range() -> None:
    close = [100, 100.2, 100.1, 99.9, 100.0, 99.8] * 8 + [99.5, 99.0, 98.5, 98.0, 97.5, 97.0, 96.5, 96.0, 95.5]
    prices = pd.DataFrame({"close": close})

    analysis = build_curve_shape_analysis(prices, current_price=95.5, lookback_rows=80, impulse_bars=8)

    assert analysis["status"] == "ok"
    assert analysis["label"] == "breakdown_after_range"
    assert analysis["direction"] == "bearish"
    assert analysis["recommended_option_bias"] == "put"


def test_curve_shape_labels_range_chop() -> None:
    close = [100.0, 100.1, 99.95, 100.05, 99.9, 100.0, 100.08, 99.98] * 12
    prices = pd.DataFrame({"close": close})

    analysis = build_curve_shape_analysis(prices, current_price=99.98, lookback_rows=80, impulse_bars=8)

    assert analysis["status"] == "ok"
    assert analysis["label"] == "range_chop"
    assert analysis["recommended_option_bias"] == "hold"


def test_curve_shape_labels_late_downtrend_exhaustion() -> None:
    close = [104, 103.8, 104.1, 103.9, 104.0] * 8 + [103, 102, 101, 100, 99, 98, 97, 97.3, 97.6]
    prices = pd.DataFrame({"close": close})

    analysis = build_curve_shape_analysis(prices, current_price=97.6, lookback_rows=80, impulse_bars=8)

    assert analysis["status"] == "ok"
    assert analysis["label"] == "late_downtrend_exhaustion"
    assert analysis["recommended_option_bias"] == "hold"
