from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.stochastic_pipeline.models import run_stochastic_analysis, run_stochastic_backtest



def _make_price_series() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2022-01-03", periods=320)
    log_returns = rng.normal(loc=0.0004, scale=0.012, size=len(index))
    close = 100.0 * np.exp(np.cumsum(log_returns))
    return pd.DataFrame({"SPY": close}, index=index)



def test_run_stochastic_analysis_returns_expected_sections() -> None:
    prices = _make_price_series()

    result = run_stochastic_analysis(prices, horizon_days=20, num_paths=300, seed=7)

    assert {"history", "log_returns", "gbm", "jump", "garch", "egarch", "regime", "comparison", "last_price"}.issubset(result)
    assert len(result["gbm"]["price_cone"]) == 21
    assert len(result["jump"]["price_cone"]) == 21
    assert result["jump"]["parameters"]["jump_intensity_daily"] >= 0.0
    assert len(result["garch"].volatility_forecast) == 20
    assert len(result["egarch"].volatility_forecast) == 20
    assert len(result["regime"].price_cone) == 21
    assert result["regime"].current_regime in {
        "low_volatility_normal",
        "high_volatility_trend",
        "crash_stress",
        "explosive_momentum",
    }
    assert np.allclose(result["regime"].forecast_state_probabilities.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
    assert (result["garch"].volatility_forecast["annualized_volatility"] > 0.0).all()
    assert (result["egarch"].volatility_forecast["annualized_volatility"] > 0.0).all()



def test_stochastic_comparison_contains_all_models() -> None:
    prices = _make_price_series()

    result = run_stochastic_analysis(prices, horizon_days=15, num_paths=250, seed=11)

    models = result["comparison"]["model"].tolist()
    assert models == ["GBM", "Jump Diffusion", "GARCH(1,1)", "EGARCH(1,1)", "Regime HMM"]
    assert pd.notna(result["comparison"].loc[result["comparison"]["model"] == "GARCH(1,1)", "log_likelihood"]).all()
    assert pd.notna(result["comparison"].loc[result["comparison"]["model"] == "EGARCH(1,1)", "log_likelihood"]).all()
    assert pd.notna(result["comparison"].loc[result["comparison"]["model"] == "Regime HMM", "log_likelihood"]).all()


def test_run_stochastic_backtest_returns_detail_and_summary() -> None:
    prices = _make_price_series()

    result = run_stochastic_backtest(prices, horizon_days=15, evaluation_windows=4, step_days=10, num_paths=200, seed=5)

    assert {"detail", "summary", "horizon_days", "evaluation_windows", "step_days", "ticker"}.issubset(result)
    assert not result["detail"].empty
    assert not result["summary"].empty
    assert set(result["summary"]["model"].tolist()) == {"GBM", "Jump Diffusion", "GARCH(1,1)", "EGARCH(1,1)", "Regime HMM"}
    assert {
        "upper_cone_breach",
        "lower_cone_breach",
        "cone_breach_label",
        "realized_annualized_volatility",
        "volatility_forecast_error",
        "volatility_forecast_abs_error",
    }.issubset(result["detail"].columns)
    assert {
        "median_terminal_error",
        "upper_breach_rate",
        "lower_breach_rate",
        "mean_realized_annualized_volatility",
        "mean_volatility_forecast_error",
        "median_volatility_forecast_error",
        "mean_volatility_forecast_abs_error",
    }.issubset(result["summary"].columns)
    assert result["detail"]["terminal_abs_error"].ge(0.0).all()
    assert result["summary"]["coverage_10_90"].between(0.0, 1.0).all()
    assert result["summary"]["upper_breach_rate"].between(0.0, 1.0).all()
    assert result["summary"]["lower_breach_rate"].between(0.0, 1.0).all()
    assert result["detail"]["volatility_forecast_abs_error"].ge(0.0).all()
    assert set(result["detail"]["cone_breach_label"].unique()) <= {"inside", "upper", "lower"}
