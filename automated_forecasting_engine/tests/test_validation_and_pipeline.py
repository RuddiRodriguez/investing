from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_forecasting_engine import ForecastConfig, ForecastingEngine
from market_forecasting_engine.backtest import backtest_validation_signals
from market_forecasting_engine.basing_points import analyze_basing_points
from market_forecasting_engine.chapter_10_patterns import analyze_chapter_10_patterns
from market_forecasting_engine.chapter_11_patterns import analyze_chapter_11_patterns
from market_forecasting_engine.chapter_12_gaps import analyze_chapter_12_gaps
from market_forecasting_engine.chapter_13_support_resistance import analyze_chapter_13_support_resistance
from market_forecasting_engine.chapter_14_trendlines import analyze_chapter_14_trendlines
from market_forecasting_engine.chapter_15_major_trendlines import analyze_chapter_15_major_trendlines
from market_forecasting_engine.chapter_16_market_context import analyze_chapter_16_market_context
from market_forecasting_engine.chapter_17_governance_context import analyze_chapter_17_governance_context
from market_forecasting_engine.chapter_18_tactics import analyze_chapter_18_tactical_problem
from market_forecasting_engine.chapter_19_validation import apply_chapter_19_validation
from market_forecasting_engine.chapter_20_selection import apply_chapter_20_ticker_suitability
from market_forecasting_engine.chapter_21_chart_selection import apply_chapter_21_chart_selection
from market_forecasting_engine.chapter_23_30_trade_risk import apply_chapter_23_30_trade_risk_plan
from market_forecasting_engine.chapter_31_42_portfolio_risk import apply_chapter_31_42_portfolio_capital_risk
from market_forecasting_engine.chapter_39_43_discipline import apply_chapter_39_43_discipline_governance
from market_forecasting_engine.chapter_9_patterns import analyze_chapter_9_patterns
from market_forecasting_engine.dow_theory import analyze_dow_theory
from market_forecasting_engine.factor_evaluation import evaluate_factors
from market_forecasting_engine.features import add_forward_return_targets, build_feature_frame
from market_forecasting_engine.models import (
    HistoricalMeanReturn,
    KalmanFilterReturnCandidate,
    LSTMReturnCandidate,
    RecentMeanReturn,
    default_candidates,
)
from market_forecasting_engine.plots import write_plot_artifacts
from market_forecasting_engine.reversal_patterns import analyze_reversal_patterns
from market_forecasting_engine.triangle_patterns import analyze_triangle_patterns
from market_forecasting_engine import portfolio
from market_forecasting_engine.pairs_trading import analyze_pair, rank_cointegrated_pairs
from market_forecasting_engine.validation import (
    apply_ml4t_selection_adjustments,
    make_walk_forward_splits,
    select_candidate,
    validate_candidates,
)


def _market_prices(rows: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2024-01-02", periods=rows)
    trend = np.linspace(0, 0.30, rows)
    cycle = np.sin(np.arange(rows) / 11) * 0.01
    noise = rng.normal(0.0, 0.009, rows)
    close = 100 * np.exp(np.cumsum(0.0005 + trend / rows + cycle / 10 + noise))
    volume = rng.integers(1_000_000, 2_000_000, rows)
    return pd.DataFrame({"close": close, "volume": volume}, index=dates)


def test_walk_forward_splits_are_expanding_and_non_overlapping() -> None:
    splits = make_walk_forward_splits(
        n_rows=220,
        min_training_rows=100,
        validation_window=30,
        step_size=25,
        max_splits=3,
    )

    assert len(splits) == 3
    for train_idx, validation_idx in splits:
        assert train_idx.max() < validation_idx.min()
        assert train_idx.min() == 0


def test_walk_forward_splits_apply_purge_and_embargo() -> None:
    splits = make_walk_forward_splits(
        n_rows=220,
        min_training_rows=100,
        validation_window=30,
        step_size=30,
        max_splits=2,
        purge_window=5,
        embargo_window=2,
    )

    for train_idx, validation_idx in splits:
        assert train_idx.max() <= validation_idx.min() - 8


def test_select_candidate_uses_lower_error_metric() -> None:
    dates = pd.bdate_range("2024-01-02", periods=150)
    features = pd.DataFrame({"x": np.arange(150)}, index=dates)
    target = pd.Series(np.r_[np.zeros(100), np.ones(50) * 0.01], index=dates)
    candidates = [HistoricalMeanReturn(), RecentMeanReturn(window=20)]

    results = validate_candidates(
        candidates=candidates,
        features=features,
        target=target,
        horizon_days=1,
        min_training_rows=80,
        validation_window=20,
        step_size=20,
        max_splits=2,
    )
    _, summary, _ = select_candidate(results, selection_metric="mae")

    assert summary.metrics["mae"] >= 0
    assert summary.model_name in {"historical_mean_return", "recent_mean_return"}


def test_chapter_9_adjusted_metric_can_change_candidate_selection() -> None:
    dates = pd.bdate_range("2024-01-02", periods=150)
    features = pd.DataFrame({"x": np.arange(150)}, index=dates)
    target = pd.Series(np.r_[np.zeros(100), np.ones(50) * 0.01], index=dates)
    results = validate_candidates(
        candidates=[HistoricalMeanReturn(), RecentMeanReturn(window=20)],
        features=features,
        target=target,
        horizon_days=1,
        min_training_rows=80,
        validation_window=20,
        step_size=20,
        max_splits=2,
    )
    first = results[0][1]
    second = results[1][1]
    first.metrics["mae"] = 0.01
    second.metrics["mae"] = 0.02
    first.metrics["chapter_9_adjusted_mae"] = 0.05
    second.metrics["chapter_9_adjusted_mae"] = 0.02

    _, summary, _ = select_candidate(results, selection_metric="mae")

    assert summary.model_name == second.model_name


def test_chapter_9_selection_adjustments_are_written_to_candidate_metrics() -> None:
    dates = pd.bdate_range("2024-01-02", periods=150)
    features = pd.DataFrame({"x": np.arange(150)}, index=dates)
    target = pd.Series(np.sin(np.arange(150) / 7) * 0.01, index=dates)
    results = validate_candidates(
        candidates=[HistoricalMeanReturn(), RecentMeanReturn(window=20)],
        features=features,
        target=target,
        horizon_days=1,
        min_training_rows=80,
        validation_window=20,
        step_size=20,
        max_splits=2,
    )

    audit = apply_ml4t_selection_adjustments(results, target=target, selection_metric="mae")

    assert audit["adjusted_candidate_count"] == len(results)
    for _, summary, _ in results:
        assert "chapter_9_selection_penalty" in summary.metrics
        assert "chapter_9_adjusted_mae" in summary.metrics
        assert summary.metrics["chapter_9_selection_adjustment_applied"] == 1.0


def test_default_candidates_register_advanced_model_families_when_available() -> None:
    names = {
        candidate.name
        for candidate in default_candidates(
            include_lightgbm=False,
            include_statistical_models=True,
            include_lstm=importlib.util.find_spec("torch") is not None,
        )
    }

    assert "kalman_filter" in names
    assert "ridge_alpha_1_0" in names
    assert "lasso_alpha_0_0001" in names
    assert "elastic_net_alpha_0_0005_l1_0_25" in names
    if importlib.util.find_spec("statsmodels") is not None:
        assert any(name.startswith("arima_") for name in names)
        assert any(name.startswith("sarima_") for name in names)
        assert any(name.startswith("var_") for name in names)
    if importlib.util.find_spec("arch") is not None:
        assert any(name.startswith("garch_") for name in names)
    if importlib.util.find_spec("torch") is not None:
        assert "lstm" in names


def test_expanded_search_registers_multiple_tuning_variants() -> None:
    names = {
        candidate.name
        for candidate in default_candidates(
            include_lightgbm=False,
            include_statistical_models=True,
            include_lstm=False,
            search_level="expanded",
        )
    }

    assert {
        "ridge_alpha_10_0",
        "ridge_alpha_100_0",
        "lasso_alpha_0_00001",
        "lasso_alpha_0_001",
        "elastic_net_alpha_0_0001_l1_0_10",
        "random_forest_smoother",
        "gradient_boosting_deeper",
    }.issubset(names)
    if importlib.util.find_spec("statsmodels") is not None:
        assert sum(name.startswith("arima_") for name in names) >= 3
    if importlib.util.find_spec("arch") is not None:
        assert sum(name.startswith("garch_") for name in names) >= 3


@pytest.mark.skipif(importlib.util.find_spec("optuna") is None, reason="optuna is not installed")
def test_optuna_tuning_candidate_fits_with_inner_walk_forward() -> None:
    rng = np.random.default_rng(23)
    dates = pd.bdate_range("2024-01-02", periods=90)
    features = pd.DataFrame(
        {
            "momentum": np.linspace(-1.0, 1.0, 90),
            "noise": rng.normal(size=90),
            "volume_z": rng.normal(size=90),
        },
        index=dates,
    )
    target = pd.Series(features["momentum"] * 0.004 + rng.normal(0.0, 0.001, 90), index=dates)
    candidate = next(
        candidate
        for candidate in default_candidates(
            include_lightgbm=False,
            include_statistical_models=False,
            include_lstm=False,
            tuning_mode="optuna",
            optuna_trials=2,
            optuna_inner_splits=2,
            optuna_families=("elastic_net",),
        )
        if candidate.name == "optuna_elastic_net"
    )

    fitted = candidate.clone().fit(features.iloc[:75], target.iloc[:75])
    predicted = fitted.predict(features.iloc[75:80])
    params = fitted.parameters()

    assert predicted.shape == (5,)
    assert np.isfinite(predicted).all()
    assert params["tuning"] == "optuna"
    assert params["trials_completed"] >= 1


@pytest.mark.skipif(importlib.util.find_spec("lightgbm") is None, reason="lightgbm is not installed")
def test_lightgbm_candidate_preserves_feature_names_on_predict() -> None:
    rng = np.random.default_rng(17)
    dates = pd.bdate_range("2024-01-02", periods=90)
    features = pd.DataFrame(rng.normal(size=(90, 4)), columns=["alpha", "beta", "gamma", "delta"], index=dates)
    features.iloc[:5, 2] = np.nan
    target = pd.Series(rng.normal(0.0, 0.01, 90), index=dates)
    candidate = next(
        candidate
        for candidate in default_candidates(include_lightgbm=True, include_statistical_models=False, include_lstm=False)
        if candidate.name.startswith("lightgbm")
    )

    model = candidate.clone().fit(features.iloc[:70], target.iloc[:70])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predicted = model.predict(features.iloc[70:75])

    assert predicted.shape == (5,)
    assert not any("does not have valid feature names" in str(item.message) for item in caught)


def test_kalman_filter_candidate_fit_predicts() -> None:
    dates = pd.bdate_range("2024-01-02", periods=80)
    features = pd.DataFrame({"x": np.arange(80)}, index=dates)
    target = pd.Series(np.sin(np.arange(80) / 10) * 0.01, index=dates)

    model = KalmanFilterReturnCandidate().fit(features, target)
    predicted = model.predict(features.tail(5))

    assert predicted.shape == (5,)
    assert np.isfinite(predicted).all()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_lstm_candidate_fit_predicts_small_series() -> None:
    dates = pd.bdate_range("2024-01-02", periods=70)
    features = pd.DataFrame({"x": np.arange(70)}, index=dates)
    target = pd.Series(np.sin(np.arange(70) / 7) * 0.01, index=dates)

    model = LSTMReturnCandidate(sequence_length=8, hidden_size=4, max_epochs=2, min_rows=50).fit(features, target)
    predicted = model.predict(features.tail(4))

    assert predicted.shape == (4,)
    assert np.isfinite(predicted).all()


def test_forecasting_engine_returns_governed_multi_horizon_report() -> None:
    config = ForecastConfig(
        ticker="TEST",
        horizons=(1, 5, 30),
        min_training_rows=120,
        validation_window=30,
        step_size=30,
        max_splits=3,
        include_lightgbm=False,
        include_statistical_models=False,
    )

    report = ForecastingEngine(config).run(_market_prices())

    assert report["ticker"] == "TEST"
    assert report["suggested_action"] in {"Buy", "Hold", "Sell"}
    assert report["risk_level"] in {"Low", "Medium", "High"}
    assert report["technical_view"]["trend_state"]["state"] in {"Bullish", "Neutral", "Bearish"}
    assert report["technical_view"]["dow_theory"]["primary_trend"]["state"] in {"Bullish", "Neutral", "Bearish"}
    assert report["technical_view"]["magee_basing_points"]["preferred"]["trend_state"] in {"Long", "Short", "Neutral"}
    reversal_view = report["technical_view"]["reversal_patterns"]
    assert reversal_view["preferred"]["pattern"] in {"HeadAndShouldersTop", "HeadAndShouldersBottom"}
    assert set(reversal_view["timeframes"]) == {"daily", "weekly", "monthly"}
    assert "dormant_bottoms" in reversal_view["optional_methods"]
    triangle_view = report["technical_view"]["triangle_patterns"]
    assert "preferred" in triangle_view
    assert set(triangle_view["timeframes"]) == {"daily", "weekly", "monthly"}
    chapter_9_view = report["technical_view"]["chapter_9_patterns"]
    assert "preferred" in chapter_9_view
    assert set(chapter_9_view["timeframes"]) == {"daily", "weekly", "monthly"}
    assert "rectangle_patterns" in report["technical_view"]
    assert "multi_top_bottom_patterns" in report["technical_view"]
    chapter_10_view = report["technical_view"]["chapter_10_patterns"]
    assert "preferred" in chapter_10_view
    assert set(chapter_10_view["timeframes"]) == {"daily", "weekly", "monthly"}
    assert "chapter_10_structural_patterns" in report["technical_view"]
    assert "chapter_10_short_term_events" in report["technical_view"]
    chapter_11_view = report["technical_view"]["chapter_11_patterns"]
    assert "preferred" in chapter_11_view
    assert set(chapter_11_view["timeframes"]) == {"daily", "weekly", "monthly"}
    assert "chapter_11_continuation_patterns" in report["technical_view"]
    assert "chapter_11_head_and_shoulders_continuation" in report["technical_view"]
    chapter_12_view = report["technical_view"]["chapter_12_gaps"]
    assert "preferred" in chapter_12_view
    assert set(chapter_12_view["timeframes"]) == {"daily", "weekly", "monthly"}
    assert "chapter_12_classified_gaps" in report["technical_view"]
    assert "chapter_12_island_reversals" in report["technical_view"]
    chapter_13_view = report["technical_view"]["chapter_13_support_resistance"]
    assert "preferred" in chapter_13_view
    assert set(chapter_13_view["timeframes"]) == {"daily", "weekly", "monthly"}
    assert "chapter_13_support_zones" in report["technical_view"]
    assert "chapter_13_resistance_zones" in report["technical_view"]
    chapter_14_view = report["technical_view"]["chapter_14_trendlines"]
    assert "preferred" in chapter_14_view
    assert set(chapter_14_view["timeframes"]) == {"daily", "weekly", "monthly"}
    assert "chapter_14_channels" in report["technical_view"]
    assert "chapter_14_fan_lines" in report["technical_view"]
    chapter_15_view = report["technical_view"]["chapter_15_major_trendlines"]
    assert "stock_major_trend" in chapter_15_view
    assert "broad_market_confirmation" in chapter_15_view
    assert "chapter_15_scale_comparison" in report["technical_view"]
    assert "chapter_15_broad_market_confirmation" in report["technical_view"]
    assert "dow_theory" in report["diagnostics"]
    assert "magee_basing_points" in report["diagnostics"]
    assert "reversal_patterns" in report["diagnostics"]
    assert "triangle_patterns" in report["diagnostics"]
    assert "chapter_9_patterns" in report["diagnostics"]
    assert "chapter_10_patterns" in report["diagnostics"]
    assert "chapter_11_patterns" in report["diagnostics"]
    assert "chapter_12_gaps" in report["diagnostics"]
    assert "chapter_13_support_resistance" in report["diagnostics"]
    assert "chapter_14_trendlines" in report["diagnostics"]
    assert "chapter_15_major_trendlines" in report["diagnostics"]
    chapter_16_view = report["technical_view"]["chapter_16_market_context"]
    assert chapter_16_view["decision_policy"]["influences_final_action"] is False
    assert "donchian_context" in chapter_16_view
    assert "chapter_16_donchian_context" in report["technical_view"]
    assert "chapter_16_futures_risk_context" in report["technical_view"]
    assert "chapter_16_market_context" in report["diagnostics"]
    chapter_17_view = report["technical_view"]["chapter_17_governance_context"]
    assert chapter_17_view["decision_policy"]["influences_final_action"] is False
    assert chapter_17_view["decision_policy"]["intended_consumer"] == "human_or_llm_decision_layer"
    assert "llm_decision_packet" in chapter_17_view
    assert "chapter_17_llm_decision_packet" in report["technical_view"]
    assert "chapter_17_decision_fragility" in report["technical_view"]
    assert "chapter_17_governance_context" in report["diagnostics"]
    chapter_18_view = report["decision_view"]["chapter_18_tactical_problem"]
    assert chapter_18_view["decision_policy"]["influences_final_action"] is True
    assert chapter_18_view["llm_review"]["status"] == "disabled"
    assert chapter_18_view["llm_safety_gate"]["final_action"] == report["suggested_action"]
    assert "chapter_18_tactical_problem" in report["technical_view"]
    assert "chapter_18_tactical_plan" in report["technical_view"]
    assert "chapter_18_llm_review" in report["technical_view"]
    assert "chapter_18_tactical_problem" in report["diagnostics"]
    chapter_19_view = report["operations_view"]["chapter_19_validation"]
    assert chapter_19_view["decision_policy"]["mode"] == "conditional_operational_validation_gate"
    assert chapter_19_view["action_gate"]["validated_action"] == report["suggested_action"]
    assert chapter_19_view["status"] in {"pass", "warn", "fail"}
    assert "chapter_19_validation" in report["decision_view"]
    assert "chapter_19_validation" in report["technical_view"]
    assert "chapter_19_validation" in report["diagnostics"]
    chapter_20_view = report["selection_view"]["chapter_20_ticker_suitability"]
    assert chapter_20_view["decision_policy"]["influences_final_action"] is False
    assert chapter_20_view["profile_fit"]["primary_profile"] in {
        "short_term_trader",
        "intermediate_trader",
        "long_term_investor",
        "speculative_satellite",
        "index_or_diversifier",
    }
    assert "chapter_20_ticker_suitability" in report["technical_view"]
    assert "chapter_20_ticker_suitability" in report["diagnostics"]
    chapter_21_view = report["selection_view"]["chapter_21_chart_selection"]
    assert chapter_21_view["decision_policy"]["influences_final_action"] is False
    assert chapter_21_view["chart_selection"]["chart_book_bucket"] in {
        "trade_candidates",
        "active_review",
        "watchlist",
        "monitor_only",
        "excluded",
    }
    assert "chapter_21_chart_selection" in report["technical_view"]
    assert "chapter_21_chart_selection" in report["diagnostics"]
    trade_risk_view = report["trade_risk_view"]["chapter_23_30_trade_risk_plan"]
    assert trade_risk_view["decision_policy"]["influences_final_action"] is False
    assert trade_risk_view["commitment"]["commitment_type"] in {
        "candidate_long_commitment",
        "candidate_short_commitment",
        "risk_reduction_or_exit",
        "active_review_no_new_commitment",
        "watchlist_no_new_commitment",
        "monitor_only",
        "no_new_commitment",
    }
    assert "chapter_23_30_trade_risk_plan" in report["technical_view"]
    assert "chapter_23_30_trade_risk_plan" in report["diagnostics"]
    portfolio_capital_view = report["portfolio_view"]["chapter_31_42_portfolio_capital_risk"]
    assert portfolio_capital_view["decision_policy"]["influences_final_action"] is False
    assert portfolio_capital_view["portfolio_capital_gate"]["allocation_status"] in {
        "not_applicable",
        "blocked_pending_inputs",
        "ready_with_rules",
    }
    assert "chapter_31_42_portfolio_capital_risk" in report["technical_view"]
    assert "chapter_31_42_portfolio_capital_risk" in report["diagnostics"]
    discipline_view = report["discipline_view"]["chapter_39_43_discipline_governance"]
    assert discipline_view["decision_policy"]["influences_final_action"] is False
    assert discipline_view["status"] in {"consistent", "review_needed"}
    assert discipline_view["discipline_gate"]["plan_adherence"] in {"consistent", "needs_manual_review"}
    assert "chapter_39_43_discipline_governance" in report["technical_view"]
    assert "chapter_39_43_discipline_governance" in report["diagnostics"]
    assert "decision_view" in report
    assert "portfolio_view" in report
    assert "operations_view" in report
    assert "selection_view" in report
    assert "trade_risk_view" in report
    assert "discipline_view" in report
    chapter_1_ml4t = report["technical_view"]["chapter_1_ml4t_workflow"]
    assert chapter_1_ml4t["workflow_complete"] is True
    assert set(chapter_1_ml4t["workflow_stages"]) == {
        "data",
        "features",
        "targets",
        "model_selection",
        "signals",
        "backtest",
        "risk_action",
    }
    assert chapter_1_ml4t["strategy_breadth"]["horizon_count"] == 3
    assert "forecast_layer" in chapter_1_ml4t["forecast_to_action_separation"]
    chapter_2_market_data = report["technical_view"]["chapter_2_market_data"]
    assert chapter_2_market_data["source"]["row_count"] > 0
    assert chapter_2_market_data["field_coverage"]["has_volume"] is True
    assert "liquidity_tradability" in chapter_2_market_data
    assert chapter_2_market_data["tradability_gate"]["status"] in {"pass", "warn", "fail"}
    chapter_3_alternative_data = report["technical_view"]["chapter_3_alternative_data"]
    assert chapter_3_alternative_data["status"] == "not_supplied"
    assert "hybrid_collection_design" in chapter_3_alternative_data
    assert chapter_3_alternative_data["decision_policy"]["influences_model_fitting"] is True
    chapter_4_alpha_research = report["technical_view"]["chapter_4_alpha_research"]
    assert chapter_4_alpha_research["decision_policy"]["influences_final_action"] is False
    assert chapter_4_alpha_research["decision_policy"]["influences_model_fitting"] is True
    assert set(chapter_4_alpha_research["horizons"]) == {"1", "5", "30"}
    assert "recommendations" in chapter_4_alpha_research
    assert chapter_4_alpha_research["wavelet_denoising_assessment"]["tested"] is True
    chapter_5_portfolio_evaluation = report["technical_view"]["chapter_5_portfolio_evaluation"]
    assert chapter_5_portfolio_evaluation["decision_policy"]["influences_final_action"] is False
    assert set(chapter_5_portfolio_evaluation["horizons"]) == {"1", "5", "30"}
    assert chapter_5_portfolio_evaluation["allocation_policy"]["status"] in {
        "candidate_for_implementation",
        "tested_not_implemented",
    }
    assert "portfolio_construction_candidates" in chapter_5_portfolio_evaluation
    chapter_6_ml_process = report["technical_view"]["chapter_6_ml_process"]
    assert chapter_6_ml_process["decision_policy"]["influences_final_action"] is False
    assert chapter_6_ml_process["decision_policy"]["influences_model_selection"] is True
    assert chapter_6_ml_process["promotion_policy"]["status"] in {"candidate_for_gate_promotion", "report_only"}
    assert set(chapter_6_ml_process["horizons"]) == {"1", "5", "30"}
    assert "directional_classification" in chapter_6_ml_process["horizons"]["1"]
    assert "feature_information" in chapter_6_ml_process["horizons"]["1"]
    chapter_7_linear_models = report["technical_view"]["chapter_7_linear_models"]
    assert chapter_7_linear_models["decision_policy"]["influences_final_action"] is False
    assert chapter_7_linear_models["model_registry_policy"]["elastic_net"] == "kept_distinct_mixed_l1_l2_shrinkage"
    assert set(chapter_7_linear_models["horizons"]) == {"1", "5", "30"}
    assert "linear_candidate_comparison" in chapter_7_linear_models["horizons"]["1"]
    assert "residual_diagnostics" in chapter_7_linear_models["horizons"]["1"]
    chapter_8_backtesting = report["technical_view"]["chapter_8_backtesting"]
    assert chapter_8_backtesting["decision_policy"]["influences_final_action"] is False
    assert chapter_8_backtesting["decision_policy"]["influences_model_selection"] is True
    assert chapter_8_backtesting["promotion_policy"]["status"] in {
        "candidate_for_event_driven_research",
        "diagnostic_only",
    }
    assert set(chapter_8_backtesting["horizons"]) == {"1", "5", "30"}
    assert "realism_audit" in chapter_8_backtesting["horizons"]["1"]
    assert "cost_slippage_sensitivity" in chapter_8_backtesting["horizons"]["1"]
    chapter_9_time_series = report["technical_view"]["chapter_9_time_series"]
    assert chapter_9_time_series["decision_policy"]["influences_final_action"] is False
    assert chapter_9_time_series["decision_policy"]["influences_model_selection"] is True
    assert chapter_9_time_series["model_policy"]["new_arima_garch_models_added"] is False
    assert chapter_9_time_series["model_policy"]["selection_adjustment"] == "enabled"
    assert chapter_9_time_series["model_policy"]["pairs_trading_module"] == "available_as_market_forecasting_engine.pairs_trading"
    assert "series_diagnostics" in chapter_9_time_series
    assert set(chapter_9_time_series["horizons"]) == {"1", "5", "30"}
    assert "residual_white_noise_audit" in chapter_9_time_series["horizons"]["1"]
    assert "chapter_9_selection_penalty" in report["candidate_results"]["1"][0]["metrics"]
    assert "chapter_9_adjusted_mae" in report["candidate_results"]["1"][0]["metrics"]
    assert "chapter_6_process_selection_penalty" in report["candidate_results"]["1"][0]["metrics"]
    assert "chapter_8_backtest_selection_penalty" in report["candidate_results"]["1"][0]["metrics"]
    assert "chapter_10_bayesian_selection_penalty" in report["candidate_results"]["1"][0]["metrics"]
    chapter_10_bayesian_ml = report["technical_view"]["chapter_10_bayesian_ml"]
    assert chapter_10_bayesian_ml["decision_policy"]["influences_model_selection"] is True
    assert chapter_10_bayesian_ml["decision_policy"]["influences_forecast_confidence"] is True
    assert chapter_10_bayesian_ml["heavy_bayesian_policy"]["enabled"] is False
    assert chapter_10_bayesian_ml["horizons"]["1"]["heavy_bayesian_path"]["status"] == "disabled"
    assert "ml4t_feature_selection_policy" in report["governance"]["model_cards"]["1"]
    assert "ml4t_selection_adjustment" in report["governance"]["model_cards"]["1"]
    assert "chapter_3_alternative_data" in report["diagnostics"]
    assert "chapter_4_alpha_research" in report["diagnostics"]
    assert "chapter_5_portfolio_evaluation" in report["diagnostics"]
    assert "chapter_6_ml_process" in report["diagnostics"]
    assert "chapter_7_linear_models" in report["diagnostics"]
    assert "chapter_8_backtesting" in report["diagnostics"]
    assert "chapter_9_time_series" in report["diagnostics"]
    assert "chapter_1_ml4t_workflow" in report["diagnostics"]
    assert "chapter_2_market_data" in report["diagnostics"]
    assert (
        report["governance"]["technical_method_cards"]["chapter_1_ml4t_workflow"]["name"]
        == "jansen_ml4t_chapter_1_workflow_audit"
    )
    assert (
        report["governance"]["technical_method_cards"]["chapter_2_market_data"]["name"]
        == "jansen_ml4t_chapter_2_market_data_diagnostics"
    )
    assert (
        report["governance"]["technical_method_cards"]["chapter_3_alternative_data"]["name"]
        == "jansen_ml4t_chapter_3_alternative_data_registry"
    )
    assert (
        report["governance"]["technical_method_cards"]["chapter_4_alpha_research"]["name"]
        == "jansen_ml4t_chapter_4_alpha_research"
    )
    assert (
        report["governance"]["technical_method_cards"]["chapter_5_portfolio_evaluation"]["name"]
        == "jansen_ml4t_chapter_5_portfolio_evaluation"
    )
    assert (
        report["governance"]["technical_method_cards"]["chapter_6_ml_process"]["name"]
        == "jansen_ml4t_chapter_6_ml_process"
    )
    assert (
        report["governance"]["technical_method_cards"]["chapter_7_linear_models"]["name"]
        == "jansen_ml4t_chapter_7_linear_models"
    )
    assert (
        report["governance"]["technical_method_cards"]["chapter_8_backtesting"]["name"]
        == "jansen_ml4t_chapter_8_backtesting"
    )
    assert (
        report["governance"]["technical_method_cards"]["chapter_9_time_series"]["name"]
        == "jansen_ml4t_chapter_9_time_series"
    )
    assert (
        report["governance"]["technical_method_cards"]["chapter_10_bayesian_ml"]["name"]
        == "jansen_ml4t_chapter_10_bayesian_ml"
    )
    assert report["part_i_suggested_action"] in {"Buy", "Hold", "Sell"}
    assert "technical_history_quality" in report["technical_view"]
    assert set(report["technical_view"]["support_resistance_by_timeframe"]) == {"daily", "weekly", "monthly"}
    assert report["technical_view"]["chart_metadata"]["recommended_scale"] == "log"
    assert "decision_diagnostics" in report["technical_view"]
    assert "dow_action_filter" in report["technical_view"]
    assert "reversal_action_filter" in report["technical_view"]
    assert "triangle_action_filter" in report["technical_view"]
    assert "chapter_9_action_filter" in report["technical_view"]
    assert "chapter_10_action_filter" in report["technical_view"]
    assert "chapter_11_action_filter" in report["technical_view"]
    assert "chapter_12_gap_filter" in report["technical_view"]
    assert "chapter_13_zone_filter" in report["technical_view"]
    assert "chapter_14_trendline_filter" in report["technical_view"]
    assert "chapter_15_major_trendline_filter" in report["technical_view"]
    assert "market_only_vs_enriched" in report["technical_view"]
    assert len(report["forecasts"]) == 3
    assert set(report["governance"]["model_cards"]) == {"1", "5", "30"}
    assert report["governance"]["technical_method_card"]["name"] == "dow_theory_inspired_market_action"
    assert report["governance"]["technical_method_cards"]["magee_basing_points"]["name"] == "magee_basing_points"
    assert report["governance"]["technical_method_cards"]["reversal_patterns"]["name"] == "edwards_magee_reversal_patterns"
    assert report["governance"]["technical_method_cards"]["triangle_patterns"]["name"] == "edwards_magee_triangle_patterns"
    assert report["governance"]["technical_method_cards"]["chapter_9_patterns"]["name"] == "edwards_magee_chapter_9_patterns"
    assert report["governance"]["technical_method_cards"]["chapter_10_patterns"]["name"] == "edwards_magee_chapter_10_patterns"
    assert report["governance"]["technical_method_cards"]["chapter_11_patterns"]["name"] == "edwards_magee_chapter_11_continuation_patterns"
    assert report["governance"]["technical_method_cards"]["chapter_12_gaps"]["name"] == "edwards_magee_chapter_12_gaps"
    assert report["governance"]["technical_method_cards"]["chapter_13_support_resistance"]["name"] == "edwards_magee_chapter_13_support_resistance"
    assert report["governance"]["technical_method_cards"]["chapter_14_trendlines"]["name"] == "edwards_magee_chapter_14_trendlines_channels"
    assert report["governance"]["technical_method_cards"]["chapter_15_major_trendlines"]["name"] == "edwards_magee_chapter_15_major_trendlines"
    assert report["governance"]["technical_method_cards"]["chapter_16_market_context"]["name"] == "edwards_magee_chapter_16_market_context"
    assert report["governance"]["technical_method_cards"]["chapter_17_governance_context"]["name"] == "edwards_magee_chapter_17_governance_context"
    assert report["governance"]["tactical_method_cards"]["chapter_18_tactical_problem"]["name"] == "edwards_magee_chapter_18_tactical_problem"
    assert report["governance"]["operational_method_cards"]["chapter_19_validation"]["name"] == "edwards_magee_chapter_19_operational_validation"
    assert report["governance"]["selection_method_cards"]["chapter_20_ticker_suitability"]["name"] == "edwards_magee_chapter_20_ticker_suitability"
    assert report["governance"]["selection_method_cards"]["chapter_21_chart_selection"]["name"] == "edwards_magee_chapter_21_chart_selection"
    assert report["governance"]["trade_risk_method_cards"]["chapter_23_30_trade_risk_plan"]["name"] == "edwards_magee_chapters_23_30_trade_risk_plan"
    assert (
        report["governance"]["portfolio_method_cards"]["chapter_31_42_portfolio_capital_risk"]["name"]
        == "edwards_magee_chapters_31_38_40_42_portfolio_capital_risk"
    )
    assert (
        report["governance"]["discipline_method_cards"]["chapter_39_43_discipline_governance"]["name"]
        == "edwards_magee_chapters_39_43_discipline_governance"
    )
    assert report["governance"]["feature_registry_summary"]["feature_count"] == len(report["governance"]["feature_columns"])
    assert report["governance"]["feature_registry_summary"]["family_counts"]["volatility_risk"] >= 1
    assert len(report["governance"]["feature_registry"]) == len(report["governance"]["feature_columns"])
    assert set(report["backtests"]) == {"1", "5", "30"}
    assert set(report["diagnostics"]["factor_evaluation"]) == {"1", "5", "30"}
    assert "technical_structure" in report["diagnostics"]
    assert "dow_chapter_4_defects" in report["diagnostics"]
    assert set(report["diagnostics"]["selected_validation_predictions"]) == {"1", "5", "30"}
    decision = report["technical_view"]["decision_diagnostics"]
    assert decision["raw_action"] in {"Buy", "Hold", "Sell"}
    if decision["action"] == "Hold":
        assert decision["hold_reason"] in {
            "NoEdge",
            "RiskBlocked",
            "TrendDoubt",
            "RangeWait",
            "RegimeBlocked",
            "BasingStopTooClose",
            "BelowBasingStop",
            "LongTermTrendIntact",
            "ConfirmedReversalTop",
            "ConfirmedReversalBottom",
            "ReversalBlocked",
            "TriangleWait",
            "TriangleFailure",
            "TriangleBreakoutConflict",
            "TriangleBlocked",
            "RectangleWait",
            "RectangleFailure",
            "RectangleBreakoutConflict",
            "ConfirmedDoubleTop",
            "ConfirmedDoubleBottom",
            "ConfirmedTripleTop",
            "ConfirmedTripleBottom",
            "Chapter9Blocked",
            "BroadeningTopConflict",
            "WedgeBreakConflict",
            "DiamondBreakConflict",
            "ShortTermExhaustion",
            "Chapter10Blocked",
            "ContinuationConflict",
            "StaleContinuation",
            "FailedContinuation",
            "Chapter11Blocked",
            "ExhaustionGap",
            "IslandReversal",
            "GapBreakawayConflict",
            "Chapter12Blocked",
            "ResistanceTooClose",
            "SupportTooClose",
            "SupportFailure",
            "ResistanceBreakout",
            "Chapter13Blocked",
            "UpTrendlineBreak",
            "DownTrendlineBreak",
            "ActiveUpTrendline",
            "ActiveDownTrendline",
            "BullishFanBreak",
            "BearishFanBreak",
            "Chapter14Blocked",
            "MajorUpTrendlineBreak",
            "MajorDownTrendlineBreak",
            "MajorBullTrendIntact",
            "BroadMarketMajorDivergence",
            "Chapter15Blocked",
            "NoForecast",
        }
    for forecast in report["forecasts"]:
        assert forecast["selected_model"]
        assert forecast["lower_price"] <= forecast["predicted_price"] <= forecast["upper_price"]
        assert 0.5 <= forecast["directional_confidence"] <= 0.99
        assert forecast["confidence_interval_method"] in {
            "empirical_validation_residual_quantile",
            "normal_residual_fallback",
        }
        assert forecast["calibration_sample_size"] > 0
        assert isinstance(forecast["trade_quality"], dict)
        assert "mae" in forecast["validation_metrics"]
        assert "holdout_mae" in forecast["validation_metrics"]
        assert "deflated_sharpe_ratio" in forecast["validation_metrics"]


def test_chapter_31_42_portfolio_capital_risk_discloses_missing_account_inputs_without_overriding_action() -> None:
    report = {
        "ticker": "TEST",
        "suggested_action": "Buy",
        "current_price": 120.0,
        "risk_level": "Medium",
        "portfolio_view": {
            "position_context": {
                "status": "not_supplied",
                "reason": "No account data supplied.",
            }
        },
        "trade_risk_view": {
            "chapter_23_30_trade_risk_plan": {
                "commitment": {
                    "commitment_type": "candidate_long_commitment",
                    "entry_plan": "PlanLongEntry",
                },
                "chapter_26_position_sizing": {
                    "risk_budget_pct": 0.0075,
                    "risk_per_share": 4.0,
                },
                "execution_summary": {
                    "risk_budget_pct": 0.0075,
                    "initial_stop": 116.0,
                },
            }
        },
        "governance": {},
    }

    portfolio_capital = apply_chapter_31_42_portfolio_capital_risk(
        report,
        prices=_market_prices(),
        target_column="close",
    )

    assert report["suggested_action"] == "Buy"
    assert portfolio_capital["decision_policy"]["influences_final_action"] is False
    assert portfolio_capital["account_inputs"]["status"] == "not_supplied"
    assert portfolio_capital["portfolio_capital_gate"]["allocation_status"] == "blocked_pending_inputs"
    assert "Account equity is missing" in portfolio_capital["portfolio_capital_gate"]["blocking_reasons"][0]


def test_chapter_39_43_discipline_governance_flags_consistent_no_commitment_plan() -> None:
    report = {
        "ticker": "TEST",
        "suggested_action": "Hold",
        "risk_level": "Medium",
        "selection_metric": "mae",
        "data_version": "abc",
        "model_version": "test",
        "decision_view": {
            "chapter_18_tactical_problem": {
                "final_action": "Hold",
            },
        },
        "operations_view": {
            "chapter_19_validation": {
                "status": "pass",
                "action_gate": {
                    "validated_action": "Hold",
                    "action_override": False,
                },
            },
        },
        "selection_view": {
            "chapter_21_chart_selection": {
                "chart_selection": {
                    "chart_book_bucket": "active_review",
                },
            },
        },
        "trade_risk_view": {
            "chapter_23_30_trade_risk_plan": {
                "commitment": {
                    "commitment_type": "active_review_no_new_commitment",
                    "entry_plan": "WatchActively",
                },
            },
        },
        "portfolio_view": {
            "chapter_31_42_portfolio_capital_risk": {
                "portfolio_capital_gate": {
                    "allocation_status": "not_applicable",
                    "state": "NoNewAllocation",
                },
            },
        },
        "candidate_results": {"1": []},
        "backtests": {
            "1": {
                "cumulative_return": 0.02,
                "sharpe_ratio": 0.50,
                "hit_rate": 0.55,
                "max_drawdown": -0.03,
            }
        },
        "governance": {
            "model_cards": {"1": {}},
        },
        "diagnostics": {
            "selected_validation_predictions": {"1": [{"validation_date": "2024-01-02"}]},
            "validation_design": {"purge_window": 1},
        },
    }

    discipline = apply_chapter_39_43_discipline_governance(report)

    assert discipline["status"] == "consistent"
    assert discipline["decision_policy"]["influences_final_action"] is False
    assert discipline["discipline_gate"]["plan_adherence"] == "consistent"
    assert discipline["discipline_gate"]["new_capital_policy"] == "no_new_capital_commitment"


def test_dow_theory_diagnostics_capture_confirmation_and_close_signals() -> None:
    prices = _market_prices(rows=420)
    prices["benchmark_SPY"] = prices["close"] * 0.95
    diagnostics = analyze_dow_theory(prices)

    assert diagnostics["primary_trend"]["state"] in {"Bullish", "Neutral", "Bearish"}
    assert diagnostics["secondary_trend"]["state"] in {"Bullish", "Neutral", "Bearish"}
    assert diagnostics["minor_trend"]["state"] in {"Bullish", "Neutral", "Bearish"}
    assert diagnostics["trend_confirmation"]["status"] in {
        "Confirmed",
        "MixedConfirmation",
        "Divergent",
        "Unconfirmed",
        "StockTrendNeutral",
    }
    assert "benchmark_SPY" in diagnostics["context_trends"]
    assert "close_breakout" in diagnostics["close_confirmed_signals"]
    assert "confirms_primary_trend" in diagnostics["volume_confirmation"]
    assert "state" in diagnostics["line_pattern"]
    assert "secondary_candidate" in diagnostics["retracement"] or diagnostics["retracement"]["state"].startswith("Insufficient")
    chapter_4 = diagnostics["chapter_4_defect_diagnostics"]
    assert "signal_lag" in chapter_4
    assert "sensitivity_analysis" in chapter_4
    assert "regime_backtest" in chapter_4
    assert 0.0 <= chapter_4["ambiguity_score"] <= 1.0
    assert diagnostics["technical_method_card"]["chapter_4_controls"]["regime_backtest"]


def test_magee_basing_points_compute_stops_and_backtests() -> None:
    prices = _market_prices(rows=520)
    prices["open"] = prices["close"].shift(1).fillna(prices["close"]) * 0.995
    prices["high"] = prices["close"] * 1.015
    prices["low"] = prices["close"] * 0.985

    diagnostics = analyze_basing_points(prices)

    assert diagnostics["primary_timeframe"] == "weekly"
    assert diagnostics["preferred"]["trend_state"] in {"Long", "Short", "Neutral"}
    assert diagnostics["preferred"]["stop_status"] in {"AboveBasingStop", "BelowBasingStop", "NoActiveStop"}
    assert set(diagnostics["timeframes"]) == {"daily", "weekly"}
    weekly = diagnostics["timeframes"]["weekly"]
    assert weekly["preferred_variant"] in {"variant_1", "variant_2"}
    assert "long_only" in weekly["variants"]["variant_1"]["backtests"]
    assert "long_short" in weekly["variants"]["variant_2"]["backtests"]
    assert diagnostics["technical_method_card"]["variant_1"]["exit_signal"]


def test_reversal_patterns_detect_confirmed_head_and_shoulders_top() -> None:
    segment_lengths = [25, 15, 12, 16, 12, 12, 18, 20]
    levels = [100, 110, 130, 115, 150, 114, 132, 105, 100]
    close_values: list[float] = []
    current = levels[0]
    for target, length in zip(levels[1:], segment_lengths):
        close_values.extend(np.linspace(current, target, length + 1)[1:])
        current = target

    dates = pd.bdate_range("2023-01-02", periods=len(close_values))
    close = pd.Series(close_values, index=dates)
    volume = pd.Series(1_000_000.0, index=dates)
    for position, value in {39: 2_000_000.0, 67: 1_800_000.0, 91: 850_000.0, 103: 2_300_000.0}.items():
        volume.iloc[max(0, position - 1) : min(len(volume), position + 2)] = value
    prices = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close) * 0.997,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    diagnostics = analyze_reversal_patterns(prices)
    preferred = diagnostics["preferred"]

    assert preferred["pattern"] == "HeadAndShouldersTop"
    assert preferred["status"] in {"Confirmed", "PullbackToNeckline", "ObjectiveReached"}
    assert preferred["neckline_break_date"] is not None
    assert preferred["measured_objective"] < preferred["neckline_break_price"]
    assert preferred["volume_confirmation"]["right_shoulder_quieter"] is True
    assert diagnostics["technical_method_card"]["head_and_shoulders_top"]["confirmation"]["break_rule"]
    assert diagnostics["technical_method_card"]["version"] == "chapter_7_alignment_v1"


def test_reversal_patterns_detect_confirmed_head_and_shoulders_bottom() -> None:
    segment_lengths = [20, 20, 14, 16, 14, 16, 18, 20]
    levels = [150, 130, 100, 116, 84, 119, 98, 130, 138]
    close_values: list[float] = []
    current = levels[0]
    for target, length in zip(levels[1:], segment_lengths):
        close_values.extend(np.linspace(current, target, length + 1)[1:])
        current = target

    dates = pd.bdate_range("2023-01-02", periods=len(close_values))
    close = pd.Series(close_values, index=dates)
    volume = pd.Series(1_000_000.0, index=dates)
    for position, value in {39: 2_100_000.0, 69: 1_600_000.0, 99: 800_000.0, 112: 2_600_000.0}.items():
        volume.iloc[max(0, position - 1) : min(len(volume), position + 2)] = value
    prices = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close) * 0.997,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    diagnostics = analyze_reversal_patterns(prices)
    bottom = diagnostics["timeframes"]["daily"]["head_and_shoulders_bottom"]

    assert bottom["pattern"] == "HeadAndShouldersBottom"
    assert bottom["status"] in {"Confirmed", "ThrowbackToNeckline", "ObjectiveReached"}
    assert bottom["neckline_break_date"] is not None
    assert bottom["measured_objective"] > bottom["neckline_break_price"]
    assert bottom["prior_trend"]["state"] == "PriorDecline"
    assert diagnostics["technical_method_card"]["head_and_shoulders_bottom"]["confirmation"]["break_rule"]


def test_dormant_bottom_optional_method_detects_quiet_base_breakout() -> None:
    dates = pd.bdate_range("2022-01-03", periods=240)
    close_values = np.r_[
        np.linspace(160, 90, 70),
        90 + np.sin(np.arange(130)) * 1.0,
        np.linspace(93, 114, 40),
    ]
    volume_values = np.r_[
        np.full(70, 1_500_000.0),
        np.full(130, 300_000.0),
        np.full(40, 900_000.0),
    ]
    close = pd.Series(close_values, index=dates)
    prices = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume_values,
        },
        index=dates,
    )

    diagnostics = analyze_reversal_patterns(prices)
    dormant = diagnostics["optional_methods"]["dormant_bottoms"]["preferred"]

    assert dormant["pattern"] == "DormantBottom"
    assert dormant["status"] in {"Breakout", "Candidate"}
    assert dormant["optional"] is True
    assert dormant["decision_use"] == "context_only"
    assert dormant["volume_confirmation"]["breakout_volume_expansion"] is True


def test_triangle_patterns_detect_ascending_breakout_with_objective() -> None:
    points = [
        (0, 80),
        (20, 110),
        (30, 120),
        (40, 90),
        (50, 119),
        (60, 97),
        (70, 120),
        (80, 104),
        (90, 119),
        (100, 110),
        (106, 126),
        (116, 128),
    ]
    close_values: list[float] = []
    for (start_position, start_value), (end_position, end_value) in zip(points, points[1:]):
        length = end_position - start_position
        close_values.extend(np.linspace(start_value, end_value, length + 1)[:-1])
    close_values.append(points[-1][1])

    dates = pd.bdate_range("2023-01-02", periods=len(close_values))
    close = pd.Series(close_values, index=dates)
    volume = pd.Series(1_000_000.0, index=dates)
    volume.iloc[104:108] = 2_500_000.0
    prices = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close),
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    diagnostics = analyze_triangle_patterns(prices)
    preferred = diagnostics["preferred"]

    assert preferred["pattern"] == "AscendingTriangle"
    assert preferred["status"] == "Breakout"
    assert preferred["direction"] == "bullish"
    assert preferred["measured_objective"] > preferred["breakout_close"]
    assert preferred["boundaries"]["upper_touch_count"] >= 2
    assert preferred["boundaries"]["lower_touch_count"] >= 2
    assert preferred["volume_confirmation"]["breakout_volume_expansion"] is True
    assert diagnostics["technical_method_card"]["version"] == "chapter_8_alignment_v1"


def test_chapter_9_patterns_detect_rectangle_breakout_with_objective() -> None:
    points = [
        (0, 95),
        (18, 112),
        (34, 101),
        (50, 113),
        (66, 100),
        (82, 112),
        (98, 102),
        (108, 120),
        (118, 122),
    ]
    close_values: list[float] = []
    for (start_position, start_value), (end_position, end_value) in zip(points, points[1:]):
        length = end_position - start_position
        close_values.extend(np.linspace(start_value, end_value, length + 1)[:-1])
    close_values.append(points[-1][1])

    dates = pd.bdate_range("2023-01-02", periods=len(close_values))
    close = pd.Series(close_values, index=dates)
    volume = pd.Series(np.linspace(1_500_000.0, 850_000.0, len(close)), index=dates)
    volume.iloc[106:110] = 2_300_000.0
    prices = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close),
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    diagnostics = analyze_chapter_9_patterns(prices)
    rectangle = diagnostics["rectangle_patterns"]["preferred"]

    assert rectangle["pattern"] == "Rectangle"
    assert rectangle["status"] in {"Breakout", "ObjectiveReached"}
    assert rectangle["direction"] == "bullish"
    assert rectangle["measured_objective"] > rectangle["breakout_close"]
    assert rectangle["boundaries"]["upper_touch_count"] >= 2
    assert rectangle["boundaries"]["lower_touch_count"] >= 2
    assert rectangle["volume_confirmation"]["breakout_volume_expansion"] is True
    assert diagnostics["technical_method_card"]["version"] == "chapter_9_alignment_v1"


def test_chapter_9_patterns_detect_confirmed_double_top() -> None:
    points = [
        (0, 82),
        (55, 120),
        (90, 100),
        (135, 121),
        (165, 93),
        (180, 90),
    ]
    close_values: list[float] = []
    for (start_position, start_value), (end_position, end_value) in zip(points, points[1:]):
        length = end_position - start_position
        close_values.extend(np.linspace(start_value, end_value, length + 1)[:-1])
    close_values.append(points[-1][1])

    dates = pd.bdate_range("2023-01-02", periods=len(close_values))
    close = pd.Series(close_values, index=dates)
    volume = pd.Series(1_000_000.0, index=dates)
    volume.iloc[52:58] = 2_000_000.0
    volume.iloc[132:138] = 1_200_000.0
    prices = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close),
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    diagnostics = analyze_chapter_9_patterns(prices)
    double_top = diagnostics["multi_top_bottom_patterns"]["preferred"]

    assert double_top["pattern"] == "DoubleTop"
    assert double_top["status"] in {"Confirmed", "ObjectiveReached", "PullbackToConfirmation"}
    assert double_top["direction"] == "bearish"
    assert double_top["confirmation_level"] < double_top["level"]
    assert double_top["measured_objective"] < double_top["confirmation_level"]
    assert double_top["prior_trend"]["state"] == "PriorAdvance"
    assert double_top["volume_confirmation"]["pivot_volumes_decline"] is True


def test_chapter_10_patterns_detect_broadening_top_confirmation() -> None:
    points = [
        (0, 70),
        (45, 100),
        (60, 90),
        (80, 112),
        (100, 84),
        (120, 126),
        (138, 78),
        (150, 80),
    ]
    close_values: list[float] = []
    for (start_position, start_value), (end_position, end_value) in zip(points, points[1:]):
        length = end_position - start_position
        close_values.extend(np.linspace(start_value, end_value, length + 1)[:-1])
    close_values.append(points[-1][1])

    dates = pd.bdate_range("2023-01-02", periods=len(close_values))
    close = pd.Series(close_values, index=dates)
    volume = pd.Series(1_000_000.0, index=dates)
    for position, value in {45: 2_000_000.0, 80: 1_800_000.0, 120: 2_200_000.0, 138: 2_300_000.0}.items():
        volume.iloc[max(0, position - 1) : min(len(volume), position + 2)] = value
    prices = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close),
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    diagnostics = analyze_chapter_10_patterns(prices)
    broadening = diagnostics["timeframes"]["daily"]["broadening"]

    assert broadening["pattern"] == "BroadeningTop"
    assert broadening["status"] in {"Confirmed", "PullbackToBoundary", "ObjectiveReached"}
    assert broadening["direction"] == "bearish"
    assert broadening["confirmation_level"] < broadening["points"]["top_3"]["price"]
    assert broadening["measured_objective"] < broadening["confirmation_level"]
    assert broadening["prior_trend"]["state"] == "PriorAdvance"
    assert diagnostics["technical_method_card"]["version"] == "chapter_10_alignment_v1"


def test_chapter_10_patterns_detect_rising_wedge_breakdown() -> None:
    points = [
        (0, 80),
        (25, 95),
        (35, 88),
        (55, 102),
        (65, 96),
        (85, 108),
        (95, 104),
        (106, 96),
        (116, 94),
    ]
    close_values: list[float] = []
    for (start_position, start_value), (end_position, end_value) in zip(points, points[1:]):
        length = end_position - start_position
        close_values.extend(np.linspace(start_value, end_value, length + 1)[:-1])
    close_values.append(points[-1][1])

    dates = pd.bdate_range("2023-01-02", periods=len(close_values))
    close = pd.Series(close_values, index=dates)
    volume = pd.Series(np.linspace(1_500_000.0, 700_000.0, len(close)), index=dates)
    prices = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close),
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    diagnostics = analyze_chapter_10_patterns(prices)
    wedge = diagnostics["timeframes"]["daily"]["wedge"]

    assert wedge["pattern"] == "RisingWedge"
    assert wedge["status"] == "Breakdown"
    assert wedge["direction"] == "bearish"
    assert wedge["measured_objective"] < wedge["latest_close"]
    assert wedge["boundaries"]["width_compression_pct"] > 0
    assert wedge["volume_confirmation"]["volume_contracts_inside_pattern"] is True


def test_chapter_10_patterns_detect_key_reversal_top_event() -> None:
    rows = 80
    dates = pd.bdate_range("2023-01-02", periods=rows)
    close = pd.Series(np.linspace(80, 120, rows), index=dates)
    open_price = close.shift(1).fillna(close)
    high = close * 1.01
    low = close * 0.99
    volume = pd.Series(1_000_000.0, index=dates)
    open_price.iloc[-1] = 124.0
    high.iloc[-1] = 128.0
    low.iloc[-1] = 114.0
    close.iloc[-1] = 115.0
    volume.iloc[-1] = 2_500_000.0
    prices = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    diagnostics = analyze_chapter_10_patterns(prices)
    event = diagnostics["short_term_events"]["preferred"]
    features = build_feature_frame(prices)

    assert event["pattern"] == "KeyReversalTop"
    assert event["status"] == "Observed"
    assert event["direction"] == "bearish"
    assert event["score"] >= 0.78
    assert features["structure_key_reversal_top"].iloc[-1] == 1.0


def test_chapter_11_patterns_detect_bullish_flag_breakout() -> None:
    points = [
        (0, 90),
        (30, 122),
        (36, 118),
        (42, 120),
        (48, 116),
        (54, 118),
        (56, 128),
    ]
    close_values: list[float] = []
    for (start_position, start_value), (end_position, end_value) in zip(points, points[1:]):
        length = end_position - start_position
        close_values.extend(np.linspace(start_value, end_value, length + 1)[:-1])
    close_values.append(points[-1][1])

    dates = pd.bdate_range("2023-01-02", periods=len(close_values))
    close = pd.Series(close_values, index=dates)
    volume = pd.Series(1_000_000.0, index=dates)
    volume.iloc[:31] = np.linspace(1_800_000.0, 1_300_000.0, 31)
    volume.iloc[31:55] = np.linspace(1_000_000.0, 600_000.0, 24)
    volume.iloc[-1] = 2_000_000.0
    prices = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close),
            "high": close * 1.006,
            "low": close * 0.994,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    diagnostics = analyze_chapter_11_patterns(prices)
    flag = diagnostics["timeframes"]["daily"]["flag"]

    assert flag["pattern"] == "Flag"
    assert flag["status"] == "Breakout"
    assert flag["direction"] == "bullish"
    assert flag["measured_objective"] > flag["breakout_close"]
    assert flag["volume_confirmation"]["volume_contracts_inside_pattern"] is True
    assert flag["volume_confirmation"]["breakout_volume_expansion"] is True
    assert diagnostics["continuation_patterns"]["preferred"]["pattern"] == "Flag"
    assert diagnostics["technical_method_card"]["version"] == "chapter_11_alignment_v1"


def test_chapter_12_gaps_detect_breakaway_gap() -> None:
    rows = 100
    dates = pd.bdate_range("2023-01-02", periods=rows)
    close = pd.Series(100 + np.sin(np.arange(rows) / 3) * 0.7, index=dates)
    close.iloc[70:] = np.linspace(112, 130, rows - 70)
    high = close + 0.8
    low = close - 0.8
    open_price = close.shift(1).fillna(close)
    open_price.iloc[70:] = close.iloc[70:] - 0.2
    volume = pd.Series(1_000_000.0, index=dates)
    volume.iloc[70] = 2_500_000.0
    prices = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    diagnostics = analyze_chapter_12_gaps(prices)
    gap = diagnostics["timeframes"]["daily"]["preferred_gap"]

    assert gap["pattern"] == "BreakawayGap"
    assert gap["status"] == "Open"
    assert gap["direction"] == "bullish"
    assert gap["gap_zone"]["lower"] < gap["gap_zone"]["upper"]
    assert gap["context"]["breakout_context"] is True
    assert gap["context"]["volume_extreme"] is True
    assert diagnostics["classified_gaps"]["preferred"]["pattern"] == "BreakawayGap"
    assert diagnostics["technical_method_card"]["version"] == "chapter_12_alignment_v1"


def test_chapter_13_support_resistance_detects_old_top_as_support() -> None:
    points = [
        (0, 90),
        (25, 110),
        (40, 100),
        (65, 126),
        (82, 112),
        (100, 121),
    ]
    close_values: list[float] = []
    for (start_position, start_value), (end_position, end_value) in zip(points, points[1:]):
        length = end_position - start_position
        close_values.extend(np.linspace(start_value, end_value, length + 1)[:-1])
    close_values.append(points[-1][1])

    dates = pd.bdate_range("2023-01-02", periods=len(close_values))
    close = pd.Series(close_values, index=dates)
    high = close + 0.7
    low = close - 0.7
    volume = pd.Series(1_000_000.0, index=dates)
    prices = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close),
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    diagnostics = analyze_chapter_13_support_resistance(prices)
    support = diagnostics["timeframes"]["daily"]["nearest_support"]
    resistance = diagnostics["timeframes"]["daily"]["nearest_resistance"]

    assert support["pattern"] == "SupportResistanceZone"
    assert support["role"] == "support"
    assert support["role_reversal"] == "OldTopAsSupport"
    assert support["remaining_strength"] >= 0.45
    assert resistance["role"] == "resistance"
    assert diagnostics["technical_method_card"]["version"] == "chapter_13_alignment_v1"


def test_chapter_14_trendlines_detect_decisive_uptrend_break() -> None:
    dates = pd.bdate_range("2023-01-02", periods=160)
    positions = np.arange(len(dates))
    close = pd.Series(
        100.0 + positions * 0.25 + np.sin(positions / 5.0) * 2.0,
        index=dates,
    )
    close.iloc[-5:] = close.iloc[-5:] - np.linspace(1.0, 8.0, 5)
    prices = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close),
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=dates,
    )

    diagnostics = analyze_chapter_14_trendlines(prices)
    trendline = diagnostics["timeframes"]["daily"]["preferred_trendline"]

    assert trendline["pattern"] == "Chapter14Trendline"
    assert trendline["kind"] == "uptrend"
    assert trendline["status"] in {"DecisiveBreak", "BorderlineBreak", "InnerLineBreak"}
    assert trendline["authority_score"] >= 0.50
    assert diagnostics["timeframes"]["daily"]["channel"]["pattern"] == "Chapter14Channel"
    assert diagnostics["technical_method_card"]["version"] == "chapter_14_alignment_v1"


def test_chapter_15_major_trendlines_handle_optional_benchmark_context() -> None:
    dates = pd.bdate_range("2017-01-02", periods=252 * 8)
    positions = np.arange(len(dates))
    close = pd.Series(
        50.0 * np.exp(positions / (252 * 8) * 1.0) * (1.0 + np.sin(positions / 40.0) * 0.04),
        index=dates,
    )
    close.iloc[-60:] = close.iloc[-60:] * np.linspace(1.0, 0.82, 60)
    prices = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=dates,
    )

    no_context = analyze_chapter_15_major_trendlines(prices)
    assert no_context["broad_market_confirmation"]["status"] == "Unavailable"
    assert no_context["stock_major_trend"]["major_trendline"]["pattern"] == "Chapter15MajorTrendline"
    assert no_context["stock_major_trend"]["major_trendline"]["scale"] in {"log", "linear"}

    prices["benchmark_spy"] = 50.0 * np.exp(positions / (252 * 8) * 0.8)
    with_context = analyze_chapter_15_major_trendlines(prices)
    assert "benchmark_spy" in with_context["context_major_trends"]
    assert with_context["broad_market_confirmation"]["status"] in {"Confirmed", "Divergent", "Mixed"}
    assert with_context["technical_method_card"]["version"] == "chapter_15_alignment_v1"


def test_chapter_16_market_context_is_report_only_with_optional_futures_data() -> None:
    dates = pd.bdate_range("2020-01-02", periods=252 * 4)
    positions = np.arange(len(dates))
    close = pd.Series(
        100.0 * np.exp(positions / len(dates) * 0.55) * (1.0 + np.sin(positions / 19.0) * 0.035),
        index=dates,
    )
    prices = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close),
            "high": close * 1.012,
            "low": close * 0.988,
            "close": close,
            "volume": 5000.0 + positions * 2,
            "open_interest": 10_000.0 + positions * 5,
            "contract_multiplier": 100.0,
            "margin_requirement": 5500.0,
        },
        index=dates,
    )

    diagnostics = analyze_chapter_16_market_context(
        prices,
        ticker="GC=F",
        security_metadata={"asset_class": "future"},
    )

    assert diagnostics["decision_policy"]["mode"] == "report_only"
    assert diagnostics["decision_policy"]["influences_final_action"] is False
    assert diagnostics["asset_context"]["asset_class"] == "future"
    assert diagnostics["open_interest"]["status"] == "Measured"
    assert diagnostics["donchian_context"]["overall_state"] in {"LongBreakout", "ShortBreakout", "NoBreakout"}
    assert diagnostics["futures_risk_context"]["contract_multiplier"] == 100.0
    assert diagnostics["technical_method_card"]["version"] == "chapter_16_report_only_v1"


def test_chapter_17_governance_context_packages_llm_report_only_context() -> None:
    prices = _market_prices()
    features = build_feature_frame(prices)
    forecasts = [
        {
            "horizon_days": 5,
            "expected_direction": "Upward",
            "expected_return": 0.012,
            "directional_confidence": 0.54,
            "validation_metrics": {"mae": 0.02},
        }
    ]
    decision = {
        "hold_reason": "NoEdge",
        "edge_to_error_ratio": 0.60,
        "blocking_reasons": ["Preferred horizon directional confidence is below 55%."],
        "supporting_reasons": ["Overall risk is Medium."],
    }
    filters = {
        "dow": {
            "input_action": "Buy",
            "filtered_action": "Hold",
            "filter_applied": True,
            "blocking_reasons": ["Dow primary regime is not clear enough."],
        },
        "magee": {"input_action": "Hold", "filtered_action": "Hold", "filter_applied": False},
    }
    contexts = {
        "trend_state": {"state": "Bullish", "confidence": 0.65},
        "dow_theory": {"primary_trend": {"state": "Bearish", "confidence": 0.60}},
        "magee_basing_points": {"preferred": {"trend_state": "Long", "score": 0.60}},
        "chapter_16_market_context": {"donchian_context": {"overall_state": "LongBreakout"}},
    }
    backtests = {
        "5": {
            "rows": 20,
            "cumulative_return": -0.01,
            "benchmark_cumulative_return": 0.02,
            "sharpe_ratio": -0.2,
            "max_drawdown": -0.05,
            "hit_rate": 0.45,
            "trades": 3,
            "turnover": 0.20,
        }
    }

    diagnostics = analyze_chapter_17_governance_context(
        prices=prices,
        features=features,
        forecasts=forecasts,
        raw_action="Buy",
        final_action="Hold",
        risk_level="Medium",
        decision_diagnostics=decision,
        action_filters=filters,
        technical_contexts=contexts,
        backtests=backtests,
        data_quality_report={"status": "warn", "warnings": [{"code": "example"}]},
        technical_history_quality={"sufficient_for_classical_technical_analysis": False, "warnings": ["short history"]},
        market_feature_comparison={"status": "not_compared"},
        security_metadata={},
        data_manifest={},
    )

    assert diagnostics["decision_policy"]["mode"] == "report_only"
    assert diagnostics["decision_policy"]["influences_final_action"] is False
    assert diagnostics["llm_decision_packet"]["engine_final_action"] == "Hold"
    assert diagnostics["computer_humility"]["decision_fragility"]["level"] in {"Medium", "High"}
    assert diagnostics["computer_humility"]["filter_stack_review"]["applied_filter_count"] == 1
    assert diagnostics["method_performance_ledger"]["weak_horizons"] == [5]
    assert diagnostics["technical_method_card"]["version"] == "chapter_17_report_only_v1"


def test_chapter_18_tactical_problem_applies_rule_first_llm_safety_gate() -> None:
    prices = _market_prices()
    latest_close = float(prices["close"].iloc[-1])
    forecasts = [
        {
            "horizon_days": 5,
            "expected_direction": "Upward",
            "expected_return": 0.06,
            "directional_confidence": 0.72,
            "predicted_price": latest_close * 1.10,
            "validation_metrics": {"mae": 0.02},
        }
    ]
    contexts = {
        "magee_basing_points": {"preferred": {"active_basing_stop": latest_close * 0.96}},
        "chapter_13_support_resistance": {
            "support_zones": {"nearest": {"lower": latest_close * 0.95, "center": latest_close * 0.96}},
            "resistance_zones": {"nearest": {"center": latest_close * 1.12, "upper": latest_close * 1.13}},
        },
    }

    diagnostics = analyze_chapter_18_tactical_problem(
        prices=prices,
        forecasts=forecasts,
        part_i_action="Buy",
        raw_action="Buy",
        risk_level="Medium",
        decision_diagnostics={"blocking_reasons": [], "supporting_reasons": []},
        technical_contexts=contexts,
        latest_features={},
        tactical_profile="intermediate",
        enable_llm_review=True,
        llm_review_override={
            "status": "executed",
            "provider": "test",
            "recommended_action": "Hold",
            "confidence": 0.8,
            "rationale": "Reward exists, but downgrade for test safety.",
        },
    )

    assert diagnostics["rule_based_action"] == "Buy"
    assert diagnostics["final_action"] == "Hold"
    assert diagnostics["llm_safety_gate"]["status"] == "accepted_downgrade"
    assert diagnostics["trade_plan"]["stop_plan"]["status"] == "Selected"
    assert diagnostics["trade_plan"]["target_plan"]["reward_to_risk"] >= 1.5
    assert diagnostics["technical_method_card"]["version"] == "chapter_18_tactical_plan_v1"


def test_chapter_18_tactical_problem_rejects_llm_upgrade_from_hold() -> None:
    prices = _market_prices()
    latest_close = float(prices["close"].iloc[-1])
    diagnostics = analyze_chapter_18_tactical_problem(
        prices=prices,
        forecasts=[
            {
                "horizon_days": 5,
                "expected_direction": "Upward",
                "expected_return": 0.01,
                "directional_confidence": 0.52,
                "predicted_price": latest_close * 1.02,
            }
        ],
        part_i_action="Hold",
        raw_action="Buy",
        risk_level="Medium",
        decision_diagnostics={"hold_reason": "NoEdge", "blocking_reasons": ["No edge"]},
        technical_contexts={},
        tactical_profile="intermediate",
        enable_llm_review=True,
        llm_review_override={
            "status": "executed",
            "provider": "test",
            "recommended_action": "Buy",
            "confidence": 0.9,
            "rationale": "Invalid test upgrade.",
        },
    )

    assert diagnostics["rule_based_action"] == "Hold"
    assert diagnostics["final_action"] == "Hold"
    assert diagnostics["llm_safety_gate"]["status"] == "rejected"
    assert "cannot upgrade" in diagnostics["llm_safety_gate"]["reason"]


def test_chapter_19_validation_blocks_directional_action_when_run_is_not_auditable() -> None:
    report = {
        "ticker": "TEST",
        "suggested_action": "Buy",
        "current_price": 100.0,
        "decision_view": {
            "final_governed_action": "Buy",
            "llm_safety_gate": {"final_action": "Buy"},
        },
        "technical_view": {
            "trend_state": {"state": "Bullish"},
            "dow_theory": {"primary_trend": {"state": "Bullish"}},
            "magee_basing_points": {"preferred": {"trend_state": "Long"}},
            "chapter_17_governance_context": {"decision_policy": {"mode": "report_only"}},
            "chapter_18_tactical_problem": {"final_action": "Buy"},
            "decision_diagnostics": {"action": "Buy"},
        },
        "diagnostics": {
            "data_quality": {"status": "fail", "warnings": [{"severity": "high", "code": "duplicate_dates"}]},
            "selected_validation_predictions": {},
        },
        "data_manifest": {"ticker": "TEST"},
        "data_version": "abc",
        "forecasts": [],
        "candidate_results": {},
        "backtests": {},
        "governance": {
            "config": {"min_training_rows": 10},
            "model_cards": {},
            "feature_registry": [],
            "technical_method_cards": {"chapter_17_governance_context": {"name": "chapter_17"}},
            "tactical_method_cards": {"chapter_18_tactical_problem": {"name": "chapter_18"}},
        },
    }

    validation = apply_chapter_19_validation(report, prices=_market_prices(rows=60))

    assert validation["status"] == "fail"
    assert validation["action_gate"]["hard_block_new_commitments"] is True
    assert validation["action_gate"]["validated_action"] == "Hold"
    assert report["suggested_action"] == "Hold"
    assert report["decision_view"]["final_operational_action"] == "Hold"


def test_chapter_20_ticker_suitability_scores_completed_report_without_overriding_action() -> None:
    prices = _market_prices(rows=260)
    prices["open"] = prices["close"].shift(1).fillna(prices["close"]) * 0.997
    prices["high"] = prices["close"] * 1.015
    prices["low"] = prices["close"] * 0.985
    original_action = "Buy"
    report = {
        "ticker": "TEST",
        "suggested_action": original_action,
        "risk_level": "Medium",
        "forecasts": [
            {
                "horizon_days": 5,
                "directional_confidence": 0.66,
                "validation_metrics": {"mae": 0.015, "holdout_mae": 0.014},
            }
        ],
        "operations_view": {"chapter_19_validation": {"status": "pass"}},
        "decision_view": {
            "chapter_18_tactical_problem": {
                "trade_plan": {"reward_to_risk": 2.0},
                "rule_gate": {"hard_blockers": []},
            }
        },
        "technical_view": {
            "trend_state": {"state": "Bullish"},
            "decision_diagnostics": {"edge_to_error_ratio": 1.6},
            "chapter_17_governance_context": {
                "computer_humility": {
                    "decision_fragility": {"level": "Low"},
                    "method_conflict_score": {"level": "Low"},
                }
            },
        },
        "governance": {"security_metadata": {"asset_class": "equity", "sector": "Technology"}},
        "diagnostics": {},
    }

    suitability = apply_chapter_20_ticker_suitability(report, prices=prices)

    assert report["suggested_action"] == original_action
    assert suitability["decision_policy"]["mode"] == "selection_suitability_report_only"
    assert suitability["profile_fit"]["primary_profile"] in suitability["profile_scores"]
    assert suitability["profile_fit"]["selection_hint"] in {
        "active_review",
        "keep_watching",
        "monitor_only",
        "avoid_for_now",
    }
    assert suitability["llm_integration"]["status"] == "planned"
    assert report["governance"]["selection_method_cards"]["chapter_20_ticker_suitability"]["version"] == "chapter_20_selection_profile_v1"


def test_chapter_21_chart_selection_uses_chapter_20_without_overriding_action() -> None:
    report = {
        "ticker": "TEST",
        "as_of_date": "2026-05-27",
        "current_price": 100.0,
        "suggested_action": "Buy",
        "risk_level": "Medium",
        "operations_view": {
            "chapter_19_validation": {
                "status": "pass",
                "action_gate": {"hard_block_new_commitments": False},
            }
        },
        "selection_view": {
            "chapter_20_ticker_suitability": {
                "profile_fit": {
                    "primary_profile": "intermediate_trader",
                    "classification": "suitable_trade_candidate",
                    "suitability_score": 0.78,
                    "selection_hint": "active_review",
                    "trade_candidate_eligible": True,
                    "active_review_eligible": True,
                    "watchlist_eligible": True,
                },
                "component_scores": {
                    "tactical_readiness": {"score": 0.80},
                },
            }
        },
        "governance": {"selection_method_cards": {}},
    }

    selection = apply_chapter_21_chart_selection(report)

    assert report["suggested_action"] == "Buy"
    assert selection["decision_policy"]["mode"] == "chart_selection_report_only"
    assert selection["chart_selection"]["chart_book_bucket"] == "trade_candidates"
    assert selection["chart_selection"]["trade_candidate"] is True
    assert selection["review_plan"]["review_cadence"] == "daily_until_resolved"
    assert report["governance"]["selection_method_cards"]["chapter_21_chart_selection"]["version"] == "chapter_21_chart_book_selection_v1"


def test_chapter_23_30_trade_risk_plan_builds_position_and_stop_controls_without_overriding_action() -> None:
    prices = _market_prices(rows=260)
    prices["open"] = prices["close"].shift(1).fillna(prices["close"]) * 0.997
    prices["high"] = prices["close"] * 1.015
    prices["low"] = prices["close"] * 0.985
    report = {
        "ticker": "TEST",
        "as_of_date": "2026-05-27",
        "current_price": 100.0,
        "suggested_action": "Buy",
        "risk_level": "Medium",
        "decision_view": {
            "chapter_18_tactical_problem": {
                "trade_plan": {
                    "stop_plan": {"status": "Selected", "level": 95.0, "distance_pct": 0.05, "source": "chapter_13_support_zone"},
                    "target_plan": {"status": "Selected", "level": 112.0, "reward_to_risk": 2.4},
                    "reward_to_risk": 2.4,
                    "max_loss_pct": 0.05,
                }
            }
        },
        "operations_view": {"chapter_19_validation": {"status": "pass", "action_gate": {"hard_block_new_commitments": False}}},
        "selection_view": {
            "chapter_20_ticker_suitability": {
                "profile_fit": {"primary_profile": "intermediate_trader"},
                "instrument_habit_profile": {
                    "status": "measured",
                    "realized_volatility_20d": 0.25,
                    "atr_pct_20d": 0.02,
                    "max_drawdown_252d": -0.12,
                },
            },
            "chapter_21_chart_selection": {
                "chart_selection": {"chart_book_bucket": "trade_candidates"},
            },
        },
        "technical_view": {
            "chapter_13_support_resistance": {
                "support_zones": {"nearest": {"lower": 94.0, "center": 96.0, "upper": 98.0}},
                "resistance_zones": {"nearest": {"lower": 110.0, "center": 112.0, "upper": 114.0}},
            },
            "chapter_14_trendlines": {
                "trendlines": {"preferred": {"kind": "uptrend", "status": "Active", "effective_decisive_break": False}}
            },
            "magee_basing_points": {"preferred": {"trend_state": "Long", "active_basing_stop": 95.0, "stop_status": "AboveBasingStop"}},
        },
        "governance": {"trade_risk_method_cards": {}},
    }

    trade_risk = apply_chapter_23_30_trade_risk_plan(report, prices=prices)

    assert report["suggested_action"] == "Buy"
    assert trade_risk["decision_policy"]["mode"] == "trade_risk_plan_report_only"
    assert trade_risk["commitment"]["commitment_type"] == "candidate_long_commitment"
    assert trade_risk["chapter_26_position_sizing"]["example_units_per_100k_account"] is not None
    assert trade_risk["chapter_27_stop_order_plan"]["initial_stop"] == 95.0
    assert trade_risk["chapter_25_margin_short_policy"]["margin_allowed"] is False
    assert report["governance"]["trade_risk_method_cards"]["chapter_23_30_trade_risk_plan"]["version"] == "chapter_23_30_trade_risk_v1"


def test_market_only_vs_enriched_comparison_runs_when_context_features_exist() -> None:
    prices = _market_prices()
    prices["benchmark_SPY"] = prices["close"] * (1 + np.sin(np.arange(len(prices)) / 17) * 0.02)
    config = ForecastConfig(
        ticker="TEST",
        horizons=(1,),
        min_training_rows=120,
        validation_window=30,
        step_size=30,
        max_splits=2,
        include_lightgbm=False,
        include_statistical_models=False,
    )

    report = ForecastingEngine(config).run(prices)
    comparison = report["technical_view"]["market_only_vs_enriched"]

    assert comparison["status"] == "compared"
    assert "benchmark_spy" in comparison["external_columns"]
    assert comparison["horizons"][0]["status"] == "compared"


def test_factor_evaluation_and_backtest_helpers_return_metrics() -> None:
    prices = _market_prices()
    features = build_feature_frame(prices)
    supervised = add_forward_return_targets(features, prices, horizons=(1,))

    factors = evaluate_factors(features, supervised, horizons=(1,), top_n=5)

    assert "1" in factors
    assert len(factors["1"]) > 0
    assert {"feature", "rank_ic", "quantile_spread_return", "top_quantile_turnover"}.issubset(factors["1"][0])

    records = [
        {
            "validation_date": str(date.date()),
            "actual_log_return": actual,
            "predicted_log_return": predicted,
        }
        for date, actual, predicted in zip(
            pd.bdate_range("2024-01-02", periods=5),
            [0.01, -0.02, 0.015, 0.005, -0.01],
            [0.008, -0.01, -0.004, 0.002, -0.005],
        )
    ]
    backtest = backtest_validation_signals(records, horizon_days=1, transaction_cost_bps=1.0)

    assert backtest["rows"] == 5
    assert "cumulative_return" in backtest
    assert "benchmark_cumulative_return" in backtest
    assert backtest["execution_timing"] == "vectorized_same_period_close_to_close"
    assert len(backtest["slippage_sensitivity"]) == 5
    assert backtest["trade_ledger_summary"]["trade_count"] >= 1


def test_pairs_trading_helper_is_standalone_multi_ticker_research() -> None:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-02", periods=180)
    base = 100 + np.cumsum(rng.normal(0.0, 0.7, len(dates)))
    asset_a = pd.Series(base + rng.normal(0.0, 0.4, len(dates)), index=dates)
    asset_b = pd.Series(base * 1.01 + rng.normal(0.0, 0.4, len(dates)), index=dates)
    asset_c = pd.Series(80 + np.cumsum(rng.normal(0.0, 1.5, len(dates))), index=dates)

    pair = analyze_pair(asset_a, asset_b, symbol_a="AAA", symbol_b="BBB", min_rows=120)
    assert pair["decision_policy"]["influences_single_ticker_forecast"] is False
    assert pair["rows"] == 180
    assert pair["status"] in {"candidate", "not_cointegrated"}
    assert "cointegration" in pair
    assert pair["signal"] in {"long_spread", "short_spread", "neutral_exit_zone", "watch"}

    ranked = rank_cointegrated_pairs(pd.DataFrame({"AAA": asset_a, "BBB": asset_b, "CCC": asset_c}), min_rows=120)
    assert len(ranked) == 3
    assert {"symbol_a", "symbol_b", "hedge_ratio", "spread_zscore"}.issubset(ranked[0])


def test_write_plot_artifacts_creates_forecast_and_validation_plots(tmp_path) -> None:
    prices = _market_prices()
    config = ForecastConfig(
        ticker="TEST",
        horizons=(1, 5),
        min_training_rows=120,
        validation_window=30,
        step_size=30,
        max_splits=2,
        include_lightgbm=False,
        include_statistical_models=False,
    )
    report = ForecastingEngine(config).run(prices)

    artifacts = write_plot_artifacts(report, prices, tmp_path)

    assert {
        "forecast_plot",
        "forecast_plotly",
        "technical_chart",
        "technical_chart_plotly",
        "technical_clean_chart",
        "technical_clean_chart_plotly",
        "technical_daily_chart",
        "technical_daily_chart_plotly",
        "technical_weekly_chart",
        "technical_weekly_chart_plotly",
        "technical_monthly_chart",
        "technical_monthly_chart_plotly",
        "validation_plot_1d",
        "validation_plot_5d",
        "validation_plotly_1d",
        "validation_plotly_5d",
    }.issubset(artifacts)
    for path in artifacts.values():
        output = tmp_path / Path(path).relative_to(tmp_path)
        assert output.exists()
        assert output.stat().st_size > 0
        if output.suffix == ".html":
            assert "Plotly.newPlot" in output.read_text(encoding="utf-8")


def test_portfolio_pdf_text_extraction_maps_holdings(monkeypatch: pytest.MonkeyPatch) -> None:
    sample_text = """
BROKERAGE
PCS. / NOMINAL SECURITY NAME PRICE PER PIECE VALUE IN EUR
0.019553 Pcs. Taiwan Semiconduct.Manufact.Co 345.50 6.76
ISIN: US8740391003
1.046876 Pcs. Automation & Robotics USD (Acc) 17.03 17.83
ISIN: IE00BYZK4552
CRYPTO WALLET
0.004877 Pcs. Ethereum 1,835.13 8.95
ETH 20.05.2026
CASH
Current account 42.26 EUR
"""
    monkeypatch.setattr(portfolio, "_extract_pdf_text", lambda _: sample_text)

    holdings = portfolio.extract_portfolio_holdings("statement.pdf")

    assert len(holdings) == 3
    assert holdings[0].security_name == "Taiwan Semiconduct.Manufact.Co"
    assert holdings[0].symbol_candidates == ("TSM",)
    assert holdings[1].symbol_candidates[0] == "2B76.DE"
    assert holdings[2].symbol_candidates == ("ETH-USD",)
    assert holdings[2].statement_price == 1835.13
