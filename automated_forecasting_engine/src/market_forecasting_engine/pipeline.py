from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from market_forecasting_engine.backtest import backtest_validation_signals
from market_forecasting_engine.basing_points import analyze_basing_points
from market_forecasting_engine.chapter_1_ml4t_workflow import analyze_chapter_1_ml4t_workflow
from market_forecasting_engine.chapter_10_bayesian_ml import analyze_chapter_10_bayesian_ml
from market_forecasting_engine.chapter_10_patterns import analyze_chapter_10_patterns
from market_forecasting_engine.chapter_11_patterns import analyze_chapter_11_patterns
from market_forecasting_engine.chapter_11_tree_models import analyze_chapter_11_tree_models
from market_forecasting_engine.chapter_12_boosting_models import analyze_chapter_12_boosting_models
from market_forecasting_engine.chapter_12_gaps import analyze_chapter_12_gaps
from market_forecasting_engine.chapter_13_support_resistance import analyze_chapter_13_support_resistance
from market_forecasting_engine.chapter_14_trendlines import analyze_chapter_14_trendlines
from market_forecasting_engine.chapter_15_major_trendlines import analyze_chapter_15_major_trendlines
from market_forecasting_engine.chapter_16_market_context import analyze_chapter_16_market_context
from market_forecasting_engine.chapter_17_deep_learning import analyze_chapter_17_deep_learning
from market_forecasting_engine.chapter_17_governance_context import analyze_chapter_17_governance_context
from market_forecasting_engine.chapter_18_cnn import analyze_chapter_18_cnn
from market_forecasting_engine.chapter_18_tactics import analyze_chapter_18_tactical_problem
from market_forecasting_engine.chapter_19_validation import apply_chapter_19_validation
from market_forecasting_engine.chapter_2_market_data import analyze_chapter_2_market_data
from market_forecasting_engine.chapter_3_alternative_data import analyze_chapter_3_alternative_data
from market_forecasting_engine.chapter_20_selection import apply_chapter_20_ticker_suitability
from market_forecasting_engine.chapter_21_chart_selection import apply_chapter_21_chart_selection
from market_forecasting_engine.chapter_23_30_trade_risk import apply_chapter_23_30_trade_risk_plan
from market_forecasting_engine.chapter_31_42_portfolio_risk import apply_chapter_31_42_portfolio_capital_risk
from market_forecasting_engine.chapter_39_43_discipline import apply_chapter_39_43_discipline_governance
from market_forecasting_engine.chapter_4_alpha_research import analyze_chapter_4_alpha_research
from market_forecasting_engine.chapter_5_portfolio_evaluation import analyze_chapter_5_portfolio_evaluation
from market_forecasting_engine.chapter_6_ml_process import analyze_chapter_6_ml_process
from market_forecasting_engine.chapter_7_linear_models import analyze_chapter_7_linear_models
from market_forecasting_engine.chapter_8_backtesting import analyze_chapter_8_backtesting
from market_forecasting_engine.chapter_9_patterns import analyze_chapter_9_patterns
from market_forecasting_engine.chapter_9_time_series import analyze_chapter_9_time_series
from market_forecasting_engine.calendar import summarize_calendar_alignment
from market_forecasting_engine.data import data_version_hash, normalize_price_frame
from market_forecasting_engine.data_manifest import build_data_manifest
from market_forecasting_engine.data_quality import build_data_quality_report
from market_forecasting_engine.daily_trade import add_trading_bars, add_trading_minutes, build_intraday_feature_frame
from market_forecasting_engine.dip_buy import annotate_mean_reversion_dip_buy
from market_forecasting_engine.dow_theory import analyze_dow_theory
from market_forecasting_engine.factor_evaluation import evaluate_factors
from market_forecasting_engine.feature_registry import build_feature_registry
from market_forecasting_engine.features import add_forward_return_targets, build_feature_frame
from market_forecasting_engine.governance import build_model_card
from market_forecasting_engine.models import default_candidates
from market_forecasting_engine.reversal_patterns import analyze_reversal_patterns
from market_forecasting_engine.risk import risk_level, suggested_action
from market_forecasting_engine.schema import ForecastConfig, HorizonForecast
from market_forecasting_engine.security_master import resolve_security_metadata
from market_forecasting_engine.technical_structure import latest_structure_snapshot
from market_forecasting_engine.triangle_patterns import analyze_triangle_patterns
from market_forecasting_engine.validation import (
    apply_ml4t_selection_adjustments,
    select_candidate,
    validate_candidates,
    validation_summaries_as_dict,
)


MARKET_DATA_COLUMNS = {"open", "high", "low", "close", "volume", "dividends", "stock_splits"}
HIGHER_IS_BETTER_METRICS = {"directional_accuracy", "sharpe_ratio", "hit_rate", "profit_factor"}


class ForecastingEngine:
    """Automated model selection and governance pipeline for stock forecasts."""

    def __init__(self, config: ForecastConfig, progress_callback: Callable[[dict[str, object]], None] | None = None) -> None:
        self.config = config
        self.progress_callback = progress_callback

    def _emit_progress(self, event: str, **payload: object) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback({"event": event, "ticker": self.config.ticker, **payload})
        except Exception:
            return

    def run(
        self,
        prices: pd.DataFrame,
        data_manifest: dict[str, Any] | None = None,
        data_quality_report: dict[str, Any] | None = None,
        security_metadata: dict[str, Any] | None = None,
        long_term_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_prices = normalize_price_frame(prices, target_column=self.config.target_column)
        if len(normalized_prices) < self.config.min_training_rows:
            raise ValueError(
                f"Need at least {self.config.min_training_rows} price rows; received {len(normalized_prices)}."
            )

        data_hash = data_version_hash(normalized_prices)
        if data_quality_report is None:
            data_quality_report = build_data_quality_report(
                normalized_prices,
                target_column=self.config.target_column,
            )
        if security_metadata is None:
            security_metadata = resolve_security_metadata(
                ticker=self.config.ticker,
                prices=normalized_prices,
            )
        if data_manifest is None:
            data_manifest = build_data_manifest(
                prices=normalized_prices,
                ticker=self.config.ticker,
                target_column=self.config.target_column,
                calendar_summary=summarize_calendar_alignment(normalized_prices),
                security_metadata=security_metadata,
            )
        if long_term_context:
            data_manifest.setdefault(
                "long_term_sources",
                {
                    "status": long_term_context.get("status"),
                    "providers_requested": long_term_context.get("providers_requested", []),
                    "provider_summaries": long_term_context.get("provider_summaries", {}),
                    "artifacts": long_term_context.get("artifacts", {}),
                    "model_feature_policy": long_term_context.get("model_feature_policy", {}),
                },
            )

        features = build_feature_frame(normalized_prices, target_column=self.config.target_column)
        if self.config.forecast_interval_minutes is not None and self.config.forecast_interval_minutes < 18 * 60:
            intraday_features = build_intraday_feature_frame(normalized_prices, target_column=self.config.target_column)
            new_columns = [column for column in intraday_features.columns if column not in features.columns]
            if new_columns:
                features = pd.concat([features, intraday_features[new_columns]], axis=1)
        supervised = add_forward_return_targets(
            features=features,
            prices=normalized_prices,
            horizons=self.config.horizons,
            target_column=self.config.target_column,
        )
        factor_evaluation = evaluate_factors(
            features=features,
            supervised=supervised,
            horizons=self.config.horizons,
            top_n=self.config.factor_top_n,
        )
        feature_registry = build_feature_registry(features, factor_evaluation=factor_evaluation)
        structure_snapshot = latest_structure_snapshot(features)
        technical_history_quality = _technical_history_quality(
            normalized_prices,
            target_column=self.config.target_column,
            data_quality_report=data_quality_report,
        )
        support_resistance_by_timeframe = _support_resistance_by_timeframe(
            normalized_prices,
            target_column=self.config.target_column,
        )
        dow_theory = analyze_dow_theory(
            normalized_prices,
            target_column=self.config.target_column,
            transaction_cost_bps=self.config.transaction_cost_bps,
        )
        magee_basing_points = analyze_basing_points(
            normalized_prices,
            target_column=self.config.target_column,
            transaction_cost_bps=self.config.transaction_cost_bps,
        )
        reversal_patterns = analyze_reversal_patterns(
            normalized_prices,
            target_column=self.config.target_column,
        )
        triangle_patterns = analyze_triangle_patterns(
            normalized_prices,
            target_column=self.config.target_column,
        )
        chapter_9_patterns = analyze_chapter_9_patterns(
            normalized_prices,
            target_column=self.config.target_column,
        )
        chapter_10_patterns = analyze_chapter_10_patterns(
            normalized_prices,
            target_column=self.config.target_column,
        )
        chapter_11_patterns = analyze_chapter_11_patterns(
            normalized_prices,
            target_column=self.config.target_column,
        )
        chapter_12_gaps = analyze_chapter_12_gaps(
            normalized_prices,
            target_column=self.config.target_column,
        )
        chapter_13_support_resistance = analyze_chapter_13_support_resistance(
            normalized_prices,
            target_column=self.config.target_column,
        )
        chapter_14_trendlines = analyze_chapter_14_trendlines(
            normalized_prices,
            target_column=self.config.target_column,
        )
        chapter_15_major_trendlines = analyze_chapter_15_major_trendlines(
            normalized_prices,
            target_column=self.config.target_column,
        )
        chapter_16_market_context = analyze_chapter_16_market_context(
            normalized_prices,
            target_column=self.config.target_column,
            ticker=self.config.ticker,
            security_metadata=security_metadata,
        )
        latest_features = features.iloc[[-1]]
        latest_price = float(normalized_prices[self.config.target_column].iloc[-1])
        as_of_timestamp = pd.Timestamp(normalized_prices.index[-1])
        as_of_date = str(as_of_timestamp.date())

        forecasts: list[dict[str, Any]] = []
        all_candidate_results: dict[str, list[dict[str, object]]] = {}
        model_cards: dict[str, Any] = {}
        selected_validation_predictions: dict[str, list[dict[str, Any]]] = {}

        for horizon in self.config.horizons:
            self._emit_progress("horizon_started", horizon_days=horizon)
            forecast, validation_results, model_card, validation_records = self._forecast_horizon(
                horizon=horizon,
                supervised=supervised,
                latest_features=latest_features,
                latest_price=latest_price,
                price_series=normalized_prices[self.config.target_column],
                data_hash=data_hash,
                factor_evaluation=factor_evaluation,
                data_manifest=data_manifest,
            )
            forecasts.append(forecast.to_dict())
            all_candidate_results[str(horizon)] = validation_summaries_as_dict(validation_results)
            model_cards[str(horizon)] = model_card
            selected_validation_predictions[str(horizon)] = validation_records
            self._emit_progress(
                "horizon_completed",
                horizon_days=horizon,
                selected_model=forecast.selected_model,
                expected_direction=forecast.expected_direction,
                directional_confidence=forecast.directional_confidence,
            )

        risk = _portfolio_risk_level(forecasts)
        raw_action = suggested_action(forecasts, risk=risk)
        dow_action_filter = _apply_dow_regime_filter(raw_action, dow_theory)
        magee_action_filter = _apply_magee_basing_filter(dow_action_filter["filtered_action"], magee_basing_points)
        reversal_action_filter = _apply_reversal_pattern_filter(
            magee_action_filter["filtered_action"],
            reversal_patterns,
        )
        triangle_action_filter = _apply_triangle_pattern_filter(
            reversal_action_filter["filtered_action"],
            triangle_patterns,
        )
        chapter_9_action_filter = _apply_chapter_9_pattern_filter(
            triangle_action_filter["filtered_action"],
            chapter_9_patterns,
        )
        chapter_10_action_filter = _apply_chapter_10_pattern_filter(
            chapter_9_action_filter["filtered_action"],
            chapter_10_patterns,
        )
        chapter_11_action_filter = _apply_chapter_11_continuation_filter(
            chapter_10_action_filter["filtered_action"],
            chapter_11_patterns,
        )
        chapter_12_gap_filter = _apply_chapter_12_gap_filter(
            chapter_11_action_filter["filtered_action"],
            chapter_12_gaps,
        )
        chapter_13_zone_filter = _apply_chapter_13_support_resistance_filter(
            chapter_12_gap_filter["filtered_action"],
            chapter_13_support_resistance,
        )
        chapter_14_trendline_filter = _apply_chapter_14_trendline_filter(
            chapter_13_zone_filter["filtered_action"],
            chapter_14_trendlines,
        )
        chapter_15_major_trendline_filter = _apply_chapter_15_major_trendline_filter(
            chapter_14_trendline_filter["filtered_action"],
            chapter_15_major_trendlines,
        )
        action = chapter_15_major_trendline_filter["filtered_action"]
        trend_view = _trend_view(
            forecasts=forecasts,
            latest_features=features.iloc[-1],
            latest_price=latest_price,
            structure_snapshot=structure_snapshot,
            support_resistance_by_timeframe=support_resistance_by_timeframe,
        )
        decision_diagnostics = _decision_diagnostics(
            forecasts=forecasts,
            action=action,
            raw_action=raw_action,
            risk=risk,
            dow_action_filter=dow_action_filter,
            magee_action_filter=magee_action_filter,
            reversal_action_filter=reversal_action_filter,
            triangle_action_filter=triangle_action_filter,
            chapter_9_action_filter=chapter_9_action_filter,
            chapter_10_action_filter=chapter_10_action_filter,
            chapter_11_action_filter=chapter_11_action_filter,
            chapter_12_gap_filter=chapter_12_gap_filter,
            chapter_13_zone_filter=chapter_13_zone_filter,
            chapter_14_trendline_filter=chapter_14_trendline_filter,
            chapter_15_major_trendline_filter=chapter_15_major_trendline_filter,
            dow_theory=dow_theory,
            magee_basing_points=magee_basing_points,
            reversal_patterns=reversal_patterns,
            triangle_patterns=triangle_patterns,
            chapter_9_patterns=chapter_9_patterns,
            chapter_10_patterns=chapter_10_patterns,
            chapter_11_patterns=chapter_11_patterns,
            chapter_12_gaps=chapter_12_gaps,
            chapter_13_support_resistance=chapter_13_support_resistance,
            chapter_14_trendlines=chapter_14_trendlines,
            chapter_15_major_trendlines=chapter_15_major_trendlines,
        )
        market_feature_comparison = self._market_only_feature_comparison(
            normalized_prices=normalized_prices,
            full_forecasts=forecasts,
        )
        backtests = {
            horizon: backtest_validation_signals(
                records,
                horizon_days=int(horizon),
                transaction_cost_bps=self.config.transaction_cost_bps,
            )
            for horizon, records in selected_validation_predictions.items()
        }
        chapter_1_ml4t_workflow = analyze_chapter_1_ml4t_workflow(
            features=features,
            supervised=supervised,
            forecasts=forecasts,
            factor_evaluation=factor_evaluation,
            candidate_results=all_candidate_results,
            selected_validation_predictions=selected_validation_predictions,
            backtests=backtests,
            data_manifest=data_manifest,
            data_quality_report=data_quality_report,
            final_action=action,
        )
        chapter_2_market_data = analyze_chapter_2_market_data(
            prices=normalized_prices,
            data_manifest=data_manifest,
            data_quality_report=data_quality_report,
            target_column=self.config.target_column,
        )
        chapter_3_alternative_data = analyze_chapter_3_alternative_data(
            data_manifest=data_manifest,
            data_quality_report=data_quality_report,
            market_feature_comparison=market_feature_comparison,
        )
        chapter_4_alpha_research = analyze_chapter_4_alpha_research(
            features=features,
            supervised=supervised,
            factor_evaluation=factor_evaluation,
            feature_registry=feature_registry,
            horizons=self.config.horizons,
        )
        chapter_5_portfolio_evaluation = analyze_chapter_5_portfolio_evaluation(
            backtests=backtests,
            selected_validation_predictions=selected_validation_predictions,
            factor_evaluation=factor_evaluation,
            market_feature_comparison=market_feature_comparison,
            horizons=self.config.horizons,
        )
        chapter_6_ml_process = analyze_chapter_6_ml_process(
            features=features,
            supervised=supervised,
            forecasts=forecasts,
            candidate_results=all_candidate_results,
            selected_validation_predictions=selected_validation_predictions,
            horizons=self.config.horizons,
        )
        chapter_7_linear_models = analyze_chapter_7_linear_models(
            forecasts=forecasts,
            candidate_results=all_candidate_results,
            selected_validation_predictions=selected_validation_predictions,
            model_cards=model_cards,
            horizons=self.config.horizons,
            selection_metric=self.config.selection_metric,
        )
        chapter_8_backtesting = analyze_chapter_8_backtesting(
            backtests=backtests,
            candidate_results=all_candidate_results,
            selected_validation_predictions=selected_validation_predictions,
            horizons=self.config.horizons,
        )
        chapter_9_time_series = analyze_chapter_9_time_series(
            prices=normalized_prices,
            forecasts=forecasts,
            candidate_results=all_candidate_results,
            selected_validation_predictions=selected_validation_predictions,
            model_cards=model_cards,
            horizons=self.config.horizons,
            target_column=self.config.target_column,
        )
        chapter_10_bayesian_ml = analyze_chapter_10_bayesian_ml(
            forecasts=forecasts,
            candidate_results=all_candidate_results,
            selected_validation_predictions=selected_validation_predictions,
            model_cards=model_cards,
            horizons=self.config.horizons,
            enable_heavy_bayesian=self.config.enable_bayesian_heavy,
            mcmc_draws=self.config.bayesian_mcmc_draws,
            mcmc_tune=self.config.bayesian_mcmc_tune,
        )
        chapter_11_tree_models = analyze_chapter_11_tree_models(
            forecasts=forecasts,
            candidate_results=all_candidate_results,
            model_cards=model_cards,
            horizons=self.config.horizons,
            selection_metric=self.config.selection_metric,
        )
        chapter_12_boosting_models = analyze_chapter_12_boosting_models(
            forecasts=forecasts,
            candidate_results=all_candidate_results,
            model_cards=model_cards,
            horizons=self.config.horizons,
            selection_metric=self.config.selection_metric,
        )
        chapter_17_deep_learning = analyze_chapter_17_deep_learning(
            features=features,
            forecasts=forecasts,
            candidate_results=all_candidate_results,
            model_cards=model_cards,
            horizons=self.config.horizons,
            deep_learning_profile=self.config.deep_learning_profile,
            include_lstm=self.config.include_lstm,
            selection_metric=self.config.selection_metric,
        )
        chapter_18_cnn = analyze_chapter_18_cnn(
            forecasts=forecasts,
            candidate_results=all_candidate_results,
            model_cards=model_cards,
            horizons=self.config.horizons,
            deep_learning_profile=self.config.deep_learning_profile,
            selection_metric=self.config.selection_metric,
        )
        chapter_17_governance_context = analyze_chapter_17_governance_context(
            prices=normalized_prices,
            features=features,
            forecasts=forecasts,
            raw_action=raw_action,
            final_action=action,
            risk_level=risk,
            decision_diagnostics=decision_diagnostics,
            action_filters={
                "dow": dow_action_filter,
                "magee": magee_action_filter,
                "reversal": reversal_action_filter,
                "triangle": triangle_action_filter,
                "chapter_9": chapter_9_action_filter,
                "chapter_10": chapter_10_action_filter,
                "chapter_11": chapter_11_action_filter,
                "chapter_12": chapter_12_gap_filter,
                "chapter_13": chapter_13_zone_filter,
                "chapter_14": chapter_14_trendline_filter,
                "chapter_15": chapter_15_major_trendline_filter,
            },
            technical_contexts={
                "trend_state": trend_view,
                "dow_theory": dow_theory,
                "magee_basing_points": magee_basing_points,
                "reversal_patterns": reversal_patterns,
                "triangle_patterns": triangle_patterns,
                "chapter_9_patterns": chapter_9_patterns,
                "chapter_10_patterns": chapter_10_patterns,
                "chapter_11_patterns": chapter_11_patterns,
                "chapter_12_gaps": chapter_12_gaps,
                "chapter_13_support_resistance": chapter_13_support_resistance,
                "chapter_14_trendlines": chapter_14_trendlines,
                "chapter_15_major_trendlines": chapter_15_major_trendlines,
                "chapter_16_market_context": chapter_16_market_context,
            },
            backtests=backtests,
            data_quality_report=data_quality_report,
            technical_history_quality=technical_history_quality,
            market_feature_comparison=market_feature_comparison,
            security_metadata=security_metadata,
            data_manifest=data_manifest,
            target_column=self.config.target_column,
        )
        chapter_18_tactical_problem = analyze_chapter_18_tactical_problem(
            prices=normalized_prices,
            forecasts=forecasts,
            part_i_action=action,
            raw_action=raw_action,
            risk_level=risk,
            decision_diagnostics=decision_diagnostics,
            technical_contexts={
                "trend_state": trend_view,
                "dow_theory": dow_theory,
                "magee_basing_points": magee_basing_points,
                "reversal_patterns": reversal_patterns,
                "triangle_patterns": triangle_patterns,
                "chapter_9_patterns": chapter_9_patterns,
                "chapter_10_patterns": chapter_10_patterns,
                "chapter_11_patterns": chapter_11_patterns,
                "chapter_12_gaps": chapter_12_gaps,
                "chapter_13_support_resistance": chapter_13_support_resistance,
                "chapter_14_trendlines": chapter_14_trendlines,
                "chapter_15_major_trendlines": chapter_15_major_trendlines,
                "chapter_16_market_context": chapter_16_market_context,
            },
            latest_features=features.iloc[-1],
            chapter_17_llm_packet=chapter_17_governance_context.get("llm_decision_packet", {}),
            tactical_profile=self.config.tactical_profile,
            enable_llm_review=self.config.enable_llm_review,
            llm_provider=self.config.llm_provider,
            llm_model=self.config.llm_model,
            llm_temperature=self.config.llm_temperature,
            llm_reasoning_effort=self.config.llm_reasoning_effort,
            llm_timeout_seconds=self.config.llm_timeout_seconds,
            llm_env_file=self.config.llm_env_file,
            target_column=self.config.target_column,
            long_term_context=long_term_context,
        )
        final_action = chapter_18_tactical_problem["final_action"]
        final_decision_reasoning = _final_decision_reasoning(
            final_action=final_action,
            part_i_action=action,
            raw_action=raw_action,
            risk_level=risk,
            decision_diagnostics=decision_diagnostics,
            chapter_18_tactical_problem=chapter_18_tactical_problem,
            forecasts=forecasts,
            long_term_context=long_term_context,
        )
        report = {
            "ticker": self.config.ticker.upper(),
            "as_of_date": as_of_date,
            "as_of_timestamp": as_of_timestamp.isoformat(),
            "forecast_interval": self.config.forecast_interval,
            "forecast_interval_minutes": self.config.forecast_interval_minutes,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "current_price": latest_price,
            "horizons": list(self.config.horizons),
            "forecasts": forecasts,
            "part_i_suggested_action": action,
            "suggested_action": final_action,
            "final_decision_reasoning": final_decision_reasoning,
            "risk_level": risk,
            "risk_warning": _risk_warning(risk),
            "decision_view": {
                "part_i_action": action,
                "rule_based_tactical_action": chapter_18_tactical_problem.get("rule_based_action"),
                "final_governed_action": final_action,
                "chapter_18_tactical_problem": chapter_18_tactical_problem,
                "tactical_plan": chapter_18_tactical_problem.get("trade_plan", {}),
                "llm_review": chapter_18_tactical_problem.get("llm_review", {}),
                "llm_safety_gate": chapter_18_tactical_problem.get("llm_safety_gate", {}),
                "long_term_context": long_term_context or {},
                "final_decision_reasoning": final_decision_reasoning,
            },
            "portfolio_view": {
                "mark_to_market": chapter_18_tactical_problem.get("mark_to_market", {}),
                "position_context": {
                    "status": "not_supplied",
                    "reason": "Single-ticker forecast runs do not include shares, cost basis, account liquidity, or tax constraints.",
                },
            },
            "operations_view": {},
            "selection_view": {},
            "trade_risk_view": {},
            "discipline_view": {},
            "technical_view": {
                "chapter_1_ml4t_workflow": chapter_1_ml4t_workflow,
                "chapter_2_market_data": chapter_2_market_data,
                "chapter_3_alternative_data": chapter_3_alternative_data,
                "chapter_4_alpha_research": chapter_4_alpha_research,
                "chapter_5_portfolio_evaluation": chapter_5_portfolio_evaluation,
                "chapter_6_ml_process": chapter_6_ml_process,
                "chapter_7_linear_models": chapter_7_linear_models,
                "chapter_8_backtesting": chapter_8_backtesting,
                "chapter_9_time_series": chapter_9_time_series,
                "chapter_10_bayesian_ml": chapter_10_bayesian_ml,
                "chapter_11_tree_models": chapter_11_tree_models,
                "chapter_12_boosting_models": chapter_12_boosting_models,
                "chapter_17_deep_learning": chapter_17_deep_learning,
                "chapter_18_cnn": chapter_18_cnn,
                "trend_state": trend_view,
                "dow_theory": dow_theory,
                "magee_basing_points": magee_basing_points,
                "reversal_patterns": reversal_patterns,
                "triangle_patterns": triangle_patterns,
                "chapter_9_patterns": chapter_9_patterns,
                "rectangle_patterns": chapter_9_patterns.get("rectangle_patterns", {}),
                "multi_top_bottom_patterns": chapter_9_patterns.get("multi_top_bottom_patterns", {}),
                "chapter_10_patterns": chapter_10_patterns,
                "chapter_10_structural_patterns": chapter_10_patterns.get("structural_patterns", {}),
                "chapter_10_short_term_events": chapter_10_patterns.get("short_term_events", {}),
                "chapter_11_patterns": chapter_11_patterns,
                "chapter_11_continuation_patterns": chapter_11_patterns.get("continuation_patterns", {}),
                "chapter_11_head_and_shoulders_continuation": chapter_11_patterns.get("head_and_shoulders_continuation", {}),
                "chapter_12_gaps": chapter_12_gaps,
                "chapter_12_classified_gaps": chapter_12_gaps.get("classified_gaps", {}),
                "chapter_12_island_reversals": chapter_12_gaps.get("island_reversals", {}),
                "chapter_13_support_resistance": chapter_13_support_resistance,
                "chapter_13_support_zones": chapter_13_support_resistance.get("support_zones", {}),
                "chapter_13_resistance_zones": chapter_13_support_resistance.get("resistance_zones", {}),
                "chapter_14_trendlines": chapter_14_trendlines,
                "chapter_14_channels": chapter_14_trendlines.get("channels", {}),
                "chapter_14_fan_lines": chapter_14_trendlines.get("fan_lines", {}),
                "chapter_15_major_trendlines": chapter_15_major_trendlines,
                "chapter_15_scale_comparison": chapter_15_major_trendlines.get("scale_comparison", {}),
                "chapter_15_broad_market_confirmation": chapter_15_major_trendlines.get("broad_market_confirmation", {}),
                "chapter_16_market_context": chapter_16_market_context,
                "chapter_16_donchian_context": chapter_16_market_context.get("donchian_context", {}),
                "chapter_16_futures_risk_context": chapter_16_market_context.get("futures_risk_context", {}),
                "chapter_17_governance_context": chapter_17_governance_context,
                "jansen_chapter_17_deep_learning": chapter_17_deep_learning,
                "jansen_chapter_18_cnn": chapter_18_cnn,
                "chapter_17_llm_decision_packet": chapter_17_governance_context.get("llm_decision_packet", {}),
                "chapter_17_decision_fragility": chapter_17_governance_context.get("computer_humility", {}).get("decision_fragility", {}),
                "chapter_18_tactical_problem": chapter_18_tactical_problem,
                "chapter_18_tactical_plan": chapter_18_tactical_problem.get("trade_plan", {}),
                "chapter_18_llm_review": chapter_18_tactical_problem.get("llm_review", {}),
                "long_term_source_context": long_term_context or {},
                "technical_history_quality": technical_history_quality,
                "support_resistance_by_timeframe": support_resistance_by_timeframe,
                "chart_metadata": _default_chart_metadata(
                    prices=normalized_prices,
                    target_column=self.config.target_column,
                ),
                "decision_diagnostics": decision_diagnostics,
                "dow_action_filter": dow_action_filter,
                "magee_action_filter": magee_action_filter,
                "reversal_action_filter": reversal_action_filter,
                "triangle_action_filter": triangle_action_filter,
                "chapter_9_action_filter": chapter_9_action_filter,
                "chapter_10_action_filter": chapter_10_action_filter,
                "chapter_11_action_filter": chapter_11_action_filter,
                "chapter_12_gap_filter": chapter_12_gap_filter,
                "chapter_13_zone_filter": chapter_13_zone_filter,
                "chapter_14_trendline_filter": chapter_14_trendline_filter,
                "chapter_15_major_trendline_filter": chapter_15_major_trendline_filter,
                "market_only_vs_enriched": market_feature_comparison,
            },
            "selection_metric": self.config.selection_metric,
            "data_version": data_hash,
            "data_manifest": data_manifest,
            "model_version": self.config.model_version,
            "candidate_results": all_candidate_results,
            "backtests": backtests,
            "governance": {
                "config": self.config.to_dict(),
                "feature_columns": list(features.columns),
                "feature_registry_summary": {
                    "feature_count": feature_registry["feature_count"],
                    "family_counts": feature_registry["family_counts"],
                    "source_counts": feature_registry["source_counts"],
                },
                "feature_registry": feature_registry["entries"],
                "technical_method_card": dow_theory.get("technical_method_card", {}),
                "technical_method_cards": {
                    "chapter_1_ml4t_workflow": chapter_1_ml4t_workflow.get("technical_method_card", {}),
                    "chapter_2_market_data": chapter_2_market_data.get("technical_method_card", {}),
                    "chapter_3_alternative_data": chapter_3_alternative_data.get("technical_method_card", {}),
                    "chapter_4_alpha_research": chapter_4_alpha_research.get("technical_method_card", {}),
                    "chapter_5_portfolio_evaluation": chapter_5_portfolio_evaluation.get("technical_method_card", {}),
                    "chapter_6_ml_process": chapter_6_ml_process.get("technical_method_card", {}),
                    "chapter_7_linear_models": chapter_7_linear_models.get("technical_method_card", {}),
                    "chapter_8_backtesting": chapter_8_backtesting.get("technical_method_card", {}),
                    "chapter_9_time_series": chapter_9_time_series.get("technical_method_card", {}),
                    "chapter_10_bayesian_ml": chapter_10_bayesian_ml.get("technical_method_card", {}),
                    "chapter_11_tree_models": chapter_11_tree_models.get("technical_method_card", {}),
                    "chapter_12_boosting_models": chapter_12_boosting_models.get("technical_method_card", {}),
                    "jansen_chapter_17_deep_learning": chapter_17_deep_learning.get("technical_method_card", {}),
                    "jansen_chapter_18_cnn": chapter_18_cnn.get("technical_method_card", {}),
                    "dow_theory": dow_theory.get("technical_method_card", {}),
                    "magee_basing_points": magee_basing_points.get("technical_method_card", {}),
                    "reversal_patterns": reversal_patterns.get("technical_method_card", {}),
                    "triangle_patterns": triangle_patterns.get("technical_method_card", {}),
                    "chapter_9_patterns": chapter_9_patterns.get("technical_method_card", {}),
                    "chapter_10_patterns": chapter_10_patterns.get("technical_method_card", {}),
                    "chapter_11_patterns": chapter_11_patterns.get("technical_method_card", {}),
                    "chapter_12_gaps": chapter_12_gaps.get("technical_method_card", {}),
                    "chapter_13_support_resistance": chapter_13_support_resistance.get("technical_method_card", {}),
                    "chapter_14_trendlines": chapter_14_trendlines.get("technical_method_card", {}),
                    "chapter_15_major_trendlines": chapter_15_major_trendlines.get("technical_method_card", {}),
                    "chapter_16_market_context": chapter_16_market_context.get("technical_method_card", {}),
                    "chapter_17_governance_context": chapter_17_governance_context.get("technical_method_card", {}),
                    "long_term_source_context": (long_term_context or {}).get("technical_method_card", {}),
                },
                "tactical_method_cards": {
                    "chapter_18_tactical_problem": chapter_18_tactical_problem.get("technical_method_card", {}),
                },
                "operational_method_cards": {},
                "selection_method_cards": {},
                "trade_risk_method_cards": {},
                "portfolio_method_cards": {},
                "discipline_method_cards": {},
                "model_cards": model_cards,
                "security_metadata": security_metadata,
                "long_term_source_context": long_term_context or {},
            },
            "diagnostics": {
                "chapter_1_ml4t_workflow": chapter_1_ml4t_workflow,
                "chapter_2_market_data": chapter_2_market_data,
                "chapter_3_alternative_data": chapter_3_alternative_data,
                "chapter_4_alpha_research": chapter_4_alpha_research,
                "chapter_5_portfolio_evaluation": chapter_5_portfolio_evaluation,
                "chapter_6_ml_process": chapter_6_ml_process,
                "chapter_7_linear_models": chapter_7_linear_models,
                "chapter_8_backtesting": chapter_8_backtesting,
                "chapter_9_time_series": chapter_9_time_series,
                "chapter_10_bayesian_ml": chapter_10_bayesian_ml,
                "chapter_11_tree_models": chapter_11_tree_models,
                "chapter_12_boosting_models": chapter_12_boosting_models,
                "chapter_17_deep_learning": chapter_17_deep_learning,
                "chapter_18_cnn": chapter_18_cnn,
                "selected_validation_predictions": selected_validation_predictions,
                "factor_evaluation": factor_evaluation,
                "technical_structure": structure_snapshot,
                "dow_theory": dow_theory,
                "dow_chapter_4_defects": dow_theory.get("chapter_4_defect_diagnostics", {}),
                "magee_basing_points": magee_basing_points,
                "reversal_patterns": reversal_patterns,
                "triangle_patterns": triangle_patterns,
                "chapter_9_patterns": chapter_9_patterns,
                "rectangle_patterns": chapter_9_patterns.get("rectangle_patterns", {}),
                "multi_top_bottom_patterns": chapter_9_patterns.get("multi_top_bottom_patterns", {}),
                "chapter_10_patterns": chapter_10_patterns,
                "chapter_10_structural_patterns": chapter_10_patterns.get("structural_patterns", {}),
                "chapter_10_short_term_events": chapter_10_patterns.get("short_term_events", {}),
                "chapter_11_patterns": chapter_11_patterns,
                "chapter_11_continuation_patterns": chapter_11_patterns.get("continuation_patterns", {}),
                "chapter_11_head_and_shoulders_continuation": chapter_11_patterns.get("head_and_shoulders_continuation", {}),
                "chapter_12_gaps": chapter_12_gaps,
                "chapter_12_classified_gaps": chapter_12_gaps.get("classified_gaps", {}),
                "chapter_12_island_reversals": chapter_12_gaps.get("island_reversals", {}),
                "chapter_13_support_resistance": chapter_13_support_resistance,
                "chapter_13_support_zones": chapter_13_support_resistance.get("support_zones", {}),
                "chapter_13_resistance_zones": chapter_13_support_resistance.get("resistance_zones", {}),
                "chapter_14_trendlines": chapter_14_trendlines,
                "chapter_14_channels": chapter_14_trendlines.get("channels", {}),
                "chapter_14_fan_lines": chapter_14_trendlines.get("fan_lines", {}),
                "chapter_15_major_trendlines": chapter_15_major_trendlines,
                "chapter_15_scale_comparison": chapter_15_major_trendlines.get("scale_comparison", {}),
                "chapter_15_broad_market_confirmation": chapter_15_major_trendlines.get("broad_market_confirmation", {}),
                "chapter_16_market_context": chapter_16_market_context,
                "chapter_16_donchian_context": chapter_16_market_context.get("donchian_context", {}),
                "chapter_16_futures_risk_context": chapter_16_market_context.get("futures_risk_context", {}),
                "chapter_17_governance_context": chapter_17_governance_context,
                "jansen_chapter_17_deep_learning": chapter_17_deep_learning,
                "jansen_chapter_18_cnn": chapter_18_cnn,
                "chapter_17_llm_decision_packet": chapter_17_governance_context.get("llm_decision_packet", {}),
                "chapter_17_decision_fragility": chapter_17_governance_context.get("computer_humility", {}).get("decision_fragility", {}),
                "chapter_18_tactical_problem": chapter_18_tactical_problem,
                "chapter_18_tactical_plan": chapter_18_tactical_problem.get("trade_plan", {}),
                "chapter_18_llm_review": chapter_18_tactical_problem.get("llm_review", {}),
                "long_term_source_context": long_term_context or {},
                "data_quality": data_quality_report,
                "decision_diagnostics": decision_diagnostics,
                "dow_action_filter": dow_action_filter,
                "magee_action_filter": magee_action_filter,
                "reversal_action_filter": reversal_action_filter,
                "triangle_action_filter": triangle_action_filter,
                "chapter_9_action_filter": chapter_9_action_filter,
                "chapter_10_action_filter": chapter_10_action_filter,
                "chapter_11_action_filter": chapter_11_action_filter,
                "chapter_12_gap_filter": chapter_12_gap_filter,
                "chapter_13_zone_filter": chapter_13_zone_filter,
                "chapter_14_trendline_filter": chapter_14_trendline_filter,
                "chapter_15_major_trendline_filter": chapter_15_major_trendline_filter,
                "market_only_vs_enriched": market_feature_comparison,
                "validation_design": {
                    "purge_window": self.config.purge_window,
                    "embargo_window": self.config.embargo_window,
                    "final_holdout_fraction": self.config.final_holdout_fraction,
                    "multiple_testing_adjustment": "deflated_sharpe_ratio_approximation",
                    "tuning_mode": self.config.tuning_mode,
                    "optuna_trials": self.config.optuna_trials if self.config.tuning_mode == "optuna" else 0,
                    "optuna_timeout_seconds": self.config.optuna_timeout_seconds,
                    "optuna_inner_splits": self.config.optuna_inner_splits if self.config.tuning_mode == "optuna" else 0,
                    "optuna_families": list(self.config.optuna_families) if self.config.tuning_mode == "optuna" else [],
                    "optuna_holdout_policy": "Optuna tuning is nested inside candidate fit slices; final holdout is not used for tuning or selection.",
                },
            },
        }
        apply_chapter_19_validation(
            report,
            prices=normalized_prices,
            target_column=self.config.target_column,
        )
        apply_chapter_20_ticker_suitability(
            report,
            prices=normalized_prices,
            target_column=self.config.target_column,
        )
        apply_chapter_21_chart_selection(report)
        annotate_mean_reversion_dip_buy(
            report,
            prices=normalized_prices,
            target_column=self.config.target_column,
        )
        apply_chapter_23_30_trade_risk_plan(
            report,
            prices=normalized_prices,
            target_column=self.config.target_column,
        )
        apply_chapter_31_42_portfolio_capital_risk(
            report,
            prices=normalized_prices,
            target_column=self.config.target_column,
        )
        apply_chapter_39_43_discipline_governance(report)
        return report

    def _forecast_horizon(
        self,
        horizon: int,
        supervised: pd.DataFrame,
        latest_features: pd.DataFrame,
        latest_price: float,
        price_series: pd.Series,
        data_hash: str,
        factor_evaluation: dict[str, list[dict[str, Any]]],
        data_manifest: dict[str, Any],
    ) -> tuple[HorizonForecast, list[tuple[Any, Any, pd.DataFrame]], dict[str, Any], list[dict[str, Any]]]:
        target_column = f"target_log_return_{horizon}d"
        training_frame = supervised.dropna(subset=[target_column])
        feature_columns = [column for column in training_frame.columns if not column.startswith("target_")]
        feature_selection_policy = _ml4t_feature_selection_policy(
            feature_columns=feature_columns,
            factor_rows=factor_evaluation.get(str(horizon), []),
            data_manifest=data_manifest,
        )
        excluded_features = set(feature_selection_policy.get("excluded_features", []))
        feature_columns = [column for column in feature_columns if str(column) not in excluded_features]
        training_frame = training_frame[feature_columns + [target_column]]

        if len(training_frame) < 30:
            raise ValueError(f"Horizon {horizon}d has only {len(training_frame)} supervised rows.")

        features = training_frame[feature_columns]
        target = training_frame[target_column]
        candidates = default_candidates(
            random_state=self.config.random_state,
            include_lightgbm=self.config.include_lightgbm,
            include_statistical_models=self.config.include_statistical_models,
            include_lstm=self.config.include_lstm,
            deep_learning_profile=self.config.deep_learning_profile,
            search_level=self.config.search_level,
            tuning_mode=self.config.tuning_mode,
            optuna_trials=self.config.optuna_trials,
            optuna_timeout_seconds=self.config.optuna_timeout_seconds,
            optuna_inner_splits=self.config.optuna_inner_splits,
            optuna_families=self.config.optuna_families,
        )
        validation_results = validate_candidates(
            candidates=candidates,
            features=features,
            target=target,
            horizon_days=horizon,
            min_training_rows=self.config.min_training_rows,
            validation_window=self.config.validation_window,
            step_size=self.config.step_size,
            max_splits=self.config.max_splits,
            purge_window=horizon if self.config.purge_window is None else self.config.purge_window,
            embargo_window=self.config.embargo_window,
            final_holdout_fraction=self.config.final_holdout_fraction,
            validation_workers=self.config.validation_workers,
            progress_callback=self.progress_callback,
        )
        ml4t_selection_adjustment = apply_ml4t_selection_adjustments(
            validation_results,
            target=target,
            selection_metric=self.config.selection_metric,
        )
        selected_candidate, selected_summary, selected_predictions = select_candidate(
            validation_results,
            selection_metric=self.config.selection_metric,
        )

        fitted_model = selected_candidate.clone().fit(features, target)
        selected_log_return = float(fitted_model.predict(latest_features[feature_columns])[0])
        ensemble = _ensemble_forecast(
            validation_results=validation_results,
            features=features,
            target=target,
            latest_features=latest_features[feature_columns],
            feature_columns=feature_columns,
            selection_metric=self.config.selection_metric,
        )
        expected_log_return = float(ensemble.get("expected_log_return", selected_log_return))
        expected_return = float(np.expm1(expected_log_return))
        trade_quality = _predict_trade_quality(
            selected_candidate=selected_candidate,
            features=features,
            supervised=supervised,
            feature_columns=feature_columns,
            latest_features=latest_features,
            horizon=horizon,
        )

        residual_std = float(ensemble.get("residual_std", selected_summary.metrics.get("residual_std", 0.0)))
        interval = _empirical_forecast_interval(
            expected_log_return=expected_log_return,
            validation_predictions=ensemble.get("validation_predictions", selected_predictions),
            confidence_level=self.config.confidence_level,
            residual_std=residual_std,
        )
        lower_log_return = interval["lower_log_return"]
        upper_log_return = interval["upper_log_return"]
        predicted_price = float(latest_price * np.exp(expected_log_return))
        lower_price = float(latest_price * np.exp(lower_log_return))
        upper_price = float(latest_price * np.exp(upper_log_return))
        confidence = _empirical_directional_confidence(
            expected_log_return=expected_log_return,
            validation_predictions=ensemble.get("validation_predictions", selected_predictions),
            validation_metrics=selected_summary.metrics,
        )
        confidence *= float(selected_summary.metrics.get("chapter_10_confidence_multiplier", 1.0))
        confidence = float(min(0.99, max(0.50, confidence)))
        direction = _direction_label(expected_return, selected_summary.metrics.get("mae", 0.0))
        forecast_date = _forecast_timestamp(
            pd.Timestamp(latest_features.index[-1]),
            horizon,
            interval_minutes=self.config.forecast_interval_minutes,
            index=price_series.index,
        )
        model_diagnostics = fitted_model.model_diagnostics()

        forecast = HorizonForecast(
            horizon_days=horizon,
            forecast_date=_format_forecast_timestamp(forecast_date, self.config.forecast_interval_minutes),
            selected_model=selected_summary.model_name,
            selected_model_family=selected_summary.model_family,
            selection_metric=self.config.selection_metric,
            expected_log_return=expected_log_return,
            expected_return=expected_return,
            predicted_price=predicted_price,
            lower_price=lower_price,
            upper_price=upper_price,
            expected_direction=direction,
            directional_confidence=confidence,
            confidence_interval_method=interval["method"],
            calibration_sample_size=interval["sample_size"],
            trade_quality=trade_quality,
            validation_metrics=selected_summary.metrics,
        )

        model_card = build_model_card(
            ticker=self.config.ticker.upper(),
            horizon_days=horizon,
            selected_model=selected_summary.model_name,
            selected_model_family=selected_summary.model_family,
            model_parameters=selected_summary.model_parameters,
            training_window={
                "start_date": str(features.index[0].date()),
                "end_date": str(features.index[-1].date()),
                "rows": int(len(features)),
            },
            feature_columns=feature_columns,
            selection_metric=self.config.selection_metric,
            validation_metrics=selected_summary.metrics,
            confidence_interval={
                "level": self.config.confidence_level,
                "lower_price": lower_price,
                "upper_price": upper_price,
                "residual_std_log_return": residual_std,
                "method": interval["method"],
                "lower_residual_quantile": interval["lower_residual_quantile"],
                "upper_residual_quantile": interval["upper_residual_quantile"],
                "sample_size": interval["sample_size"],
            },
            data_version=data_hash,
            model_version=self.config.model_version,
        )
        if model_diagnostics:
            model_card["model_diagnostics"] = model_diagnostics
        if ensemble:
            model_card["ensemble"] = {key: value for key, value in ensemble.items() if key != "validation_predictions"}
        model_card["ml4t_feature_selection_policy"] = feature_selection_policy
        model_card["ml4t_selection_adjustment"] = ml4t_selection_adjustment
        model_card["chapter_9_time_series_selection_adjustment"] = ml4t_selection_adjustment
        model_card["trade_quality"] = trade_quality
        model_card["selected_validation_predictions"] = {
            "rows": int(len(selected_predictions)),
            "start_date": str(selected_predictions.index[0].date()),
            "end_date": str(selected_predictions.index[-1].date()),
        }

        validation_records = _validation_prediction_records(selected_predictions, price_series)
        return forecast, validation_results, model_card, validation_records

    def _market_only_feature_comparison(
        self,
        normalized_prices: pd.DataFrame,
        full_forecasts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        external_columns = [
            column
            for column in normalized_prices.columns
            if str(column).lower() not in MARKET_DATA_COLUMNS and str(column).lower() != self.config.target_column
        ]
        if not external_columns:
            return {
                "status": "not_applicable",
                "reason": "No external, relative, macro, event, or panel features were supplied.",
                "external_columns": [],
            }

        market_columns = [
            column for column in normalized_prices.columns if str(column).lower() in MARKET_DATA_COLUMNS
        ]
        market_prices = normalize_price_frame(
            normalized_prices[market_columns],
            target_column=self.config.target_column,
        )
        market_features = build_feature_frame(market_prices, target_column=self.config.target_column)
        market_supervised = add_forward_return_targets(
            features=market_features,
            prices=market_prices,
            horizons=self.config.horizons,
            target_column=self.config.target_column,
        )

        comparisons = []
        full_by_horizon = {int(item["horizon_days"]): item for item in full_forecasts}
        for horizon in self.config.horizons:
            try:
                market_summary = self._select_market_only_summary(market_supervised, horizon=horizon)
            except Exception as exc:
                comparisons.append(
                    {
                        "horizon_days": int(horizon),
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                continue
            full_forecast = full_by_horizon[int(horizon)]
            metric = self.config.selection_metric
            full_metric = float(full_forecast["validation_metrics"].get(metric, np.nan))
            market_metric = float(market_summary.metrics.get(metric, np.nan))
            full_better = _metric_is_better(full_metric, market_metric, metric)
            comparisons.append(
                {
                    "horizon_days": int(horizon),
                    "status": "compared",
                    "selection_metric": metric,
                    "enriched_model": full_forecast["selected_model"],
                    "market_only_model": market_summary.model_name,
                    "enriched_metric": full_metric,
                    "market_only_metric": market_metric,
                    "enriched_features_helped": full_better,
                    "metric_delta_enriched_minus_market": float(full_metric - market_metric),
                }
            )

        helped = [item.get("enriched_features_helped") for item in comparisons if item.get("status") == "compared"]
        return {
            "status": "compared",
            "external_columns": [str(column) for column in external_columns],
            "principle": "External/context features must beat the market-only feature set in walk-forward validation to justify their use.",
            "enriched_features_helped_any_horizon": bool(any(helped)),
            "enriched_features_helped_all_horizons": bool(helped and all(helped)),
            "horizons": comparisons,
        }

    def _select_market_only_summary(self, supervised: pd.DataFrame, horizon: int) -> Any:
        target_column = f"target_log_return_{horizon}d"
        training_frame = supervised.dropna(subset=[target_column])
        feature_columns = [column for column in training_frame.columns if not column.startswith("target_")]
        training_frame = training_frame[feature_columns + [target_column]]
        if len(training_frame) < 30:
            raise ValueError(f"Horizon {horizon}d has only {len(training_frame)} market-only rows.")
        candidates = default_candidates(
            random_state=self.config.random_state,
            include_lightgbm=self.config.include_lightgbm,
            include_statistical_models=self.config.include_statistical_models,
            include_lstm=self.config.include_lstm,
            deep_learning_profile=self.config.deep_learning_profile,
            search_level=self.config.search_level,
        )
        validation_results = validate_candidates(
            candidates=candidates,
            features=training_frame[feature_columns],
            target=training_frame[target_column],
            horizon_days=horizon,
            min_training_rows=self.config.min_training_rows,
            validation_window=self.config.validation_window,
            step_size=self.config.step_size,
            max_splits=self.config.max_splits,
            purge_window=horizon if self.config.purge_window is None else self.config.purge_window,
            embargo_window=self.config.embargo_window,
            final_holdout_fraction=self.config.final_holdout_fraction,
            validation_workers=self.config.validation_workers,
            progress_callback=self.progress_callback,
        )
        _, selected_summary, _ = select_candidate(validation_results, selection_metric=self.config.selection_metric)
        return selected_summary


def _ml4t_feature_selection_policy(
    *,
    feature_columns: list[str],
    factor_rows: list[dict[str, Any]],
    data_manifest: dict[str, Any],
) -> dict[str, Any]:
    available = {str(column) for column in feature_columns}
    exclusions: dict[str, str] = {}
    for row in factor_rows:
        feature = str(row.get("feature", ""))
        if feature not in available:
            continue
        rank_ic = _safe_float(row.get("rank_ic"))
        quantile_spread = _safe_float(row.get("quantile_spread_return"))
        rows = int(row.get("rows", 0) or 0)
        if rows >= 80 and abs(rank_ic) < 0.005 and abs(quantile_spread) < 0.001:
            exclusions[feature] = "chapter_4_low_alpha_evidence"

    unsafe_alt_data = _alternative_data_is_unsafe(data_manifest)
    if unsafe_alt_data:
        for feature in available:
            lowered = feature.lower()
            if lowered.startswith("alt_") or lowered.startswith("exo_alt_") or "sentiment" in lowered or "news" in lowered:
                exclusions.setdefault(feature, "chapter_3_alternative_data_quality")

    max_exclusions = max(0, len(feature_columns) - 10)
    ordered_exclusions = sorted(exclusions)[:max_exclusions]
    return {
        "status": "applied" if ordered_exclusions else "no_exclusions",
        "excluded_features": ordered_exclusions,
        "excluded_feature_count": int(len(ordered_exclusions)),
        "input_feature_count": int(len(feature_columns)),
        "output_feature_count": int(len(feature_columns) - len(ordered_exclusions)),
        "reasons": {feature: exclusions[feature] for feature in ordered_exclusions},
        "model_fitting_consequence": "Excluded weak alpha or unsafe alternative-data features before candidate validation.",
    }


def _alternative_data_is_unsafe(data_manifest: dict[str, Any]) -> bool:
    for source in data_manifest.get("alternative_sources", []) or []:
        if not isinstance(source, dict):
            continue
        registry = source.get("registry") if isinstance(source.get("registry"), dict) else source
        if registry.get("point_in_time_safe") is False:
            return True
        if registry.get("provider_status") not in {None, "ok"}:
            return True
    return False


def _safe_float(value: Any) -> float:
    try:
        number = float(value if value is not None else 0.0)
    except Exception:
        return 0.0
    return number if np.isfinite(number) else 0.0


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return number if np.isfinite(number) else None


def _final_decision_reasoning(
    *,
    final_action: str,
    part_i_action: str,
    raw_action: str,
    risk_level: str,
    decision_diagnostics: dict[str, Any],
    chapter_18_tactical_problem: dict[str, Any],
    forecasts: list[dict[str, Any]],
    long_term_context: dict[str, Any] | None,
) -> dict[str, Any]:
    llm_review = chapter_18_tactical_problem.get("llm_review", {}) or {}
    safety_gate = chapter_18_tactical_problem.get("llm_safety_gate", {}) or {}
    preferred_forecast = chapter_18_tactical_problem.get("preferred_forecast") or (forecasts[0] if forecasts else {})
    raw_json = llm_review.get("raw_json") if isinstance(llm_review.get("raw_json"), dict) else {}
    rationale = llm_review.get("rationale") or raw_json.get("rationale")
    final_advice = raw_json.get("final_advice") if isinstance(raw_json.get("final_advice"), dict) else None
    if final_advice is None:
        final_advice = _fallback_final_advice(chapter_18_tactical_problem)
    risk_notes = llm_review.get("risk_notes") or raw_json.get("risk_notes") or []
    if not isinstance(risk_notes, list):
        risk_notes = [str(risk_notes)]

    long_term = long_term_context or {}
    provider_summaries = long_term.get("provider_summaries", {}) if isinstance(long_term.get("provider_summaries"), dict) else {}
    decision_relevance = long_term.get("decision_relevance", {}) if isinstance(long_term.get("decision_relevance"), dict) else {}
    conflicts = long_term.get("conflicts", []) if isinstance(long_term.get("conflicts"), list) else []
    stale_fields = long_term.get("stale_fields", []) if isinstance(long_term.get("stale_fields"), list) else []

    return {
        "final_action": final_action,
        "decision_source": "llm_review_with_safety_gate" if llm_review.get("status") == "executed" else "rule_based_tactical_gate",
        "llm_status": llm_review.get("status"),
        "llm_model": llm_review.get("model"),
        "llm_recommended_action": llm_review.get("recommended_action"),
        "llm_rationale": rationale,
        "final_advice": final_advice,
        "llm_risk_notes": risk_notes[:8],
        "safety_gate_status": safety_gate.get("status"),
        "safety_gate_reason": safety_gate.get("reason"),
        "rule_based_action": chapter_18_tactical_problem.get("rule_based_action"),
        "part_i_action": part_i_action,
        "raw_model_action": raw_action,
        "risk_level": risk_level,
        "hold_reason": decision_diagnostics.get("hold_reason"),
        "blocking_reasons": list(decision_diagnostics.get("blocking_reasons", []) or [])[:12],
        "supporting_reasons": list(decision_diagnostics.get("supporting_reasons", []) or [])[:12],
        "preferred_forecast": {
            "horizon_days": preferred_forecast.get("horizon_days"),
            "expected_direction": preferred_forecast.get("expected_direction"),
            "expected_return": preferred_forecast.get("expected_return"),
            "directional_confidence": preferred_forecast.get("directional_confidence"),
            "predicted_price": preferred_forecast.get("predicted_price"),
        },
        "long_term_sources": {
            "status": long_term.get("status"),
            "providers_requested": long_term.get("providers_requested", []),
            "provider_status": {
                provider: summary.get("status")
                for provider, summary in provider_summaries.items()
                if isinstance(summary, dict)
            },
            "decision_relevance": decision_relevance,
            "conflict_count": len(conflicts),
            "stale_field_count": len(stale_fields),
            "summary": long_term.get("summary") or long_term.get("consolidated_summary"),
        },
        "audit_note": (
            "Final action is determined by the Chapter 18 tactical safety gate. "
            "Executed LLM review may downgrade or support the rule-based action, but cannot bypass hard blockers."
        ),
    }


def _fallback_final_advice(chapter_18_tactical_problem: dict[str, Any]) -> dict[str, Any]:
    trade_plan = chapter_18_tactical_problem.get("trade_plan", {}) or {}
    action = str(chapter_18_tactical_problem.get("final_action") or trade_plan.get("rule_based_action") or "Hold")
    current_price = _finite_or_none(trade_plan.get("current_price"))
    stop_plan = trade_plan.get("stop_plan", {}) if isinstance(trade_plan.get("stop_plan"), dict) else {}
    target_plan = trade_plan.get("target_plan", {}) if isinstance(trade_plan.get("target_plan"), dict) else {}
    stop = _finite_or_none(stop_plan.get("level"))
    target = _finite_or_none(target_plan.get("level"))
    if action == "Buy":
        headline = f"Buy only if execution is near the current tactical level; target {target}, stop {stop}."
        action_now = "buy_now"
        why_not_buy_now = ""
        why_not_sell_now = "The governed action is Buy, not Sell."
    elif action == "Sell":
        headline = f"Sell or reduce exposure near the current tactical level; downside objective {target}, invalidation {stop}."
        action_now = "sell_now"
        why_not_buy_now = "The governed action is Sell, so adding exposure is blocked."
        why_not_sell_now = ""
    else:
        headline = "Hold. Wait for a better risk/reward setup before adding or exiting."
        action_now = "hold"
        why_not_buy_now = "The governed tactical action is Hold; no active buy commitment passed the gate."
        why_not_sell_now = "The governed tactical action is Hold; no active sell commitment passed the gate."
    return {
        "headline": headline,
        "action_now": action_now,
        "buy_now_price": current_price if action == "Buy" else None,
        "buy_lower_price": None,
        "buy_above_breakout_price": None,
        "sell_or_trim_price": current_price if action == "Sell" else None,
        "take_profit_price": target if action == "Buy" else None,
        "stop_loss_price": stop,
        "invalidation_price": stop,
        "expected_base_case": trade_plan.get("entry_policy") or "Follow the governed tactical action until a fresh forecast changes the setup.",
        "why_not_buy_now": why_not_buy_now,
        "why_not_sell_now": why_not_sell_now,
    }


def run_forecast(
    prices: pd.DataFrame,
    config: ForecastConfig,
    data_manifest: dict[str, Any] | None = None,
    data_quality_report: dict[str, Any] | None = None,
    security_metadata: dict[str, Any] | None = None,
    long_term_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return ForecastingEngine(config).run(
        prices,
        data_manifest=data_manifest,
        data_quality_report=data_quality_report,
        security_metadata=security_metadata,
        long_term_context=long_term_context,
    )


def _forecast_timestamp(
    as_of_date: pd.Timestamp,
    horizon: int,
    interval_minutes: float | None = None,
    index: pd.DatetimeIndex | None = None,
) -> pd.Timestamp:
    if interval_minutes is not None and interval_minutes < 18 * 60:
        if index is not None:
            return add_trading_bars(pd.DatetimeIndex(index), as_of_date, horizon, interval_minutes)
        return add_trading_minutes(as_of_date, float(interval_minutes) * horizon)
    return pd.bdate_range(as_of_date.normalize(), periods=horizon + 1)[-1]


def _format_forecast_timestamp(forecast_timestamp: pd.Timestamp, interval_minutes: float | None = None) -> str:
    if interval_minutes is not None and interval_minutes < 18 * 60:
        return forecast_timestamp.isoformat()
    return str(forecast_timestamp.date())


def _predict_trade_quality(
    selected_candidate: Any,
    features: pd.DataFrame,
    supervised: pd.DataFrame,
    feature_columns: list[str],
    latest_features: pd.DataFrame,
    horizon: int,
) -> dict[str, float]:
    targets = {
        "upside_breakout_probability": f"target_upside_breakout_{horizon}d",
        "downside_breakdown_probability": f"target_downside_breakdown_{horizon}d",
        "reward_to_risk_score": f"target_reward_to_risk_{horizon}d",
        "direction_probability": f"target_direction_{horizon}d",
    }
    output: dict[str, float] = {}
    for output_name, target_name in targets.items():
        if target_name not in supervised.columns:
            continue
        target = supervised[target_name].replace([np.inf, -np.inf], np.nan).dropna()
        aligned_features = features.reindex(target.index)[feature_columns]
        valid = aligned_features.notna().any(axis=1)
        aligned_features = aligned_features.loc[valid]
        target = target.loc[valid]
        if len(target) < 40 or target.nunique(dropna=True) < 2:
            continue
        try:
            model = selected_candidate.clone().fit(aligned_features, target)
            prediction = float(model.predict(latest_features[feature_columns])[0])
        except Exception:
            continue
        if output_name.endswith("probability"):
            prediction = min(1.0, max(0.0, prediction))
        output[output_name] = float(prediction)
    return output


def _ensemble_forecast(
    *,
    validation_results: list[tuple[Any, Any, pd.DataFrame]],
    features: pd.DataFrame,
    target: pd.Series,
    latest_features: pd.DataFrame,
    feature_columns: list[str],
    selection_metric: str,
) -> dict[str, Any]:
    if not validation_results:
        return {}
    rows = []
    for candidate, summary, predictions in validation_results:
        metrics = summary.metrics
        mae_value = float(metrics.get("mae", np.inf))
        holdout_mae = float(metrics.get("holdout_mae", mae_value))
        directional_accuracy = float(metrics.get("directional_accuracy", 0.0))
        holdout_directional_accuracy = float(metrics.get("holdout_directional_accuracy", directional_accuracy))
        if not np.isfinite(mae_value) or mae_value <= 0:
            continue
        if holdout_directional_accuracy < 0.42 or directional_accuracy < 0.42:
            continue
        if holdout_mae > max(0.018, mae_value * 2.5):
            continue
        try:
            fitted = candidate.clone().fit(features, target)
            prediction = float(fitted.predict(latest_features)[0])
        except Exception:
            continue
        if not np.isfinite(prediction):
            continue
        score = 1.0 / max(mae_value, holdout_mae, 1e-6)
        if selection_metric in HIGHER_IS_BETTER_METRICS:
            score *= max(0.05, float(metrics.get(selection_metric, directional_accuracy)))
        rows.append(
            {
                "candidate": candidate,
                "summary": summary,
                "predictions": predictions,
                "prediction": prediction,
                "weight_score": float(score),
            }
        )

    if len(rows) < 2:
        return {}
    total = sum(row["weight_score"] for row in rows)
    if total <= 0:
        return {}
    weights = [row["weight_score"] / total for row in rows]
    expected_log_return = float(sum(weight * row["prediction"] for weight, row in zip(weights, rows)))
    validation_predictions = _weighted_validation_predictions(rows, weights)
    residuals = validation_predictions["actual"] - validation_predictions["predicted"]
    residual_std = float(residuals.std()) if len(residuals.dropna()) > 1 else 0.0
    members = [
        {
            "model": row["summary"].model_name,
            "family": row["summary"].model_family,
            "weight": float(weight),
            "mae": float(row["summary"].metrics.get("mae", np.nan)),
            "holdout_mae": float(row["summary"].metrics.get("holdout_mae", np.nan)),
            "directional_accuracy": float(row["summary"].metrics.get("directional_accuracy", np.nan)),
        }
        for weight, row in zip(weights, rows)
    ]
    return {
        "method": "validation_weighted_ensemble",
        "expected_log_return": expected_log_return,
        "member_count": int(len(members)),
        "members": members,
        "residual_std": residual_std,
        "validation_predictions": validation_predictions,
    }


def _weighted_validation_predictions(rows: list[dict[str, Any]], weights: list[float]) -> pd.DataFrame:
    prediction_frames = []
    for row, weight in zip(rows, weights):
        predictions = row["predictions"][["actual", "predicted", "split_train_end"]].copy()
        predictions["weighted_predicted"] = predictions["predicted"] * weight
        prediction_frames.append(predictions)
    stacked = pd.concat(prediction_frames, keys=range(len(prediction_frames)), names=["member", "timestamp"])
    grouped = stacked.groupby(level="timestamp")
    output = pd.DataFrame(
        {
            "actual": grouped["actual"].mean(),
            "predicted": grouped["weighted_predicted"].sum(),
            "split_train_end": grouped["split_train_end"].last(),
        }
    )
    return output.sort_index()


def _empirical_forecast_interval(
    expected_log_return: float,
    validation_predictions: pd.DataFrame,
    confidence_level: float,
    residual_std: float,
) -> dict[str, Any]:
    residuals = (validation_predictions["actual"] - validation_predictions["predicted"]).dropna().to_numpy(dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) >= 10:
        alpha = max(0.0, min(1.0, 1.0 - confidence_level))
        lower_residual = float(np.quantile(residuals, alpha / 2))
        upper_residual = float(np.quantile(residuals, 1 - alpha / 2))
        method = "empirical_validation_residual_quantile"
    else:
        z = 1.2815515655446004 if confidence_level <= 0.80 else 1.6448536269514722
        lower_residual = float(-z * residual_std)
        upper_residual = float(z * residual_std)
        method = "normal_residual_fallback"

    lower_residual = min(lower_residual, 0.0)
    upper_residual = max(upper_residual, 0.0)
    lower_log_return = expected_log_return + lower_residual
    upper_log_return = expected_log_return + upper_residual
    if lower_log_return > upper_log_return:
        lower_log_return, upper_log_return = upper_log_return, lower_log_return

    return {
        "lower_log_return": float(lower_log_return),
        "upper_log_return": float(upper_log_return),
        "lower_residual_quantile": float(lower_residual),
        "upper_residual_quantile": float(upper_residual),
        "sample_size": int(len(residuals)),
        "method": method,
    }


def _empirical_directional_confidence(
    expected_log_return: float,
    validation_predictions: pd.DataFrame,
    validation_metrics: dict[str, float],
) -> float:
    expected_sign = np.sign(expected_log_return)
    if expected_sign == 0:
        return 0.50

    frame = validation_predictions[["actual", "predicted"]].dropna()
    frame = frame[np.isfinite(frame["actual"]) & np.isfinite(frame["predicted"])]
    same_signal = frame[np.sign(frame["predicted"]) == expected_sign]
    if len(same_signal) >= 10:
        confidence = float((np.sign(same_signal["actual"]) == expected_sign).mean())
    elif len(frame) >= 10:
        confidence = float(validation_metrics.get("directional_accuracy", 0.50))
    else:
        confidence = 0.50
    return float(min(0.95, max(0.50, confidence)))


def _validation_prediction_records(predictions: pd.DataFrame, price_series: pd.Series) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    base_prices = price_series.reindex(predictions.index)
    for date, row in predictions.iterrows():
        base_price = float(base_prices.loc[date])
        actual_log_return = float(row["actual"])
        predicted_log_return = float(row["predicted"])
        records.append(
            {
                "validation_date": str(pd.Timestamp(date).date()),
                "split_train_end": str(row["split_train_end"]),
                "base_price": base_price,
                "actual_log_return": actual_log_return,
                "predicted_log_return": predicted_log_return,
                "actual_future_price": float(base_price * np.exp(actual_log_return)),
                "predicted_future_price": float(base_price * np.exp(predicted_log_return)),
            }
        )
    return records


def _direction_label(expected_return: float, validation_mae: float) -> str:
    threshold = max(0.0025, validation_mae * 0.25)
    if expected_return > threshold:
        return "Upward"
    if expected_return < -threshold:
        return "Downward"
    return "Flat"


def _technical_history_quality(
    prices: pd.DataFrame,
    target_column: str,
    data_quality_report: dict[str, Any],
) -> dict[str, Any]:
    rows = int(len(prices))
    years = float(rows / 252) if rows else 0.0
    target = target_column.lower()
    has_ohlc = all(column in prices.columns for column in ("open", "high", "low", target))
    has_volume = "volume" in prices.columns and pd.to_numeric(prices["volume"], errors="coerce").notna().mean() > 0.80
    warnings = []
    if rows < 252:
        warnings.append("Less than one trading year of history.")
    if rows < 504:
        warnings.append("Less than two trading years; longer chart background is preferred.")
    if not has_ohlc:
        warnings.append("Incomplete OHLC data limits chart-pattern and range-based features.")
    if not has_volume:
        warnings.append("Volume is missing or sparse, so supply/demand confirmation is weaker.")
    if data_quality_report.get("status") == "fail":
        warnings.append("Data quality report has high-severity warnings.")
    sufficient = rows >= 504 and has_ohlc and has_volume and data_quality_report.get("status") != "fail"
    return {
        "rows": rows,
        "approx_years": years,
        "start_date": str(prices.index.min().date()) if rows else None,
        "end_date": str(prices.index.max().date()) if rows else None,
        "has_ohlc": bool(has_ohlc),
        "has_volume": bool(has_volume),
        "sufficient_for_classical_technical_analysis": bool(sufficient),
        "warnings": warnings,
    }


def _trend_view(
    forecasts: list[dict[str, Any]],
    latest_features: pd.Series,
    latest_price: float,
    structure_snapshot: dict[str, float],
    support_resistance_by_timeframe: dict[str, Any],
) -> dict[str, Any]:
    score = 0.0
    evidence: list[str] = []
    warnings: list[str] = []

    forecast_by_horizon = {int(item["horizon_days"]): item for item in forecasts}
    for horizon, weight in ((5, 1.0), (30, 1.4), (1, 0.4)):
        forecast = forecast_by_horizon.get(horizon)
        if not forecast:
            continue
        direction = str(forecast["expected_direction"])
        confidence = float(forecast["directional_confidence"])
        if direction == "Upward":
            score += weight * confidence
            evidence.append(f"{horizon}d forecast points upward with {confidence:.0%} directional confidence.")
        elif direction == "Downward":
            score -= weight * confidence
            evidence.append(f"{horizon}d forecast points downward with {confidence:.0%} directional confidence.")

    checks = {
        "short_trend_state": float(latest_features.get("structure_trend_state_short", 0.0)),
        "long_trend_state": float(latest_features.get("structure_trend_state_long", 0.0)),
        "close_above_sma_50": float(latest_features.get("structure_close_above_sma_50", 0.0)),
        "close_above_sma_200": float(latest_features.get("structure_close_above_sma_200", 0.0)),
        "breakout_63d": float(latest_features.get("structure_breakout_63d", 0.0)),
        "breakdown_63d": float(latest_features.get("structure_breakdown_63d", 0.0)),
        "breakout_volume_confirmed_63d": float(latest_features.get("structure_breakout_volume_confirmed_63d", 0.0)),
        "breakdown_volume_confirmed_63d": float(latest_features.get("structure_breakdown_volume_confirmed_63d", 0.0)),
        "failed_breakout_63d": float(latest_features.get("structure_failed_breakout_63d", 0.0)),
        "failed_breakdown_63d": float(latest_features.get("structure_failed_breakdown_63d", 0.0)),
    }
    score += 0.5 * checks["short_trend_state"]
    score += 0.5 * checks["long_trend_state"]
    score += 0.35 if checks["close_above_sma_50"] > 0 else -0.35
    score += 0.35 if checks["close_above_sma_200"] > 0 else -0.35
    if checks["breakout_volume_confirmed_63d"] > 0:
        score += 1.0
        evidence.append("63d breakout is volume-confirmed.")
    elif checks["breakout_63d"] > 0:
        score += 0.6
        evidence.append("63d breakout is present but not volume-confirmed.")
    if checks["breakdown_volume_confirmed_63d"] > 0:
        score -= 1.0
        warnings.append("63d breakdown is volume-confirmed.")
    elif checks["breakdown_63d"] > 0:
        score -= 0.6
        warnings.append("63d breakdown is present.")
    if checks["failed_breakout_63d"] > 0:
        score -= 0.7
        warnings.append("Recent breakout appears to have failed.")
    if checks["failed_breakdown_63d"] > 0:
        score += 0.4
        evidence.append("Recent breakdown appears to have failed.")

    daily_levels = support_resistance_by_timeframe.get("daily", {})
    support = daily_levels.get("support", latest_features.get("structure_support_63d", np.nan))
    resistance = daily_levels.get("resistance", latest_features.get("structure_resistance_63d", np.nan))
    invalidation_level = float(support) if pd.notna(support) else None
    upside_reference = float(resistance) if pd.notna(resistance) else None
    if invalidation_level is not None:
        evidence.append(f"63d support/invalidation reference is {invalidation_level:.2f}.")
    if upside_reference is not None:
        evidence.append(f"63d resistance/upside reference is {upside_reference:.2f}.")

    if score >= 1.25:
        trend_state = "Bullish"
    elif score <= -1.25:
        trend_state = "Bearish"
    else:
        trend_state = "Neutral"
    confidence = float(min(0.90, max(0.35, 0.50 + abs(score) * 0.08)))
    return {
        "state": trend_state,
        "score": float(score),
        "confidence": confidence,
        "current_price": latest_price,
        "invalidation_level": invalidation_level,
        "upside_reference": upside_reference,
        "evidence": evidence[:12],
        "warnings": warnings[:12],
        "structure_snapshot": structure_snapshot,
    }


def _support_resistance_by_timeframe(prices: pd.DataFrame, target_column: str) -> dict[str, Any]:
    frames = {
        "daily": prices,
        "weekly": _resample_ohlcv(prices, "W-FRI"),
        "monthly": _resample_ohlcv(prices, "ME"),
    }
    result: dict[str, Any] = {}
    target = target_column.lower()
    for name, frame in frames.items():
        if frame.empty:
            result[name] = {"rows": 0, "support": None, "resistance": None, "latest_close": None}
            continue
        high = pd.to_numeric(frame["high"], errors="coerce") if "high" in frame.columns else pd.to_numeric(frame[target], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce") if "low" in frame.columns else pd.to_numeric(frame[target], errors="coerce")
        close = pd.to_numeric(frame[target], errors="coerce")
        window = min(63, max(5, len(frame) - 1)) if name == "daily" else min(52, max(5, len(frame) - 1)) if name == "weekly" else min(36, max(5, len(frame) - 1))
        support = low.shift(1).rolling(window).min().iloc[-1] if len(frame) > window else np.nan
        resistance = high.shift(1).rolling(window).max().iloc[-1] if len(frame) > window else np.nan
        latest_close = close.iloc[-1]
        result[name] = {
            "rows": int(len(frame)),
            "start_date": str(frame.index.min().date()),
            "end_date": str(frame.index.max().date()),
            "lookback_bars": int(window),
            "support": float(support) if pd.notna(support) else None,
            "resistance": float(resistance) if pd.notna(resistance) else None,
            "latest_close": float(latest_close) if pd.notna(latest_close) else None,
            "distance_to_support": float((latest_close - support) / latest_close) if pd.notna(latest_close) and pd.notna(support) and latest_close else None,
            "distance_to_resistance": float((resistance - latest_close) / latest_close) if pd.notna(latest_close) and pd.notna(resistance) and latest_close else None,
        }
    return result


def _resample_ohlcv(prices: pd.DataFrame, rule: str) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()
    aggregations: dict[str, str] = {}
    if "open" in prices.columns:
        aggregations["open"] = "first"
    if "high" in prices.columns:
        aggregations["high"] = "max"
    if "low" in prices.columns:
        aggregations["low"] = "min"
    if "close" in prices.columns:
        aggregations["close"] = "last"
    if "volume" in prices.columns:
        aggregations["volume"] = "sum"
    for optional in ("dividends", "stock_splits"):
        if optional in prices.columns:
            aggregations[optional] = "sum"
    frame = prices.resample(rule).agg(aggregations)
    return frame.dropna(subset=["close"]) if "close" in frame.columns else frame.dropna(how="all")


def _default_chart_metadata(prices: pd.DataFrame, target_column: str) -> dict[str, Any]:
    return {
        "recommended_scale": "log",
        "available_scales": ["log", "linear"],
        "default_price_display": "ohlc_with_close_overlay",
        "supported_timeframes": ["daily", "weekly", "monthly"],
        "target_column": target_column.lower(),
        "has_ohlc": all(column in prices.columns for column in ("open", "high", "low", target_column.lower())),
        "has_volume": "volume" in prices.columns,
        "support_resistance_method": {
            "daily": "prior 63 trading-session high/low range",
            "weekly": "prior 52 weekly-bar high/low range",
            "monthly": "prior 36 monthly-bar high/low range",
        },
        "magee_basing_points_overlay": {
            "included": True,
            "variants": ["variant_1_wave_low_stops", "variant_2_wave_low_plus_wave_high_stops"],
            "primary_timeframe": "weekly",
        },
        "reversal_patterns_overlay": {
            "included": True,
            "patterns": ["head_and_shoulders_top", "head_and_shoulders_bottom", "dormant_bottom_optional"],
            "confirmation": "3% close beyond neckline in the reversal direction",
        },
        "triangle_patterns_overlay": {
            "included": True,
            "patterns": ["symmetrical_triangle", "ascending_triangle", "descending_triangle"],
            "confirmation": "close beyond boundary with apex timing and volume reliability checks",
        },
        "chapter_9_patterns_overlay": {
            "included": True,
            "patterns": ["rectangle", "double_top", "double_bottom", "triple_top", "triple_bottom"],
            "confirmation": "close beyond rectangle boundary or the intervening top/bottom confirmation level",
        },
        "chapter_10_patterns_overlay": {
            "included": True,
            "patterns": ["broadening_top", "diamond", "rising_wedge", "falling_wedge", "one_day_events"],
            "confirmation": "structural close beyond active boundary; one-day events are shown as tactical warnings",
        },
        "chapter_11_patterns_overlay": {
            "included": True,
            "patterns": ["flag", "pennant", "head_and_shoulders_continuation", "scallop_context_optional"],
            "confirmation": "close beyond the short consolidation boundary in the prior mast direction",
        },
        "chapter_12_gaps_overlay": {
            "included": True,
            "patterns": ["common_gap", "breakaway_gap", "runaway_gap", "exhaustion_gap", "island_reversal"],
            "confirmation": "true range gap zone with fill/open state and corporate-action exclusion",
        },
        "chapter_13_support_resistance_overlay": {
            "included": True,
            "patterns": ["volume_weighted_support_zone", "volume_weighted_resistance_zone", "round_number_zone"],
            "confirmation": "historical pivot/volume zones with role reversal and attack-count diagnostics",
        },
        "chapter_14_trendlines_overlay": {
            "included": True,
            "patterns": ["basic_trendline", "double_trendline", "trend_channel", "three_fan_lines"],
            "confirmation": "confirmed pivot trendlines with 3% close penetration and pullback diagnostics",
        },
        "chapter_15_major_trendlines_overlay": {
            "included": True,
            "patterns": ["monthly_major_trendline", "scale_comparison", "broad_market_confirmation_optional"],
            "confirmation": "major monthly trendline with log-vs-linear scale selection; benchmarks are optional",
        },
        "chapter_16_market_context_overlay": {
            "included": True,
            "patterns": ["donchian_20_55_channels", "trending_vs_trading_context", "optional_open_interest_context"],
            "confirmation": "report-only commodity/futures market context; it does not influence final action",
        },
        "clean_signal_chart": {
            "included": True,
            "default_scale": "linear",
            "purpose": "production-readable companion chart with reduced annotation density and ranked active evidence",
            "visible_layers": [
                "price_or_ohlc",
                "sma_20",
                "sma_50",
                "donchian_20_channel",
                "nearest_daily_support_resistance",
                "ranked_signal_table",
            ],
        },
    }


def _apply_dow_regime_filter(raw_action: str, dow_theory: dict[str, Any]) -> dict[str, Any]:
    primary_state = dow_theory.get("primary_trend", {}).get("state", "Neutral")
    confirmation_status = dow_theory.get("trend_confirmation", {}).get("status", "Unavailable")
    continuation_state = dow_theory.get("continuation_rule", {}).get("state", "WaitForClearConfirmation")
    line_state = dow_theory.get("line_pattern", {}).get("state", "NoCompactLine")
    ambiguity = float(
        dow_theory.get("chapter_4_defect_diagnostics", {})
        .get("sensitivity_analysis", {})
        .get("ambiguity_score", 0.0)
        or 0.0
    )
    filtered_action = raw_action
    blockers: list[str] = []
    warnings: list[str] = []

    if raw_action == "Buy":
        if primary_state == "Bearish":
            blockers.append("Dow primary regime is Bearish, blocking a Buy signal.")
        if confirmation_status == "Divergent":
            blockers.append("Benchmark/sector confirmation is divergent, blocking a Buy signal.")
        if continuation_state in {"TreatTrendAsDoubt", "WaitForClearConfirmation"}:
            blockers.append("Dow continuation rule is not clear enough for a Buy signal.")
    elif raw_action == "Sell":
        if primary_state == "Bullish":
            blockers.append("Dow primary regime is Bullish, blocking a Sell signal.")
        if confirmation_status == "Divergent":
            blockers.append("Benchmark/sector confirmation is divergent, blocking a Sell signal.")
        if continuation_state in {"TreatTrendAsDoubt", "WaitForClearConfirmation"}:
            blockers.append("Dow continuation rule is not clear enough for a Sell signal.")

    if raw_action in {"Buy", "Sell"} and line_state == "ActiveLine":
        blockers.append("Price is inside an active Dow line/range, so directional action waits for a break.")
    if ambiguity > 0.45:
        warnings.append("Dow primary trend is sensitive to reasonable lookback/threshold changes.")
        if raw_action in {"Buy", "Sell"}:
            blockers.append("Dow ambiguity is high, so the directional action is held for review.")

    if blockers:
        filtered_action = "Hold"

    return {
        "raw_action": raw_action,
        "filtered_action": filtered_action,
        "filter_applied": filtered_action != raw_action,
        "primary_trend": primary_state,
        "confirmation_status": confirmation_status,
        "continuation_state": continuation_state,
        "line_state": line_state,
        "ambiguity_score": ambiguity,
        "blocking_reasons": blockers,
        "warnings": warnings,
    }


def _apply_magee_basing_filter(action: str, magee_basing_points: dict[str, Any]) -> dict[str, Any]:
    preferred = magee_basing_points.get("preferred", {})
    trend_state = preferred.get("trend_state", "Neutral")
    stop_status = preferred.get("stop_status", "NoActiveStop")
    stop_distance = preferred.get("stop_distance_pct")
    timeframe = preferred.get("timeframe", "weekly")
    variant = preferred.get("variant", "variant_2")
    filtered_action = action
    blockers: list[str] = []
    warnings: list[str] = []

    distance = float(stop_distance) if stop_distance is not None else None
    if action == "Buy":
        if stop_status == "BelowBasingStop" or trend_state == "Short":
            blockers.append("Magee basing-points trend is below its active stop, blocking a Buy signal.")
        elif distance is not None and 0.0 <= distance < 0.03:
            blockers.append("Price is less than 3% above the Magee basing stop, so the Buy signal is held for stop-risk review.")
    elif action == "Sell":
        if trend_state == "Long" and stop_status == "AboveBasingStop":
            blockers.append("Magee long-term basing trend remains above its active stop, blocking a Sell signal.")

    if distance is not None and distance > 0.20:
        warnings.append("Price is far above the Magee basing stop; position risk to the stop is wide.")
    if stop_status == "NoActiveStop":
        warnings.append("Magee basing stop is unavailable for the preferred timeframe.")

    if blockers:
        filtered_action = "Hold"

    return {
        "input_action": action,
        "filtered_action": filtered_action,
        "filter_applied": filtered_action != action,
        "timeframe": timeframe,
        "variant": variant,
        "trend_state": trend_state,
        "stop_status": stop_status,
        "active_basing_stop": preferred.get("active_basing_stop"),
        "stop_distance_pct": distance,
        "blocking_reasons": blockers,
        "warnings": warnings,
    }


def _apply_reversal_pattern_filter(action: str, reversal_patterns: dict[str, Any]) -> dict[str, Any]:
    preferred = reversal_patterns.get("preferred", {})
    pattern = preferred.get("pattern", "NoPattern")
    direction = preferred.get("direction")
    status = preferred.get("status", "NoPattern")
    timeframe = preferred.get("timeframe")
    score = float(preferred.get("score") or 0.0)
    objective_status = preferred.get("objective_status", "Unavailable")
    measured_objective = preferred.get("measured_objective")
    latest_margin = preferred.get("latest_neckline_margin_pct")
    filtered_action = action
    blockers: list[str] = []
    warnings: list[str] = []

    active_confirmed_top = pattern == "HeadAndShouldersTop" and status in {"Confirmed", "PullbackToNeckline"}
    active_confirmed_bottom = pattern == "HeadAndShouldersBottom" and status in {"Confirmed", "ThrowbackToNeckline"}
    if action == "Buy" and active_confirmed_top and score >= 0.55:
        blockers.append("Confirmed Head-and-Shoulders Top is active, blocking a fresh Buy signal.")
    elif action == "Sell" and active_confirmed_bottom and score >= 0.55:
        blockers.append("Confirmed Head-and-Shoulders Bottom is active, blocking a fresh Sell signal.")
    elif status == "Candidate" and score >= 0.65:
        warnings.append(f"{pattern} candidate is forming; neckline confirmation is not present yet.")
    elif status == "ObjectiveReached":
        warnings.append(f"A prior {pattern} reached its measured objective; treat the reversal signal as played out.")
    elif status == "Failed":
        warnings.append(f"A prior {pattern} failed by reclaiming the neckline.")

    if active_confirmed_top and measured_objective is not None:
        warnings.append("Measured objective remains part of the downside risk map.")
    if active_confirmed_bottom and measured_objective is not None:
        warnings.append("Measured objective remains part of the upside recovery map.")
    optional_methods = reversal_patterns.get("optional_methods", {})
    complex_status = optional_methods.get("complex_head_and_shoulders", {}).get("preferred", {}).get("status")
    dormant = optional_methods.get("dormant_bottoms", {}).get("preferred", {})
    if complex_status == "WarningOnly":
        warnings.append("Complex Head-and-Shoulders warning is present; use manual chart review before acting.")
    if dormant.get("status") in {"Breakout", "Candidate", "BaseForming"}:
        warnings.append("Optional Dormant Bottom diagnostic is active; treat it as accumulation context, not an automatic signal.")
    if blockers:
        filtered_action = "Hold"

    return {
        "input_action": action,
        "filtered_action": filtered_action,
        "filter_applied": filtered_action != action,
        "pattern": pattern,
        "direction": direction,
        "status": status,
        "timeframe": timeframe,
        "score": score,
        "objective_status": objective_status,
        "measured_objective": measured_objective,
        "latest_neckline_margin_pct": latest_margin,
        "blocking_reasons": blockers,
        "warnings": warnings,
    }


def _apply_triangle_pattern_filter(action: str, triangle_patterns: dict[str, Any]) -> dict[str, Any]:
    preferred = triangle_patterns.get("preferred", {})
    pattern = preferred.get("pattern", "NoTriangle")
    status = preferred.get("status", "NoPattern")
    direction = preferred.get("direction", "undetermined")
    timeframe = preferred.get("timeframe")
    score = float(preferred.get("score") or 0.0)
    apex_timing = preferred.get("apex", {}).get("timing")
    filtered_action = action
    blockers: list[str] = []
    warnings: list[str] = []

    if pattern == "SymmetricalTriangle" and status == "Candidate" and action in {"Buy", "Sell"}:
        blockers.append("Unbroken Symmetrical Triangle is active; directional action waits for a decisive breakout.")
    if action == "Buy" and status in {"Breakdown", "Retest"} and direction == "bearish" and score >= 0.55:
        blockers.append("Confirmed bearish triangle breakdown conflicts with a fresh Buy signal.")
    if action == "Sell" and status in {"Breakout", "Retest"} and direction == "bullish" and score >= 0.55:
        blockers.append("Confirmed bullish triangle breakout conflicts with a fresh Sell signal.")
    if action == "Buy" and status == "FailedBreakout":
        blockers.append("Recent triangle upside breakout failed, blocking a fresh Buy signal.")
    if action == "Sell" and status == "FailedBreakdown":
        blockers.append("Recent triangle downside breakdown failed, blocking a fresh Sell signal.")

    if status == "LateApex" or apex_timing == "LateApex":
        warnings.append("Triangle is in the late-apex zone; breakout reliability is reduced.")
    if preferred.get("volume_confirmation", {}).get("breakout_volume_expansion") is False:
        warnings.append("Triangle breakout lacks volume expansion.")
    if pattern in {"AscendingTriangle", "DescendingTriangle"} and status == "Candidate":
        warnings.append(f"{pattern} is forming but remains unconfirmed until boundary breakout.")

    if blockers:
        filtered_action = "Hold"

    return {
        "input_action": action,
        "filtered_action": filtered_action,
        "filter_applied": filtered_action != action,
        "pattern": pattern,
        "status": status,
        "direction": direction,
        "timeframe": timeframe,
        "score": score,
        "apex_timing": apex_timing,
        "measured_objective": preferred.get("measured_objective"),
        "blocking_reasons": blockers,
        "warnings": warnings,
    }


def _apply_chapter_9_pattern_filter(action: str, chapter_9_patterns: dict[str, Any]) -> dict[str, Any]:
    rectangles = chapter_9_patterns.get("rectangle_patterns", {})
    multi_patterns = chapter_9_patterns.get("multi_top_bottom_patterns", {})
    rectangle = rectangles.get("preferred", {})
    multi = multi_patterns.get("preferred", {})
    rectangle_status = rectangle.get("status", "NoPattern")
    rectangle_direction = rectangle.get("direction", "undetermined")
    rectangle_score = float(rectangle.get("score") or 0.0)
    multi_pattern = multi.get("pattern", "NoMultiTopBottom")
    multi_status = multi.get("status", "NoPattern")
    multi_direction = multi.get("direction", "warning_only")
    multi_score = float(multi.get("score") or 0.0)
    filtered_action = action
    blockers: list[str] = []
    warnings: list[str] = []

    if action in {"Buy", "Sell"} and rectangle_status == "Candidate" and rectangle_score >= 0.55:
        blockers.append("Unbroken Rectangle is active; directional action waits for a confirmed range break.")
    if action == "Buy" and rectangle_status in {"Breakdown", "Retest", "FalseBreakout"} and rectangle_direction == "bearish" and rectangle_score >= 0.55:
        blockers.append("Chapter 9 rectangle breakdown conflicts with a fresh Buy signal.")
    if action == "Sell" and rectangle_status in {"Breakout", "Retest", "FalseBreakdown"} and rectangle_direction == "bullish" and rectangle_score >= 0.55:
        blockers.append("Chapter 9 rectangle breakout conflicts with a fresh Sell signal.")
    if action == "Buy" and rectangle_status == "PrematureBreakout":
        blockers.append("Rectangle upside break returned inside the range, blocking a fresh Buy signal.")
    if action == "Sell" and rectangle_status == "PrematureBreakdown":
        blockers.append("Rectangle downside break returned inside the range, blocking a fresh Sell signal.")

    if rectangle_status in {"Candidate", "PrematureBreakout", "PrematureBreakdown", "FalseBreakout", "FalseBreakdown"}:
        warnings.extend(str(note) for note in rectangle.get("reliability_notes", [])[:2])
    if rectangle.get("volume_confirmation", {}).get("volume_contracts_inside_pattern") is False:
        warnings.append("Rectangle volume did not contract clearly inside the range.")
    if rectangle.get("volume_confirmation", {}).get("breakout_volume_expansion") is False:
        warnings.append("Rectangle boundary break lacks volume expansion.")

    confirmed_multi = multi_status in {"Confirmed", "PullbackToConfirmation"}
    is_confirmed_top = multi_pattern in {"DoubleTop", "TripleTop"} and confirmed_multi and multi_direction == "bearish"
    is_confirmed_bottom = multi_pattern in {"DoubleBottom", "TripleBottom"} and confirmed_multi and multi_direction == "bullish"
    if action == "Buy" and is_confirmed_top and multi_score >= 0.55:
        blockers.append(f"Confirmed {multi_pattern} conflicts with a fresh Buy signal.")
    if action == "Sell" and is_confirmed_bottom and multi_score >= 0.55:
        blockers.append(f"Confirmed {multi_pattern} conflicts with a fresh Sell signal.")
    if multi_status == "Suspected" and multi_score >= 0.45:
        warnings.append(f"{multi_pattern} is suspected but unconfirmed; wait for the Chapter 9 confirmation level.")
    if multi_status == "ObjectiveReached":
        warnings.append(f"{multi_pattern} reached its measured objective; treat the signal as played out.")
    warnings.extend(str(note) for note in multi.get("reliability_notes", [])[:2])

    if blockers:
        filtered_action = "Hold"

    return {
        "input_action": action,
        "filtered_action": filtered_action,
        "filter_applied": filtered_action != action,
        "rectangle_pattern": rectangle.get("pattern", "Rectangle"),
        "rectangle_status": rectangle_status,
        "rectangle_direction": rectangle_direction,
        "rectangle_score": rectangle_score,
        "rectangle_objective": rectangle.get("measured_objective"),
        "multi_top_bottom_pattern": multi_pattern,
        "multi_top_bottom_status": multi_status,
        "multi_top_bottom_direction": multi_direction,
        "multi_top_bottom_score": multi_score,
        "multi_top_bottom_objective": multi.get("measured_objective"),
        "blocking_reasons": blockers,
        "warnings": warnings,
    }


def _apply_chapter_10_pattern_filter(action: str, chapter_10_patterns: dict[str, Any]) -> dict[str, Any]:
    structural = chapter_10_patterns.get("structural_patterns", {}).get("preferred", {})
    event = chapter_10_patterns.get("short_term_events", {}).get("preferred", {})
    pattern = structural.get("pattern", "NoChapter10StructuralPattern")
    status = structural.get("status", "NoPattern")
    direction = structural.get("direction", "undetermined")
    score = float(structural.get("score") or 0.0)
    event_pattern = event.get("pattern", "NoShortTermEvent")
    event_direction = event.get("direction", "undetermined")
    event_score = float(event.get("score") or 0.0)
    filtered_action = action
    blockers: list[str] = []
    warnings: list[str] = []

    if action == "Buy" and pattern in {"BroadeningTop", "FlatToppedBroadening"} and status in {"Confirmed", "PullbackToBoundary"} and score >= 0.55:
        blockers.append(f"Confirmed {pattern} conflicts with a fresh Buy signal.")
    if action == "Buy" and pattern == "RisingWedge" and status in {"Breakdown", "Retest"} and score >= 0.55:
        blockers.append("Rising Wedge broke down, blocking a fresh Buy signal.")
    if action == "Sell" and pattern == "FallingWedge" and status in {"Breakout", "Retest"} and score >= 0.55:
        blockers.append("Falling Wedge broke upward, blocking a fresh Sell signal.")
    if action == "Buy" and pattern == "Diamond" and status in {"Breakdown", "Retest"} and direction == "bearish" and score >= 0.55:
        blockers.append("Diamond downside break conflicts with a fresh Buy signal.")
    if action == "Sell" and pattern == "Diamond" and status in {"Breakout", "Retest"} and direction == "bullish" and score >= 0.55:
        blockers.append("Diamond upside break conflicts with a fresh Sell signal.")

    if status == "Candidate" and score >= 0.50:
        warnings.append(f"{pattern} is forming but remains unconfirmed.")
    if status == "UpsideBreakout" and pattern in {"BroadeningTop", "FlatToppedBroadening"}:
        warnings.append("Broadening-pattern upside break is uncommon; require follow-through.")
    warnings.extend(str(note) for note in structural.get("reliability_notes", [])[:2])

    bearish_event = event_pattern in {"OneDayReversalTop", "KeyReversalTop", "SpikeTop"} or event_direction == "bearish"
    bullish_event = event_pattern in {"OneDayReversalBottom", "KeyReversalBottom", "SpikeBottom", "SellingClimax"} or event_direction == "bullish"
    if action == "Buy" and bearish_event and event_score >= 0.78:
        blockers.append(f"{event_pattern} is a strong short-term exhaustion warning against a fresh Buy signal.")
    if action == "Sell" and bullish_event and event_score >= 0.78:
        blockers.append(f"{event_pattern} is a strong short-term exhaustion warning against a fresh Sell signal.")
    if event_pattern != "NoShortTermEvent":
        warnings.extend(str(note) for note in event.get("reliability_notes", [])[:2])

    if blockers:
        filtered_action = "Hold"

    return {
        "input_action": action,
        "filtered_action": filtered_action,
        "filter_applied": filtered_action != action,
        "structural_pattern": pattern,
        "structural_status": status,
        "structural_direction": direction,
        "structural_score": score,
        "structural_objective": structural.get("measured_objective"),
        "short_term_event": event_pattern,
        "short_term_event_direction": event_direction,
        "short_term_event_score": event_score,
        "short_term_event_invalidation": event.get("invalidation_level"),
        "blocking_reasons": blockers,
        "warnings": warnings,
    }


def _apply_chapter_11_continuation_filter(action: str, chapter_11_patterns: dict[str, Any]) -> dict[str, Any]:
    continuation = chapter_11_patterns.get("continuation_patterns", {}).get("preferred", {})
    hs = chapter_11_patterns.get("head_and_shoulders_continuation", {}).get("preferred", {})
    candidates = [
        pattern
        for pattern in (continuation, hs)
        if isinstance(pattern, dict) and pattern.get("status") not in {None, "NoPattern", "InsufficientData"}
    ]
    preferred = max(candidates, key=_chapter_11_filter_rank) if candidates else chapter_11_patterns.get("preferred", {})
    pattern = preferred.get("pattern", "NoChapter11Pattern")
    status = preferred.get("status", "NoPattern")
    direction = preferred.get("direction", "undetermined")
    score = float(preferred.get("score") or 0.0)
    filtered_action = action
    blockers: list[str] = []
    warnings: list[str] = []
    supporting: list[str] = []

    bullish_confirmed = direction == "bullish" and status in {"Breakout", "Confirmed"}
    bearish_confirmed = direction == "bearish" and status in {"Breakdown", "Confirmed"}
    bullish_candidate = direction == "bullish" and status == "Candidate"
    bearish_candidate = direction == "bearish" and status == "Candidate"
    failed_bullish = direction == "failed_bullish" or (
        status == "FailedBreakout" and preferred.get("expected_breakout_direction") == "up"
    )
    failed_bearish = direction == "failed_bearish" or (
        status == "FailedBreakdown" and preferred.get("expected_breakout_direction") == "down"
    )

    if action == "Buy" and bearish_confirmed and score >= 0.55:
        blockers.append(f"Confirmed Chapter 11 bearish continuation ({pattern}) conflicts with a fresh Buy signal.")
    if action == "Sell" and bullish_confirmed and score >= 0.55:
        blockers.append(f"Confirmed Chapter 11 bullish continuation ({pattern}) conflicts with a fresh Sell signal.")
    if action == "Buy" and failed_bullish and score >= 0.45:
        blockers.append(f"Chapter 11 bullish continuation failed ({pattern}), blocking a fresh Buy signal.")
    if action == "Sell" and failed_bearish and score >= 0.45:
        blockers.append(f"Chapter 11 bearish continuation failed ({pattern}), blocking a fresh Sell signal.")

    if action == "Buy" and bullish_confirmed and score >= 0.55:
        supporting.append(f"Confirmed Chapter 11 bullish continuation ({pattern}) supports the Buy signal.")
    if action == "Sell" and bearish_confirmed and score >= 0.55:
        supporting.append(f"Confirmed Chapter 11 bearish continuation ({pattern}) supports the Sell signal.")
    if action == "Buy" and bullish_candidate and score >= 0.60:
        supporting.append(f"Chapter 11 bullish continuation candidate ({pattern}) aligns with the Buy signal, pending breakout.")
    if action == "Sell" and bearish_candidate and score >= 0.60:
        supporting.append(f"Chapter 11 bearish continuation candidate ({pattern}) aligns with the Sell signal, pending breakdown.")

    if status == "Candidate" and score >= 0.50:
        warnings.append(f"{pattern} is forming but remains unconfirmed until it breaks in the mast direction.")
    if status == "Stale":
        warnings.append(f"{pattern} has exceeded the Chapter 11 reliability window.")
    if status in {"FailedBreakout", "FailedBreakdown"}:
        warnings.append(f"{pattern} broke against its expected continuation direction.")
    if status == "ObjectiveReached":
        warnings.append(f"{pattern} already reached its measured continuation objective.")
    volume = preferred.get("volume_confirmation", {})
    if volume.get("volume_contracts_inside_pattern") is False:
        warnings.append(f"{pattern} volume did not contract during consolidation.")
    if volume.get("breakout_volume_expansion") is False:
        warnings.append(f"{pattern} breakout lacks volume expansion.")
    warnings.extend(str(note) for note in preferred.get("reliability_notes", [])[:2])

    if blockers:
        filtered_action = "Hold"

    return {
        "input_action": action,
        "filtered_action": filtered_action,
        "filter_applied": filtered_action != action,
        "pattern": pattern,
        "status": status,
        "direction": direction,
        "timeframe": preferred.get("timeframe"),
        "score": score,
        "measured_objective": preferred.get("measured_objective"),
        "measured_move_pct": preferred.get("measured_move_pct"),
        "breakout_date": preferred.get("breakout_date"),
        "breakout_close": preferred.get("breakout_close"),
        "blocking_reasons": blockers,
        "supporting_reasons": supporting,
        "warnings": warnings,
    }


def _chapter_11_filter_rank(pattern: dict[str, Any]) -> tuple[float, float, str]:
    status_rank = {
        "Breakout": 5.0,
        "Breakdown": 5.0,
        "Confirmed": 4.5,
        "Candidate": 2.0,
        "Stale": 1.0,
        "FailedBreakout": 0.8,
        "FailedBreakdown": 0.8,
        "ObjectiveReached": 0.6,
        "PossibleSequence": 0.2,
    }.get(str(pattern.get("status")), 0.0)
    pattern_bonus = {"Flag": 0.25, "Pennant": 0.25, "HeadAndShouldersContinuation": 0.15}.get(str(pattern.get("pattern")), 0.0)
    timeframe_bonus = {"daily": 0.20, "weekly": 0.05, "chart": 0.10}.get(str(pattern.get("timeframe")), 0.0)
    score = float(pattern.get("score") or 0.0)
    date = pattern.get("breakout_date") or pattern.get("boundaries", {}).get("latest_date") or ""
    return (status_rank + pattern_bonus + timeframe_bonus, score, str(date))


def _apply_chapter_12_gap_filter(action: str, chapter_12_gaps: dict[str, Any]) -> dict[str, Any]:
    gap = chapter_12_gaps.get("classified_gaps", {}).get("preferred", {})
    island = chapter_12_gaps.get("island_reversals", {}).get("preferred", {})
    candidates = [
        pattern
        for pattern in (gap, island)
        if isinstance(pattern, dict) and pattern.get("status") not in {None, "NoPattern", "InsufficientData", "Ignored", "Excluded"}
    ]
    preferred = max(candidates, key=_chapter_12_filter_rank) if candidates else chapter_12_gaps.get("preferred", {})
    pattern = preferred.get("pattern", "NoChapter12Gap")
    status = preferred.get("status", "NoPattern")
    direction = preferred.get("direction", "undetermined")
    score = float(preferred.get("score") or 0.0)
    filtered_action = action
    blockers: list[str] = []
    warnings: list[str] = []
    supporting: list[str] = []

    bullish_confirmation = pattern in {"BreakawayGap", "RunawayGap"} and direction == "bullish" and score >= 0.55
    bearish_confirmation = pattern in {"BreakawayGap", "RunawayGap"} and direction == "bearish" and score >= 0.55
    bullish_warning = (
        (pattern == "ExhaustionGap" and direction == "bullish_warning")
        or (pattern == "IslandReversal" and direction == "bullish")
    )
    bearish_warning = (
        (pattern == "ExhaustionGap" and direction == "bearish_warning")
        or (pattern == "IslandReversal" and direction == "bearish")
    )

    if action == "Buy" and bearish_confirmation and score >= 0.60:
        blockers.append(f"Chapter 12 {pattern} confirms downside pressure against a fresh Buy signal.")
    if action == "Sell" and bullish_confirmation and score >= 0.60:
        blockers.append(f"Chapter 12 {pattern} confirms upside pressure against a fresh Sell signal.")
    if action == "Buy" and bearish_warning and score >= 0.55:
        blockers.append(f"Chapter 12 {pattern} warns of upside exhaustion or bearish island reversal, blocking a fresh Buy signal.")
    if action == "Sell" and bullish_warning and score >= 0.55:
        blockers.append(f"Chapter 12 {pattern} warns of downside exhaustion or bullish island reversal, blocking a fresh Sell signal.")

    if action == "Buy" and bullish_confirmation:
        supporting.append(f"Chapter 12 {pattern} supports the Buy signal.")
    if action == "Sell" and bearish_confirmation:
        supporting.append(f"Chapter 12 {pattern} supports the Sell signal.")

    if pattern == "CommonGap":
        warnings.append("Chapter 12 common/area gap is context only and does not drive the decision.")
    if pattern == "RunawayGap" and preferred.get("objective_reached") is True:
        warnings.append("Runaway gap measured objective has already been reached; use it as exit/risk context.")
    if pattern == "ExhaustionGap":
        warnings.append("Exhaustion gap is a stop/warning condition; wait for the next pattern before trusting continuation.")
    if pattern == "IslandReversal":
        warnings.append("Island Reversal is tactical reversal evidence and can be late for entry.")
    if preferred.get("habitual_gap_warning"):
        warnings.append("This issue/timeframe gaps frequently; downgrade single-gap reliability.")
    warnings.extend(str(note) for note in preferred.get("reliability_notes", [])[:2])

    if blockers:
        filtered_action = "Hold"

    return {
        "input_action": action,
        "filtered_action": filtered_action,
        "filter_applied": filtered_action != action,
        "pattern": pattern,
        "status": status,
        "direction": direction,
        "gap_direction": preferred.get("gap_direction"),
        "timeframe": preferred.get("timeframe"),
        "score": score,
        "gap_zone": preferred.get("gap_zone"),
        "fill_state": preferred.get("fill_state"),
        "measured_objective": preferred.get("measured_objective"),
        "objective_reached": preferred.get("objective_reached"),
        "blocking_reasons": blockers,
        "supporting_reasons": supporting,
        "warnings": warnings,
    }


def _chapter_12_filter_rank(pattern: dict[str, Any]) -> tuple[float, float, str]:
    pattern_rank = {
        "IslandReversal": 6.0,
        "ExhaustionGap": 5.0,
        "RunawayGap": 4.0,
        "BreakawayGap": 3.5,
        "CommonGap": 1.0,
    }.get(str(pattern.get("pattern")), 0.0)
    status_rank = {
        "Confirmed": 1.0,
        "Open": 0.8,
        "ExhaustionWarning": 0.8,
        "ClosedQuickly": 0.7,
        "PartiallyFilled": 0.5,
        "Filled": 0.3,
    }.get(str(pattern.get("status")), 0.0)
    score = float(pattern.get("score") or 0.0)
    date = pattern.get("date") or pattern.get("end_date") or ""
    return (pattern_rank + status_rank, score, str(date))


def _apply_chapter_13_support_resistance_filter(
    action: str,
    chapter_13_support_resistance: dict[str, Any],
) -> dict[str, Any]:
    support = chapter_13_support_resistance.get("support_zones", {}).get("nearest", {})
    resistance = chapter_13_support_resistance.get("resistance_zones", {}).get("nearest", {})
    preferred = chapter_13_support_resistance.get("preferred", {})
    active = _chapter_13_active_state(chapter_13_support_resistance)
    filtered_action = action
    blockers: list[str] = []
    warnings: list[str] = []
    supporting: list[str] = []

    support_strength = float(support.get("remaining_strength") or support.get("score") or 0.0)
    resistance_strength = float(resistance.get("remaining_strength") or resistance.get("score") or 0.0)
    support_distance = float(support.get("distance_to_zone_pct") or 999.0)
    resistance_distance = float(resistance.get("distance_to_zone_pct") or 999.0)
    support_failure = bool(active.get("support_failure")) and bool(active.get("volume_confirmed"))
    resistance_breakout = bool(active.get("resistance_breakout")) and bool(active.get("volume_confirmed"))

    if action == "Buy" and support_failure and support_strength >= 0.45:
        blockers.append("Chapter 13 volume-confirmed support failure conflicts with a fresh Buy signal.")
    if action == "Sell" and resistance_breakout and resistance_strength >= 0.45:
        blockers.append("Chapter 13 volume-confirmed resistance breakout conflicts with a fresh Sell signal.")
    if action == "Buy" and resistance_distance <= 0.035 and resistance_strength >= 0.55:
        blockers.append("Strong Chapter 13 resistance is too close for a fresh Buy signal.")
    if action == "Sell" and support_distance <= 0.035 and support_strength >= 0.55:
        blockers.append("Strong Chapter 13 support is too close for a fresh Sell signal.")

    if action == "Buy" and support_distance <= 0.035 and support_strength >= 0.50 and not support_failure:
        supporting.append("Price is near a strong Chapter 13 support zone.")
    if action == "Sell" and resistance_distance <= 0.035 and resistance_strength >= 0.50 and not resistance_breakout:
        supporting.append("Price is near a strong Chapter 13 resistance zone.")
    if action == "Buy" and resistance_breakout:
        supporting.append("Chapter 13 volume-confirmed resistance breakout supports the Buy signal.")
    if action == "Sell" and support_failure:
        supporting.append("Chapter 13 volume-confirmed support failure supports the Sell signal.")

    if resistance_distance <= 0.06 and resistance_strength >= 0.45:
        warnings.append("Nearest resistance may cap upside or reduce reward/risk.")
    if support_distance <= 0.06 and support_strength >= 0.45:
        warnings.append("Nearest support may limit downside follow-through.")
    for zone in (support, resistance):
        if zone.get("role_reversal") in {"OldTopAsSupport", "OldBottomAsResistance"}:
            warnings.extend(str(note) for note in zone.get("reliability_notes", [])[:1])

    if blockers:
        filtered_action = "Hold"

    return {
        "input_action": action,
        "filtered_action": filtered_action,
        "filter_applied": filtered_action != action,
        "pattern": preferred.get("pattern", "NoChapter13Zone"),
        "status": preferred.get("status", "NoPattern"),
        "nearest_support": support,
        "nearest_resistance": resistance,
        "support_distance_pct": support.get("distance_to_zone_pct"),
        "resistance_distance_pct": resistance.get("distance_to_zone_pct"),
        "support_strength": support_strength,
        "resistance_strength": resistance_strength,
        "support_failure": support_failure,
        "resistance_breakout": resistance_breakout,
        "blocking_reasons": blockers,
        "supporting_reasons": supporting,
        "warnings": warnings,
    }


def _chapter_13_active_state(chapter_13_support_resistance: dict[str, Any]) -> dict[str, Any]:
    for timeframe in ("weekly", "monthly", "daily"):
        active = chapter_13_support_resistance.get("timeframes", {}).get(timeframe, {}).get("active_state", {})
        if active.get("support_failure") or active.get("resistance_breakout"):
            return active
    return chapter_13_support_resistance.get("timeframes", {}).get("weekly", {}).get("active_state", {})


def _apply_chapter_14_trendline_filter(
    action: str,
    chapter_14_trendlines: dict[str, Any],
) -> dict[str, Any]:
    trendline = chapter_14_trendlines.get("trendlines", {}).get("preferred", {})
    channel = chapter_14_trendlines.get("channels", {}).get("preferred", {})
    fan = chapter_14_trendlines.get("fan_lines", {}).get("preferred", {})
    filtered_action = action
    blockers: list[str] = []
    warnings: list[str] = []
    supporting: list[str] = []

    kind = str(trendline.get("kind", ""))
    status = str(trendline.get("status", ""))
    direction = str(trendline.get("direction", ""))
    authority = float(trendline.get("authority_score") or trendline.get("score") or 0.0)
    effective_break = bool(trendline.get("effective_decisive_break"))
    distance_to_line = float(trendline.get("distance_to_line_pct") or 999.0)

    if action == "Buy" and kind == "uptrend" and effective_break and authority >= 0.50:
        blockers.append("Chapter 14 decisive uptrend-line break conflicts with a fresh Buy signal.")
    if action == "Sell" and kind == "downtrend" and effective_break and authority >= 0.50:
        blockers.append("Chapter 14 decisive downtrend-line break conflicts with a fresh Sell signal.")
    if action == "Sell" and kind == "uptrend" and status == "Active" and authority >= 0.58 and 0.0 <= distance_to_line <= 0.05:
        blockers.append("Authoritative Chapter 14 uptrend line remains intact near price, blocking a fresh Sell signal.")
    if action == "Buy" and kind == "downtrend" and status == "Active" and authority >= 0.58 and 0.0 <= distance_to_line <= 0.05:
        blockers.append("Authoritative Chapter 14 downtrend line remains intact near price, blocking a fresh Buy signal.")

    if action == "Sell" and kind == "uptrend" and effective_break:
        supporting.append("Chapter 14 decisive uptrend-line break supports the Sell signal.")
    if action == "Buy" and kind == "downtrend" and effective_break:
        supporting.append("Chapter 14 decisive downtrend-line break supports the Buy signal.")
    if action == "Buy" and kind == "uptrend" and status == "Active" and authority >= 0.50:
        supporting.append("Chapter 14 active uptrend line supports the Buy signal.")
    if action == "Sell" and kind == "downtrend" and status == "Active" and authority >= 0.50:
        supporting.append("Chapter 14 active downtrend line supports the Sell signal.")

    fan_status = str(fan.get("status", ""))
    fan_direction = str(fan.get("direction", ""))
    if action == "Buy" and fan_status == "ThirdFanBreakDownside":
        blockers.append("Chapter 14 bearish third fan-line break conflicts with a fresh Buy signal.")
    if action == "Sell" and fan_status == "ThirdFanBreakUpside":
        blockers.append("Chapter 14 bullish third fan-line break conflicts with a fresh Sell signal.")
    if action == "Buy" and fan_status == "ThirdFanBreakUpside":
        supporting.append("Chapter 14 bullish third fan-line break supports the Buy signal.")
    if action == "Sell" and fan_status == "ThirdFanBreakDownside":
        supporting.append("Chapter 14 bearish third fan-line break supports the Sell signal.")

    if status == "InnerLineBreak":
        warnings.append("Chapter 14 inner trendline is broken, but the outer double trendline remains intact.")
    if status == "ShakeoutWarning":
        warnings.append("Chapter 14 saw an intraday trendline penetration without closing confirmation.")
    if status == "PullbackToBrokenLine":
        warnings.append("Chapter 14 pullback to a broken trendline failed to reclaim the line.")
    if channel.get("status") == "ReturnLineFailure":
        warnings.append("Chapter 14 channel shows deterioration from failure to reach the return line.")
    if channel.get("status") == "ReturnLineBreakout":
        supporting.append("Chapter 14 channel return-line breakout shows trend acceleration.")
    if fan_status == "FanLinesDeveloping":
        warnings.append(f"Chapter 14 {fan_direction} fan lines are developing but not yet confirmed.")

    if blockers:
        filtered_action = "Hold"

    return {
        "input_action": action,
        "filtered_action": filtered_action,
        "filter_applied": filtered_action != action,
        "pattern": trendline.get("pattern", "NoChapter14Trendline"),
        "status": status or "NoPattern",
        "direction": direction,
        "kind": kind,
        "authority_score": trendline.get("authority_score"),
        "effective_decisive_break": effective_break,
        "channel_status": channel.get("status"),
        "fan_status": fan.get("status"),
        "blocking_reasons": blockers,
        "supporting_reasons": supporting,
        "warnings": warnings,
    }


def _apply_chapter_15_major_trendline_filter(
    action: str,
    chapter_15_major_trendlines: dict[str, Any],
) -> dict[str, Any]:
    stock = chapter_15_major_trendlines.get("stock_major_trend", {})
    trendline = stock.get("major_trendline", {})
    regime = stock.get("major_regime", {})
    confirmation = chapter_15_major_trendlines.get("broad_market_confirmation", {})
    scale = stock.get("scale_comparison", {})
    filtered_action = action
    blockers: list[str] = []
    warnings: list[str] = []
    supporting: list[str] = []

    kind = str(trendline.get("kind", ""))
    status = str(trendline.get("status", ""))
    direction = str(trendline.get("direction", ""))
    authority = float(trendline.get("authority_score") or trendline.get("score") or 0.0)
    effective_break = bool(trendline.get("effective_major_break"))
    major_state = str(regime.get("state", "Neutral"))

    if action == "Buy" and kind == "major_uptrend" and effective_break and authority >= 0.50:
        blockers.append("Chapter 15 major uptrend-line break conflicts with an aggressive Buy signal.")
    if action == "Sell" and kind == "major_uptrend" and status == "ActiveMajorTrendline" and major_state == "Bullish" and authority >= 0.55:
        blockers.append("Chapter 15 major bull trendline remains intact, blocking a premature Sell signal.")
    if action == "Sell" and kind == "major_downtrend" and effective_break and authority >= 0.50:
        blockers.append("Chapter 15 major downtrend-line break conflicts with an aggressive Sell signal.")

    conflicting_contexts = confirmation.get("conflicting_contexts", [])
    confirming_contexts = confirmation.get("confirming_contexts", [])
    if action == "Buy" and _context_major_state_count(conflicting_contexts, "Bearish") >= 1:
        blockers.append("Supplied broad-market Chapter 15 context is Bearish while the stock signal is Buy.")
    if action == "Sell" and _context_major_state_count(conflicting_contexts, "Bullish") >= 1:
        blockers.append("Supplied broad-market Chapter 15 context is Bullish while the stock signal is Sell.")

    if action == "Buy" and major_state == "Bullish" and not effective_break:
        supporting.append("Chapter 15 major trend context is Bullish.")
    if action == "Sell" and kind == "major_uptrend" and effective_break:
        supporting.append("Chapter 15 major uptrend-line break supports the Sell signal.")
    if action == "Buy" and kind == "major_downtrend" and effective_break:
        supporting.append("Chapter 15 major downtrend-line break warns that the prior bear trend may be ending.")
    if confirmation.get("status") == "Confirmed" and confirming_contexts:
        supporting.append("Supplied broad-market Chapter 15 context confirms the stock major trend.")

    if confirmation.get("status") == "Unavailable":
        warnings.append("Chapter 15 broad-market confirmation is unavailable because no benchmark/index context was supplied.")
    if scale.get("major_trend_shape") in {"mixed_or_unclear", "decelerating_investment_preferred_style"}:
        warnings.append("Chapter 15 scale/shape diagnostics are not clean enough for high confidence.")
    if trendline.get("major_bear_warning_only"):
        warnings.append("Chapter 15 major bear trendline is warning-only and should not be used alone.")
    if confirmation.get("status") == "Divergent":
        warnings.append("Supplied broad-market Chapter 15 context diverges from the stock major trend.")

    if blockers:
        filtered_action = "Hold"

    return {
        "input_action": action,
        "filtered_action": filtered_action,
        "filter_applied": filtered_action != action,
        "pattern": trendline.get("pattern", "NoChapter15MajorTrendline"),
        "status": status or "NoPattern",
        "direction": direction,
        "kind": kind,
        "major_state": major_state,
        "scale": trendline.get("scale"),
        "authority_score": trendline.get("authority_score"),
        "effective_major_break": effective_break,
        "broad_market_confirmation_status": confirmation.get("status"),
        "blocking_reasons": blockers,
        "supporting_reasons": supporting,
        "warnings": warnings,
    }


def _context_major_state_count(contexts: list[dict[str, Any]], state: str) -> int:
    return sum(1 for item in contexts if str(item.get("state")) == state)


def _decision_diagnostics(
    forecasts: list[dict[str, Any]],
    action: str,
    raw_action: str,
    risk: str,
    dow_action_filter: dict[str, Any],
    magee_action_filter: dict[str, Any],
    reversal_action_filter: dict[str, Any],
    triangle_action_filter: dict[str, Any],
    chapter_9_action_filter: dict[str, Any],
    chapter_10_action_filter: dict[str, Any],
    chapter_11_action_filter: dict[str, Any],
    chapter_12_gap_filter: dict[str, Any],
    chapter_13_zone_filter: dict[str, Any],
    chapter_14_trendline_filter: dict[str, Any],
    chapter_15_major_trendline_filter: dict[str, Any],
    dow_theory: dict[str, Any],
    magee_basing_points: dict[str, Any],
    reversal_patterns: dict[str, Any],
    triangle_patterns: dict[str, Any],
    chapter_9_patterns: dict[str, Any],
    chapter_10_patterns: dict[str, Any],
    chapter_11_patterns: dict[str, Any],
    chapter_12_gaps: dict[str, Any],
    chapter_13_support_resistance: dict[str, Any],
    chapter_14_trendlines: dict[str, Any],
    chapter_15_major_trendlines: dict[str, Any],
) -> dict[str, Any]:
    if not forecasts:
        return {
            "action": action,
            "raw_action": raw_action,
            "hold_reason": "NoForecast",
            "blocking_reasons": ["No forecasts were produced."],
            "supporting_reasons": [],
        }
    preferred = next((forecast for forecast in forecasts if int(forecast["horizon_days"]) == 5), forecasts[0])
    expected_return = float(preferred["expected_return"])
    confidence = float(preferred["directional_confidence"])
    validation_mae = float(preferred["validation_metrics"].get("mae", 0.03))
    threshold = max(0.02, validation_mae)
    edge_ratio = float(abs(expected_return) / threshold) if threshold > 0 else 0.0

    blocking = []
    supporting = []
    hold_reason = None
    if risk == "High":
        blocking.append("Overall risk is High, so the engine blocks directional action.")
        hold_reason = hold_reason or "RiskBlocked"
    if confidence < 0.55:
        blocking.append("Preferred horizon directional confidence is below 55%.")
        hold_reason = hold_reason or "NoEdge"
    if abs(expected_return) <= threshold:
        blocking.append("Preferred horizon expected return does not exceed validation-error threshold.")
        hold_reason = hold_reason or "NoEdge"
    if dow_action_filter.get("filter_applied"):
        blocking.extend(str(reason) for reason in dow_action_filter.get("blocking_reasons", []))
        hold_reason = hold_reason or "RegimeBlocked"
    if magee_action_filter.get("filter_applied"):
        blocking.extend(str(reason) for reason in magee_action_filter.get("blocking_reasons", []))
        hold_reason = hold_reason or _magee_hold_reason(magee_basing_points) or "RegimeBlocked"
    if reversal_action_filter.get("filter_applied"):
        blocking.extend(str(reason) for reason in reversal_action_filter.get("blocking_reasons", []))
        hold_reason = hold_reason or _reversal_hold_reason(reversal_patterns) or "ReversalBlocked"
    if triangle_action_filter.get("filter_applied"):
        blocking.extend(str(reason) for reason in triangle_action_filter.get("blocking_reasons", []))
        hold_reason = hold_reason or _triangle_hold_reason(triangle_patterns) or "TriangleBlocked"
    if chapter_9_action_filter.get("filter_applied"):
        blocking.extend(str(reason) for reason in chapter_9_action_filter.get("blocking_reasons", []))
        hold_reason = hold_reason or _chapter_9_hold_reason(chapter_9_patterns) or "Chapter9Blocked"
    if chapter_10_action_filter.get("filter_applied"):
        blocking.extend(str(reason) for reason in chapter_10_action_filter.get("blocking_reasons", []))
        hold_reason = hold_reason or _chapter_10_hold_reason(chapter_10_patterns) or "Chapter10Blocked"
    if chapter_11_action_filter.get("filter_applied"):
        blocking.extend(str(reason) for reason in chapter_11_action_filter.get("blocking_reasons", []))
        hold_reason = hold_reason or _chapter_11_hold_reason(chapter_11_patterns) or "Chapter11Blocked"
    if chapter_12_gap_filter.get("filter_applied"):
        blocking.extend(str(reason) for reason in chapter_12_gap_filter.get("blocking_reasons", []))
        hold_reason = hold_reason or _chapter_12_hold_reason(chapter_12_gaps) or "Chapter12Blocked"
    if chapter_13_zone_filter.get("filter_applied"):
        blocking.extend(str(reason) for reason in chapter_13_zone_filter.get("blocking_reasons", []))
        hold_reason = hold_reason or _chapter_13_hold_reason(chapter_13_support_resistance) or "Chapter13Blocked"
    if chapter_14_trendline_filter.get("filter_applied"):
        blocking.extend(str(reason) for reason in chapter_14_trendline_filter.get("blocking_reasons", []))
        hold_reason = hold_reason or _chapter_14_hold_reason(chapter_14_trendlines) or "Chapter14Blocked"
    if chapter_15_major_trendline_filter.get("filter_applied"):
        blocking.extend(str(reason) for reason in chapter_15_major_trendline_filter.get("blocking_reasons", []))
        hold_reason = hold_reason or _chapter_15_hold_reason(chapter_15_major_trendlines) or "Chapter15Blocked"
    elif dow_action_filter.get("line_state") == "ActiveLine" and action == "Hold":
        hold_reason = hold_reason or "RangeWait"
    elif dow_action_filter.get("continuation_state") in {"TreatTrendAsDoubt", "WaitForClearConfirmation"} and action == "Hold":
        hold_reason = hold_reason or "TrendDoubt"
    elif action == "Hold":
        hold_reason = hold_reason or _magee_hold_reason(magee_basing_points)
    if action == "Buy" and expected_return > threshold:
        supporting.append("Preferred horizon expected return exceeds validation-error threshold.")
    if action == "Sell" and expected_return < -threshold:
        supporting.append("Preferred horizon expected return is below the negative validation-error threshold.")
    if risk != "High":
        supporting.append(f"Overall risk is {risk}.")
    if confidence >= 0.55:
        supporting.append("Preferred horizon confidence clears the 55% action gate.")
    if action != raw_action:
        supporting.append("The model signal was filtered by the technical-regime gate.")
    supporting.extend(str(reason) for reason in chapter_11_action_filter.get("supporting_reasons", []))
    supporting.extend(str(reason) for reason in chapter_12_gap_filter.get("supporting_reasons", []))
    supporting.extend(str(reason) for reason in chapter_13_zone_filter.get("supporting_reasons", []))
    supporting.extend(str(reason) for reason in chapter_14_trendline_filter.get("supporting_reasons", []))
    supporting.extend(str(reason) for reason in chapter_15_major_trendline_filter.get("supporting_reasons", []))
    if action == "Hold" and hold_reason is None:
        hold_reason = _dow_hold_reason(dow_theory) or "NoEdge"

    return {
        "action": action,
        "raw_action": raw_action,
        "hold_reason": hold_reason if action == "Hold" else None,
        "hold_reason_legend": {
            "NoEdge": "Forecast edge is smaller than validation error or confidence is weak.",
            "RiskBlocked": "Model or validation risk is too high for directional action.",
            "TrendDoubt": "Primary/intermediate technical state is not clear enough.",
            "RangeWait": "Price is inside a sideways line/range and needs a break.",
            "RegimeBlocked": "Forecast direction conflicts with Dow-style regime controls.",
            "BasingStopTooClose": "Price is too close to the Magee basing stop for a fresh directional action.",
            "BelowBasingStop": "Price is below the active Magee basing stop.",
            "LongTermTrendIntact": "Magee long-term trend remains intact above its active stop.",
            "ConfirmedReversalTop": "A confirmed active Head-and-Shoulders Top conflicts with a fresh Buy action.",
            "ConfirmedReversalBottom": "A confirmed active Head-and-Shoulders Bottom conflicts with a fresh Sell action.",
            "ReversalBlocked": "Forecast direction conflicts with confirmed reversal-pattern controls.",
            "TriangleWait": "Price is inside an unbroken Symmetrical Triangle and needs a decisive breakout.",
            "TriangleFailure": "A recent triangle breakout or breakdown failed.",
            "TriangleBreakoutConflict": "Forecast direction conflicts with a confirmed triangle breakout.",
            "TriangleBlocked": "Forecast direction conflicts with triangle-pattern controls.",
            "RectangleWait": "Price is inside an unbroken Chapter 9 rectangle and needs a confirmed break.",
            "RectangleFailure": "A rectangle breakout or breakdown returned inside the range or reversed.",
            "RectangleBreakoutConflict": "Forecast direction conflicts with a confirmed rectangle break.",
            "ConfirmedDoubleTop": "A confirmed Double Top conflicts with a fresh Buy action.",
            "ConfirmedDoubleBottom": "A confirmed Double Bottom conflicts with a fresh Sell action.",
            "ConfirmedTripleTop": "A confirmed Triple Top conflicts with a fresh Buy action.",
            "ConfirmedTripleBottom": "A confirmed Triple Bottom conflicts with a fresh Sell action.",
            "Chapter9Blocked": "Forecast direction conflicts with Chapter 9 pattern controls.",
            "BroadeningTopConflict": "A confirmed Chapter 10 broadening top conflicts with a fresh Buy action.",
            "WedgeBreakConflict": "Forecast direction conflicts with a confirmed wedge break.",
            "DiamondBreakConflict": "Forecast direction conflicts with a confirmed diamond break.",
            "ShortTermExhaustion": "A strong one-day reversal, spike, or selling climax conflicts with the fresh action.",
            "Chapter10Blocked": "Forecast direction conflicts with Chapter 10 reversal phenomena.",
            "ContinuationConflict": "Forecast direction conflicts with a confirmed Chapter 11 continuation.",
            "StaleContinuation": "A continuation pattern has exceeded its normal Chapter 11 reliability window.",
            "FailedContinuation": "A flag or pennant broke against its expected continuation direction.",
            "Chapter11Blocked": "Forecast direction conflicts with Chapter 11 continuation-pattern controls.",
            "ExhaustionGap": "A Chapter 12 exhaustion gap warns that the recent move may have ended.",
            "IslandReversal": "A Chapter 12 Island Reversal conflicts with the fresh directional action.",
            "GapBreakawayConflict": "Forecast direction conflicts with a Chapter 12 breakaway or runaway gap.",
            "Chapter12Blocked": "Forecast direction conflicts with Chapter 12 gap controls.",
            "ResistanceTooClose": "Strong Chapter 13 resistance is too close for a fresh Buy action.",
            "SupportTooClose": "Strong Chapter 13 support is too close for a fresh Sell action.",
            "SupportFailure": "Chapter 13 support failure warns of trend deterioration.",
            "ResistanceBreakout": "Chapter 13 resistance breakout conflicts with a fresh Sell action.",
            "Chapter13Blocked": "Forecast direction conflicts with Chapter 13 support/resistance controls.",
            "UpTrendlineBreak": "Chapter 14 decisive uptrend-line break conflicts with a fresh Buy action.",
            "DownTrendlineBreak": "Chapter 14 decisive downtrend-line break conflicts with a fresh Sell action.",
            "ActiveUpTrendline": "Chapter 14 authoritative uptrend line remains intact near price.",
            "ActiveDownTrendline": "Chapter 14 authoritative downtrend line remains intact near price.",
            "BullishFanBreak": "Chapter 14 bullish third fan-line break conflicts with a fresh Sell action.",
            "BearishFanBreak": "Chapter 14 bearish third fan-line break conflicts with a fresh Buy action.",
            "Chapter14Blocked": "Forecast direction conflicts with Chapter 14 trendline/channel controls.",
            "MajorUpTrendlineBreak": "Chapter 15 major uptrend-line break conflicts with a fresh Buy action.",
            "MajorDownTrendlineBreak": "Chapter 15 major downtrend-line break conflicts with a fresh Sell action.",
            "MajorBullTrendIntact": "Chapter 15 major bull trendline remains intact.",
            "BroadMarketMajorDivergence": "Supplied benchmark/index major trend conflicts with the stock action.",
            "Chapter15Blocked": "Forecast direction conflicts with Chapter 15 major-trendline controls.",
            "NoForecast": "No forecast was produced.",
        },
        "preferred_horizon_days": int(preferred["horizon_days"]),
        "preferred_expected_return": expected_return,
        "preferred_directional_confidence": confidence,
        "validation_error_threshold": threshold,
        "edge_to_error_ratio": edge_ratio,
        "risk_level": risk,
        "dow_action_filter": dow_action_filter,
        "magee_action_filter": magee_action_filter,
        "reversal_action_filter": reversal_action_filter,
        "triangle_action_filter": triangle_action_filter,
        "chapter_9_action_filter": chapter_9_action_filter,
        "chapter_10_action_filter": chapter_10_action_filter,
        "chapter_11_action_filter": chapter_11_action_filter,
        "chapter_12_gap_filter": chapter_12_gap_filter,
        "chapter_13_zone_filter": chapter_13_zone_filter,
        "chapter_14_trendline_filter": chapter_14_trendline_filter,
        "chapter_15_major_trendline_filter": chapter_15_major_trendline_filter,
        "blocking_reasons": blocking,
        "supporting_reasons": supporting,
    }


def _dow_hold_reason(dow_theory: dict[str, Any]) -> str | None:
    if dow_theory.get("line_pattern", {}).get("state") == "ActiveLine":
        return "RangeWait"
    if dow_theory.get("trend_confirmation", {}).get("status") in {"Divergent", "MixedConfirmation"}:
        return "TrendDoubt"
    if dow_theory.get("continuation_rule", {}).get("state") in {"TreatTrendAsDoubt", "WaitForClearConfirmation"}:
        return "TrendDoubt"
    ambiguity = (
        dow_theory.get("chapter_4_defect_diagnostics", {})
        .get("sensitivity_analysis", {})
        .get("ambiguity_score")
    )
    if ambiguity is not None and float(ambiguity) > 0.45:
        return "TrendDoubt"
    return None


def _magee_hold_reason(magee_basing_points: dict[str, Any]) -> str | None:
    preferred = magee_basing_points.get("preferred", {})
    trend_state = preferred.get("trend_state")
    stop_status = preferred.get("stop_status")
    stop_distance = preferred.get("stop_distance_pct")
    if stop_status == "BelowBasingStop" or trend_state == "Short":
        return "BelowBasingStop"
    if stop_distance is not None and 0.0 <= float(stop_distance) < 0.03:
        return "BasingStopTooClose"
    if trend_state == "Long" and stop_status == "AboveBasingStop":
        return "LongTermTrendIntact"
    return None


def _reversal_hold_reason(reversal_patterns: dict[str, Any]) -> str | None:
    preferred = reversal_patterns.get("preferred", {})
    if preferred.get("pattern") == "HeadAndShouldersTop" and preferred.get("status") in {"Confirmed", "PullbackToNeckline"}:
        return "ConfirmedReversalTop"
    if preferred.get("pattern") == "HeadAndShouldersBottom" and preferred.get("status") in {"Confirmed", "ThrowbackToNeckline"}:
        return "ConfirmedReversalBottom"
    return None


def _triangle_hold_reason(triangle_patterns: dict[str, Any]) -> str | None:
    preferred = triangle_patterns.get("preferred", {})
    pattern = preferred.get("pattern")
    status = preferred.get("status")
    if pattern == "SymmetricalTriangle" and status == "Candidate":
        return "TriangleWait"
    if status in {"FailedBreakout", "FailedBreakdown"}:
        return "TriangleFailure"
    if status in {"Breakout", "Breakdown", "Retest"}:
        return "TriangleBreakoutConflict"
    return None


def _chapter_9_hold_reason(chapter_9_patterns: dict[str, Any]) -> str | None:
    rectangle = chapter_9_patterns.get("rectangle_patterns", {}).get("preferred", {})
    rectangle_status = rectangle.get("status")
    if rectangle_status == "Candidate":
        return "RectangleWait"
    if rectangle_status in {"FalseBreakout", "FalseBreakdown", "PrematureBreakout", "PrematureBreakdown"}:
        return "RectangleFailure"
    if rectangle_status in {"Breakout", "Breakdown", "Retest"}:
        return "RectangleBreakoutConflict"
    multi = chapter_9_patterns.get("multi_top_bottom_patterns", {}).get("preferred", {})
    pattern = multi.get("pattern")
    status = multi.get("status")
    if status in {"Confirmed", "PullbackToConfirmation"}:
        if pattern == "DoubleTop":
            return "ConfirmedDoubleTop"
        if pattern == "DoubleBottom":
            return "ConfirmedDoubleBottom"
        if pattern == "TripleTop":
            return "ConfirmedTripleTop"
        if pattern == "TripleBottom":
            return "ConfirmedTripleBottom"
    return None


def _chapter_10_hold_reason(chapter_10_patterns: dict[str, Any]) -> str | None:
    structural = chapter_10_patterns.get("structural_patterns", {}).get("preferred", {})
    pattern = structural.get("pattern")
    status = structural.get("status")
    direction = structural.get("direction")
    if pattern in {"BroadeningTop", "FlatToppedBroadening"} and status in {"Confirmed", "PullbackToBoundary"}:
        return "BroadeningTopConflict"
    if pattern in {"RisingWedge", "FallingWedge"} and status in {"Breakout", "Breakdown", "Retest"}:
        return "WedgeBreakConflict"
    if pattern == "Diamond" and status in {"Breakout", "Breakdown", "Retest"} and direction in {"bullish", "bearish"}:
        return "DiamondBreakConflict"
    event = chapter_10_patterns.get("short_term_events", {}).get("preferred", {})
    if event.get("pattern") in {
        "OneDayReversalTop",
        "OneDayReversalBottom",
        "KeyReversalTop",
        "KeyReversalBottom",
        "SpikeTop",
        "SpikeBottom",
        "SellingClimax",
    }:
        return "ShortTermExhaustion"
    return None


def _chapter_11_hold_reason(chapter_11_patterns: dict[str, Any]) -> str | None:
    continuation = chapter_11_patterns.get("continuation_patterns", {}).get("preferred", {})
    hs = chapter_11_patterns.get("head_and_shoulders_continuation", {}).get("preferred", {})
    candidates = [
        pattern
        for pattern in (continuation, hs)
        if isinstance(pattern, dict) and pattern.get("status") not in {None, "NoPattern", "InsufficientData"}
    ]
    if not candidates:
        return None
    preferred = max(candidates, key=_chapter_11_filter_rank)
    status = preferred.get("status")
    if status in {"FailedBreakout", "FailedBreakdown"}:
        return "FailedContinuation"
    if status == "Stale":
        return "StaleContinuation"
    if status in {"Breakout", "Breakdown", "Confirmed"}:
        return "ContinuationConflict"
    return None


def _chapter_12_hold_reason(chapter_12_gaps: dict[str, Any]) -> str | None:
    gap = chapter_12_gaps.get("classified_gaps", {}).get("preferred", {})
    island = chapter_12_gaps.get("island_reversals", {}).get("preferred", {})
    candidates = [
        pattern
        for pattern in (gap, island)
        if isinstance(pattern, dict) and pattern.get("status") not in {None, "NoPattern", "InsufficientData", "Ignored", "Excluded"}
    ]
    if not candidates:
        return None
    preferred = max(candidates, key=_chapter_12_filter_rank)
    pattern = preferred.get("pattern")
    if pattern == "IslandReversal":
        return "IslandReversal"
    if pattern == "ExhaustionGap":
        return "ExhaustionGap"
    if pattern in {"BreakawayGap", "RunawayGap"}:
        return "GapBreakawayConflict"
    return None


def _chapter_13_hold_reason(chapter_13_support_resistance: dict[str, Any]) -> str | None:
    active = _chapter_13_active_state(chapter_13_support_resistance)
    if active.get("support_failure") and active.get("volume_confirmed"):
        return "SupportFailure"
    if active.get("resistance_breakout") and active.get("volume_confirmed"):
        return "ResistanceBreakout"
    support = chapter_13_support_resistance.get("support_zones", {}).get("nearest", {})
    resistance = chapter_13_support_resistance.get("resistance_zones", {}).get("nearest", {})
    if float(resistance.get("distance_to_zone_pct") or 999.0) <= 0.035 and float(resistance.get("remaining_strength") or 0.0) >= 0.55:
        return "ResistanceTooClose"
    if float(support.get("distance_to_zone_pct") or 999.0) <= 0.035 and float(support.get("remaining_strength") or 0.0) >= 0.55:
        return "SupportTooClose"
    return None


def _chapter_14_hold_reason(chapter_14_trendlines: dict[str, Any]) -> str | None:
    trendline = chapter_14_trendlines.get("trendlines", {}).get("preferred", {})
    fan = chapter_14_trendlines.get("fan_lines", {}).get("preferred", {})
    kind = trendline.get("kind")
    status = trendline.get("status")
    if bool(trendline.get("effective_decisive_break")) and kind == "uptrend":
        return "UpTrendlineBreak"
    if bool(trendline.get("effective_decisive_break")) and kind == "downtrend":
        return "DownTrendlineBreak"
    if status == "Active" and kind == "uptrend":
        return "ActiveUpTrendline"
    if status == "Active" and kind == "downtrend":
        return "ActiveDownTrendline"
    if fan.get("status") == "ThirdFanBreakUpside":
        return "BullishFanBreak"
    if fan.get("status") == "ThirdFanBreakDownside":
        return "BearishFanBreak"
    return None


def _chapter_15_hold_reason(chapter_15_major_trendlines: dict[str, Any]) -> str | None:
    trendline = chapter_15_major_trendlines.get("stock_major_trend", {}).get("major_trendline", {})
    confirmation = chapter_15_major_trendlines.get("broad_market_confirmation", {})
    kind = trendline.get("kind")
    status = trendline.get("status")
    if confirmation.get("status") == "Divergent":
        return "BroadMarketMajorDivergence"
    if bool(trendline.get("effective_major_break")) and kind == "major_uptrend":
        return "MajorUpTrendlineBreak"
    if bool(trendline.get("effective_major_break")) and kind == "major_downtrend":
        return "MajorDownTrendlineBreak"
    if status == "ActiveMajorTrendline" and kind == "major_uptrend":
        return "MajorBullTrendIntact"
    return None


def _portfolio_risk_level(forecasts: list[dict[str, Any]]) -> str:
    levels = [
        risk_level(
            validation_metrics=forecast["validation_metrics"],
            directional_confidence=float(forecast["directional_confidence"]),
        )
        for forecast in forecasts
    ]
    if "High" in levels:
        return "High"
    if "Medium" in levels:
        return "Medium"
    return "Low"


def _risk_warning(risk: str) -> str:
    if risk == "High":
        return "High model or market risk; use as research only and require manual review."
    if risk == "Medium":
        return "Moderate uncertainty; validate against market context and position limits."
    return "Lower validation risk, but forecasts remain uncertain and non-binding."


def _metric_is_better(left: float, right: float, metric: str) -> bool:
    if not np.isfinite(left) or not np.isfinite(right):
        return False
    if metric in HIGHER_IS_BETTER_METRICS:
        return left > right
    return left < right
