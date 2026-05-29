"""Standalone advanced live stock pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import PipelineConfig
from .features import add_forward_targets, build_feature_panel, validate_prices
from .live_data import LiveDataClient, split_assets_and_benchmark
from .model import TabularAlphaModel
from .portfolio import BUY, CASH, HOLD, SELL, PortfolioOptimizer
from .regime import RegimeDetector
from .risk import RiskEngine


@dataclass(frozen=True)
class PipelineResult:
    as_of_date: pd.Timestamp
    regime: str
    decisions: pd.DataFrame
    target_weights: pd.Series
    diagnostics: dict[str, Any]


class AdvancedLivePipeline:
    """Download live data, score stocks, and produce portfolio-ready decisions."""

    def __init__(self, config: PipelineConfig, data_client: LiveDataClient | None = None):
        self.config = config
        self.data_client = data_client or LiveDataClient(config)
        self.regime_detector = RegimeDetector()
        self.risk_engine = RiskEngine(config)
        self.optimizer = PortfolioOptimizer(config)

    def run(self) -> PipelineResult:
        market_prices = self.data_client.fetch_prices()
        fundamentals = self.data_client.fetch_fundamentals()
        news = self.data_client.fetch_news_signals()
        asset_prices, benchmark = split_assets_and_benchmark(market_prices, self.config)
        sectors = _sector_map(fundamentals)
        return self.run_from_frames(asset_prices, benchmark, fundamentals, news, sectors)

    def run_from_frames(
        self,
        prices: pd.DataFrame,
        benchmark: pd.Series | None = None,
        fundamentals: pd.DataFrame | None = None,
        news: pd.DataFrame | None = None,
        sectors: dict[str, str] | None = None,
    ) -> PipelineResult:
        clean = validate_prices(prices, self.config.min_history_days)
        features = build_feature_panel(clean, self.config, benchmark=benchmark, fundamentals=fundamentals, news=news)
        modeled = add_forward_targets(features, clean, self.config, benchmark=benchmark)
        as_of_date = clean.index[-1]
        training = self._training_slice(modeled, as_of_date)
        latest = features.loc[[as_of_date]]

        regime = self.regime_detector.detect(clean, benchmark=benchmark)
        regime_score = self.regime_detector.score(regime)
        weights = self.regime_detector.expert_weights(regime)
        alpha_model = TabularAlphaModel(self.config).fit(training)
        alpha = alpha_model.predict(latest)
        alpha["regime_signal"] = regime_score
        risk = self.risk_engine.score(latest, alpha)
        decisions = self._decide(latest, alpha, risk, regime, weights)
        target_weights = self.optimizer.optimize(decisions, sectors=sectors)
        position_size = pd.Series(decisions.index.get_level_values("ticker"), index=decisions.index).map(target_weights)
        decisions["position_size"] = position_size.fillna(0.0)
        decisions = decisions.sort_values(["decision", "risk_adjusted_score"], ascending=[True, False])

        return PipelineResult(
            as_of_date=as_of_date,
            regime=regime,
            decisions=decisions,
            target_weights=target_weights,
            diagnostics={
                "training_rows": len(training),
                "feature_columns": alpha_model.feature_columns,
                "model_errors": alpha_model.errors,
                "expert_weights": weights,
                "cache_dir": str(self.config.cache_dir),
                "primary_horizon": self.config.primary_horizon,
                "decision_counts": decisions["decision"].value_counts().to_dict(),
                "expected_excess_return_summary": decisions["expected_excess_return"].describe().to_dict(),
                "position_size_summary": decisions["position_size"].describe().to_dict(),
            },
        )

    def _training_slice(self, modeled: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
        cutoff = as_of_date - pd.tseries.offsets.BDay(max(self.config.horizons))
        training = modeled.loc[modeled.index.get_level_values("date") <= cutoff].copy()
        if self.config.train_window_days is not None:
            start = as_of_date - pd.tseries.offsets.BDay(self.config.train_window_days)
            training = training.loc[training.index.get_level_values("date") >= start]
        target = f"target_excess_return_{self.config.primary_horizon}d"
        training = training.dropna(subset=[target])
        if len(training) < self.config.min_model_rows:
            raise ValueError(
                f"Need at least {self.config.min_model_rows} leakage-safe model rows. Found {len(training)}."
            )
        return training

    def _decide(
        self,
        features: pd.DataFrame,
        alpha: pd.DataFrame,
        risk: pd.DataFrame,
        regime: str,
        weights: dict[str, float],
    ) -> pd.DataFrame:
        frame = pd.DataFrame(index=features.index)
        frame["technical_signal"] = features["technical_signal"].fillna(0.0)
        frame["fundamental_signal"] = features["fundamental_signal"].fillna(0.0)
        frame["news_signal"] = features["news_signal"].fillna(0.0)
        frame["graph_signal"] = features["graph_signal"].fillna(0.0)
        frame["tabular_contribution"] = alpha["tabular_signal"] * weights["tabular"]
        frame["technical_contribution"] = frame["technical_signal"] * weights["technical"]
        frame["fundamental_contribution"] = frame["fundamental_signal"] * weights["fundamental"]
        frame["news_contribution"] = frame["news_signal"] * weights["news"]
        frame["graph_contribution"] = frame["graph_signal"] * weights["graph"]
        frame["regime_contribution"] = alpha["regime_signal"] * weights["regime"]
        frame["ensemble_score"] = frame.filter(like="_contribution").sum(axis=1)
        frame["expected_excess_return"] = (
            alpha["expected_model_return"] + frame["ensemble_score"] * 0.015 - risk["risk_penalty"]
        )
        frame["confidence"] = (
            0.45
            + alpha["model_confidence"] * 0.35
            + frame["ensemble_score"].abs().clip(0.0, 1.0) * 0.20
            - risk["risk_score"] * 0.20
        ).clip(0.0, 0.99)
        frame["outperform_probability"] = alpha["outperform_probability"]
        frame = frame.join(risk)
        frame["lower_bound"] = frame["expected_excess_return"] - frame["uncertainty"]
        frame["upper_bound"] = frame["expected_excess_return"] + frame["uncertainty"]
        frame["risk_adjusted_score"] = (
            frame["expected_excess_return"] / frame["expected_volatility"].replace(0.0, pd.NA)
        ).fillna(0.0)
        frame["alpha_score"] = (
            frame["expected_excess_return"] * frame["confidence"] * (1.0 - frame["risk_score"])
        )
        frame["final_score"] = frame["alpha_score"] - frame["uncertainty"] * 0.10
        frame["regime"] = regime
        frame["decision"] = [
            self._action(row)
            for row in frame[
                ["expected_excess_return", "confidence", "uncertainty", "risk_score", "upper_bound", "lower_bound"]
            ].itertuples(
                index=False
            )
        ]
        frame["main_drivers"] = self._drivers(frame)
        frame["main_risks"] = self._risks(features, frame)
        frame["reason_codes"] = self._reason_codes(frame)
        return frame

    def _action(self, row: tuple[float, float, float, float, float, float]) -> str:
        expected, confidence, uncertainty, risk_score, upper_bound, lower_bound = row
        if (
            expected >= self.config.buy_threshold
            and confidence >= self.config.min_confidence
            and uncertainty <= self.config.max_uncertainty
            and risk_score <= self.config.max_risk_score
            and lower_bound > 0.0
        ):
            return BUY
        if risk_score > 0.90:
            return SELL
        if (
            expected <= self.config.sell_threshold
            and confidence >= self.config.min_sell_confidence
            and upper_bound < 0.01
        ):
            return SELL
        return HOLD

    def _drivers(self, frame: pd.DataFrame) -> pd.Series:
        labels = {
            "tabular_contribution": "learned alpha",
            "technical_contribution": "technical momentum",
            "fundamental_contribution": "fundamental quality/value",
            "news_contribution": "positive live-news signal",
            "graph_contribution": "relationship spillover",
            "regime_contribution": "supportive regime",
        }
        rows = []
        for _, row in frame.iterrows():
            selected = [label for column, label in labels.items() if row[column] > 0.02]
            if row["expected_excess_return"] > 0:
                selected.append("positive expected excess return")
            rows.append(selected[:4] or ["no strong positive driver"])
        return pd.Series(rows, index=frame.index)

    def _risks(self, features: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
        rows = []
        for index, row in frame.iterrows():
            selected = []
            if row["risk_score"] > 0.60:
                selected.append("high risk score")
            if row["uncertainty"] > self.config.max_uncertainty:
                selected.append("wide uncertainty range")
            if features.at[index, "drawdown_252d"] < -0.20:
                selected.append("large drawdown")
            if features.at[index, "volatility_60d"] > 0.40:
                selected.append("high realized volatility")
            if row["outperform_probability"] < 0.50:
                selected.append("weak outperformance probability")
            rows.append(selected[:4] or ["normal model and market risk"])
        return pd.Series(rows, index=frame.index)

    def _reason_codes(self, frame: pd.DataFrame) -> pd.Series:
        rows = []
        for _, row in frame.iterrows():
            reasons = []
            if row["decision"] == BUY:
                reasons.append("expected excess return above buy threshold")
                reasons.append("confidence above buy threshold")
            elif row["decision"] == SELL:
                reasons.append("expected excess return below sell threshold")
                if row["confidence"] >= self.config.min_sell_confidence:
                    reasons.append("confidence above sell threshold")
            else:
                reasons.append("signal inside hold zone")

            if row["risk_score"] > 0.60:
                reasons.append("elevated risk score")
            elif row["risk_score"] < 0.40:
                reasons.append("risk score relatively low")

            contribution_labels = {
                "technical_contribution": "technical momentum",
                "fundamental_contribution": "fundamental quality/value",
                "news_contribution": "news sentiment",
                "graph_contribution": "relationship spillover",
                "regime_contribution": "market regime",
                "tabular_contribution": "learned alpha",
            }
            strongest = max(contribution_labels, key=lambda column: abs(float(row[column])))
            reasons.append(f"largest contribution: {contribution_labels[strongest]}")
            rows.append(reasons)
        return pd.Series(rows, index=frame.index)


def result_records(result: PipelineResult) -> list[dict[str, Any]]:
    output = []
    for row in result.decisions.reset_index().to_dict(orient="records"):
        output.append(
            {
                "ticker": row["ticker"],
                "decision": row["decision"],
                "confidence": round(float(row["confidence"]), 4),
                "horizon": f"{result.diagnostics.get('primary_horizon', 'primary')} trading days",
                "expected_excess_return": round(float(row["expected_excess_return"]), 6),
                "lower_bound": round(float(row["lower_bound"]), 6),
                "upper_bound": round(float(row["upper_bound"]), 6),
                "expected_volatility": round(float(row["expected_volatility"]), 6),
                "risk_adjusted_score": round(float(row["risk_adjusted_score"]), 4),
                "alpha_score": round(float(row["alpha_score"]), 6),
                "final_score": round(float(row["final_score"]), 6),
                "regime": result.regime,
                "position_size": round(float(row["position_size"]), 6),
                "risk_score": round(float(row["risk_score"]), 4),
                "uncertainty": round(float(row["uncertainty"]), 6),
                "technical_signal": round(float(row["technical_signal"]), 6),
                "fundamental_signal": round(float(row["fundamental_signal"]), 6),
                "news_signal": round(float(row["news_signal"]), 6),
                "graph_signal": round(float(row["graph_signal"]), 6),
                "regime_signal": round(float(row["regime_contribution"]), 6),
                "risk_penalty": round(float(row["risk_penalty"]), 6),
                "main_positive_drivers": row["main_drivers"],
                "main_risks": row["main_risks"],
                "reason_codes": row["reason_codes"],
            }
        )
    return output


def _sector_map(fundamentals: pd.DataFrame | None) -> dict[str, str] | None:
    if fundamentals is None or fundamentals.empty or "sector" not in fundamentals.columns:
        return None
    sectors = fundamentals["sector"].dropna()
    if sectors.empty:
        return None
    return {str(ticker).upper(): str(sector) for ticker, sector in sectors.items()}
