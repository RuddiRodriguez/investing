from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration for one forecast run."""

    ticker: str
    horizons: tuple[int, ...] = (1, 5, 30)
    target_column: str = "close"
    min_training_rows: int = 180
    validation_window: int = 45
    step_size: int = 20
    max_splits: int = 8
    selection_metric: str = "mae"
    confidence_level: float = 0.80
    purge_window: int | None = None
    embargo_window: int = 0
    final_holdout_fraction: float = 0.15
    transaction_cost_bps: float = 5.0
    factor_top_n: int = 25
    random_state: int = 42
    include_lightgbm: bool = True
    include_statistical_models: bool = True
    include_lstm: bool = False
    search_level: str = "fast"
    tuning_mode: str = "fixed"
    optuna_trials: int = 25
    optuna_timeout_seconds: int | None = None
    optuna_inner_splits: int = 3
    optuna_families: tuple[str, ...] = ("lightgbm", "elastic_net", "random_forest", "gradient_boosting")
    tactical_profile: str = "intermediate"
    enable_llm_review: bool = False
    llm_provider: str = "openai"
    llm_model: str | None = None
    llm_temperature: float = 0.0
    llm_reasoning_effort: str = "none"
    llm_timeout_seconds: int = 30
    llm_env_file: str | None = None
    enable_bayesian_heavy: bool = False
    bayesian_mcmc_draws: int = 300
    bayesian_mcmc_tune: int = 300
    forecast_interval: str = "1d"
    forecast_interval_minutes: float | None = None
    model_version: str = "0.1.0"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["horizons"] = list(self.horizons)
        data["optuna_families"] = list(self.optuna_families)
        return data


@dataclass(frozen=True)
class CandidateValidation:
    model_name: str
    model_family: str
    model_parameters: dict[str, Any]
    train_rows: int
    validation_rows: int
    train_start_date: str
    train_end_date: str
    validation_start_date: str
    validation_end_date: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class HorizonForecast:
    horizon_days: int
    forecast_date: str
    selected_model: str
    selected_model_family: str
    selection_metric: str
    expected_log_return: float
    expected_return: float
    predicted_price: float
    lower_price: float
    upper_price: float
    expected_direction: str
    directional_confidence: float
    confidence_interval_method: str
    calibration_sample_size: int
    trade_quality: dict[str, float]
    validation_metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
