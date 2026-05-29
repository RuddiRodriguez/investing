"""Configuration for the standalone advanced live pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = PACKAGE_DIR / ".cache"


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime settings for the advanced stock-signal system."""

    tickers: tuple[str, ...]
    start_date: str = "2018-01-01"
    end_date: str | None = None
    benchmark: str | None = "SPY"
    interval: str = "1d"
    horizons: tuple[int, ...] = (5, 20, 60)
    primary_horizon: int = 20
    min_history_days: int = 260
    train_window_days: int | None = 756
    min_model_rows: int = 120
    max_model_rows: int = 30_000
    buy_threshold: float = 0.03
    sell_threshold: float = -0.04
    min_confidence: float = 0.60
    min_sell_confidence: float = 0.60
    max_uncertainty: float = 0.12
    max_risk_score: float = 0.78
    max_position_size: float = 0.20
    max_sector_weight: float = 0.45
    volatility_target: float = 0.16
    transaction_cost_bps: float = 10.0
    cache_dir: Path = DEFAULT_CACHE_DIR
    cache_ttl_hours: float | None = 24.0
    force_refresh: bool = False
    random_state: int = 7

    def normalized_tickers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(ticker.strip().upper() for ticker in self.tickers if ticker.strip()))

    def all_market_symbols(self) -> tuple[str, ...]:
        symbols = list(self.normalized_tickers())
        if self.benchmark:
            symbols.append(self.benchmark.upper())
        return tuple(dict.fromkeys(symbols))
