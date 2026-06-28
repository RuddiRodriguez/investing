"""Governable automated stock price forecasting engine."""

__all__ = [
    "DailyTradeConfig",
    "ForecastConfig",
    "ForecastingEngine",
    "build_daily_trade_plan",
    "run_forecast",
]


def __getattr__(name: str):
    if name in {"ForecastingEngine", "run_forecast"}:
        from market_forecasting_engine.pipeline import ForecastingEngine, run_forecast

        return {"ForecastingEngine": ForecastingEngine, "run_forecast": run_forecast}[name]
    if name in {"DailyTradeConfig", "build_daily_trade_plan"}:
        from market_forecasting_engine.daily_trade import DailyTradeConfig, build_daily_trade_plan

        return {"DailyTradeConfig": DailyTradeConfig, "build_daily_trade_plan": build_daily_trade_plan}[name]
    if name == "ForecastConfig":
        from market_forecasting_engine.schema import ForecastConfig

        return ForecastConfig
    raise AttributeError(name)
