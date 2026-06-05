from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd

from market_forecasting_engine.option_ticker_selector import (
    OptionTickerSelectorConfig,
    evaluate_option_ticker,
    select_option_ticker,
)
from market_forecasting_engine.paper_options_agent import apply_option_profile_defaults
from market_forecasting_engine.paper_options_auto_agent import build_parser, _selector_config_from_args


class FakeBroker:
    def account(self) -> dict:
        return {"equity": "100000", "options_buying_power": "100000", "status": "ACTIVE"}

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        ticker = path.rsplit("/", 1)[-1]
        return {"symbol": ticker, "status": "active", "tradable": True, "attributes": ["has_options"]}

    def option_contracts(self, **kwargs) -> list[dict]:
        underlying = kwargs["underlying_symbols"]
        option_type = kwargs.get("option_type") or "call"
        if underlying == "BAD":
            return []
        return [
            {
                "symbol": f"{underlying}260605{'C' if option_type == 'call' else 'P'}00100000",
                "name": f"{underlying} test option",
                "status": "active",
                "tradable": True,
                "expiration_date": "2026-06-05",
                "type": option_type,
                "strike_price": "100",
                "open_interest": "2500" if underlying == "GOOD" else "20",
            }
        ]

    def option_snapshots(self, symbols: list[str]) -> dict:
        symbol = symbols[0]
        is_good = symbol.startswith("GOOD")
        return {
            symbol: {
                "latestQuote": {
                    "bp": 4.95 if is_good else 4.00,
                    "ap": 5.05 if is_good else 4.80,
                },
                "greeks": {"delta": 0.35, "gamma": 0.02, "theta": -0.02, "vega": 0.10},
            }
        }


def test_selector_picks_best_executable_liquid_ticker(monkeypatch) -> None:
    monkeypatch.setattr(
        "market_forecasting_engine.option_ticker_selector.load_prices_with_provider",
        lambda *args, **kwargs: SimpleNamespace(frame=_prices()),
    )
    monkeypatch.setattr(
        "market_forecasting_engine.option_ticker_selector.build_daily_trade_plan",
        lambda prices, config: _plan(config.ticker, expected_return=0.006),
    )

    selection = select_option_ticker(
        broker=FakeBroker(),  # type: ignore[arg-type]
        config=OptionTickerSelectorConfig(
            tickers=("BAD", "WEAK", "GOOD"),
            min_selector_score=40.0,
            max_spread_pct=0.20,
            min_abs_forecast_return=0.001,
            min_open_interest=0,
            min_trend_strength_pct=0.001,
        ),
        now=datetime(2026, 6, 4, 15, 0, tzinfo=UTC),
    )

    assert selection["selected_ticker"] == "GOOD"
    assert selection["selected"]["eligible"] is True
    assert [row["ticker"] for row in selection["eligible_candidates"]] == ["GOOD"]
    assert next(row for row in selection["candidates"] if row["ticker"] == "WEAK")["eligible"] is False


def test_evaluate_ticker_blocks_when_forecast_edge_is_too_small(monkeypatch) -> None:
    monkeypatch.setattr(
        "market_forecasting_engine.option_ticker_selector.load_prices_with_provider",
        lambda *args, **kwargs: SimpleNamespace(frame=_prices()),
    )
    monkeypatch.setattr(
        "market_forecasting_engine.option_ticker_selector.build_daily_trade_plan",
        lambda prices, config: _plan(config.ticker, expected_return=0.0001),
    )

    row = evaluate_option_ticker(
        broker=FakeBroker(),  # type: ignore[arg-type]
        ticker="GOOD",
        account={"equity": "100000"},
        config=OptionTickerSelectorConfig(
            tickers=("GOOD",),
            min_selector_score=40.0,
            max_spread_pct=0.20,
            min_abs_forecast_return=0.001,
        ),
        now=datetime(2026, 6, 4, 15, 0, tzinfo=UTC),
    )

    assert row["eligible"] is False
    assert "forecast_edge_below_selector_min" in row["reasons"]


def test_auto_options_default_horizons_match_eth_style_short_path() -> None:
    args = apply_option_profile_defaults(build_parser().parse_args([]))
    config = _selector_config_from_args(args)

    assert args.forecast_hours == "0.25,0.5,0.75,1"
    assert args.alpaca_data_feed == "iex"
    assert args.close_before_expiry_hours == 12.0
    assert args.target_delta == 0.45
    assert args.max_delta_distance == 0.30
    assert config.forecast_hours == (0.25, 0.5, 0.75, 1.0)
    assert config.alpaca_data_feed == "iex"
    assert config.enable_market_regime_filter is True
    assert config.enable_impulse_entry is True
    assert config.enable_late_entry_filter is True
    assert config.target_delta == 0.45
    assert config.max_delta_distance == 0.30


def _prices() -> pd.DataFrame:
    index = pd.date_range("2026-06-04 13:30:00+00:00", periods=200, freq="1min")
    close = pd.Series([100 + idx * 0.01 for idx in range(len(index))], index=index)
    return pd.DataFrame(
        {
            "open": close - 0.02,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": 1_000_000,
        },
        index=index,
    )


def _plan(ticker: str, *, expected_return: float) -> dict:
    latest_price = 101.99
    return {
        "ticker": ticker,
        "latest_price": latest_price,
        "forecasts": [
            {
                "horizon_hours": 1.0,
                "forecast_timestamp": "2026-06-04T16:00:00+00:00",
                "predicted_price": latest_price * (1.0 + expected_return),
                "expected_return": expected_return,
            }
        ],
    }
