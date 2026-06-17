from __future__ import annotations

from contextlib import contextmanager
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_forecasting_engine.alpaca_broker import AlpacaPaperBroker
from market_forecasting_engine.alpaca_options_trader import OptionExecutionConfig, build_real_option_trade_plan
from market_forecasting_engine.daily_trade import DailyTradeConfig, build_daily_trade_plan
from market_forecasting_engine.data import normalize_price_frame
from market_forecasting_engine.data_providers import DataRequest, load_prices_with_provider
from market_forecasting_engine.llm_trader.run import openai_client_for_provider, resolve_llm_model, resolve_llm_provider
from market_forecasting_engine.openai_responses import call_response
from market_forecasting_engine.risk_profiles import risk_profile_for_name


DEFAULT_OPTION_TICKER_UNIVERSE = (
    "SPY",
    "QQQ",
    "NVDA",
    "TSLA",
    "AAPL",
    "MSFT",
    "AMZN",
    "META",
    "AMD",
    "PLTR",
)


@dataclass(frozen=True)
class OptionTickerSelectorConfig:
    tickers: tuple[str, ...] = DEFAULT_OPTION_TICKER_UNIVERSE
    provider: str = "alpaca"
    alpaca_data_feed: str = "iex"
    interval: str = "1m"
    lookback_days: int = 20
    forecast_hours: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    risk_profile: str = "aggressive"
    max_training_rows: int = 3500
    min_dte: int = 1
    max_dte: int = 14
    allow_0dte: bool = False
    max_contract_premium: float | None = None
    max_total_debit: float = 1500.0
    risk_budget_pct: float | None = None
    max_position_equity_pct: float = 0.02
    max_spread_pct: float = 0.15
    max_contracts: int = 1
    target_delta: float = 0.45
    max_delta_distance: float = 0.30
    require_greeks: bool = True
    max_theta_edge_ratio: float = 0.75
    max_theta_premium_pct_per_day: float = 0.35
    min_open_interest: int = 0
    enable_market_regime_filter: bool = True
    allow_range_edge_reversal_entry: bool = False
    market_regime_lookback_rows: int = 120
    market_regime_breakout_buffer_pct: float = 0.001
    market_regime_middle_zone_width: float = 0.30
    min_trend_strength_pct: float = 0.003
    enable_impulse_entry: bool = True
    impulse_lookback_bars: int = 12
    min_impulse_move_pct: float = 0.006
    min_impulse_directional_bars: int = 7
    enable_late_entry_filter: bool = True
    max_late_entry_move_pct: float = 0.018
    max_ema_extension_pct: float = 0.010
    exhaustion_reversal_bars: int = 2
    limit_price_offset_pct: float = 0.03
    stop_loss_pct: float = 0.10
    take_profit_pct: float = 0.55
    stop_limit_offset_pct: float = 0.08
    min_selector_score: float = 55.0
    min_abs_forecast_return: float = 0.001
    min_intraday_rows: int = 120
    max_underlying_price: float | None = None
    enable_llm_selection: bool = False
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_use_web: bool = False


def select_option_ticker(
    *,
    broker: AlpacaPaperBroker,
    config: OptionTickerSelectorConfig,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    account = broker.account()
    rows = [
        evaluate_option_ticker(
            broker=broker,
            ticker=ticker,
            account=account,
            config=config,
            now=now,
        )
        for ticker in config.tickers
    ]
    hard_passed = [row for row in rows if row["eligible"]]
    deterministic = sorted(hard_passed, key=lambda row: row["score"], reverse=True)
    llm_decision = None
    selected = deterministic[0] if deterministic else None
    if config.enable_llm_selection and deterministic:
        llm_decision = choose_ticker_with_llm(candidates=deterministic[:8], config=config, now=now)
        selected_ticker = str((llm_decision or {}).get("selected_ticker") or "").upper()
        selected = next((row for row in deterministic if row["ticker"] == selected_ticker), selected)
    if selected is not None and float(selected["score"]) < float(config.min_selector_score):
        selected = None
    return {
        "selected_ticker": None if selected is None else selected["ticker"],
        "selected": selected,
        "checked_at_utc": now.isoformat(),
        "selector_config": _selector_config_payload(config),
        "account": {
            key: account.get(key)
            for key in ("status", "trading_blocked", "account_blocked", "options_buying_power", "buying_power", "cash", "equity")
        },
        "llm_decision": llm_decision,
        "candidates": rows,
        "eligible_candidates": deterministic,
        "blocked_reason": None if selected is not None else "no_ticker_passed_selector_gates",
    }


def evaluate_option_ticker(
    *,
    broker: AlpacaPaperBroker,
    ticker: str,
    account: dict[str, Any],
    config: OptionTickerSelectorConfig,
    now: datetime,
) -> dict[str, Any]:
    ticker = ticker.upper().strip()
    reasons: list[str] = []
    asset = _safe_asset(broker, ticker, reasons)
    prices = _safe_load_prices(ticker=ticker, config=config, now=now, reasons=reasons)
    price_metrics = _price_metrics(prices)
    latest_price = _float_or_none(price_metrics.get("latest_price"))
    if config.max_underlying_price is not None and latest_price is not None and latest_price > float(config.max_underlying_price):
        reasons.append("underlying_price_above_selector_max")
    plan: dict[str, Any] | None = None
    forecast: dict[str, Any] | None = None
    trade_plan: dict[str, Any] | None = None
    if prices is not None and len(prices) >= int(config.min_intraday_rows):
        try:
            plan = build_daily_trade_plan(
                prices,
                DailyTradeConfig(
                    ticker=ticker,
                    interval=config.interval,
                    forecast_hours=config.forecast_hours,
                    minimum_score_to_trade=1.5 if config.risk_profile == "aggressive" else 2.0,
                ),
            )
            forecast = _primary_forecast(plan)
            forecast["account_equity"] = _float_or_none(account.get("equity"))
        except Exception as exc:
            reasons.append(f"forecast_failed:{str(exc)[:160]}")
    else:
        reasons.append("not_enough_intraday_rows")
    if asset and not _asset_has_options(asset):
        reasons.append("underlying_has_no_listed_options")
    if asset and asset.get("tradable") is False:
        reasons.append("underlying_not_tradable")
    if forecast is not None:
        expected_return = abs(_float_or_none(forecast.get("expected_return")) or 0.0)
        if expected_return < float(config.min_abs_forecast_return):
            reasons.append("forecast_edge_below_selector_min")
        try:
            trade_plan = build_real_option_trade_plan(
                broker=broker,
                underlying=ticker,
                underlying_price=float(plan["latest_price"] if plan else forecast.get("spot")),
                forecast=forecast,
                config=_option_execution_config(ticker=ticker, config=config),
                prices=prices,
                now=now,
            )
        except Exception as exc:
            reasons.append(f"option_plan_failed:{str(exc)[:160]}")
    if trade_plan is None or trade_plan.get("action") != "buy_option":
        reasons.append(str((trade_plan or {}).get("reason") or "no_executable_option_plan"))
    liquidity = _option_liquidity_metrics(trade_plan or {})
    score = _selector_score(
        forecast=forecast or {},
        price_metrics=price_metrics,
        trade_plan=trade_plan or {},
        liquidity=liquidity,
        config=config,
    )
    eligible = bool(trade_plan and trade_plan.get("action") == "buy_option" and score >= config.min_selector_score and not _hard_blocking_reasons(reasons))
    return {
        "ticker": ticker,
        "eligible": eligible,
        "score": round(score, 2),
        "reasons": list(dict.fromkeys(reasons)),
        "asset": {key: asset.get(key) for key in ("symbol", "status", "tradable", "attributes")} if asset else None,
        "price_metrics": price_metrics,
        "forecast": forecast,
        "option_trade_plan_summary": _trade_plan_summary(trade_plan or {}),
        "liquidity": liquidity,
    }


def choose_ticker_with_llm(
    *,
    candidates: list[dict[str, Any]],
    config: OptionTickerSelectorConfig,
    now: datetime,
) -> dict[str, Any] | None:
    provider = resolve_llm_provider(config.llm_provider)
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        return {"status": "skipped", "reason": "OPENAI_API_KEY_missing", "provider": provider}
    if provider == "huggingface" and not (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")):
        return {"status": "skipped", "reason": "HF_TOKEN_or_HUGGINGFACE_API_KEY_missing", "provider": provider}
    compact = [
        {
            "ticker": row["ticker"],
            "score": row["score"],
            "forecast_direction": (row.get("forecast") or {}).get("expected_direction"),
            "expected_return": (row.get("forecast") or {}).get("expected_return"),
            "intraday_volatility_pct": (row.get("price_metrics") or {}).get("intraday_volatility_pct"),
            "avg_dollar_volume": (row.get("price_metrics") or {}).get("avg_dollar_volume"),
            "spread_pct": (row.get("liquidity") or {}).get("best_spread_pct"),
            "open_interest": (row.get("liquidity") or {}).get("best_open_interest"),
            "accepted_contracts": (row.get("liquidity") or {}).get("accepted_count"),
            "selected_contract": ((row.get("option_trade_plan_summary") or {}).get("selected_contract") or {}).get("symbol"),
        }
        for row in candidates
    ]
    tools = [{"type": "web_search_preview"}] if config.llm_use_web and provider == "openai" else []
    try:
        _payload, _response, parsed = call_response(
            client=openai_client_for_provider(provider, timeout=60.0),
            provider=provider,
            model=resolve_llm_model(config.llm_model, provider=provider),
            system_message=(
                "You rank already-filtered US option day-trading tickers. "
                "Never select a ticker outside the candidate list. "
                "Do not override broker/liquidity gates. Prefer tight spreads, high liquidity, clean forecast edge, and controlled volatility."
            ),
            user_message=json.dumps(
                {
                    "as_of_utc": now.isoformat(),
                    "risk_profile": config.risk_profile,
                    "candidates": compact,
                    "instruction": "Select exactly one ticker for the paper options agent or return NO_TRADE if none is suitable.",
                },
                indent=2,
                default=str,
            ),
            json_schema={
                "type": "json_schema",
                "name": "option_ticker_selection",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "selected_ticker": {"type": "string"},
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                        "risks": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["selected_ticker", "confidence", "reason", "risks"],
                },
                "strict": True,
            },
            tools=tools,
            usage_context={"process": "option_ticker_selector", "risk_profile": config.risk_profile, "provider": provider},
        )
        parsed["status"] = "ok"
        parsed["provider"] = provider
        return parsed
    except Exception as exc:
        return {"status": "error", "reason": str(exc)[:300]}


def write_selection_report(output_dir: Path, selection: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = selection.get("selected_ticker") or "NO_TRADE"
    path = output_dir / f"option_ticker_selection_{selected}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(selection, indent=2, default=str) + "\n", encoding="utf-8")
    latest = output_dir / "option_ticker_selection_latest.json"
    latest.write_text(json.dumps(selection, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _safe_asset(broker: AlpacaPaperBroker, ticker: str, reasons: list[str]) -> dict[str, Any] | None:
    try:
        return broker._request("GET", f"/v2/assets/{ticker}")
    except Exception as exc:
        reasons.append(f"asset_lookup_failed:{str(exc)[:160]}")
        return None


def _safe_load_prices(*, ticker: str, config: OptionTickerSelectorConfig, now: datetime, reasons: list[str]) -> pd.DataFrame | None:
    start = (now - timedelta(days=int(config.lookback_days))).isoformat().replace("+00:00", "Z")
    try:
        feed = str(config.alpaca_data_feed or "").strip().lower()
        env_feed = feed if config.provider.lower() == "alpaca" and feed and feed != "auto" else None
        with _temporary_env("ALPACA_DATA_FEED", env_feed):
            result = load_prices_with_provider(
                config.provider,
                DataRequest(ticker=ticker, start=start, interval=config.interval, target_column="close"),
                store=None,
                use_cache=False,
                refresh_cache=True,
            )
        frame = normalize_price_frame(result.frame, target_column="close")
        if config.max_training_rows and len(frame) > int(config.max_training_rows):
            frame = frame.tail(int(config.max_training_rows))
        return frame
    except Exception as exc:
        reasons.append(f"price_load_failed:{str(exc)[:160]}")
        return None


@contextmanager
def _temporary_env(name: str, value: str | None):
    old_value = os.environ.get(name)
    if value is not None:
        os.environ[name] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old_value


def _price_metrics(prices: pd.DataFrame | None) -> dict[str, Any]:
    if prices is None or prices.empty:
        return {"rows": 0}
    frame = normalize_price_frame(prices, target_column="close")
    close = frame["close"].astype(float)
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    volume = frame["volume"].astype(float) if "volume" in frame.columns else pd.Series(0.0, index=frame.index)
    dollar_volume = close.reindex(volume.index).astype(float) * volume
    return {
        "rows": int(len(frame)),
        "latest_price": round(float(close.iloc[-1]), 4),
        "intraday_volatility_pct": round(float(returns.tail(390).std() * np.sqrt(min(390, max(len(returns), 1))) * 100.0), 4) if len(returns) else 0.0,
        "latest_return_pct": round(float(returns.iloc[-1] * 100.0), 4) if len(returns) else 0.0,
        "avg_dollar_volume": round(float(dollar_volume.tail(390).mean()), 2) if len(dollar_volume) else 0.0,
        "latest_timestamp": pd.Timestamp(frame.index[-1]).isoformat(),
    }


def _primary_forecast(plan: dict[str, Any]) -> dict[str, Any]:
    forecasts = sorted(plan.get("forecasts", []), key=lambda row: float(row.get("horizon_hours") or 0.0))
    if not forecasts:
        raise RuntimeError("No forecasts were generated.")
    forecast = dict(forecasts[0])
    forecast.setdefault("spot", plan.get("latest_price"))
    forecast.setdefault("forecast_date", forecast.get("forecast_timestamp"))
    expected_return = float(forecast.get("expected_return") or 0.0)
    forecast["expected_direction"] = "Upward" if expected_return > 0 else "Downward" if expected_return < 0 else "Flat"
    return forecast


def _option_execution_config(*, ticker: str, config: OptionTickerSelectorConfig) -> OptionExecutionConfig:
    profile = risk_profile_for_name(config.risk_profile)
    return OptionExecutionConfig(
        underlying=ticker,
        risk_profile=config.risk_profile,
        min_dte=config.min_dte,
        max_dte=config.max_dte,
        allow_0dte=config.allow_0dte,
        max_contract_premium=config.max_contract_premium,
        max_total_debit=config.max_total_debit,
        risk_budget_pct=float(config.risk_budget_pct if config.risk_budget_pct is not None else profile.risk_budget_pct),
        max_position_equity_pct=config.max_position_equity_pct,
        max_spread_pct=config.max_spread_pct,
        max_contracts=config.max_contracts,
        target_delta=config.target_delta,
        max_delta_distance=config.max_delta_distance,
        require_greeks=config.require_greeks,
        max_theta_edge_ratio=config.max_theta_edge_ratio,
        max_theta_premium_pct_per_day=config.max_theta_premium_pct_per_day,
        min_open_interest=config.min_open_interest,
        enable_market_regime_filter=config.enable_market_regime_filter,
        allow_range_edge_reversal_entry=config.allow_range_edge_reversal_entry,
        market_regime_lookback_rows=config.market_regime_lookback_rows,
        market_regime_breakout_buffer_pct=config.market_regime_breakout_buffer_pct,
        market_regime_middle_zone_width=config.market_regime_middle_zone_width,
        min_trend_strength_pct=config.min_trend_strength_pct,
        enable_impulse_entry=config.enable_impulse_entry,
        impulse_lookback_bars=config.impulse_lookback_bars,
        min_impulse_move_pct=config.min_impulse_move_pct,
        min_impulse_directional_bars=config.min_impulse_directional_bars,
        enable_late_entry_filter=config.enable_late_entry_filter,
        max_late_entry_move_pct=config.max_late_entry_move_pct,
        max_ema_extension_pct=config.max_ema_extension_pct,
        exhaustion_reversal_bars=config.exhaustion_reversal_bars,
        limit_price_offset_pct=config.limit_price_offset_pct,
        stop_loss_pct=config.stop_loss_pct,
        take_profit_pct=config.take_profit_pct,
        stop_limit_offset_pct=config.stop_limit_offset_pct,
    )


def _selector_score(
    *,
    forecast: dict[str, Any],
    price_metrics: dict[str, Any],
    trade_plan: dict[str, Any],
    liquidity: dict[str, Any],
    config: OptionTickerSelectorConfig,
) -> float:
    if trade_plan.get("action") != "buy_option":
        return 0.0
    expected_return = abs(_float_or_none(forecast.get("expected_return")) or 0.0)
    spread_pct = _float_or_none(liquidity.get("best_spread_pct"))
    open_interest = _float_or_none(liquidity.get("best_open_interest")) or 0.0
    accepted_count = _float_or_none(liquidity.get("accepted_count")) or 0.0
    intraday_vol = (_float_or_none(price_metrics.get("intraday_volatility_pct")) or 0.0) / 100.0
    avg_dollar_volume = _float_or_none(price_metrics.get("avg_dollar_volume")) or 0.0
    edge_score = min(expected_return / 0.01, 1.0) * 25.0
    spread_score = 0.0 if spread_pct is None else max(0.0, 1.0 - spread_pct / max(config.max_spread_pct, 1e-9)) * 25.0
    oi_score = min(open_interest / 1000.0, 1.0) * 15.0
    candidate_score = min(accepted_count / 5.0, 1.0) * 10.0
    volatility_score = min(intraday_vol / 0.035, 1.0) * 15.0
    volume_score = min(avg_dollar_volume / 10_000_000.0, 1.0) * 10.0
    return float(edge_score + spread_score + oi_score + candidate_score + volatility_score + volume_score)


def _asset_has_options(asset: dict[str, Any]) -> bool:
    attrs = asset.get("attributes") or []
    return "has_options" in attrs if isinstance(attrs, list) else "has_options" in str(attrs)


def _hard_blocking_reasons(reasons: list[str]) -> list[str]:
    prefixes = (
        "asset_lookup_failed",
        "price_load_failed",
        "forecast_failed",
        "option_plan_failed",
    )
    exact = {
        "underlying_has_no_listed_options",
        "underlying_not_tradable",
        "not_enough_intraday_rows",
        "forecast_edge_below_selector_min",
        "no_executable_option_plan",
        "no_tradable_option_contracts",
        "market_regime_blocks_directional_entry",
        "no_contract_passed_execution_gates",
        "position_size_below_one_contract",
        "underlying_price_above_selector_max",
    }
    return [reason for reason in reasons if reason in exact or any(reason.startswith(prefix) for prefix in prefixes)]


def _option_liquidity_metrics(trade_plan: dict[str, Any]) -> dict[str, Any]:
    selected = trade_plan.get("selected_contract") or {}
    top_candidates = trade_plan.get("top_candidates") or []
    accepted = [row for row in top_candidates if row.get("accepted", True)]
    best = selected or (accepted[0] if accepted else {})
    return {
        "accepted_count": int(trade_plan.get("accepted_count") or len(accepted)),
        "candidate_count": int(trade_plan.get("candidate_count") or len(top_candidates)),
        "best_spread_pct": _float_or_none(best.get("spread_pct")),
        "best_open_interest": _float_or_none(best.get("open_interest")),
        "best_bid": _float_or_none(best.get("bid")),
        "best_ask": _float_or_none(best.get("ask")),
        "best_mid": _float_or_none(best.get("mid")),
        "best_premium": _float_or_none(best.get("premium")),
    }


def _trade_plan_summary(trade_plan: dict[str, Any]) -> dict[str, Any]:
    selected = trade_plan.get("selected_contract") or {}
    order = trade_plan.get("order") or {}
    return {
        "action": trade_plan.get("action"),
        "reason": trade_plan.get("reason"),
        "option_type": trade_plan.get("option_type"),
        "market_regime": trade_plan.get("market_regime"),
        "accepted_count": trade_plan.get("accepted_count"),
        "candidate_count": trade_plan.get("candidate_count"),
        "selected_contract": {
            key: selected.get(key)
            for key in ("symbol", "name", "option_type", "expiration_date", "dte", "strike", "bid", "ask", "mid", "spread_pct", "delta", "open_interest", "premium", "score")
        },
        "trade_quality": trade_plan.get("trade_quality") or selected.get("trade_quality"),
        "order": {key: order.get(key) for key in ("symbol", "side", "type", "qty", "limit_price", "time_in_force")},
        "risk": trade_plan.get("risk"),
    }


def _selector_config_payload(config: OptionTickerSelectorConfig) -> dict[str, Any]:
    payload = dict(config.__dict__)
    payload["tickers"] = list(config.tickers)
    payload["forecast_hours"] = list(config.forecast_hours)
    return payload


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed
