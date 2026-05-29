"""Function-based paper-trading simulator for the advanced pipeline."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .live_data import LiveDataClient, split_assets_and_benchmark
from .pipeline import AdvancedLivePipeline, result_records


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "simulation_runs" / "advanced_pipeline"
TRADING_DAYS_PER_MONTH = 21
STATE_FILE = "agent_state.json"
DEFAULT_PORTFOLIO_SYMBOL_MAP = {
    "US67066G1040": "NVDA",
    "NL0010273215": "ASML",
    "US0378331005": "AAPL",
    "US11135F1012": "AVGO",
    "US5324571083": "LLY",
    "US30231G1022": "XOM",
    "GB00BP6MXD84": "SHEL",
    "US8740391003": "TSM",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "IE00BK5BQT80": "VWCE.DE",
    "IE00B4L5Y983": "EUNL.DE",
    "IE00BYZK4552": "2B76.DE",
    "IE00B1XNHC34": "IQQH.DE",
    "LU1681048804": "500.PA",
}


def build_config(
    tickers: tuple[str, ...],
    benchmark: str | None,
    cache_ttl_hours: float | None,
    force_refresh: bool,
) -> PipelineConfig:
    return PipelineConfig(
        tickers=tickers,
        benchmark=benchmark,
        cache_ttl_hours=cache_ttl_hours,
        force_refresh=force_refresh,
    )


def fetch_market_inputs(config: PipelineConfig, include_side_data: bool) -> dict[str, Any]:
    client = LiveDataClient(config)
    market_prices = client.fetch_prices()
    asset_prices, benchmark = split_assets_and_benchmark(market_prices, config)
    fundamentals = client.fetch_fundamentals() if include_side_data else None
    news = client.fetch_news_signals() if include_side_data else None
    sectors = None
    if fundamentals is not None and not fundamentals.empty and "sector" in fundamentals.columns:
        sectors = {
            str(ticker).upper(): str(sector)
            for ticker, sector in fundamentals["sector"].dropna().items()
        }
    return {
        "asset_prices": asset_prices,
        "benchmark": benchmark,
        "fundamentals": fundamentals,
        "news": news,
        "sectors": sectors,
    }


def choose_start_index(
    price_index: pd.DatetimeIndex,
    min_history_days: int,
    simulation_days: int,
    seed: int,
    start_date: str | None = None,
    require_full_horizon: bool = True,
) -> int:
    min_start = min_history_days
    max_start = len(price_index) - simulation_days - 1 if require_full_horizon else len(price_index) - 1
    if max_start < min_start:
        raise ValueError("Not enough history to simulate a full month with the requested warm-up period.")

    if start_date is not None:
        start_timestamp = pd.Timestamp(start_date)
        candidate_positions = np.where(price_index <= start_timestamp)[0]
        if len(candidate_positions) == 0:
            raise ValueError(
                f"Start date {start_date} is earlier than the first available trading day {price_index.min().date()}."
            )
        selected = int(candidate_positions[-1])
        if selected < min_start or selected > max_start:
            if selected < min_start:
                raise ValueError(
                    "Requested start date does not leave enough warm-up history. "
                    f"Earliest valid start date is {price_index[min_start].date()}."
                )
            raise ValueError(
                "Requested start date does not leave enough future trading days for the requested simulation horizon. "
                f"Latest valid start date for a {simulation_days}-day historical run is {price_index[max_start].date()}. "
                "Use --continuous to start from the latest available trading day and wait for new bars."
            )
        return selected

    generator = np.random.default_rng(seed)
    return int(generator.integers(min_start, max_start + 1))


def make_run_directory(run_root: Path, run_id: str | None, resume: bool = False) -> Path:
    resolved_root = Path(run_root)
    resolved_root.mkdir(parents=True, exist_ok=True)
    final_run_id = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = resolved_root / final_run_id
    if run_dir.exists():
        if resume:
            return run_dir
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_symbol_map(raw_mapping: str | None) -> dict[str, str]:
    mapping = dict(DEFAULT_PORTFOLIO_SYMBOL_MAP)
    if not raw_mapping:
        return mapping

    for item in raw_mapping.split(","):
        pair = item.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"Invalid symbol mapping '{pair}'. Use SOURCE=TARGET format.")
        source, target = pair.split("=", 1)
        source_key = source.strip().upper()
        target_value = target.strip().upper()
        if not source_key or not target_value:
            raise ValueError(f"Invalid symbol mapping '{pair}'. Use SOURCE=TARGET format.")
        mapping[source_key] = target_value
    return mapping


def initialize_state(tickers: list[str], initial_cash: float) -> dict[str, Any]:
    return {
        "cash": float(initial_cash),
        "units": pd.Series(0.0, index=tickers, dtype=float),
        "avg_cost": pd.Series(np.nan, index=tickers, dtype=float),
        "entry_date": pd.Series(pd.NaT, index=tickers, dtype="datetime64[ns]"),
        "realized_pnl": 0.0,
    }


def resolve_portfolio_ticker(
    symbol: Any,
    name: Any,
    tickers: list[str],
    symbol_map: dict[str, str],
) -> str | None:
    ticker_set = {ticker.upper() for ticker in tickers}
    candidates = [str(value).strip().upper() for value in (symbol, name) if pd.notna(value) and str(value).strip()]
    for candidate in candidates:
        if candidate in symbol_map:
            mapped = symbol_map[candidate]
            return mapped if mapped in ticker_set else None
        if candidate in ticker_set:
            return candidate
    return None


def build_initial_state_from_transactions(
    transactions_path: Path,
    tickers: list[str],
    as_of_date: str | None,
    symbol_map: dict[str, str],
) -> tuple[dict[str, Any], pd.Timestamp, dict[str, Any]]:
    symbol_lookup = {**DEFAULT_PORTFOLIO_SYMBOL_MAP, **symbol_map}
    transactions = pd.read_csv(transactions_path)
    if transactions.empty:
        raise ValueError(f"Transaction export is empty: {transactions_path}")

    transactions["date"] = pd.to_datetime(transactions["date"], errors="coerce").dt.normalize()
    transactions["datetime"] = pd.to_datetime(transactions.get("datetime"), errors="coerce", utc=True)
    if as_of_date is None:
        portfolio_as_of = transactions["date"].dropna().max()
    else:
        portfolio_as_of = pd.Timestamp(as_of_date).normalize()

    filtered = transactions.loc[transactions["date"].notna() & (transactions["date"] <= portfolio_as_of)].copy()
    if filtered.empty:
        raise ValueError(
            f"No transactions are available on or before portfolio as-of date {portfolio_as_of.date()}."
        )

    filtered.sort_values(["date", "datetime", "transaction_id"], inplace=True, na_position="last")
    for column in ["shares", "amount", "fee", "tax"]:
        filtered[column] = pd.to_numeric(filtered.get(column), errors="coerce").fillna(0.0)
    filtered["cash_effect"] = filtered["amount"] + filtered["fee"] + filtered["tax"]

    state = initialize_state(tickers, initial_cash=0.0)
    state["cash"] = float(filtered["cash_effect"].sum())

    excluded_assets: dict[str, dict[str, Any]] = {}
    tracked_assets: dict[str, dict[str, Any]] = {}
    trading_rows = filtered.loc[filtered["category"].eq("TRADING") & filtered["type"].isin(["BUY", "SELL"])]
    for _, row in trading_rows.iterrows():
        resolved_ticker = resolve_portfolio_ticker(row.get("symbol"), row.get("name"), tickers, symbol_lookup)
        share_delta = float(row["shares"]) if str(row["type"]).upper() == "BUY" else -float(row["shares"])
        cash_effect = float(row["cash_effect"])
        if resolved_ticker is None:
            excluded_key = str(row.get("symbol") or row.get("name") or "UNKNOWN").strip().upper()
            excluded = excluded_assets.setdefault(
                excluded_key,
                {
                    "source_symbol": None if pd.isna(row.get("symbol")) else str(row.get("symbol")),
                    "name": None if pd.isna(row.get("name")) else str(row.get("name")),
                    "shares": 0.0,
                    "cash_effect": 0.0,
                },
            )
            excluded["shares"] += share_delta
            excluded["cash_effect"] += cash_effect
            continue

        tracked = tracked_assets.setdefault(
            resolved_ticker,
            {
                "source_symbol": None if pd.isna(row.get("symbol")) else str(row.get("symbol")),
                "name": None if pd.isna(row.get("name")) else str(row.get("name")),
                "simulation_ticker": resolved_ticker,
                "shares": 0.0,
                "cash_effect": 0.0,
            },
        )
        tracked["shares"] += share_delta
        tracked["cash_effect"] += cash_effect

        current_units = float(state["units"][resolved_ticker])
        new_units = current_units + share_delta
        if str(row["type"]).upper() == "BUY":
            current_cost = 0.0 if np.isnan(state["avg_cost"][resolved_ticker]) else current_units * float(state["avg_cost"][resolved_ticker])
            incremental_cost = -cash_effect
            state["units"][resolved_ticker] = new_units
            state["avg_cost"][resolved_ticker] = (current_cost + incremental_cost) / new_units if new_units > 0 else np.nan
            if pd.isna(state["entry_date"][resolved_ticker]) or current_units <= 0:
                state["entry_date"][resolved_ticker] = row["date"]
        else:
            state["units"][resolved_ticker] = max(0.0, new_units)
            if state["units"][resolved_ticker] <= 1e-10:
                state["units"][resolved_ticker] = 0.0
                state["avg_cost"][resolved_ticker] = np.nan
                state["entry_date"][resolved_ticker] = pd.NaT

    portfolio_seed_snapshot = [
        {
            "asset_scope": "cash",
            "tracked_by_simulation": True,
            "simulation_ticker": "CASH",
            "source_symbol": "EUR",
            "name": "Cash",
            "shares": float(state["cash"]),
            "avg_cost": 1.0,
            "entry_date": None,
            "cash_effect": float(state["cash"]),
            "portfolio_as_of_date": str(portfolio_as_of.date()),
        }
    ]
    for ticker, tracked in tracked_assets.items():
        units = float(state["units"][ticker])
        if abs(units) <= 1e-12:
            continue
        portfolio_seed_snapshot.append(
            {
                "asset_scope": "tracked",
                "tracked_by_simulation": True,
                "simulation_ticker": ticker,
                "source_symbol": tracked["source_symbol"],
                "name": tracked["name"],
                "shares": units,
                "avg_cost": None if np.isnan(state["avg_cost"][ticker]) else float(state["avg_cost"][ticker]),
                "entry_date": None if pd.isna(state["entry_date"][ticker]) else str(pd.Timestamp(state["entry_date"][ticker]).date()),
                "cash_effect": float(tracked["cash_effect"]),
                "portfolio_as_of_date": str(portfolio_as_of.date()),
            }
        )
    for excluded in excluded_assets.values():
        if abs(float(excluded["shares"])) <= 1e-12:
            continue
        portfolio_seed_snapshot.append(
            {
                "asset_scope": "excluded",
                "tracked_by_simulation": False,
                "simulation_ticker": None,
                "source_symbol": excluded["source_symbol"],
                "name": excluded["name"],
                "shares": float(excluded["shares"]),
                "avg_cost": None,
                "entry_date": None,
                "cash_effect": float(excluded["cash_effect"]),
                "portfolio_as_of_date": str(portfolio_as_of.date()),
            }
        )

    metadata = {
        "portfolio_source": str(transactions_path),
        "portfolio_as_of_date": str(portfolio_as_of.date()),
        "portfolio_seed_cash": float(state["cash"]),
        "portfolio_seed_positions": {
            ticker: {
                "units": float(state["units"][ticker]),
                "avg_cost": None if np.isnan(state["avg_cost"][ticker]) else float(state["avg_cost"][ticker]),
                "entry_date": None if pd.isna(state["entry_date"][ticker]) else str(pd.Timestamp(state["entry_date"][ticker]).date()),
            }
            for ticker in tickers
            if abs(float(state["units"][ticker])) > 1e-12
        },
        "excluded_portfolio_assets": list(excluded_assets.values()),
        "portfolio_seed_snapshot": portfolio_seed_snapshot,
    }
    return state, portfolio_as_of, metadata


def serialize_state(state: dict[str, Any], benchmark_value: float) -> dict[str, Any]:
    return {
        "cash": float(state["cash"]),
        "realized_pnl": float(state["realized_pnl"]),
        "benchmark_value": float(benchmark_value),
        "units": {ticker: float(value) for ticker, value in state["units"].items()},
        "avg_cost": {
            ticker: (None if pd.isna(value) else float(value))
            for ticker, value in state["avg_cost"].items()
        },
        "entry_date": {
            ticker: (None if pd.isna(value) else pd.Timestamp(value).isoformat())
            for ticker, value in state["entry_date"].items()
        },
    }


def deserialize_state(
    payload: dict[str, Any] | None,
    tickers: list[str],
    initial_cash: float,
    default_state: dict[str, Any] | None = None,
    default_benchmark_value: float | None = None,
) -> tuple[dict[str, Any], float]:
    if not payload:
        fallback_state = default_state if default_state is not None else initialize_state(tickers, initial_cash)
        fallback_benchmark = default_benchmark_value if default_benchmark_value is not None else float(initial_cash)
        return fallback_state, fallback_benchmark

    state = initialize_state(tickers, initial_cash)
    state["cash"] = float(payload.get("cash", initial_cash))
    state["realized_pnl"] = float(payload.get("realized_pnl", 0.0))
    state["units"] = pd.Series(payload.get("units", {}), dtype=float).reindex(tickers).fillna(0.0)
    avg_cost = pd.Series(payload.get("avg_cost", {}), dtype=object).reindex(tickers)
    state["avg_cost"] = pd.to_numeric(avg_cost, errors="coerce")
    entry_date = pd.Series(payload.get("entry_date", {}), dtype=object).reindex(tickers)
    state["entry_date"] = pd.to_datetime(entry_date, errors="coerce")
    benchmark_fallback = default_benchmark_value if default_benchmark_value is not None else float(initial_cash)
    return state, float(payload.get("benchmark_value", benchmark_fallback))


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_frames(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def _read_csv(name: str) -> pd.DataFrame:
        path = run_dir / name
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    return (
        _read_csv("daily_summary.csv"),
        _read_csv("trades.csv"),
        _read_csv("positions.csv"),
        _read_csv("decision_snapshots.csv"),
    )


def persist_state(run_dir: Path, state: dict[str, Any], benchmark_value: float) -> None:
    (run_dir / STATE_FILE).write_text(
        json.dumps(serialize_state(state, benchmark_value), indent=2, default=str),
        encoding="utf-8",
    )


def load_state(
    run_dir: Path,
    tickers: list[str],
    initial_cash: float,
    default_state: dict[str, Any] | None = None,
    default_benchmark_value: float | None = None,
) -> tuple[dict[str, Any], float]:
    payload = load_json_file(run_dir / STATE_FILE)
    return deserialize_state(payload, tickers, initial_cash, default_state, default_benchmark_value)


def portfolio_value(state: dict[str, Any], prices: pd.Series) -> float:
    return float(state["cash"] + (state["units"] * prices).sum())


def current_weights(state: dict[str, Any], prices: pd.Series) -> pd.Series:
    total_value = portfolio_value(state, prices)
    if total_value <= 0:
        weights = pd.Series(0.0, index=list(prices.index) + ["CASH"], dtype=float)
        weights["CASH"] = 1.0
        return weights

    asset_weights = (state["units"] * prices) / total_value
    weights = asset_weights.reindex(prices.index).fillna(0.0)
    weights["CASH"] = state["cash"] / total_value
    return weights


def normalize_target_weights(target_weights: pd.Series, tickers: list[str]) -> pd.Series:
    normalized = pd.Series(0.0, index=tickers + ["CASH"], dtype=float)
    if target_weights is None or target_weights.empty:
        normalized["CASH"] = 1.0
        return normalized

    upper = {str(index).upper(): float(value) for index, value in target_weights.items()}
    for ticker in tickers:
        normalized[ticker] = upper.get(ticker, 0.0)
    normalized["CASH"] = upper.get("CASH", max(0.0, 1.0 - normalized[tickers].sum()))
    total = normalized.sum()
    if total <= 0:
        normalized["CASH"] = 1.0
        return normalized
    return normalized / total


def run_pipeline_signal(
    history_prices: pd.DataFrame,
    history_benchmark: pd.Series | None,
    config: PipelineConfig,
    fundamentals: pd.DataFrame | None,
    news: pd.DataFrame | None,
    sectors: dict[str, str] | None,
) -> dict[str, Any]:
    result = AdvancedLivePipeline(config).run_from_frames(
        prices=history_prices,
        benchmark=history_benchmark,
        fundamentals=fundamentals,
        news=news,
        sectors=sectors,
    )
    return {
        "as_of_date": result.as_of_date,
        "regime": result.regime,
        "decisions": result.decisions,
        "target_weights": result.target_weights,
        "records": result_records(result),
        "diagnostics": result.diagnostics,
    }


def rebalance_state(
    state: dict[str, Any],
    desired_weights: pd.Series,
    prices_today: pd.Series,
    signal_date: pd.Timestamp,
) -> tuple[dict[str, Any], pd.DataFrame]:
    total_value = portfolio_value(state, prices_today)
    desired_asset_values = desired_weights.reindex(prices_today.index).fillna(0.0) * total_value
    trade_rows: list[dict[str, Any]] = []

    for ticker in prices_today.index:
        price = float(prices_today[ticker])
        current_units = float(state["units"].get(ticker, 0.0))
        current_value = current_units * price
        target_value = float(desired_asset_values[ticker])
        delta_value = target_value - current_value
        delta_units = delta_value / price if price > 0 else 0.0
        action = "HOLD"
        realized_pnl = 0.0

        if delta_value > 1e-8:
            action = "BUY"
            state["cash"] -= delta_value
            new_units = current_units + delta_units
            current_cost = 0.0 if np.isnan(state["avg_cost"][ticker]) else current_units * float(state["avg_cost"][ticker])
            state["avg_cost"][ticker] = (current_cost + delta_value) / new_units if new_units > 0 else np.nan
            if pd.isna(state["entry_date"][ticker]) or current_units <= 0:
                state["entry_date"][ticker] = signal_date
            state["units"][ticker] = new_units
        elif delta_value < -1e-8:
            action = "SELL"
            sell_units = min(current_units, -delta_units)
            average_cost = float(state["avg_cost"][ticker]) if not np.isnan(state["avg_cost"][ticker]) else price
            realized_pnl = sell_units * (price - average_cost)
            state["realized_pnl"] += realized_pnl
            state["cash"] -= delta_value
            new_units = max(0.0, current_units + delta_units)
            state["units"][ticker] = new_units
            if new_units <= 1e-10:
                state["units"][ticker] = 0.0
                state["avg_cost"][ticker] = np.nan
                state["entry_date"][ticker] = pd.NaT

        trade_rows.append(
            {
                "signal_date": signal_date,
                "ticker": ticker,
                "action": action,
                "price": price,
                "trade_value": delta_value,
                "trade_units": delta_units,
                "realized_pnl": realized_pnl,
                "post_trade_units": float(state["units"][ticker]),
                "post_trade_cash": float(state["cash"]),
            }
        )

    return state, pd.DataFrame(trade_rows)


def build_positions_snapshot(
    state: dict[str, Any],
    prices_today: pd.Series,
    signal_date: pd.Timestamp,
    decisions: pd.DataFrame,
    desired_weights: pd.Series,
) -> pd.DataFrame:
    total_value = portfolio_value(state, prices_today)
    rows = []
    for ticker in prices_today.index:
        units = float(state["units"][ticker])
        price = float(prices_today[ticker])
        average_cost = float(state["avg_cost"][ticker]) if not np.isnan(state["avg_cost"][ticker]) else np.nan
        market_value = units * price
        weight = market_value / total_value if total_value > 0 else 0.0
        unrealized_pnl = (price - average_cost) * units if units > 0 and not np.isnan(average_cost) else 0.0
        entry_date = state["entry_date"][ticker]
        holding_days = int((signal_date - entry_date).days) if pd.notna(entry_date) else 0
        decision = decisions.xs(ticker, level="ticker").iloc[0]["decision"]
        rows.append(
            {
                "signal_date": signal_date,
                "ticker": ticker,
                "decision": decision,
                "price": price,
                "units": units,
                "average_cost": average_cost,
                "market_value": market_value,
                "portfolio_weight": weight,
                "target_weight": float(desired_weights.get(ticker, 0.0)),
                "unrealized_pnl": unrealized_pnl,
                "entry_date": entry_date,
                "holding_days": holding_days,
            }
        )
    rows.append(
        {
            "signal_date": signal_date,
            "ticker": "CASH",
            "decision": "CASH",
            "price": 1.0,
            "units": float(state["cash"]),
            "average_cost": 1.0,
            "market_value": float(state["cash"]),
            "portfolio_weight": float(state["cash"] / total_value) if total_value > 0 else 1.0,
            "target_weight": float(desired_weights.get("CASH", 0.0)),
            "unrealized_pnl": 0.0,
            "entry_date": signal_date,
            "holding_days": 0,
        }
    )
    return pd.DataFrame(rows)


def update_state_to_next_close(state: dict[str, Any], next_prices: pd.Series) -> dict[str, Any]:
    state["units"] = state["units"].reindex(next_prices.index).fillna(0.0)
    return state


def build_daily_row(
    signal_date: pd.Timestamp,
    next_date: pd.Timestamp,
    regime: str,
    start_value: float,
    end_value: float,
    benchmark_value: float,
    next_benchmark_value: float,
    decisions: pd.DataFrame,
    state: dict[str, Any],
    target_weights: pd.Series,
) -> dict[str, Any]:
    daily_return = end_value / start_value - 1.0 if start_value > 0 else 0.0
    benchmark_return = next_benchmark_value / benchmark_value - 1.0 if benchmark_value > 0 else 0.0
    return {
        "signal_date": signal_date,
        "next_date": next_date,
        "regime": regime,
        "portfolio_value_start": start_value,
        "portfolio_value_end": end_value,
        "daily_pnl": end_value - start_value,
        "daily_return": daily_return,
        "benchmark_value_start": benchmark_value,
        "benchmark_value_end": next_benchmark_value,
        "benchmark_return": benchmark_return,
        "cash": float(state["cash"]),
        "cash_weight": float(target_weights.get("CASH", 0.0)),
        "gross_exposure": float(target_weights.drop(labels=["CASH"], errors="ignore").sum()),
        "buy_count": int((decisions["decision"] == "BUY").sum()),
        "hold_count": int((decisions["decision"] == "HOLD").sum()),
        "sell_count": int((decisions["decision"] == "SELL").sum()),
        "realized_pnl_cumulative": float(state["realized_pnl"]),
    }


def summarize_run(metadata: dict[str, Any], daily: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    if daily.empty:
        return metadata | {"status": metadata.get("status", "created")}
    equity = daily["portfolio_value_end"]
    benchmark = daily["benchmark_value_end"]
    drawdown = equity / equity.cummax() - 1.0
    return metadata | {
        "status": metadata.get("status", "completed"),
        "completed_steps": int(len(daily)),
        "final_portfolio_value": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 0 else 0.0,
        "benchmark_return": float(benchmark.iloc[-1] / benchmark.iloc[0] - 1.0) if len(benchmark) > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((daily["daily_return"] > 0).mean()),
        "trade_count": int((trades["action"] != "HOLD").sum()) if not trades.empty else 0,
    }


def write_run_artifacts(
    run_dir: Path,
    metadata: dict[str, Any],
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    decisions: pd.DataFrame,
) -> None:
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    daily.to_csv(run_dir / "daily_summary.csv", index=False)
    trades.to_csv(run_dir / "trades.csv", index=False)
    positions.to_csv(run_dir / "positions.csv", index=False)
    decisions.to_csv(run_dir / "decision_snapshots.csv", index=False)
    portfolio_seed_snapshot = metadata.get("portfolio_seed_snapshot")
    if portfolio_seed_snapshot:
        pd.DataFrame(portfolio_seed_snapshot).to_csv(run_dir / "portfolio_seed_snapshot.csv", index=False)
    summary = summarize_run(metadata, daily, trades)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def initialize_metadata(
    prices: pd.DataFrame,
    config: PipelineConfig,
    tickers: list[str],
    initial_cash: float,
    simulation_days: int,
    seed: int,
    start_index: int,
    continuous: bool,
    poll_seconds: float,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    end_index = len(prices.index) - 2 if continuous else start_index + simulation_days - 1
    metadata = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tickers": tickers,
        "benchmark": config.benchmark,
        "initial_cash": initial_cash,
        "simulation_days": simulation_days,
        "seed": seed,
        "start_index": int(start_index),
        "start_date": str(prices.index[start_index].date()),
        "current_index": int(start_index),
        "end_index": int(end_index),
        "continuous": bool(continuous),
        "poll_seconds": float(poll_seconds),
        "data_refresh": "fresh_on_every_poll" if continuous else "fresh_on_every_run",
        "signal_frequency": "daily_close",
        "decision_frequency": "once_per_trading_day",
        "rebalance_frequency": "once_per_trading_day",
        "execution_timing": "rebalance_at_signal_close_mark_to_next_close",
        "dashboard_refresh_seconds": 10,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


def execute_step(
    current_index: int,
    prices: pd.DataFrame,
    benchmark: pd.Series | None,
    fundamentals: pd.DataFrame | None,
    news: pd.DataFrame | None,
    sectors: dict[str, str] | None,
    config: PipelineConfig,
    state: dict[str, Any],
    benchmark_value: float,
    signal_fn,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    signal_date = prices.index[current_index]
    next_date = prices.index[current_index + 1]
    history_prices = prices.iloc[: current_index + 1]
    history_benchmark = benchmark.iloc[: current_index + 1] if benchmark is not None else None
    prices_today = prices.iloc[current_index]
    prices_next = prices.iloc[current_index + 1]
    benchmark_today = float(benchmark.iloc[current_index]) if benchmark is not None else benchmark_value
    benchmark_next = float(benchmark.iloc[current_index + 1]) if benchmark is not None else benchmark_value

    signal_output = signal_fn(history_prices, history_benchmark, config, fundamentals, news, sectors)
    desired_weights = normalize_target_weights(signal_output["target_weights"], list(prices.columns))
    start_value = portfolio_value(state, prices_today)
    state, trade_frame = rebalance_state(state, desired_weights, prices_today, signal_date)
    positions_frame = build_positions_snapshot(state, prices_today, signal_date, signal_output["decisions"], desired_weights)
    end_value = float(state["cash"] + (state["units"] * prices_next).sum())
    next_benchmark_value = benchmark_value
    if benchmark is not None and benchmark_today > 0:
        next_benchmark_value = benchmark_value * (benchmark_next / benchmark_today)

    payload = {
        "signal_date": signal_date,
        "next_date": next_date,
        "signal_output": signal_output,
        "desired_weights": desired_weights,
        "trade_frame": trade_frame,
        "positions_frame": positions_frame,
        "daily_row": build_daily_row(
            signal_date,
            next_date,
            signal_output["regime"],
            start_value,
            end_value,
            benchmark_value,
            next_benchmark_value,
            signal_output["decisions"],
            state,
            desired_weights,
        ),
    }
    return state, next_benchmark_value, payload


def simulate_month(
    config: PipelineConfig,
    market_inputs: dict[str, Any],
    run_dir: Path,
    initial_cash: float,
    simulation_days: int,
    seed: int,
    start_date: str | None,
    step_seconds: float,
    signal_fn=run_pipeline_signal,
) -> dict[str, Any]:
    prices = market_inputs["asset_prices"]
    benchmark = market_inputs["benchmark"]
    fundamentals = market_inputs["fundamentals"]
    news = market_inputs["news"]
    sectors = market_inputs["sectors"]
    tickers = list(prices.columns)

    start_index = choose_start_index(prices.index, config.min_history_days, simulation_days, seed, start_date)
    state = initialize_state(tickers, initial_cash)
    benchmark_value = float(initial_cash)

    metadata = initialize_metadata(
        prices,
        config,
        tickers,
        initial_cash,
        simulation_days,
        seed,
        start_index,
        continuous=False,
        poll_seconds=step_seconds,
    )

    daily_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []

    for step in range(simulation_days):
        current_index = start_index + step
        state, benchmark_value, payload = execute_step(
            current_index,
            prices,
            benchmark,
            fundamentals,
            news,
            sectors,
            config,
            state,
            benchmark_value,
            signal_fn,
        )

        signal_date = payload["signal_date"]
        next_date = payload["next_date"]
        daily_rows.append(payload["daily_row"])
        trade_rows.extend(payload["trade_frame"].to_dict(orient="records"))
        position_rows.extend(payload["positions_frame"].to_dict(orient="records"))
        for record in payload["signal_output"]["records"]:
            decision_rows.append({"signal_date": signal_date, **record})

        metadata["completed_steps"] = step + 1
        metadata["latest_signal_date"] = str(signal_date.date())
        metadata["latest_next_date"] = str(next_date.date())
        metadata["current_index"] = current_index + 1
        write_run_artifacts(
            run_dir,
            metadata,
            pd.DataFrame(daily_rows),
            pd.DataFrame(trade_rows),
            pd.DataFrame(position_rows),
            pd.DataFrame(decision_rows),
        )
        persist_state(run_dir, state, benchmark_value)
        if step_seconds > 0:
            time.sleep(step_seconds)

    metadata["status"] = "completed"
    write_run_artifacts(
        run_dir,
        metadata,
        pd.DataFrame(daily_rows),
        pd.DataFrame(trade_rows),
        pd.DataFrame(position_rows),
        pd.DataFrame(decision_rows),
    )
    return summarize_run(metadata, pd.DataFrame(daily_rows), pd.DataFrame(trade_rows))


def run_resumable_agent(
    config: PipelineConfig,
    market_inputs: dict[str, Any],
    run_dir: Path,
    initial_cash: float,
    simulation_days: int,
    seed: int,
    start_date: str | None,
    step_seconds: float,
    continuous: bool,
    poll_seconds: float,
    initial_state: dict[str, Any] | None = None,
    initial_metadata: dict[str, Any] | None = None,
    signal_fn=run_pipeline_signal,
) -> dict[str, Any]:
    tickers = list(market_inputs["asset_prices"].columns)
    metadata = load_json_file(run_dir / "metadata.json")
    daily, trades, positions, decisions = load_run_frames(run_dir)
    seed_benchmark_value: float | None = None

    if not metadata:
        prices = market_inputs["asset_prices"]
        start_index = choose_start_index(
            prices.index,
            config.min_history_days,
            simulation_days,
            seed,
            start_date,
            require_full_horizon=not continuous,
        )
        metadata = initialize_metadata(
            prices,
            config,
            tickers,
            initial_cash,
            simulation_days,
            seed,
            start_index,
            continuous=continuous,
            poll_seconds=poll_seconds,
            extra_metadata=initial_metadata,
        )
        if initial_state is not None:
            seed_benchmark_value = portfolio_value(initial_state, prices.iloc[start_index])
    else:
        metadata["simulation_days"] = simulation_days
        metadata["continuous"] = bool(continuous or metadata.get("continuous", False))
        metadata["poll_seconds"] = float(poll_seconds)
        if not metadata["continuous"]:
            metadata["end_index"] = int(metadata["start_index"]) + simulation_days - 1

    state, benchmark_value = load_state(
        run_dir,
        tickers,
        initial_cash,
        default_state=initial_state,
        default_benchmark_value=seed_benchmark_value,
    )

    try:
        while True:
            if metadata.get("continuous", False):
                market_inputs = fetch_market_inputs(config, include_side_data=market_inputs["fundamentals"] is not None)
            prices = market_inputs["asset_prices"]
            benchmark = market_inputs["benchmark"]
            fundamentals = market_inputs["fundamentals"]
            news = market_inputs["news"]
            sectors = market_inputs["sectors"]

            current_index = int(metadata.get("current_index", metadata.get("start_index", 0)))
            end_index = int(metadata.get("end_index", current_index + simulation_days - 1))

            if current_index + 1 >= len(prices.index):
                metadata["status"] = "waiting_for_data"
                write_run_artifacts(run_dir, metadata, daily, trades, positions, decisions)
                persist_state(run_dir, state, benchmark_value)
                if not continuous:
                    break
                time.sleep(poll_seconds)
                continue

            if not continuous and current_index > end_index:
                metadata["status"] = "completed"
                break

            state, benchmark_value, payload = execute_step(
                current_index,
                prices,
                benchmark,
                fundamentals,
                news,
                sectors,
                config,
                state,
                benchmark_value,
                signal_fn,
            )

            daily = pd.concat([daily, pd.DataFrame([payload["daily_row"]])], ignore_index=True)
            trades = pd.concat([trades, payload["trade_frame"]], ignore_index=True)
            positions = pd.concat([positions, payload["positions_frame"]], ignore_index=True)
            decision_records = pd.DataFrame(
                [{"signal_date": payload["signal_date"], **record} for record in payload["signal_output"]["records"]]
            )
            decisions = pd.concat([decisions, decision_records], ignore_index=True)

            metadata["status"] = "running"
            metadata["completed_steps"] = int(metadata.get("completed_steps", 0)) + 1
            metadata["latest_signal_date"] = str(payload["signal_date"].date())
            metadata["latest_next_date"] = str(payload["next_date"].date())
            metadata["current_index"] = current_index + 1
            if not continuous and metadata["current_index"] > end_index:
                metadata["status"] = "completed"

            write_run_artifacts(run_dir, metadata, daily, trades, positions, decisions)
            persist_state(run_dir, state, benchmark_value)

            if metadata["status"] == "completed":
                break
            if step_seconds > 0:
                time.sleep(step_seconds)
    except KeyboardInterrupt:
        metadata["status"] = "stopped"
        write_run_artifacts(run_dir, metadata, daily, trades, positions, decisions)
        persist_state(run_dir, state, benchmark_value)
        return summarize_run(metadata, daily, trades)

    write_run_artifacts(run_dir, metadata, daily, trades, positions, decisions)
    persist_state(run_dir, state, benchmark_value)
    return summarize_run(metadata, daily, trades)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers, for example TSLA,AMD,GOOGL")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--simulation-days", type=int, default=TRADING_DAYS_PER_MONTH)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--portfolio-transactions", default=None, help="CSV export used to seed real cash and holdings.")
    parser.add_argument("--portfolio-as-of", default=None, help="Portfolio snapshot date used when seeding from transactions.")
    parser.add_argument(
        "--portfolio-symbol-map",
        default=None,
        help="Comma-separated SOURCE=TARGET mappings, for example US67066G1040=NVDA,NL0010273215=ASML",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--step-seconds", type=float, default=1.0)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--cache-ttl-hours", type=float, default=0.0)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--include-side-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = tuple(ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip())
    symbol_map = parse_symbol_map(args.portfolio_symbol_map)
    config = build_config(
        tickers=tickers,
        benchmark=args.benchmark,
        cache_ttl_hours=None if args.use_cache else args.cache_ttl_hours,
        force_refresh=not args.use_cache,
    )
    market_inputs = fetch_market_inputs(config, include_side_data=args.include_side_data)
    initial_state = None
    initial_metadata = None
    effective_start_date = args.start_date
    if args.portfolio_transactions:
        initial_state, portfolio_as_of, initial_metadata = build_initial_state_from_transactions(
            transactions_path=Path(args.portfolio_transactions),
            tickers=list(market_inputs["asset_prices"].columns),
            as_of_date=args.portfolio_as_of or args.start_date,
            symbol_map=symbol_map,
        )
        if effective_start_date is None:
            effective_start_date = str(portfolio_as_of.date())
    run_dir = make_run_directory(Path(args.run_root), args.run_id, resume=args.resume)
    summary = run_resumable_agent(
        config=config,
        market_inputs=market_inputs,
        run_dir=run_dir,
        initial_cash=args.initial_cash,
        simulation_days=args.simulation_days,
        seed=args.seed,
        start_date=effective_start_date,
        step_seconds=args.step_seconds,
        continuous=args.continuous,
        poll_seconds=args.poll_seconds,
        initial_state=initial_state,
        initial_metadata=initial_metadata,
    )
    print(f"Run directory: {run_dir}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()