from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_universe_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Universe CSV `{path}` is empty.")
    frame.columns = [_safe_label(str(column)) for column in frame.columns]
    ticker_column = _ticker_column(frame)
    if ticker_column is None:
        raise ValueError("Universe CSV must include a ticker or symbol column.")
    frame[ticker_column] = frame[ticker_column].astype(str).str.upper()
    return frame.rename(columns={ticker_column: "ticker"})


def parse_universe_tickers(value: str | None) -> list[str]:
    if not value:
        return []
    return sorted({item.strip().upper() for item in value.split(",") if item.strip()})


def build_panel_frame(price_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a date/ticker MultiIndex panel from individual normalized frames."""

    pieces = []
    for ticker, frame in price_frames.items():
        piece = frame.copy()
        piece["ticker"] = ticker.upper()
        piece.index = pd.DatetimeIndex(piece.index)
        piece.index.name = "date"
        pieces.append(piece.reset_index().set_index(["date", "ticker"]))
    if not pieces:
        return pd.DataFrame()
    panel = pd.concat(pieces).sort_index()
    return panel[~panel.index.duplicated(keep="last")]


def build_cross_sectional_panel_features(panel: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
    """Build date-wise cross-sectional ranks from a date/ticker panel."""

    if panel.empty:
        return pd.DataFrame(index=panel.index)
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names[:2] != ["date", "ticker"]:
        raise ValueError("Panel must use a date/ticker MultiIndex.")
    if price_column not in panel.columns:
        raise ValueError(f"Panel must include `{price_column}`.")

    close = pd.to_numeric(panel[price_column], errors="coerce")
    log_close = np.log(close.replace(0, np.nan))
    by_ticker = log_close.groupby(level="ticker")
    ret_1d = by_ticker.diff()
    momentum_5d = by_ticker.diff(5)
    momentum_21d = by_ticker.diff(21)
    volatility_20d = ret_1d.groupby(level="ticker").rolling(20).std().droplevel(0)

    features = pd.DataFrame(index=panel.index)
    features["cs_return_1d_rank"] = _date_rank(ret_1d)
    features["cs_momentum_5d_rank"] = _date_rank(momentum_5d)
    features["cs_momentum_21d_rank"] = _date_rank(momentum_21d)
    features["cs_volatility_20d_rank"] = _date_rank(volatility_20d)
    features["cs_residual_momentum_21d"] = momentum_21d - momentum_21d.groupby(level="date").transform("mean")

    if "volume" in panel.columns:
        volume = pd.to_numeric(panel["volume"], errors="coerce")
        dollar_volume = close * volume
        average_dollar_volume_20d = dollar_volume.groupby(level="ticker").rolling(20).mean().droplevel(0)
        amihud_20d = (ret_1d.abs() / dollar_volume.replace(0, np.nan)).groupby(level="ticker").rolling(20).mean().droplevel(0)
        features["cs_dollar_volume_20d_rank"] = _date_rank(average_dollar_volume_20d)
        features["cs_illiquidity_20d_rank"] = _date_rank(amihud_20d, ascending=False)

    if "sector" in panel.columns:
        features = _add_sector_neutral_ranks(features, panel, momentum_21d, volatility_20d)

    return features.replace([np.inf, -np.inf], np.nan)


def select_ticker_panel_features(panel_features: pd.DataFrame, ticker: str, prefix: str = "panel_") -> pd.DataFrame:
    if panel_features.empty:
        return pd.DataFrame()
    symbol = ticker.upper()
    if symbol not in panel_features.index.get_level_values("ticker"):
        return pd.DataFrame()
    selected = panel_features.xs(symbol, level="ticker").copy()
    selected.columns = [f"{prefix}{column}" for column in selected.columns]
    selected.index = pd.DatetimeIndex(selected.index)
    return selected


def rank_universe_from_panel(
    panel: pd.DataFrame,
    panel_features: pd.DataFrame,
    price_column: str = "close",
    top_n: int = 25,
) -> list[dict[str, Any]]:
    """Rank universe members using market-action and cross-sectional evidence."""

    if panel.empty or panel_features.empty:
        return []
    rows = []
    for ticker in sorted(panel.index.get_level_values("ticker").unique()):
        asset = panel.xs(ticker, level="ticker").sort_index()
        asset_features = panel_features.xs(ticker, level="ticker").sort_index()
        if asset.empty or asset_features.empty or price_column not in asset.columns:
            continue
        latest_date = asset.index.intersection(asset_features.index).max()
        if pd.isna(latest_date):
            continue
        close = pd.to_numeric(asset[price_column], errors="coerce")
        latest_close = float(close.loc[latest_date])
        momentum_21d = float(np.log(close / close.shift(21)).loc[latest_date]) if len(close) > 21 else 0.0
        momentum_63d = float(np.log(close / close.shift(63)).loc[latest_date]) if len(close) > 63 else 0.0
        feature_row = asset_features.loc[latest_date]
        momentum_rank = _safe_float(feature_row.get("cs_momentum_21d_rank", 0.5), 0.5)
        return_rank = _safe_float(feature_row.get("cs_return_1d_rank", 0.5), 0.5)
        volume_rank = _safe_float(feature_row.get("cs_dollar_volume_20d_rank", 0.5), 0.5)
        volatility_rank = _safe_float(feature_row.get("cs_volatility_20d_rank", 0.5), 0.5)
        illiquidity_rank = _safe_float(feature_row.get("cs_illiquidity_20d_rank", 0.5), 0.5)
        residual_momentum = _safe_float(feature_row.get("cs_residual_momentum_21d", 0.0), 0.0)
        score = (
            0.35 * momentum_rank
            + 0.15 * return_rank
            + 0.15 * volume_rank
            + 0.15 * (1.0 - volatility_rank)
            + 0.10 * (1.0 - illiquidity_rank)
            + 0.10 * np.tanh(residual_momentum * 20.0)
        )
        rows.append(
            {
                "ticker": str(ticker),
                "as_of_date": str(pd.Timestamp(latest_date).date()),
                "score": float(score),
                "latest_close": latest_close,
                "momentum_21d": momentum_21d,
                "momentum_63d": momentum_63d,
                "cross_sectional_momentum_rank": momentum_rank,
                "cross_sectional_return_rank": return_rank,
                "cross_sectional_dollar_volume_rank": volume_rank,
                "cross_sectional_volatility_rank": volatility_rank,
                "cross_sectional_illiquidity_rank": illiquidity_rank,
                "residual_momentum_21d": residual_momentum,
            }
        )
    ranked = sorted(rows, key=lambda item: item["score"], reverse=True)
    for position, row in enumerate(ranked, start=1):
        row["rank"] = position
    return ranked[:top_n]


def summarize_universe(tickers: list[str], panel: pd.DataFrame | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ticker_count": int(len(tickers)),
        "tickers": sorted({ticker.upper() for ticker in tickers}),
    }
    if panel is not None and not panel.empty:
        dates = panel.index.get_level_values("date")
        summary.update(
            {
                "panel_rows": int(len(panel)),
                "start_date": str(pd.Timestamp(dates.min()).date()),
                "end_date": str(pd.Timestamp(dates.max()).date()),
                "columns": [str(column) for column in panel.columns],
            }
        )
    return summary


def _date_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    return series.groupby(level="date").rank(pct=True, ascending=ascending)


def _add_sector_neutral_ranks(
    features: pd.DataFrame,
    panel: pd.DataFrame,
    momentum_21d: pd.Series,
    volatility_20d: pd.Series,
) -> pd.DataFrame:
    working = pd.DataFrame(
        {
            "sector": panel["sector"],
            "momentum_21d": momentum_21d,
            "volatility_20d": volatility_20d,
        },
        index=panel.index,
    ).reset_index()
    working["cs_sector_momentum_21d_rank"] = working.groupby(["date", "sector"])["momentum_21d"].rank(pct=True)
    working["cs_sector_volatility_20d_rank"] = working.groupby(["date", "sector"])["volatility_20d"].rank(pct=True)
    sector_features = working.set_index(["date", "ticker"])[
        ["cs_sector_momentum_21d_rank", "cs_sector_volatility_20d_rank"]
    ]
    return features.join(sector_features, how="left")


def _ticker_column(frame: pd.DataFrame) -> str | None:
    for candidate in ("ticker", "symbol", "asset", "security"):
        if candidate in frame.columns:
            return candidate
    return None


def _safe_label(label: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in label).strip("_")


def _safe_float(value: Any, default: float) -> float:
    try:
        output = float(value)
    except Exception:
        return default
    return output if np.isfinite(output) else default
