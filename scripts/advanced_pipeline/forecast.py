"""Local stock forecasting with direct multi-horizon LightGBM models."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote_plus
from urllib.request import urlopen

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .cache import DataCache

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

DEFAULT_HISTORY_START = "2010-01-01"
OPTUNA_RANDOM_SEED = 42
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MAX_WALK_FORWARD_FOLDS = 12
DEFAULT_EVALUATION_FOLDS = 8
DEFAULT_DECISION_MAX_HORIZON_DAYS = 3
MIN_EMBARGO_DAYS = 10
MIN_INTERVAL_TRAIN_ROWS = 80
MIN_INTERVAL_CALIBRATION_ROWS = 40
INTERVAL_ALPHA = 0.1
DEPLOYMENT_MAE_EDGE = 0.97
MIN_INTERVAL_COVERAGE = 0.85
MAX_ABS_VALIDATION_BIAS = 0.005
DEFAULT_TRANSACTION_COST_RATE = 0.0025
DEFAULT_SAFETY_MARGIN_RATE = 0.0025
RESIDUAL_DRIFT_LOOKBACK = 3
SEMICONDUCTOR_CLASSIFIER_UNIVERSE = (
    "NVDA",
    "AMD",
    "AVGO",
    "QCOM",
    "MU",
    "TSM",
    "ASML",
    "ARM",
    "AMAT",
    "LRCX",
    "KLAC",
    "MRVL",
)

POSITIVE_NEWS_WORDS = {
    "beat",
    "beats",
    "bullish",
    "buyback",
    "contract",
    "gain",
    "gains",
    "growth",
    "guidance",
    "launch",
    "outperform",
    "partnership",
    "profit",
    "record",
    "strong",
    "surge",
    "upgrade",
    "upside",
    "win",
}
NEGATIVE_NEWS_WORDS = {
    "bearish",
    "cut",
    "cuts",
    "decline",
    "declines",
    "delay",
    "downgrade",
    "fall",
    "falls",
    "fraud",
    "investigation",
    "lawsuit",
    "loss",
    "losses",
    "miss",
    "misses",
    "recall",
    "risk",
    "warning",
    "weak",
}
FORECAST_FEATURE_COLUMNS = [
    "log_return_1d",
    "log_return_2d",
    "log_return_3d",
    "log_return_5d",
    "log_return_10d",
    "log_return_20d",
    "log_return_60d",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "volatility_60d",
    "ma_distance_5d",
    "ma_distance_10d",
    "ma_distance_20d",
    "ma_distance_50d",
    "ma_distance_100d",
    "drawdown_20d",
    "range_position_20d",
    "drawdown_60d",
    "range_position_60d",
    "rsi_14d",
    "breakout_distance_20d",
    "breakout_distance_60d",
    "up_day_ratio_10d",
    "large_up_day_ratio_20d",
    "momentum_consistency_20d",
    "spy_relative_return_1d",
    "spy_relative_return_5d",
    "spy_relative_return_20d",
    "qqq_relative_return_1d",
    "qqq_relative_return_5d",
    "qqq_relative_return_20d",
    "qqq_relative_acceleration_5_20d",
    "xlk_relative_return_1d",
    "xlk_relative_return_5d",
    "xlk_relative_return_20d",
    "soxx_relative_return_5d",
    "soxx_relative_return_20d",
    "vix_level",
    "vix_return_1d",
    "vix_return_5d",
    "vix_level_z_20d",
    "market_stress_interaction",
    "relative_volume_20d",
    "volume_z_20d",
    "dollar_volume_trend_20d",
    "spy_trend_20d",
    "qqq_trend_20d",
    "xlk_trend_20d",
    "market_regime_spread_20d",
    "volatility_regime_20d",
    "trend_strength_20_60d",
    "overnight_gap_1d",
    "intraday_return_1d",
    "atr_14d",
    "atr_regime_14_60d",
    "gap_trend_interaction",
    "volume_return_confirmation_5d",
    "momentum_acceleration_5_20d",
    "benchmark_vol_spread_20d",
    "qqq_correlation_60d",
    "semiconductor_leadership_20d",
    "news_sentiment_1d",
    "news_sentiment_3d",
    "news_sentiment_7d",
    "news_volume_1d",
    "news_volume_3d",
    "news_volume_7d",
    "news_positive_share_7d",
    "news_negative_share_7d",
    "news_sentiment_acceleration_3_7d",
    "news_volume_shock_1_7d",
    "weekday",
    "month",
]


@dataclass(frozen=True)
class ForecastArtifacts:
    ticker: str
    benchmark_symbol: str
    max_horizon_days: int
    feature_columns: tuple[str, ...]
    threshold_return: float
    training_cutoff: str
    fitted_at: str
    median_models: dict[int, Any]
    lower_models: dict[int, Any]
    upper_models: dict[int, Any]
    probability_models: dict[int, Any]
    probability_calibrators: dict[int, Any]
    trust_models: dict[int, Any]
    interval_adjustments: dict[int, float]
    validation_metrics: dict[int, dict[str, float]]
    news_feature_mode: str


@dataclass(frozen=True)
class PeerTrainingBundle:
    ticker: str
    training: pd.DataFrame


def model_file_for_ticker(ticker: str) -> Path:
    return MODEL_DIR / f"{ticker.upper()}_lgbm_model.joblib"


def metadata_file_for_ticker(ticker: str) -> Path:
    return MODEL_DIR / f"{ticker.upper()}_lgbm_metadata.json"


def default_progress_callback(step: int, total: int, message: str) -> None:
    return None


def classifier_peer_universe(ticker: str) -> tuple[str, ...]:
    symbol = ticker.upper()
    if symbol not in SEMICONDUCTOR_CLASSIFIER_UNIVERSE:
        return ()
    return tuple(peer for peer in SEMICONDUCTOR_CLASSIFIER_UNIVERSE if peer != symbol)


def default_probability_calibrator() -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit([0.0, 1.0], [0.5, 0.5])


def fit_probability_calibrator(prediction: np.ndarray, target: pd.Series) -> IsotonicRegression:
    actual = pd.Series(target).astype(float)
    if actual.nunique() < 2 or len(actual) < 20:
        return default_probability_calibrator()
    clipped_prediction = np.clip(np.asarray(prediction, dtype=float), 0.0, 1.0)
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(clipped_prediction, actual.to_numpy())


def apply_probability_calibrator(calibrator: Any, probability: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    if calibrator is None:
        return values
    return np.clip(np.asarray(calibrator.predict(values), dtype=float), 0.0, 1.0)


def clean_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.apply(pd.to_numeric, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan)
    return clean.fillna(clean.median()).fillna(0.0)


def sanitize_positive_series(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    clean = clean.where(clean > 0.0)
    return clean.ffill().bfill()


def build_trust_feature_frame(
    feature_frame: pd.DataFrame,
    raw_probability: np.ndarray,
    calibrated_probability: np.ndarray,
    regressor_prediction: np.ndarray,
) -> pd.DataFrame:
    trust = pd.DataFrame(index=feature_frame.index)
    raw = np.asarray(raw_probability, dtype=float)
    calibrated = np.asarray(calibrated_probability, dtype=float)
    regression = np.asarray(regressor_prediction, dtype=float)
    trust["raw_probability"] = raw
    trust["calibrated_probability"] = calibrated
    trust["probability_edge"] = np.abs(calibrated - 0.5) * 2.0
    trust["regressor_prediction"] = regression
    trust["probability_regression_alignment"] = (calibrated - 0.5) * regression
    for column in [
        "volatility_20d",
        "volatility_regime_20d",
        "market_stress_interaction",
        "qqq_relative_return_5d",
        "qqq_relative_acceleration_5_20d",
        "breakout_distance_20d",
        "relative_volume_20d",
        "volume_return_confirmation_5d",
        "semiconductor_leadership_20d",
        "momentum_consistency_20d",
    ]:
        trust[column] = feature_frame[column] if column in feature_frame.columns else 0.0
    return clean_numeric_frame(trust)


def fit_trust_model(features: pd.DataFrame, target: np.ndarray) -> Any | None:
    actual = np.asarray(target, dtype=int)
    if len(actual) < 40 or np.unique(actual).size < 2:
        return None
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=OPTUNA_RANDOM_SEED),
    )
    model.fit(clean_numeric_frame(features), actual)
    return model


def predict_trust_probability(model: Any | None, features: pd.DataFrame) -> np.ndarray:
    if model is None:
        return np.full(len(features), 0.5, dtype=float)
    return np.clip(model.predict_proba(clean_numeric_frame(features))[:, 1], 0.0, 1.0)


def fit_probability_calibration_and_trust(
    features: pd.DataFrame,
    target: pd.Series,
    probability_target: pd.Series,
    regressor_params: dict[str, Any] | None,
    classifier_params: dict[str, Any] | None,
    embargo_days: int,
    horizon: int,
    peer_training_bundles: tuple[PeerTrainingBundle, ...],
    threshold_return: float,
) -> tuple[Any, Any]:
    probability_split = split_training_calibration_rows(features, probability_target, embargo_days=embargo_days)
    target_split = split_training_calibration_rows(features, target, embargo_days=embargo_days)
    if probability_split is None or target_split is None:
        return default_probability_calibrator(), None

    fit_x, calibration_x, fit_probability_y, calibration_probability_y = probability_split
    _, _, fit_target_y, calibration_target_y = target_split
    classifier_fit_x, classifier_fit_y, _ = augment_classifier_training_set(
        fit_x,
        fit_probability_y,
        peer_training_bundles=peer_training_bundles,
        horizon=horizon,
        threshold_return=threshold_return,
        peer_target_column=f"target_signal_excess_log_return_{horizon}d",
        as_of_date=pd.Timestamp(fit_x.index.max()),
    )
    classifier = build_classifier_model(params=classifier_params)
    classifier.fit(classifier_fit_x, classifier_fit_y)
    calibration_raw_probability = classifier.predict_proba(calibration_x)[:, 1]
    calibrator = fit_probability_calibrator(calibration_raw_probability, calibration_probability_y)
    calibration_probability = apply_probability_calibrator(calibrator, calibration_raw_probability)

    regressor = build_regressor_model(objective="regression", params=regressor_params)
    regressor.fit(fit_x, fit_target_y)
    calibration_regression = regressor.predict(calibration_x)
    trust_target = ((calibration_probability >= 0.5).astype(int) == calibration_probability_y.to_numpy()).astype(int)
    trust_features = build_trust_feature_frame(
        calibration_x,
        calibration_raw_probability,
        calibration_probability,
        calibration_regression,
    )
    trust_model = fit_trust_model(trust_features, trust_target)
    return calibrator, trust_model


def require_openai_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def fetch_forecast_market_data(
    ticker: str,
    prediction_start_date: str,
    history_start: str = DEFAULT_HISTORY_START,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    import yfinance as yf

    symbols = [ticker.upper(), "SPY", "QQQ", "XLK", "SOXX", "^VIX"]
    cache = DataCache(cache_dir or (Path(__file__).resolve().parent / ".cache"), ttl_hours=24.0)
    params = {
        "symbols": symbols,
        "start": history_start,
        "end": prediction_start_date,
        "auto_adjust": True,
        "interval": "1d",
    }
    key = cache.key_for("forecast_prices", params)
    if not force_refresh:
        cached = cache.read_frame(key)
        if cached is not None and not cached.empty:
            return cached

    raw = yf.download(
        tickers=symbols,
        start=history_start,
        end=None,
        interval="1d",
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise ValueError("Yahoo Finance returned no market data for forecasting.")
    if not isinstance(raw.columns, pd.MultiIndex) or "Close" not in raw.columns.get_level_values(0):
        raise ValueError("Yahoo Finance response did not include Close prices for forecasting.")

    prices = raw["Close"].copy()
    prices.columns = [str(column).upper() for column in prices.columns]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index().apply(pd.to_numeric, errors="coerce")
    prices = prices.where(prices > 0.0).ffill().bfill().dropna(how="all")
    if ticker.upper() not in prices.columns:
        raise ValueError(f"Ticker {ticker.upper()} is missing from the downloaded market data.")

    selected_price_columns = [column for column in [ticker.upper(), "SPY", "QQQ", "XLK", "SOXX", "^VIX"] if column in prices.columns]
    output = prices.loc[:, selected_price_columns].copy()

    for field in ["Open", "High", "Low"]:
        if field in raw.columns.get_level_values(0):
            field_frame = raw[field].copy()
            if isinstance(field_frame, pd.Series):
                field_frame = field_frame.to_frame(name=ticker.upper())
            field_frame.columns = [str(column).upper() for column in field_frame.columns]
            field_frame.index = pd.to_datetime(field_frame.index).tz_localize(None)
            field_frame = field_frame.sort_index().apply(pd.to_numeric, errors="coerce")
            if ticker.upper() in field_frame.columns:
                output[f"{field.upper()}_{ticker.upper()}"] = sanitize_positive_series(field_frame[ticker.upper()])

    if "Volume" in raw.columns.get_level_values(0):
        volumes = raw["Volume"].copy()
        if isinstance(volumes, pd.Series):
            volumes = volumes.to_frame(name=ticker.upper())
        volumes.columns = [str(column).upper() for column in volumes.columns]
        volumes.index = pd.to_datetime(volumes.index).tz_localize(None)
        volumes = volumes.sort_index().apply(pd.to_numeric, errors="coerce")
        for symbol in [ticker.upper(), "SPY", "QQQ"]:
            if symbol in volumes.columns:
                    output[f"VOLUME_{symbol}"] = volumes[symbol].clip(lower=0.0)

    cache.write_frame(key, output, params)
    return output


def fetch_historical_news_features(
    ticker: str,
    prediction_start_date: str,
    history_start: str = DEFAULT_HISTORY_START,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
    use_openai: bool = True,
    allow_network: bool = True,
) -> tuple[pd.DataFrame, str]:
    cache = DataCache(cache_dir or (Path(__file__).resolve().parent / ".cache"), ttl_hours=24.0)
    params = {
        "ticker": ticker.upper(),
        "history_start": history_start,
        "prediction_start_date": prediction_start_date,
        "use_openai": bool(use_openai and require_openai_client() is not None),
        "source": "gdelt_doc_2",
    }
    key = cache.key_for("forecast_news", params)
    if not force_refresh:
        cached = cache.read_frame(key)
        if cached is not None:
            mode = "openai" if params["use_openai"] and not cached.empty and "openai_score" in cached.columns else "keyword"
            return cached, mode

    if not allow_network:
        empty = pd.DataFrame(columns=["date", "title", "sentiment_score"])
        return empty, "none"

    news_items = fetch_gdelt_news_items(
        ticker=ticker,
        history_start=history_start,
        prediction_start_date=prediction_start_date,
    )
    if news_items.empty:
        empty = pd.DataFrame(columns=["date", "title", "sentiment_score"])
        cache.write_frame(key, empty, params)
        return empty, "none"

    client = require_openai_client() if use_openai else None
    if client is not None:
        scores = score_headlines_with_openai(client, news_items["title"].tolist())
        news_items["sentiment_score"] = scores
        news_items["openai_score"] = scores
        mode = "openai"
    else:
        news_items["sentiment_score"] = news_items["title"].map(score_headline_keywords)
        mode = "keyword"

    cache.write_frame(key, news_items, params)
    return news_items, mode


def fetch_gdelt_news_items(ticker: str, history_start: str, prediction_start_date: str) -> pd.DataFrame:
    import yfinance as yf

    start = pd.Timestamp(history_start).normalize()
    end = pd.Timestamp(prediction_start_date).normalize()
    try:
        info = yf.Ticker(ticker.upper()).get_info()
        short_name = str(info.get("shortName") or info.get("longName") or "").strip()
    except Exception:
        short_name = ""

    query_terms = [f'"{ticker.upper()}"']
    if short_name:
        query_terms.append(f'"{short_name}"')
    query = " OR ".join(query_terms)

    rows: list[dict[str, Any]] = []
    window_start = start
    while window_start <= end:
        window_end = min(window_start + pd.Timedelta(days=89), end)
        url = (
            "https://api.gdeltproject.org/api/v2/doc/doc?"
            f"query={quote_plus(query)}&mode=ArtList&format=json&maxrecords=250&sort=DateDesc"
            f"&startdatetime={window_start.strftime('%Y%m%d000000')}"
            f"&enddatetime={window_end.strftime('%Y%m%d235959')}"
        )
        try:
            with urlopen(url, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            payload = {}
        for article in payload.get("articles", []) or []:
            title = str(article.get("title") or "").strip()
            seen_date = article.get("seendate") or article.get("date") or article.get("socialimage")
            if not title:
                continue
            article_date = pd.to_datetime(seen_date, errors="coerce")
            if pd.isna(article_date):
                continue
            article_date = pd.Timestamp(article_date).tz_localize(None).normalize()
            if article_date < start or article_date > end:
                continue
            rows.append(
                {
                    "date": article_date,
                    "title": title,
                    "source": article.get("domain") or article.get("sourceCountry") or "gdelt",
                }
            )
        window_start = window_end + pd.Timedelta(days=1)

    if not rows:
        return pd.DataFrame(columns=["date", "title", "source"])
    frame = pd.DataFrame(rows).drop_duplicates(subset=["date", "title"]).sort_values(["date", "title"])
    return frame.reset_index(drop=True)


def score_headline_keywords(title: str) -> float:
    words = set(str(title).lower().replace("-", " ").split())
    positive = len(words & POSITIVE_NEWS_WORDS)
    negative = len(words & NEGATIVE_NEWS_WORDS)
    total = max(positive + negative, 1)
    return float((positive - negative) / total)


def score_headlines_with_openai(client: OpenAI, titles: list[str]) -> list[float]:
    if not titles:
        return []
    scored: list[float] = []
    batch_size = 25
    for start in range(0, len(titles), batch_size):
        batch = titles[start : start + batch_size]
        prompt = {
            "headlines": [{"id": index, "title": title} for index, title in enumerate(batch)],
            "task": "Return JSON array with fields id and score where score is a finance sentiment score between -1 and 1.",
        }
        try:
            response = client.responses.create(
                model="gpt-4.1-mini",
                instructions=(
                    "You score finance headlines for expected short-term stock impact. "
                    "Return only valid JSON. Use score in [-1,1], where positive means bullish and negative means bearish."
                ),
                input=json.dumps(prompt),
            )
            payload = json.loads((response.output_text or "[]").strip())
            by_id = {int(item["id"]): float(item["score"]) for item in payload}
            scored.extend([float(np.clip(by_id.get(index, score_headline_keywords(title)), -1.0, 1.0)) for index, title in enumerate(batch)])
        except Exception:
            scored.extend([score_headline_keywords(title) for title in batch])
    return scored


def aggregate_historical_news_features(news_items: pd.DataFrame, price_index: pd.DatetimeIndex) -> pd.DataFrame:
    base = pd.DataFrame(index=price_index)
    base.index = pd.to_datetime(base.index).tz_localize(None)
    if news_items.empty:
        return base.assign(
            news_sentiment_1d=0.0,
            news_sentiment_3d=0.0,
            news_sentiment_7d=0.0,
            news_volume_1d=0.0,
            news_volume_3d=0.0,
            news_volume_7d=0.0,
            news_positive_share_7d=0.0,
            news_negative_share_7d=0.0,
            news_sentiment_acceleration_3_7d=0.0,
            news_volume_shock_1_7d=0.0,
        )

    frame = news_items.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["sentiment_score"] = pd.to_numeric(frame["sentiment_score"], errors="coerce").fillna(0.0)
    grouped = frame.groupby("date").agg(
        news_sentiment_1d=("sentiment_score", "mean"),
        news_volume_1d=("sentiment_score", "size"),
        news_positive_1d=("sentiment_score", lambda values: float((values > 0.15).mean() if len(values) else 0.0)),
        news_negative_1d=("sentiment_score", lambda values: float((values < -0.15).mean() if len(values) else 0.0)),
    )
    grouped = grouped.reindex(base.index).fillna(0.0)
    output = pd.DataFrame(index=base.index)
    output["news_sentiment_1d"] = grouped["news_sentiment_1d"]
    output["news_sentiment_3d"] = grouped["news_sentiment_1d"].rolling(3, min_periods=1).mean()
    output["news_sentiment_7d"] = grouped["news_sentiment_1d"].rolling(7, min_periods=1).mean()
    output["news_volume_1d"] = grouped["news_volume_1d"]
    output["news_volume_3d"] = grouped["news_volume_1d"].rolling(3, min_periods=1).sum()
    output["news_volume_7d"] = grouped["news_volume_1d"].rolling(7, min_periods=1).sum()
    output["news_positive_share_7d"] = grouped["news_positive_1d"].rolling(7, min_periods=1).mean()
    output["news_negative_share_7d"] = grouped["news_negative_1d"].rolling(7, min_periods=1).mean()
    output["news_sentiment_acceleration_3_7d"] = output["news_sentiment_3d"] - output["news_sentiment_7d"]
    output["news_volume_shock_1_7d"] = output["news_volume_1d"] / output["news_volume_7d"].replace(0.0, np.nan)
    return output.fillna(0.0)


def add_forecast_features(prices: pd.DataFrame, news_features: pd.DataFrame | None = None) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("No prices available to build forecast features.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("Forecast prices must use a DatetimeIndex.")

    clean = prices.sort_index().copy().ffill().dropna(how="all")
    price_columns = [column for column in clean.columns if not str(column).startswith("VOLUME_")]
    if not price_columns:
        raise ValueError("Forecast prices must include at least one close-price column.")

    target_symbol = str(price_columns[0]).upper()
    close = clean[target_symbol]
    returns = np.log(close / close.shift(1))
    frame = pd.DataFrame(index=clean.index)
    for window in (1, 2, 3, 5, 10, 20, 60):
        frame[f"log_return_{window}d"] = np.log(close / close.shift(window))
    for window in (5, 10, 20, 60):
        frame[f"volatility_{window}d"] = returns.rolling(window).std() * np.sqrt(252)
    for window in (5, 10, 20, 50, 60, 100):
        frame[f"ma_distance_{window}d"] = close / close.rolling(window).mean() - 1.0
    for window in (20, 60):
        rolling_high = close.rolling(window).max()
        rolling_low = close.rolling(window).min()
        frame[f"drawdown_{window}d"] = close / rolling_high - 1.0
        frame[f"range_position_{window}d"] = (close - rolling_low) / (rolling_high - rolling_low).replace(0.0, np.nan)

    frame["rsi_14d"] = calculate_rsi(close, 14)
    frame["breakout_distance_20d"] = close / close.rolling(20).max() - 1.0
    frame["breakout_distance_60d"] = close / close.rolling(60).max() - 1.0
    frame["up_day_ratio_10d"] = returns.gt(0.0).rolling(10).mean()
    frame["large_up_day_ratio_20d"] = returns.gt(np.log1p(0.02)).rolling(20).mean()
    frame["momentum_consistency_20d"] = returns.rolling(20).mean() / returns.rolling(20).std().replace(0.0, np.nan)

    spy = clean["SPY"] if "SPY" in clean.columns else close
    qqq = clean["QQQ"] if "QQQ" in clean.columns else close
    xlk = clean["XLK"] if "XLK" in clean.columns else qqq
    soxx = clean["SOXX"] if "SOXX" in clean.columns else xlk
    vix = clean["^VIX"] if "^VIX" in clean.columns else pd.Series(index=clean.index, data=20.0)
    frame["spy_relative_return_1d"] = np.log(close / close.shift(1)) - np.log(spy / spy.shift(1))
    frame["spy_relative_return_5d"] = np.log(close / close.shift(5)) - np.log(spy / spy.shift(5))
    frame["spy_relative_return_20d"] = np.log(close / close.shift(20)) - np.log(spy / spy.shift(20))
    frame["qqq_relative_return_1d"] = np.log(close / close.shift(1)) - np.log(qqq / qqq.shift(1))
    frame["qqq_relative_return_5d"] = np.log(close / close.shift(5)) - np.log(qqq / qqq.shift(5))
    frame["qqq_relative_return_20d"] = np.log(close / close.shift(20)) - np.log(qqq / qqq.shift(20))
    frame["qqq_relative_acceleration_5_20d"] = frame["qqq_relative_return_5d"] - frame["qqq_relative_return_20d"]
    frame["xlk_relative_return_1d"] = np.log(close / close.shift(1)) - np.log(xlk / xlk.shift(1))
    frame["xlk_relative_return_5d"] = np.log(close / close.shift(5)) - np.log(xlk / xlk.shift(5))
    frame["xlk_relative_return_20d"] = np.log(close / close.shift(20)) - np.log(xlk / xlk.shift(20))
    frame["soxx_relative_return_5d"] = np.log(close / close.shift(5)) - np.log(soxx / soxx.shift(5))
    frame["soxx_relative_return_20d"] = np.log(close / close.shift(20)) - np.log(soxx / soxx.shift(20))
    frame["vix_level"] = vix
    frame["vix_return_1d"] = np.log(vix / vix.shift(1))
    frame["vix_return_5d"] = np.log(vix / vix.shift(5))
    vix_mean = vix.rolling(20).mean()
    vix_std = vix.rolling(20).std()
    frame["vix_level_z_20d"] = (vix - vix_mean) / vix_std.replace(0.0, np.nan)
    frame["market_stress_interaction"] = frame["log_return_5d"] * frame["vix_level_z_20d"].fillna(0.0)
    frame["spy_trend_20d"] = spy / spy.rolling(20).mean() - 1.0
    frame["qqq_trend_20d"] = qqq / qqq.rolling(20).mean() - 1.0
    frame["xlk_trend_20d"] = xlk / xlk.rolling(20).mean() - 1.0
    frame["market_regime_spread_20d"] = frame["spy_trend_20d"] - frame["qqq_trend_20d"]
    frame["volatility_regime_20d"] = frame["volatility_20d"] / frame["volatility_60d"].replace(0.0, np.nan)
    frame["trend_strength_20_60d"] = frame["ma_distance_20d"] - frame["ma_distance_60d"]
    frame["momentum_acceleration_5_20d"] = frame["log_return_5d"] - frame["log_return_20d"]
    frame["semiconductor_leadership_20d"] = frame["soxx_relative_return_20d"] - frame["xlk_relative_return_20d"]
    qqq_returns = np.log(qqq / qqq.shift(1))
    qqq_corr = returns.rolling(60).corr(qqq_returns)
    qqq_volatility_20d = qqq_returns.rolling(20).std() * np.sqrt(252)
    frame["benchmark_vol_spread_20d"] = frame["volatility_20d"] - qqq_volatility_20d
    frame["qqq_correlation_60d"] = qqq_corr

    open_series = clean.get(f"OPEN_{target_symbol}")
    high_series = clean.get(f"HIGH_{target_symbol}")
    low_series = clean.get(f"LOW_{target_symbol}")
    if open_series is None or high_series is None or low_series is None:
        frame["overnight_gap_1d"] = 0.0
        frame["intraday_return_1d"] = 0.0
        frame["atr_14d"] = 0.0
        frame["atr_regime_14_60d"] = 0.0
    else:
        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                (high_series - low_series).abs(),
                (high_series - previous_close).abs(),
                (low_series - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_14d = true_range.rolling(14).mean() / close.replace(0.0, np.nan)
        atr_60d = true_range.rolling(60).mean() / close.replace(0.0, np.nan)
        frame["overnight_gap_1d"] = np.log(open_series / previous_close)
        frame["intraday_return_1d"] = np.log(close / open_series.replace(0.0, np.nan))
        frame["atr_14d"] = atr_14d
        frame["atr_regime_14_60d"] = atr_14d / atr_60d.replace(0.0, np.nan)
        frame["gap_trend_interaction"] = frame["overnight_gap_1d"] * frame["log_return_5d"]

    volume = clean.get(f"VOLUME_{target_symbol}", pd.Series(index=clean.index, dtype="float64"))
    if volume.empty or volume.dropna().empty:
        frame["relative_volume_20d"] = 0.0
        frame["volume_z_20d"] = 0.0
        frame["dollar_volume_trend_20d"] = 0.0
        frame["volume_return_confirmation_5d"] = 0.0
    else:
        volume_ma20 = volume.rolling(20).mean()
        volume_std20 = volume.rolling(20).std()
        dollar_volume = close * volume
        frame["relative_volume_20d"] = volume / volume_ma20.replace(0.0, np.nan)
        frame["volume_z_20d"] = (volume - volume_ma20) / volume_std20.replace(0.0, np.nan)
        frame["dollar_volume_trend_20d"] = np.log(dollar_volume / dollar_volume.rolling(20).mean())
        frame["volume_return_confirmation_5d"] = frame["relative_volume_20d"] * frame["log_return_5d"]

    if "gap_trend_interaction" not in frame.columns:
        frame["gap_trend_interaction"] = 0.0

    if news_features is not None and not news_features.empty:
        aligned_news = news_features.reindex(frame.index).fillna(0.0)
        for column in aligned_news.columns:
            frame[column] = aligned_news[column]
    else:
        for column in [
            "news_sentiment_1d",
            "news_sentiment_3d",
            "news_sentiment_7d",
            "news_volume_1d",
            "news_volume_3d",
            "news_volume_7d",
            "news_positive_share_7d",
            "news_negative_share_7d",
            "news_sentiment_acceleration_3_7d",
            "news_volume_shock_1_7d",
        ]:
            frame[column] = 0.0
    frame = frame.replace([np.inf, -np.inf], np.nan).shift(1)
    frame["weekday"] = frame.index.weekday
    frame["month"] = frame.index.month
    return frame


def calculate_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.rolling(window).mean()
    average_loss = losses.rolling(window).mean()
    rs = average_gain / average_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def benchmark_symbol_from_prices(prices: pd.DataFrame, target_symbol: str) -> str:
    for symbol in ["QQQ", "XLK", "SPY"]:
        if symbol in prices.columns and symbol != target_symbol:
            return symbol
    return target_symbol


def signal_benchmark_symbol_from_prices(prices: pd.DataFrame, target_symbol: str) -> str:
    for symbol in ["SOXX", "QQQ", "XLK", "SPY"]:
        if symbol in prices.columns and symbol != target_symbol:
            return symbol
    return benchmark_symbol_from_prices(prices, target_symbol)


def benchmark_relative_feature_columns(benchmark_symbol: str) -> tuple[str | None, str | None]:
    feature_map = {
        "SPY": ("spy_relative_return_1d", "spy_relative_return_5d"),
        "QQQ": ("qqq_relative_return_1d", "qqq_relative_return_5d"),
        "XLK": ("xlk_relative_return_1d", "xlk_relative_return_5d"),
    }
    return feature_map.get(str(benchmark_symbol).upper(), (None, None))


def validation_embargo_days(max_horizon_days: int, horizon: int) -> int:
    return int(max(max_horizon_days, horizon, MIN_EMBARGO_DAYS))


def estimate_benchmark_forward_log_return(
    benchmark_series: pd.Series,
    cutoff: pd.Timestamp,
    horizon: int,
) -> float:
    history = benchmark_series.loc[benchmark_series.index <= cutoff].dropna()
    if len(history) < 21:
        return 0.0
    benchmark_return_5d = float(np.log(history.iloc[-1] / history.iloc[-6]) / 5.0) if len(history) >= 6 else 0.0
    benchmark_return_20d = float(np.log(history.iloc[-1] / history.iloc[-21]) / 20.0) if len(history) >= 21 else benchmark_return_5d
    blended_daily_return = 0.6 * benchmark_return_5d + 0.4 * benchmark_return_20d
    return blended_daily_return * horizon


def add_forecast_targets(
    feature_frame: pd.DataFrame,
    price_frame: pd.DataFrame,
    max_horizon_days: int,
) -> pd.DataFrame:
    output = feature_frame.copy()
    target_symbol = str(price_frame.columns[0]).upper()
    benchmark_symbol = benchmark_symbol_from_prices(price_frame, target_symbol)
    signal_benchmark_symbol = signal_benchmark_symbol_from_prices(price_frame, target_symbol)
    target_series = price_frame[target_symbol]
    benchmark_series = price_frame[benchmark_symbol]
    signal_benchmark_series = price_frame[signal_benchmark_symbol]
    for horizon in range(1, max_horizon_days + 1):
        asset_log_return = np.log(target_series.shift(-horizon) / target_series)
        benchmark_log_return = np.log(benchmark_series.shift(-horizon) / benchmark_series)
        signal_benchmark_log_return = np.log(signal_benchmark_series.shift(-horizon) / signal_benchmark_series)
        output[f"target_log_return_{horizon}d"] = asset_log_return
        output[f"target_excess_log_return_{horizon}d"] = asset_log_return - benchmark_log_return
        output[f"target_signal_excess_log_return_{horizon}d"] = asset_log_return - signal_benchmark_log_return
    return output


def build_training_data(
    prices: pd.DataFrame,
    prediction_start_date: str,
    max_horizon_days: int,
    news_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Timestamp, int]:
    feature_frame = add_forecast_features(prices, news_features=news_features)
    cutoff = resolve_prediction_start_date(prices.index, prediction_start_date)
    target_frame = add_forecast_targets(feature_frame, prices[[prices.columns[0], *[column for column in prices.columns[1:] if not str(column).startswith(("OPEN_", "HIGH_", "LOW_", "VOLUME_"))]]], max_horizon_days)
    training = target_frame.loc[target_frame.index < cutoff].copy()
    training = training.dropna(subset=FORECAST_FEATURE_COLUMNS)
    if training.empty:
        raise ValueError("No training rows remain after applying the prediction start date and feature requirements.")
    return training, prices.iloc[:, 0], cutoff, int(prices.index.get_loc(cutoff))


def build_classifier_peer_training_bundles(
    ticker: str,
    prediction_start_date: str,
    max_horizon_days: int,
) -> tuple[PeerTrainingBundle, ...]:
    bundles: list[PeerTrainingBundle] = []
    for peer_ticker in classifier_peer_universe(ticker):
        try:
            peer_prices = fetch_forecast_market_data(peer_ticker, prediction_start_date)
            peer_training, _, _, _ = build_training_data(
                peer_prices,
                prediction_start_date,
                max_horizon_days,
                news_features=None,
            )
        except Exception:
            continue
        bundles.append(PeerTrainingBundle(ticker=peer_ticker, training=peer_training))
    return tuple(bundles)


def augment_classifier_training_set(
    base_features: pd.DataFrame,
    base_target: pd.Series,
    peer_training_bundles: tuple[PeerTrainingBundle, ...],
    horizon: int,
    threshold_return: float,
    peer_target_column: str | None = None,
    as_of_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Series, int]:
    if as_of_date is None:
        if base_features.empty:
            return base_features, base_target, 0
        as_of_date = pd.Timestamp(base_features.index.max())

    preferred_target_name = peer_target_column or f"target_signal_excess_log_return_{horizon}d"
    fallback_target_name = f"target_excess_log_return_{horizon}d"
    legacy_target_name = f"target_log_return_{horizon}d"
    feature_frames = [base_features]
    target_frames = [base_target]
    peer_rows_added = 0
    for bundle in peer_training_bundles:
        available_target_name = next(
            (
                column_name
                for column_name in (preferred_target_name, fallback_target_name, legacy_target_name)
                if column_name in bundle.training.columns
            ),
            None,
        )
        if available_target_name is None:
            continue
        peer_rows = bundle.training.loc[bundle.training.index <= as_of_date].dropna(subset=[available_target_name]).copy()
        if len(peer_rows) < 40:
            continue
        feature_frames.append(peer_rows[FORECAST_FEATURE_COLUMNS])
        peer_target = (np.exp(peer_rows[available_target_name]) - 1.0 >= threshold_return).astype(int)
        target_frames.append(peer_target)
        peer_rows_added += len(peer_rows)

    classifier_features = pd.concat(feature_frames, axis=0)
    classifier_target = pd.concat(target_frames, axis=0)
    return classifier_features, classifier_target, peer_rows_added


def train_forecast_artifacts(
    ticker: str,
    prices: pd.DataFrame,
    prediction_start_date: str,
    max_horizon_days: int,
    threshold_return: float = 0.0,
    use_openai_news: bool = True,
    decision_horizon_cap: int | None = DEFAULT_DECISION_MAX_HORIZON_DAYS,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ForecastArtifacts:
    progress = progress_callback or default_progress_callback
    prediction_start_date = str(resolve_prediction_start_date(prices.index, prediction_start_date).date())
    effective_max_horizon_days = int(max_horizon_days)
    if decision_horizon_cap is not None:
        effective_max_horizon_days = min(effective_max_horizon_days, int(decision_horizon_cap))
    if effective_max_horizon_days < 1:
        raise ValueError("Forecast training needs at least one horizon day.")

    progress(0, effective_max_horizon_days * 6 + 1, f"Loading historical news for {ticker}")
    news_items, news_mode = fetch_historical_news_features(
        ticker=ticker,
        prediction_start_date=prediction_start_date,
        use_openai=use_openai_news,
    )
    news_features = aggregate_historical_news_features(news_items, prices.index)
    training, _, cutoff, cutoff_position = build_training_data(
        prices,
        prediction_start_date,
        effective_max_horizon_days,
        news_features=news_features,
    )
    if len(training) < 260:
        raise ValueError(f"Need at least 260 training rows to fit the forecast model. Found {len(training)}.")

    total_steps = effective_max_horizon_days * 6 + 1
    current_step = 0
    median_models: dict[int, Any] = {}
    lower_models: dict[int, Any] = {}
    upper_models: dict[int, Any] = {}
    probability_models: dict[int, Any] = {}
    probability_calibrators: dict[int, Any] = {}
    trust_models: dict[int, Any] = {}
    interval_adjustments: dict[int, float] = {}
    validation_metrics: dict[int, dict[str, float]] = {}
    benchmark_symbol = benchmark_symbol_from_prices(prices, ticker.upper())
    signal_benchmark_symbol = signal_benchmark_symbol_from_prices(prices, ticker.upper())
    relative_return_1d_column, relative_return_5d_column = benchmark_relative_feature_columns(benchmark_symbol)
    signal_relative_return_1d_column, signal_relative_return_5d_column = benchmark_relative_feature_columns(signal_benchmark_symbol)
    peer_training_bundles = build_classifier_peer_training_bundles(
        ticker=ticker,
        prediction_start_date=prediction_start_date,
        max_horizon_days=effective_max_horizon_days,
    )

    for horizon in range(1, effective_max_horizon_days + 1):
        target_name = f"target_log_return_{horizon}d"
        signal_target_name = f"target_signal_excess_log_return_{horizon}d"
        horizon_embargo_days = validation_embargo_days(effective_max_horizon_days, horizon)
        latest_safe_position = cutoff_position - horizon
        if latest_safe_position < 0:
            raise ValueError(f"Prediction start date {prediction_start_date} does not leave history for horizon {horizon}.")
        allowed_index = prices.index[: latest_safe_position + 1]
        horizon_rows = training.loc[training.index.isin(allowed_index)].dropna(subset=[target_name]).copy()
        if len(horizon_rows) < 120:
            raise ValueError(f"Not enough rows to train horizon {horizon}. Need at least 120, found {len(horizon_rows)}.")

        horizon_features = horizon_rows[FORECAST_FEATURE_COLUMNS]
        target = horizon_rows[f"target_excess_log_return_{horizon}d"]
        raw_probability_target = horizon_rows[signal_target_name]
        probability_target = (np.exp(raw_probability_target) - 1.0 >= threshold_return).astype(int)

        current_step += 1
        progress(current_step, total_steps, f"Tuning median forecast model for {ticker} day {horizon}")
        regressor_params, regressor_trials = tune_regressor_hyperparameters(
            horizon_features,
            target,
            max_horizon_days=effective_max_horizon_days,
            embargo_days=horizon_embargo_days,
        )

        current_step += 1
        progress(current_step, total_steps, f"Training median forecast model for {ticker} day {horizon}")
        median_models[horizon] = build_regressor_model(objective="regression", params=regressor_params)
        median_models[horizon].fit(horizon_features, target)

        current_step += 1
        progress(current_step, total_steps, f"Training lower confidence model for {ticker} day {horizon}")
        lower_models[horizon] = build_regressor_model(objective="quantile", alpha=0.1, params=regressor_params)
        lower_models[horizon].fit(horizon_features, target)

        current_step += 1
        progress(current_step, total_steps, f"Training upper confidence model for {ticker} day {horizon}")
        upper_models[horizon] = build_regressor_model(objective="quantile", alpha=0.9, params=regressor_params)
        upper_models[horizon].fit(horizon_features, target)
        interval_adjustments[horizon] = calibrate_interval_adjustment(
            horizon_features,
            target,
            regressor_params=regressor_params,
            embargo_days=horizon_embargo_days,
        )

        current_step += 1
        progress(current_step, total_steps, f"Tuning probability model for {ticker} day {horizon}")
        classifier_training_features, classifier_training_target, classifier_peer_rows = augment_classifier_training_set(
            horizon_features,
            probability_target,
            peer_training_bundles=peer_training_bundles,
            horizon=horizon,
            threshold_return=threshold_return,
            peer_target_column=signal_target_name,
            as_of_date=pd.Timestamp(horizon_rows.index.max()),
        )
        classifier_params, classifier_trials = tune_classifier_hyperparameters(
            classifier_training_features,
            classifier_training_target,
            max_horizon_days=effective_max_horizon_days,
            embargo_days=horizon_embargo_days,
        )
        probability_calibrators[horizon], trust_models[horizon] = fit_probability_calibration_and_trust(
            horizon_features,
            target,
            probability_target,
            regressor_params=regressor_params,
            classifier_params=classifier_params,
            embargo_days=horizon_embargo_days,
            horizon=horizon,
            peer_training_bundles=peer_training_bundles,
            threshold_return=threshold_return,
        )

        current_step += 1
        progress(current_step, total_steps, f"Training probability model for {ticker} day {horizon}")
        probability_models[horizon] = build_classifier_model(params=classifier_params)
        probability_models[horizon].fit(classifier_training_features, classifier_training_target)

        validation_metrics[horizon] = evaluate_horizon_models(
            horizon_features,
            target,
            probability_target,
            horizon_rows.index,
            regressor_params=regressor_params,
            classifier_params=classifier_params,
            embargo_days=horizon_embargo_days,
            horizon=horizon,
            relative_return_1d_column=relative_return_1d_column,
            relative_return_5d_column=relative_return_5d_column,
            signal_relative_return_1d_column=signal_relative_return_1d_column,
            signal_relative_return_5d_column=signal_relative_return_5d_column,
            peer_training_bundles=peer_training_bundles,
            threshold_return=threshold_return,
            stored_calibrator=probability_calibrators[horizon],
            stored_trust_model=trust_models[horizon],
        )
        validation_metrics[horizon]["regressor_optuna_trials"] = regressor_trials
        validation_metrics[horizon]["classifier_optuna_trials"] = classifier_trials
        validation_metrics[horizon]["interval_adjustment"] = interval_adjustments[horizon]
        validation_metrics[horizon]["benchmark_symbol"] = benchmark_symbol
        validation_metrics[horizon]["signal_benchmark_symbol"] = signal_benchmark_symbol
        validation_metrics[horizon]["classifier_training_rows"] = int(len(classifier_training_features))
        validation_metrics[horizon]["classifier_peer_rows"] = int(classifier_peer_rows)
        validation_metrics[horizon]["classifier_peer_tickers"] = ",".join(bundle.ticker for bundle in peer_training_bundles)

    return ForecastArtifacts(
        ticker=ticker.upper(),
        benchmark_symbol=benchmark_symbol,
        max_horizon_days=effective_max_horizon_days,
        feature_columns=tuple(FORECAST_FEATURE_COLUMNS),
        threshold_return=float(threshold_return),
        training_cutoff=str(cutoff.date()),
        fitted_at=str(pd.Timestamp.utcnow()),
        median_models=median_models,
        lower_models=lower_models,
        upper_models=upper_models,
        probability_models=probability_models,
        probability_calibrators=probability_calibrators,
        trust_models=trust_models,
        interval_adjustments=interval_adjustments,
        validation_metrics=validation_metrics,
        news_feature_mode=news_mode,
    )


def base_regressor_params() -> dict[str, Any]:
    return {
        "n_estimators": 500,
        "learning_rate": 0.02,
        "num_leaves": 15,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "min_child_samples": 40,
        "reg_alpha": 0.1,
        "reg_lambda": 0.25,
        "random_state": OPTUNA_RANDOM_SEED,
        "n_jobs": 1,
        "verbosity": -1,
        "force_col_wise": True,
    }


def base_classifier_params() -> dict[str, Any]:
    return {
        "n_estimators": 400,
        "learning_rate": 0.02,
        "num_leaves": 15,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "min_child_samples": 40,
        "reg_alpha": 0.1,
        "reg_lambda": 0.25,
        "random_state": OPTUNA_RANDOM_SEED,
        "n_jobs": 1,
        "verbosity": -1,
        "force_col_wise": True,
    }


def build_regressor_model(objective: str, alpha: float | None = None, params: dict[str, Any] | None = None) -> LGBMRegressor:
    model_params: dict[str, Any] = {
        "objective": objective,
        **base_regressor_params(),
    }
    if params is not None:
        model_params.update(
            {
                key: value
                for key, value in params.items()
                if key not in {"objective", "alpha", "metric", "verbosity"} and value is not None
            }
        )
    if objective == "quantile":
        model_params["alpha"] = 0.5 if alpha is None else alpha
        model_params["metric"] = "quantile"
    else:
        model_params.pop("alpha", None)
        model_params["metric"] = "l2"
    return LGBMRegressor(**model_params)


def build_classifier_model(params: dict[str, Any] | None = None) -> LGBMClassifier:
    model_params: dict[str, Any] = {
        "objective": "binary",
        **base_classifier_params(),
    }
    if params is not None:
        model_params.update(
            {
                key: value
                for key, value in params.items()
                if key not in {"objective", "alpha", "metric", "class_weight", "verbosity"} and value is not None
            }
        )
    return LGBMClassifier(**model_params)


def choose_optuna_trials(row_count: int, max_horizon_days: int) -> int:
    if row_count >= 1000:
        trials = 8
    elif row_count >= 500:
        trials = 6
    else:
        trials = 4
    if max_horizon_days >= 10:
        trials = max(3, trials - 2)
    return trials


def optuna_regressor_space(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 250, 700),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.04, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 40),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 90),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def optuna_classifier_space(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.04, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 40),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 90),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def split_train_validation_rows(
    total_rows: int,
    min_train_rows: int = 180,
    min_validation_rows: int = 40,
    embargo_days: int = 0,
) -> int:
    required_rows = min_train_rows + min_validation_rows + max(0, embargo_days)
    if total_rows < required_rows:
        raise ValueError(
            f"Need at least {required_rows} rows for train/validation with embargo. Found {total_rows}."
        )
    validation_rows = max(min_validation_rows, int(round(total_rows * 0.2)))
    validation_rows = min(validation_rows, total_rows - min_train_rows - max(0, embargo_days))
    if validation_rows < min_validation_rows:
        raise ValueError(
            f"Could not reserve at least {min_validation_rows} validation rows from {total_rows} total rows."
        )
    return int(validation_rows)


def walk_forward_validation_slices(
    total_rows: int,
    embargo_days: int,
    min_train_rows: int,
    min_validation_rows: int,
    max_folds: int = DEFAULT_MAX_WALK_FORWARD_FOLDS,
) -> list[tuple[int, int, int]]:
    first_validation_start = min_train_rows + embargo_days
    available_validation_span = total_rows - first_validation_start
    if available_validation_span < min_validation_rows:
        raise ValueError(
            f"Need at least {min_train_rows + min_validation_rows + max(0, embargo_days)} rows for walk-forward validation. Found {total_rows}."
        )
    candidate_span = max(1, total_rows - min_validation_rows - first_validation_start + 1)
    validation_rows = max(min_validation_rows, int(candidate_span / max(1, max_folds)))
    last_validation_start = total_rows - validation_rows
    candidate_starts = list(range(first_validation_start, last_validation_start + 1, validation_rows))
    if not candidate_starts or candidate_starts[-1] != last_validation_start:
        candidate_starts.append(last_validation_start)

    slices: list[tuple[int, int, int]] = []
    for validation_start in candidate_starts[-max_folds:]:
        train_end = validation_start - embargo_days
        validation_end = min(total_rows, validation_start + validation_rows)
        if train_end < min_train_rows:
            continue
        if validation_end - validation_start < min_validation_rows:
            continue
        slices.append((train_end, validation_start, validation_end))
    return slices


def split_nested_walk_forward_slices(
    total_rows: int,
    embargo_days: int,
    min_train_rows: int,
    min_validation_rows: int,
    max_folds: int = DEFAULT_MAX_WALK_FORWARD_FOLDS,
    evaluation_folds: int = DEFAULT_EVALUATION_FOLDS,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    slices = walk_forward_validation_slices(
        total_rows,
        embargo_days=embargo_days,
        min_train_rows=min_train_rows,
        min_validation_rows=min_validation_rows,
        max_folds=max_folds,
    )
    if len(slices) <= 1:
        return slices, slices
    evaluation_count = min(max(1, evaluation_folds), len(slices) - 1)
    return slices[:-evaluation_count], slices[-evaluation_count:]


def split_training_calibration_rows(
    features: pd.DataFrame,
    target: pd.Series,
    embargo_days: int,
    min_train_rows: int = MIN_INTERVAL_TRAIN_ROWS,
    min_calibration_rows: int = MIN_INTERVAL_CALIBRATION_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] | None:
    total_rows = len(features)
    required_rows = min_train_rows + min_calibration_rows + max(0, embargo_days)
    if total_rows < required_rows:
        return None
    calibration_rows = max(min_calibration_rows, int(round(total_rows * 0.2)))
    calibration_rows = min(calibration_rows, total_rows - min_train_rows - max(0, embargo_days))
    if calibration_rows < min_calibration_rows:
        return None
    calibration_start = total_rows - calibration_rows
    training_end = calibration_start - max(0, embargo_days)
    if training_end < min_train_rows:
        return None
    return (
        features.iloc[:training_end],
        features.iloc[calibration_start:],
        target.iloc[:training_end],
        target.iloc[calibration_start:],
    )


def calibrate_interval_adjustment(
    features: pd.DataFrame,
    target: pd.Series,
    regressor_params: dict[str, Any] | None,
    embargo_days: int,
) -> float:
    split = split_training_calibration_rows(features, target, embargo_days=embargo_days)
    if split is None:
        return 0.0
    fit_x, calibration_x, fit_y, calibration_y = split
    lower_regressor = build_regressor_model(objective="quantile", alpha=0.1, params=regressor_params)
    upper_regressor = build_regressor_model(objective="quantile", alpha=0.9, params=regressor_params)
    lower_regressor.fit(fit_x, fit_y)
    upper_regressor.fit(fit_x, fit_y)
    lower_prediction = lower_regressor.predict(calibration_x)
    upper_prediction = upper_regressor.predict(calibration_x)
    calibration_actual = calibration_y.to_numpy()
    nonconformity = np.maximum.reduce(
        [
            lower_prediction - calibration_actual,
            calibration_actual - upper_prediction,
            np.zeros(len(calibration_actual), dtype=float),
        ]
    )
    if len(nonconformity) == 0:
        return 0.0
    quantile_level = min(1.0, np.ceil((len(nonconformity) + 1) * (1.0 - INTERVAL_ALPHA)) / len(nonconformity))
    return float(np.quantile(nonconformity, quantile_level, method="higher"))


def training_validation_split(
    features: pd.DataFrame,
    target: pd.Series,
    min_train_rows: int = 80,
    min_validation_rows: int = 30,
    embargo_days: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] | None:
    try:
        slices = walk_forward_validation_slices(
            len(features),
            embargo_days=embargo_days,
            min_train_rows=min_train_rows,
            min_validation_rows=min_validation_rows,
            max_folds=1,
        )
    except ValueError:
        return None
    if not slices:
        return None
    train_end, validation_start, validation_end = slices[-1]
    return (
        features.iloc[:train_end],
        features.iloc[validation_start:validation_end],
        target.iloc[:train_end],
        target.iloc[validation_start:validation_end],
    )


def tune_regressor_hyperparameters(
    features: pd.DataFrame,
    target: pd.Series,
    max_horizon_days: int,
    embargo_days: int,
) -> tuple[dict[str, Any], int]:
    tuning_slices, _ = split_nested_walk_forward_slices(
        len(features),
        embargo_days=embargo_days,
        min_train_rows=90,
        min_validation_rows=25,
        max_folds=DEFAULT_MAX_WALK_FORWARD_FOLDS,
        evaluation_folds=DEFAULT_EVALUATION_FOLDS,
    )
    if not tuning_slices:
        return base_regressor_params(), 0
    trial_count = choose_optuna_trials(len(features), max_horizon_days)

    sampler = optuna.samplers.TPESampler(seed=OPTUNA_RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.trial.Trial) -> float:
        trial_params = optuna_regressor_space(trial)
        fold_scores: list[float] = []
        for train_end, validation_start, validation_end in tuning_slices:
            train_x = features.iloc[:train_end]
            validation_x = features.iloc[validation_start:validation_end]
            train_y = target.iloc[:train_end]
            validation_y = target.iloc[validation_start:validation_end]
            model = build_regressor_model(objective="regression", params=trial_params)
            model.fit(train_x, train_y)
            prediction = model.predict(validation_x)
            fold_scores.append(mean_absolute_error(validation_y, prediction))
        return float(np.mean(fold_scores))

    study.optimize(objective, n_trials=trial_count, show_progress_bar=False)
    return {**base_regressor_params(), **study.best_params}, trial_count


def tune_classifier_hyperparameters(
    features: pd.DataFrame,
    target: pd.Series,
    max_horizon_days: int,
    embargo_days: int,
) -> tuple[dict[str, Any], int]:
    tuning_slices, _ = split_nested_walk_forward_slices(
        len(features),
        embargo_days=embargo_days,
        min_train_rows=90,
        min_validation_rows=25,
        max_folds=DEFAULT_MAX_WALK_FORWARD_FOLDS,
        evaluation_folds=DEFAULT_EVALUATION_FOLDS,
    )
    if not tuning_slices:
        return base_classifier_params(), 0
    trial_count = choose_optuna_trials(len(features), max_horizon_days)

    sampler = optuna.samplers.TPESampler(seed=OPTUNA_RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.trial.Trial) -> float:
        trial_params = optuna_classifier_space(trial)
        fold_scores: list[float] = []
        for train_end, validation_start, validation_end in tuning_slices:
            train_x = features.iloc[:train_end]
            validation_x = features.iloc[validation_start:validation_end]
            train_y = target.iloc[:train_end]
            validation_y = target.iloc[validation_start:validation_end]
            model = build_classifier_model(params=trial_params)
            model.fit(train_x, train_y)
            probability = model.predict_proba(validation_x)[:, 1]
            fold_scores.append(brier_score(validation_y, probability))
        return float(np.mean(fold_scores))

    study.optimize(objective, n_trials=trial_count, show_progress_bar=False)
    return {**base_classifier_params(), **study.best_params}, trial_count


def mean_absolute_error(actual: pd.Series, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual.to_numpy() - predicted)))


def root_mean_squared_error(actual: pd.Series, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual.to_numpy() - predicted) ** 2)))


def classification_accuracy(actual: pd.Series, predicted: np.ndarray) -> float:
    return float((actual.to_numpy() == predicted).mean())


def brier_score(actual: pd.Series, probability: np.ndarray) -> float:
    return float(np.mean((actual.to_numpy() - probability) ** 2))


def evaluate_horizon_models(
    features: pd.DataFrame,
    target: pd.Series,
    probability_target: pd.Series,
    row_index: pd.Index,
    regressor_params: dict[str, Any] | None = None,
    classifier_params: dict[str, Any] | None = None,
    embargo_days: int = 1,
    horizon: int = 1,
    relative_return_1d_column: str | None = None,
    relative_return_5d_column: str | None = None,
    signal_relative_return_1d_column: str | None = None,
    signal_relative_return_5d_column: str | None = None,
    peer_training_bundles: tuple[PeerTrainingBundle, ...] = (),
    threshold_return: float = 0.0,
    stored_calibrator: Any | None = None,
    stored_trust_model: Any | None = None,
) -> dict[str, float | int | str]:
    _, evaluation_slices = split_nested_walk_forward_slices(
        len(features),
        embargo_days=embargo_days,
        min_train_rows=180,
        min_validation_rows=40,
        max_folds=DEFAULT_MAX_WALK_FORWARD_FOLDS,
        evaluation_folds=DEFAULT_EVALUATION_FOLDS,
    )
    if not evaluation_slices:
        raise ValueError("Not enough rows to run walk-forward validation.")

    fold_metrics: list[dict[str, float | int | str]] = []
    for train_end, validation_start, validation_end in evaluation_slices:
        train_x = features.iloc[:train_end]
        validation_x = features.iloc[validation_start:validation_end]
        train_y = target.iloc[:train_end]
        validation_y = target.iloc[validation_start:validation_end]
        train_probability_y = probability_target.iloc[:train_end]
        validation_probability_y = probability_target.iloc[validation_start:validation_end]

        regressor = build_regressor_model(objective="regression", params=regressor_params)
        regressor.fit(train_x, train_y)
        model_train_prediction = regressor.predict(train_x)
        model_validation_prediction = regressor.predict(validation_x)
        interval_adjustment = calibrate_interval_adjustment(
            train_x,
            train_y,
            regressor_params=regressor_params,
            embargo_days=embargo_days,
        )
        lower_regressor = build_regressor_model(objective="quantile", alpha=0.1, params=regressor_params)
        lower_regressor.fit(train_x, train_y)
        lower_validation_prediction = lower_regressor.predict(validation_x) - interval_adjustment
        upper_regressor = build_regressor_model(objective="quantile", alpha=0.9, params=regressor_params)
        upper_regressor.fit(train_x, train_y)
        upper_validation_prediction = upper_regressor.predict(validation_x) + interval_adjustment

        median_prediction = float(train_y.median())
        zero_prediction = 0.0
        dummy_train_prediction = np.full(len(train_y), median_prediction)
        dummy_validation_prediction = np.full(len(validation_y), median_prediction)
        zero_train_prediction = np.full(len(train_y), zero_prediction)
        zero_validation_prediction = np.full(len(validation_y), zero_prediction)
        last_train_prediction = train_x["log_return_1d"].to_numpy()
        last_validation_prediction = validation_x["log_return_1d"].to_numpy()
        rolling_mean_train_prediction = (train_x["log_return_5d"] / 5.0).to_numpy()
        rolling_mean_validation_prediction = (validation_x["log_return_5d"] / 5.0).to_numpy()
        if relative_return_1d_column and relative_return_1d_column in train_x.columns:
            relative_last_train_prediction = train_x[relative_return_1d_column].to_numpy() * horizon
            relative_last_validation_prediction = validation_x[relative_return_1d_column].to_numpy() * horizon
        else:
            relative_last_train_prediction = np.zeros(len(train_y))
            relative_last_validation_prediction = np.zeros(len(validation_y))
        if relative_return_5d_column and relative_return_5d_column in train_x.columns:
            relative_momentum_train_prediction = train_x[relative_return_5d_column].to_numpy() * (horizon / 5.0)
            relative_momentum_validation_prediction = validation_x[relative_return_5d_column].to_numpy() * (horizon / 5.0)
        else:
            relative_momentum_train_prediction = np.zeros(len(train_y))
            relative_momentum_validation_prediction = np.zeros(len(validation_y))

        classifier_train_x, classifier_train_y, peer_classifier_rows = augment_classifier_training_set(
            train_x,
            train_probability_y,
            peer_training_bundles=peer_training_bundles,
            horizon=horizon,
            threshold_return=threshold_return,
            peer_target_column=f"target_signal_excess_log_return_{horizon}d",
            as_of_date=pd.Timestamp(row_index[train_end - 1]),
        )
        fold_calibrator, fold_trust_model = fit_probability_calibration_and_trust(
            train_x,
            train_y,
            train_probability_y,
            regressor_params=regressor_params,
            classifier_params=classifier_params,
            embargo_days=embargo_days,
            horizon=horizon,
            peer_training_bundles=peer_training_bundles,
            threshold_return=threshold_return,
        )
        classifier = build_classifier_model(params=classifier_params)
        classifier.fit(classifier_train_x, classifier_train_y)
        raw_train_probability = classifier.predict_proba(train_x)[:, 1]
        raw_validation_probability = classifier.predict_proba(validation_x)[:, 1]
        model_train_probability = apply_probability_calibrator(fold_calibrator, raw_train_probability)
        model_validation_probability = apply_probability_calibrator(fold_calibrator, raw_validation_probability)
        trust_train_probability = predict_trust_probability(
            fold_trust_model,
            build_trust_feature_frame(train_x, raw_train_probability, model_train_probability, model_train_prediction),
        )
        trust_validation_probability = predict_trust_probability(
            fold_trust_model,
            build_trust_feature_frame(validation_x, raw_validation_probability, model_validation_probability, model_validation_prediction),
        )
        trusted_train_probability = model_train_probability * trust_train_probability
        trusted_validation_probability = model_validation_probability * trust_validation_probability
        model_train_class = (model_train_probability >= 0.5).astype(int)
        model_validation_class = (model_validation_probability >= 0.5).astype(int)
        trusted_train_class = (trusted_train_probability >= 0.5).astype(int)
        trusted_validation_class = (trusted_validation_probability >= 0.5).astype(int)

        dummy_class_probability = float(train_probability_y.mean())
        dummy_train_probability = np.full(len(train_probability_y), dummy_class_probability)
        dummy_validation_probability = np.full(len(validation_probability_y), dummy_class_probability)
        dummy_train_class = np.full(len(train_probability_y), int(dummy_class_probability >= 0.5))
        dummy_validation_class = np.full(len(validation_probability_y), int(dummy_class_probability >= 0.5))
        if signal_relative_return_1d_column and signal_relative_return_1d_column in train_x.columns:
            signal_last_train_prediction = train_x[signal_relative_return_1d_column].to_numpy() * horizon
            signal_last_validation_prediction = validation_x[signal_relative_return_1d_column].to_numpy() * horizon
        else:
            signal_last_train_prediction = last_train_prediction
            signal_last_validation_prediction = last_validation_prediction
        if signal_relative_return_5d_column and signal_relative_return_5d_column in train_x.columns:
            signal_rolling_train_prediction = train_x[signal_relative_return_5d_column].to_numpy() * (horizon / 5.0)
            signal_rolling_validation_prediction = validation_x[signal_relative_return_5d_column].to_numpy() * (horizon / 5.0)
        else:
            signal_rolling_train_prediction = rolling_mean_train_prediction
            signal_rolling_validation_prediction = rolling_mean_validation_prediction
        last_return_train_class = (signal_last_train_prediction >= 0.0).astype(int)
        last_return_validation_class = (signal_last_validation_prediction >= 0.0).astype(int)
        rolling_mean_train_class = (signal_rolling_train_prediction >= 0.0).astype(int)
        rolling_mean_validation_class = (signal_rolling_validation_prediction >= 0.0).astype(int)
        rolling_mean_10d_train_prediction = (train_x["log_return_10d"] / 10.0).to_numpy()
        rolling_mean_10d_validation_prediction = (validation_x["log_return_10d"] / 10.0).to_numpy()
        rolling_mean_10d_train_class = (rolling_mean_10d_train_prediction >= 0.0).astype(int)
        rolling_mean_10d_validation_class = (rolling_mean_10d_validation_prediction >= 0.0).astype(int)
        validation_event_rate = float(validation_probability_y.mean())
        validation_predicted_probability_mean = float(np.mean(model_validation_probability))

        fold_metrics.append(
            {
                "train_rows": int(len(train_x)),
                "validation_rows": int(len(validation_x)),
                "train_end_index": int(train_end - 1),
                "validation_start_index": int(validation_start),
                "validation_end_index": int(validation_end - 1),
                "model_train_mae_log_return": mean_absolute_error(train_y, model_train_prediction),
                "dummy_train_mae_log_return": mean_absolute_error(train_y, dummy_train_prediction),
                "model_validation_mae_log_return": mean_absolute_error(validation_y, model_validation_prediction),
                "dummy_validation_mae_log_return": mean_absolute_error(validation_y, dummy_validation_prediction),
                "zero_validation_mae_log_return": mean_absolute_error(validation_y, zero_validation_prediction),
                "last_return_validation_mae_log_return": mean_absolute_error(validation_y, last_validation_prediction),
                "rolling_mean_validation_mae_log_return": mean_absolute_error(
                    validation_y, rolling_mean_validation_prediction
                ),
                "rolling_mean_10d_validation_mae_log_return": mean_absolute_error(
                    validation_y, rolling_mean_10d_validation_prediction
                ),
                "benchmark_match_validation_mae_log_return": mean_absolute_error(
                    validation_y, zero_validation_prediction
                ),
                "relative_last_validation_mae_log_return": mean_absolute_error(
                    validation_y, relative_last_validation_prediction
                ),
                "relative_momentum_validation_mae_log_return": mean_absolute_error(
                    validation_y, relative_momentum_validation_prediction
                ),
                "model_train_rmse_log_return": root_mean_squared_error(train_y, model_train_prediction),
                "dummy_train_rmse_log_return": root_mean_squared_error(train_y, dummy_train_prediction),
                "model_validation_rmse_log_return": root_mean_squared_error(validation_y, model_validation_prediction),
                "dummy_validation_rmse_log_return": root_mean_squared_error(validation_y, dummy_validation_prediction),
                "model_train_threshold_accuracy": classification_accuracy(train_probability_y, model_train_class),
                "dummy_train_threshold_accuracy": classification_accuracy(train_probability_y, dummy_train_class),
                "model_validation_threshold_accuracy": classification_accuracy(
                    validation_probability_y, model_validation_class
                ),
                "dummy_validation_threshold_accuracy": classification_accuracy(
                    validation_probability_y, dummy_validation_class
                ),
                "last_return_validation_threshold_accuracy": classification_accuracy(
                    validation_probability_y, last_return_validation_class
                ),
                "rolling_mean_validation_threshold_accuracy": classification_accuracy(
                    validation_probability_y, rolling_mean_validation_class
                ),
                "rolling_mean_10d_validation_threshold_accuracy": classification_accuracy(
                    validation_probability_y, rolling_mean_10d_validation_class
                ),
                "model_train_brier": brier_score(train_probability_y, model_train_probability),
                "dummy_train_brier": brier_score(train_probability_y, dummy_train_probability),
                "model_validation_brier": brier_score(validation_probability_y, model_validation_probability),
                "dummy_validation_brier": brier_score(validation_probability_y, dummy_validation_probability),
                "trusted_validation_brier": brier_score(validation_probability_y, trusted_validation_probability),
                "trusted_validation_threshold_accuracy": classification_accuracy(
                    validation_probability_y, trusted_validation_class
                ),
                "mean_trust_probability": float(np.mean(trust_validation_probability)),
                "classifier_training_rows": int(len(classifier_train_x)),
                "classifier_peer_rows": int(peer_classifier_rows),
                "validation_event_rate": validation_event_rate,
                "validation_predicted_probability_mean": validation_predicted_probability_mean,
                "validation_probability_calibration_gap": validation_predicted_probability_mean - validation_event_rate,
                "model_validation_interval_coverage": float(
                    ((validation_y.to_numpy() >= lower_validation_prediction) & (validation_y.to_numpy() <= upper_validation_prediction)).mean()
                ),
                "model_validation_interval_width": float(np.mean(upper_validation_prediction - lower_validation_prediction)),
                "interval_adjustment": float(interval_adjustment),
                "model_validation_bias_log_return": float(np.mean(validation_y.to_numpy() - model_validation_prediction)),
                "validation_tp": int(((validation_probability_y.to_numpy() == 1) & (model_validation_class == 1)).sum()),
                "validation_tn": int(((validation_probability_y.to_numpy() == 0) & (model_validation_class == 0)).sum()),
                "validation_fp": int(((validation_probability_y.to_numpy() == 0) & (model_validation_class == 1)).sum()),
                "validation_fn": int(((validation_probability_y.to_numpy() == 1) & (model_validation_class == 0)).sum()),
                "mae_log_return": mean_absolute_error(validation_y, model_validation_prediction),
                "directional_accuracy": classification_accuracy(validation_probability_y, model_validation_class),
                "dummy_mae_log_return": mean_absolute_error(validation_y, dummy_validation_prediction),
                "dummy_directional_accuracy": classification_accuracy(validation_probability_y, dummy_validation_class),
            }
        )

    def fold_mean(metric_name: str) -> float:
        return float(np.mean([float(fold[metric_name]) for fold in fold_metrics]))

    first_validation_start = int(fold_metrics[0]["validation_start_index"])
    last_train_end = int(fold_metrics[-1]["train_end_index"])
    last_validation_end = int(fold_metrics[-1]["validation_end_index"])
    return {
        "train_rows": int(fold_metrics[-1]["train_rows"]),
        "validation_rows": int(sum(int(fold["validation_rows"]) for fold in fold_metrics)),
        "train_start_date": str(pd.Timestamp(row_index[0]).date()),
        "train_end_date": str(pd.Timestamp(row_index[last_train_end]).date()),
        "validation_start_date": str(pd.Timestamp(row_index[first_validation_start]).date()),
        "validation_end_date": str(pd.Timestamp(row_index[last_validation_end]).date()),
        "walk_forward_folds": int(len(evaluation_slices)),
        "tuning_walk_forward_folds": int(
            max(
                0,
                len(
                    split_nested_walk_forward_slices(
                        len(features),
                        embargo_days=embargo_days,
                        min_train_rows=180,
                        min_validation_rows=40,
                        max_folds=DEFAULT_MAX_WALK_FORWARD_FOLDS,
                        evaluation_folds=DEFAULT_EVALUATION_FOLDS,
                    )[0]
                ),
            )
        ),
        "embargo_days": int(embargo_days),
        "model_train_mae_log_return": fold_mean("model_train_mae_log_return"),
        "dummy_train_mae_log_return": fold_mean("dummy_train_mae_log_return"),
        "model_validation_mae_log_return": fold_mean("model_validation_mae_log_return"),
        "dummy_validation_mae_log_return": fold_mean("dummy_validation_mae_log_return"),
        "zero_validation_mae_log_return": fold_mean("zero_validation_mae_log_return"),
        "last_return_validation_mae_log_return": fold_mean("last_return_validation_mae_log_return"),
        "rolling_mean_validation_mae_log_return": fold_mean("rolling_mean_validation_mae_log_return"),
        "rolling_mean_10d_validation_mae_log_return": fold_mean("rolling_mean_10d_validation_mae_log_return"),
        "benchmark_match_validation_mae_log_return": fold_mean("benchmark_match_validation_mae_log_return"),
        "relative_last_validation_mae_log_return": fold_mean("relative_last_validation_mae_log_return"),
        "relative_momentum_validation_mae_log_return": fold_mean("relative_momentum_validation_mae_log_return"),
        "model_train_rmse_log_return": fold_mean("model_train_rmse_log_return"),
        "dummy_train_rmse_log_return": fold_mean("dummy_train_rmse_log_return"),
        "model_validation_rmse_log_return": fold_mean("model_validation_rmse_log_return"),
        "dummy_validation_rmse_log_return": fold_mean("dummy_validation_rmse_log_return"),
        "model_train_threshold_accuracy": fold_mean("model_train_threshold_accuracy"),
        "dummy_train_threshold_accuracy": fold_mean("dummy_train_threshold_accuracy"),
        "model_validation_threshold_accuracy": fold_mean("model_validation_threshold_accuracy"),
        "dummy_validation_threshold_accuracy": fold_mean("dummy_validation_threshold_accuracy"),
        "last_return_validation_threshold_accuracy": fold_mean("last_return_validation_threshold_accuracy"),
        "rolling_mean_validation_threshold_accuracy": fold_mean("rolling_mean_validation_threshold_accuracy"),
        "rolling_mean_10d_validation_threshold_accuracy": fold_mean("rolling_mean_10d_validation_threshold_accuracy"),
        "model_train_brier": fold_mean("model_train_brier"),
        "dummy_train_brier": fold_mean("dummy_train_brier"),
        "model_validation_brier": fold_mean("model_validation_brier"),
        "dummy_validation_brier": fold_mean("dummy_validation_brier"),
        "trusted_validation_brier": fold_mean("trusted_validation_brier"),
        "trusted_validation_threshold_accuracy": fold_mean("trusted_validation_threshold_accuracy"),
        "mean_trust_probability": fold_mean("mean_trust_probability"),
        "classifier_training_rows": int(round(fold_mean("classifier_training_rows"))),
        "classifier_peer_rows": int(round(fold_mean("classifier_peer_rows"))),
        "validation_event_rate": fold_mean("validation_event_rate"),
        "validation_predicted_probability_mean": fold_mean("validation_predicted_probability_mean"),
        "validation_probability_calibration_gap": fold_mean("validation_probability_calibration_gap"),
        "model_validation_interval_coverage": fold_mean("model_validation_interval_coverage"),
        "model_validation_interval_width": fold_mean("model_validation_interval_width"),
        "interval_adjustment": fold_mean("interval_adjustment"),
        "model_validation_bias_log_return": fold_mean("model_validation_bias_log_return"),
        "validation_tp": int(sum(int(fold["validation_tp"]) for fold in fold_metrics)),
        "validation_tn": int(sum(int(fold["validation_tn"]) for fold in fold_metrics)),
        "validation_fp": int(sum(int(fold["validation_fp"]) for fold in fold_metrics)),
        "validation_fn": int(sum(int(fold["validation_fn"]) for fold in fold_metrics)),
        "mae_log_return": fold_mean("mae_log_return"),
        "directional_accuracy": fold_mean("directional_accuracy"),
        "dummy_mae_log_return": fold_mean("dummy_mae_log_return"),
        "dummy_directional_accuracy": fold_mean("dummy_directional_accuracy"),
    }


def best_baseline_mae(metrics: dict[str, Any]) -> float:
    return float(
        np.nanmin(
            [
                metrics.get("dummy_validation_mae_log_return", np.nan),
                metrics.get("zero_validation_mae_log_return", np.nan),
                metrics.get("last_return_validation_mae_log_return", np.nan),
                metrics.get("rolling_mean_validation_mae_log_return", np.nan),
                metrics.get("rolling_mean_10d_validation_mae_log_return", np.nan),
                metrics.get("benchmark_match_validation_mae_log_return", np.nan),
                metrics.get("relative_last_validation_mae_log_return", np.nan),
                metrics.get("relative_momentum_validation_mae_log_return", np.nan),
            ]
        )
    )


def best_baseline_accuracy(metrics: dict[str, Any]) -> float:
    return float(
        np.nanmax(
            [
                metrics.get("dummy_validation_threshold_accuracy", np.nan),
                metrics.get("last_return_validation_threshold_accuracy", np.nan),
                metrics.get("rolling_mean_validation_threshold_accuracy", np.nan),
                metrics.get("rolling_mean_10d_validation_threshold_accuracy", np.nan),
            ]
        )
    )


def evaluate_deployment_gate(
    metrics: dict[str, Any],
    predicted_return: float,
    recent_residual_drift: float | None,
    trust_probability: float | None = None,
    trade_probability: float | None = None,
    transaction_cost_rate: float = DEFAULT_TRANSACTION_COST_RATE,
    safety_margin_rate: float = DEFAULT_SAFETY_MARGIN_RATE,
) -> dict[str, Any]:
    best_mae = best_baseline_mae(metrics)
    best_accuracy = best_baseline_accuracy(metrics)
    reasons: list[str] = []
    if metrics.get("model_validation_mae_log_return", np.inf) > best_mae * DEPLOYMENT_MAE_EDGE:
        reasons.append("mae_edge_too_small")
    if metrics.get("model_validation_threshold_accuracy", -np.inf) <= best_accuracy:
        reasons.append("directional_accuracy_not_above_baseline")
    if metrics.get("model_validation_interval_coverage", -np.inf) < MIN_INTERVAL_COVERAGE:
        reasons.append("interval_coverage_too_low")
    if abs(metrics.get("model_validation_bias_log_return", np.inf)) >= MAX_ABS_VALIDATION_BIAS:
        reasons.append("validation_bias_too_large")
    if predicted_return <= transaction_cost_rate + safety_margin_rate:
        reasons.append("predicted_edge_below_cost_buffer")
    if recent_residual_drift is not None and abs(recent_residual_drift) >= safety_margin_rate:
        reasons.append("recent_residual_drift_too_large")
    if trust_probability is not None and trust_probability < 0.55:
        reasons.append("trust_probability_too_low")
    if trade_probability is not None and trade_probability < 0.20:
        reasons.append("trade_probability_too_low")
    return {
        "approved": not reasons,
        "best_baseline_mae": best_mae,
        "best_baseline_accuracy": best_accuracy,
        "reasons": reasons,
        "suggested_action": "TRADE" if not reasons else "NO_TRADE",
    }


def select_primary_decision_horizon(
    forecast: pd.DataFrame,
    validation_metrics: dict[int, dict[str, Any]],
    max_decision_horizon: int = 3,
) -> int:
    if forecast.empty:
        return 1

    candidate_rows = forecast.loc[forecast["horizon_day"] <= min(max_decision_horizon, len(forecast))]
    if candidate_rows.empty:
        return int(forecast.iloc[-1]["horizon_day"])

    best_horizon = int(candidate_rows.iloc[0]["horizon_day"])
    best_score: tuple[float, float, float, float, float] | None = None
    for _, row in candidate_rows.iterrows():
        horizon = int(row["horizon_day"])
        metrics = validation_metrics.get(horizon, {})
        trusted_accuracy = float(
            metrics.get(
                "trusted_validation_threshold_accuracy",
                metrics.get("model_validation_threshold_accuracy", float("-inf")),
            )
        )
        baseline_accuracy = best_baseline_accuracy(metrics) if metrics else float("-inf")
        accuracy_edge = trusted_accuracy - baseline_accuracy
        trade_probability = float(row.get("trade_probability", 0.0))
        trust_probability = float(row.get("trust_probability", 0.0))
        predicted_return = float(np.exp(row.get("predicted_log_return", 0.0)) - 1.0)
        # Prefer horizons where trusted accuracy clears the baseline, then stronger trade setup,
        # then higher expected return, with shorter horizons winning ties.
        score = (accuracy_edge, trade_probability, trust_probability, predicted_return, -float(horizon))
        if best_score is None or score > best_score:
            best_score = score
            best_horizon = horizon
    return best_horizon


def recent_residual_drift(forecast: pd.DataFrame, lookback: int = RESIDUAL_DRIFT_LOOKBACK) -> float | None:
    observed = forecast.loc[forecast["residual"].notna(), "residual"].tail(lookback)
    if observed.empty:
        return None
    return float(observed.mean())


def save_forecast_artifacts(artifacts: ForecastArtifacts) -> None:
    joblib.dump(artifacts, model_file_for_ticker(artifacts.ticker))
    metadata = {
        "ticker": artifacts.ticker,
        "max_horizon_days": artifacts.max_horizon_days,
        "feature_columns": list(artifacts.feature_columns),
        "threshold_return": artifacts.threshold_return,
        "training_cutoff": artifacts.training_cutoff,
        "fitted_at": artifacts.fitted_at,
        "interval_adjustments": artifacts.interval_adjustments,
        "validation_metrics": artifacts.validation_metrics,
        "news_feature_mode": artifacts.news_feature_mode,
    }
    metadata_file_for_ticker(artifacts.ticker).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_forecast_artifacts(ticker: str) -> ForecastArtifacts:
    model_path = model_file_for_ticker(ticker)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved forecast model exists for {ticker.upper()}. Train the model first in the Stock Forecast tab."
        )
    artifacts = joblib.load(model_path)
    if not isinstance(artifacts, ForecastArtifacts):
        raise TypeError(f"Saved artifact for {ticker.upper()} is not a ForecastArtifacts instance.")
    return artifacts


def next_trading_dates(index: pd.DatetimeIndex, prediction_start: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    normalized_index = pd.DatetimeIndex(pd.to_datetime(index)).tz_localize(None).sort_values()
    prediction_start = pd.Timestamp(prediction_start).tz_localize(None)
    future_dates = normalized_index[normalized_index > prediction_start]
    if len(future_dates) >= horizon:
        return pd.DatetimeIndex(future_dates[:horizon])

    generated: list[pd.Timestamp] = list(future_dates)
    current = max(prediction_start, normalized_index.max())
    while len(generated) < horizon:
        current = current + pd.offsets.BDay(1)
        generated.append(pd.Timestamp(current))
    return pd.DatetimeIndex(generated[:horizon])


def resolve_prediction_start_date(index: pd.DatetimeIndex, prediction_start_date: str | pd.Timestamp) -> pd.Timestamp:
    normalized_index = pd.DatetimeIndex(pd.to_datetime(index)).tz_localize(None).normalize().sort_values()
    requested_date = pd.Timestamp(prediction_start_date).tz_localize(None).normalize()
    available_dates = normalized_index[normalized_index <= requested_date]
    if available_dates.empty:
        earliest_date = normalized_index.min().date()
        raise ValueError(
            f"Prediction start date {requested_date.date()} is before the first available market date {earliest_date}."
        )
    return pd.Timestamp(available_dates[-1])


def generate_forecast(
    ticker: str,
    prices: pd.DataFrame,
    prediction_start_date: str,
    forecast_horizon_days: int,
    artifacts: ForecastArtifacts | None = None,
    use_openai_news: bool = True,
    allow_news_refresh: bool = True,
) -> dict[str, Any]:
    artifacts = artifacts or load_forecast_artifacts(ticker)
    normalized_prices = prices.sort_index().copy()
    cutoff = resolve_prediction_start_date(normalized_prices.index, prediction_start_date)
    if forecast_horizon_days > artifacts.max_horizon_days:
        raise ValueError(
            f"Requested horizon {forecast_horizon_days} exceeds trained horizon {artifacts.max_horizon_days}."
        )

    news_items, news_mode = fetch_historical_news_features(
        ticker=ticker,
        prediction_start_date=prediction_start_date,
        use_openai=use_openai_news,
        allow_network=allow_news_refresh,
    )
    news_features = aggregate_historical_news_features(news_items, prices.index)
    feature_frame = add_forecast_features(prices, news_features=news_features)
    feature_row = feature_frame.loc[cutoff, list(artifacts.feature_columns)]
    feature_input = feature_row.to_frame().T
    price_series = normalized_prices.iloc[:, 0]
    benchmark_symbol = artifacts.benchmark_symbol if artifacts.benchmark_symbol in normalized_prices.columns else benchmark_symbol_from_prices(normalized_prices, ticker.upper())
    signal_benchmark_symbol = signal_benchmark_symbol_from_prices(normalized_prices, ticker.upper())
    benchmark_series = normalized_prices[benchmark_symbol]
    interval_adjustments = getattr(artifacts, "interval_adjustments", {})
    probability_calibrators = getattr(artifacts, "probability_calibrators", {})
    trust_models = getattr(artifacts, "trust_models", {})
    current_price = float(price_series.loc[cutoff])
    forecast_index = next_trading_dates(prices.index, cutoff, forecast_horizon_days)
    history = price_series.loc[price_series.index <= cutoff].rename("price").reset_index()
    history.columns = ["date", "price"]

    rows: list[dict[str, Any]] = []
    for horizon in range(1, forecast_horizon_days + 1):
        forecast_date = forecast_index[horizon - 1]
        expected_benchmark_log_return = estimate_benchmark_forward_log_return(benchmark_series, cutoff, horizon)
        predicted_excess_log_return = float(artifacts.median_models[horizon].predict(feature_input)[0])
        interval_adjustment = float(interval_adjustments.get(horizon, 0.0))
        lower_excess_log_return = float(artifacts.lower_models[horizon].predict(feature_input)[0]) - interval_adjustment
        upper_excess_log_return = float(artifacts.upper_models[horizon].predict(feature_input)[0]) + interval_adjustment
        median_log_return = predicted_excess_log_return + expected_benchmark_log_return
        lower_log_return = lower_excess_log_return + expected_benchmark_log_return
        upper_log_return = upper_excess_log_return + expected_benchmark_log_return
        raw_probability_threshold_hit = float(artifacts.probability_models[horizon].predict_proba(feature_input)[0, 1])
        probability_threshold_hit = float(
            apply_probability_calibrator(
                probability_calibrators.get(horizon),
                np.asarray([raw_probability_threshold_hit], dtype=float),
            )[0]
        )
        trust_probability = float(
            predict_trust_probability(
                trust_models.get(horizon),
                build_trust_feature_frame(
                    feature_input,
                    np.asarray([raw_probability_threshold_hit], dtype=float),
                    np.asarray([probability_threshold_hit], dtype=float),
                    np.asarray([predicted_excess_log_return], dtype=float),
                ),
            )[0]
        )
        trade_probability = float(probability_threshold_hit * trust_probability)
        trade_decision = "TRADE" if trade_probability >= 0.20 and trust_probability >= 0.55 else "NO_TRADE"
        actual_price = float(price_series.loc[forecast_date]) if forecast_date in price_series.index else np.nan
        predicted_price = current_price * float(np.exp(median_log_return))
        rows.append(
            {
                "date": forecast_date,
                "forecast_date": forecast_date,
                "horizon_day": horizon,
                "horizon_days": horizon,
                "predicted_log_return": median_log_return,
                "predicted_excess_log_return": predicted_excess_log_return,
                "lower_log_return": lower_log_return,
                "lower_excess_log_return": lower_excess_log_return,
                "upper_log_return": upper_log_return,
                "upper_excess_log_return": upper_excess_log_return,
                "expected_benchmark_log_return": expected_benchmark_log_return,
                "predicted_price": predicted_price,
                "lower_price": current_price * float(np.exp(lower_log_return)),
                "upper_price": current_price * float(np.exp(upper_log_return)),
                "raw_probability_threshold_hit": raw_probability_threshold_hit,
                "probability_threshold_hit": probability_threshold_hit,
                "trust_probability": trust_probability,
                "trade_probability": trade_probability,
                "trade_decision": trade_decision,
                "actual_price": actual_price,
                "residual": actual_price - predicted_price if pd.notna(actual_price) else np.nan,
                "interval_adjustment": interval_adjustment,
            }
        )

    forecast = pd.DataFrame(rows).set_index("date", drop=False)
    residual_drift = recent_residual_drift(forecast)
    decision_horizon = select_primary_decision_horizon(forecast, artifacts.validation_metrics) if not forecast.empty else 1
    decision_row = forecast.loc[forecast["horizon_day"] == decision_horizon].iloc[-1] if not forecast.empty else None
    decision_metrics = artifacts.validation_metrics.get(decision_horizon, {}) if decision_row is not None else {}
    decision_gate = evaluate_deployment_gate(
        decision_metrics,
        predicted_return=float(np.exp(decision_row["predicted_log_return"]) - 1.0) if decision_row is not None else 0.0,
        recent_residual_drift=residual_drift,
        trust_probability=float(decision_row["trust_probability"]) if decision_row is not None else None,
        trade_probability=float(decision_row["trade_probability"]) if decision_row is not None else None,
    ) if decision_metrics else {}
    return {
        "ticker": ticker.upper(),
        "prediction_start_date": str(cutoff.date()),
        "anchor_price": current_price,
        "history": history,
        "forecast": forecast,
        "metadata": {
            "ticker": artifacts.ticker,
            "benchmark_symbol": benchmark_symbol,
            "signal_benchmark_symbol": signal_benchmark_symbol,
            "max_horizon_days": artifacts.max_horizon_days,
            "threshold_return": artifacts.threshold_return,
            "training_cutoff": artifacts.training_cutoff,
            "fitted_at": artifacts.fitted_at,
            "validation_metrics": artifacts.validation_metrics,
            "recent_residual_drift": residual_drift,
            "decision_horizon": decision_horizon,
            "deployment_gate": decision_gate,
            "suggested_action": decision_gate.get("suggested_action") if decision_gate else None,
            "news_feature_mode": news_mode if not news_items.empty else artifacts.news_feature_mode,
        },
    }
