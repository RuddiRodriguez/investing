#!/usr/bin/env python3
"""
Transformer outperformance trading script.

What this script predicts:
    Probability that TICKER will outperform BENCHMARK over the next N trading days.

Important:
    The model is a classifier. It does NOT directly predict an exact future price.
    Any "approximate future price" in this script is derived from historical conditional
    return distributions and the model probability. It is only a rough scenario estimate,
    not a true price forecast.

Main fixes versus the original notebook:
    1. Drops rows where future returns are NaN before creating usable training rows.
       This avoids labeling unresolved future rows as 0.
    2. Removes the old price plot that used realized future returns to construct
       "predicted" prices. That was leakage.
    3. Adds clearer plots:
       - training/validation learning curve
       - probability of outperformance versus actual outcome
       - approximate future price scenario without using future realized returns
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Reproducibility / device
# -----------------------------

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device() -> torch.device:
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = select_device()


# -----------------------------
# Config
# -----------------------------

@dataclass
class Config:
    ticker: str = "NVDA"
    benchmark: str = "SPY"
    start_date: str = "2016-01-01"
    end_date: str | None = None
    sequence_length: int = 32
    forecast_horizon_days: int = 5
    train_ratio: float = 0.70
    valid_ratio: float = 0.15
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    bullish_threshold: float = 0.60
    bearish_threshold: float = 0.40
    output_dir: str = "transformer_trading_output"
    focus_date: str | None = None
    window_back_days: int = 7
    window_forward_days: int = 5


# -----------------------------
# Data and features
# -----------------------------

def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def download_ohlcv(ticker: str, benchmark: str, start: str, end: str | None = None) -> pd.DataFrame:
    raw = yf.download(
        [ticker.upper(), benchmark.upper()],
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw.empty:
        raise ValueError("No market data was returned from Yahoo Finance.")

    def extract_symbol_frame(symbol: str) -> pd.DataFrame:
        symbol = symbol.upper()
        if isinstance(raw.columns, pd.MultiIndex):
            frame = raw[symbol].copy()
        else:
            frame = raw.copy()
        frame.columns = [str(column).lower() for column in frame.columns]
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        return frame[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")

    stock = extract_symbol_frame(ticker)
    benchmark_frame = extract_symbol_frame(benchmark).rename(columns=lambda name: f"benchmark_{name}")
    combined = stock.join(benchmark_frame[["benchmark_close"]], how="inner")
    combined = combined.dropna().sort_index()
    if combined.empty:
        raise ValueError("No overlapping stock and benchmark history was found.")
    return combined


def add_features(
    frame: pd.DataFrame,
    forecast_horizon_days: int,
    require_future_returns: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    df = frame.copy()

    df["log_return_1d"] = np.log(df["close"] / df["close"].shift(1))
    df["log_return_5d"] = np.log(df["close"] / df["close"].shift(5))
    df["log_return_10d"] = np.log(df["close"] / df["close"].shift(10))

    df["volatility_10d"] = df["log_return_1d"].rolling(10).std()
    df["volatility_20d"] = df["log_return_1d"].rolling(20).std()

    df["range_pct"] = (df["high"] - df["low"]) / df["close"]

    df["volume_log"] = np.log1p(df["volume"])
    volume_mean = df["volume_log"].rolling(20).mean()
    volume_std = df["volume_log"].rolling(20).std()
    df["volume_zscore_20d"] = (df["volume_log"] - volume_mean) / volume_std

    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["close_to_sma_10"] = df["close"] / df["sma_10"] - 1.0
    df["close_to_sma_20"] = df["close"] / df["sma_20"] - 1.0
    df["close_to_sma_50"] = df["close"] / df["sma_50"] - 1.0

    df["benchmark_log_return_5d"] = np.log(df["benchmark_close"] / df["benchmark_close"].shift(5))
    df["benchmark_rel_5d"] = df["log_return_5d"] - df["benchmark_log_return_5d"]

    df["rsi_14"] = compute_rsi(df["close"], window=14)

    # Future returns are only known after forecast_horizon_days.
    # The last rows will be NaN and must be removed before target creation is trusted.
    df["future_stock_log_return"] = np.log(df["close"].shift(-forecast_horizon_days) / df["close"])
    df["future_benchmark_log_return"] = np.log(
        df["benchmark_close"].shift(-forecast_horizon_days) / df["benchmark_close"]
    )

    feature_columns = [
        "log_return_1d",
        "log_return_5d",
        "log_return_10d",
        "volatility_10d",
        "volatility_20d",
        "range_pct",
        "volume_zscore_20d",
        "close_to_sma_10",
        "close_to_sma_20",
        "close_to_sma_50",
        "benchmark_rel_5d",
        "rsi_14",
    ]

    if require_future_returns:
        required = feature_columns + ["future_stock_log_return", "future_benchmark_log_return"]
    else:
        # For inference/plotting near the most recent date, future returns may not exist yet.
        # We keep those rows so the model can still create probability and approximate-price scenarios.
        required = feature_columns

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()

    known_future = df["future_stock_log_return"].notna() & df["future_benchmark_log_return"].notna()
    df["target_outperform"] = np.where(
        known_future,
        (df["future_stock_log_return"] > df["future_benchmark_log_return"]).astype(float),
        np.nan,
    )

    return df, feature_columns


# -----------------------------
# Dataset / model
# -----------------------------

@dataclass
class SequenceBundle:
    features: np.ndarray
    targets: np.ndarray
    dates: pd.DatetimeIndex
    close_prices: np.ndarray
    future_stock_returns: np.ndarray
    future_benchmark_returns: np.ndarray


class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, num_features: int, d_model: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_projection = nn.Linear(num_features, d_model)
        self.position_encoding = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.position_encoding(x)
        x = self.encoder(x)
        pooled = x[:, -1, :]
        return self.classifier(pooled).squeeze(-1)


def build_sequence_bundle(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    sequence_length: int,
) -> SequenceBundle:
    values = frame[list(feature_columns)].to_numpy(dtype=np.float32)
    targets = frame["target_outperform"].to_numpy(dtype=np.float32)
    dates = pd.DatetimeIndex(frame.index)
    close_prices = frame["close"].to_numpy(dtype=np.float32)
    future_stock_returns = frame["future_stock_log_return"].to_numpy(dtype=np.float32)
    future_benchmark_returns = frame["future_benchmark_log_return"].to_numpy(dtype=np.float32)

    sequence_features = []
    sequence_targets = []
    sequence_dates = []
    sequence_close = []
    sequence_future_stock_returns = []
    sequence_future_benchmark_returns = []

    for end_index in range(sequence_length, len(frame)):
        start_index = end_index - sequence_length
        sequence_features.append(values[start_index:end_index])
        sequence_targets.append(targets[end_index])
        sequence_dates.append(dates[end_index])
        sequence_close.append(close_prices[end_index])
        sequence_future_stock_returns.append(future_stock_returns[end_index])
        sequence_future_benchmark_returns.append(future_benchmark_returns[end_index])

    return SequenceBundle(
        features=np.asarray(sequence_features, dtype=np.float32),
        targets=np.asarray(sequence_targets, dtype=np.float32),
        dates=pd.DatetimeIndex(sequence_dates),
        close_prices=np.asarray(sequence_close, dtype=np.float32),
        future_stock_returns=np.asarray(sequence_future_stock_returns, dtype=np.float32),
        future_benchmark_returns=np.asarray(sequence_future_benchmark_returns, dtype=np.float32),
    )


def split_bundle(bundle: SequenceBundle, train_ratio: float, valid_ratio: float):
    total = len(bundle.features)
    train_end = int(total * train_ratio)
    valid_end = int(total * (train_ratio + valid_ratio))

    train_slice = slice(0, train_end)
    valid_slice = slice(train_end, valid_end)
    test_slice = slice(valid_end, total)

    scaler = StandardScaler()
    train_2d = bundle.features[train_slice].reshape(-1, bundle.features.shape[-1])
    scaler.fit(train_2d)

    def transform(features: np.ndarray) -> np.ndarray:
        flat = features.reshape(-1, features.shape[-1])
        scaled = scaler.transform(flat)
        return scaled.reshape(features.shape).astype(np.float32)

    partitions = {}
    for name, subset_slice in {"train": train_slice, "valid": valid_slice, "test": test_slice}.items():
        partitions[name] = {
            "features": transform(bundle.features[subset_slice]),
            "targets": bundle.targets[subset_slice],
            "dates": bundle.dates[subset_slice],
            "close_prices": bundle.close_prices[subset_slice],
            "future_stock_returns": bundle.future_stock_returns[subset_slice],
            "future_benchmark_returns": bundle.future_benchmark_returns[subset_slice],
        }

    return partitions, scaler


def scale_sequence_features(features: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    flat = features.reshape(-1, features.shape[-1])
    scaled = scaler.transform(flat)
    return scaled.reshape(features.shape).astype(np.float32)


@torch.no_grad()
def predict_probabilities(model: nn.Module, features: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    probabilities = []
    dataset = torch.tensor(features, dtype=torch.float32)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities)


def bundle_to_partition(bundle: SequenceBundle, scaled_features: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "features": scaled_features,
        "targets": bundle.targets,
        "dates": bundle.dates,
        "close_prices": bundle.close_prices,
        "future_stock_returns": bundle.future_stock_returns,
        "future_benchmark_returns": bundle.future_benchmark_returns,
    }


def make_loaders(partitions: dict[str, dict[str, np.ndarray]], batch_size: int):
    loaders = {}
    for name, partition in partitions.items():
        dataset = SequenceDataset(partition["features"], partition["targets"])
        loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=(name == "train"))
    return loaders


def train_one_epoch(model, loader, criterion, optimizer) -> float:
    model.train()
    losses = []
    for features, targets in loader:
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, criterion) -> dict:
    model.eval()
    losses = []
    probabilities = []
    targets_all = []

    for features, targets in loader:
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        logits = model(features)
        loss = criterion(logits, targets)
        probs = torch.sigmoid(logits)

        losses.append(loss.item())
        probabilities.append(probs.cpu().numpy())
        targets_all.append(targets.cpu().numpy())

    probabilities_np = np.concatenate(probabilities)
    targets_np = np.concatenate(targets_all)
    predictions_np = (probabilities_np >= 0.5).astype(int)
    accuracy = accuracy_score(targets_np, predictions_np)

    try:
        auc = roc_auc_score(targets_np, probabilities_np)
    except ValueError:
        auc = np.nan

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy),
        "auc": float(auc) if not np.isnan(auc) else np.nan,
        "probabilities": probabilities_np,
        "targets": targets_np,
    }


# -----------------------------
# Approximate price logic
# -----------------------------

@dataclass
class PriceScenarioParams:
    benchmark_base_log_return: float
    median_excess_if_outperform: float
    median_excess_if_underperform: float
    q20_excess_if_outperform: float
    q80_excess_if_outperform: float
    q20_excess_if_underperform: float
    q80_excess_if_underperform: float


def fit_price_scenario_params(train_partition: dict[str, np.ndarray]) -> PriceScenarioParams:
    stock = train_partition["future_stock_returns"]
    bench = train_partition["future_benchmark_returns"]
    targets = train_partition["targets"].astype(bool)
    excess = stock - bench

    if targets.sum() == 0 or (~targets).sum() == 0:
        raise ValueError("Training data needs both outperform and underperform examples.")

    return PriceScenarioParams(
        benchmark_base_log_return=float(np.median(bench)),
        median_excess_if_outperform=float(np.median(excess[targets])),
        median_excess_if_underperform=float(np.median(excess[~targets])),
        q20_excess_if_outperform=float(np.quantile(excess[targets], 0.20)),
        q80_excess_if_outperform=float(np.quantile(excess[targets], 0.80)),
        q20_excess_if_underperform=float(np.quantile(excess[~targets], 0.20)),
        q80_excess_if_underperform=float(np.quantile(excess[~targets], 0.80)),
    )


def approximate_future_prices(
    close_prices: np.ndarray,
    probabilities: np.ndarray,
    params: PriceScenarioParams,
) -> pd.DataFrame:
    """
    Converts classification probabilities into approximate price scenarios.

    This does NOT use realized future returns.
    It uses:
        expected benchmark log return = historical median benchmark future return from train
        expected excess log return = weighted historical excess return conditional on class

    p close to 1 uses more of historical outperform excess.
    p close to 0 uses more of historical underperform excess.
    """
    p = probabilities.astype(float)

    median_excess = (
        p * params.median_excess_if_outperform
        + (1.0 - p) * params.median_excess_if_underperform
    )

    low_excess = (
        p * params.q20_excess_if_outperform
        + (1.0 - p) * params.q20_excess_if_underperform
    )

    high_excess = (
        p * params.q80_excess_if_outperform
        + (1.0 - p) * params.q80_excess_if_underperform
    )

    approx_log_return = params.benchmark_base_log_return + median_excess
    low_log_return = params.benchmark_base_log_return + low_excess
    high_log_return = params.benchmark_base_log_return + high_excess

    return pd.DataFrame(
        {
            "approx_log_return": approx_log_return,
            "approx_future_close": close_prices * np.exp(approx_log_return),
            "approx_future_close_low": close_prices * np.exp(low_log_return),
            "approx_future_close_high": close_prices * np.exp(high_log_return),
            "approx_future_return_pct": (np.exp(approx_log_return) - 1.0) * 100.0,
        }
    )


@dataclass
class PriceConfidenceParams:
    residual_q05: float
    residual_q16: float
    residual_q50: float
    residual_q84: float
    residual_q95: float
    mae_log_return: float
    rmse_log_return: float
    median_absolute_price_error_pct: float
    coverage_68: float
    coverage_90: float


def fit_price_confidence_params(calibration_predictions: pd.DataFrame) -> PriceConfidenceParams:
    """
    Fits empirical confidence bands from validation-set residuals.

    residual = realized future stock log return - approximate model log return

    This is useful because the model does not predict a full price distribution.
    The interval is calibrated from historical out-of-sample errors, not from the
    raw neural network probability.
    """
    residual = (
        calibration_predictions["future_stock_log_return"].astype(float).to_numpy()
        - calibration_predictions["approx_log_return"].astype(float).to_numpy()
    )
    actual_close = calibration_predictions["actual_future_close"].astype(float).to_numpy()
    approx_close = calibration_predictions["approx_future_close"].astype(float).to_numpy()

    q05, q16, q50, q84, q95 = np.quantile(residual, [0.05, 0.16, 0.50, 0.84, 0.95])

    ci68_low = calibration_predictions["close"].astype(float).to_numpy() * np.exp(
        calibration_predictions["approx_log_return"].astype(float).to_numpy() + q16
    )
    ci68_high = calibration_predictions["close"].astype(float).to_numpy() * np.exp(
        calibration_predictions["approx_log_return"].astype(float).to_numpy() + q84
    )
    ci90_low = calibration_predictions["close"].astype(float).to_numpy() * np.exp(
        calibration_predictions["approx_log_return"].astype(float).to_numpy() + q05
    )
    ci90_high = calibration_predictions["close"].astype(float).to_numpy() * np.exp(
        calibration_predictions["approx_log_return"].astype(float).to_numpy() + q95
    )

    return PriceConfidenceParams(
        residual_q05=float(q05),
        residual_q16=float(q16),
        residual_q50=float(q50),
        residual_q84=float(q84),
        residual_q95=float(q95),
        mae_log_return=float(np.mean(np.abs(residual))),
        rmse_log_return=float(np.sqrt(np.mean(residual**2))),
        median_absolute_price_error_pct=float(np.median(np.abs(approx_close / actual_close - 1.0)) * 100.0),
        coverage_68=float(np.mean((actual_close >= ci68_low) & (actual_close <= ci68_high))),
        coverage_90=float(np.mean((actual_close >= ci90_low) & (actual_close <= ci90_high))),
    )


def add_empirical_confidence_bands(
    predictions: pd.DataFrame,
    confidence: PriceConfidenceParams,
) -> pd.DataFrame:
    """Adds empirical confidence bands around the approximate future close."""
    out = predictions.copy()
    close = out["close"].astype(float).to_numpy()
    approx_lr = out["approx_log_return"].astype(float).to_numpy()

    out["empirical_ci68_low"] = close * np.exp(approx_lr + confidence.residual_q16)
    out["empirical_ci68_high"] = close * np.exp(approx_lr + confidence.residual_q84)
    out["empirical_ci90_low"] = close * np.exp(approx_lr + confidence.residual_q05)
    out["empirical_ci90_high"] = close * np.exp(approx_lr + confidence.residual_q95)
    out["empirical_residual_median_adjusted_close"] = close * np.exp(approx_lr + confidence.residual_q50)
    return out


def build_prediction_frame(
    partition: dict[str, np.ndarray],
    result: dict,
    scenario_params: PriceScenarioParams,
) -> pd.DataFrame:
    """Builds a prediction dataframe for validation or test partitions."""
    predictions = pd.DataFrame(
        {
            "date": partition["dates"],
            "close": partition["close_prices"],
            "probability_outperform": result["probabilities"],
            "target_outperform": partition["targets"],
            "future_stock_log_return": partition["future_stock_returns"],
            "future_benchmark_log_return": partition["future_benchmark_returns"],
        }
    )
    predictions["predicted_outperform"] = (predictions["probability_outperform"] >= 0.5).astype(int)
    predictions["actual_future_close"] = predictions["close"] * np.exp(predictions["future_stock_log_return"])
    predictions["stock_future_return_pct"] = (np.exp(predictions["future_stock_log_return"]) - 1.0) * 100.0
    predictions["benchmark_future_return_pct"] = (
        np.exp(predictions["future_benchmark_log_return"]) - 1.0
    ) * 100.0

    approx_prices = approximate_future_prices(
        close_prices=predictions["close"].to_numpy(),
        probabilities=predictions["probability_outperform"].to_numpy(),
        params=scenario_params,
    )
    return pd.concat([predictions.reset_index(drop=True), approx_prices], axis=1)


# -----------------------------
# Plots
# -----------------------------

def plot_training_history(history_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(history_df["epoch"], history_df["train_loss"], label="Train loss")
    ax1.plot(history_df["epoch"], history_df["valid_loss"], label="Validation loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(history_df["epoch"], history_df["valid_auc"], label="Validation AUC", linestyle="--")
    ax2.set_ylabel("Validation AUC")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    ax1.set_title("Training history")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_probability_panel(
    predictions: pd.DataFrame,
    ticker: str,
    benchmark: str,
    horizon: int,
    bullish_threshold: float,
    bearish_threshold: float,
    output_path: Path,
) -> None:
    dates = pd.to_datetime(predictions["date"])

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(dates, predictions["probability_outperform"], label="Model probability of outperforming")
    ax.axhline(0.5, linestyle="--", linewidth=1, label="Neutral 0.50")
    ax.axhline(bullish_threshold, linestyle=":", linewidth=1.5, label=f"Bullish threshold {bullish_threshold:.2f}")
    ax.axhline(bearish_threshold, linestyle=":", linewidth=1.5, label=f"Bearish threshold {bearish_threshold:.2f}")

    actual_outperform = predictions["target_outperform"].astype(bool).values
    ax.scatter(
        dates[actual_outperform],
        predictions.loc[actual_outperform, "probability_outperform"],
        marker="^",
        s=35,
        label="Actually outperformed",
    )
    ax.scatter(
        dates[~actual_outperform],
        predictions.loc[~actual_outperform, "probability_outperform"],
        marker="v",
        s=35,
        label="Actually underperformed",
    )

    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{ticker} vs {benchmark}: probability of outperformance over next {horizon} trading days")
    ax.set_xlabel("Signal date")
    ax.set_ylabel("Probability")
    ax.grid(True, alpha=0.3)

    locator = mdates.AutoDateLocator(minticks=8, maxticks=14)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_approx_price_panel(
    predictions: pd.DataFrame,
    ticker: str,
    benchmark: str,
    horizon: int,
    output_path: Path,
) -> None:
    """
    Clear price plot:
        - close at signal date
        - actual future close, known in historical test data
        - approximate future close generated without future leakage
        - scenario band

    The approximate line is not a direct model price prediction.
    """
    dates = pd.to_datetime(predictions["date"])

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(dates, predictions["close"], label="Close at signal date", linewidth=1.5)
    ax.plot(dates, predictions["actual_future_close"], label=f"Actual close after {horizon} trading days", linewidth=1.5)
    ax.plot(
        dates,
        predictions["approx_future_close"],
        label="Approx future close from probability, no future leakage",
        linewidth=2.0,
    )
    ax.fill_between(
        dates,
        predictions["approx_future_close_low"].astype(float).values,
        predictions["approx_future_close_high"].astype(float).values,
        alpha=0.18,
        label="Approx scenario band",
    )

    ax.set_title(
        f"{ticker}: historical actual price vs approximate price scenario\n"
        f"Approx price = close * exp(median benchmark return from train + probability-weighted excess return from train)"
    )
    ax.set_xlabel("Signal date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)

    calculation_note = (
        "Calculation: approx_future_close = close_t * exp(B_train + "
        "p * E_out_train + (1-p) * E_under_train). "
        "B_train is the historical median benchmark future log return. "
        "E_out/E_under are historical median excess log returns when the stock did/did not outperform. "
        "This is an approximate scenario, not a direct price prediction."
    )
    fig.text(0.01, 0.01, calculation_note, fontsize=8, ha="left", va="bottom", wrap=True)

    locator = mdates.AutoDateLocator(minticks=8, maxticks=14)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.legend(loc="best")
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)




def plot_approx_price_with_confidence_panel(
    predictions: pd.DataFrame,
    confidence: PriceConfidenceParams,
    ticker: str,
    benchmark: str,
    horizon: int,
    output_path: Path,
) -> None:
    """
    Price plot with empirical confidence intervals calibrated on validation residuals.

    The intervals are not neural-network certainty. They are historical error bands:
        residual = actual future log return - approximate log return
    using validation-set residual quantiles.
    """
    dates = pd.to_datetime(predictions["date"])

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(dates, predictions["close"], label="Close at signal date", linewidth=1.3)
    ax.plot(dates, predictions["actual_future_close"], label=f"Actual close after {horizon} trading days", linewidth=1.4)
    ax.plot(
        dates,
        predictions["approx_future_close"],
        label="Approx future close from model probability",
        linewidth=2.0,
    )

    ax.fill_between(
        dates,
        predictions["empirical_ci90_low"].astype(float).values,
        predictions["empirical_ci90_high"].astype(float).values,
        alpha=0.12,
        label="Empirical 90% band from validation errors",
    )
    ax.fill_between(
        dates,
        predictions["empirical_ci68_low"].astype(float).values,
        predictions["empirical_ci68_high"].astype(float).values,
        alpha=0.22,
        label="Empirical 68% band from validation errors",
    )

    ax.set_title(
        f"{ticker}: approximate price scenario with empirical confidence bands\n"
        f"Bands calibrated on validation residuals; benchmark = {benchmark}; horizon = {horizon} trading days"
    )
    ax.set_xlabel("Signal date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)

    note = (
        "Approx price = close_t * exp(B_train + p*E_out_train + (1-p)*E_under_train). "
        "Confidence bands = approx_log_return + validation residual quantiles. "
        f"Validation median abs price error: {confidence.median_absolute_price_error_pct:.2f}%. "
        f"Validation coverage: 68% band captured {confidence.coverage_68:.1%}, "
        f"90% band captured {confidence.coverage_90:.1%}."
    )
    fig.text(0.01, 0.01, note, fontsize=8, ha="left", va="bottom", wrap=True)

    locator = mdates.AutoDateLocator(minticks=8, maxticks=14)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.legend(loc="best")
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_price_residuals_panel(
    calibration_predictions: pd.DataFrame,
    confidence: PriceConfidenceParams,
    ticker: str,
    horizon: int,
    output_path: Path,
) -> None:
    residual_pct = (
        calibration_predictions["actual_future_close"].astype(float).to_numpy()
        / calibration_predictions["approx_future_close"].astype(float).to_numpy()
        - 1.0
    ) * 100.0

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.hist(residual_pct, bins=35, alpha=0.75)
    ax.axvline(0.0, linestyle="--", linewidth=1.2, label="Zero error")
    ax.axvline(np.percentile(residual_pct, 5), linestyle=":", linewidth=1.2, label="5th / 95th percentiles")
    ax.axvline(np.percentile(residual_pct, 95), linestyle=":", linewidth=1.2)
    ax.set_title(f"{ticker}: validation price approximation errors over {horizon} trading days")
    ax.set_xlabel("Actual future close / approx future close - 1, %")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    note = (
        f"Median absolute price error: {confidence.median_absolute_price_error_pct:.2f}%. "
        f"MAE log-return: {confidence.mae_log_return:.4f}. "
        f"RMSE log-return: {confidence.rmse_log_return:.4f}."
    )
    fig.text(0.01, 0.01, note, fontsize=8, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_price_confidence_method(
    output_path: Path,
    ticker: str,
    benchmark: str,
    horizon: int,
    confidence: PriceConfidenceParams,
) -> None:
    text = f"""Empirical price confidence method
=================================

What the model predicts
-----------------------
The Transformer predicts probability of outperformance, not an exact price.

How the approximate price is created
------------------------------------
approx_future_close = close_t * exp(B_train + p * E_out_train + (1 - p) * E_under_train)

where p is the model probability that {ticker} outperforms {benchmark} over the next {horizon} trading days.

How the confidence bands are created
------------------------------------
The bands are empirical error bands calibrated on the validation set.

For every validation prediction:

    residual = actual_future_stock_log_return - approx_log_return

Then the script takes residual quantiles:

    68% band: residual q16 to q84
    90% band: residual q05 to q95

For a new signal:

    lower_band = close_t * exp(approx_log_return + residual_low_quantile)
    upper_band = close_t * exp(approx_log_return + residual_high_quantile)

Interpretation
--------------
These are approximate empirical confidence intervals, not guaranteed statistical confidence intervals.
They answer: historically, on the validation period, how far was the approximate price from the realized price?

Calibration diagnostics from validation set
-------------------------------------------
median_absolute_price_error_pct: {confidence.median_absolute_price_error_pct:.4f}%
mae_log_return: {confidence.mae_log_return:.8f}
rmse_log_return: {confidence.rmse_log_return:.8f}
coverage_68: {confidence.coverage_68:.4f}
coverage_90: {confidence.coverage_90:.4f}

Residual quantiles
------------------
q05: {confidence.residual_q05:.8f}
q16: {confidence.residual_q16:.8f}
q50: {confidence.residual_q50:.8f}
q84: {confidence.residual_q84:.8f}
q95: {confidence.residual_q95:.8f}
"""
    output_path.write_text(text, encoding="utf-8")


def select_focus_window(
    predictions: pd.DataFrame,
    focus_date: str,
    back_days: int,
    forward_days: int,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Selects a row window around a focus signal date using available trading rows.

    If the exact focus date is not a trading day, the nearest previous available
    signal date is used. If the focus date is before the first available row, the
    first available row is used.
    """
    if back_days < 0 or forward_days < 0:
        raise ValueError("back_days and forward_days must be >= 0.")

    out = predictions.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)

    focus_ts = pd.to_datetime(focus_date).tz_localize(None)
    dates = pd.DatetimeIndex(out["date"])
    previous_or_equal = np.where(dates <= focus_ts)[0]
    if len(previous_or_equal) == 0:
        focus_idx = 0
    else:
        focus_idx = int(previous_or_equal[-1])

    start_idx = max(0, focus_idx - back_days)
    end_idx = min(len(out), focus_idx + forward_days + 1)
    window = out.iloc[start_idx:end_idx].copy()
    used_focus_date = pd.Timestamp(out.loc[focus_idx, "date"])
    return window, used_focus_date


def plot_focus_window_price_with_confidence(
    predictions: pd.DataFrame,
    focus_date: str,
    back_days: int,
    forward_days: int,
    confidence: PriceConfidenceParams,
    ticker: str,
    benchmark: str,
    horizon: int,
    output_path: Path,
) -> pd.DataFrame:
    """
    Creates a focused plot around a chosen signal date.

    It supports recent dates where the realized future close is not available yet:
    actual_future_close will be NaN and therefore not drawn for those rows.
    Approximate prices and confidence bands are drawn whenever the signal-date
    close and features are available.
    """
    window, used_focus_date = select_focus_window(predictions, focus_date, back_days, forward_days)
    dates = pd.to_datetime(window["date"])

    fig, ax = plt.subplots(figsize=(12, 6.5))

    ax.plot(dates, window["close"], marker="o", label="Close at signal date", linewidth=1.4)
    ax.plot(
        dates,
        window["actual_future_close"],
        marker="o",
        label=f"Actual close after {horizon} trading days, if already known",
        linewidth=1.4,
    )
    ax.plot(
        dates,
        window["approx_future_close"],
        marker="o",
        label="Approx future close from model probability",
        linewidth=2.0,
    )

    ax.fill_between(
        dates,
        window["empirical_ci90_low"].astype(float).values,
        window["empirical_ci90_high"].astype(float).values,
        alpha=0.12,
        label="Empirical 90% band",
    )
    ax.fill_between(
        dates,
        window["empirical_ci68_low"].astype(float).values,
        window["empirical_ci68_high"].astype(float).values,
        alpha=0.22,
        label="Empirical 68% band",
    )

    ax.axvline(used_focus_date, linestyle="--", linewidth=1.2, label=f"Focus signal date: {used_focus_date.date()}")

    ax.set_title(
        f"{ticker}: focused approximate price scenario around {pd.to_datetime(focus_date).date()}\n"
        f"{back_days} trading rows back, {forward_days} trading rows forward; benchmark = {benchmark}; horizon = {horizon} trading days"
    )
    ax.set_xlabel("Signal date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)

    known_count = int(window["actual_future_close"].notna().sum())
    total_count = int(len(window))
    note = (
        "Actual future close is blank when the horizon has not happened yet. "
        "Approx/CI uses only signal-date features, historical train scenario params, and validation residual bands. "
        f"Known actual future closes in this window: {known_count}/{total_count}. "
        f"Validation median abs price error: {confidence.median_absolute_price_error_pct:.2f}%."
    )
    fig.text(0.01, 0.01, note, fontsize=8, ha="left", va="bottom", wrap=True)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.legend(loc="best")
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return window


def write_price_calculation_method(
    output_path: Path,
    ticker: str,
    benchmark: str,
    horizon: int,
    params: PriceScenarioParams,
) -> None:
    text = f"""Price scenario calculation method
=================================

Model output
------------
The Transformer is a classifier. It predicts only this probability:

    p = probability that {ticker} outperforms {benchmark} over the next {horizon} trading days

It does not directly predict an exact future price.

Approximate price formula
-------------------------
For each signal date t:

    approx_future_close = close_t * exp(approx_log_return)

where:

    approx_log_return = B_train + p * E_out_train + (1 - p) * E_under_train

Definitions
-----------
B_train:
    Historical median future log return of {benchmark} in the training set.

E_out_train:
    Historical median excess log return of {ticker} versus {benchmark} in the training set,
    using only cases where {ticker} actually outperformed {benchmark}.

E_under_train:
    Historical median excess log return of {ticker} versus {benchmark} in the training set,
    using only cases where {ticker} did not outperform {benchmark}.

Scenario band
-------------
The low/high band uses the same formula but replaces median excess returns with
20th and 80th percentile conditional excess returns from the training set.

Important interpretation
------------------------
This price is an approximate scenario derived from historical behavior and model probability.
It is not a direct price forecast and it does not use realized future returns.

Parameters used
---------------
benchmark_base_log_return: {params.benchmark_base_log_return:.8f}
median_excess_if_outperform: {params.median_excess_if_outperform:.8f}
median_excess_if_underperform: {params.median_excess_if_underperform:.8f}
q20_excess_if_outperform: {params.q20_excess_if_outperform:.8f}
q80_excess_if_outperform: {params.q80_excess_if_outperform:.8f}
q20_excess_if_underperform: {params.q20_excess_if_underperform:.8f}
q80_excess_if_underperform: {params.q80_excess_if_underperform:.8f}
"""
    output_path.write_text(text, encoding="utf-8")


# -----------------------------
# Main pipeline
# -----------------------------

def run_pipeline(cfg: Config) -> dict:
    set_seed(SEED)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{cfg.ticker.upper()}_transformer_outperformance.pt"

    print(f"Using device: {DEVICE}")
    print(f"Downloading {cfg.ticker.upper()} and {cfg.benchmark.upper()}...")

    raw_prices = download_ohlcv(cfg.ticker, cfg.benchmark, cfg.start_date, cfg.end_date)
    feature_frame, feature_columns = add_features(
        raw_prices,
        cfg.forecast_horizon_days,
        require_future_returns=True,
    )
    inference_frame, _ = add_features(
        raw_prices,
        cfg.forecast_horizon_days,
        require_future_returns=False,
    )

    bundle = build_sequence_bundle(feature_frame, feature_columns, cfg.sequence_length)
    partitions, scaler = split_bundle(bundle, cfg.train_ratio, cfg.valid_ratio)
    loaders = make_loaders(partitions, cfg.batch_size)

    model = TransformerClassifier(
        num_features=len(feature_columns),
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    history = []
    best_state = None
    best_valid_auc = -np.inf

    print("Training...")
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer)
        valid_metrics = evaluate(model, loaders["valid"], criterion)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_metrics["loss"],
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_auc": valid_metrics["auc"],
            }
        )

        valid_auc = valid_metrics["auc"] if not np.isnan(valid_metrics["auc"]) else -np.inf
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_acc={valid_metrics['accuracy']:.3f} "
            f"valid_auc={valid_metrics['auc']:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": feature_columns,
            "ticker": cfg.ticker.upper(),
            "benchmark": cfg.benchmark.upper(),
            "sequence_length": cfg.sequence_length,
            "forecast_horizon_days": cfg.forecast_horizon_days,
            "model_args": {
                "num_features": len(feature_columns),
                "d_model": cfg.d_model,
                "nhead": cfg.nhead,
                "num_layers": cfg.num_layers,
                "dropout": cfg.dropout,
            },
        },
        model_path,
    )

    valid_result = evaluate(model, loaders["valid"], criterion)
    test_result = evaluate(model, loaders["test"], criterion)

    metrics_table = pd.DataFrame(
        [
            {
                "split": "validation",
                "loss": valid_result["loss"],
                "accuracy": valid_result["accuracy"],
                "auc": valid_result["auc"],
            },
            {
                "split": "test",
                "loss": test_result["loss"],
                "accuracy": test_result["accuracy"],
                "auc": test_result["auc"],
            },
        ]
    )

    scenario_params = fit_price_scenario_params(partitions["train"])

    valid_predictions = build_prediction_frame(
        partition=partitions["valid"],
        result=valid_result,
        scenario_params=scenario_params,
    )
    confidence_params = fit_price_confidence_params(valid_predictions)

    test_predictions = build_prediction_frame(
        partition=partitions["test"],
        result=test_result,
        scenario_params=scenario_params,
    )
    test_predictions = add_empirical_confidence_bands(test_predictions, confidence_params)

    # Full inference frame keeps recent rows where the future outcome is still unknown.
    # This is used for focus-date plots around today/recent dates.
    inference_bundle = build_sequence_bundle(inference_frame, feature_columns, cfg.sequence_length)
    inference_features_scaled = scale_sequence_features(inference_bundle.features, scaler)
    inference_probabilities = predict_probabilities(model, inference_features_scaled, cfg.batch_size)
    inference_partition = bundle_to_partition(inference_bundle, inference_features_scaled)
    inference_result = {
        "probabilities": inference_probabilities,
        "targets": inference_bundle.targets,
    }
    all_predictions = build_prediction_frame(
        partition=inference_partition,
        result=inference_result,
        scenario_params=scenario_params,
    )
    all_predictions = add_empirical_confidence_bands(all_predictions, confidence_params)

    history_df = pd.DataFrame(history)

    metrics_path = output_dir / "metrics.csv"
    predictions_path = output_dir / "test_predictions.csv"
    all_predictions_path = output_dir / "all_predictions_including_unresolved_future.csv"
    valid_predictions_path = output_dir / "validation_predictions_for_confidence.csv"
    history_path = output_dir / "training_history.csv"
    config_path = output_dir / "config.json"
    scenario_path = output_dir / "price_scenario_params.json"
    confidence_path = output_dir / "price_confidence_params.json"

    metrics_table.to_csv(metrics_path, index=False)
    test_predictions.to_csv(predictions_path, index=False)
    all_predictions.to_csv(all_predictions_path, index=False)
    valid_predictions.to_csv(valid_predictions_path, index=False)
    history_df.to_csv(history_path, index=False)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    with open(scenario_path, "w", encoding="utf-8") as f:
        json.dump(asdict(scenario_params), f, indent=2)

    with open(confidence_path, "w", encoding="utf-8") as f:
        json.dump(asdict(confidence_params), f, indent=2)

    write_price_calculation_method(
        output_dir / "price_calculation_method.txt",
        cfg.ticker.upper(),
        cfg.benchmark.upper(),
        cfg.forecast_horizon_days,
        scenario_params,
    )
    write_price_confidence_method(
        output_dir / "price_confidence_method.txt",
        cfg.ticker.upper(),
        cfg.benchmark.upper(),
        cfg.forecast_horizon_days,
        confidence_params,
    )

    plot_training_history(history_df, output_dir / "plot_training_history.png")
    plot_probability_panel(
        test_predictions,
        cfg.ticker.upper(),
        cfg.benchmark.upper(),
        cfg.forecast_horizon_days,
        cfg.bullish_threshold,
        cfg.bearish_threshold,
        output_dir / "plot_probability_outperformance.png",
    )
    plot_approx_price_panel(
        test_predictions,
        cfg.ticker.upper(),
        cfg.benchmark.upper(),
        cfg.forecast_horizon_days,
        output_dir / "plot_approx_price_scenario.png",
    )
    plot_approx_price_with_confidence_panel(
        test_predictions,
        confidence_params,
        cfg.ticker.upper(),
        cfg.benchmark.upper(),
        cfg.forecast_horizon_days,
        output_dir / "plot_approx_price_with_confidence.png",
    )
    plot_price_residuals_panel(
        valid_predictions,
        confidence_params,
        cfg.ticker.upper(),
        cfg.forecast_horizon_days,
        output_dir / "plot_validation_price_errors.png",
    )

    focus_window_path = None
    focus_plot_path = None
    if cfg.focus_date:
        focus_plot_path = output_dir / "plot_focus_window_price_with_confidence.png"
        focus_window = plot_focus_window_price_with_confidence(
            all_predictions,
            cfg.focus_date,
            cfg.window_back_days,
            cfg.window_forward_days,
            confidence_params,
            cfg.ticker.upper(),
            cfg.benchmark.upper(),
            cfg.forecast_horizon_days,
            focus_plot_path,
        )
        focus_window_path = output_dir / "focus_window_predictions.csv"
        focus_window.to_csv(focus_window_path, index=False)

    latest = all_predictions.iloc[-1]
    latest_probability = float(latest["probability_outperform"])
    trade_view = (
        "bullish_relative"
        if latest_probability >= cfg.bullish_threshold
        else "bearish_relative"
        if latest_probability <= cfg.bearish_threshold
        else "neutral"
    )

    forecast_summary = {
        "ticker": cfg.ticker.upper(),
        "benchmark": cfg.benchmark.upper(),
        "date": str(pd.to_datetime(latest["date"]).date()),
        "latest_close": float(latest["close"]),
        "forecast_horizon_days": cfg.forecast_horizon_days,
        "probability_of_outperformance": latest_probability,
        "trade_view": trade_view,
        "approx_future_close": float(latest["approx_future_close"]),
        "approx_future_close_low": float(latest["approx_future_close_low"]),
        "approx_future_close_high": float(latest["approx_future_close_high"]),
        "empirical_ci68_low": float(latest["empirical_ci68_low"]),
        "empirical_ci68_high": float(latest["empirical_ci68_high"]),
        "empirical_ci90_low": float(latest["empirical_ci90_low"]),
        "empirical_ci90_high": float(latest["empirical_ci90_high"]),
        "approx_future_return_pct": float(latest["approx_future_return_pct"]),
        "validation_median_absolute_price_error_pct": confidence_params.median_absolute_price_error_pct,
        "validation_coverage_68": confidence_params.coverage_68,
        "validation_coverage_90": confidence_params.coverage_90,
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "focus_date": cfg.focus_date,
        "window_back_days": cfg.window_back_days,
        "window_forward_days": cfg.window_forward_days,
        "focus_plot_path": str(focus_plot_path) if focus_plot_path is not None else None,
        "focus_window_predictions_path": str(focus_window_path) if focus_window_path is not None else None,
    }

    with open(output_dir / "forecast_summary.json", "w", encoding="utf-8") as f:
        json.dump(forecast_summary, f, indent=2)

    print("\nMetrics")
    print(metrics_table.to_string(index=False))

    print("\nLatest forecast")
    print(json.dumps(forecast_summary, indent=2))

    print("\nFiles written")
    for path in [
        model_path,
        metrics_path,
        predictions_path,
        all_predictions_path,
        valid_predictions_path,
        history_path,
        output_dir / "plot_training_history.png",
        output_dir / "plot_probability_outperformance.png",
        output_dir / "plot_approx_price_scenario.png",
        output_dir / "plot_approx_price_with_confidence.png",
        output_dir / "plot_validation_price_errors.png",
        output_dir / "plot_focus_window_price_with_confidence.png",
        output_dir / "focus_window_predictions.csv",
        output_dir / "price_calculation_method.txt",
        output_dir / "price_confidence_method.txt",
        output_dir / "forecast_summary.json",
    ]:
        if Path(path).exists():
            print(f"- {path}")

    return forecast_summary


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Transformer outperformance trading classifier.")
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--forecast-horizon-days", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="transformer_trading_output")
    parser.add_argument("--focus-date", default=None, help="Optional signal date for a focused price plot, YYYY-MM-DD.")
    parser.add_argument("--window-back-days", type=int, default=7, help="Trading rows before focus date to show.")
    parser.add_argument("--window-forward-days", type=int, default=5, help="Trading rows after focus date to show.")

    args = parser.parse_args()

    return Config(
        ticker=args.ticker,
        benchmark=args.benchmark,
        start_date=args.start_date,
        end_date=args.end_date,
        sequence_length=args.sequence_length,
        forecast_horizon_days=args.forecast_horizon_days,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        focus_date=args.focus_date,
        window_back_days=args.window_back_days,
        window_forward_days=args.window_forward_days,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_pipeline(cfg)
