"""Transformer-based stock outperformance analysis for the dashboard."""

from __future__ import annotations

import math
import os
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


MODEL_ROOT = Path(__file__).resolve().parents[2] / "models" / "transformer_outperformance"
MODEL_ROOT.mkdir(parents=True, exist_ok=True)
SEED = 42
ProgressCallback = Callable[[int, int, str], None]


def _progress(progress_callback: ProgressCallback | None, step: int, total: int, message: str) -> None:
    if progress_callback is not None:
        progress_callback(step, total, message)


def select_torch_device() -> torch.device:
    forced_device = os.getenv("INVEST_TRANSFORMER_DEVICE", "").strip().lower()
    if forced_device:
        if forced_device == "mps":
            if platform.system() != "Darwin" or not torch.backends.mps.is_available():
                raise RuntimeError("INVEST_TRANSFORMER_DEVICE=mps was requested, but Apple MPS is not available.")
            return torch.device("mps")
        if forced_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("INVEST_TRANSFORMER_DEVICE=cuda was requested, but CUDA is not available.")
            return torch.device("cuda")
        if forced_device == "cpu":
            return torch.device("cpu")
        raise RuntimeError("INVEST_TRANSFORMER_DEVICE must be one of: cpu, cuda, mps.")

    # The macOS MPS backend is unstable here during dashboard-triggered training,
    # so default to CPU unless the user explicitly opts in via env var.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_random_seed(seed: int = SEED) -> None:
    # Runtime reseeding is disabled here because it triggers a native crash in
    # the interactive dashboard training path on this machine.
    _ = seed


def model_path_for_ticker(ticker: str, benchmark: str) -> Path:
    safe_name = f"{ticker.upper()}_vs_{benchmark.upper()}_transformer.pt"
    return MODEL_ROOT / safe_name


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def build_transformer_market_frame(prices: pd.DataFrame, ticker: str, benchmark: str) -> pd.DataFrame:
    symbol = ticker.upper()
    benchmark_symbol = benchmark.upper()
    required_columns = {
        "open": f"OPEN_{symbol}",
        "high": f"HIGH_{symbol}",
        "low": f"LOW_{symbol}",
        "close": symbol,
        "volume": f"VOLUME_{symbol}",
        "benchmark_close": benchmark_symbol,
    }
    missing = [name for name, column in required_columns.items() if column not in prices.columns]
    if missing:
        raise ValueError(
            f"Transformer analysis for {symbol} vs {benchmark_symbol} requires OHLCV plus benchmark close data. "
            f"Missing: {', '.join(missing)}."
        )

    frame = pd.DataFrame(index=pd.to_datetime(prices.index, errors="coerce"))
    for name, column in required_columns.items():
        frame[name] = pd.to_numeric(prices[column], errors="coerce")
    frame = frame.dropna().sort_index()
    if len(frame) < 160:
        raise ValueError(f"Need at least 160 valid observations for transformer analysis. Found {len(frame)}.")
    return frame


def add_features(frame: pd.DataFrame, forecast_horizon_days: int) -> tuple[pd.DataFrame, list[str]]:
    df = frame.copy()
    df["log_return_1d"] = np.log(df["close"] / df["close"].shift(1))
    df["log_return_5d"] = np.log(df["close"] / df["close"].shift(5))
    df["log_return_10d"] = np.log(df["close"] / df["close"].shift(10))
    df["volatility_10d"] = df["log_return_1d"].rolling(10).std()
    df["volatility_20d"] = df["log_return_1d"].rolling(20).std()
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["volume_log"] = np.log1p(df["volume"])
    df["volume_zscore_20d"] = (df["volume_log"] - df["volume_log"].rolling(20).mean()) / df["volume_log"].rolling(20).std()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["close_to_sma_10"] = df["close"] / df["sma_10"] - 1.0
    df["close_to_sma_20"] = df["close"] / df["sma_20"] - 1.0
    df["close_to_sma_50"] = df["close"] / df["sma_50"] - 1.0
    df["benchmark_log_return_5d"] = np.log(df["benchmark_close"] / df["benchmark_close"].shift(5))
    df["benchmark_log_return_10d"] = np.log(df["benchmark_close"] / df["benchmark_close"].shift(10))
    df["benchmark_rel_5d"] = df["log_return_5d"] - df["benchmark_log_return_5d"]
    df["benchmark_rel_10d"] = df["log_return_10d"] - df["benchmark_log_return_10d"]
    df["rsi_14"] = compute_rsi(df["close"], window=14)
    df["future_stock_log_return"] = np.log(df["close"].shift(-forecast_horizon_days) / df["close"])
    df["future_benchmark_log_return"] = np.log(df["benchmark_close"].shift(-forecast_horizon_days) / df["benchmark_close"])
    df["target_outperform"] = (df["future_stock_log_return"] > df["future_benchmark_log_return"]).astype(float)

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
        "benchmark_rel_10d",
        "rsi_14",
    ]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns + ["target_outperform"]).copy()
    return df, feature_columns


@dataclass
class SequenceBundle:
    features: np.ndarray
    targets: np.ndarray
    dates: pd.DatetimeIndex
    close_prices: np.ndarray
    benchmark_close_prices: np.ndarray
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
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
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


def build_sequence_bundle(frame: pd.DataFrame, feature_columns: list[str], sequence_length: int) -> SequenceBundle:
    values = frame[feature_columns].to_numpy(dtype=np.float32)
    targets = frame["target_outperform"].to_numpy(dtype=np.float32)
    dates = pd.DatetimeIndex(frame.index)
    close_prices = frame["close"].to_numpy(dtype=np.float32)
    benchmark_close_prices = frame["benchmark_close"].to_numpy(dtype=np.float32)
    future_stock_returns = frame["future_stock_log_return"].to_numpy(dtype=np.float32)
    future_benchmark_returns = frame["future_benchmark_log_return"].to_numpy(dtype=np.float32)

    sequence_features: list[np.ndarray] = []
    sequence_targets: list[float] = []
    sequence_dates: list[pd.Timestamp] = []
    sequence_close: list[float] = []
    sequence_benchmark_close: list[float] = []
    sequence_future_stock_returns: list[float] = []
    sequence_future_benchmark_returns: list[float] = []

    for end_index in range(sequence_length, len(frame)):
        start_index = end_index - sequence_length
        sequence_features.append(values[start_index:end_index])
        sequence_targets.append(float(targets[end_index]))
        sequence_dates.append(dates[end_index])
        sequence_close.append(float(close_prices[end_index]))
        sequence_benchmark_close.append(float(benchmark_close_prices[end_index]))
        sequence_future_stock_returns.append(float(future_stock_returns[end_index]))
        sequence_future_benchmark_returns.append(float(future_benchmark_returns[end_index]))

    return SequenceBundle(
        features=np.asarray(sequence_features, dtype=np.float32),
        targets=np.asarray(sequence_targets, dtype=np.float32),
        dates=pd.DatetimeIndex(sequence_dates),
        close_prices=np.asarray(sequence_close, dtype=np.float32),
        benchmark_close_prices=np.asarray(sequence_benchmark_close, dtype=np.float32),
        future_stock_returns=np.asarray(sequence_future_stock_returns, dtype=np.float32),
        future_benchmark_returns=np.asarray(sequence_future_benchmark_returns, dtype=np.float32),
    )


def split_bundle(bundle: SequenceBundle, train_ratio: float, valid_ratio: float) -> tuple[dict[str, dict[str, Any]], StandardScaler]:
    total = len(bundle.features)
    train_end = int(total * train_ratio)
    valid_end = int(total * (train_ratio + valid_ratio))
    train_end = min(max(train_end, 64), total - 32)
    valid_end = min(max(valid_end, train_end + 16), total)

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

    partitions: dict[str, dict[str, Any]] = {}
    for name, subset_slice in {"train": train_slice, "valid": valid_slice, "test": test_slice}.items():
        partitions[name] = {
            "features": transform(bundle.features[subset_slice]),
            "targets": bundle.targets[subset_slice],
            "dates": bundle.dates[subset_slice],
            "close_prices": bundle.close_prices[subset_slice],
            "benchmark_close_prices": bundle.benchmark_close_prices[subset_slice],
            "future_stock_returns": bundle.future_stock_returns[subset_slice],
            "future_benchmark_returns": bundle.future_benchmark_returns[subset_slice],
        }
    return partitions, scaler


def make_loaders(partitions: dict[str, dict[str, Any]], batch_size: int) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for name, partition in partitions.items():
        dataset = SequenceDataset(partition["features"], partition["targets"])
        loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=(name == "train"), num_workers=0)
    return loaders


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    losses: list[float] = []
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    probabilities: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        logits = model(features)
        loss = criterion(logits, targets)
        probs = torch.sigmoid(logits)
        losses.append(float(loss.item()))
        probabilities.append(probs.cpu().numpy())
        targets_all.append(targets.cpu().numpy())

    if not probabilities:
        return {"loss": np.nan, "accuracy": np.nan, "auc": np.nan, "probabilities": np.array([]), "targets": np.array([])}
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


@dataclass
class TransformerArtifacts:
    ticker: str
    benchmark: str
    analysis_end_date: str
    forecast_horizon_days: int
    sequence_length: int
    feature_columns: list[str]
    model_args: dict[str, Any]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    state_dict: dict[str, torch.Tensor]
    history_df: pd.DataFrame
    validation_metrics: dict[str, dict[str, float]]
    historical_predictions: pd.DataFrame
    model_path: Path


def _serialize_artifacts(artifacts: TransformerArtifacts) -> dict[str, Any]:
    return {
        "ticker": artifacts.ticker,
        "benchmark": artifacts.benchmark,
        "analysis_end_date": artifacts.analysis_end_date,
        "forecast_horizon_days": artifacts.forecast_horizon_days,
        "sequence_length": artifacts.sequence_length,
        "feature_columns": artifacts.feature_columns,
        "model_args": artifacts.model_args,
        "scaler_mean": artifacts.scaler_mean,
        "scaler_scale": artifacts.scaler_scale,
        "state_dict": {key: value.detach().cpu() for key, value in artifacts.state_dict.items()},
        "history": artifacts.history_df.to_dict(orient="records"),
        "validation_metrics": artifacts.validation_metrics,
        "historical_predictions": artifacts.historical_predictions.to_dict(orient="records"),
    }


def _deserialize_artifacts(payload: dict[str, Any], path: Path) -> TransformerArtifacts:
    historical_predictions = pd.DataFrame(payload.get("historical_predictions", []))
    for column in ["date"]:
        if column in historical_predictions.columns:
            historical_predictions[column] = pd.to_datetime(historical_predictions[column], errors="coerce")
    return TransformerArtifacts(
        ticker=str(payload["ticker"]),
        benchmark=str(payload["benchmark"]),
        analysis_end_date=str(payload["analysis_end_date"]),
        forecast_horizon_days=int(payload["forecast_horizon_days"]),
        sequence_length=int(payload["sequence_length"]),
        feature_columns=list(payload["feature_columns"]),
        model_args=dict(payload["model_args"]),
        scaler_mean=np.asarray(payload["scaler_mean"], dtype=np.float32),
        scaler_scale=np.asarray(payload["scaler_scale"], dtype=np.float32),
        state_dict=dict(payload["state_dict"]),
        history_df=pd.DataFrame(payload.get("history", [])),
        validation_metrics=dict(payload.get("validation_metrics", {})),
        historical_predictions=historical_predictions,
        model_path=path,
    )


def save_transformer_artifacts(artifacts: TransformerArtifacts) -> None:
    artifacts.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_serialize_artifacts(artifacts), artifacts.model_path)


def load_transformer_artifacts(ticker: str, benchmark: str) -> TransformerArtifacts:
    path = model_path_for_ticker(ticker, benchmark)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved transformer model exists for {ticker.upper()} vs {benchmark.upper()}. Enable retrain to train one first."
        )
    payload = torch.load(path, map_location="cpu")
    return _deserialize_artifacts(payload, path)


def _build_prediction_row(partition: dict[str, Any], probabilities: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": partition["dates"],
            "close": partition["close_prices"],
            "benchmark_close": partition["benchmark_close_prices"],
            "probability_outperform": probabilities,
            "target_outperform": partition["targets"],
            "future_stock_log_return": partition["future_stock_returns"],
            "future_benchmark_log_return": partition["future_benchmark_returns"],
        }
    )
    frame["predicted_outperform"] = (frame["probability_outperform"] >= 0.5).astype(int)
    frame["stock_future_return_pct"] = (np.exp(frame["future_stock_log_return"]) - 1.0) * 100.0
    frame["benchmark_future_return_pct"] = (np.exp(frame["future_benchmark_log_return"]) - 1.0) * 100.0
    return frame


def train_transformer_artifacts(
    ticker: str,
    benchmark: str,
    prices: pd.DataFrame,
    analysis_end_date: str,
    *,
    forecast_horizon_days: int = 5,
    sequence_length: int = 32,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    batch_size: int = 64,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    progress_callback: ProgressCallback | None = None,
) -> TransformerArtifacts:
    set_random_seed(SEED)
    device = select_torch_device()
    if device.type == "mps":
        torch.set_float32_matmul_precision("high")

    total_steps = epochs + 5
    _progress(progress_callback, 1, total_steps, f"Preparing transformer data for {ticker.upper()} vs {benchmark.upper()}")
    market_frame = build_transformer_market_frame(prices, ticker, benchmark)
    capped_frame = market_frame.loc[market_frame.index <= pd.Timestamp(analysis_end_date)].copy()
    if len(capped_frame) < 160:
        raise ValueError(f"Not enough history exists before {analysis_end_date} to train the transformer model.")
    feature_frame, feature_columns = add_features(capped_frame, forecast_horizon_days)
    bundle = build_sequence_bundle(feature_frame, feature_columns, sequence_length)
    if len(bundle.features) < 96:
        raise ValueError("Not enough sequence rows exist after feature engineering. Reduce sequence length or move the date forward.")
    partitions, scaler = split_bundle(bundle, train_ratio, valid_ratio)
    if len(partitions["test"]["features"]) == 0:
        raise ValueError("The test split is empty. Increase lookback or adjust the split ratios.")
    loaders = make_loaders(partitions, batch_size)

    _progress(progress_callback, 2, total_steps, "Building transformer model")
    model = TransformerClassifier(
        num_features=len(feature_columns),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_valid_auc = -np.inf
    for epoch in range(1, epochs + 1):
        _progress(progress_callback, epoch + 2, total_steps, f"Training transformer epoch {epoch}/{epochs}")
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        valid_metrics = evaluate(model, loaders["valid"], criterion, device)
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

    if best_state is not None:
        model.load_state_dict(best_state)

    _progress(progress_callback, total_steps - 1, total_steps, "Evaluating validation and test splits")
    valid_result = evaluate(model, loaders["valid"], criterion, device)
    test_result = evaluate(model, loaders["test"], criterion, device)
    historical_predictions = _build_prediction_row(partitions["test"], test_result["probabilities"])
    history_df = pd.DataFrame(history)
    validation_metrics = {
        "validation": {
            "loss": float(valid_result["loss"]),
            "accuracy": float(valid_result["accuracy"]),
            "auc": float(valid_result["auc"]) if not np.isnan(valid_result["auc"]) else np.nan,
        },
        "test": {
            "loss": float(test_result["loss"]),
            "accuracy": float(test_result["accuracy"]),
            "auc": float(test_result["auc"]) if not np.isnan(test_result["auc"]) else np.nan,
        },
    }

    artifacts = TransformerArtifacts(
        ticker=ticker.upper(),
        benchmark=benchmark.upper(),
        analysis_end_date=str(pd.Timestamp(analysis_end_date).date()),
        forecast_horizon_days=forecast_horizon_days,
        sequence_length=sequence_length,
        feature_columns=feature_columns,
        model_args={
            "num_features": len(feature_columns),
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dropout": dropout,
        },
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
        history_df=history_df,
        validation_metrics=validation_metrics,
        historical_predictions=historical_predictions,
        model_path=model_path_for_ticker(ticker, benchmark),
    )
    save_transformer_artifacts(artifacts)
    _progress(progress_callback, total_steps, total_steps, f"Saved transformer model to {artifacts.model_path.name}")
    return artifacts


def _load_model_from_artifacts(artifacts: TransformerArtifacts, device: torch.device) -> TransformerClassifier:
    model = TransformerClassifier(**artifacts.model_args).to(device)
    model.load_state_dict(artifacts.state_dict)
    model.eval()
    return model


def _transform_sequence(features: np.ndarray, artifacts: TransformerArtifacts) -> np.ndarray:
    scaler_mean = artifacts.scaler_mean
    scaler_scale = np.where(artifacts.scaler_scale == 0.0, 1.0, artifacts.scaler_scale)
    return ((features - scaler_mean) / scaler_scale).astype(np.float32)


@torch.no_grad()
def _predict_probability(model: TransformerClassifier, sequence: np.ndarray, device: torch.device) -> float:
    tensor = torch.tensor(sequence[None, ...], dtype=torch.float32, device=device)
    probability = torch.sigmoid(model(tensor)).item()
    return float(probability)


def generate_transformer_analysis(
    ticker: str,
    benchmark: str,
    prices: pd.DataFrame,
    analysis_end_date: str,
    artifacts: TransformerArtifacts,
) -> dict[str, Any]:
    if artifacts.ticker != ticker.upper() or artifacts.benchmark != benchmark.upper():
        raise ValueError("Loaded transformer artifacts do not match the selected ticker and benchmark.")
    if artifacts.analysis_end_date != str(pd.Timestamp(analysis_end_date).date()):
        raise ValueError(
            f"Saved transformer model was trained through {artifacts.analysis_end_date}. Enable retrain to analyze {analysis_end_date}."
        )

    device = select_torch_device()
    if device.type == "mps":
        torch.set_float32_matmul_precision("high")
    model = _load_model_from_artifacts(artifacts, device)

    market_frame = build_transformer_market_frame(prices, ticker, benchmark)
    capped_frame = market_frame.loc[market_frame.index <= pd.Timestamp(analysis_end_date)].copy()
    feature_frame, feature_columns = add_features(capped_frame, artifacts.forecast_horizon_days)
    if feature_columns != artifacts.feature_columns:
        raise ValueError("Saved transformer model feature set does not match the current feature pipeline. Retrain the model.")
    bundle = build_sequence_bundle(feature_frame, feature_columns, artifacts.sequence_length)
    if len(bundle.features) == 0:
        raise ValueError("No valid sequences are available for transformer inference at the selected date.")

    latest_sequence = _transform_sequence(bundle.features[-1], artifacts)
    probability = _predict_probability(model, latest_sequence, device)
    cutoff = pd.Timestamp(analysis_end_date)
    actual_future = market_frame.loc[market_frame.index > cutoff, ["close", "benchmark_close"]].head(artifacts.forecast_horizon_days).copy()
    actual_future = actual_future.reset_index().rename(columns={"index": "date"})
    if not actual_future.empty:
        actual_future["stock_return_pct"] = actual_future["close"] / float(capped_frame["close"].iloc[-1]) - 1.0
        actual_future["benchmark_return_pct"] = actual_future["benchmark_close"] / float(capped_frame["benchmark_close"].iloc[-1]) - 1.0
        actual_future["excess_return_pct"] = actual_future["stock_return_pct"] - actual_future["benchmark_return_pct"]

    forecast_summary = {
        "ticker": ticker.upper(),
        "benchmark": benchmark.upper(),
        "date": cutoff,
        "latest_close": float(capped_frame["close"].iloc[-1]),
        "latest_benchmark_close": float(capped_frame["benchmark_close"].iloc[-1]),
        "forecast_horizon_days": artifacts.forecast_horizon_days,
        "probability_of_outperformance": probability,
        "trade_view": "bullish_relative" if probability >= 0.60 else "bearish_relative" if probability <= 0.40 else "neutral",
        "model_path": str(artifacts.model_path),
    }
    historical_predictions = artifacts.historical_predictions.copy()
    if "date" in historical_predictions.columns:
        historical_predictions["date"] = pd.to_datetime(historical_predictions["date"], errors="coerce")

    return {
        "summary": forecast_summary,
        "history": market_frame.reset_index().rename(columns={"index": "date"}),
        "historical_predictions": historical_predictions,
        "actual_future": actual_future,
        "training_history": artifacts.history_df.copy(),
        "validation_metrics": artifacts.validation_metrics,
    }