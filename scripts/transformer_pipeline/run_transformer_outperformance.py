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


@dataclass(frozen=True)
class TransformerConfig:
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
    seed: int = 42
    output_dir: Path = Path("notebooks") / "transformer_trading" / "models"
    show_plot: bool = True
    save_plot: bool = True


@dataclass(frozen=True)
class SequenceBundle:
    features: np.ndarray
    targets: np.ndarray
    dates: pd.DatetimeIndex
    close_prices: np.ndarray
    future_stock_returns: np.ndarray
    future_benchmark_returns: np.ndarray


@dataclass(frozen=True)
class SplitData:
    features: np.ndarray
    targets: np.ndarray
    dates: pd.DatetimeIndex
    close_prices: np.ndarray
    future_stock_returns: np.ndarray
    future_benchmark_returns: np.ndarray


@dataclass(frozen=True)
class TrainingArtifacts:
    history: pd.DataFrame
    validation_metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    forecast_summary: dict[str, object]
    model_path: Path
    metrics_path: Path
    predictions_path: Path
    history_path: Path
    summary_path: Path
    plot_path: Path


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


def parse_args() -> TransformerConfig:
    parser = argparse.ArgumentParser(description="Train a transformer that predicts stock outperformance versus a benchmark.")
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--forecast-horizon-days", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("notebooks") / "transformer_trading" / "models")
    parser.add_argument("--hide-plot", action="store_true")
    parser.add_argument("--skip-save-plot", action="store_true")
    args = parser.parse_args()
    return TransformerConfig(
        ticker=args.ticker.upper(),
        benchmark=args.benchmark.upper(),
        start_date=args.start_date,
        end_date=args.end_date,
        sequence_length=args.sequence_length,
        forecast_horizon_days=args.forecast_horizon_days,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seed=args.seed,
        output_dir=args.output_dir,
        show_plot=not args.hide_plot,
        save_plot=not args.skip_save_plot,
    )


def project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "requirements.txt").exists() and (candidate / "app.py").exists():
            return candidate
    return current


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device() -> torch.device:
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def download_ohlcv(ticker: str, benchmark: str, start: str, end: str | None) -> pd.DataFrame:
    raw = yf.download([ticker, benchmark], start=start, end=end, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    if raw.empty:
        raise ValueError("No market data was returned from Yahoo Finance.")

    def extract_symbol_frame(symbol: str) -> pd.DataFrame:
        frame = raw[symbol].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
        frame.columns = [column.lower() for column in frame.columns]
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        columns = ["open", "high", "low", "close", "volume"]
        return frame[columns].apply(pd.to_numeric, errors="coerce")

    stock = extract_symbol_frame(ticker)
    benchmark_frame = extract_symbol_frame(benchmark).rename(columns=lambda column: f"benchmark_{column}")
    combined = stock.join(benchmark_frame[["benchmark_close"]], how="inner").dropna().sort_index()
    if combined.empty:
        raise ValueError("No overlapping stock and benchmark history was found.")
    return combined


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
    df["benchmark_rel_5d"] = df["log_return_5d"] - df["benchmark_log_return_5d"]
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
        "rsi_14",
    ]
    clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns + ["target_outperform"]).copy()
    return clean, feature_columns


def build_sequence_bundle(frame: pd.DataFrame, feature_columns: Iterable[str], sequence_length: int) -> SequenceBundle:
    values = frame[list(feature_columns)].to_numpy(dtype=np.float32)
    targets = frame["target_outperform"].to_numpy(dtype=np.float32)
    dates = pd.DatetimeIndex(frame.index)
    close_prices = frame["close"].to_numpy(dtype=np.float32)
    future_stock_returns = frame["future_stock_log_return"].to_numpy(dtype=np.float32)
    future_benchmark_returns = frame["future_benchmark_log_return"].to_numpy(dtype=np.float32)

    sequence_features: list[np.ndarray] = []
    sequence_targets: list[float] = []
    sequence_dates: list[pd.Timestamp] = []
    sequence_close: list[float] = []
    sequence_future_stock_returns: list[float] = []
    sequence_future_benchmark_returns: list[float] = []

    for end_index in range(sequence_length, len(frame)):
        start_index = end_index - sequence_length
        sequence_features.append(values[start_index:end_index])
        sequence_targets.append(float(targets[end_index]))
        sequence_dates.append(dates[end_index])
        sequence_close.append(float(close_prices[end_index]))
        sequence_future_stock_returns.append(float(future_stock_returns[end_index]))
        sequence_future_benchmark_returns.append(float(future_benchmark_returns[end_index]))

    return SequenceBundle(
        features=np.asarray(sequence_features, dtype=np.float32),
        targets=np.asarray(sequence_targets, dtype=np.float32),
        dates=pd.DatetimeIndex(sequence_dates),
        close_prices=np.asarray(sequence_close, dtype=np.float32),
        future_stock_returns=np.asarray(sequence_future_stock_returns, dtype=np.float32),
        future_benchmark_returns=np.asarray(sequence_future_benchmark_returns, dtype=np.float32),
    )


def split_bundle(bundle: SequenceBundle, train_ratio: float, valid_ratio: float) -> tuple[dict[str, SplitData], StandardScaler]:
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
        return scaler.transform(flat).reshape(features.shape).astype(np.float32)

    def build_split(data_slice: slice) -> SplitData:
        return SplitData(
            features=transform(bundle.features[data_slice]),
            targets=bundle.targets[data_slice],
            dates=bundle.dates[data_slice],
            close_prices=bundle.close_prices[data_slice],
            future_stock_returns=bundle.future_stock_returns[data_slice],
            future_benchmark_returns=bundle.future_benchmark_returns[data_slice],
        )

    partitions = {
        "train": build_split(train_slice),
        "valid": build_split(valid_slice),
        "test": build_split(test_slice),
    }
    return partitions, scaler


def make_loaders(partitions: dict[str, SplitData], batch_size: int) -> dict[str, DataLoader]:
    return {
        name: DataLoader(
            SequenceDataset(split.features, split.targets),
            batch_size=batch_size,
            shuffle=(name == "train"),
        )
        for name, split in partitions.items()
    }


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
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, object]:
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


def train_model(
    config: TransformerConfig,
    feature_columns: list[str],
    loaders: dict[str, DataLoader],
    device: torch.device,
) -> tuple[TransformerClassifier, pd.DataFrame]:
    model = TransformerClassifier(
        num_features=len(feature_columns),
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_valid_auc = -np.inf

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        valid_metrics = evaluate(model, loaders["valid"], criterion, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "valid_loss": float(valid_metrics["loss"]),
                "valid_accuracy": float(valid_metrics["accuracy"]),
                "valid_auc": float(valid_metrics["auc"]) if not np.isnan(valid_metrics["auc"]) else np.nan,
            }
        )
        valid_auc = valid_metrics["auc"] if not np.isnan(valid_metrics["auc"]) else -np.inf
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} valid_loss={float(valid_metrics['loss']):.4f} "
            f"valid_acc={float(valid_metrics['accuracy']):.3f} valid_auc={float(valid_metrics['auc']):.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history)


def evaluate_splits(
    model: TransformerClassifier,
    partitions: dict[str, SplitData],
    loaders: dict[str, DataLoader],
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    criterion = nn.BCEWithLogitsLoss()
    valid_result = evaluate(model, loaders["valid"], criterion, device)
    test_result = evaluate(model, loaders["test"], criterion, device)

    metrics_table = pd.DataFrame(
        [
            {
                "split": "validation",
                "loss": float(valid_result["loss"]),
                "accuracy": float(valid_result["accuracy"]),
                "auc": float(valid_result["auc"]) if not np.isnan(valid_result["auc"]) else np.nan,
            },
            {
                "split": "test",
                "loss": float(test_result["loss"]),
                "accuracy": float(test_result["accuracy"]),
                "auc": float(test_result["auc"]) if not np.isnan(test_result["auc"]) else np.nan,
            },
        ]
    )

    test_predictions = pd.DataFrame(
        {
            "date": partitions["test"].dates,
            "close": partitions["test"].close_prices,
            "probability_outperform": test_result["probabilities"],
            "target_outperform": partitions["test"].targets,
            "future_stock_log_return": partitions["test"].future_stock_returns,
            "future_benchmark_log_return": partitions["test"].future_benchmark_returns,
        }
    )
    test_predictions["date"] = pd.to_datetime(test_predictions["date"], errors="coerce")
    test_predictions = test_predictions.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    test_predictions["predicted_outperform"] = (test_predictions["probability_outperform"] >= 0.5).astype(int)
    test_predictions["stock_future_return_pct"] = (np.exp(test_predictions["future_stock_log_return"]) - 1.0) * 100.0
    test_predictions["benchmark_future_return_pct"] = (np.exp(test_predictions["future_benchmark_log_return"]) - 1.0) * 100.0
    return metrics_table, test_predictions


def build_forecast_summary(config: TransformerConfig, test_predictions: pd.DataFrame, model_path: Path) -> dict[str, object]:
    latest_probability = float(test_predictions["probability_outperform"].iloc[-1]) if not test_predictions.empty else float("nan")
    latest_date = test_predictions["date"].iloc[-1] if not test_predictions.empty else None
    latest_close = float(test_predictions["close"].iloc[-1]) if not test_predictions.empty else float("nan")
    return {
        "ticker": config.ticker,
        "benchmark": config.benchmark,
        "date": latest_date.isoformat() if latest_date is not None else None,
        "latest_close": latest_close,
        "forecast_horizon_days": config.forecast_horizon_days,
        "probability_of_outperformance": latest_probability,
        "trade_view": "bullish_relative" if latest_probability >= 0.60 else "bearish_relative" if latest_probability <= 0.40 else "neutral",
        "model_path": str(model_path),
    }


def output_paths(config: TransformerConfig) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{config.ticker}_vs_{config.benchmark}_transformer"
    return {
        "model": config.output_dir / f"{stem}.pt",
        "metrics": config.output_dir / f"{stem}_metrics.csv",
        "predictions": config.output_dir / f"{stem}_predictions.csv",
        "history": config.output_dir / f"{stem}_history.csv",
        "summary": config.output_dir / f"{stem}_summary.json",
        "plot": config.output_dir / f"{stem}_plots.png",
    }


def save_outputs(
    config: TransformerConfig,
    feature_columns: list[str],
    scaler: StandardScaler,
    model: TransformerClassifier,
    metrics_table: pd.DataFrame,
    history_df: pd.DataFrame,
    test_predictions: pd.DataFrame,
    forecast_summary: dict[str, object],
    output_file_map: dict[str, Path],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": feature_columns,
            "ticker": config.ticker,
            "benchmark": config.benchmark,
            "sequence_length": config.sequence_length,
            "forecast_horizon_days": config.forecast_horizon_days,
            "model_args": {
                "num_features": len(feature_columns),
                "d_model": config.d_model,
                "nhead": config.nhead,
                "num_layers": config.num_layers,
                "dropout": config.dropout,
            },
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        output_file_map["model"],
    )
    metrics_table.to_csv(output_file_map["metrics"], index=False)
    history_df.to_csv(output_file_map["history"], index=False)
    test_predictions.to_csv(output_file_map["predictions"], index=False)
    output_file_map["summary"].write_text(json.dumps(forecast_summary, indent=2), encoding="utf-8")


def plot_results(
    config: TransformerConfig,
    feature_frame: pd.DataFrame,
    history_df: pd.DataFrame,
    test_predictions: pd.DataFrame,
    plot_path: Path,
) -> None:
    price_dates = pd.to_datetime(feature_frame.index, errors="coerce")
    prediction_dates = pd.to_datetime(test_predictions["date"], errors="coerce")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    history_df.plot(x="epoch", y=["train_loss", "valid_loss"], ax=axes[0], title="Training Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(price_dates, feature_frame["close"], label=f"{config.ticker} close", color="black", linewidth=1.5)
    axes[1].scatter(
        prediction_dates,
        test_predictions["close"],
        c=test_predictions["probability_outperform"],
        cmap="viridis",
        s=30,
        label="Test probability",
    )
    axes[1].set_title(f"{config.ticker} price with transformer probabilities")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(prediction_dates, test_predictions["probability_outperform"], label="Probability of outperformance", color="tab:blue")
    axes[2].axhline(0.5, color="tab:red", linestyle="--", linewidth=1)
    axes[2].set_title(
        f"Probability that {config.ticker} outperforms {config.benchmark} over the next {config.forecast_horizon_days} days"
    )
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    all_dates = pd.Index(price_dates.dropna()).append(pd.Index(prediction_dates.dropna()))
    if not all_dates.empty:
        date_min = all_dates.min()
        date_max = all_dates.max()
        axes[1].set_xlim(date_min, date_max)
        axes[2].set_xlim(prediction_dates.min() if not prediction_dates.dropna().empty else date_min, date_max)

    date_locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    date_formatter = mdates.ConciseDateFormatter(date_locator)
    axes[2].xaxis.set_major_locator(date_locator)
    axes[2].xaxis.set_major_formatter(date_formatter)

    fig.autofmt_xdate()
    plt.tight_layout()

    if config.save_plot:
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    if config.show_plot:
        plt.show()
    else:
        plt.close(fig)


def validate_config(config: TransformerConfig) -> None:
    if not 0.0 < config.train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0.0 < config.valid_ratio < 1.0:
        raise ValueError("valid_ratio must be between 0 and 1.")
    if config.train_ratio + config.valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must leave room for a test split.")
    if config.sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")
    if config.forecast_horizon_days < 1:
        raise ValueError("forecast_horizon_days must be at least 1.")


def run_pipeline(config: TransformerConfig) -> TrainingArtifacts:
    validate_config(config)
    set_seed(config.seed)
    device = select_device()
    if device.type == "mps":
        torch.set_float32_matmul_precision("high")

    print(f"Using device: {device}")
    prices = download_ohlcv(config.ticker, config.benchmark, config.start_date, config.end_date)
    feature_frame, feature_columns = add_features(prices, config.forecast_horizon_days)
    bundle = build_sequence_bundle(feature_frame, feature_columns, config.sequence_length)
    partitions, scaler = split_bundle(bundle, config.train_ratio, config.valid_ratio)
    loaders = make_loaders(partitions, config.batch_size)

    model, history_df = train_model(config, feature_columns, loaders, device)
    metrics_table, test_predictions = evaluate_splits(model, partitions, loaders, device)
    output_file_map = output_paths(config)
    forecast_summary = build_forecast_summary(config, test_predictions, output_file_map["model"])

    save_outputs(
        config,
        feature_columns,
        scaler,
        model,
        metrics_table,
        history_df,
        test_predictions,
        forecast_summary,
        output_file_map,
    )
    plot_results(config, feature_frame, history_df, test_predictions, output_file_map["plot"])

    return TrainingArtifacts(
        history=history_df,
        validation_metrics=metrics_table,
        test_predictions=test_predictions,
        forecast_summary=forecast_summary,
        model_path=output_file_map["model"],
        metrics_path=output_file_map["metrics"],
        predictions_path=output_file_map["predictions"],
        history_path=output_file_map["history"],
        summary_path=output_file_map["summary"],
        plot_path=output_file_map["plot"],
    )


def main() -> None:
    config = parse_args()
    root = project_root()
    if not config.output_dir.is_absolute():
        config = TransformerConfig(**{**asdict(config), "output_dir": root / config.output_dir})

    artifacts = run_pipeline(config)
    print("\nValidation metrics")
    print(artifacts.validation_metrics.to_string(index=False))
    print("\nLatest forecast summary")
    print(json.dumps(artifacts.forecast_summary, indent=2))
    print("\nSaved files")
    print(f"model: {artifacts.model_path}")
    print(f"metrics: {artifacts.metrics_path}")
    print(f"history: {artifacts.history_path}")
    print(f"predictions: {artifacts.predictions_path}")
    print(f"summary: {artifacts.summary_path}")
    if config.save_plot:
        print(f"plot: {artifacts.plot_path}")


if __name__ == "__main__":
    main()