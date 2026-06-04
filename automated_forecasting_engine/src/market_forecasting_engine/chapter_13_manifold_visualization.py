from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from market_forecasting_engine.chapter_13_unsupervised_risk import build_returns_matrix


def build_chapter_13_manifold_embedding(
    prices_by_ticker: dict[str, pd.DataFrame],
    *,
    method: str = "tsne",
    min_history: int = 120,
    random_state: int = 0,
    n_components: int = 2,
) -> dict[str, Any]:
    """Return 2D/3D coordinates for visualizing a portfolio universe.

    This module is intentionally visualization-only. The coordinates should not
    directly drive trading decisions because t-SNE/UMAP embeddings can shift
    materially with hyperparameters, sample windows, and universe membership.
    """

    returns = build_returns_matrix(prices_by_ticker, min_history=min_history)
    if returns.shape[1] < 3:
        return {"status": "InsufficientUniverse", "reason": "Need at least three assets for manifold visualization."}
    features = returns.T
    scaled = StandardScaler().fit_transform(features)
    selected = str(method or "tsne").lower()
    if selected == "pca":
        embedding = PCA(n_components=min(n_components, scaled.shape[0], scaled.shape[1]), random_state=random_state).fit_transform(scaled)
        model_name = "pca"
    elif selected == "umap":
        try:
            import umap  # type: ignore
        except Exception as exc:
            return {
                "status": "DependencyMissing",
                "reason": f"UMAP requires umap-learn: {type(exc).__name__}: {exc}",
                "fallback": "Use method='tsne' or install umap-learn.",
            }
        embedding = umap.UMAP(n_components=n_components, random_state=random_state).fit_transform(scaled)
        model_name = "umap"
    else:
        perplexity = max(2, min(30, (scaled.shape[0] - 1) // 2))
        embedding = TSNE(n_components=n_components, perplexity=perplexity, init="pca", learning_rate="auto", random_state=random_state).fit_transform(scaled)
        model_name = "tsne"
    rows = []
    for ticker, coords in zip(features.index, embedding, strict=False):
        row = {"ticker": str(ticker)}
        for index, value in enumerate(np.ravel(coords), start=1):
            row[f"x{index}"] = round(float(value), 8)
        rows.append(row)
    return {
        "status": "available",
        "method": model_name,
        "policy": "Visualization-only; do not use manifold coordinates as direct live trading gates.",
        "asset_count": int(returns.shape[1]),
        "observation_count": int(returns.shape[0]),
        "coordinates": rows,
    }
