from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import FastICA, PCA


PriceFetcher = Callable[[str, str | None, str | None], pd.DataFrame]


@dataclass(frozen=True)
class Chapter13Config:
    start: str | None = "2020-01-01"
    end: str | None = None
    max_components: int = 5
    min_history: int = 120
    winsor_quantile: float = 0.025
    include_ica: bool = False


def analyze_chapter_13_unsupervised_risk(
    prices_by_ticker: dict[str, pd.DataFrame],
    *,
    current_values: dict[str, float] | None = None,
    config: Chapter13Config | None = None,
) -> dict[str, Any]:
    """Build PCA/eigenportfolio, clustering, and HRP diagnostics for a ticker universe."""

    cfg = config or Chapter13Config()
    returns = build_returns_matrix(prices_by_ticker, min_history=cfg.min_history, winsor_quantile=cfg.winsor_quantile)
    if returns.shape[1] < 2:
        return _empty_result("InsufficientUniverse", "Chapter 13 requires at least two assets with overlapping return history.")

    pca_view = pca_risk_model(returns, max_components=cfg.max_components)
    cluster_view = hierarchical_cluster_view(returns)
    hrp_view = hierarchical_risk_parity(returns, current_values=current_values)
    ica_view = ica_risk_model(returns, max_components=cfg.max_components) if cfg.include_ica and returns.shape[1] >= 3 else None
    ticker_contexts = _ticker_contexts(returns, pca_view, cluster_view, hrp_view, current_values or {})
    return {
        "status": "available",
        "chapter": "jansen_chapter_13",
        "name": "data_driven_risk_factors_asset_allocation_unsupervised_learning",
        "method_card": chapter_13_unsupervised_method_card(cfg),
        "universe": {
            "tickers": list(returns.columns),
            "asset_count": int(returns.shape[1]),
            "observation_count": int(returns.shape[0]),
            "start": returns.index.min().isoformat(),
            "end": returns.index.max().isoformat(),
        },
        "pca": pca_view,
        "ica": ica_view,
        "clustering": cluster_view,
        "hrp": hrp_view,
        "ticker_contexts": ticker_contexts,
    }


def build_returns_matrix(
    prices_by_ticker: dict[str, pd.DataFrame],
    *,
    min_history: int = 120,
    winsor_quantile: float = 0.025,
) -> pd.DataFrame:
    closes: dict[str, pd.Series] = {}
    for ticker, prices in prices_by_ticker.items():
        if prices is None or prices.empty or "close" not in prices.columns:
            continue
        close = pd.to_numeric(prices["close"], errors="coerce").dropna()
        if len(close) >= max(3, int(min_history)):
            closes[str(ticker).upper()] = close
    if len(closes) < 2:
        return pd.DataFrame()
    close_frame = pd.DataFrame(closes).sort_index().ffill(limit=5)
    returns = close_frame.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    returns = returns.dropna(axis=1, thresh=max(3, int(min_history)))
    returns = returns.dropna(thresh=max(2, int(returns.shape[1] * 0.7)))
    returns = returns.fillna(returns.median(numeric_only=True))
    if returns.empty or returns.shape[1] < 2:
        return pd.DataFrame()
    q = float(winsor_quantile)
    if 0 < q < 0.5:
        lower = returns.quantile(q)
        upper = returns.quantile(1 - q)
        returns = returns.clip(lower=lower, upper=upper, axis=1)
    return returns


def pca_risk_model(returns: pd.DataFrame, *, max_components: int = 5) -> dict[str, Any]:
    scaled = _standardize(returns)
    n_components = max(1, min(int(max_components), scaled.shape[0], scaled.shape[1]))
    model = PCA(n_components=n_components, random_state=0)
    scores = model.fit_transform(scaled)
    loadings = pd.DataFrame(model.components_.T, index=returns.columns, columns=[f"pc{i + 1}" for i in range(n_components)])
    component_returns = pd.DataFrame(scores, index=returns.index, columns=[f"pc{i + 1}" for i in range(n_components)])
    reconstructed = model.inverse_transform(scores)
    residual = pd.DataFrame(scaled.to_numpy() - reconstructed, index=returns.index, columns=returns.columns)
    common_share = _safe_explained_share(model.explained_variance_ratio_)
    return {
        "component_count": int(n_components),
        "explained_variance_ratio": [float(x) for x in model.explained_variance_ratio_],
        "cumulative_explained_variance": [float(x) for x in np.cumsum(model.explained_variance_ratio_)],
        "common_variance_share": common_share,
        "component_latest_return": _round_dict(component_returns.iloc[-1].to_dict()),
        "component_momentum_20": _round_dict(component_returns.tail(20).mean().to_dict()),
        "component_returns_tail": _tail_records(component_returns, 5),
        "loadings": _round_nested(loadings.to_dict(orient="index")),
        "residual_volatility": _round_dict(residual.std().to_dict()),
        "eigenportfolios": eigenportfolios_from_loadings(loadings),
    }


def ica_risk_model(returns: pd.DataFrame, *, max_components: int = 5) -> dict[str, Any]:
    scaled = _standardize(returns)
    n_components = max(1, min(int(max_components), scaled.shape[0], scaled.shape[1]))
    model = FastICA(n_components=n_components, random_state=0, max_iter=1000, whiten="unit-variance")
    sources = model.fit_transform(scaled)
    columns = [f"ic{i + 1}" for i in range(n_components)]
    mixing = pd.DataFrame(model.mixing_, index=returns.columns, columns=columns)
    return {
        "component_count": int(n_components),
        "mixing": _round_nested(mixing.to_dict(orient="index")),
        "component_returns_tail": _tail_records(pd.DataFrame(sources, index=returns.index, columns=columns), 5),
    }


def eigenportfolios_from_loadings(loadings: pd.DataFrame) -> dict[str, Any]:
    portfolios: dict[str, Any] = {}
    for component in loadings.columns:
        raw = loadings[component].astype(float)
        gross = float(raw.abs().sum())
        if gross <= 1e-12:
            weights = raw * 0.0
        else:
            weights = raw / gross
        portfolios[component] = {
            "weights": _round_dict(weights.to_dict()),
            "gross_exposure": float(round(weights.abs().sum(), 8)),
            "net_exposure": float(round(weights.sum(), 8)),
            "largest_long": _top_weights(weights, largest=True),
            "largest_short": _top_weights(weights, largest=False),
            "tradability": "diagnostic_first",
            "promotion_requirement": "Promote only after rolling stability, transaction-cost, basket-execution, and drawdown tests pass.",
        }
    return portfolios


def hierarchical_cluster_view(returns: pd.DataFrame) -> dict[str, Any]:
    corr = returns.corr().fillna(0.0).clip(-1.0, 1.0)
    distance = np.sqrt((1.0 - corr) / 2.0)
    np.fill_diagonal(distance.values, 0.0)
    condensed = squareform(distance.to_numpy(), checks=False)
    links = linkage(condensed, method="single")
    order = [corr.index[i] for i in leaves_list(links)]
    clusters = _cluster_labels_from_linkage(links, list(corr.index))
    return {
        "ordered_tickers": order,
        "clusters": clusters,
        "correlation": _round_nested(corr.to_dict()),
        "distance": _round_nested(pd.DataFrame(distance, index=corr.index, columns=corr.columns).to_dict()),
        "linkage_method": "single",
    }


def hierarchical_risk_parity(returns: pd.DataFrame, *, current_values: dict[str, float] | None = None) -> dict[str, Any]:
    cov = returns.cov().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr = returns.corr().fillna(0.0).clip(-1.0, 1.0)
    distance = np.sqrt((1.0 - corr) / 2.0)
    np.fill_diagonal(distance.values, 0.0)
    links = linkage(squareform(distance.to_numpy(), checks=False), method="single")
    ordered = [corr.index[i] for i in leaves_list(links)]
    weights = pd.Series(1.0, index=ordered)
    _recursive_bisection(weights, cov.loc[ordered, ordered], ordered)
    weights = weights / max(float(weights.sum()), 1e-12)
    current = _current_weights(current_values or {}, weights.index)
    delta = weights.reindex(current.index).fillna(0.0) - current
    return {
        "status": "available",
        "ordered_tickers": ordered,
        "risk_weights": _round_dict(weights.sort_values(ascending=False).to_dict()),
        "current_weights": _round_dict(current.sort_values(ascending=False).to_dict()),
        "target_minus_current": _round_dict(delta.sort_values(ascending=False).to_dict()),
        "policy": "Use as a portfolio risk-budget overlay. It should size or block concentration, not override Buy/Hold/Sell by itself.",
    }


def chapter_13_unsupervised_method_card(config: Chapter13Config) -> dict[str, Any]:
    return {
        "name": "jansen_chapter_13_unsupervised_risk_factors_hrp",
        "version": "chapter_13_unsupervised_v1",
        "inputs": ["multi_asset_adjusted_close_prices", "current_position_values"],
        "methods": ["PCA latent risk factors", "PCA eigenportfolio diagnostics", "hierarchical clustering", "hierarchical risk parity"],
        "model_consequences": [
            "PCA factor exposures and residual features can enter per-ticker modeling frames.",
            "HRP weights adjust portfolio risk budget for per-ticker agents.",
            "Cluster concentration can block adding exposure to already crowded latent-risk groups.",
        ],
        "execution_policy": "Basket orders are dry-run unless an explicit execution flag is set.",
        "config": config.__dict__,
    }


def _ticker_contexts(
    returns: pd.DataFrame,
    pca_view: dict[str, Any],
    cluster_view: dict[str, Any],
    hrp_view: dict[str, Any],
    current_values: dict[str, float],
) -> dict[str, Any]:
    risk_weights = hrp_view.get("risk_weights", {})
    current_weights = hrp_view.get("current_weights", {})
    deltas = hrp_view.get("target_minus_current", {})
    clusters = cluster_view.get("clusters", {})
    loadings = pca_view.get("loadings", {})
    residual_vol = pca_view.get("residual_volatility", {})
    output: dict[str, Any] = {}
    for ticker in returns.columns:
        peers = [peer for peer, label in clusters.items() if label == clusters.get(ticker) and peer != ticker]
        output[ticker] = {
            "ticker": ticker,
            "pca_loadings": loadings.get(ticker, {}),
            "pca_residual_volatility": residual_vol.get(ticker),
            "cluster_id": clusters.get(ticker),
            "cluster_peers": peers,
            "hrp_risk_weight": risk_weights.get(ticker),
            "current_portfolio_weight": current_weights.get(ticker),
            "target_minus_current_weight": deltas.get(ticker),
            "position_value": current_values.get(ticker),
        }
    return output


def _recursive_bisection(weights: pd.Series, cov: pd.DataFrame, ordered: list[str]) -> None:
    clusters = [ordered]
    while clusters:
        next_clusters: list[list[str]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left, right = cluster[:split], cluster[split:]
            left_var = _cluster_variance(cov, left)
            right_var = _cluster_variance(cov, right)
            allocation = 1.0 - left_var / max(left_var + right_var, 1e-12)
            weights.loc[left] *= allocation
            weights.loc[right] *= 1.0 - allocation
            next_clusters.extend([left, right])
        clusters = next_clusters


def _cluster_variance(cov: pd.DataFrame, tickers: list[str]) -> float:
    sub = cov.loc[tickers, tickers]
    diag = np.diag(sub.to_numpy())
    inv_diag = 1.0 / np.maximum(diag, 1e-12)
    ivp = inv_diag / inv_diag.sum()
    return float(ivp @ sub.to_numpy() @ ivp)


def _cluster_labels_from_linkage(links: np.ndarray, tickers: list[str]) -> dict[str, int]:
    from scipy.cluster.hierarchy import fcluster

    count = max(2, min(6, int(round(np.sqrt(len(tickers))))))
    labels = fcluster(links, t=count, criterion="maxclust")
    return {ticker: int(label) for ticker, label in zip(tickers, labels, strict=False)}


def _standardize(frame: pd.DataFrame) -> pd.DataFrame:
    centered = frame - frame.mean()
    scale = frame.std().replace(0.0, np.nan)
    return (centered / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _current_weights(values: dict[str, float], index: pd.Index) -> pd.Series:
    series = pd.Series({str(k).upper(): float(v or 0.0) for k, v in values.items()}, dtype=float).reindex(index).fillna(0.0)
    total = float(series.clip(lower=0).sum())
    return series * 0.0 if total <= 0 else series / total


def _safe_explained_share(values: np.ndarray) -> float:
    return float(round(float(np.nansum(values)), 8))


def _tail_records(frame: pd.DataFrame, n: int) -> list[dict[str, Any]]:
    tail = frame.tail(n).copy()
    tail.index = tail.index.astype(str)
    return [{"date": index, **{k: round(float(v), 8) for k, v in row.items()}} for index, row in tail.iterrows()]


def _top_weights(weights: pd.Series, *, largest: bool) -> list[dict[str, Any]]:
    selected = weights.sort_values(ascending=not largest).head(5)
    return [{"ticker": str(k), "weight": round(float(v), 8)} for k, v in selected.items() if (v > 0 if largest else v < 0)]


def _round_dict(values: dict[Any, Any], digits: int = 8) -> dict[str, Any]:
    return {str(k): (round(float(v), digits) if _is_number(v) else v) for k, v in values.items()}


def _round_nested(values: dict[Any, Any], digits: int = 8) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, dict):
            output[str(key)] = _round_nested(value, digits)
        elif _is_number(value):
            output[str(key)] = round(float(value), digits)
        else:
            output[str(key)] = value
    return output


def _is_number(value: Any) -> bool:
    try:
        return np.isfinite(float(value))
    except Exception:
        return False


def _empty_result(status: str, reason: str) -> dict[str, Any]:
    return {"status": status, "reason": reason, "chapter": "jansen_chapter_13"}
