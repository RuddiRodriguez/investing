from __future__ import annotations

import importlib.util
import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd


def analyze_chapter_4_alpha_research(
    *,
    features: pd.DataFrame,
    supervised: pd.DataFrame,
    factor_evaluation: dict[str, list[dict[str, Any]]],
    feature_registry: dict[str, Any],
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    """Build an Alphalens-inspired alpha-factor research report."""

    registry_by_feature = {
        str(entry.get("name")): entry for entry in feature_registry.get("entries", []) if isinstance(entry, dict)
    }
    horizon_reports = {}
    recommendation_buckets: dict[str, list[dict[str, Any]]] = {
        "keep": [],
        "watch": [],
        "penalize": [],
        "drop": [],
    }

    for horizon in horizons:
        key = str(horizon)
        rows = factor_evaluation.get(key, [])
        classified = [
            _classify_factor(row, registry_by_feature.get(str(row.get("feature")), {}), features, supervised, horizon)
            for row in rows
        ]
        for item in classified:
            bucket = item["recommendation"]
            recommendation_buckets.setdefault(bucket, []).append(
                {
                    "horizon_days": int(horizon),
                    "feature": item["feature"],
                    "economic_family": item["economic_family"],
                    "rank_ic": item["rank_ic"],
                    "turnover": item["top_quantile_turnover"],
                    "reason": item["recommendation_reason"],
                }
            )
        family_summary = _family_summary(classified)
        horizon_reports[key] = {
            "horizon_days": int(horizon),
            "alpha_quality_score": _alpha_quality_score(classified),
            "top_validated_factors": [item for item in classified if item["quality_label"] == "validated"][:10],
            "top_watch_factors": [item for item in classified if item["recommendation"] == "watch"][:10],
            "top_penalized_factors": [item for item in classified if item["recommendation"] == "penalize"][:10],
            "top_rejected_factors": [item for item in classified if item["recommendation"] == "drop"][:10],
            "top_abs_rank_ic": sorted(classified, key=lambda item: abs(item["rank_ic"]), reverse=True)[:10],
            "top_quantile_spread": sorted(classified, key=lambda item: abs(item["quantile_spread_return"]), reverse=True)[:10],
            "turnover_warnings": [
                item for item in classified if item["top_quantile_turnover"] > 0.35 and abs(item["rank_ic"]) >= 0.02
            ][:10],
            "sentiment_factor_summary": _sentiment_summary(classified),
            "family_summary": family_summary,
            "family_concentration_warning": _family_concentration_warning(family_summary),
        }

    status = "pass" if any(report["top_validated_factors"] for report in horizon_reports.values()) else "warn"
    return {
        "chapter": 4,
        "name": "Financial Feature Engineering - How to Research Alpha Factors",
        "status": status,
        "decision_policy": {
            "influences_final_action": False,
            "influences_model_fitting": True,
            "mode": "feature_quality_filter",
            "reason": "Low-alpha features can be excluded before candidate validation; raw factor diagnostics remain available for audit.",
        },
        "horizons": horizon_reports,
        "recommendations": {
            bucket: sorted(items, key=lambda item: abs(float(item.get("rank_ic", 0.0))), reverse=True)[:25]
            for bucket, items in recommendation_buckets.items()
        },
        "wavelet_denoising_assessment": _wavelet_denoising_assessment(
            features=features,
            supervised=supervised,
            factor_evaluation=factor_evaluation,
            horizons=horizons,
        ),
        "technical_method_card": chapter_4_alpha_research_method_card(),
    }


def chapter_4_alpha_research_method_card() -> dict[str, Any]:
    return {
        "name": "jansen_ml4t_chapter_4_alpha_research",
        "version": "chapter_4_alpha_research_v1",
        "source": "Machine Learning for Algorithmic Trading, 2nd ed., Chapter 4",
        "purpose": "Evaluate alpha-factor quality using IC, quantile spreads, turnover, stability, and economic family taxonomy.",
        "decision_policy": "feature_quality_filter",
        "implemented_components": [
            "economic_factor_taxonomy",
            "rank_ic_quality_gate",
            "quantile_return_table",
            "monotonicity_score",
            "turnover_penalty_recommendations",
            "sentiment_factor_summary",
            "wavelet_denoising_validation_test",
        ],
        "not_implemented": [
            "TA-Lib dependency",
            "Zipline/Alphalens runtime dependency",
            "wavelet denoising features",
        ],
    }


def _classify_factor(
    row: dict[str, Any],
    registry_entry: dict[str, Any],
    features: pd.DataFrame,
    supervised: pd.DataFrame,
    horizon: int,
) -> dict[str, Any]:
    feature = str(row.get("feature"))
    rank_ic = _finite(row.get("rank_ic"))
    quantile_spread = _finite(row.get("quantile_spread_return"))
    turnover = _finite(row.get("top_quantile_turnover"))
    stability = row.get("rank_ic_stability", {}) if isinstance(row.get("rank_ic_stability"), dict) else {}
    positive_share = _finite(stability.get("positive_share"))
    ic_std = _finite(stability.get("std"))
    quantile_table = _quantile_return_table(features.get(feature), supervised.get(f"target_log_return_{horizon}d"))
    monotonicity = _monotonicity_score(quantile_table)
    economic_family = _economic_family(feature, str(registry_entry.get("family", "")))
    quality_label, recommendation, reason = _quality_recommendation(
        rank_ic=rank_ic,
        quantile_spread=quantile_spread,
        turnover=turnover,
        positive_share=positive_share,
        monotonicity=monotonicity,
        rows=int(row.get("rows", 0) or 0),
    )
    return {
        "feature": feature,
        "feature_family": registry_entry.get("family"),
        "economic_family": economic_family,
        "rows": int(row.get("rows", 0) or 0),
        "rank_ic": rank_ic,
        "pearson_ic": _finite(row.get("pearson_ic")),
        "quantile_spread_return": quantile_spread,
        "top_quantile_turnover": turnover,
        "rank_ic_stability": {
            "mean": _finite(stability.get("mean")),
            "std": ic_std,
            "positive_share": positive_share,
            "ic_t_stat_proxy": _ic_t_stat_proxy(_finite(stability.get("mean")), ic_std),
        },
        "quantile_return_table": quantile_table,
        "monotonicity_score": monotonicity,
        "quality_label": quality_label,
        "recommendation": recommendation,
        "recommendation_reason": reason,
    }


def _quality_recommendation(
    *,
    rank_ic: float,
    quantile_spread: float,
    turnover: float,
    positive_share: float,
    monotonicity: float,
    rows: int,
) -> tuple[str, str, str]:
    abs_ic = abs(rank_ic)
    if rows < 80:
        return "under_sampled", "watch", "Insufficient observations for a stable alpha decision."
    if abs_ic >= 0.03 and positive_share >= 0.60 and turnover <= 0.35 and monotonicity >= 0.50:
        return "validated", "keep", "IC, stability, turnover, and quantile monotonicity are acceptable."
    if abs_ic >= 0.03 and turnover > 0.35:
        return "expensive", "penalize", "Predictive signal has high top-quantile turnover."
    if abs_ic >= 0.015 or abs(quantile_spread) >= 0.005 or monotonicity >= 0.50:
        return "weak_or_unstable", "watch", "Some signal is present, but stability or monotonicity is not strong enough."
    return "rejected", "drop", "Low IC, weak quantile spread, and weak monotonicity."


def _quantile_return_table(factor: pd.Series | None, target: pd.Series | None) -> list[dict[str, float]]:
    if factor is None or target is None:
        return []
    pair = pd.concat([factor.rename("factor"), target.rename("target")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 40 or pair["factor"].nunique(dropna=True) < 5:
        return []
    try:
        quantiles = pd.qcut(pair["factor"], q=5, labels=False, duplicates="drop")
    except ValueError:
        return []
    frame = pair.assign(quantile=quantiles).dropna()
    if frame["quantile"].nunique() < 2:
        return []
    output = []
    for quantile, group in frame.groupby("quantile"):
        output.append(
            {
                "quantile": int(quantile) + 1,
                "mean_forward_return": _finite(group["target"].mean()),
                "rows": int(len(group)),
            }
        )
    return output


def _monotonicity_score(quantile_table: list[dict[str, float]]) -> float:
    if len(quantile_table) < 3:
        return 0.0
    means = [float(item["mean_forward_return"]) for item in sorted(quantile_table, key=lambda item: item["quantile"])]
    diffs = np.diff(means)
    nonzero = diffs[np.abs(diffs) > 1e-12]
    if len(nonzero) == 0:
        return 0.0
    return float(max((nonzero > 0).mean(), (nonzero < 0).mean()))


def _economic_family(feature: str, registry_family: str) -> str:
    name = feature.lower()
    family = registry_family.lower()
    if "alt_news_sentiment" in name or "positive_share" in name or "negative_share" in name:
        return "sentiment"
    if "alt_news_volume" in name or "alt_news_relevance" in name:
        return "attention_alternative_data"
    if family == "relative_strength" or "relative" in name:
        return "relative_strength"
    if family in {"momentum", "returns", "trend_overlap"} or "momentum" in name or "return" in name:
        return "momentum"
    if family == "volatility_risk" or "volatility" in name or "atr" in name or "drawdown" in name:
        return "volatility"
    if family == "liquidity_volume" or "volume" in name or "liquidity" in name or "money_flow" in name:
        return "liquidity"
    if family in {"chart_structure", "trend_structure", "chart_gap"}:
        return "technical_structure"
    if family == "external_context" or name.startswith("exo_macro") or name.startswith("exo_rates"):
        return "macro_context"
    if family == "calendar":
        return "calendar"
    return "other"


def _family_summary(classified: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in classified:
        groups[str(item["economic_family"])].append(item)
    output = {}
    for family, rows in groups.items():
        output[family] = {
            "factor_count": int(len(rows)),
            "validated_count": int(sum(item["quality_label"] == "validated" for item in rows)),
            "mean_abs_rank_ic": _finite(np.mean([abs(float(item["rank_ic"])) for item in rows])),
            "mean_turnover": _finite(np.mean([float(item["top_quantile_turnover"]) for item in rows])),
        }
    return output


def _family_concentration_warning(family_summary: dict[str, dict[str, float]]) -> str | None:
    counts = Counter({family: int(stats["factor_count"]) for family, stats in family_summary.items()})
    total = sum(counts.values())
    if total <= 0:
        return None
    family, count = counts.most_common(1)[0]
    if count / total > 0.60:
        return f"Alpha candidates are concentrated in {family}; consider broader economic hypotheses."
    return None


def _sentiment_summary(classified: list[dict[str, Any]]) -> dict[str, Any]:
    sentiment = [
        item for item in classified if item["economic_family"] in {"sentiment", "attention_alternative_data"}
    ]
    if not sentiment:
        return {
            "status": "not_present",
            "factor_count": 0,
            "validated_count": 0,
        }
    return {
        "status": "present",
        "factor_count": int(len(sentiment)),
        "validated_count": int(sum(item["quality_label"] == "validated" for item in sentiment)),
        "best_factor": max(sentiment, key=lambda item: abs(float(item["rank_ic"])))["feature"],
        "best_abs_rank_ic": max(abs(float(item["rank_ic"])) for item in sentiment),
        "recommendations": dict(Counter(item["recommendation"] for item in sentiment)),
    }


def _alpha_quality_score(classified: list[dict[str, Any]]) -> int:
    if not classified:
        return 0
    validated = sum(item["quality_label"] == "validated" for item in classified)
    watch = sum(item["recommendation"] == "watch" for item in classified)
    penalized = sum(item["recommendation"] == "penalize" for item in classified)
    return int(max(0, min(100, validated * 12 + watch * 4 - penalized * 3)))


def _wavelet_denoising_assessment(
    *,
    features: pd.DataFrame,
    supervised: pd.DataFrame,
    factor_evaluation: dict[str, list[dict[str, Any]]],
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    if importlib.util.find_spec("pywt") is None:
        return {
            "status": "not_implemented",
            "tested": False,
            "reason": "PyWavelets is not installed in the project venv; no validation improvement was established, so no wavelet features were added.",
            "next_step": "Install PyWavelets only if a controlled validation experiment shows denoised factors beat raw factors.",
        }
    pywt = importlib.import_module("pywt")
    comparisons = []
    tested_keys: set[tuple[int, str]] = set()
    for horizon in horizons:
        target = supervised.get(f"target_log_return_{horizon}d")
        if target is None:
            continue
        rows = sorted(
            factor_evaluation.get(str(horizon), []),
            key=lambda row: abs(_finite(row.get("rank_ic"))),
            reverse=True,
        )[:5]
        for row in rows:
            feature = str(row.get("feature"))
            key = (int(horizon), feature)
            if key in tested_keys or feature not in features:
                continue
            tested_keys.add(key)
            raw = features[feature]
            if raw.nunique(dropna=True) < 8:
                continue
            denoised = _rolling_wavelet_denoise(raw, pywt=pywt)
            raw_rank_ic = _rank_ic(raw, target)
            denoised_rank_ic = _rank_ic(denoised, target)
            if not np.isfinite(raw_rank_ic) or not np.isfinite(denoised_rank_ic):
                continue
            delta = abs(denoised_rank_ic) - abs(raw_rank_ic)
            comparisons.append(
                {
                    "horizon_days": int(horizon),
                    "feature": feature,
                    "raw_rank_ic": _finite(raw_rank_ic),
                    "denoised_rank_ic": _finite(denoised_rank_ic),
                    "abs_rank_ic_delta": _finite(delta),
                    "improved": bool(delta > 0.0),
                }
            )
    comparisons = sorted(comparisons, key=lambda item: abs(float(item["abs_rank_ic_delta"])), reverse=True)
    if not comparisons:
        return {
            "status": "tested_not_implemented",
            "tested": True,
            "pywt_version": str(getattr(pywt, "__version__", "unknown")),
            "comparison_method": "rolling_point_in_time_wavelet_denoise_vs_raw_rank_ic",
            "tested_factor_count": 0,
            "improved_factor_count": 0,
            "improved_share": 0.0,
            "best_abs_rank_ic_delta": 0.0,
            "reason": "No eligible non-constant factors had enough observations for a controlled wavelet comparison; no wavelet model features were added.",
            "comparisons": [],
        }
    improved_count = sum(1 for item in comparisons if item["improved"])
    improved_share = improved_count / len(comparisons)
    best_delta = max(float(item["abs_rank_ic_delta"]) for item in comparisons)
    candidate = improved_share >= 0.50 and best_delta >= 0.01
    if candidate:
        status = "candidate_for_future_implementation"
        reason = (
            "Wavelet denoising improved enough alpha diagnostics to justify a later walk-forward model experiment; "
            "no model features were added in this run."
        )
    else:
        status = "tested_not_implemented"
        reason = "Wavelet denoising did not show enough incremental rank-IC improvement; no wavelet model features were added."
    return {
        "status": status,
        "tested": True,
        "pywt_version": str(getattr(pywt, "__version__", "unknown")),
        "comparison_method": "rolling_point_in_time_wavelet_denoise_vs_raw_rank_ic",
        "tested_factor_count": int(len(comparisons)),
        "improved_factor_count": int(improved_count),
        "improved_share": _finite(improved_share),
        "best_abs_rank_ic_delta": _finite(best_delta),
        "reason": reason,
        "comparisons": comparisons[:20],
    }


def _rolling_wavelet_denoise(
    series: pd.Series,
    *,
    pywt: Any,
    window: int = 64,
    min_periods: int = 24,
    wavelet: str = "db2",
) -> pd.Series:
    values = series.replace([np.inf, -np.inf], np.nan).astype(float)
    output = pd.Series(np.nan, index=series.index, dtype=float)
    for end in range(len(values)):
        sample = values.iloc[max(0, end - window + 1) : end + 1].dropna()
        if len(sample) < min_periods or sample.nunique(dropna=True) < 4:
            continue
        denoised = _wavelet_denoise_array(sample.to_numpy(dtype=float), pywt=pywt, wavelet=wavelet)
        if len(denoised):
            output.iloc[end] = float(denoised[-1])
    return output


def _wavelet_denoise_array(values: np.ndarray, *, pywt: Any, wavelet: str) -> np.ndarray:
    if len(values) < 8 or np.nanstd(values) <= 1e-12:
        return values
    centered = values - float(np.nanmean(values))
    max_level = pywt.dwt_max_level(len(centered), pywt.Wavelet(wavelet).dec_len)
    level = min(2, max_level)
    if level <= 0:
        return values
    coeffs = pywt.wavedec(centered, wavelet=wavelet, mode="periodization", level=level)
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745 if len(detail) else 0.0
    threshold = float(sigma * math.sqrt(2.0 * math.log(len(centered)))) if sigma > 0 else 0.0
    filtered = [coeffs[0]]
    filtered.extend(pywt.threshold(coeff, threshold, mode="soft") for coeff in coeffs[1:])
    reconstructed = pywt.waverec(filtered, wavelet=wavelet, mode="periodization")[: len(values)]
    return reconstructed + float(np.nanmean(values))


def _rank_ic(factor: pd.Series, target: pd.Series) -> float:
    pair = (
        pd.concat([factor.rename("factor"), target.rename("target")], axis=1)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(pair) < 50 or pair["factor"].nunique(dropna=True) < 5 or pair["target"].nunique(dropna=True) < 5:
        return 0.0
    return _finite(pair["factor"].rank().corr(pair["target"].rank()))


def _ic_t_stat_proxy(mean: float, std: float) -> float:
    if std <= 1e-12:
        return 0.0
    return float(mean / std * math.sqrt(4))


def _finite(value: Any) -> float:
    try:
        numeric = float(value if value is not None else 0.0)
    except Exception:
        return 0.0
    return numeric if np.isfinite(numeric) else 0.0
