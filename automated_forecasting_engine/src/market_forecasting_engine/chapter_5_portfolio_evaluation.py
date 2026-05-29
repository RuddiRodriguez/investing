from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def analyze_chapter_5_portfolio_evaluation(
    *,
    backtests: dict[str, dict[str, Any]],
    selected_validation_predictions: dict[str, list[dict[str, Any]]],
    factor_evaluation: dict[str, list[dict[str, Any]]],
    market_feature_comparison: dict[str, Any],
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    """Build a Chapter 5 portfolio-performance and stability diagnostic."""

    horizon_reports = {}
    benchmark_wins = 0
    stable_horizons = 0
    for horizon in horizons:
        key = str(horizon)
        records = selected_validation_predictions.get(key, [])
        tear_sheet = _performance_tear_sheet(records, horizon_days=int(horizon))
        backtest = backtests.get(key, {})
        benchmark_comparison = _benchmark_comparison(backtest, tear_sheet)
        if benchmark_comparison["beats_benchmark"]:
            benchmark_wins += 1
        if benchmark_comparison["stable_improvement"]:
            stable_horizons += 1
        horizon_reports[key] = {
            "horizon_days": int(horizon),
            "backtest_summary": backtest,
            "performance_tear_sheet": tear_sheet,
            "benchmark_comparison": benchmark_comparison,
            "fundamental_law_proxy": _fundamental_law_proxy(
                factor_evaluation.get(key, []),
                validation_rows=int(backtest.get("rows", len(records)) or 0),
            ),
        }

    enriched_policy = _enriched_feature_policy(market_feature_comparison)
    allocation_policy = _allocation_policy(
        horizon_count=len(horizons),
        stable_horizons=stable_horizons,
        enriched_policy=enriched_policy,
    )
    return {
        "chapter": 5,
        "name": "Portfolio Optimization and Performance Evaluation",
        "status": "pass" if allocation_policy["stable_improvement"] else "warn",
        "decision_policy": {
            "influences_final_action": False,
            "mode": "diagnostic_only",
            "reason": "Chapter 5 evaluates strategy and portfolio evidence but does not change forecast actions or position sizes.",
        },
        "horizons": horizon_reports,
        "benchmark_win_rate": _finite(benchmark_wins / max(len(horizons), 1)),
        "stable_improvement_horizon_count": int(stable_horizons),
        "enriched_feature_policy": enriched_policy,
        "allocation_policy": allocation_policy,
        "portfolio_construction_candidates": _portfolio_construction_candidates(),
        "technical_method_card": chapter_5_portfolio_evaluation_method_card(),
    }


def chapter_5_portfolio_evaluation_method_card() -> dict[str, Any]:
    return {
        "name": "jansen_ml4t_chapter_5_portfolio_evaluation",
        "version": "chapter_5_portfolio_evaluation_v1",
        "source": "Machine Learning for Algorithmic Trading, 2nd ed., Chapter 5",
        "purpose": "Evaluate risk-adjusted strategy performance, benchmark-relative behavior, and whether portfolio/allocation changes are justified.",
        "decision_policy": "diagnostic_only",
        "implemented_components": [
            "performance_tear_sheet",
            "benchmark_alpha_beta",
            "drawdown_tail_and_var_metrics",
            "information_ratio_proxy",
            "fundamental_law_proxy",
            "stable_improvement_gate",
        ],
        "not_implemented": [
            "Zipline dependency",
            "Pyfolio dependency",
            "automatic mean_variance_position_sizing",
            "automatic Kelly sizing",
            "automatic HRP allocation",
        ],
    }


def _performance_tear_sheet(records: list[dict[str, Any]], horizon_days: int) -> dict[str, Any]:
    frame = pd.DataFrame(records)
    if frame.empty:
        return _empty_tear_sheet()
    frame["validation_date"] = pd.to_datetime(frame["validation_date"], errors="coerce")
    frame = frame.dropna(subset=["validation_date"]).sort_values("validation_date")
    if frame.empty:
        return _empty_tear_sheet()
    predicted = pd.to_numeric(frame.get("predicted_log_return"), errors="coerce").fillna(0.0)
    actual = pd.to_numeric(frame.get("actual_log_return"), errors="coerce").fillna(0.0)
    signal = np.sign(predicted).astype(float)
    gross_log = signal * actual
    benchmark_log = actual
    strategy_returns = np.expm1(gross_log.to_numpy(dtype=float))
    benchmark_returns = np.expm1(benchmark_log.to_numpy(dtype=float))
    periods_per_year = 252 / max(int(horizon_days), 1)
    equity = pd.Series(np.exp(gross_log.cumsum()).to_numpy(dtype=float), index=frame["validation_date"])
    benchmark_equity = pd.Series(np.exp(benchmark_log.cumsum()).to_numpy(dtype=float), index=frame["validation_date"])
    alpha_beta = _alpha_beta(strategy_returns, benchmark_returns, periods_per_year)
    return {
        "rows": int(len(frame)),
        "start_date": str(frame["validation_date"].iloc[0].date()),
        "end_date": str(frame["validation_date"].iloc[-1].date()),
        "cumulative_return": _finite(float(equity.iloc[-1] - 1.0)),
        "benchmark_cumulative_return": _finite(float(benchmark_equity.iloc[-1] - 1.0)),
        "annualized_return": _annualized_return(float(equity.iloc[-1]), len(frame), periods_per_year),
        "annualized_volatility": _annualized_volatility(strategy_returns, periods_per_year),
        "sharpe_ratio": _sharpe(strategy_returns, periods_per_year),
        "sortino_ratio": _sortino(strategy_returns, periods_per_year),
        "calmar_ratio": _calmar(strategy_returns, equity, periods_per_year),
        "max_drawdown": _max_drawdown(equity),
        "omega_ratio": _omega(strategy_returns),
        "tail_ratio": _tail_ratio(strategy_returns),
        "period_value_at_risk_5pct": _finite(float(np.quantile(strategy_returns, 0.05))) if len(strategy_returns) else 0.0,
        "hit_rate": _finite(float((strategy_returns > 0).mean())) if len(strategy_returns) else 0.0,
        "profit_factor": _profit_factor(strategy_returns),
        "benchmark_sharpe_ratio": _sharpe(benchmark_returns, periods_per_year),
        "benchmark_max_drawdown": _max_drawdown(benchmark_equity),
        "alpha": alpha_beta["alpha"],
        "beta": alpha_beta["beta"],
        "information_ratio": _information_ratio(strategy_returns, benchmark_returns, periods_per_year),
    }


def _benchmark_comparison(backtest: dict[str, Any], tear_sheet: dict[str, Any]) -> dict[str, Any]:
    strategy_sharpe = _number(backtest.get("sharpe_ratio", tear_sheet.get("sharpe_ratio")))
    benchmark_sharpe = _number(backtest.get("benchmark_sharpe_ratio", tear_sheet.get("benchmark_sharpe_ratio")))
    strategy_return = _number(backtest.get("cumulative_return", tear_sheet.get("cumulative_return")))
    benchmark_return = _number(backtest.get("benchmark_cumulative_return", tear_sheet.get("benchmark_cumulative_return")))
    strategy_drawdown = _number(backtest.get("max_drawdown", tear_sheet.get("max_drawdown")))
    benchmark_drawdown = _number(backtest.get("benchmark_max_drawdown", tear_sheet.get("benchmark_max_drawdown")))
    sharpe_delta = strategy_sharpe - benchmark_sharpe
    return_delta = strategy_return - benchmark_return
    drawdown_delta = strategy_drawdown - benchmark_drawdown
    beats = sharpe_delta > 0.05 and return_delta > 0.0
    stable = beats and drawdown_delta >= -0.02
    return {
        "beats_benchmark": bool(beats),
        "stable_improvement": bool(stable),
        "sharpe_delta": _finite(sharpe_delta),
        "cumulative_return_delta": _finite(return_delta),
        "max_drawdown_delta": _finite(drawdown_delta),
        "reason": (
            "Strategy beat benchmark on return and Sharpe without materially worse drawdown."
            if stable
            else "Strategy did not beat benchmark consistently enough for allocation changes."
        ),
    }


def _fundamental_law_proxy(factor_rows: list[dict[str, Any]], validation_rows: int) -> dict[str, Any]:
    if not factor_rows:
        return {
            "status": "insufficient_factor_evidence",
            "best_abs_rank_ic": 0.0,
            "effective_breadth": 0,
            "information_ratio_proxy": 0.0,
        }
    abs_ics = [abs(_number(row.get("rank_ic"))) for row in factor_rows]
    useful = [value for value in abs_ics if value >= 0.015]
    breadth = min(len(useful), max(int(np.sqrt(max(validation_rows, 0))), 1))
    best_ic = max(abs_ics) if abs_ics else 0.0
    mean_useful_ic = float(np.mean(useful)) if useful else 0.0
    ir_proxy = mean_useful_ic * float(np.sqrt(max(breadth, 0)))
    return {
        "status": "available" if useful else "weak_factor_breadth",
        "best_abs_rank_ic": _finite(best_ic),
        "mean_useful_abs_rank_ic": _finite(mean_useful_ic),
        "effective_breadth": int(breadth),
        "information_ratio_proxy": _finite(ir_proxy),
        "interpretation": "Higher values require both stronger IC and more independent validated signals.",
    }


def _enriched_feature_policy(market_feature_comparison: dict[str, Any]) -> dict[str, Any]:
    horizons = market_feature_comparison.get("horizons", []) if isinstance(market_feature_comparison, dict) else []
    helped = [bool(row.get("enriched_features_helped")) for row in horizons if isinstance(row, dict)]
    stable = bool(helped) and all(helped)
    return {
        "status": "stable_improvement" if stable else "not_stable",
        "stable_improvement": stable,
        "helped_horizon_count": int(sum(helped)),
        "tested_horizon_count": int(len(helped)),
        "reason": (
            "Enriched features improved every tested horizon."
            if stable
            else "Enriched features did not improve every tested horizon, so they remain evidence rather than allocation policy."
        ),
    }


def _allocation_policy(
    *,
    horizon_count: int,
    stable_horizons: int,
    enriched_policy: dict[str, Any],
) -> dict[str, Any]:
    stable = stable_horizons == horizon_count and horizon_count > 0 and bool(enriched_policy["stable_improvement"])
    return {
        "status": "candidate_for_implementation" if stable else "tested_not_implemented",
        "stable_improvement": bool(stable),
        "reason": (
            "Backtest and enriched-feature evidence were stable across horizons; allocation research can move to a controlled implementation."
            if stable
            else "Stable improvement was not present across all horizons; no portfolio optimizer or sizing change was added."
        ),
        "required_before_activation": [
            "multi_ticker_or_portfolio_context",
            "walk_forward_out_of_sample_weight_test",
            "transaction_cost_and_turnover_budget",
            "drawdown_not_worse_than_benchmark",
        ],
    }


def _portfolio_construction_candidates() -> list[dict[str, Any]]:
    return [
        {
            "name": "equal_weight",
            "implementation_status": "benchmark_only",
            "reason": "Useful 1/N benchmark and hard to overfit.",
        },
        {
            "name": "inverse_volatility",
            "implementation_status": "candidate",
            "reason": "Uses more reliable volatility estimates instead of noisy expected returns.",
        },
        {
            "name": "minimum_variance",
            "implementation_status": "candidate_after_multiticker_test",
            "reason": "Requires stable covariance estimates and multi-asset context.",
        },
        {
            "name": "hierarchical_risk_parity",
            "implementation_status": "candidate_after_multiticker_test",
            "reason": "Promising for correlated assets, but needs a real universe panel.",
        },
        {
            "name": "kelly_sizing",
            "implementation_status": "not_recommended_now",
            "reason": "Too sensitive to noisy edge estimates for this single-run pipeline.",
        },
    ]


def _alpha_beta(strategy_returns: np.ndarray, benchmark_returns: np.ndarray, periods_per_year: float) -> dict[str, float]:
    if len(strategy_returns) < 3 or len(benchmark_returns) != len(strategy_returns):
        return {"alpha": 0.0, "beta": 0.0}
    benchmark_var = float(np.var(benchmark_returns, ddof=1))
    if benchmark_var <= 1e-12:
        beta = 0.0
    else:
        beta = float(np.cov(strategy_returns, benchmark_returns, ddof=1)[0, 1] / benchmark_var)
    alpha = float((np.mean(strategy_returns) - beta * np.mean(benchmark_returns)) * periods_per_year)
    return {"alpha": _finite(alpha), "beta": _finite(beta)}


def _annualized_return(final_equity: float, periods: int, periods_per_year: float) -> float:
    if periods <= 0 or final_equity <= 0:
        return 0.0
    return _finite(float(final_equity ** (periods_per_year / periods) - 1.0))


def _annualized_volatility(returns: np.ndarray, periods_per_year: float) -> float:
    if len(returns) < 2:
        return 0.0
    return _finite(float(np.std(returns, ddof=1) * np.sqrt(periods_per_year)))


def _sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    vol = _annualized_volatility(returns, periods_per_year)
    if vol <= 1e-12:
        return 0.0
    return _finite(float(np.mean(returns) * periods_per_year / vol))


def _sortino(returns: np.ndarray, periods_per_year: float) -> float:
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = float(np.std(downside, ddof=1))
    if downside_std <= 1e-12:
        return 0.0
    return _finite(float(np.mean(returns) / downside_std * np.sqrt(periods_per_year)))


def _calmar(returns: np.ndarray, equity: pd.Series, periods_per_year: float) -> float:
    drawdown = abs(_max_drawdown(equity))
    if drawdown <= 1e-12:
        return 0.0
    annual_return = float(np.mean(returns) * periods_per_year) if len(returns) else 0.0
    return _finite(annual_return / drawdown)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_high = equity.cummax()
    return _finite(float((equity / running_high - 1.0).min()))


def _omega(returns: np.ndarray, threshold: float = 0.0) -> float:
    gains = np.maximum(returns - threshold, 0).sum()
    losses = np.maximum(threshold - returns, 0).sum()
    if losses <= 1e-12:
        return 999.0 if gains > 0 else 0.0
    return _finite(float(gains / losses))


def _tail_ratio(returns: np.ndarray) -> float:
    if len(returns) < 10:
        return 0.0
    left = abs(float(np.quantile(returns, 0.05)))
    right = abs(float(np.quantile(returns, 0.95)))
    if left <= 1e-12:
        return 999.0 if right > 0 else 0.0
    return _finite(right / left)


def _information_ratio(strategy_returns: np.ndarray, benchmark_returns: np.ndarray, periods_per_year: float) -> float:
    if len(strategy_returns) < 2 or len(strategy_returns) != len(benchmark_returns):
        return 0.0
    active = strategy_returns - benchmark_returns
    tracking_error = float(np.std(active, ddof=1))
    if tracking_error <= 1e-12:
        return 0.0
    return _finite(float(np.mean(active) / tracking_error * np.sqrt(periods_per_year)))


def _profit_factor(returns: np.ndarray) -> float:
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    if losses < 0:
        return _finite(float(gains / abs(losses)))
    return 999.0 if gains > 0 else 0.0


def _empty_tear_sheet() -> dict[str, Any]:
    return {
        "rows": 0,
        "cumulative_return": 0.0,
        "annualized_return": 0.0,
        "annualized_volatility": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "calmar_ratio": 0.0,
        "max_drawdown": 0.0,
        "omega_ratio": 0.0,
        "tail_ratio": 0.0,
        "period_value_at_risk_5pct": 0.0,
        "hit_rate": 0.0,
        "profit_factor": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "information_ratio": 0.0,
    }


def _number(value: Any) -> float:
    try:
        number = float(value if value is not None else 0.0)
    except Exception:
        return 0.0
    return number if np.isfinite(number) else 0.0


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0
