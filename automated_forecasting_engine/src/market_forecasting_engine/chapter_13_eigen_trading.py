from __future__ import annotations

from typing import Any


def build_eigen_trading_plan(
    chapter_13: dict[str, Any],
    *,
    max_component_gross_notional: float = 500.0,
    min_abs_weight: float = 0.03,
) -> dict[str, Any]:
    """Create dry-run basket plans from PCA eigenportfolios.

    Eigenportfolio trading is portfolio-level basket trading. This helper does
    not submit orders; it produces a reviewed plan that a portfolio executor can
    use after explicit approval and broker/liquidity checks.
    """

    if chapter_13.get("status") != "available":
        return {"status": "unavailable", "reason": chapter_13.get("reason", "chapter_13_unavailable")}
    pca = chapter_13.get("pca", {})
    eigenportfolios = pca.get("eigenportfolios", {})
    momentum = pca.get("component_momentum_20", {})
    plans = []
    for component, portfolio in eigenportfolios.items():
        signal = _float(momentum.get(component))
        if abs(signal) <= 1e-9:
            direction = "hold"
        else:
            direction = "long_eigenportfolio" if signal > 0 else "short_eigenportfolio"
        orders = []
        for ticker, weight in (portfolio.get("weights") or {}).items():
            weight = _float(weight)
            if abs(weight) < float(min_abs_weight):
                continue
            side_multiplier = 1.0 if direction == "long_eigenportfolio" else -1.0 if direction == "short_eigenportfolio" else 0.0
            signed_notional = side_multiplier * weight * float(max_component_gross_notional)
            if abs(signed_notional) <= 0:
                continue
            orders.append(
                {
                    "ticker": ticker,
                    "side": "buy" if signed_notional > 0 else "sell",
                    "signed_notional": round(float(signed_notional), 2),
                    "source_weight": round(float(weight), 8),
                    "order_type": "basket_limit_required",
                    "submit_eligible": False,
                    "reason": "eigen_trading_requires_explicit_portfolio_executor_and_live_prices",
                }
            )
        plans.append(
            {
                "component": component,
                "direction": direction,
                "momentum_20": round(float(signal), 8),
                "explained_variance_ratio": _component_explained_variance(pca, component),
                "gross_notional_cap": float(max_component_gross_notional),
                "orders": orders,
                "status": "dry_run" if orders else "no_trade",
            }
        )
    return {
        "status": "available",
        "policy": "Eigenportfolio plans are basket-level and dry-run by default; they do not override per-ticker Buy/Hold/Sell decisions.",
        "plans": plans,
    }


def _component_explained_variance(pca: dict[str, Any], component: str) -> float | None:
    try:
        index = int(str(component).replace("pc", "")) - 1
        values = pca.get("explained_variance_ratio", [])
        return round(float(values[index]), 8)
    except Exception:
        return None


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0
