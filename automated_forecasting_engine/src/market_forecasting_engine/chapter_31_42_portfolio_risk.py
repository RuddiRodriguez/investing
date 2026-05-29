from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def analyze_chapter_31_42_portfolio_capital_risk(
    report: dict[str, Any],
    prices: pd.DataFrame | None = None,
    target_column: str = "close",
) -> dict[str, Any]:
    """Build portfolio/capital/risk controls for Chapters 31, 38, and 40-42."""

    target = target_column.lower()
    portfolio_context = report.get("portfolio_view", {}).get("position_context", {})
    account_inputs = _account_inputs(portfolio_context)
    trade_risk = report.get("trade_risk_view", {}).get("chapter_23_30_trade_risk_plan", {})
    diversification = _chapter_31_diversification(report)
    balance = _chapter_38_balance(report, diversification)
    capital_budget = _chapter_40_capital_budget(report, account_inputs, trade_risk)
    capital_application = _chapter_41_capital_application(report, account_inputs, trade_risk, capital_budget)
    portfolio_risk = _chapter_42_portfolio_risk(report, prices=prices, target_column=target, account_inputs=account_inputs, trade_risk=trade_risk)
    gate = _portfolio_capital_gate(
        report=report,
        account_inputs=account_inputs,
        trade_risk=trade_risk,
        diversification=diversification,
        portfolio_risk=portfolio_risk,
    )

    return {
        "principle": (
            "Chapters 31, 38, and 40-42 move from one chart to portfolio control: diversify, keep balance, "
            "decide how much capital is actually usable, apply it gradually, and measure ordinary plus catastrophic risk."
        ),
        "state": gate["state"],
        "status": gate["status"],
        "decision_policy": {
            "mode": "portfolio_capital_risk_report_only",
            "influences_final_action": False,
            "intended_consumer": "portfolio_allocator_human_or_llm_reviewer",
            "reason": "This layer sizes and constrains portfolio exposure; it does not override the single-ticker forecast action.",
        },
        "account_inputs": account_inputs,
        "portfolio_capital_gate": gate,
        "chapter_31_diversification": diversification,
        "chapter_38_balance": balance,
        "chapter_40_capital_budget": capital_budget,
        "chapter_41_capital_application": capital_application,
        "chapter_42_portfolio_risk": portfolio_risk,
        "capital_summary": {
            "commitment_type": trade_risk.get("commitment", {}).get("commitment_type"),
            "allocation_status": gate["allocation_status"],
            "risk_budget_pct": capital_budget.get("risk_budget_pct"),
            "max_loss_amount": capital_budget.get("max_loss_amount"),
            "max_new_notional": capital_application.get("max_new_notional"),
            "portfolio_risk_state": portfolio_risk.get("risk_state"),
        },
        "llm_integration": {
            "status": "planned",
            "note": (
                "A later LLM portfolio reviewer can explain allocation tradeoffs across tickers, but it must not bypass "
                "diversification, max-loss, capital, or drawdown gates."
            ),
        },
        "technical_method_card": chapter_31_42_portfolio_capital_risk_method_card(target_column=target),
    }


def apply_chapter_31_42_portfolio_capital_risk(
    report: dict[str, Any],
    prices: pd.DataFrame | None = None,
    target_column: str = "close",
) -> dict[str, Any]:
    portfolio_risk = analyze_chapter_31_42_portfolio_capital_risk(
        report,
        prices=prices,
        target_column=target_column,
    )
    report.setdefault("portfolio_view", {})["chapter_31_42_portfolio_capital_risk"] = portfolio_risk
    report.setdefault("technical_view", {})["chapter_31_42_portfolio_capital_risk"] = portfolio_risk
    report.setdefault("diagnostics", {})["chapter_31_42_portfolio_capital_risk"] = portfolio_risk
    report.setdefault("governance", {}).setdefault("portfolio_method_cards", {})[
        "chapter_31_42_portfolio_capital_risk"
    ] = portfolio_risk["technical_method_card"]
    return portfolio_risk


def chapter_31_42_portfolio_capital_risk_method_card(target_column: str = "close") -> dict[str, Any]:
    return {
        "name": "edwards_magee_chapters_31_38_40_42_portfolio_capital_risk",
        "version": "chapter_31_42_portfolio_risk_v1",
        "target_column": target_column.lower(),
        "decision_policy": "report_only_portfolio_capital_risk_no_action_override",
        "implemented_controls": [
            "chapter_31_diversification_context",
            "chapter_38_balance_context",
            "chapter_40_capital_budget",
            "chapter_41_capital_application_plan",
            "chapter_42_position_and_portfolio_risk",
            "missing_account_input_disclosure",
        ],
        "chapter_alignment": [
            "avoid_all_in_one_basket",
            "balance_and_diversify_exposure",
            "do_not_use_capital_before_chart_record_is_ready",
            "apply_capital_gradually_and_pragmatically",
            "measure_position_risk_portfolio_risk_and_drawdown",
        ],
        "future_llm_integration": (
            "A later LLM can review portfolio tradeoffs across tickers only after rule-based exposure and capital gates run."
        ),
    }


def _account_inputs(position_context: dict[str, Any]) -> dict[str, Any]:
    equity = _first_finite(
        position_context.get("account_equity"),
        position_context.get("portfolio_value"),
        position_context.get("net_liquidation_value"),
        position_context.get("net_value"),
    )
    current_value = _first_finite(
        position_context.get("current_value"),
        position_context.get("market_value"),
        position_context.get("position_value"),
        position_context.get("current_value_eur"),
    )
    quantity = _first_finite(position_context.get("quantity"), position_context.get("shares"))
    cost_basis = _first_finite(position_context.get("cost_basis"), position_context.get("average_cost"))
    cash = _first_finite(position_context.get("cash"), position_context.get("available_cash"))
    weight = _first_finite(position_context.get("weight"))
    if weight is None and equity and current_value is not None and equity > 0:
        weight = current_value / equity
    missing = []
    if equity is None:
        missing.append("account_equity")
    if current_value is None:
        missing.append("current_position_value")
    if quantity is None:
        missing.append("quantity")
    return {
        "status": "supplied" if not missing else "partial" if len(missing) < 3 else "not_supplied",
        "account_equity": _round(equity),
        "current_position_value": _round(current_value),
        "quantity": _round(quantity),
        "cost_basis": _round(cost_basis),
        "available_cash": _round(cash),
        "current_weight": _round(weight),
        "missing_inputs": missing,
        "source_status": position_context.get("status", "unknown"),
    }


def _chapter_31_diversification(report: dict[str, Any]) -> dict[str, Any]:
    metadata = report.get("governance", {}).get("security_metadata", {})
    data_manifest = report.get("data_manifest", {})
    universe = data_manifest.get("universe", {}) if isinstance(data_manifest, dict) else {}
    return {
        "status": "requires_universe_context",
        "ticker": report.get("ticker"),
        "sector": metadata.get("sector"),
        "industry": metadata.get("industry"),
        "asset_class": metadata.get("asset_class") or metadata.get("quote_type"),
        "universe_size": universe.get("ticker_count") or universe.get("count"),
        "single_ticker_limitation": "Diversification cannot be validated from one ticker report.",
        "policy": "Before allocation, compare sector, industry, asset class, and correlation against the rest of the portfolio.",
    }


def _chapter_38_balance(report: dict[str, Any], diversification: dict[str, Any]) -> dict[str, Any]:
    risk = report.get("risk_level")
    bucket = (
        report.get("selection_view", {})
        .get("chapter_21_chart_selection", {})
        .get("chart_selection", {})
        .get("chart_book_bucket")
    )
    return {
        "status": "requires_portfolio_context",
        "risk_level": risk,
        "chart_book_bucket": bucket,
        "balance_policy": "Do not let one market opinion, sector, or high-risk bucket dominate the whole account.",
        "diversification_status": diversification.get("status"),
    }


def _chapter_40_capital_budget(
    report: dict[str, Any],
    account_inputs: dict[str, Any],
    trade_risk: dict[str, Any],
) -> dict[str, Any]:
    position_sizing = trade_risk.get("chapter_26_position_sizing", {})
    risk_budget_pct = _first_finite(position_sizing.get("risk_budget_pct"), trade_risk.get("execution_summary", {}).get("risk_budget_pct"))
    if risk_budget_pct is None:
        risk_budget_pct = {"Low": 0.010, "Medium": 0.0075, "High": 0.0035}.get(str(report.get("risk_level")), 0.005)
    equity = _first_finite(account_inputs.get("account_equity"))
    max_loss = equity * risk_budget_pct if equity is not None else None
    return {
        "status": "sized" if max_loss is not None else "formula_only",
        "risk_budget_pct": _round(risk_budget_pct),
        "max_loss_amount": _round(max_loss),
        "account_equity": _round(equity),
        "capital_use_policy": "Use only a defined risk slice of account equity; do not size from conviction alone.",
        "paper_trading_note": "If account equity is missing, treat this as a paper/formula plan until portfolio inputs are supplied.",
    }


def _chapter_41_capital_application(
    report: dict[str, Any],
    account_inputs: dict[str, Any],
    trade_risk: dict[str, Any],
    capital_budget: dict[str, Any],
) -> dict[str, Any]:
    commitment = trade_risk.get("commitment", {})
    sizing = trade_risk.get("chapter_26_position_sizing", {})
    risk_per_share = _first_finite(sizing.get("risk_per_share"))
    max_loss = _first_finite(capital_budget.get("max_loss_amount"))
    current_price = _first_finite(report.get("current_price"))
    units = max_loss / risk_per_share if max_loss is not None and risk_per_share and risk_per_share > 0 else None
    max_notional = units * current_price if units is not None and current_price is not None else None
    staged = commitment.get("commitment_type") in {"candidate_long_commitment", "candidate_short_commitment"}
    return {
        "status": "allocation_ready" if max_notional is not None and staged else "formula_only" if staged else "no_new_allocation",
        "commitment_type": commitment.get("commitment_type"),
        "max_units": _round(units),
        "max_new_notional": _round(max_notional),
        "staging_policy": (
            "Initial allocation should be partial; add only after the trade behaves as planned and risk remains inside budget."
            if staged
            else "No staged capital application while this is not a trade candidate."
        ),
        "cash_check": {
            "available_cash": account_inputs.get("available_cash"),
            "cash_sufficient": (
                None
                if max_notional is None or account_inputs.get("available_cash") is None
                else float(account_inputs["available_cash"]) >= float(max_notional)
            ),
        },
    }


def _chapter_42_portfolio_risk(
    report: dict[str, Any],
    prices: pd.DataFrame | None,
    target_column: str,
    account_inputs: dict[str, Any],
    trade_risk: dict[str, Any],
) -> dict[str, Any]:
    close = _close_series(prices, target_column)
    drawdown = _max_drawdown(close)
    realized_vol = _realized_vol(close)
    current_weight = _first_finite(account_inputs.get("current_weight"))
    stop = _first_finite(trade_risk.get("execution_summary", {}).get("initial_stop"))
    current_price = _first_finite(report.get("current_price"))
    stop_loss_pct = abs(current_price - stop) / current_price if current_price and stop else None
    weight_risk = current_weight * stop_loss_pct if current_weight is not None and stop_loss_pct is not None else None
    flags = []
    if current_weight is not None and current_weight > 0.20:
        flags.append("single_position_weight_above_20pct")
    if drawdown is not None and drawdown <= -0.35:
        flags.append("large_recent_drawdown")
    if report.get("risk_level") == "High":
        flags.append("high_model_or_market_risk")
    risk_state = "High" if len(flags) >= 2 else "Medium" if flags else "Unmeasured" if current_weight is None else "Controlled"
    return {
        "status": "measured_partial" if close is not None and not close.empty else "requires_portfolio_context",
        "risk_state": risk_state,
        "realized_volatility_60d": _round(realized_vol),
        "max_drawdown_252d": _round(drawdown),
        "current_weight": _round(current_weight),
        "stop_loss_pct": _round(stop_loss_pct),
        "estimated_portfolio_loss_at_stop_pct": _round(weight_risk),
        "risk_flags": flags,
        "ordinary_risk_policy": "Measure normal volatility and stop-loss risk before adding capital.",
        "catastrophic_risk_policy": "Keep cash, diversification, and maximum drawdown limits outside any one ticker forecast.",
    }


def _portfolio_capital_gate(
    report: dict[str, Any],
    account_inputs: dict[str, Any],
    trade_risk: dict[str, Any],
    diversification: dict[str, Any],
    portfolio_risk: dict[str, Any],
) -> dict[str, Any]:
    commitment = trade_risk.get("commitment", {}).get("commitment_type")
    if commitment not in {"candidate_long_commitment", "candidate_short_commitment"}:
        return {
            "state": "NoNewAllocation",
            "status": "no_new_allocation",
            "allocation_status": "not_applicable",
            "reason": "Trade/risk layer does not authorize a new candidate allocation.",
            "blocking_reasons": [],
        }
    blockers = []
    if account_inputs.get("account_equity") is None:
        blockers.append("Account equity is missing, so real position size cannot be finalized.")
    if diversification.get("status") == "requires_universe_context":
        blockers.append("Diversification must be checked against the full portfolio before allocation.")
    if portfolio_risk.get("risk_state") == "High":
        blockers.append("Portfolio risk state is High.")
    return {
        "state": "Blocked" if blockers else "AllocationReady",
        "status": "blocked" if blockers else "allocation_ready",
        "allocation_status": "blocked_pending_inputs" if blockers else "ready_with_rules",
        "reason": blockers[0] if blockers else "Capital allocation can proceed under the rule-based risk budget.",
        "blocking_reasons": blockers,
    }


def _close_series(prices: pd.DataFrame | None, target_column: str) -> pd.Series | None:
    if prices is None or prices.empty:
        return None
    frame = prices.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    if target_column not in frame:
        return None
    return pd.to_numeric(frame[target_column], errors="coerce").dropna()


def _max_drawdown(close: pd.Series | None) -> float | None:
    if close is None or close.empty:
        return None
    recent = close.tail(252)
    drawdown = recent / recent.cummax() - 1.0
    value = float(drawdown.min())
    return value if np.isfinite(value) else None


def _realized_vol(close: pd.Series | None) -> float | None:
    if close is None or len(close) < 10:
        return None
    returns = np.log(close.replace(0, np.nan)).diff().replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return None
    value = float(returns.tail(60).std() * np.sqrt(252))
    return value if np.isfinite(value) else None


def _first_finite(*values: Any) -> float | None:
    for value in values:
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(result):
            return result
    return None


def _round(value: Any, digits: int = 4) -> float | None:
    numeric = _first_finite(value)
    if numeric is None:
        return None
    return round(numeric, digits)
