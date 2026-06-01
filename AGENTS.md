# Project Instructions

This project is intended to become a real trading product, not an academic demo or toy backtest.

When changing the forecasting, trading, options, broker, dashboard, or agent code:

- Prefer production-grade behavior over simplified examples.
- Treat every order path as safety-critical, even for paper trading.
- Use real broker constraints when available: order types, whole-contract sizing, market hours, buying power, option contract tradability, spread, liquidity, and API limits.
- Keep autonomous agents fully auditable: write state, logs, reports, selected inputs, decisions, order payloads, execution blocks, and broker responses.
- Default to dry-run unless an explicit execution flag is present.
- Do not use market orders for options entries unless the user explicitly asks and the code includes a slippage/risk explanation.
- Size positions from account equity, risk budget, actual live premium/price, max debit, max open exposure, and max contracts. Fixed small premium caps are optional safety limits, not the primary sizing method.
- For options, use real contract symbols and live bid/ask snapshots. Synthetic options are research context only and must not be treated as executable.
- Agents should make professional autonomous decisions inside the user-provided risk envelope: entry, order type, limit price, take profit, stop loss, stale-order abandonment, hold/no-trade, and exit management.
- Avoid hidden behavior. If an agent blocks a trade, the reason must be visible in the terminal output, JSON report, and dashboard.
- Tests should cover risk gates, sizing, order payload shape, broker constraints, and dashboard parsing for any ticker, not just TSLA.

The user expects momentum toward a serious paper-to-live trading system. When implementation details are open, choose the safer and more operationally complete design.
