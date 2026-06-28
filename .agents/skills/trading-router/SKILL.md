---
name: trading-router
description: Route natural-language trading requests in `/Users/ruddigarcia/Projects/invest` to the correct existing local agent or forecast command. Use when the user asks which stock/ETF/options/crypto agent to run, asks for advice for tickers, asks to choose a ticker from a universe, asks to check expired Alpaca orders, or wants Codex to run the right local trading tool with safe options.
---

# Trading Router

## Overview

Use this skill to translate a user trading request into one existing local agent command. Existing agents remain independent; do not edit or replace them when routing.

## Workflow

1. Read the user request.
2. Use `market_forecasting_engine.trading_router_agent.plan_from_question` to get a route plan when command choice or options are not obvious.
3. If plan status is `needs_clarification`, ask only the listed question(s). Do not run a command.
4. If plan status is `ready`, inspect the chosen command before running it.
5. Default to advice-only or dry-run. Never add live execution flags unless user explicitly requested live execution and confirmed risk.
6. After running, summarize action, result, blocks, and report path.

## Safe Defaults

- Advice only unless user asks for order submission.
- Dry-run unless user asks for paper or live execution.
- No market orders.
- No live orders without explicit confirmation.
- Ask when stock vs options vs crypto is unclear.
- Ask when pure LLM vs classical forecast is unclear.
- Ask for risk size before options tools.

## Planner Helper

Use:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.trading_router_agent "give me advice for AAPL and MSFT"
```

The helper returns JSON with:
- `status`: `ready`, `needs_clarification`, or `unsupported`
- `tool`: selected local tool
- `questions`: required clarification questions
- `commands`: safe command arguments
- `report_hints`: likely report paths
- `safety`: execution assumptions

## Tool Map

- `pure_llm_stock_forecast`: one or more stocks, pure LLM forecast plus stock CEO.
- `classical_stock_forecast`: one or more stocks, classical model forecast plus stock CEO.
- `universe_stock_agent`: broad stock universe scan, one-cycle dry-run by default.
- `expired_alpaca_order_check`: expired Alpaca advice-order check, dry-run by default.
- `alpaca_paper_options`: Alpaca paper options decision; ask risk first.
- `deribit_eth_spot`: Deribit ETH/USDC spot check; dry-run by default.
- `deribit_eth_options`: Deribit ETH options; ask risk and live confirmation before execution.
- `llm_options_live_shadow`: options live-shadow simulation.

## Clarification Examples

- User: `give me advice for AAPL and MSFT`
  Ask: `Use pure LLM forecast or classical/model forecast?`
- User: `trade TSLA`
  Ask: `Stock forecast or options trade?`
- User: `TSLA options`
  Ask: `What max debit or risk profile should the paper options tool use?`
- User: `buy AAPL live`
  Ask for explicit live confirmation and amount; do not use market order.

Read `references/agent-map.md` for details when the route is unclear.
