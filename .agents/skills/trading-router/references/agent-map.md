# Trading Router Agent Map

## Stock

`market_forecasting_engine.pure_llm_stock_forecaster`
- One ticker or repeated per ticker.
- Pure LLM forecast, market-structure read, stock CEO.
- No order submission.

`market_forecasting_engine.cli`
- One ticker or repeated per ticker.
- Classical model forecast, validation, long-term sources, stock CEO.
- No order submission.

`market_forecasting_engine.virtual_trader_agent_cli`
- Broad universe scan and Alpaca paper stock plans.
- Use `--once --dry-run` unless paper submission is explicitly requested.

`market_forecasting_engine.live_trading.stocks.advice_order_agent`
- Checks open/expired Alpaca live advice orders.
- Use `--only-if-expired` by default.
- Do not add `--execute-live-orders` or `--confirm-live-order-risk` unless user explicitly asks for live submission.

## Crypto Spot

`market_forecasting_engine.live_trading.deribit_eth_usdc_daily_agent`
- ETH/USDC spot workflow.
- Dry-run unless user explicitly asks and confirms live Deribit order risk.
- Stock guidance does not apply.

## Options

`market_forecasting_engine.paper_options_agent`
- Alpaca paper options.
- Ask max debit or risk profile before running.
- Do not add `--execute-paper-orders` unless user explicitly asks.

`market_forecasting_engine.deribit_options_agent`
- Deribit ETH options.
- Ask max debit/risk.
- Do not add live confirmation flags unless user explicitly asks and confirms.

`market_forecasting_engine.llm_options_trader.agent`
- LLM options live-shadow simulation.
- Use `--simulation-only` by default.
