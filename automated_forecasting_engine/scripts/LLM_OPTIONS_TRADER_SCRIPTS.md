# LLM Options Trader Scripts

This file indexes the shell scripts for the isolated LLM options trader path.
The implementation lives in:

```text
automated_forecasting_engine/src/market_forecasting_engine/llm_options_trader/
```

## Testnet / Paper-Like Path

- `run_llm_options_trader_testnet.sh`
  - Starts `market_forecasting_engine.llm_options_trader.agent`.
  - Default state: `automated_forecasting_engine/runs/llm_options_trader_testnet`.
  - Resolves `LLM_MODEL` from `LLM_PROVIDER` when `LLM_MODEL` is not set.
  - Defaults to `LLM_PROVIDER=llm_studio` and the loaded local model `gemma-4-e4b-it`.
  - Starts the dashboard unless `RUN_DASHBOARD=0`.
  - Includes `--execute-testnet-orders`, so do not run it for code inspection.

- `stop_llm_options_trader_testnet.sh`
  - Stops combined, entry, exit, and dashboard processes tied to the testnet state.

## Dashboard Only

- `run_llm_options_trader_dashboard.sh`
  - Starts only `market_forecasting_engine.llm_options_trader.dashboard`.
  - Read-only view of state, options chain, forecast validation, shadow PnL, and LLM usage cost.

## Live Shadow

- `run_llm_options_trader_live_shadow.sh`
  - Starts the LLM options trader against live market data in shadow simulation mode.
  - Resolves `LLM_MODEL` from `LLM_PROVIDER` when `LLM_MODEL` is not set.
  - Defaults to `LLM_PROVIDER=llm_studio` and the loaded local model `gemma-4-e4b-it`.

- `stop_llm_options_trader_live_shadow.sh`
  - Stops the live-shadow processes.

- `run_llm_options_trader_live_shadow_hf.sh`
  - Live-shadow variant routed to Hugging Face/local model configuration.

- `stop_llm_options_trader_live_shadow_hf.sh`
  - Stops the Hugging Face/local live-shadow processes.

## Related Utilities

- `load_llm_studio_remote_model.sh`
  - Helper for loading the configured local/LLM Studio trader model.

LLM usage logs are centralized under:

```text
automated_forecasting_engine/runs/openai_usage/
```

## Per-Step LLM Overrides

The combined Deribit and Alpaca full-LLM agents can route entry, exit, and profit-policy calls independently:

```text
ENTRY_LLM_PROVIDER=llm_studio ENTRY_LLM_MODEL=gemma-4-e4b-it
EXIT_LLM_PROVIDER=bedrock EXIT_LLM_MODEL=<bedrock-model>
PROFIT_POLICY_LLM_PROVIDER=openai PROFIT_POLICY_LLM_MODEL=<openai-model>
```

If a step override is not set, the agent falls back to `LLM_PROVIDER` / `LLM_MODEL`, and the options-only default is LM Studio with `gemma-4-e4b-it`.
