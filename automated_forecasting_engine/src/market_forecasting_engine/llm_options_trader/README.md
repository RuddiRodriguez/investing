# LLM Options Trader Implementation Map

This directory is the isolated implementation path for the LLM-directed options trader.
It is separate from the pure forecast CLI and from the virtual stock trader.

## Entry Points

- `agent.py`: combined LLM options loop. Builds the market packet, asks the LLM for entry or exit decisions, validates order payloads, records reports, memory, forecast feedback, and broker responses.
- `entry_agent.py`: older/split entry-only loop.
- `exit_agent.py`: older/split exit-only loop.
- `alpaca_agent.py`: Alpaca shadow/options variant.
- `dashboard.py`: read-only HTTP dashboard for agent state, options chain, forecast validation, shadow PnL, and LLM usage/cost.

## Shared Implementation

- `common.py`: runtime config, broker selection, market packet building, compact prompt packets, order payload helpers, JSONL writing, and shared technical observations.
- `prompts.py`: LLM system/user prompts and JSON schemas for combined entry, exit management, and adaptive profit policy.
- `profiles.py`: trader and strategy-mode profiles.
- `memory.py`: append-only decision memory and strategy lessons.
- `forecast_ledger.py`: records forecasts and matures them against actual price bars so the agent sees forecast-error feedback.
- `chronos_forecast.py`: optional numeric time-series forecast evidence for the LLM.
- `shadow_ledger.py` and `shadow_execution.py`: live-shadow simulation state and passive fill simulation.
- `knowledge_base.py` and `knowledge/`: local strategy notes injected into the market packet.
- `alpaca_common.py`, `alpaca_prompts.py`, `alpaca_shadow_ledger.py`: Alpaca-specific packet and shadow-execution helpers.

## Current Local LLM Route

The generic options runner scripts default to LM Studio:

```text
LLM_PROVIDER=llm_studio
LLM_MODEL=gemma-4-e4b-it
LLM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
```

`LLM_MODEL` can still be overridden explicitly. The Hugging Face-specific runner remains a separate hosted-model route.

The combined agents also support separate provider/model choices per LLM branch:

```text
ENTRY_LLM_PROVIDER / ENTRY_LLM_MODEL
EXIT_LLM_PROVIDER / EXIT_LLM_MODEL
PROFIT_POLICY_LLM_PROVIDER / PROFIT_POLICY_LLM_MODEL
```

The equivalent CLI flags are `--entry-llm-provider`, `--entry-llm-model`, `--exit-llm-provider`, `--exit-llm-model`, `--profit-policy-llm-provider`, and `--profit-policy-llm-model`.
This lets entry, exit, and profit-policy decisions use different OpenAI, Hugging Face, Bedrock, or LM Studio models without changing the prompt code.

## Shell Scripts

The runnable scripts live outside this package in `automated_forecasting_engine/scripts/`:

- `run_llm_options_trader_testnet.sh`: starts the combined LLM options agent against the testnet-style state directory and can submit testnet orders.
- `stop_llm_options_trader_testnet.sh`: stops the testnet agent processes.
- `run_llm_options_trader_dashboard.sh`: starts only the read-only dashboard.
- `run_llm_options_trader_live_shadow.sh`: starts live-market shadow simulation.
- `stop_llm_options_trader_live_shadow.sh`: stops live shadow processes.
- `run_llm_options_trader_live_shadow_hf.sh`: live shadow variant using the Hugging Face/local model route.
- `stop_llm_options_trader_live_shadow_hf.sh`: stops the Hugging Face/local live shadow variant.

## Runtime State

Default testnet state:

```text
automated_forecasting_engine/runs/llm_options_trader_testnet/
```

Important files under that directory:

- `ETH_llm_agent_report.json`: latest combined-agent report.
- `logs/ETH_llm_agent.jsonl`: append-only combined-agent history.
- `memory/`: decision and strategy memory.
- `forecasts/`: Chronos and forecast-validation artifacts.
- `agent.out`, `dashboard.out`: process logs from the shell scripts.

LLM usage/cost logs are written by `market_forecasting_engine.llm_usage` under:

```text
automated_forecasting_engine/runs/openai_usage/
```

The dashboard reads these usage logs and reports total calls, token counts, estimated cost, and net PnL after LLM cost.

## Safety Notes

- Do not start `run_llm_options_trader_testnet.sh` just to inspect code; it can submit testnet orders when `--execute-testnet-orders` is present.
- Prefer inspecting `agent.py`, `common.py`, `prompts.py`, and the latest JSON/JSONL state first.
- The dashboard is read-only, but it still starts a local HTTP process.
- Live-money behavior must remain explicit and audited; use shadow mode unless deliberately testing execution.
