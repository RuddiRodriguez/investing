# SPY Optuna Forecaster Experiment

Independent SPY intraday forecasting experiment for selecting and tuning model
candidates before wiring anything into an autonomous trading agent.

The experiment:

- downloads intraday SPY OHLCV bars from the configured data provider;
- builds leakage-safe intraday features using the production feature utilities;
- creates forward-return labels for 5, 10, 15, and 30 minute horizons;
- evaluates multiple model families on chronological train/validation splits;
- tunes the winning model family with Optuna;
- writes a final report with the best three candidates, tuned model metrics,
  validation tables, and validation plots.

Default run:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python experiments/spy_optuna_forecaster/run_spy_optuna_forecaster.py
```

Useful faster run:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python experiments/spy_optuna_forecaster/run_spy_optuna_forecaster.py --trials 15 --lookback-days 30
```

Outputs are written to:

```text
experiments/spy_optuna_forecaster/runs/latest/
```

