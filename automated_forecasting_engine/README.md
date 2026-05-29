# Automated Stock Price Forecasting Engine

This is a new standalone experiment for a governable stock market price forecasting system.

The engine follows the proposed design:

- ingest historical price data from CSV or Yahoo Finance
- store raw, normalized, panel, and metadata artifacts in a local data lake
- write data manifests and data-quality reports for every governed run
- apply point-in-time availability rules for macro, rates, and event CSV data
- track security metadata, corporate-action columns, and exchange-calendar alignment
- build structured time-series features
- add technical chart-structure features such as support/resistance, trend channels, gaps, breakouts, and volume confirmation
- test multiple model candidate families
- select a model with objective validation metrics
- produce 1-day, 5-day, and 30-day forecasts with confidence intervals
- produce trade-quality diagnostics such as breakout probability and reward-to-risk score
- evaluate factor quality with rank IC, quantile spread, turnover, and stability
- backtest validation-period model signals against buy-and-hold behavior
- write reproducible governance metadata for audit and model review

This first version avoids expensive LLM-based experimentation. It uses deterministic feature engineering, statistical baselines, scikit-learn models, statsmodels/arch time-series models, and optional LightGBM support when installed.

The current venv has the advanced time-series dependencies installed:

- `statsmodels` for ARIMA, SARIMA, and VAR
- `arch` for GARCH
- `torch` for LSTM
- `pyarrow` for parquet data storage

## Run From The Existing Workspace

From `/Users/ruddigarcia/Projects/invest`:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.cli \
  --csv stock_prices.csv \
  --ticker AAPL \
  --output-dir automated_forecasting_engine/runs/aapl
```

To fetch from Yahoo Finance instead of a CSV:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.cli \
  --ticker AAPL \
  --start 2020-01-01 \
  --output-dir automated_forecasting_engine/runs/aapl
```

## Same-Session Daily Trading Variant

Use the daily-trade variant when the goal is to make an entry/exit decision during the same trading session. This path requires intraday OHLCV data such as 1-minute, 5-minute, or 15-minute bars. Daily end-of-day candles can support swing or next-day forecasts, but they do not contain the intraday path needed for VWAP, opening range, stop placement, or same-day risk control.

Fetch 5-minute Yahoo bars:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.daily_trade_cli \
  --ticker AAPL \
  --interval 5m \
  --start 2026-05-20 \
  --output automated_forecasting_engine/runs/aapl_daily_trade/report.json
```

Run from a local intraday CSV:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.daily_trade_cli \
  --ticker AAPL \
  --csv path/to/aapl_5m.csv \
  --interval 5m \
  --output-dir automated_forecasting_engine/runs/aapl_daily_trade
```

The output is an auditable same-session plan with:

- whether the input actually looks intraday
- inferred bar interval
- 1h, 2h, and 4h price forecasts by default
- VWAP and opening-range context
- long/short/no-trade decision
- entry reference, stop, take-profit, and max hold time when a setup qualifies

The daily-trade run also writes:

```text
<output-dir>/daily_trade_report.json
<output-dir>/forecast_report.json
<output-dir>/daily_trade_forecasts.csv
<output-dir>/governance/
<output-dir>/plots/forecast_<TICKER>.png
<output-dir>/plots/forecast_<TICKER>.html
<output-dir>/plots/daily_trade_<TICKER>.png
<output-dir>/plots/daily_trade_<TICKER>.html
```

Change the forecast horizons with:

```bash
--forecast-hours 1,2,3,4
```

The CLI refreshes Yahoo data on every run and uses recent intraday history for model training. If the requested `--start` has too few 5-minute bars for 1h-4h validation, it automatically expands the Yahoo training lookback while still forecasting from the newest available bar.

By default, runs with `--output-dir` also create a local data cache inside the run folder:

```text
automated_forecasting_engine/runs/<run_name>/data/
  raw/
  normalized/
  manifests/
  metadata/
```

Use a shared data lake instead:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.cli \
  --ticker AAPL \
  --start 2020-01-01 \
  --data-dir automated_forecasting_engine/data \
  --output-dir automated_forecasting_engine/runs/aapl_governed
```

Refresh provider data instead of using the normalized cache:

```bash
--refresh-data-cache
```

Technical charts default to a semilog price scale, matching the Edwards/Magee preference for percentage-aware charts. Use a linear scale when needed:

```bash
--chart-scale linear
```

Run with richer market context:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.cli \
  --ticker AAPL \
  --start 2020-01-01 \
  --benchmark SPY \
  --sector XLK \
  --vix ^VIX \
  --output-dir automated_forecasting_engine/runs/aapl_context
```

Run the wider candidate/tuning grid:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.cli \
  --ticker AAPL \
  --start 2020-01-01 \
  --benchmark SPY \
  --vix ^VIX \
  --search-level expanded \
  --output-dir automated_forecasting_engine/runs/aapl_expanded
```

Run nested Optuna optimization candidates:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.cli \
  --ticker AAPL \
  --start 2020-01-01 \
  --tune optuna \
  --optuna-trials 25 \
  --optuna-inner-splits 3 \
  --output-dir automated_forecasting_engine/runs/aapl_optuna
```

Validation controls:

```bash
--purge-window 5 --embargo-window 2 --final-holdout-fraction 0.15 --transaction-cost-bps 5
```

By default, the purge window equals the forecast horizon. The final holdout period is not used to select the winning model; it is reported as post-selection diagnostics. With `--tune optuna`, tuning is nested inside each candidate fit slice, so Optuna only sees the training portion available to that fold.

## Date-Time Based Forecast Runs

The main engine can run from a specific day and hour with `--as-of`. The model only sees bars at or before that timestamp, so this is the option to use when you want to reproduce a decision as of a known point in time.

Daily bars:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.cli \
  --ticker TSLA \
  --start 2020-01-01 \
  --as-of 2026-05-29 \
  --output-dir automated_forecasting_engine/runs/TSLA_daily_asof
```

Hourly bars:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.cli \
  --ticker TSLA \
  --start 2026-04-01 \
  --interval 1h \
  --horizons 1,2,4,8 \
  --as-of 2026-05-29T15:00:00 \
  --output-dir automated_forecasting_engine/runs/TSLA_hourly_asof
```

For live Yahoo runs where `--end` is omitted, the engine now refreshes provider data on each execution by default, so a scheduled daily/hourly run will scrape the newest available bars and append compact forecast rows to:

```text
<output-dir>/forecast_log.csv
```

It also writes forecast-history plots from that log:

```text
<output-dir>/plots/forecast_log_<TICKER>.png
<output-dir>/plots/forecast_log_<TICKER>.html
```

Use `--allow-live-cache` only when you explicitly want to reuse cached provider data.

Optional CSV inputs can add macro, rates, and event data:

```bash
--macro-csv path/to/macro.csv --rates-csv path/to/rates.csv --events-csv path/to/events.csv
```

Each CSV should have a date column first. Macro/rates CSVs should contain numeric columns. Event CSVs can include an `event_type`, `type`, `event`, or `category` column.

For point-in-time correctness, macro/rates/events CSVs may include one of these availability columns:

- `available_at`
- `availability_date`
- `release_date`
- `published_at`

If no availability column exists, add an explicit business-day release lag:

```bash
--macro-release-lag-days 1 --rates-release-lag-days 1 --events-release-lag-days 0
```

Security metadata can be supplied with:

```bash
--security-master-csv path/to/security_master.csv
```

The security master should include a `ticker` or `symbol` column. Optional columns such as `exchange`, `currency`, `sector`, `industry`, `active`, and `delisted` are copied into governance metadata.

Universe/panel metadata can be added with:

```bash
--universe-tickers MSFT,NVDA,GOOGL --build-panel
```

or:

```bash
--universe-csv path/to/universe.csv --build-panel
```

The panel is stored as a date/ticker MultiIndex dataset under the data lake and summarized in the run manifest.

Rank the universe without running a separate forecast for every ticker:

```bash
--universe-tickers AAPL,MSFT,NVDA,CRWD --rank-universe --universe-top-n 10
```

Universe ranking uses market-action and cross-sectional evidence such as momentum rank, short-term return rank, dollar-volume rank, volatility rank, illiquidity rank, and residual momentum.

## Chapter 3 Alternative News and Sentiment

The engine can download/scrape recent news, convert it into point-in-time sentiment features, and include those features in model selection. Enable this path with:

```bash
--enable-alt-news
```

Provider options:

- `yahoo_rss`: tries the old Yahoo Finance RSS endpoint first and falls back to `yahoo_news` when RSS returns 404.
- `yahoo_news`: uses `yfinance.Ticker(...).news` directly.
- `openai_web`: uses OpenAI web search to retrieve recent factual news/social/forum context. Requires `OPENAI_API_KEY`.

Sentiment modes:

- `lexicon`: deterministic positive/negative word scoring. No LLM call.
- `llm`: uses OpenAI to classify article sentiment when an API key is available.
- `hybrid`: blends lexicon and LLM sentiment. Falls back to lexicon if the LLM is unavailable.

OpenAI-backed paths use the Responses API with structured outputs. The default model is `gpt-5.4-mini-2026-03-17` with reasoning effort `none`; override with `--llm-model` and `--llm-reasoning-effort` when needed.

Provider-backed news with deterministic sentiment:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.cli \
  --ticker ASML \
  --start 2020-01-01 \
  --output-dir automated_forecasting_engine/runs/ASML_alt_news \
  --enable-alt-news \
  --alt-news-provider yahoo_news \
  --alt-news-sentiment-mode lexicon
```

Provider-backed news with hybrid LLM sentiment:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.cli \
  --ticker ASML \
  --start 2020-01-01 \
  --output-dir automated_forecasting_engine/runs/ASML_alt_news_hybrid \
  --enable-alt-news \
  --alt-news-provider yahoo_news \
  --alt-news-sentiment-mode hybrid \
  --llm-env-file path/to/.env
```

Most aggressive LLM/web retrieval:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.cli \
  --ticker ASML \
  --start 2020-01-01 \
  --output-dir automated_forecasting_engine/runs/ASML_openai_web_hybrid \
  --enable-alt-news \
  --alt-news-provider openai_web \
  --alt-news-sentiment-mode hybrid \
  --llm-env-file path/to/.env
```

The news pipeline adds rolling features such as `alt_news_sentiment_7d`, `alt_news_volume_7d`, `alt_news_positive_share_7d`, `alt_news_negative_share_7d`, and `alt_news_relevance_7d`. `build_feature_frame()` then turns them into external model features such as `exo_alt_news_sentiment_7d`, one-day changes, and rolling z-scores.

To verify that scraping worked, inspect `forecast_report.json`:

```text
data_manifest.alternative_sources[0].article_count
technical_view.chapter_3_alternative_data.registry[0].provider_status
```

To verify that LLM sentiment ran, inspect:

```text
data_manifest.alternative_sources[0].sentiment.mode
data_manifest.alternative_sources[0].sentiment.llm_status
technical_view.chapter_3_alternative_data.registry[0].quality.llm_status
```

Expected values are `article_count > 0` and, for LLM runs, `llm_status: executed`. Chapter 3 also reports whether the enriched alternative-data feature set beat the market-only baseline in walk-forward validation.

Chapter 3 has a model-fitting consequence: if an alternative source is not point-in-time safe or the provider status is not ok, alternative/news/sentiment features are excluded before candidate validation. Safe alternative features are not forced; they still have to survive normal walk-forward model selection.

## Chapter 4 Alpha Research

The Chapter 4 layer evaluates alpha factors before turning them into stronger modeling rules. It does not override the final action, but weak alpha factors can now affect model fitting.

It adds:

- alpha-factor taxonomy by economic family
- rank-IC and Pearson-IC quality labels
- quantile return tables and monotonicity checks
- top-quantile turnover warnings
- keep/watch/penalize/drop recommendations
- sentiment-factor summaries when Chapter 3 features are present
- a PyWavelets denoising test
- pre-validation feature exclusions for sufficiently sampled factors with very weak IC and quantile-spread evidence

The wavelet test compares raw factors against rolling point-in-time denoised versions using rank IC. If denoising does not clearly improve the factor diagnostics, no denoised features are added to modeling.

The feature-selection policy is stored in each model card under `ml4t_feature_selection_policy`.

Inspect these fields in `forecast_report.json`:

```text
technical_view.chapter_4_alpha_research.recommendations
technical_view.chapter_4_alpha_research.wavelet_denoising_assessment
diagnostics.chapter_4_alpha_research
```

## Chapter 5 Portfolio Evaluation

The Chapter 5 layer adds a lightweight portfolio-performance tear sheet without adding Zipline, Pyfolio, or automatic position sizing. It is report-only unless the evidence is stable across every tested horizon.

It adds:

- Sharpe, Sortino, Calmar, max drawdown, VaR, tail ratio, Omega, hit rate, and profit factor
- benchmark-relative alpha, beta, information ratio, return delta, Sharpe delta, and drawdown delta
- a Fundamental Law proxy based on rank IC and effective breadth
- a stable-improvement gate for enriched features and strategy-vs-benchmark evidence
- portfolio construction candidates such as equal weight, inverse volatility, minimum variance, HRP, and Kelly sizing

The allocation gate stays inactive unless the strategy beats the benchmark on return and Sharpe without materially worse drawdown for every horizon, and enriched features also improved every tested horizon. If that condition fails, `allocation_policy.status` is `tested_not_implemented`.

Inspect these fields in `forecast_report.json`:

```text
technical_view.chapter_5_portfolio_evaluation.allocation_policy
technical_view.chapter_5_portfolio_evaluation.enriched_feature_policy
technical_view.chapter_5_portfolio_evaluation.horizons
diagnostics.chapter_5_portfolio_evaluation
```

## Chapter 6 ML Process

The Chapter 6 layer audits the machine-learning process before any new gate is allowed to influence the final action. It now affects model selection, but not the final action gate directly.

It adds:

- mutual-information feature ranking for nonlinear feature/outcome dependence
- directional classification metrics from validation predictions
- validation-threshold tests for Buy/Sell signal precision and coverage
- bias/variance checks using validation versus final holdout metrics
- leakage-risk checks for walk-forward, purge, embargo, and holdout design
- a promotion policy that says whether the diagnostics are strong enough to consider a real action gate
- candidate-selection penalties for holdout degradation, negative deflated Sharpe, and weak direction plus weak Sharpe

The action gate is not enabled unless every tested horizon passes the Chapter 6 promotion policy and a saved-run stability check supports promotion. Otherwise, `promotion_policy.status` remains `report_only`. Model selection still uses the Chapter 6 penalties through the adjusted validation metric.

Inspect these fields in `forecast_report.json`:

```text
technical_view.chapter_6_ml_process.promotion_policy
technical_view.chapter_6_ml_process.horizons
candidate_results.<horizon>[].metrics.chapter_6_process_selection_penalty
diagnostics.chapter_6_ml_process
```

## Chapter 7 Linear Models

The Chapter 7 layer treats regularized linear models as explicit baselines and diagnostics. Ridge, Lasso, and ElasticNet are all kept because they represent distinct shrinkage regimes rather than duplicate models.

It adds:

- fixed Ridge and Lasso candidates beside the existing ElasticNet candidate
- Ridge/Lasso/ElasticNet comparison by selected validation metric
- prediction information coefficient for selected model validation predictions
- residual diagnostics for fat tails and autocorrelation
- linear coefficient explainability when the selected model exposes coefficients

The layer does not discard tree, boosting, statistical, or LSTM candidates, and it does not use OLS p-values as trading rules.

Inspect these fields in `forecast_report.json`:

```text
technical_view.chapter_7_linear_models.model_registry_policy
technical_view.chapter_7_linear_models.horizons
diagnostics.chapter_7_linear_models
```

## Chapter 8 Backtesting

The Chapter 8 layer audits whether the validation signal backtest is research-grade or close to execution-grade. It does not install Zipline or Backtrader; it strengthens the existing vectorized backtest and reports what is still missing for an event-driven simulation.

It adds:

- explicit execution timing policy for validation signals
- transaction-cost plus slippage sensitivity
- compact trade ledger summary
- turnover pressure warnings
- minimum backtest length versus tested candidate count
- vectorized-versus-event-driven realism audit

The current backtest remains a vectorized validation signal backtest. Its signal-risk metrics have candidate-ranking consequences before model selection, while the event-driven execution gate remains future work.

Inspect these fields in `forecast_report.json`:

```text
backtests
technical_view.chapter_8_backtesting.promotion_policy
technical_view.chapter_8_backtesting.horizons
diagnostics.chapter_8_backtesting
candidate_results.<horizon>[].metrics.chapter_8_backtest_selection_penalty
```

## Chapter 9 Time-Series Diagnostics

The Chapter 9 ML4T layer improves diagnostics around the time-series models that already exist in the engine. It does not add duplicate ARIMA, SARIMA, VAR, or GARCH model families.

Unlike the earlier report-only prototype, these checks now have model-selection consequences. After walk-forward validation and before candidate selection, the engine records a Chapter 9 suitability adjustment in each candidate's validation metrics. Residual autocorrelation, stationarity risk, and volatility-clustering mismatch can penalize the adjusted selection metric. The raw validation metrics are preserved, and the adjusted metrics are written beside them for audit.

It adds:

- stationarity checks for log prices and log returns
- return autocorrelation and volatility-clustering diagnostics
- Ljung-Box residual white-noise audits for selected validation predictions
- reporting of existing time-series candidates by horizon
- candidate-ranking adjustments through `chapter_9_adjusted_<selection_metric>` and `chapter_9_selection_penalty`
- a standalone pairs-trading helper for multi-ticker research

Standalone pairs-trading helper note: `market_forecasting_engine.pairs_trading` is available for multi-ticker cointegration research only. It is not called by `ForecastingEngine.run`, does not change `suggested_action`, and is not part of the single-ticker action gate. Use it only when a real multi-ticker price panel or universe is supplied.

Inspect these fields in `forecast_report.json`:

```text
technical_view.chapter_9_time_series.model_policy
technical_view.chapter_9_time_series.series_diagnostics
technical_view.chapter_9_time_series.horizons
candidate_results.<horizon>[].metrics.chapter_9_selection_penalty
candidate_results.<horizon>[].metrics.chapter_9_adjusted_<selection_metric>
diagnostics.chapter_9_time_series
```

## Chapter 10 Bayesian ML

The Chapter 10 ML4T layer adds Bayesian uncertainty to model selection and forecast confidence. It is separate from the existing Edwards/Magee `chapter_10_patterns` technical-analysis module.

By default, the implementation is conservative and lightweight:

- estimates the probability that each candidate's validation Sharpe is positive
- penalizes model selection when posterior Sharpe evidence is weak or very uncertain
- records `chapter_10_bayesian_selection_penalty` in candidate metrics
- lowers selected forecast directional confidence with `chapter_10_confidence_multiplier`
- keeps the final Buy/Hold/Sell gate unchanged

The optional heavy Bayesian path is disabled by default. Enable it only when you want PyMC/MCMC diagnostics during a run:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.cli \
  --ticker ASML \
  --start 2020-01-01 \
  --output-dir automated_forecasting_engine/runs/ASML_bayesian_heavy \
  --enable-bayesian-heavy \
  --bayesian-mcmc-draws 300 \
  --bayesian-mcmc-tune 300
```

When the heavy flag is false, `technical_view.chapter_10_bayesian_ml.horizons.<horizon>.heavy_bayesian_path.status` is `disabled`. When true, the run attempts a PyMC posterior for selected validation strategy returns and reports `executed`, `unavailable`, or `not_available`. Even when heavy Bayesian diagnostics run, they do not create a direct Bayesian trading gate and do not automatically promote a new stochastic-volatility model.

Inspect these fields in `forecast_report.json`:

```text
technical_view.chapter_10_bayesian_ml.heavy_bayesian_policy
technical_view.chapter_10_bayesian_ml.horizons
candidate_results.<horizon>[].metrics.chapter_10_bayesian_selection_penalty
candidate_results.<horizon>[].metrics.chapter_10_prob_sharpe_positive
forecasts[].validation_metrics.chapter_10_confidence_multiplier
diagnostics.chapter_10_bayesian_ml
```

## Output

The CLI prints a concise forecast summary and writes:

- `forecast_report.json`
- `governance/model_card_<ticker>.json`
- `governance/data_manifest_<ticker>.json`
- `governance/data_quality_<ticker>.json`
- `plots/forecast_<ticker>.png`
- `plots/forecast_<ticker>.html`
- `plots/technical_<ticker>.png`
- `plots/technical_<ticker>.html`
- `plots/technical_clean_<ticker>.png`
- `plots/technical_clean_<ticker>.html`
- `plots/technical_<ticker>_daily.png`
- `plots/technical_<ticker>_daily.html`
- `plots/technical_<ticker>_weekly.png`
- `plots/technical_<ticker>_weekly.html`
- `plots/technical_<ticker>_monthly.png`
- `plots/technical_<ticker>_monthly.html`
- `plots/validation_<ticker>_<horizon>d.png`
- `plots/validation_<ticker>_<horizon>d.html`

The report includes selected models, validation metrics, confidence intervals, a buy/hold/sell signal, risk level, data hash, feature set, trend view, decision diagnostics, and training windows.

The full technical chart keeps the complete/debug overlay set. The clean technical chart is the trader-facing companion: it uses a linear price axis and limits the view to price/OHLC, SMA 20, SMA 50, Donchian 20 channel, nearest daily support/resistance, recent breakout/breakdown markers, and a ranked signal table.

## Portfolio PDF Projection

The portfolio runner extracts positions from a Trade Republic net-worth PDF, maps known ISINs/names to Yahoo symbols, runs one forecast per holding, and projects each EUR position value with the selected forecast horizon's expected return.

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.portfolio_cli \
  --pdf "/Users/ruddigarcia/Downloads/Patrimonio neto 3.pdf" \
  --start 2020-01-01 \
  --horizons 1,5,30 \
  --projection-horizon 5 \
  --output-dir automated_forecasting_engine/runs/portfolio_projection
```

It writes:

- `portfolio_projection.csv`
- `portfolio_projection.json`
- `portfolio_projection.html`
- `portfolio_projection.png`
- `portfolio_projection_plotly.html`
- `ticker_reports/<symbol>/forecast_report.json`
- `extracted_holdings.json`

The projection applies forecast percentage returns to the statement's EUR values, so US, European, ETF, and crypto price currencies do not get mixed in the portfolio total.

## Autonomous LLM Trader

The autonomous trader is separate from the forecast CLI. The normal command `market_forecasting_engine.cli` still runs the rule-based forecast without needing an LLM.

The LLM trader runs one ticker forecast first, builds a compact technical packet from the governed report, then calls OpenAI Responses API using the prompt in `src/market_forecasting_engine/llm_trader/prompts/autonomous_trader.py`. Web search is enabled by default so the trader can inspect recent news, market sentiment, and visible forum/discussion context before returning a structured Buy/Hold/Sell plan.

After the trader decision, a second LLM call uses `src/market_forecasting_engine/llm_trader/prompts/nontechnical_summary.py` to translate the full decision into a beginner-friendly summary. This second prompt does not use web search; it only summarizes the trader JSON and keeps the key numbers such as buy levels, stop loss, take-profit, and sell/trim zones. Price levels are reported in dollars with the euro equivalent next to them, for example `$1,505.22 (€1,385.00)`, using a live Yahoo `EURUSD=X` conversion when available. The summary is also decision-oriented: each future trigger must say whether it is for considering a buy, sell, hold, risk reduction, profit-taking, or continued waiting.

The autonomous trader and beginner-summary calls default to `gpt-5.4-mini-2026-03-17` with reasoning effort `none`. Use `--llm-model`, `--summary-model`, `--reasoning-effort`, or `--summary-reasoning-effort` to override those defaults.

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.llm_trader.cli \
  --ticker AAPL \
  --start 2020-01-01 \
  --profile medium \
  --llm-env-file /Users/ruddigarcia/Projects/invest/.env \
  --output-dir automated_forecasting_engine/runs/aapl_llm_trader
```

Profiles:

- `aggressive`: faster entries and higher volatility tolerance, but still respects hard rule gates
- `medium`: balanced technical, sentiment, and execution-quality profile
- `conservative`: waits for stronger alignment and prioritizes capital preservation

For an existing holding:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.llm_trader.cli \
  --ticker AAPL \
  --start 2020-01-01 \
  --profile conservative \
  --holding-status owned \
  --entry-price 180 \
  --quantity 10 \
  --llm-env-file /Users/ruddigarcia/Projects/invest/.env \
  --output-dir automated_forecasting_engine/runs/aapl_owned_llm_trader
```

Useful switches:

- `--no-web-search` disables OpenAI web search and uses only the forecast packet
- `--no-summary` disables the second beginner-summary LLM call
- `--usd-eur-rate 0.92` manually overrides the USD to EUR conversion used by the beginner summary
- `--dry-run` builds the prompt packet without calling OpenAI
- `--search-context-size low|medium|high` controls how much web context the model can gather
- `--reasoning-effort none|minimal|low|medium|high` controls the trader Responses API reasoning effort
- `--trader-name` gives the trader instance a name for audit records

It writes:

- `forecast_report.json`
- `trader_decision.json`
- `trader_decision_only.json`
- `trader_summary.json`
- `trader_prompt_packet.json`

## Terminal Watch Agent

The watch agent is not an LLM agent during the watch loop. On startup it uses watcher memory for the ticker/profile. If no memory exists, it runs the full forecast + LLM trader pipeline once, saves that decision, and then only watches live price levels. It will not call the forecast or LLM again until the memory is older than `--refresh-after-hours` hours, or unless you pass `--force-refresh`.

Run it and let it create or reuse its remembered LLM trader decision:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.watch_agent.cli \
  --ticker ASML \
  --profile medium \
  --holding-status not_owned \
  --llm-env-file /Users/ruddigarcia/Projects/invest/.env \
  --once
```

Force a fresh forecast + LLM trader decision at startup:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.watch_agent.cli \
  --ticker ASML \
  --profile medium \
  --holding-status not_owned \
  --llm-env-file /Users/ruddigarcia/Projects/invest/.env \
  --force-refresh \
  --once
```

You can also point it at a completed LLM trader folder:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.watch_agent.cli \
  --ticker ASML \
  --profile medium \
  --run-dir automated_forecasting_engine/runs/asml_llm_trader \
  --holding-status not_owned
```

By default it checks every hour. For a quick one-time check without live price fetching:

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.watch_agent.cli \
  --ticker ASML \
  --profile medium \
  --run-dir automated_forecasting_engine/runs/asml_llm_trader \
  --holding-status not_owned \
  --price 1500 \
  --once
```

It prints capitalized terminal actions such as:

```text
ACTION: BUY
*** BUY ASML ***
```

Use `--holding-status owned` if the ticker is already in your portfolio. In that mode the watcher prints `SELL` when the stop loss, sell-near level, or take-profit level is reached.

For macOS hourly scheduling, use launchd. The watcher should run once, check the latest live price, print/log only if something changed or an alert fired, then exit. launchd starts it again one hour later. If the Mac is asleep at the scheduled time, the job runs the next time launchd gets a chance after the machine is awake; it does not keep a Python process running all day. The full forecast + LLM trader decision is refreshed when the remembered advice is older than `REFRESH_AFTER_HOURS`, which defaults to 12 hours.

Install a LaunchAgent with the installer script:

```bash
automated_forecasting_engine/scripts/install_watch_agent_launchd.sh \
  --ticker NVDA \
  --profile conservative \
  --label com.marketforecasting.watchagent.nvda.conservative
```

The label must be unique. Existing agents are not affected unless you intentionally reuse the same label with `--replace`.

The installer defaults to quiet mode, so unchanged hourly `HOLD` checks do not fill the terminal log. Use `--print-unchanged` if you want every hourly check printed.

You can still install the example LaunchAgent manually:

```bash
mkdir -p ~/Library/LaunchAgents
cp automated_forecasting_engine/scripts/com.marketforecasting.watchagent.plist.example \
  ~/Library/LaunchAgents/com.marketforecasting.watchagent.asml.medium.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.marketforecasting.watchagent.asml.medium.plist
launchctl kickstart -k gui/$(id -u)/com.marketforecasting.watchagent.asml.medium
```

The LaunchAgent calls:

```bash
automated_forecasting_engine/scripts/run_watch_agent_once.sh
```

Configure ticker/profile by editing the copied plist environment variables:

- `TICKER`
- `PROFILE`
- `HOLDING_STATUS`
- `REFRESH_AFTER_HOURS`
- `LLM_ENV_FILE`
- `QUIET_UNCHANGED`

The watcher overwrites the remembered LLM trader run at:

```text
automated_forecasting_engine/runs/watch_agent_state/llm_run/<TICKER>_<profile>/
```

It appends every hourly check to daily JSONL logs:

```text
automated_forecasting_engine/runs/watch_agent_state/logs/<TICKER>_<profile>_YYYYMMDD.jsonl
```

Progress is printed when the watcher does real work, such as the daily full forecast + LLM refresh. In a normal terminal run you see it directly. In launchd, use the stdout log for the agent label:

```bash
tail -f automated_forecasting_engine/runs/watch_agent_state/com.marketforecasting.watchagent.asml.medium.stdout.log
```

The progress output shows whether the watcher refreshed the full forecast + LLM trader decision, fetched the current price, wrote the daily decision log, and finished the one-shot wake-up. Use `--quiet-unchanged` to suppress unchanged checks, or `--no-progress` if you only want the final `ACTION` lines.

To print the latest full beginner report for every active launchd trader:

```bash
automated_forecasting_engine/scripts/print_active_trader_summaries.py
```

Use `--compact` if you only want the short summary, action note, and recheck triggers.

For repeat updates, run the provided shell job:

```bash
automated_forecasting_engine/scripts/update_portfolio_projection.sh
```

Useful overrides:

```bash
PORTFOLIO_PDF="/path/to/new_statement.pdf" \
OUTPUT_DIR="automated_forecasting_engine/runs/portfolio_projection_latest" \
PROJECTION_HORIZON=5 \
automated_forecasting_engine/scripts/update_portfolio_projection.sh
```

The data manifest records providers, request parameters, raw/normalized artifact paths, hashes, context sources, point-in-time policy, security metadata, exchange-calendar alignment, and universe/panel metadata. The data-quality report checks missing data, duplicate dates, non-positive prices, stale prices, return outliers, and missing trading sessions.

The validation plots use walk-forward out-of-sample predictions from the selected model, not in-sample fitted values. The forecast plot shows recent actual prices, forecast horizon points, and confidence intervals. The Plotly `.html` versions are interactive and expose actual/predicted values in hover labels.

The technical chart is the Edwards/Magee market-action view. It uses OHLC/candlestick-style price bars, close overlay, moving averages, support/resistance, Magee basing-point stops, Head-and-Shoulders Top/Bottom neckline and objective overlays, triangle boundary/objective overlays, Chapter 9 rectangle and double/triple top-bottom overlays, Chapter 10 broadening/wedge/diamond and one-day event overlays, Chapter 11 flag/pennant continuation overlays, Chapter 12 gap-zone overlays, Chapter 13 support/resistance zone bands, Chapter 14 trendline/channel/fan-line overlays, Chapter 15 major-trendline overlays, Chapter 16 Donchian market-context overlays, optional Dormant Bottom base zones, breakouts, breakdowns, gaps, and volume. Daily, weekly, and monthly versions are written so short-term, intermediate, and major support/resistance can be inspected separately.

Forecast intervals are calibrated from the selected model's walk-forward validation residuals. The report records the interval method and calibration sample size for each horizon.

The report also includes:

- `diagnostics.technical_structure`
- `diagnostics.factor_evaluation`
- `technical_view.trend_state`
- `technical_view.dow_theory`
- `technical_view.magee_basing_points`
- `technical_view.reversal_patterns`
- `technical_view.triangle_patterns`
- `technical_view.chapter_9_patterns`
- `technical_view.rectangle_patterns`
- `technical_view.multi_top_bottom_patterns`
- `technical_view.chapter_10_patterns`
- `technical_view.chapter_10_structural_patterns`
- `technical_view.chapter_10_short_term_events`
- `technical_view.chapter_11_patterns`
- `technical_view.chapter_11_continuation_patterns`
- `technical_view.chapter_11_head_and_shoulders_continuation`
- `technical_view.chapter_12_gaps`
- `technical_view.chapter_12_classified_gaps`
- `technical_view.chapter_12_island_reversals`
- `technical_view.chapter_13_support_resistance`
- `technical_view.chapter_13_support_zones`
- `technical_view.chapter_13_resistance_zones`
- `technical_view.chapter_14_trendlines`
- `technical_view.chapter_14_channels`
- `technical_view.chapter_14_fan_lines`
- `technical_view.chapter_15_major_trendlines`
- `technical_view.chapter_15_scale_comparison`
- `technical_view.chapter_15_broad_market_confirmation`
- `technical_view.chapter_16_market_context`
- `technical_view.chapter_16_donchian_context`
- `technical_view.chapter_16_futures_risk_context`
- `technical_view.chapter_17_governance_context`
- `technical_view.chapter_17_llm_decision_packet`
- `technical_view.chapter_17_decision_fragility`
- `technical_view.chapter_18_tactical_problem`
- `technical_view.chapter_18_tactical_plan`
- `technical_view.chapter_18_llm_review`
- `technical_view.chapter_19_validation`
- `technical_view.chapter_20_ticker_suitability`
- `technical_view.chapter_21_chart_selection`
- `technical_view.chapter_23_30_trade_risk_plan`
- `technical_view.chapter_31_42_portfolio_capital_risk`
- `technical_view.chapter_39_43_discipline_governance`
- `decision_view`
- `operations_view`
- `operations_view.chapter_19_validation`
- `selection_view`
- `selection_view.chapter_20_ticker_suitability`
- `selection_view.chapter_21_chart_selection`
- `trade_risk_view`
- `trade_risk_view.chapter_23_30_trade_risk_plan`
- `portfolio_view.mark_to_market`
- `portfolio_view.chapter_31_42_portfolio_capital_risk`
- `discipline_view`
- `discipline_view.chapter_39_43_discipline_governance`
- `diagnostics.dow_theory`
- `diagnostics.magee_basing_points`
- `diagnostics.reversal_patterns`
- `diagnostics.triangle_patterns`
- `diagnostics.chapter_9_patterns`
- `diagnostics.chapter_10_patterns`
- `diagnostics.chapter_11_patterns`
- `diagnostics.chapter_12_gaps`
- `diagnostics.chapter_13_support_resistance`
- `diagnostics.chapter_14_trendlines`
- `diagnostics.chapter_15_major_trendlines`
- `diagnostics.chapter_16_market_context`
- `diagnostics.chapter_17_governance_context`
- `diagnostics.chapter_18_tactical_problem`
- `diagnostics.chapter_19_validation`
- `diagnostics.chapter_20_ticker_suitability`
- `diagnostics.chapter_21_chart_selection`
- `diagnostics.chapter_23_30_trade_risk_plan`
- `diagnostics.chapter_31_42_portfolio_capital_risk`
- `diagnostics.chapter_39_43_discipline_governance`
- `technical_view.technical_history_quality`
- `technical_view.support_resistance_by_timeframe`
- `technical_view.chart_metadata`
- `technical_view.decision_diagnostics`
- `technical_view.dow_action_filter`
- `technical_view.magee_action_filter`
- `technical_view.reversal_action_filter`
- `technical_view.triangle_action_filter`
- `technical_view.chapter_9_action_filter`
- `technical_view.chapter_10_action_filter`
- `technical_view.chapter_11_action_filter`
- `technical_view.chapter_12_gap_filter`
- `technical_view.chapter_13_zone_filter`
- `technical_view.chapter_14_trendline_filter`
- `technical_view.chapter_15_major_trendline_filter`
- `technical_view.market_only_vs_enriched`
- `diagnostics.dow_chapter_4_defects`
- `governance.feature_registry_summary`
- `governance.feature_registry`
- `governance.technical_method_card`
- `governance.portfolio_method_cards`
- `governance.discipline_method_cards`
- `governance.technical_method_cards`
- `governance.tactical_method_cards`
- `governance.operational_method_cards`
- `governance.selection_method_cards`
- `governance.trade_risk_method_cards`
- `backtests`
- `candidate_results[*].metrics.holdout_*`
- `candidate_results[*].metrics.deflated_sharpe_ratio`
- `forecasts[*].trade_quality`

## Current Candidate Models

Statistical and structural baselines:

- historical mean return
- recent mean return
- exponential smoothing return
- ARIMA
- SARIMA
- GARCH
- VAR
- Kalman filter

Machine learning:

- Elastic Net
- Random Forest
- Gradient Boosting
- LightGBM, when installed and enabled
- LSTM, when explicitly enabled with `--include-lstm`

`--search-level expanded` adds more ARIMA/SARIMA orders, GARCH `p/q` variants, VAR lag settings, and ML hyperparameter variants. The engine chooses among them with the same walk-forward objective metrics used for every other candidate.

`--tune optuna` adds optimized machine-learning candidates alongside the fixed candidates. Supported Optuna families are `lightgbm`, `elastic_net`, `random_forest`, and `gradient_boosting`; use `--optuna-families lightgbm,elastic_net` to restrict the search. The inner Optuna objective is walk-forward MAE, and the final holdout remains unavailable to tuning and model selection.

`--tactical-profile short_term|intermediate|long_term` controls the Chapter 18 stop, reward/risk, and holding-horizon assumptions. `--enable-llm-review` adds the optional governed OpenAI reviewer; use `--llm-env-file /path/to/.env` when the API key is stored outside the shell environment. The reviewer uses the same Responses API default model, `gpt-5.4-mini-2026-03-17`, with reasoning effort `none` unless overridden. The LLM can downgrade a directional action to `Hold`, but it cannot upgrade a rule-based `Hold` or override hard blockers.

The GARCH candidate records volatility diagnostics in the model card. The VAR candidate can use exogenous market, sector, volatility, macro, rate, and event features when those inputs are provided.

## Book-Driven Additions

Inspired by Edwards/Magee, the engine now models market structure rather than only raw returns:

- trend-first market-action view
- decision blockers/supporting reasons
- technical-history sufficiency diagnostics
- market-only versus enriched-feature validation comparison
- annotated technical chart artifacts
- OHLC/candlestick-style daily, weekly, and monthly technical charts
- semilog technical chart scale by default
- timeframe-specific support/resistance levels
- Dow Theory-inspired primary, secondary, and minor trend hierarchy
- close-confirmed breakout/breakdown diagnostics
- benchmark/sector trend confirmation when context series are supplied
- secondary retracement classification
- volume-trend confirmation
- compact sideways line/range detection
- continuation-until-reversal diagnostic
- Chapter 4 Dow Theory defect controls: signal-lag diagnostics, sensitivity/ambiguity scoring, and technical-regime backtests
- Dow-style regime filter for final `Buy`/`Hold`/`Sell` actions
- Magee basing-points procedure with daily and weekly confirmed wave highs/lows
- Magee Variant 1 and Variant 2 stairstep stop diagnostics and backtests
- Magee action filter for final `Buy`/`Hold`/`Sell` actions
- Chapter 6 Head-and-Shoulders Top detection with confirmed pivots, neckline break confirmation, volume context, pullback status, and measured objective
- Chapter 7 Head-and-Shoulders Bottom detection with prior downtrend validation, upside neckline confirmation, volume context, throwback status, and measured objective
- monthly reversal-pattern scans for completed major formations
- complex Head-and-Shoulders warning diagnostics for broad/multiple formations
- optional Dormant Bottom diagnostics for quiet accumulation bases
- reversal-pattern action filter for final `Buy`/`Hold`/`Sell` actions
- Chapter 8 triangle diagnostics for Symmetrical, Ascending, and Descending Triangles with apex timing, boundary breakout, volume confirmation, retest state, and measured objective
- triangle-pattern action filter for wait states, failed breakouts, and forecast-direction conflicts
- Chapter 9 rectangle diagnostics with horizontal supply/demand boundaries, breakout/breakdown status, volume contraction, retest state, premature/false break notes, and measured objective
- Chapter 9 Double/Triple Top and Bottom diagnostics with prior-trend validation, equal-level tolerance, confirmation level, volume context, and measured objective
- Chapter 9 action filter for range wait states, rectangle break conflicts, and confirmed double/triple reversal conflicts
- Chapter 10 Broadening Top, Right-Angled Broadening, Diamond, Rising Wedge, and Falling Wedge diagnostics with boundary breaks, volume context, retest state, and measured objectives
- Chapter 10 one-day event diagnostics for One-Day Reversals, Selling Climaxes, Spikes, Runaway Days, and Key Reversal Days
- Chapter 10 action filter for broadening-top conflicts, wedge/diamond break conflicts, and strong short-term exhaustion events
- Chapter 11 Flag and Pennant continuation diagnostics with mast validation, short consolidation windows, volume contraction, breakout volume, and half-mast objectives
- Chapter 11 Head-and-Shoulders continuation and optional Scallop context diagnostics
- Chapter 11 action filter that adds supporting reasons when continuation agrees with the model and blocks failed or conflicting continuation patterns
- Chapter 12 gap diagnostics for Common/Area, Breakaway, Runaway/Measuring, Exhaustion, and Island Reversal gaps
- Chapter 12 gap-zone state tracking for open, partially filled, filled, and quickly closed gaps
- Chapter 12 action filter that supports matching breakaway/runaway gaps and blocks exhaustion or island-reversal conflicts
- Chapter 13 support/resistance zone diagnostics with volume-weighted zone strength, role reversal, round-number context, attack counts, and multi-timeframe state
- Chapter 13 action filter that blocks fresh buys into nearby strong resistance, blocks fresh sells into nearby strong support, and respects volume-confirmed support failures or resistance breakouts
- Chapter 14 pivot-confirmed trendline, double-trendline, trend-channel, pullback, and Three-Fan diagnostics
- Chapter 14 action filter that blocks fresh buys after decisive uptrend-line breaks, blocks fresh sells after decisive downtrend-line breaks, and respects active authoritative lines near price
- Chapter 15 major monthly trendline diagnostics with log-vs-linear scale comparison, major-trend shape classification, post-base anchor selection, and optional broad-market confirmation
- Chapter 15 action filter that treats major uptrend-line breaks as risk controls, keeps major bear trendline breaks warning-only unless other evidence agrees, and never requires benchmark data to run
- Chapter 16 report-only market context with trending-versus-trading diagnostics, Donchian 20/55 channel state, optional open-interest context, seasonality, and futures-style risk notes
- Chapter 17 report-only governance context for downstream human/LLM review: decision fragility, method conflicts, filter-stack pressure, volume context, mark-to-market discipline, portfolio risk gaps, and optional hedging context
- Chapter 18 tactical plan with profile-specific stop selection, target selection, reward/risk gate, mark-to-market discipline, optional LLM review, and post-LLM safety gate
- Chapter 18 LLM policy that keeps rules in control: the LLM may review, explain, or downgrade to `Hold`, but cannot create a directional action when rules say `Hold`
- Chapter 19 operational validation for repeatable run discipline: data source, chart/artifact records, validation records, technical packet completeness, governance records, and a conditional `Hold` gate for non-auditable runs
- Chapter 20 ticker suitability classification for short-term trading, intermediate trading, long-term investing, speculative satellite monitoring, and index/diversifier use
- Chapter 21 chart-book selection that promotes completed reports into trade candidates, active review, watchlist, monitor-only, or excluded buckets
- Chapters 23-30 trade/risk plan with speculative controls, probable move bands, margin/short policy, risk-budget sizing, protective/progressive stops, pivot confirmation, trendline execution, and support/resistance execution
- Chapters 31, 38, and 40-42 portfolio capital/risk layer with diversification checks, balance policy, account-equity risk budget, staged capital application, drawdown/volatility risk, and missing-input disclosure
- Chapters 39 and 43 discipline/governance layer with trial/error evidence, method-change rules, plan-adherence checks, capital-gate warnings, and review triggers
- split `Hold` reasons such as `NoEdge`, `RiskBlocked`, `TrendDoubt`, `RangeWait`, `RegimeBlocked`, `BasingStopTooClose`, `BelowBasingStop`, `LongTermTrendIntact`, `ConfirmedReversalTop`, `ConfirmedReversalBottom`, `TriangleWait`, `TriangleFailure`, `TriangleBreakoutConflict`, `RectangleWait`, `RectangleFailure`, `RectangleBreakoutConflict`, `ConfirmedDoubleTop`, `ConfirmedDoubleBottom`, `ConfirmedTripleTop`, `ConfirmedTripleBottom`, `BroadeningTopConflict`, `WedgeBreakConflict`, `DiamondBreakConflict`, `ShortTermExhaustion`, `ContinuationConflict`, `StaleContinuation`, `FailedContinuation`, `ExhaustionGap`, `IslandReversal`, `GapBreakawayConflict`, `ResistanceTooClose`, `SupportTooClose`, `SupportFailure`, `ResistanceBreakout`, `UpTrendlineBreak`, `DownTrendlineBreak`, `ActiveUpTrendline`, `ActiveDownTrendline`, `BullishFanBreak`, `BearishFanBreak`, `MajorUpTrendlineBreak`, `MajorDownTrendlineBreak`, `MajorBullTrendIntact`, and `BroadMarketMajorDivergence`
- technical method cards with Dow, Magee, reversal-pattern, triangle, Chapter 9, Chapter 10, Chapter 11, Chapter 12, Chapter 13, Chapter 14, Chapter 15, Chapter 16, and Chapter 17 lookbacks, thresholds, pivot rules, line rules, stop rules, context rules, and confirmation rules
- tactical method cards with Chapter 18 profile, stop, target, reward/risk, LLM-review, and safety-gate rules
- operational method cards with Chapter 19 validation checks and action-gate policy
- selection method cards with Chapter 20 profile scoring and Chapter 21 chart-book selection policy
- trade/risk method cards with Chapters 23-30 execution and risk-control rules
- portfolio method cards with Chapters 31, 38, and 40-42 capital and portfolio-risk rules
- discipline method cards with Chapters 39 and 43 evidence and plan-adherence rules
- universe ranking for the "what stock" question
- support and resistance distance
- range position
- breakout/breakdown flags

## When To Use Chapter 7 Reversal Diagnostics

Use `technical_view.reversal_patterns` as a governance and interpretation layer, not as a standalone price forecast. The statistical/ML model still estimates expected return; the reversal layer explains whether the chart structure supports or conflicts with that forecast.

Use Head-and-Shoulders Top diagnostics when:

- the forecast is bullish but the stock recently had a large advance
- you need to know whether a fresh `Buy` conflicts with a confirmed distribution pattern
- the weekly or monthly chart is showing a neckline break or pullback to a broken neckline

Use Head-and-Shoulders Bottom diagnostics when:

- the forecast is bearish but the stock recently had a large decline
- you need to know whether a fresh `Sell` conflicts with a confirmed accumulation pattern
- upside breakout volume is strong enough to validate the bottom

Use Complex Head-and-Shoulders diagnostics only for manual review. They are warning-only because complex patterns are less clean and easier to overfit.

Use Dormant Bottom diagnostics only as an optional long-horizon accumulation screen. They are most useful for thin, quiet, or forgotten stocks after a major decline, where price forms a long flat base and then breaks out on stronger volume. Do not use Dormant Bottoms as an automatic `Buy` rule for liquid large-cap stocks; treat them as context that deserves chart review and position-size discipline.

## When To Use Chapter 8 Triangle Diagnostics

Use `technical_view.triangle_patterns` as a continuation/reversal context layer. Triangles often pause an existing move rather than reverse it, so the engine does not treat a triangle by itself as a Buy or Sell signal.

Use Symmetrical Triangle diagnostics when:

- price is compressing between falling resistance and rising support
- the forecast is directional but price has not broken either boundary
- you need a `TriangleWait` reason until a decisive breakout or breakdown appears

Use Ascending Triangle diagnostics when:

- resistance is flat, lows are rising, and you want evidence of accumulation
- the model is bullish and price has broken above resistance
- upside breakout volume is present or absent and needs to be audited

Use Descending Triangle diagnostics when:

- support is flat, highs are falling, and you want evidence of distribution
- the model is bearish and price has broken below support
- a fresh Buy conflicts with a confirmed downside breakdown

Treat late-apex triangle breakouts with caution. Chapter 8 emphasizes that breakouts are usually more reliable before the pattern pushes too far toward the apex; the report marks this with `apex.timing` and lower reliability notes.

Triangle diagnostics are most useful for timing and risk control around model forecasts. They should not replace walk-forward validation, risk metrics, or portfolio exposure limits.

## When To Use Chapter 9 Rectangle And Double/Triple Diagnostics

Use `technical_view.chapter_9_patterns` when price is moving sideways between repeated supply and demand zones or when two or three major highs/lows look similar.

Use Rectangle diagnostics when:

- price is oscillating between horizontal support and resistance
- the model is directional but price has not yet broken the range
- you need a `RectangleWait` reason until a confirmed boundary break appears
- a forecast conflicts with a confirmed rectangle breakout or breakdown

Use Double/Triple Top and Bottom diagnostics conservatively:

- suspected formations are warnings only
- tops confirm only after price breaks the intervening valley
- bottoms confirm only after price breaks the intervening peak
- confirmed tops can block fresh `Buy` actions
- confirmed bottoms can block fresh `Sell` actions

Chapter 9 diagnostics are stricter than ordinary support/resistance features. They require repeated touches, meaningful pattern width, prior-trend context for double/triple formations, and confirmation before they can veto a model signal.

## When To Use Chapter 10 Reversal Phenomena

Use `technical_view.chapter_10_patterns` when the chart shows disorder, exhaustion, or a sharp one-day change in control. These diagnostics are deliberately conservative because Chapter 10 patterns are rarer and more tactical than the major reversal formations.

Use structural Chapter 10 diagnostics when:

- a Broadening Top or Right-Angled Broadening pattern appears after an advance
- a Rising Wedge breaks down and conflicts with a fresh `Buy`
- a Falling Wedge breaks upward and conflicts with a fresh `Sell`
- a Diamond breaks decisively and conflicts with the model direction

Use short-term Chapter 10 event diagnostics when:

- a Key Reversal Day or One-Day Reversal appears after a strong move
- a Selling Climax argues against panic selling after a sharp decline
- a Spike or Runaway Day needs follow-through before trusting the move

One-day Chapter 10 events are tactical. They can become model features and warnings, and very strong opposing events can force `Hold`, but they should not become automatic standalone `Buy` or `Sell` rules.

## When To Use Chapter 11 Continuation Diagnostics

Use `technical_view.chapter_11_patterns` when price pauses after a sharp move and you need to know whether the pause is a healthy continuation or a warning.

Use Flag and Pennant diagnostics when:

- there is a sharp mast immediately before the consolidation
- the consolidation is short, normally no more than about 3-4 trading weeks
- volume contracts inside the pause and ideally expands on breakout
- you need a half-mast objective for the next projected move

Use Chapter 11 controls conservatively. Confirmed continuation patterns can support a matching `Buy` or `Sell`, but candidates mostly add context until they break. Failed continuations can block the matching directional action, and stale patterns are warning evidence because true flags and pennants should not drift sideways for months.

## When To Use Chapter 12 Gap Diagnostics

Use `technical_view.chapter_12_gaps` when the chart shows a true non-overlapping price range between adjacent bars. The engine separates true range gaps from ordinary opening gaps, filters dividend/split-driven gaps, and downgrades issues that gap habitually.

Use the gap classes this way:

- Common/Area Gap: context only, often inside congestion
- Breakaway Gap: supports a confirmed breakout from support, resistance, trendline, or congestion
- Runaway/Measuring Gap: supports an existing fast move and provides a measured objective
- Exhaustion Gap: warning that a fast move may be ending, especially if it fills quickly or appears with extreme volume
- Island Reversal: tactical reversal warning when opposite gaps isolate a compact trading range

Chapter 12 does not assume every gap must close. Open breakaway and runaway gaps can be bullish or bearish evidence; quick closure is more important when judging exhaustion.

## When To Use Chapter 13 Support And Resistance Diagnostics

Use `technical_view.chapter_13_support_resistance` when price is close to an important historical supply/demand zone, when a forecast points directly into nearby overhead resistance, or when a bearish forecast is already near strong support.

Use the zone diagnostics this way:

- Support zone: can support a matching `Buy` signal when price is close and the zone remains intact
- Resistance zone: can support a matching `Sell` signal when price is close and the zone remains intact
- OldTopAsSupport: prior resistance that now acts as support after a breakout
- OldBottomAsResistance: prior support that now acts as resistance after a breakdown
- SupportFailure: volume-confirmed break below support that can block a fresh `Buy`
- ResistanceBreakout: volume-confirmed break above resistance that can block a fresh `Sell`
- PsychologicalRoundNumber: context zone only unless price action confirms it

Daily zones are tactical, weekly zones are intermediate, and monthly zones are major-context diagnostics. Zone strength is based on volume near the zone, number of touches, distance price moved away, age, attack count, role reversal, and round-number context.

Chapter 13 does not replace the forecasting model. It is a reward/risk and governance layer that explains whether the model signal has enough chart room to work.

## When To Use Chapter 14 Trendline And Channel Diagnostics

Use `technical_view.chapter_14_trendlines` when you need to know whether a trend is still intact, whether a line break is decisive enough to matter, or whether price is deteriorating inside a trend channel before the basic line breaks.

Use the diagnostics this way:

- Active uptrend line: supports a matching `Buy` and can block a premature `Sell` when the line is authoritative and close
- Active downtrend line: supports a matching `Sell` and can block a premature `Buy` when the line is authoritative and close
- DecisiveBreak: close beyond the relevant line by about 3%; can block the conflicting model action
- BorderlineBreak: smaller close penetration with confirming volume; warning or blocker depending on context
- InnerLineBreak: double-trendline warning where the outer line has not yet broken
- ReturnLineFailure: channel deterioration when price fails to reach the return line
- PullbackToBrokenLine: failed reclaim of a broken line after the initial penetration
- ThirdFanBreakUpside/Downside: corrective Three-Fan signal, used only for secondary corrections

Chapter 14 is a governance layer, not a standalone trading system. The engine keeps the model forecast, risk metrics, support/resistance, gaps, and pattern context in the final decision.

## When To Use Chapter 15 Major Trendline Diagnostics

Use `technical_view.chapter_15_major_trendlines` when you need a long-horizon monthly view of the stock's major trend, especially before treating a model forecast as an aggressive directional signal.

Use the diagnostics this way:

- Major uptrend line: long-range bull-regime support that can block a fresh `Sell` when intact and authoritative
- Major uptrend-line break: risk-control warning that can block a fresh `Buy`, especially when the break is beyond the individual-stock threshold
- Major downtrend line: long-range bear-regime context; breaks are warning-only because Chapter 15 treats major bear lines as less dependable
- Scale comparison: records whether linear or semilog better fits the stock's long historical habit
- Broad-market confirmation: confirms or conflicts with the stock's major trend only when benchmark, sector, market, or index columns are supplied

Benchmark data is optional. If no `benchmark_`, `sector_`, `market_`, or `index_` columns exist, `technical_view.chapter_15_broad_market_confirmation.status` is `Unavailable` and the stock-only Chapter 15 diagnostics still run. The missing benchmark warning is informational; it does not block the pipeline or force a final `Hold`.

The engine uses a wider break threshold for individual stocks and a tighter one for supplied broad-market/index series: about 3% for the stock and about 2% for benchmark-style context.

## When To Use Chapter 16 Market Context

Use `technical_view.chapter_16_market_context` to inspect whether the instrument is behaving like a trending market or a trading/range market. This layer is always calculated, but it is report-only: `decision_policy.influences_final_action` is always `false`.

Chapter 16 diagnostics include:

- Donchian 20/55-day breakout context, inspired by Turtle-style trend-following systems
- trending-versus-trading market classification using recent path efficiency and return
- optional open-interest diagnostics when an `open_interest` column is supplied
- simple month-of-year seasonality profile when enough years are available
- ATR-based futures-style risk context, including contract risk if `contract_multiplier` and `margin_requirement` are supplied
- pattern-reliability notes for commodity/futures-like instruments

For normal stocks, Chapter 16 is mainly a descriptive context layer. For futures or commodities, it becomes more informative if the input data includes columns such as `open_interest`, `contract_multiplier`, `margin_requirement`, `days_to_expiration`, or `roll_adjusted`. Missing futures-specific columns do not fail the run and do not force a final `Hold`.

## When To Use Chapter 17 Governance Context

Use `technical_view.chapter_17_governance_context` as the compact review packet for a human or LLM decision layer. Chapter 17 is always calculated and always report-only: `decision_policy.influences_final_action` is `false`.

Chapter 17 diagnostics include:

- `llm_decision_packet`: raw action, final action, risk level, hold reason, preferred forecast, top blockers, and missing context
- `computer_humility.decision_fragility`: how fragile the engine decision is, based on confidence, edge/error, risk, filter pressure, conflicts, and data quality
- `computer_humility.method_conflict_score`: how split the technical methods are between bullish and bearish evidence
- `computer_humility.filter_stack_review`: which governance filters changed or pressured the model signal
- `volume_context_summary`: relative-volume interpretation, with the Chapter 17 warning that volume confirms price but does not stand alone
- `mark_to_market_discipline`: observed price return and drawdown from the available run history
- `portfolio_risk_context`: portfolio inputs that are present or missing, such as weight, beta, cost basis, and covariance
- `method_performance_ledger`: validation-period signal backtest summary by horizon
- `optional_hedging_context`: hedge-review context only, not an options/futures recommendation

This section exists because a later LLM decision process needs broad context, not only the final `Buy`/`Hold`/`Sell`. It deliberately exposes uncertainty, missing inputs, and conflict instead of hiding them behind one score.

## When To Use Chapter 18 Tactical Plan

Use `decision_view.chapter_18_tactical_problem` as the first Part II layer. It converts the Part I governed action into a tactical plan with an entry policy, stop, objective, reward/risk test, and mark-to-market context.

Chapter 18 diagnostics include:

- `trade_plan`: candidate action, rule-based action, final action, stop plan, target plan, invalidation reason, reward/risk, and expected holding horizon
- `rule_gate`: hard blockers, warnings, minimum reward/risk, maximum loss budget, and profile confidence gate
- `mark_to_market`: current price, one-day return, 20-day return, 252-session drawdown, and missing position context
- `llm_review`: optional OpenAI tactical review when `--enable-llm-review` is used
- `llm_safety_gate`: deterministic post-LLM control that rejects upgrades from `Hold`, Buy/Sell flips, and any attempt to bypass hard blockers

Use `short_term` for tighter tactical swings, `intermediate` for the default 5-30 trading-day plan, and `long_term` when wider position swings are acceptable. The LLM reviewer is optional and subordinate to the rule gate.

## When To Use Chapter 19 Operational Validation

Use `operations_view.chapter_19_validation` as the end-of-run operational check. It validates whether the forecast report, charts, validation records, data-quality report, model cards, and method cards are complete enough to trust as an auditable routine.

Chapter 19 diagnostics include:

- `data_routine`: data quality, manifest, data version, target column, row count, price index, and current-price checks
- `forecast_routine`: horizon forecasts, selected models, validation metrics, candidate results, backtests, and validation predictions
- `technical_routine`: required Chapter 17 and Chapter 18 decision packets
- `artifact_routine`: plot and audit artifact existence when `--output-dir` is used
- `governance_routine`: config, model cards, feature registry, and method-card records
- `action_gate`: conditional downgrade to `Hold` only when the run is not auditable enough for a fresh `Buy` or `Sell`

This layer is not a forecasting model. It is the automated equivalent of checking that the daily charting routine, data source, records, and computer-generated artifacts are complete before acting.

## When To Use Chapter 20 Ticker Suitability

Use `selection_view.chapter_20_ticker_suitability` after the report has completed Chapter 18 and Chapter 19. It answers whether the ticker is the kind of instrument worth following for a specific profile; it does not make the final trade decision stronger.

Chapter 20 diagnostics include:

- `profile_fit`: primary profile, suitability score, classification, and Chapter 21 selection hint
- `profile_scores`: rule-based scores for `short_term_trader`, `intermediate_trader`, `long_term_investor`, `speculative_satellite`, and `index_or_diversifier`
- `component_scores`: operational readiness, validation fit, technical clarity, liquidity, movement personality, risk fit, and tactical readiness
- `instrument_habit_profile`: recent volatility, ATR percent, trend efficiency, drawdown, volume, and dollar volume
- `chapter_21_readiness`: whether the ticker should move to active review, watchlist, monitor-only, or avoid
- `chapter_22_diversification_readiness`: reminder that sector/correlation/concentration checks require a multi-ticker universe
- `llm_integration`: planned downstream LLM review note

This layer is rule-based first. A later LLM can explain and rank completed ticker reports, but it must remain subordinate to Chapter 18 tactical gates and Chapter 19 validation.

## When To Use Chapter 21 Chart Selection

Use `selection_view.chapter_21_chart_selection` after Chapter 20 has scored ticker suitability. Chapter 21 decides whether the ticker deserves active chart-book attention; it does not make a `Buy` or `Sell` stronger.

Chapter 21 diagnostics include:

- `chart_selection`: chart-book bucket, action, trade-candidate flag, active-review flag, and reason
- `priority_score`: weighted score from Chapter 20 suitability, Chapter 19 operational readiness, Chapter 18 tactical readiness, urgency, and chart artifacts
- `chart_artifact_readiness`: whether expected Plotly chart artifacts exist when an output directory was used
- `review_plan`: review cadence and required timeframes
- `chart_book_row`: compact row for future multi-ticker chart-book tables
- `chapter_22_readiness`: handoff note for diversification and concentration checks
- `llm_integration`: planned downstream LLM review note

Use the buckets this way: `trade_candidates` and `active_review` deserve current chart attention, `watchlist` deserves periodic review, `monitor_only` should not consume daily attention, and `excluded` should stay out until new evidence appears.

## When To Use Chapters 23-30 Trade/Risk Plan

Use `trade_risk_view.chapter_23_30_trade_risk_plan` after Chapter 21. This layer translates the selected ticker state into execution and risk controls; it does not override `suggested_action`.

Chapters 23-30 diagnostics include:

- `commitment`: whether the ticker is a long candidate, short candidate, risk-reduction candidate, active-review-only, watchlist-only, monitor-only, or no-new-commitment
- `chapter_23_high_risk_controls`: speculative and mania controls with position-size multiplier
- `chapter_24_probable_moves`: normal one-day, five-day, and thirty-day move bands
- `chapter_25_margin_short_policy`: margin and short-selling restrictions
- `chapter_26_position_sizing`: risk-budget sizing formula and per-100k account example when stop distance is available
- `chapter_27_stop_order_plan`: initial protective stop and progressive stop rule
- `chapter_28_top_bottom_confirmation`: confirmed pivot/basing-stop policy before moving stops
- `chapter_29_trendline_execution`: trendline entry/exit rules
- `chapter_30_support_resistance_execution`: entry, target, and invalidation zones from support/resistance
- `llm_integration`: planned second-version LLM reviewer note

This layer is broader than `Buy`/`Hold`/`Sell`: it decides whether capital can be committed, how much risk is allowed, where the stop sits, when stops move, and which chart levels govern execution.

## When To Use Chapters 31, 38, 40-42 Portfolio Capital/Risk

Use `portfolio_view.chapter_31_42_portfolio_capital_risk` after the trade/risk plan when you need to decide whether a single-ticker signal can become real portfolio exposure. This layer is report-only and does not override `suggested_action`.

It checks diversification context, portfolio balance, account-equity risk budget, staged capital application, current position weight, realized volatility, drawdown, and stop-loss exposure. If account equity, current position value, quantity, or full portfolio context are missing, it records `blocked_pending_inputs` instead of inventing a size.

## When To Use Chapters 39 and 43 Discipline Governance

Use `discipline_view.chapter_39_43_discipline_governance` at the end of the flow as the process audit. It compares Chapter 18 action, Chapter 19 validation, Chapter 21 chart selection, Chapters 23-30 trade/risk commitment, and the portfolio capital gate.

This layer warns when the forecast is directional but the execution/capital plan does not authorize new commitment, when validation blocks allocation, or when weak backtest evidence tempts a method change without proper logged evidence. It is report-only: it preserves the current action but tells you whether the plan is consistent or needs manual review.

The feature layer also includes:

- volume-confirmed breakouts
- gap up/down flags
- gap fill, breakaway, and continuation flags
- trendline/channel slope and position
- confirmed pivot support/resistance
- support/resistance touch counts and age
- breakout retest and failed breakout/breakdown flags
- rectangle/consolidation and channel-compression features

Inspired by Jansen, the engine now adds:

- factor diagnostics before trusting features
- a governed feature registry with feature family, source, lookback, availability, missingness, and factor diagnostics
- explicit relative-strength features when benchmark or sector context is supplied
- volatility/risk features such as ATR, true range, range-based volatility, drawdown, and 52-week distance
- liquidity and volume-flow features such as dollar volume, Amihud illiquidity, OBV, accumulation/distribution, and money flow
- cross-sectional rank features when a universe panel is built
- purged and embargoed time-series validation
- final untouched holdout diagnostics
- multiple-testing-aware `deflated_sharpe_ratio`
- validation-period signal backtests

## Tests

```bash
PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m pytest automated_forecasting_engine/tests
```

## Notes

This is research software, not financial advice or an execution system. Forecasts and signals must be reviewed with risk controls before any real trading decision.
