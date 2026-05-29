# ETF Investing Helper

`ETF Investing Helper` is a small Streamlit app for beginners who want a simple, rules-based way to review ETFs.

The app is designed for people who want a `3-12 month` investing process instead of daily trading. It keeps the method intentionally simple:

- check which ETFs have been strongest in recent months
- keep only ETFs that still have a healthy long-term direction
- hold the top `N` ETFs
- move to cash when no ETF passes the filter

## What The App Does

- downloads historical ETF prices from Yahoo Finance or reads a user-supplied CSV
- runs a simple backtest with monthly or quarterly rebalancing
- shows portfolio growth, drops from peaks, current allocation, and past portfolio changes
- can optionally call OpenAI to explain the latest signals in simpler language with recent market context
- exposes parameters so you can compare different horizons and rebalance schedules

## Project Structure

- `/Users/ruddigarcia/Projects/invest/app.py`: Streamlit UI
- `/Users/ruddigarcia/Projects/invest/src/strategy.py`: core strategy and backtest logic
- `/Users/ruddigarcia/Projects/invest/src/data.py`: market data helpers
- `/Users/ruddigarcia/Projects/invest/tests/test_strategy.py`: local tests for pure strategy behavior

## Run Locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app.py
```

## Download Yahoo Finance Prices

Use the simple script to download historical prices for any Yahoo Finance ticker symbol:

```bash
python scripts/download_yahoo_prices.py
```

Edit these values at the top of `scripts/download_yahoo_prices.py`:

```python
TICKER = "AAPL"
START_DATE = "2020-01-01"
END_DATE = None
OUTPUT_FILE = "stock_prices.csv"
```

Then plot the downloaded close price:

```bash
python scripts/plot_stock_prices.py
```

Test the simplest direction prediction method in a graph:

```bash
python scripts/direction/method_1_graph_test.py
```

Test the momentum direction method in a graph:

```bash
python scripts/direction/method_2_graph_test.py
```

Test the moving average crossover method in a graph:

```bash
python scripts/direction/method_3_graph_test.py
```

Test the indicator voting method in a graph:

```bash
python scripts/direction/method_4_graph_test.py
```

Run the improved logistic regression direction method:

```bash
python scripts/direction/method_5_logistic_regression.py
```

Test the improved logistic regression method in a graph:

```bash
python scripts/direction/method_5_logistic_regression_graph_test.py
```

Try method 5 with LightGBM:

```bash
python scripts/direction/method_5_lgbm.py
```

Test the LightGBM method in a graph:

```bash
python scripts/direction/method_5_lgbm_graph_test.py
```

## Assess Risk

Run the basic risk methods:

```bash
python scripts/risk/method_1_volatility.py
python scripts/risk/method_2_drawdown.py
python scripts/risk/method_3_var.py
python scripts/risk/method_4_expected_shortfall.py
python scripts/risk/method_5_signal_risk.py
```

Validate whether the risk assessment is doing a decent job:

```bash
python scripts/risk/validate_risk_assessment.py
```

The validation checks whether higher estimated risk buckets lead to worse future volatility/losses, and whether 95% VaR is breached near 5% of the time.

## Predict Outperformance

Run the advanced LightGBM outperformance model:

```bash
python scripts/outperformance/method_1_lgbm_outperformance.py
```

It uses `stock_prices.csv` as the stock and downloads `SPY` as the benchmark into `benchmark_prices.csv`.

## Advanced Live Stock Pipeline

The advanced implementation is standalone in `/Users/ruddigarcia/Projects/invest/scripts/advanced_pipeline`.
It does not import the Streamlit app or the beginner strategy modules.

Run it with live Yahoo Finance data:

```bash
python -m scripts.advanced_pipeline.cli --tickers AAPL,MSFT,NVDA,AMZN --benchmark SPY
```

The pipeline downloads live adjusted prices, fundamentals, and recent headline data, then caches identical requests under:

```text
/Users/ruddigarcia/Projects/invest/scripts/advanced_pipeline/.cache
```

Use `--force-refresh` to bypass the cache. The output includes each ticker's decision, confidence, expected excess return, expected volatility, risk score, position size, main drivers, and main risks.

## Deploy On Streamlit Community Cloud

Deploy directly from the GitHub repository:

- Repository: `RuddiRodriguez/investing`
- Branch: `main`
- Main file path: `app.py`

If you enable the optional AI explanation, add this secret in the Streamlit app settings:

```toml
OPENAI_API_KEY="your_api_key_here"
```

The repository includes:

- `/Users/ruddigarcia/Projects/invest/runtime.txt` to pin the Python version used by Streamlit Cloud
- `/Users/ruddigarcia/Projects/invest/.streamlit/config.toml` for Streamlit runtime defaults

## Optional OpenAI Interpretation

If you want the app to produce a narrative interpretation of the latest allocation, set:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Then enable `OpenAI interpretation` in the sidebar.

The app uses the OpenAI `Responses API`. It can also optionally use `web_search` so the explanation reflects current market conditions and recent news rather than only the local backtest.

Important constraints:

- treat the AI output as a helper, not an automatic decision maker
- verify any news or macro claims before using them
- a `Buy`, `Hold`, `Reduce`, or `Stay In Cash` label is only a guide on top of the model

## CSV Format

If you do not want to download from Yahoo Finance, upload a CSV where:

- the first column is a date
- the remaining columns are ETF price series
- column names are the ticker labels you want to display

Example:

```csv
Date,VWCE.DE,EUNL.DE,SPYI.DE,VAGF.DE
2024-01-02,98.1,76.5,120.2,24.1
2024-01-03,98.6,76.9,121.0,24.0
```

## Notes

- This app is educational and research-oriented.
- It does not include taxes, spreads, or broker-specific execution details.
- A simple model is usually more robust than a complicated one you cannot explain.
