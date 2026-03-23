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
