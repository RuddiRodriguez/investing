# Advanced Live Stock Pipeline

This folder is a standalone implementation of the advanced pipeline. It does not import the app or `src` strategy code.

## Run

```bash
python -m scripts.advanced_pipeline.cli --tickers AAPL,MSFT,NVDA,AMZN --benchmark SPY
```

The pipeline downloads live Yahoo Finance data, then caches identical requests in:

```text
scripts/advanced_pipeline/.cache
```

Use `--force-refresh` to bypass the cache.

## What It Builds

- live adjusted-close price panel
- cached Yahoo fundamentals
- cached recent Yahoo headline signals
- multi-horizon forward excess-return targets
- tabular alpha model
- technical, fundamental, news, graph, and regime experts
- risk and uncertainty gate
- buy / hold / sell decision table
- constrained portfolio weights with cash fallback
