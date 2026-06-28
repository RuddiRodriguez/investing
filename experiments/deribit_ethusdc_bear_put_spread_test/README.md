# Deribit ETH_USDC Bear Put Spread Notebook Test

Isolated experimental flow.

Purpose:

- Test the Alpaca bear put spread notebook idea on Deribit.
- Use ETH_USDC as the spot context.
- Use Deribit ETH option instruments for the put spread.
- Default to dry-run.
- Allow live only with explicit flags.
- Cap live risk at 5 EUR.
- Write clear JSON report.

Important difference:

- On Deribit, `ETH_USDC` is spot.
- Crypto options are listed by currency, for example `ETH`.
- Option instruments look like `ETH-27JUN26-3000-P`.

Run:

```bash
python experiments/deribit_ethusdc_bear_put_spread_test/deribit_bear_put_spread_dry_run.py
```

Continuous Deribit testnet trader:

```bash
python experiments/deribit_ethusdc_bear_put_spread_test/deribit_bear_put_spread_dry_run.py \
  --account-mode testnet \
  --execute-testnet-orders \
  --continuous \
  --check-interval-seconds 60
```

Continuous BTC_USDC testnet trader with loose filters:

```bash
python experiments/deribit_ethusdc_bear_put_spread_test/deribit_bear_put_spread_dry_run.py \
  --account-mode testnet \
  --spot-instrument BTC_USDC \
  --option-currency BTC \
  --execute-testnet-orders \
  --continuous \
  --check-interval-seconds 60 \
  --loose-filters \
  --strike-range 0.50 \
  --oi-threshold 0 \
  --iv-min 0 \
  --iv-max 5 \
  --short-delta-min -1 \
  --short-delta-max 0 \
  --long-delta-min -1 \
  --long-delta-max 0 \
  --vega-min -999 \
  --vega-max 999
```

Live dry-run:

```bash
python experiments/deribit_ethusdc_bear_put_spread_test/deribit_bear_put_spread_dry_run.py \
  --account-mode live \
  --max-eur 5
```

Live order, only if the notebook filters find a valid spread:

```bash
python experiments/deribit_ethusdc_bear_put_spread_test/deribit_bear_put_spread_dry_run.py \
  --account-mode live \
  --max-eur 5 \
  --execute-live-orders \
  --confirm-live-deribit-options-orders \
  --i-understand-this-is-real-money
```

Output:

```text
experiments/deribit_ethusdc_bear_put_spread_test/reports/latest_report.json
```

Safety:

- Uses existing DeribitOptionsBroker.
- Testnet has no debit cap.
- Live submit requires three flags.
- Blocks when estimated risk is over 5 EUR.
- Does not change existing agents.
