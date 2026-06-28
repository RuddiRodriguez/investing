# Alpaca Paper 1 Bear Put Spread ETH/USD Test

Isolated experimental flow.

Purpose:

- Start from Alpaca bear put spread example.
- Test against Alpaca paper account 1.
- Try ETH/USD first.
- Do not submit orders.
- Write clear JSON report.

Important result:

- ETH/USD on Alpaca is crypto spot.
- Bear put spread needs option contracts.
- Alpaca paper shows ETH/USD as tradable crypto, but no option contracts.
- This experiment should block before order planning for ETH/USD.

Run:

```bash
python experiments/alpaca_paper1_bear_put_eth_test/bear_put_spread_dry_run.py \
  --env-file /Users/ruddigarcia/Projects/invest/.env \
  --underlying ETH/USD
```

Output:

```text
experiments/alpaca_paper1_bear_put_eth_test/reports/latest_report.json
```

Safety:

- Read-only broker calls.
- No order submission code path.
- No market orders.
- No changes to existing agents.
- No shared module imports.
