from argparse import Namespace

from deribit_bear_put_spread_dry_run import run_notebook_flow


class FakeBroker:
    base_url = "https://test.deribit.com/api/v2"

    def order_book(self, instrument_name, *, depth):
        if instrument_name == "ETH_USDC":
            return {"instrument_name": instrument_name, "mark_price": 2500.0, "best_bid_price": 2499.0, "best_ask_price": 2501.0}
        if instrument_name.endswith("2400-P"):
            return {
                "instrument_name": instrument_name,
                "best_bid_price": 0.03,
                "best_ask_price": 0.04,
                "stats": {"open_interest": 10},
            }
        return {
            "instrument_name": instrument_name,
            "best_bid_price": 0.031,
            "best_ask_price": 0.041,
            "stats": {"open_interest": 10},
        }

    def instruments(self, *, currency, kind, expired):
        return [
            {
                "instrument_name": "ETH-31JUL26-2400-P",
                "option_type": "put",
                "strike": 2400,
                "expiration_timestamp": 1785542400000,
                "base_currency": "ETH",
                "min_trade_amount": 1,
            },
            {
                "instrument_name": "ETH-31JUL26-2600-P",
                "option_type": "put",
                "strike": 2600,
                "expiration_timestamp": 1785542400000,
                "base_currency": "ETH",
                "min_trade_amount": 1,
            },
        ]


def test_notebook_flow_stays_dry_run_and_blocks_without_matching_notebook_filters():
    args = Namespace(
        spot_instrument="ETH_USDC",
        option_currency="ETH",
        max_eur=5.0,
        eur_usdc_rate=1.08,
        risk_free_rate=0.05,
        target_profit_percentage=0.4,
        delta_stop_loss=-0.50,
        iv_stop_loss=0.40,
        min_expiration_days=21,
        max_expiration_days=60,
        strike_range=0.10,
        oi_threshold=1.0,
        account_mode="testnet",
        rolling=True,
        loose_filters=False,
        iv_min=None,
        iv_max=None,
        short_delta_min=None,
        short_delta_max=None,
        long_delta_min=None,
        long_delta_max=None,
        vega_min=None,
        vega_max=None,
        chain_refresh_seconds=900,
        trade_cooldown_seconds=1800,
        max_rolls_per_hour=1,
        stop_after_one_complete_cycle=False,
    )

    report = run_notebook_flow(args=args, broker=FakeBroker())

    assert report["dry_run"] is True
    assert report["submit_orders"] is False
    assert report["spot_context_instrument"] == "ETH_USDC"
    assert report["option_currency"] == "ETH"
    assert report["candidate"]["status"] in {"planned", "blocked"}
