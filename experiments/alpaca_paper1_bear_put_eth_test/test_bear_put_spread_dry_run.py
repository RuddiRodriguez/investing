from bear_put_spread_dry_run import build_report


class FakeClient:
    def account(self):
        return {
            "status": "ACTIVE",
            "trading_blocked": False,
            "account_blocked": False,
            "buying_power": "10000",
            "options_buying_power": "10000",
            "equity": "10000",
        }

    def clock(self):
        return {"is_open": True, "timestamp": "2026-06-26T12:00:00Z", "next_open": None, "next_close": None}

    def asset(self, symbol):
        return {
            "symbol": symbol,
            "name": "Ethereum / US Dollar",
            "class": "crypto",
            "status": "active",
            "tradable": True,
            "fractionable": True,
            "attributes": [],
        }

    def option_contracts(self, underlying, *, max_contracts):
        return []


def test_eth_usd_crypto_blocks_bear_put_spread():
    report = build_report(FakeClient(), "ETH/USD", max_contracts=10)

    assert report["dry_run"] is True
    assert report["submit_orders"] is False
    assert "underlying_is_crypto_spot_not_option_underlying" in report["execution_blocks"]
    assert "no_option_contracts_for_underlying" in report["execution_blocks"]
    assert report["candidate"]["status"] == "blocked"
