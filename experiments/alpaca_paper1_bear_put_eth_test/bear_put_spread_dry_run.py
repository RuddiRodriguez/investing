from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


PAPER_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"


def main() -> None:
    args = build_parser().parse_args()
    load_env_file(Path(args.env_file).expanduser())
    client = AlpacaReadOnlyClient()
    report = build_report(client, args.underlying, max_contracts=int(args.max_contracts))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Alpaca paper bear put spread dry-run.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--underlying", default="ETH/USD")
    parser.add_argument("--max-contracts", type=int, default=250)
    parser.add_argument("--output", default="experiments/alpaca_paper1_bear_put_eth_test/reports/latest_report.json")
    return parser


def build_report(client: "AlpacaReadOnlyClient", underlying: str, *, max_contracts: int) -> dict[str, Any]:
    now = datetime.now(UTC)
    account = safe_call(client.account)
    clock = safe_call(client.clock)
    asset = safe_call(lambda: client.asset(underlying))
    contracts = safe_call(lambda: client.option_contracts(underlying, max_contracts=max_contracts))

    blocks: list[str] = []
    if account["ok"] and str(account["value"].get("status", "")).upper() != "ACTIVE":
        blocks.append("account_not_active")
    if clock["ok"] and not bool(clock["value"].get("is_open")):
        blocks.append("market_closed")
    if not asset["ok"]:
        blocks.append("underlying_asset_not_found")
    elif str(asset["value"].get("class") or asset["value"].get("asset_class") or "").lower() == "crypto":
        blocks.append("underlying_is_crypto_spot_not_option_underlying")
    if not contracts["ok"]:
        blocks.append("option_contract_lookup_failed")
        option_contracts: list[dict[str, Any]] = []
    else:
        option_contracts = contracts["value"]
        if not option_contracts:
            blocks.append("no_option_contracts_for_underlying")

    put_contracts = [row for row in option_contracts if str(row.get("type", "")).lower() == "put"]
    candidate = build_bear_put_candidate(client, put_contracts)
    if candidate.get("blocks"):
        blocks.extend(candidate["blocks"])

    return {
        "generated_at": now.isoformat(),
        "mode": "isolated_alpaca_paper1_bear_put_spread_dry_run",
        "underlying": underlying,
        "dry_run": True,
        "submit_orders": False,
        "source_notebook": "https://github.com/alpacahq/alpaca-py/blob/master/examples/options/options-bear-put-spread.ipynb",
        "policy": {
            "read_only": True,
            "no_order_submission_code_path": True,
            "no_market_orders": True,
            "eth_usd_must_block_without_option_chain": True,
        },
        "account": compact_account(account),
        "clock": compact_clock(clock),
        "asset": compact_asset(asset),
        "option_contract_count": len(option_contracts),
        "put_contract_count": len(put_contracts),
        "candidate": candidate,
        "execution_blocks": unique(blocks),
    }


def build_bear_put_candidate(client: "AlpacaReadOnlyClient", puts: list[dict[str, Any]]) -> dict[str, Any]:
    if len(puts) < 2:
        return {"status": "blocked", "blocks": ["not_enough_put_contracts_for_bear_put_spread"]}

    by_expiry: dict[str, list[dict[str, Any]]] = {}
    for row in puts:
        if row.get("tradable") is False:
            continue
        expiry = str(row.get("expiration_date") or "")
        if expiry:
            by_expiry.setdefault(expiry, []).append(row)

    today = date.today()
    for expiry in sorted(by_expiry):
        rows = sorted(by_expiry[expiry], key=lambda row: float(row.get("strike_price") or 0.0))
        if len(rows) < 2:
            continue
        days_to_expiry = (date.fromisoformat(expiry) - today).days
        if days_to_expiry < 1:
            continue
        lower_put = rows[0]
        higher_put = rows[-1]
        symbols = [str(lower_put["symbol"]), str(higher_put["symbol"])]
        snapshots = safe_call(lambda: client.option_snapshots(symbols))
        quotes = option_quotes(snapshots["value"] if snapshots["ok"] else {})
        short_quote = quotes.get(symbols[0], {})
        long_quote = quotes.get(symbols[1], {})
        short_mid = mid_price(short_quote)
        long_mid = mid_price(long_quote)
        if short_mid is None or long_mid is None:
            return {
                "status": "blocked",
                "blocks": ["missing_option_quote_for_candidate"],
                "candidate_symbols": symbols,
            }
        net_debit = max(0.0, long_mid - short_mid)
        width = float(higher_put["strike_price"]) - float(lower_put["strike_price"])
        return {
            "status": "planned_read_only",
            "strategy": "bear_put_spread",
            "legs": [
                leg_summary("sell", lower_put, short_quote, short_mid),
                leg_summary("buy", higher_put, long_quote, long_mid),
            ],
            "estimated_net_debit": round(net_debit, 2),
            "estimated_max_loss": round(net_debit * 100.0, 2),
            "estimated_max_profit": round(max(0.0, width - net_debit) * 100.0, 2),
            "order_type_if_enabled_later": "limit",
            "blocks": [],
        }

    return {"status": "blocked", "blocks": ["no_valid_expiry_pair_for_bear_put_spread"]}


class AlpacaReadOnlyClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("ALPACA_TRADING_BASE_URL") or os.getenv("APCA_API_BASE_URL") or PAPER_BASE_URL
        self.data_base_url = os.getenv("ALPACA_DATA_BASE_URL") or DATA_BASE_URL
        self.key_id = os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
        self.secret_key = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
        if not self.key_id or not self.secret_key:
            raise SystemExit("Missing Alpaca paper keys.")
        if "paper-api" not in self.base_url:
            raise SystemExit(f"Refusing non-paper Alpaca endpoint: {self.base_url}")

    def account(self) -> dict[str, Any]:
        return self._request(self.base_url, "GET", "/v2/account")

    def clock(self) -> dict[str, Any]:
        return self._request(self.base_url, "GET", "/v2/clock")

    def asset(self, symbol: str) -> dict[str, Any]:
        return self._request(self.base_url, "GET", f"/v2/assets/{quote(symbol, safe='')}")

    def option_contracts(self, underlying: str, *, max_contracts: int) -> list[dict[str, Any]]:
        params = {
            "underlying_symbols": underlying.upper(),
            "status": "active",
            "expiration_date_gte": date.today().isoformat(),
            "expiration_date_lte": (date.today() + timedelta(days=60)).isoformat(),
            "limit": int(max_contracts),
        }
        payload = self._request(self.base_url, "GET", "/v2/options/contracts?" + urlencode(params))
        rows = payload.get("option_contracts") or payload.get("contracts") or []
        return rows if isinstance(rows, list) else []

    def option_snapshots(self, symbols: list[str]) -> dict[str, Any]:
        payload = self._request(self.data_base_url, "GET", "/v1beta1/options/snapshots?" + urlencode({"symbols": ",".join(symbols)}))
        snapshots = payload.get("snapshots", payload)
        return snapshots if isinstance(snapshots, dict) else {}

    def _request(self, base_url: str, method: str, path: str) -> Any:
        req = Request(
            base_url.rstrip("/") + path,
            method=method,
            headers={
                "APCA-API-KEY-ID": self.key_id,
                "APCA-API-SECRET-KEY": self.secret_key,
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(req, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} failed: {exc.code} {detail}") from exc
        return json.loads(raw) if raw else {}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def safe_call(fn: Any) -> dict[str, Any]:
    try:
        return {"ok": True, "value": fn()}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def compact_account(result: dict[str, Any]) -> dict[str, Any]:
    if not result["ok"]:
        return result
    value = result["value"]
    return {
        "ok": True,
        "status": value.get("status"),
        "trading_blocked": value.get("trading_blocked"),
        "account_blocked": value.get("account_blocked"),
        "buying_power": value.get("buying_power"),
        "options_buying_power": value.get("options_buying_power"),
        "equity": value.get("equity"),
    }


def compact_clock(result: dict[str, Any]) -> dict[str, Any]:
    if not result["ok"]:
        return result
    value = result["value"]
    return {key: value.get(key) for key in ("is_open", "timestamp", "next_open", "next_close")}


def compact_asset(result: dict[str, Any]) -> dict[str, Any]:
    if not result["ok"]:
        return result
    value = result["value"]
    return {
        "ok": True,
        "symbol": value.get("symbol"),
        "name": value.get("name"),
        "class": value.get("class") or value.get("asset_class"),
        "status": value.get("status"),
        "tradable": value.get("tradable"),
        "fractionable": value.get("fractionable"),
        "attributes": value.get("attributes"),
    }


def option_quotes(snapshots: dict[str, Any]) -> dict[str, dict[str, Any]]:
    quotes: dict[str, dict[str, Any]] = {}
    for symbol, snapshot in snapshots.items():
        quote = snapshot.get("latestQuote") or snapshot.get("latest_quote") or {}
        quotes[symbol] = quote if isinstance(quote, dict) else {}
    return quotes


def mid_price(quote: dict[str, Any]) -> float | None:
    bid = number(quote.get("bp") or quote.get("bid_price"))
    ask = number(quote.get("ap") or quote.get("ask_price"))
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return None
    return round((bid + ask) / 2.0, 2)


def leg_summary(side: str, contract: dict[str, Any], quote: dict[str, Any], mid: float) -> dict[str, Any]:
    return {
        "side": side,
        "symbol": contract.get("symbol"),
        "type": contract.get("type"),
        "strike": contract.get("strike_price"),
        "expiration": contract.get("expiration_date"),
        "bid": number(quote.get("bp") or quote.get("bid_price")),
        "ask": number(quote.get("ap") or quote.get("ask_price")),
        "mid": mid,
    }


def number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "underlying": report["underlying"],
        "dry_run": report["dry_run"],
        "submit_orders": report["submit_orders"],
        "account_status": report["account"].get("status"),
        "market_open": report["clock"].get("is_open"),
        "asset_class": report["asset"].get("class"),
        "option_contract_count": report["option_contract_count"],
        "candidate_status": report["candidate"].get("status"),
        "execution_blocks": report["execution_blocks"],
    }


if __name__ == "__main__":
    main()
