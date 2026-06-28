from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


def load_env_file() -> None:
    for path in _env_search_paths():
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return


class AlpacaPaperBroker:
    def __init__(self, *, base_url: str | None = None, key_id: str | None = None, secret_key: str | None = None) -> None:
        load_env_file()
        self.base_url = _normalize_trading_base_url(
            base_url
            or os.getenv("ALPACA_TRADING_BASE_URL")
            or os.getenv("APCA_API_BASE_URL")
            or os.getenv("ALPACA_API_BASE_URL")
            or "https://paper-api.alpaca.markets"
        )
        self.data_base_url = (os.getenv("ALPACA_DATA_BASE_URL") or "https://data.alpaca.markets").rstrip("/")
        self.key_id = key_id or os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
        self.secret_key = secret_key or os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
        if not self.key_id or not self.secret_key:
            raise RuntimeError("Alpaca paper broker requires ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY.")

    def account(self) -> dict[str, Any]:
        return self._request("GET", "/v2/account")

    def clock(self) -> dict[str, Any]:
        return self._request("GET", "/v2/clock")

    def position(self, symbol: str) -> dict[str, Any] | None:
        attempted: list[str] = []
        for candidate in _position_symbol_candidates(symbol):
            attempted.append(candidate)
            position = self._position_once(candidate)
            if position is not None:
                return position
        return None

    def _position_once(self, symbol: str) -> dict[str, Any] | None:
        try:
            return self._request("GET", f"/v2/positions/{quote(symbol, safe='')}")
        except RuntimeError as exc:
            if "404" in str(exc):
                return None
            raise

    def positions(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/v2/positions")
        return payload if isinstance(payload, list) else []

    def orders(
        self,
        *,
        status: str = "open",
        limit: int = 50,
        direction: str | None = None,
        symbols: list[str] | None = None,
        after: str | None = None,
        until: str | None = None,
        nested: bool | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"status": status, "limit": int(limit)}
        if direction:
            params["direction"] = direction
        if symbols:
            params["symbols"] = ",".join(sorted({str(symbol).upper() for symbol in symbols if str(symbol).strip()}))
        if after:
            params["after"] = after
        if until:
            params["until"] = until
        if nested is not None:
            params["nested"] = str(bool(nested)).lower()
        payload = self._request("GET", "/v2/orders?" + urlencode(params))
        return payload if isinstance(payload, list) else []

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v2/orders/{quote(order_id, safe='')}")

    def option_contracts(
        self,
        *,
        underlying_symbols: str,
        status: str = "active",
        expiration_date_gte: str | None = None,
        expiration_date_lte: str | None = None,
        option_type: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"underlying_symbols": underlying_symbols.upper(), "status": status, "limit": int(limit)}
        if expiration_date_gte:
            params["expiration_date_gte"] = expiration_date_gte
        if expiration_date_lte:
            params["expiration_date_lte"] = expiration_date_lte
        if option_type:
            params["type"] = option_type.lower()
        rows: list[dict[str, Any]] = []
        next_page_token: str | None = None
        while True:
            page_params = dict(params)
            if next_page_token:
                page_params["page_token"] = next_page_token
            payload = self._request("GET", "/v2/options/contracts?" + urlencode(page_params))
            rows.extend(payload.get("option_contracts") or payload.get("contracts") or [])
            next_page_token = payload.get("next_page_token")
            if not next_page_token:
                break
        return rows

    def option_snapshots(self, symbols: list[str], *, feed: str | None = None) -> dict[str, Any]:
        if not symbols:
            return {}
        output: dict[str, Any] = {}
        for chunk in _chunks(list(dict.fromkeys(symbols)), 100):
            params = {"symbols": ",".join(chunk)}
            if feed:
                params["feed"] = feed
            payload = self._data_request("GET", "/v1beta1/options/snapshots?" + urlencode(params))
            snapshots = payload.get("snapshots", payload)
            if isinstance(snapshots, dict):
                output.update(snapshots)
        return output

    def crypto_snapshots(self, symbols: list[str], *, location: str | None = None) -> dict[str, Any]:
        if not symbols:
            return {}
        params = {"symbols": ",".join(symbols)}
        market = location or os.getenv("ALPACA_CRYPTO_LOCATION", "us")
        payload = self._data_request("GET", f"/v1beta3/crypto/{quote(market, safe='')}/snapshots?" + urlencode(params))
        snapshots = payload.get("snapshots", payload)
        return snapshots if isinstance(snapshots, dict) else {}

    def stock_bars(
        self,
        symbol: str,
        *,
        start: str,
        end: str,
        timeframe: str = "1Min",
        feed: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "start": start,
            "end": end,
            "timeframe": timeframe,
            "limit": int(limit),
        }
        selected_feed = feed or os.getenv("ALPACA_DATA_FEED")
        if selected_feed:
            params["feed"] = selected_feed
        rows: list[dict[str, Any]] = []
        next_page_token: str | None = None
        while True:
            page_params = dict(params)
            if next_page_token:
                page_params["page_token"] = next_page_token
            payload = self._data_request("GET", f"/v2/stocks/{quote(symbol.upper(), safe='')}/bars?" + urlencode(page_params))
            page_rows = payload.get("bars", []) if isinstance(payload, dict) else []
            if isinstance(page_rows, list):
                rows.extend(page_rows)
            next_page_token = payload.get("next_page_token") if isinstance(payload, dict) else None
            if not next_page_token:
                break
        return rows

    def submit_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str = "market",
        notional: float | None = None,
        qty: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        trail_price: float | None = None,
        trail_percent: float | None = None,
        client_order_id: str | None = None,
        time_in_force: str = "gtc",
    ) -> dict[str, Any]:
        if side not in {"buy", "sell"}:
            raise ValueError("side must be buy or sell.")
        if notional is None and qty is None:
            raise ValueError("Pass notional or qty.")
        body: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if notional is not None:
            body["notional"] = str(round(float(notional), 2))
        if qty is not None:
            body["qty"] = str(float(qty))
        if limit_price is not None:
            body["limit_price"] = str(round(float(limit_price), 2))
        if stop_price is not None:
            body["stop_price"] = str(round(float(stop_price), 2))
        if trail_price is not None:
            body["trail_price"] = str(round(float(trail_price), 2))
        if trail_percent is not None:
            body["trail_percent"] = str(round(float(trail_percent), 4))
        if client_order_id:
            body["client_order_id"] = client_order_id[:48]
        return self._request("POST", "/v2/orders", body)

    def submit_market_order(
        self,
        *,
        symbol: str,
        side: str,
        notional: float | None = None,
        qty: float | None = None,
        client_order_id: str | None = None,
        time_in_force: str = "gtc",
    ) -> dict[str, Any]:
        return self.submit_order(
            symbol=symbol,
            side=side,
            order_type="market",
            notional=notional,
            qty=qty,
            client_order_id=client_order_id,
            time_in_force=time_in_force,
        )

    def submit_multileg_option_order(
        self,
        *,
        legs: list[dict[str, Any]],
        order_type: str = "limit",
        qty: int = 1,
        limit_price: float | None = None,
        time_in_force: str = "day",
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        if order_type not in {"limit", "market"}:
            raise ValueError("Multi-leg option orders support limit or market order types.")
        if int(qty) <= 0:
            raise ValueError("Multi-leg option orders require positive qty.")
        if not legs:
            raise ValueError("Multi-leg option orders require at least one leg.")
        body: dict[str, Any] = {
            "type": order_type,
            "time_in_force": time_in_force,
            "order_class": "mleg",
            "qty": str(int(qty)),
            "legs": [
                {
                    "side": str(leg["side"]),
                    "position_intent": str(leg.get("position_intent") or ("buy_to_open" if leg["side"] == "buy" else "sell_to_close")),
                    "symbol": str(leg["symbol"]),
                    "ratio_qty": str(int(leg.get("ratio_qty") or 1)),
                }
                for leg in legs
            ],
        }
        if limit_price is not None:
            body["limit_price"] = str(round(float(limit_price), 2))
        if client_order_id:
            body["client_order_id"] = client_order_id[:48]
        return self._request("POST", "/v2/orders", body)

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        return self._json_request(method, f"{self.base_url}{path}", body)

    def _data_request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        return self._json_request(method, f"{self.data_base_url}{path}", body)

    def _json_request(self, method: str, url: str, body: dict[str, Any] | None = None) -> Any:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {
            "APCA-API-KEY-ID": self.key_id or "",
            "APCA-API-SECRET-KEY": self.secret_key or "",
            "Content-Type": "application/json",
        }
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=30) as response:
                text = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8")
            except Exception:
                error_body = ""
            detail = f"HTTP Error {exc.code}: {exc.reason}"
            if error_body:
                detail = f"{detail}: {error_body}"
            raise RuntimeError(f"Alpaca broker request failed: {method} {url}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"Alpaca broker request failed: {method} {url}: {exc}") from exc
        return json.loads(text) if text else {}


def _env_search_paths() -> list[Path]:
    cwd = Path.cwd()
    return [cwd / ".env", *[parent / ".env" for parent in cwd.parents[:4]]]


def _normalize_trading_base_url(value: str) -> str:
    normalized = str(value or "").rstrip("/")
    if normalized.endswith("/v2"):
        normalized = normalized[: -len("/v2")]
    return normalized


def _position_symbol_candidates(symbol: str) -> list[str]:
    normalized = symbol.strip().upper()
    candidates = [normalized]
    if "/" in normalized:
        candidates.append(normalized.replace("/", ""))
    elif normalized.endswith("USD") and len(normalized) > 3:
        candidates.append(f"{normalized[:-3]}/USD")
    return list(dict.fromkeys(candidates))


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]
