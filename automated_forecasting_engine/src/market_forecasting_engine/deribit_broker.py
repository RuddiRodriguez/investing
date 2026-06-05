from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
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


class DeribitTestnetBroker:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> None:
        load_env_file()
        testnet = str(os.getenv("DERIBIT_TESTNET", "true")).lower() in {"1", "true", "yes"}
        default_url = "https://test.deribit.com/api/v2" if testnet else "https://www.deribit.com/api/v2"
        self.base_url = (base_url or os.getenv("DERIBIT_BASE_URL") or default_url).rstrip("/")
        self.client_id = client_id or os.getenv("DERIBIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("DERIBIT_CLIENT_SECRET")
        self._access_token: str | None = None
        if "test.deribit.com" not in self.base_url:
            raise RuntimeError("DeribitTestnetBroker is locked to testnet. Use a separate live broker implementation for production live trading.")

    def public_get(self, method: str, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", f"/public/{method}", params=params, token=None)

    def private_get(self, method: str, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", f"/private/{method}", params=params, token=self.access_token())

    def private_post(self, method: str, params: dict[str, Any] | None = None) -> Any:
        return self._request("POST", f"/private/{method}", params=params, token=self.access_token())

    def access_token(self) -> str:
        if self._access_token:
            return self._access_token
        if not self.client_id or not self.client_secret:
            raise RuntimeError("Deribit testnet auth requires DERIBIT_CLIENT_ID and DERIBIT_CLIENT_SECRET.")
        result = self.public_get(
            "auth",
            {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
        )
        token = result.get("access_token") if isinstance(result, dict) else None
        if not token:
            raise RuntimeError("Deribit auth did not return an access token.")
        self._access_token = str(token)
        return self._access_token

    def instruments(self, *, currency: str = "ETH", kind: str = "option", expired: bool = False) -> list[dict[str, Any]]:
        result = self.public_get("get_instruments", {"currency": currency.upper(), "kind": kind, "expired": str(expired).lower()})
        return result if isinstance(result, list) else []

    def ticker(self, instrument_name: str) -> dict[str, Any]:
        result = self.public_get("ticker", {"instrument_name": instrument_name})
        return result if isinstance(result, dict) else {}

    def order_book(self, instrument_name: str, *, depth: int = 10) -> dict[str, Any]:
        result = self.public_get("get_order_book", {"instrument_name": instrument_name, "depth": int(depth)})
        return result if isinstance(result, dict) else {}

    def account_summary(self, *, currency: str = "ETH") -> dict[str, Any]:
        result = self.private_get("get_account_summary", {"currency": currency.upper(), "extended": "true"})
        return result if isinstance(result, dict) else {}

    def open_orders(self, *, currency: str = "ETH", kind: str = "option") -> list[dict[str, Any]]:
        result = self.private_get("get_open_orders_by_currency", {"currency": currency.upper(), "kind": kind})
        return result if isinstance(result, list) else []

    def positions(self, *, currency: str = "ETH", kind: str = "option") -> list[dict[str, Any]]:
        result = self.private_get("get_positions", {"currency": currency.upper(), "kind": kind})
        return result if isinstance(result, list) else []

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        result = self.private_get("cancel", {"order_id": order_id})
        return result if isinstance(result, dict) else {"result": result}

    def buy_limit(
        self,
        *,
        instrument_name: str,
        amount: float,
        price: float,
        label: str | None = None,
        post_only: bool = False,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        return self._submit_order(
            side="buy",
            instrument_name=instrument_name,
            amount=amount,
            price=price,
            label=label,
            post_only=post_only,
            reduce_only=reduce_only,
        )

    def sell_limit(
        self,
        *,
        instrument_name: str,
        amount: float,
        price: float,
        label: str | None = None,
        post_only: bool = False,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        return self._submit_order(
            side="sell",
            instrument_name=instrument_name,
            amount=amount,
            price=price,
            label=label,
            post_only=post_only,
            reduce_only=reduce_only,
        )

    def _submit_order(
        self,
        *,
        side: str,
        instrument_name: str,
        amount: float,
        price: float,
        label: str | None,
        post_only: bool,
        reduce_only: bool,
    ) -> dict[str, Any]:
        if amount <= 0:
            raise ValueError("Deribit option orders require positive amount.")
        if price <= 0:
            raise ValueError("Deribit option limit orders require positive price.")
        params: dict[str, Any] = {
            "instrument_name": instrument_name,
            "amount": amount,
            "type": "limit",
            "price": price,
            "time_in_force": "good_til_cancelled",
            "post_only": str(bool(post_only)).lower(),
            "reduce_only": str(bool(reduce_only)).lower(),
        }
        if label:
            params["label"] = label[:64]
        result = self.private_get(side, params)
        return result if isinstance(result, dict) else {"result": result}

    def _request(self, method: str, path: str, *, params: dict[str, Any] | None = None, token: str | None = None) -> Any:
        query = "" if not params else "?" + urlencode(params)
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(f"{self.base_url}{path}{query}", headers=headers, method=method)
        try:
            with urlopen(request, timeout=30) as response:
                text = response.read().decode("utf-8")
        except HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8")
            except Exception:
                body = ""
            detail = f"HTTP Error {exc.code}: {exc.reason}"
            if body:
                detail = f"{detail}: {body}"
            raise RuntimeError(f"Deribit request failed: {method} {path}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"Deribit request failed: {method} {path}: {exc}") from exc
        payload = json.loads(text) if text else {}
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(f"Deribit API error: {payload['error']}")
        return payload.get("result") if isinstance(payload, dict) and "result" in payload else payload


class DeribitReadOnlyBroker:
    """Read-only Deribit account client for testnet or live account reporting.

    This class intentionally exposes no order submission, cancellation, or
    liquidation methods. Keep live account reporting separate from the testnet
    execution broker.
    """

    def __init__(
        self,
        *,
        account_mode: str = "testnet",
        base_url: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> None:
        load_env_file()
        self.account_mode = account_mode.lower().strip()
        if self.account_mode not in {"testnet", "live"}:
            raise ValueError("Deribit account_mode must be 'testnet' or 'live'.")
        default_url = "https://test.deribit.com/api/v2" if self.account_mode == "testnet" else "https://www.deribit.com/api/v2"
        self.base_url = (base_url or _mode_env("DERIBIT", self.account_mode, "BASE_URL") or default_url).rstrip("/")
        self.client_id = client_id or _deribit_mode_secret(self.account_mode, "CLIENT_ID")
        self.client_secret = client_secret or _deribit_mode_secret(self.account_mode, "CLIENT_SECRET")
        self._access_token: str | None = None
        self._access_token_expires_at = 0.0
        if self.account_mode == "live" and "test.deribit.com" in self.base_url:
            raise RuntimeError("Refusing Deribit live mode with the testnet endpoint.")
        if self.account_mode == "testnet" and "test.deribit.com" not in self.base_url:
            raise RuntimeError("Refusing Deribit testnet mode with a non-testnet endpoint.")

    def public_get(self, method: str, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", f"/public/{method}", params=params, token=None)

    def private_get(self, method: str, params: dict[str, Any] | None = None) -> Any:
        try:
            return self._request("GET", f"/private/{method}", params=params, token=self.access_token())
        except RuntimeError as exc:
            if not _looks_like_invalid_token(exc):
                raise
            self._access_token = None
            self._access_token_expires_at = 0.0
            return self._request("GET", f"/private/{method}", params=params, token=self.access_token())

    def access_token(self) -> str:
        if self._access_token and time.time() < self._access_token_expires_at - 30:
            return self._access_token
        if not self.client_id or not self.client_secret:
            prefix = "DERIBIT_LIVE" if self.account_mode == "live" else "DERIBIT_TESTNET"
            fallback = " or DERIBIT_CLIENT_ID / DERIBIT_CLIENT_SECRET" if self.account_mode == "testnet" else ""
            raise RuntimeError(f"Deribit {self.account_mode} auth requires {prefix}_CLIENT_ID and {prefix}_CLIENT_SECRET{fallback}.")
        result = self.public_get(
            "auth",
            {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
        )
        token = result.get("access_token") if isinstance(result, dict) else None
        if not token:
            raise RuntimeError("Deribit auth did not return an access token.")
        self._access_token = str(token)
        expires_in = _float(result.get("expires_in")) if isinstance(result, dict) else 0.0
        self._access_token_expires_at = time.time() + max(expires_in, 60.0)
        return self._access_token

    def account_summary(self, *, currency: str = "ETH") -> dict[str, Any]:
        result = self.private_get("get_account_summary", {"currency": currency.upper(), "extended": "true"})
        return result if isinstance(result, dict) else {}

    def positions(self, *, currency: str = "ETH", kind: str = "any") -> list[dict[str, Any]]:
        params: dict[str, Any] = {"currency": currency.upper()}
        if kind and kind != "any":
            params["kind"] = kind
        result = self.private_get("get_positions", params)
        return result if isinstance(result, list) else []

    def open_orders(self, *, currency: str = "ETH", kind: str = "any") -> list[dict[str, Any]]:
        result = self.private_get("get_open_orders_by_currency", {"currency": currency.upper(), "kind": kind})
        return result if isinstance(result, list) else []

    def order_history(self, *, currency: str = "ETH", kind: str = "any", count: int = 100) -> list[dict[str, Any]]:
        result = self.private_get(
            "get_order_history_by_currency",
            {
                "currency": currency.upper(),
                "kind": kind,
                "count": max(1, min(int(count), 1000)),
                "include_old": "true",
                "include_unfilled": "true",
            },
        )
        if isinstance(result, dict) and isinstance(result.get("orders"), list):
            return result["orders"]
        return result if isinstance(result, list) else []

    def user_trades(self, *, currency: str = "ETH", kind: str = "any", count: int = 100) -> list[dict[str, Any]]:
        result = self.private_get(
            "get_user_trades_by_currency",
            {
                "currency": currency.upper(),
                "kind": kind,
                "count": max(1, min(int(count), 1000)),
                "include_old": "true",
            },
        )
        if isinstance(result, dict) and isinstance(result.get("trades"), list):
            return result["trades"]
        return result if isinstance(result, list) else []

    def _request(self, method: str, path: str, *, params: dict[str, Any] | None = None, token: str | None = None) -> Any:
        query = "" if not params else "?" + urlencode(params)
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(f"{self.base_url}{path}{query}", headers=headers, method=method)
        try:
            with urlopen(request, timeout=30) as response:
                text = response.read().decode("utf-8")
        except HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8")
            except Exception:
                body = ""
            detail = f"HTTP Error {exc.code}: {exc.reason}"
            if body:
                detail = f"{detail}: {body}"
            raise RuntimeError(f"Deribit read-only request failed: {method} {path}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"Deribit read-only request failed: {method} {path}: {exc}") from exc
        payload = json.loads(text) if text else {}
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(f"Deribit API error: {payload['error']}")
        return payload.get("result") if isinstance(payload, dict) and "result" in payload else payload


class DeribitLiveSpotBroker(DeribitReadOnlyBroker):
    """Live Deribit spot execution client.

    This class is intentionally limited to spot instruments. It should not be
    used for options or futures execution.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> None:
        super().__init__(account_mode="live", base_url=base_url, client_id=client_id, client_secret=client_secret)

    def order_book(self, instrument_name: str, *, depth: int = 10) -> dict[str, Any]:
        result = self.public_get("get_order_book", {"instrument_name": instrument_name, "depth": int(depth)})
        return result if isinstance(result, dict) else {}

    def ticker(self, instrument_name: str) -> dict[str, Any]:
        result = self.public_get("ticker", {"instrument_name": instrument_name})
        return result if isinstance(result, dict) else {}

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        result = self.private_get("cancel", {"order_id": order_id})
        return result if isinstance(result, dict) else {"result": result}

    def submit_spot_order(
        self,
        *,
        side: str,
        instrument_name: str,
        amount: float,
        order_type: str,
        price: float | None = None,
        trigger_price: float | None = None,
        trigger: str | None = None,
        label: str | None = None,
        time_in_force: str = "good_til_cancelled",
    ) -> dict[str, Any]:
        if not instrument_name.endswith("_USDC"):
            raise ValueError("Live spot broker is currently limited to *_USDC spot instruments.")
        side = side.lower().strip()
        if side not in {"buy", "sell"}:
            raise ValueError("Deribit spot side must be buy or sell.")
        if amount <= 0:
            raise ValueError("Deribit spot orders require positive amount.")
        order_type = order_type.lower().strip()
        if order_type not in {"limit", "market", "stop_market", "stop_limit", "take_market", "take_limit", "trailing_stop"}:
            raise ValueError(f"Unsupported Deribit spot order type: {order_type}")
        params: dict[str, Any] = {
            "instrument_name": instrument_name,
            "amount": amount,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if label:
            params["label"] = label[:64]
        if order_type in {"limit", "stop_limit", "take_limit"}:
            if price is None or price <= 0:
                raise ValueError(f"Deribit {order_type} spot orders require a positive price.")
            params["price"] = price
        if order_type in {"stop_market", "stop_limit", "take_market", "take_limit", "trailing_stop"}:
            if trigger_price is not None:
                params["trigger_price"] = trigger_price
            if trigger:
                params["trigger"] = trigger
        result = self.private_get(side, params)
        return result if isinstance(result, dict) else {"result": result}


class DeribitOptionsBroker(DeribitReadOnlyBroker):
    """Deribit options execution client for explicit testnet or live mode."""

    def instruments(self, *, currency: str = "ETH", kind: str = "option", expired: bool = False) -> list[dict[str, Any]]:
        result = self.public_get("get_instruments", {"currency": currency.upper(), "kind": kind, "expired": str(expired).lower()})
        return result if isinstance(result, list) else []

    def order_book(self, instrument_name: str, *, depth: int = 10) -> dict[str, Any]:
        result = self.public_get("get_order_book", {"instrument_name": instrument_name, "depth": int(depth)})
        return result if isinstance(result, dict) else {}

    def ticker(self, instrument_name: str) -> dict[str, Any]:
        result = self.public_get("ticker", {"instrument_name": instrument_name})
        return result if isinstance(result, dict) else {}

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        result = self.private_get("cancel", {"order_id": order_id})
        return result if isinstance(result, dict) else {"result": result}

    def buy_limit(
        self,
        *,
        instrument_name: str,
        amount: float,
        price: float,
        label: str | None = None,
        post_only: bool = False,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        return self._submit_limit_order(
            side="buy",
            instrument_name=instrument_name,
            amount=amount,
            price=price,
            label=label,
            post_only=post_only,
            reduce_only=reduce_only,
        )

    def sell_limit(
        self,
        *,
        instrument_name: str,
        amount: float,
        price: float,
        label: str | None = None,
        post_only: bool = False,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        return self._submit_limit_order(
            side="sell",
            instrument_name=instrument_name,
            amount=amount,
            price=price,
            label=label,
            post_only=post_only,
            reduce_only=reduce_only,
        )

    def _submit_limit_order(
        self,
        *,
        side: str,
        instrument_name: str,
        amount: float,
        price: float,
        label: str | None,
        post_only: bool,
        reduce_only: bool,
    ) -> dict[str, Any]:
        if amount <= 0:
            raise ValueError("Deribit option orders require positive amount.")
        if price <= 0:
            raise ValueError("Deribit option limit orders require positive price.")
        params: dict[str, Any] = {
            "instrument_name": instrument_name,
            "amount": amount,
            "type": "limit",
            "price": price,
            "time_in_force": "good_til_cancelled",
            "post_only": str(bool(post_only)).lower(),
            "reduce_only": str(bool(reduce_only)).lower(),
        }
        if label:
            params["label"] = label[:64]
        result = self.private_get(side, params)
        return result if isinstance(result, dict) else {"result": result}


def _env_search_paths() -> list[Path]:
    cwd = Path.cwd()
    return [cwd / ".env", *[parent / ".env" for parent in cwd.parents[:4]]]


def _mode_env(prefix: str, mode: str, suffix: str) -> str | None:
    return os.getenv(f"{prefix}_{mode.upper()}_{suffix}")


def _deribit_mode_secret(mode: str, suffix: str) -> str | None:
    value = _mode_env("DERIBIT", mode, suffix)
    if value:
        return value
    if mode == "testnet":
        return os.getenv(f"DERIBIT_{suffix}")
    return None


def _looks_like_invalid_token(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "invalid_token" in text or "unauthorized" in text or "13009" in text


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def summarize_instrument(instrument: dict[str, Any]) -> dict[str, Any]:
    return {
        "instrument_name": instrument.get("instrument_name"),
        "base_currency": instrument.get("base_currency"),
        "quote_currency": instrument.get("quote_currency"),
        "kind": instrument.get("kind"),
        "option_type": instrument.get("option_type"),
        "strike": instrument.get("strike"),
        "expiration_timestamp": instrument.get("expiration_timestamp"),
        "is_active": instrument.get("is_active"),
        "min_trade_amount": instrument.get("min_trade_amount"),
        "tick_size": instrument.get("tick_size"),
    }
