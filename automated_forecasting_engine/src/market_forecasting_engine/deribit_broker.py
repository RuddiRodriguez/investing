from __future__ import annotations

import json
import os
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


def _env_search_paths() -> list[Path]:
    cwd = Path.cwd()
    return [cwd / ".env", *[parent / ".env" for parent in cwd.parents[:4]]]


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
