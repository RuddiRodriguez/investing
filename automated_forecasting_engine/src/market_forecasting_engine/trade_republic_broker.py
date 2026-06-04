from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator


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


class TradeRepublicReadOnlyBroker:
    """Guarded adapter for Zarathustra2/TradeRepublicApi.

    The upstream library is unofficial and authenticating can trigger device
    registration flows. This adapter keeps the first integration step read-only
    and requires explicit flags before any login path is attempted.
    """

    def __init__(
        self,
        *,
        number: str | None = None,
        pin: str | None = None,
        locale: str | None = None,
        timeout: float | None = None,
        key_dir: str | Path | None = None,
        api_factory: Callable[..., Any] | None = None,
    ) -> None:
        load_env_file()
        self.number = number or os.getenv("TRADE_REPUBLIC_PHONE_NUMBER") or os.getenv("TR_PHONE_NUMBER")
        self.pin = pin or os.getenv("TRADE_REPUBLIC_PIN") or os.getenv("TR_PIN")
        self.locale = locale or os.getenv("TRADE_REPUBLIC_LOCALE") or "en"
        self.timeout = float(timeout if timeout is not None else os.getenv("TRADE_REPUBLIC_TIMEOUT", "20"))
        self.key_dir = Path(key_dir or os.getenv("TRADE_REPUBLIC_KEY_DIR") or Path.cwd()).expanduser().resolve()
        self._api_factory = api_factory
        self._api: Any | None = None
        self._authenticated = False

    def public_instrument(self, isin: str) -> dict[str, Any]:
        result = self._api_instance(require_credentials=False).instrument(_normalize_isin(isin))
        return result if isinstance(result, dict) else {"result": result}

    def public_ticker(self, isin: str, *, exchange: str = "LSX") -> dict[str, Any]:
        result = self._api_instance(require_credentials=False).ticker(_normalize_isin(isin), exchange=exchange.upper())
        return result if isinstance(result, dict) else {"result": result}

    def public_search(
        self,
        query: str,
        *,
        page: int = 1,
        page_size: int = 20,
        instrument_type: str = "stock",
        jurisdiction: str = "DE",
    ) -> dict[str, Any]:
        result = self._api_instance(require_credentials=False).neon_search(
            query=query,
            page=int(page),
            page_size=int(page_size),
            instrument_type=instrument_type,
            jurisdiction=jurisdiction.upper(),
        )
        return result if isinstance(result, dict) else {"result": result}

    def account_snapshot(self, *, allow_login: bool = False, allow_device_registration: bool = False) -> dict[str, Any]:
        if not allow_login:
            raise RuntimeError("Trade Republic account reads require --allow-login because authentication may affect the registered device.")
        self.login_read_only(allow_device_registration=allow_device_registration)
        api = self._api_instance(require_credentials=True)
        return {
            "cash": api.cash(),
            "available_cash": api.available_cash(),
            "portfolio": api.portfolio(),
            "orders": api.orders(),
        }

    def cash_snapshot(self, *, allow_login: bool = False, allow_device_registration: bool = False) -> dict[str, Any]:
        if not allow_login:
            raise RuntimeError("Trade Republic cash reads require --allow-login because authentication may affect the registered device.")
        self.login_read_only(allow_device_registration=allow_device_registration)
        api = self._api_instance(require_credentials=True)
        return {
            "cash": api.cash(),
            "available_cash": api.available_cash(),
            "available_cash_for_payout": api.available_cash_for_payout(),
        }

    def portfolio_snapshot(self, *, allow_login: bool = False, allow_device_registration: bool = False) -> dict[str, Any]:
        if not allow_login:
            raise RuntimeError("Trade Republic portfolio reads require --allow-login because authentication may affect the registered device.")
        self.login_read_only(allow_device_registration=allow_device_registration)
        return _as_dict(self._api_instance(require_credentials=True).portfolio())

    def order_snapshot(self, *, allow_login: bool = False, allow_device_registration: bool = False) -> dict[str, Any] | list[Any]:
        if not allow_login:
            raise RuntimeError("Trade Republic order reads require --allow-login because authentication may affect the registered device.")
        self.login_read_only(allow_device_registration=allow_device_registration)
        return self._api_instance(require_credentials=True).orders()

    def timeline_movements(
        self,
        *,
        after: str | None = None,
        allow_login: bool = False,
        allow_device_registration: bool = False,
    ) -> dict[str, Any]:
        if not allow_login:
            raise RuntimeError("Trade Republic timeline reads require --allow-login because authentication may affect the registered device.")
        self.login_read_only(allow_device_registration=allow_device_registration)
        return _as_dict(self._api_instance(require_credentials=True).timeline(after=after))

    def timeline_detail(
        self,
        movement_id: str,
        *,
        allow_login: bool = False,
        allow_device_registration: bool = False,
    ) -> dict[str, Any]:
        if not allow_login:
            raise RuntimeError("Trade Republic timeline detail reads require --allow-login because authentication may affect the registered device.")
        self.login_read_only(allow_device_registration=allow_device_registration)
        return _as_dict(self._api_instance(require_credentials=True).timeline_detail(movement_id))

    def login_read_only(self, *, allow_device_registration: bool = False) -> None:
        if self._authenticated:
            return
        self._require_credentials()
        if not (self.key_dir / "key").exists() and not allow_device_registration:
            raise RuntimeError(
                "Trade Republic key file is missing. Put the upstream key file in TRADE_REPUBLIC_KEY_DIR "
                "or pass --allow-device-registration to permit the interactive registration flow."
            )
        with self._trapi_working_directory():
            self._api_instance(require_credentials=True).login()
        self._authenticated = True

    def _api_instance(self, *, require_credentials: bool) -> Any:
        if require_credentials:
            self._require_credentials()
        if self._api is None:
            factory = self._api_factory or _load_blocking_api()
            self._api = factory(self.number or "", self.pin or "", timeout=self.timeout, locale=self.locale)
        return self._api

    def _require_credentials(self) -> None:
        if not self.number or not self.pin:
            raise RuntimeError("Trade Republic auth requires TRADE_REPUBLIC_PHONE_NUMBER and TRADE_REPUBLIC_PIN.")

    @contextmanager
    def _trapi_working_directory(self) -> Iterator[None]:
        self.key_dir.mkdir(parents=True, exist_ok=True)
        old_cwd = Path.cwd()
        os.chdir(self.key_dir)
        try:
            yield
        finally:
            os.chdir(old_cwd)


def summarize_ticker(ticker: dict[str, Any]) -> dict[str, Any]:
    return {
        key: ticker.get(key)
        for key in ["bid", "ask", "last", "pre", "open", "qualityId", "leverage", "delta", "key"]
        if key in ticker
    }


def summarize_instrument(instrument: dict[str, Any]) -> dict[str, Any]:
    return {
        key: instrument.get(key)
        for key in ["id", "isin", "name", "shortName", "type", "exchangeIds", "key"]
        if key in instrument
    }


def summarize_timeline_event(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data") if isinstance(event.get("data"), dict) else event
    return {
        key: data.get(key)
        for key in ["id", "timestamp", "title", "body", "cashChangeAmount", "month", "key"]
        if key in data
    }


def summarize_timeline(timeline: dict[str, Any], *, limit: int | None = None) -> dict[str, Any]:
    rows = timeline.get("data")
    if not isinstance(rows, list):
        return timeline
    selected = rows[:limit] if limit is not None else rows
    return {
        **{key: value for key, value in timeline.items() if key != "data"},
        "movement_count": len(rows),
        "data": [summarize_timeline_event(row) if isinstance(row, dict) else row for row in selected],
    }


def _normalize_isin(isin: str) -> str:
    normalized = isin.strip().upper()
    if len(normalized) != 12 or not normalized[:2].isalpha() or not normalized[2:].isalnum():
        raise ValueError("Trade Republic requests require a 12-character ISIN, for example US88160R1014.")
    return normalized


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {"result": value}


def _load_blocking_api() -> Any:
    try:
        from trapi.api import TrBlockingApi
    except ImportError as exc:
        raise RuntimeError(
            "TradeRepublicApi or one of its runtime dependencies is not installed. "
            "Install this project with the traderepublic extra: "
            "python -m pip install -e 'automated_forecasting_engine[traderepublic]'"
        ) from exc
    return TrBlockingApi


def _env_search_paths() -> list[Path]:
    cwd = Path.cwd()
    return [cwd / ".env", *[parent / ".env" for parent in cwd.parents[:4]]]
