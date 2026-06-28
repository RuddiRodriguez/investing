from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any


DEFAULT_EXPORT_PATH = Path("snaptrade_exports/snapshot_latest.json")
DEFAULT_INTERVAL_SECONDS = 60


def main() -> None:
    load_env_file()
    args = build_parser().parse_args()
    if args.command == "loop":
        run_loop(args)
        return
    payload = run_command(args)
    write_payload(payload, args.output, args.format)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only SnapTrade portfolio/account data fetcher.")
    parser.add_argument("--client-id", default=None, help="SnapTrade clientId. Defaults to SNAPTRADE_CLIENT_ID.")
    parser.add_argument("--consumer-key", default=None, help="SnapTrade consumerKey. Defaults to SNAPTRADE_CONSUMER_KEY.")
    parser.add_argument("--user-id", default=None, help="SnapTrade userId. Defaults to SNAPTRADE_USER_ID.")
    parser.add_argument("--user-secret", default=None, help="SnapTrade userSecret. Defaults to SNAPTRADE_USER_SECRET.")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON/CSV output instead of stdout.")
    parser.add_argument("--format", choices=("json", "csv"), default="json")

    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Check SnapTrade API status.")
    commands.add_parser("users", help="List SnapTrade user IDs for this client.")
    commands.add_parser("connections", help="List brokerage connections for the configured user.")
    commands.add_parser("accounts", help="List connected brokerage accounts for the configured user.")

    snapshot = commands.add_parser("snapshot", help="Fetch connections, accounts, balances, positions, and optional recent orders/activity.")
    snapshot.add_argument("--include-orders", action=argparse.BooleanOptionalAction, default=True)
    snapshot.add_argument("--include-activities", action=argparse.BooleanOptionalAction, default=False)
    snapshot.add_argument("--include-balance-history", action=argparse.BooleanOptionalAction, default=True)
    snapshot.add_argument("--activities-days", type=int, default=90)
    snapshot.add_argument("--orders-days", type=int, default=30)

    loop = commands.add_parser("loop", help="Continuously refresh a read-only SnapTrade snapshot.")
    loop.add_argument("--output", type=Path, default=DEFAULT_EXPORT_PATH)
    loop.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
    loop.add_argument("--include-orders", action=argparse.BooleanOptionalAction, default=True)
    loop.add_argument("--include-activities", action=argparse.BooleanOptionalAction, default=False)
    loop.add_argument("--include-balance-history", action=argparse.BooleanOptionalAction, default=True)
    loop.add_argument("--activities-days", type=int, default=90)
    loop.add_argument("--orders-days", type=int, default=30)
    return parser


def run_loop(args: argparse.Namespace) -> None:
    while True:
        payload = run_command(argparse.Namespace(**{**vars(args), "command": "snapshot", "format": "json"}))
        write_payload(payload, args.output, "json")
        print(
            json.dumps(
                {
                    "broker": "snaptrade",
                    "read_only": True,
                    "output": str(args.output),
                    "fetched_at": payload.get("fetched_at"),
                    "account_count": len(payload.get("accounts") or []),
                    "error": payload.get("error"),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        time.sleep(max(5, int(args.interval_seconds)))


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    client = build_client(args)
    base = {
        "broker": "snaptrade",
        "read_only": True,
        "execution_enabled": False,
        "command": args.command,
        "fetched_at": current_timestamp(),
    }
    try:
        if args.command == "status":
            base["status"] = response_body(client.api_status.check())
            return base
        if args.command == "users":
            base["users"] = response_body(client.authentication.list_snap_trade_users())
            return base
        credentials = user_credentials(args)
        if args.command == "connections":
            base["connections"] = list_connections(client, credentials)
            return base
        if args.command == "accounts":
            base["accounts"] = list_accounts(client, credentials)
            return base
        if args.command == "snapshot":
            base.update(fetch_snapshot(client, credentials, args))
            return base
    except Exception as exc:  # SDK exceptions expose useful status/body but vary by version.
        base["error"] = safe_error(exc)
        return base
    raise ValueError(f"Unsupported SnapTrade command: {args.command}")


def build_client(args: argparse.Namespace) -> Any:
    try:
        from snaptrade_client import SnapTrade
    except ImportError as exc:
        raise RuntimeError("snaptrade-python-sdk is not installed. Install with `pip install snaptrade-python-sdk==11.0.207`.") from exc
    client_id = args.client_id or os.getenv("SNAPTRADE_CLIENT_ID")
    consumer_key = args.consumer_key or os.getenv("SNAPTRADE_CONSUMER_KEY")
    if not client_id or not consumer_key:
        raise RuntimeError("Missing SNAPTRADE_CLIENT_ID or SNAPTRADE_CONSUMER_KEY.")
    return SnapTrade(client_id=client_id, consumer_key=consumer_key)


def user_credentials(args: argparse.Namespace) -> dict[str, str]:
    user_id = args.user_id or os.getenv("SNAPTRADE_USER_ID")
    user_secret = args.user_secret or os.getenv("SNAPTRADE_USER_SECRET")
    if not user_id or not user_secret:
        raise RuntimeError("Missing SNAPTRADE_USER_ID or SNAPTRADE_USER_SECRET for user account data.")
    return {"user_id": user_id, "user_secret": user_secret}


def fetch_snapshot(client: Any, credentials: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    connections = list_connections(client, credentials)
    accounts = list_accounts(client, credentials)
    account_rows: list[dict[str, Any]] = []
    for account in accounts if isinstance(accounts, list) else []:
        account_id = first_non_empty(account, "id", "accountId", "account_id")
        row = {"account": account}
        if account_id:
            row["account_id"] = account_id
            row["balances"] = call_or_error(client.account_information.get_user_account_balance, account_id=account_id, **credentials)
            row["positions"] = call_or_error(client.account_information.get_all_account_positions, account_id=account_id, **credentials)
            row["details"] = call_or_error(client.account_information.get_user_account_details, account_id=account_id, **credentials)
            if args.include_orders:
                row["orders"] = call_or_error(
                    client.account_information.get_user_account_orders,
                    account_id=account_id,
                    state="all",
                    days=max(1, min(90, int(args.orders_days))),
                    **credentials,
                )
            if args.include_activities:
                start = date.today() - timedelta(days=max(1, int(args.activities_days)))
                row["activities"] = call_or_error(
                    client.account_information.get_account_activities,
                    account_id=account_id,
                    start_date=start.isoformat(),
                    end_date=date.today().isoformat(),
                    limit=1000,
                    offset=0,
                    **credentials,
                )
            if args.include_balance_history:
                row["balance_history"] = call_or_error(
                    client.account_information.get_account_balance_history,
                    account_id=account_id,
                    **credentials,
                )
        account_rows.append(row)
    return {"connections": connections, "accounts": account_rows}


def list_connections(client: Any, credentials: dict[str, str]) -> Any:
    return response_body(client.connections.list_brokerage_authorizations(**credentials))


def list_accounts(client: Any, credentials: dict[str, str]) -> Any:
    return response_body(client.account_information.list_user_accounts(**credentials))


def call_or_error(func: Any, **kwargs: Any) -> Any:
    try:
        return response_body(func(**kwargs))
    except Exception as exc:
        return {"error": safe_error(exc)}


def response_body(response: Any) -> Any:
    body = getattr(response, "body", response)
    return to_plain_data(body)


def to_plain_data(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_plain_data(item) for item in value]
    if hasattr(value, "to_dict"):
        return to_plain_data(value.to_dict())
    if hasattr(value, "model_dump"):
        return to_plain_data(value.model_dump())
    if hasattr(value, "__dict__"):
        return {key: to_plain_data(item) for key, item in vars(value).items() if not key.startswith("_")}
    return str(value)


def write_payload(payload: dict[str, Any], output: Path | None, output_format: str) -> None:
    if output is None:
        print(json.dumps(payload, indent=2, ensure_ascii=True, default=str))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
        return
    rows = snapshot_to_csv_rows(payload)
    import csv

    with output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "broker",
            "account_id",
            "account_name",
            "institution",
            "symbol",
            "name",
            "asset_type",
            "quantity",
            "current_price",
            "current_value",
            "currency",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def snapshot_to_csv_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    from market_forecasting_engine.unified_portfolio import normalize_snaptrade_snapshot

    normalized = normalize_snaptrade_snapshot(payload)
    return [
        {
            "broker": row.get("broker"),
            "account_id": row.get("account_id"),
            "account_name": row.get("account_name"),
            "institution": row.get("institution"),
            "symbol": row.get("ticker"),
            "name": row.get("name"),
            "asset_type": row.get("asset_type"),
            "quantity": row.get("quantity"),
            "current_price": row.get("current_price"),
            "current_value": row.get("current_value"),
            "currency": row.get("currency"),
        }
        for row in normalized
    ]


def first_non_empty(mapping: Any, *paths: str) -> Any:
    for path in paths:
        value = get_path(mapping, path)
        if value not in (None, ""):
            return value
    return None


def get_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current


def safe_error(exc: Exception) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "status": getattr(exc, "status", None),
        "reason": getattr(exc, "reason", None),
        "body": to_plain_data(getattr(exc, "body", None)),
    }


def current_timestamp() -> str:
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def _env_search_paths() -> list[Path]:
    cwd = Path.cwd()
    return [cwd / ".env", *[parent / ".env" for parent in cwd.parents[:5]]]


if __name__ == "__main__":
    main()
