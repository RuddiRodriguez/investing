from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    load_env_file()
    args = build_parser().parse_args()
    payload: dict[str, Any] = {
        "broker": "traderepublic",
        "backend": "pytr-org/pytr",
        "read_only": True,
        "execution_enabled": False,
        "command": args.command,
    }
    if not args.allow_login:
        payload["error"] = "Trade Republic read-only account access requires --allow-login."
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        raise SystemExit(1)

    command = build_pytr_command(args)
    if args.print_command:
        payload["pytr_command"] = _redacted_command(command)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return

    result = subprocess.run(command, check=False, env=_subprocess_env())
    raise SystemExit(result.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Trade Republic portfolio and movement export via pytr.")
    parser.add_argument("--allow-login", action="store_true", help="Permit read-only Trade Republic login.")
    parser.add_argument("--phone-no", default=None, help="Trade Republic phone number. Omit to use pytr credentials file/prompt.")
    parser.add_argument("--pin", default=None, help="Trade Republic PIN. Omit to use pytr credentials file/prompt.")
    parser.add_argument("--store-credentials", action="store_true", help="Let pytr store credentials/cookies for later reads.")
    parser.add_argument("--waf-token", default="playwright", help='pytr WAF token mode/value. Default: "playwright".')
    parser.add_argument("--verbosity", default="info", choices=("warning", "info", "debug"))
    parser.add_argument("--print-command", action="store_true", help="Print the redacted pytr command without running it.")

    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("login", help="Test read-only Trade Republic login.")

    portfolio = commands.add_parser("portfolio", help="Show or export current portfolio.")
    portfolio.add_argument("-o", "--output", type=Path, default=None, help="CSV output path.")
    portfolio.add_argument("--include-watchlist", action=argparse.BooleanOptionalAction, default=False)
    portfolio.add_argument("--lang", default="en")
    portfolio.add_argument("--decimal-localization", action=argparse.BooleanOptionalAction, default=False)
    portfolio.add_argument("--sort-by-column", default=None)
    portfolio.add_argument("--sort-ascending", action=argparse.BooleanOptionalAction, default=False)

    movements = commands.add_parser("movements", help="Export timeline/account movements.")
    movements.add_argument("--output-dir", type=Path, default=Path("trade_republic_exports"))
    movements.add_argument("--output-file", type=Path, default=None, help="Explicit CSV/JSON output file.")
    movements.add_argument("--format", choices=("csv", "json"), default="csv")
    movements.add_argument("--last-days", type=int, default=0, help="0 means all days; -1 reuses existing event database.")
    movements.add_argument("--days-until", type=int, default=0)
    movements.add_argument("--lang", default="en")
    movements.add_argument("--sort", action="store_true")
    movements.add_argument("--dump-raw-data", action=argparse.BooleanOptionalAction, default=False)
    movements.add_argument("--store-event-database", action=argparse.BooleanOptionalAction, default=True)

    documents = commands.add_parser("documents", help="Download timeline documents and export movements.")
    documents.add_argument("output_dir", type=Path)
    documents.add_argument("--last-days", type=int, default=0)
    documents.add_argument("--days-until", type=int, default=0)
    documents.add_argument("--filename-format", default="{iso_date} {time} {title}")
    documents.add_argument("--export-format", choices=("csv", "json"), default="csv")
    documents.add_argument("--workers", type=int, default=8)
    documents.add_argument("--flat", action="store_true")
    documents.add_argument("--lang", default="en")
    documents.add_argument("--sort", action="store_true")
    documents.add_argument("--dump-raw-data", action=argparse.BooleanOptionalAction, default=False)

    return parser


def build_pytr_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, "-m", "pytr", "--verbosity", args.verbosity]
    if args.command == "login":
        command.append("login")
        _append_auth_args(command, args)
        return command

    if args.command == "portfolio":
        command.append("portfolio")
        _append_auth_args(command, args)
        command.extend(["--lang", args.lang])
        command.extend(["--decimal-localization" if args.decimal_localization else "--no-decimal-localization"])
        command.extend(["--include-watchlist" if args.include_watchlist else "--no-include-watchlist"])
        command.extend(["--sort-ascending" if args.sort_ascending else "--no-sort-ascending"])
        if args.output is not None:
            command.extend(["--output", str(args.output)])
        if args.sort_by_column:
            command.extend(["--sort-by-column", args.sort_by_column])
        return command

    if args.command == "movements":
        command.append("export_transactions")
        _append_auth_args(command, args)
        command.extend(["--outputdir", str(args.output_dir)])
        command.extend(["--format", args.format])
        command.extend(["--last_days", str(args.last_days)])
        command.extend(["--days_until", str(args.days_until)])
        command.extend(["--lang", args.lang])
        command.extend(["--store-event-database" if args.store_event_database else "--no-store-event-database"])
        command.extend(["--dump-raw-data" if args.dump_raw_data else "--no-dump-raw-data"])
        if args.sort:
            command.append("--sort")
        if args.output_file is not None:
            command.append(str(args.output_file))
        return command

    if args.command == "documents":
        command.append("dl_docs")
        _append_auth_args(command, args)
        command.append(str(args.output_dir))
        command.extend(["--format", args.filename_format])
        command.extend(["--last_days", str(args.last_days)])
        command.extend(["--days_until", str(args.days_until)])
        command.extend(["--workers", str(args.workers)])
        command.extend(["--export-format", args.export_format])
        command.extend(["--lang", args.lang])
        command.extend(["--dump-raw-data" if args.dump_raw_data else "--no-dump-raw-data"])
        if args.flat:
            command.append("--flat")
        if args.sort:
            command.append("--sort")
        return command

    raise ValueError(f"Unsupported read-only command: {args.command}")


def _append_auth_args(command: list[str], args: argparse.Namespace) -> None:
    phone_no = args.phone_no or os.getenv("TRADE_REPUBLIC_PHONE_NUMBER") or os.getenv("TR_PHONE_NUMBER")
    pin = args.pin or os.getenv("TRADE_REPUBLIC_PIN") or os.getenv("TR_PIN")
    if phone_no:
        command.extend(["--phone_no", phone_no])
    if pin:
        command.extend(["--pin", pin])
    if args.store_credentials:
        command.append("--store_credentials")
    if args.waf_token:
        command.extend(["--waf-token", args.waf_token])


def _redacted_command(command: list[str]) -> list[str]:
    redacted = list(command)
    for index, item in enumerate(redacted[:-1]):
        if item in {"--phone_no", "--pin", "--waf-token"}:
            redacted[index + 1] = "***"
    return redacted


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    scripts_dir = str(Path(sys.executable).resolve().parent)
    env["PATH"] = scripts_dir + os.pathsep + env.get("PATH", "")
    src_dir = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    chrome_path = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if chrome_path.exists() and not env.get("PYTR_PLAYWRIGHT_EXECUTABLE_PATH"):
        env["PYTR_PLAYWRIGHT_EXECUTABLE_PATH"] = str(chrome_path)
    return env


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
    return [cwd / ".env", *[parent / ".env" for parent in cwd.parents[:4]]]


if __name__ == "__main__":
    main()
