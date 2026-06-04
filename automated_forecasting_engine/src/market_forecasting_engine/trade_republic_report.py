from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable


@dataclass
class HoldingReport:
    isin: str
    ticker: str | None
    alpaca_ticker: str | None
    name: str | None
    ticker_resolution_source: str
    current_quantity: float
    current_price: float | None
    current_value: float
    broker_avg_cost: float | None
    open_cost_basis: float
    unrealized_pl: float
    unrealized_pl_pct: float | None
    historical_buy_cash: float
    historical_sell_cash: float
    historical_net_cash: float
    transaction_buy_shares: float
    transaction_sell_shares: float
    weighted_paid_price: float | None
    weighted_market_price_at_buy: float | None
    paid_vs_market_at_buy: float | None
    paid_vs_market_at_buy_pct: float | None
    historical_price_status: str
    alpaca_weighted_price_at_buy_time: float | None = None
    alpaca_paid_vs_market_at_buy_time: float | None = None
    alpaca_paid_vs_market_at_buy_time_pct: float | None = None
    alpaca_status: str = "not_requested"


def main() -> None:
    load_env_file()
    args = build_parser().parse_args()
    report = build_report(
        portfolio_path=args.portfolio,
        transactions_path=args.transactions,
        isin_map_path=args.isin_map,
        price_history_path=args.price_history,
        fetch_yahoo=args.fetch_yahoo,
        fetch_alpaca=args.fetch_alpaca,
        resolve_tickers_llm=args.resolve_tickers_llm,
        llm_model=args.llm_model,
        update_isin_map=args.update_isin_map,
    )
    write_report(report, args.output, args.format)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Trade Republic investment/P&L report from pytr exports.")
    parser.add_argument("--portfolio", type=Path, required=True, help="pytr portfolio CSV from the read-only portfolio command.")
    parser.add_argument("--transactions", type=Path, required=True, help="pytr account_transactions CSV or JSON export.")
    parser.add_argument("--isin-map", type=Path, default=None, help="Optional CSV with columns isin,ticker,name.")
    parser.add_argument(
        "--price-history",
        type=Path,
        default=None,
        help="Optional CSV with columns date,close and either isin or ticker for historical market comparison.",
    )
    parser.add_argument(
        "--fetch-yahoo",
        action="store_true",
        help="Fetch missing historical buy-date closes from Yahoo Finance. Requires --isin-map with tickers.",
    )
    parser.add_argument(
        "--fetch-alpaca",
        action="store_true",
        help="Fetch Alpaca 1-minute bars around each buy timestamp for supported stock symbols.",
    )
    parser.add_argument(
        "--resolve-tickers-llm",
        action="store_true",
        help="Use a small LLM call to resolve missing Yahoo/Alpaca tickers from company/fund names.",
    )
    parser.add_argument(
        "--update-isin-map",
        action="store_true",
        help="Persist newly resolved LLM tickers back to --isin-map so future runs do not call the LLM for them.",
    )
    parser.add_argument("--llm-model", default=None, help="Ticker resolver model. Defaults to OPENAI_MODEL or project default.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    return parser


def build_report(
    *,
    portfolio_path: Path,
    transactions_path: Path,
    isin_map_path: Path | None = None,
    price_history_path: Path | None = None,
    fetch_yahoo: bool = False,
    fetch_alpaca: bool = False,
    resolve_tickers_llm: bool = False,
    llm_model: str | None = None,
    ticker_resolver: Any | None = None,
    update_isin_map: bool = False,
) -> dict[str, Any]:
    isin_map = load_isin_map(isin_map_path)
    prices = load_price_history(price_history_path)
    portfolio_rows = load_portfolio(portfolio_path)
    transactions = list(load_transactions(transactions_path))
    ticker_resolution: list[dict[str, Any]] = []
    if resolve_tickers_llm:
        ticker_resolution = resolve_missing_tickers_from_names(
            portfolio_rows,
            transactions,
            isin_map,
            resolver=ticker_resolver,
            model=llm_model,
        )
        if update_isin_map:
            if isin_map_path is None:
                raise RuntimeError("--update-isin-map requires --isin-map.")
            write_isin_map(isin_map_path, isin_map)
    if fetch_yahoo:
        prices.update(fetch_yahoo_prices(transactions, isin_map, existing_prices=prices))
    transaction_stats = summarize_transactions(transactions, isin_map, prices)
    if fetch_alpaca:
        merge_alpaca_execution_stats(transaction_stats, transactions, isin_map, fetch_alpaca_execution_prices(transactions, isin_map))
    holdings: list[HoldingReport] = []

    for row in portfolio_rows:
        isin = str(row.get("ISIN") or row.get("isin") or "").strip().upper()
        if not isin:
            continue
        name = str(row.get("Name") or row.get("name") or "").strip() or isin_map.get(isin, {}).get("name")
        ticker = isin_map.get(isin, {}).get("ticker")
        alpaca_ticker = isin_map.get(isin, {}).get("alpaca_ticker")
        ticker_resolution_source = isin_map.get(isin, {}).get("source") or "missing"
        quantity = parse_decimal(row.get("quantity"))
        current_price = parse_decimal(row.get("price"), default=None)
        current_value = parse_decimal(row.get("netValue"))
        avg_cost = parse_decimal(row.get("avgCost"), default=None)
        open_cost_basis = (avg_cost or Decimal("0")) * quantity
        unrealized = current_value - open_cost_basis
        stats = transaction_stats.get(isin, {})
        weighted_market = stats.get("weighted_market_price_at_buy")
        weighted_paid = stats.get("weighted_paid_price")
        alpaca_weighted = stats.get("alpaca_weighted_price_at_buy_time")
        paid_vs_market = None
        paid_vs_market_pct = None
        if weighted_market is not None and weighted_paid is not None:
            paid_vs_market = weighted_paid - weighted_market
            paid_vs_market_pct = None if weighted_market == 0 else (paid_vs_market / weighted_market) * Decimal("100")
        alpaca_paid_vs_market = None
        alpaca_paid_vs_market_pct = None
        if alpaca_weighted is not None and weighted_paid is not None:
            alpaca_paid_vs_market = weighted_paid - alpaca_weighted
            alpaca_paid_vs_market_pct = None if alpaca_weighted == 0 else (alpaca_paid_vs_market / alpaca_weighted) * Decimal("100")
        holdings.append(
            HoldingReport(
                isin=isin,
                ticker=ticker,
                alpaca_ticker=alpaca_ticker,
                name=name,
                ticker_resolution_source=ticker_resolution_source,
                current_quantity=float(quantity),
                current_price=float(current_price) if current_price is not None else None,
                current_value=float(current_value),
                broker_avg_cost=float(avg_cost) if avg_cost is not None else None,
                open_cost_basis=float(open_cost_basis),
                unrealized_pl=float(unrealized),
                unrealized_pl_pct=None if open_cost_basis == 0 else float((unrealized / open_cost_basis) * Decimal("100")),
                historical_buy_cash=float(stats.get("buy_cash", Decimal("0"))),
                historical_sell_cash=float(stats.get("sell_cash", Decimal("0"))),
                historical_net_cash=float(stats.get("net_cash", Decimal("0"))),
                transaction_buy_shares=float(stats.get("buy_shares", Decimal("0"))),
                transaction_sell_shares=float(stats.get("sell_shares", Decimal("0"))),
                weighted_paid_price=float(weighted_paid) if weighted_paid is not None else None,
                weighted_market_price_at_buy=float(weighted_market) if weighted_market is not None else None,
                paid_vs_market_at_buy=float(paid_vs_market) if paid_vs_market is not None else None,
                paid_vs_market_at_buy_pct=float(paid_vs_market_pct) if paid_vs_market_pct is not None else None,
                historical_price_status=str(stats.get("historical_price_status", "missing_price_history")),
                alpaca_weighted_price_at_buy_time=float(alpaca_weighted) if alpaca_weighted is not None else None,
                alpaca_paid_vs_market_at_buy_time=float(alpaca_paid_vs_market) if alpaca_paid_vs_market is not None else None,
                alpaca_paid_vs_market_at_buy_time_pct=float(alpaca_paid_vs_market_pct) if alpaca_paid_vs_market_pct is not None else None,
                alpaca_status=str(stats.get("alpaca_status", "not_requested" if not fetch_alpaca else "missing")),
            )
        )

    total_cost = sum(Decimal(str(item.open_cost_basis)) for item in holdings)
    total_value = sum(Decimal(str(item.current_value)) for item in holdings)
    total_pl = total_value - total_cost
    return {
        "summary": {
            "report_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "holding_count": len(holdings),
            "total_open_cost_basis": float(total_cost),
            "total_current_value": float(total_value),
            "total_unrealized_pl": float(total_pl),
            "total_unrealized_pl_pct": None if total_cost == 0 else float((total_pl / total_cost) * Decimal("100")),
            "total_historical_buy_cash": float(sum(Decimal(str(item.historical_buy_cash)) for item in holdings)),
            "total_historical_sell_cash": float(sum(Decimal(str(item.historical_sell_cash)) for item in holdings)),
            "ticker_resolution_count": len(ticker_resolution),
        },
        "ticker_resolution": ticker_resolution,
        "holdings": [asdict(item) for item in sorted(holdings, key=lambda item: item.current_value, reverse=True)],
    }


def summarize_transactions(
    transactions: Iterable[dict[str, Any]],
    isin_map: dict[str, dict[str, str]],
    prices: dict[tuple[str, str], Decimal],
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(Decimal))
    market_weight: dict[str, Decimal] = defaultdict(Decimal)
    paid_weight: dict[str, Decimal] = defaultdict(Decimal)
    market_share_weight: dict[str, Decimal] = defaultdict(Decimal)
    missing_prices: set[str] = set()
    matched_prices: set[str] = set()

    for row in transactions:
        isin = normalize_text(value_for(row, "isin")).upper()
        if not isin:
            continue
        value = parse_decimal(value_for(row, "value"), default=Decimal("0"))
        shares = abs(parse_decimal(value_for(row, "shares"), default=Decimal("0")))
        row_type = normalize_text(value_for(row, "type")).lower()
        is_buy = row_type in {"buy", "kauf"} or (value < 0 and shares > 0)
        is_sell = row_type in {"sell", "verkauf"} or (value > 0 and shares > 0)
        if is_buy:
            cash = abs(value)
            stats[isin]["buy_cash"] += cash
            stats[isin]["net_cash"] += cash
            stats[isin]["buy_shares"] += shares
            if shares > 0:
                paid_price = cash / shares
                paid_weight[isin] += paid_price * shares
                market_price = historical_price_for(row, isin, isin_map, prices)
                if market_price is None:
                    missing_prices.add(isin)
                else:
                    matched_prices.add(isin)
                    market_weight[isin] += market_price * shares
                    market_share_weight[isin] += shares
        elif is_sell:
            cash = abs(value)
            stats[isin]["sell_cash"] += cash
            stats[isin]["net_cash"] -= cash
            stats[isin]["sell_shares"] += shares

    for isin, item in stats.items():
        buy_shares = item.get("buy_shares", Decimal("0"))
        item["weighted_paid_price"] = None if buy_shares == 0 else paid_weight[isin] / buy_shares
        item["weighted_market_price_at_buy"] = (
            None if market_share_weight[isin] == 0 else market_weight[isin] / market_share_weight[isin]
        )
        if isin in matched_prices and isin in missing_prices:
            item["historical_price_status"] = "partial"
        elif isin in matched_prices:
            item["historical_price_status"] = "matched"
        else:
            item["historical_price_status"] = "missing_price_history"
    return stats


def resolve_missing_tickers_from_names(
    portfolio_rows: Iterable[dict[str, Any]],
    transactions: Iterable[dict[str, Any]],
    isin_map: dict[str, dict[str, str]],
    *,
    resolver: Any | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    items = ticker_resolution_items(portfolio_rows, transactions, isin_map)
    if not items:
        return []
    resolver = resolver or resolve_tickers_with_openai
    resolved = resolver(items, model=model)
    audit: list[dict[str, Any]] = []
    for item in resolved:
        isin = normalize_text(item.get("isin")).upper()
        if not isin:
            continue
        current = isin_map.setdefault(isin, {})
        yahoo_ticker = normalize_text(item.get("yahoo_ticker")).upper()
        alpaca_ticker = normalize_text(item.get("alpaca_ticker")).upper()
        confidence = item.get("confidence")
        if yahoo_ticker and not current.get("ticker"):
            current["ticker"] = yahoo_ticker
        if alpaca_ticker and not current.get("alpaca_ticker"):
            current["alpaca_ticker"] = alpaca_ticker
        if item.get("name") and not current.get("name"):
            current["name"] = normalize_text(item.get("name"))
        current["source"] = "llm_name_resolution"
        current["confidence"] = str(confidence) if confidence is not None else ""
        audit.append(
            {
                "isin": isin,
                "name": current.get("name") or item.get("name"),
                "yahoo_ticker": current.get("ticker"),
                "alpaca_ticker": current.get("alpaca_ticker"),
                "confidence": confidence,
                "source": "llm_name_resolution",
                "reason": item.get("reason"),
            }
        )
    return audit


def ticker_resolution_items(
    portfolio_rows: Iterable[dict[str, Any]],
    transactions: Iterable[dict[str, Any]],
    isin_map: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    names_by_isin: dict[str, str] = {}
    for row in portfolio_rows:
        isin = normalize_text(row.get("ISIN") or row.get("isin")).upper()
        name = normalize_text(row.get("Name") or row.get("name"))
        if isin and name:
            names_by_isin[isin] = name
    for row in transactions:
        isin = normalize_text(value_for(row, "isin")).upper()
        name = normalize_text(row.get("Note") or row.get("note"))
        if isin and name and isin not in names_by_isin:
            names_by_isin[isin] = name
    items: list[dict[str, str]] = []
    for isin, name in sorted(names_by_isin.items()):
        mapped = isin_map.get(isin, {})
        yahoo = mapped.get("ticker", "")
        alpaca = mapped.get("alpaca_ticker", "")
        needs_yahoo = not yahoo
        needs_alpaca = not alpaca or not _alpaca_stock_symbol_supported(alpaca)
        if needs_yahoo or needs_alpaca:
            items.append(
                {
                    "isin": isin,
                    "name": name,
                    "existing_yahoo_ticker": yahoo,
                    "existing_alpaca_ticker": alpaca,
                    "needs_yahoo": str(needs_yahoo).lower(),
                    "needs_alpaca": str(needs_alpaca).lower(),
                }
            )
    return items


def resolve_tickers_with_openai(items: list[dict[str, str]], *, model: str | None = None) -> list[dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Ticker LLM resolution requires OPENAI_API_KEY or a test resolver.")
    from openai import OpenAI
    from market_forecasting_engine.openai_models import DEFAULT_OPENAI_MODEL
    from market_forecasting_engine.openai_responses import call_response

    selected_model = model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    client = OpenAI(api_key=api_key, timeout=30)
    _, _, parsed = call_response(
        client=client,
        model=selected_model,
        system_message=(
            "Resolve public market tickers from Trade Republic instrument names and ISINs. "
            "Return only high-confidence Yahoo Finance and Alpaca stock symbols. "
            "Use null when the instrument is an ETF, crypto, non-US listing, or ambiguous for Alpaca. "
            "Do not invent a ticker when uncertain."
        ),
        user_message=json.dumps({"items": items}, ensure_ascii=True),
        json_schema=ticker_resolution_schema(),
        reasoning_effort="none",
        usage_context={"purpose": "trade_republic_ticker_resolution", "item_count": len(items)},
    )
    results = parsed.get("items", [])
    return results if isinstance(results, list) else []


def ticker_resolution_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "ticker_resolution",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["items"],
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["isin", "name", "yahoo_ticker", "alpaca_ticker", "confidence", "reason"],
                        "properties": {
                            "isin": {"type": "string"},
                            "name": {"type": "string"},
                            "yahoo_ticker": {"type": ["string", "null"]},
                            "alpaca_ticker": {"type": ["string", "null"]},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "reason": {"type": "string"},
                        },
                    },
                }
            },
        },
        "strict": True,
    }


def historical_price_for(
    row: dict[str, Any],
    isin: str,
    isin_map: dict[str, dict[str, str]],
    prices: dict[tuple[str, str], Decimal],
) -> Decimal | None:
    date = normalize_text(value_for(row, "date"))[:10]
    ticker = isin_map.get(isin, {}).get("ticker")
    return prices.get((isin, date)) or (prices.get((ticker, date)) if ticker else None)


def load_portfolio(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file, delimiter=";"))


def load_transactions(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        if text.startswith("["):
            return json.loads(text)
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    with path.open(newline="", encoding="utf-8") as file:
        sample = file.read(4096)
        file.seek(0)
        delimiter = ";" if sample.count(";") >= sample.count(",") else ","
        return list(csv.DictReader(file, delimiter=delimiter))


def load_isin_map(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        return {
            normalize_text(row.get("isin")).upper(): {
                "ticker": normalize_text(row.get("ticker")).upper(),
                "alpaca_ticker": normalize_text(row.get("alpaca_ticker")).upper(),
                "name": normalize_text(row.get("name")),
                "source": normalize_text(row.get("source")) or "manual_map",
                "confidence": normalize_text(row.get("confidence")),
            }
            for row in rows
            if normalize_text(row.get("isin"))
        }


def write_isin_map(path: Path, isin_map: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["isin", "ticker", "alpaca_ticker", "name", "source", "confidence"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for isin, values in sorted(isin_map.items()):
            writer.writerow(
                {
                    "isin": isin,
                    "ticker": values.get("ticker", ""),
                    "alpaca_ticker": values.get("alpaca_ticker", ""),
                    "name": values.get("name", ""),
                    "source": values.get("source", ""),
                    "confidence": values.get("confidence", ""),
                }
            )


def load_price_history(path: Path | None) -> dict[tuple[str, str], Decimal]:
    if path is None:
        return {}
    output: dict[tuple[str, str], Decimal] = {}
    with path.open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        for row in rows:
            symbol = normalize_text(row.get("isin") or row.get("ticker")).upper()
            date = normalize_text(row.get("date"))[:10]
            close = parse_decimal(row.get("close"), default=None)
            if symbol and date and close is not None:
                output[(symbol, date)] = close
    return output


def fetch_yahoo_prices(
    transactions: Iterable[dict[str, Any]],
    isin_map: dict[str, dict[str, str]],
    *,
    existing_prices: dict[tuple[str, str], Decimal] | None = None,
) -> dict[tuple[str, str], Decimal]:
    existing_prices = existing_prices or {}
    requests_by_ticker: dict[str, set[str]] = defaultdict(set)
    for row in transactions:
        isin = normalize_text(value_for(row, "isin")).upper()
        if not isin:
            continue
        value = parse_decimal(value_for(row, "value"), default=Decimal("0"))
        shares = abs(parse_decimal(value_for(row, "shares"), default=Decimal("0")))
        row_type = normalize_text(value_for(row, "type")).lower()
        is_buy = row_type in {"buy", "kauf"} or (value < 0 and shares > 0)
        if not is_buy:
            continue
        ticker = isin_map.get(isin, {}).get("ticker")
        date = normalize_text(value_for(row, "date"))[:10]
        if ticker and date and (ticker, date) not in existing_prices:
            requests_by_ticker[ticker].add(date)

    if not requests_by_ticker:
        return {}
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Yahoo price fetching requires yfinance. Install project dependencies first.") from exc

    fetched: dict[tuple[str, str], Decimal] = {}
    for ticker, dates in requests_by_ticker.items():
        start_date = min(datetime.fromisoformat(date).date() for date in dates)
        end_date = max(datetime.fromisoformat(date).date() for date in dates) + timedelta(days=7)
        data = yf.download(
            ticker,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if data.empty:
            continue
        close_series = data["Close"]
        if hasattr(close_series, "columns"):
            close_series = close_series.iloc[:, 0]
        available = {index.date().isoformat(): Decimal(str(value)) for index, value in close_series.dropna().items()}
        sorted_dates = sorted(available)
        for date in dates:
            selected_date = date if date in available else next((candidate for candidate in sorted_dates if candidate >= date), None)
            if selected_date:
                fetched[(ticker, date)] = available[selected_date]
    return fetched


def merge_alpaca_execution_stats(
    stats: dict[str, dict[str, Any]],
    transactions: Iterable[dict[str, Any]],
    isin_map: dict[str, dict[str, str]],
    alpaca_prices: dict[tuple[str, str], dict[str, Any]],
) -> None:
    alpaca_weight: dict[str, Decimal] = defaultdict(Decimal)
    alpaca_share_weight: dict[str, Decimal] = defaultdict(Decimal)
    statuses: dict[str, set[str]] = defaultdict(set)
    for row in transactions:
        isin = normalize_text(value_for(row, "isin")).upper()
        if not isin:
            continue
        value = parse_decimal(value_for(row, "value"), default=Decimal("0"))
        shares = abs(parse_decimal(value_for(row, "shares"), default=Decimal("0")))
        row_type = normalize_text(value_for(row, "type")).lower()
        is_buy = row_type in {"buy", "kauf"} or (value < 0 and shares > 0)
        if not is_buy or shares <= 0:
            continue
        timestamp = normalize_text(value_for(row, "date"))
        record = alpaca_prices.get((isin, timestamp))
        if not record:
            statuses[isin].add("missing")
            continue
        status = str(record.get("status") or "missing")
        statuses[isin].add(status)
        price = record.get("price")
        if status == "matched" and price is not None:
            alpaca_weight[isin] += Decimal(str(price)) * shares
            alpaca_share_weight[isin] += shares
    for isin, item in stats.items():
        if alpaca_share_weight[isin] > 0:
            item["alpaca_weighted_price_at_buy_time"] = alpaca_weight[isin] / alpaca_share_weight[isin]
        values = statuses.get(isin, set())
        if "matched" in values and len(values) == 1:
            item["alpaca_status"] = "matched"
        elif "matched" in values:
            item["alpaca_status"] = "partial"
        elif values:
            item["alpaca_status"] = ",".join(sorted(values))
        else:
            item["alpaca_status"] = "no_buy_transactions"


def fetch_alpaca_execution_prices(
    transactions: Iterable[dict[str, Any]],
    isin_map: dict[str, dict[str, str]],
    *,
    window_minutes: int = 5,
) -> dict[tuple[str, str], dict[str, Any]]:
    from market_forecasting_engine.alpaca_broker import AlpacaPaperBroker

    broker = AlpacaPaperBroker()
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for row in transactions:
        isin = normalize_text(value_for(row, "isin")).upper()
        if not isin:
            continue
        value = parse_decimal(value_for(row, "value"), default=Decimal("0"))
        shares = abs(parse_decimal(value_for(row, "shares"), default=Decimal("0")))
        row_type = normalize_text(value_for(row, "type")).lower()
        is_buy = row_type in {"buy", "kauf"} or (value < 0 and shares > 0)
        if not is_buy or shares <= 0:
            continue
        timestamp_text = normalize_text(value_for(row, "date"))
        ticker = isin_map.get(isin, {}).get("alpaca_ticker") or isin_map.get(isin, {}).get("ticker")
        key = (isin, timestamp_text)
        if not ticker:
            output[key] = {"status": "missing_ticker"}
            continue
        if not _alpaca_stock_symbol_supported(ticker):
            output[key] = {"status": "unsupported_symbol", "ticker": ticker}
            continue
        timestamp = parse_trade_republic_timestamp(timestamp_text)
        start = (timestamp - timedelta(minutes=window_minutes)).isoformat().replace("+00:00", "Z")
        end = (timestamp + timedelta(minutes=window_minutes)).isoformat().replace("+00:00", "Z")
        try:
            bars = broker.stock_bars(ticker, start=start, end=end, timeframe="1Min", limit=100)
        except Exception as exc:
            output[key] = {"status": "alpaca_error", "ticker": ticker, "message": str(exc)}
            continue
        if not bars:
            output[key] = {"status": "no_bars", "ticker": ticker}
            continue
        nearest = min(bars, key=lambda bar: abs(parse_alpaca_timestamp(str(bar.get("t"))) - timestamp))
        output[key] = {
            "status": "matched",
            "ticker": ticker,
            "price": nearest.get("c"),
            "bar_timestamp": nearest.get("t"),
            "bar_open": nearest.get("o"),
            "bar_high": nearest.get("h"),
            "bar_low": nearest.get("l"),
            "bar_close": nearest.get("c"),
            "bar_volume": nearest.get("v"),
        }
    return output


def parse_trade_republic_timestamp(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_alpaca_timestamp(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    return datetime.fromisoformat(normalized).astimezone(timezone.utc)


def _alpaca_stock_symbol_supported(ticker: str) -> bool:
    return bool(ticker) and "." not in ticker and "/" not in ticker and ticker.upper() not in {"ETHUSD", "BTCUSD", "SOLUSD"}


def write_report(report: dict[str, Any], output: Path, format_: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if format_ == "json":
        output.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        return
    with output.open("w", newline="", encoding="utf-8") as file:
        rows = report["holdings"]
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["isin"])
        writer.writeheader()
        writer.writerows(rows)


def value_for(row: dict[str, Any], key: str) -> Any:
    normalized = {str(k).strip().lower().replace(" ", "_"): v for k, v in row.items()}
    aliases = {
        "date": ["date", "datum"],
        "type": ["type", "typ"],
        "value": ["value", "wert", "amount"],
        "isin": ["isin"],
        "shares": ["shares", "anteile", "quantity"],
    }
    for candidate in aliases[key]:
        if candidate in normalized:
            return normalized[candidate]
    return None


def parse_decimal(value: Any, default: Decimal | None = Decimal("0")) -> Decimal | None:
    if value is None or value == "":
        return default
    text = str(value).strip().replace("\u00a0", "").replace(" ", "")
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return default


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"null", "none", "n/a", "na"} else text


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
