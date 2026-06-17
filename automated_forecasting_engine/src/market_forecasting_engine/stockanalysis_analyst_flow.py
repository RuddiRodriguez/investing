from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import pandas as pd
from bs4 import BeautifulSoup


BASE_URL = "https://stockanalysis.com"
DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class VisibleAnalyst:
    rank: int
    name: str
    slug: str
    company: str
    sector: str
    success_rate: float | None
    average_return: float | None
    ratings: int | None
    last_rating: str | None
    url: str


@dataclass(frozen=True)
class AnalystTickerCandidate:
    analyst: VisibleAnalyst
    source_symbol: str
    forecast_symbol: str
    company_name: str
    rating_action: str
    rating: str
    price_target: str
    current_price: str
    upside: str
    updated: str
    source_url: str


def select_recent_visible_analyst_tickers(max_tickers: int = 1, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> list[AnalystTickerCandidate]:
    analysts = scrape_visible_analysts(timeout=timeout)
    candidates: list[AnalystTickerCandidate] = []
    for analyst in analysts:
        candidates.extend(scrape_visible_analyst_tickers(analyst, timeout=timeout))
        if len(candidates) >= max_tickers:
            break
    candidates = sorted(candidates, key=lambda item: (_date_key(item.updated), -item.analyst.rank), reverse=True)
    return candidates[:max_tickers]


def find_visible_analyst_ratings_for_symbol(
    symbol: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_analysts: int | None = None,
) -> list[dict[str, Any]]:
    """Find visible StockAnalysis analyst-page ratings that mention one ticker."""

    target = symbol.upper().strip()
    if not target:
        return []
    analysts = scrape_visible_analysts(timeout=timeout)
    if max_analysts is not None:
        analysts = analysts[: max(0, int(max_analysts))]
    matches: list[AnalystTickerCandidate] = []
    for analyst in analysts:
        try:
            candidates = scrape_visible_analyst_tickers(analyst, timeout=timeout)
        except Exception:
            continue
        for candidate in candidates:
            if candidate.source_symbol.upper().strip() == target or candidate.forecast_symbol.upper().strip() == target:
                matches.append(candidate)
    matches = sorted(matches, key=lambda item: (_date_key(item.updated), -item.analyst.rank), reverse=True)
    return [_candidate_dict(match) for match in matches]


def scrape_visible_analysts(*, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> list[VisibleAnalyst]:
    html = _get_html(f"{BASE_URL}/analysts/", timeout=timeout)
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return []
    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["td", "th"])]
        link = tr.find("a", href=True)
        if len(cells) < 8 or link is None:
            continue
        slug = link["href"].rstrip("/").split("/")[-1]
        rows.append(
            VisibleAnalyst(
                rank=int(_number(cells[0]) or 0),
                name=link.get_text(" ", strip=True),
                slug=slug,
                company=cells[2],
                sector=cells[3],
                success_rate=_percent(cells[4]),
                average_return=_percent(cells[5]),
                ratings=int(_number(cells[6]) or 0),
                last_rating=cells[7],
                url=urljoin(BASE_URL, link["href"]),
            )
        )
    return sorted(rows, key=lambda item: _date_key(item.last_rating), reverse=True)


def scrape_visible_analyst_tickers(analyst: VisibleAnalyst, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> list[AnalystTickerCandidate]:
    html = _get_html(analyst.url, timeout=timeout)
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    row_candidates = _candidates_from_table_rows(analyst, table)
    if row_candidates:
        return row_candidates
    links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if not href.startswith("/stocks/"):
            continue
        parts = href.strip("/").split("/")
        if len(parts) != 2 or parts[0] != "stocks":
            continue
        text = link.get_text(" ", strip=True)
        if not _looks_like_stock_symbol(text):
            continue
        links.append((text, href))
    seen: set[str] = set()
    candidates = []
    table_text = soup.find("table").get_text(" | ", strip=True) if soup.find("table") else ""
    for source_symbol, href in links:
        if source_symbol in seen:
            continue
        seen.add(source_symbol)
        candidates.append(_candidate_from_table_text(analyst, source_symbol, href, table_text))
    return candidates


def _candidates_from_table_rows(analyst: VisibleAnalyst, table: Any) -> list[AnalystTickerCandidate]:
    if table is None:
        return []
    candidates: list[AnalystTickerCandidate] = []
    for tr in table.find_all("tr")[1:]:
        cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["td", "th"])]
        if cells and not cells[0]:
            cells = cells[1:]
        if len(cells) < 7:
            continue
        link = tr.find("a")
        source_symbol = link.get_text(" ", strip=True) if link else _symbol_from_stock_cell(cells[0])
        if not _looks_like_stock_symbol(source_symbol):
            continue
        company_name = cells[0].replace(source_symbol, "", 1).strip()
        action_rating = cells[1].split()
        rating = action_rating[-1] if action_rating else ""
        rating_action = cells[1].replace(rating, "").strip()
        candidates.append(
            AnalystTickerCandidate(
                analyst=analyst,
                source_symbol=source_symbol,
                forecast_symbol=map_stockanalysis_symbol_to_forecast_symbol(source_symbol),
                company_name=company_name,
                rating_action=rating_action,
                rating=rating,
                price_target=cells[2],
                current_price=cells[3],
                upside=cells[4],
                updated=cells[6],
                source_url=urljoin(BASE_URL, link.get("href")) if link and link.get("href") else analyst.url,
            )
        )
    return candidates


def _symbol_from_stock_cell(value: str) -> str:
    parts = value.split()
    if len(parts) >= 2 and parts[0].endswith(":"):
        return f"{parts[0]} {parts[1]}"
    return parts[0] if parts else ""


def run_analyst_selected_forecasts(args: argparse.Namespace) -> dict[str, Any]:
    candidates = select_recent_visible_analyst_tickers(args.max_tickers, timeout=float(args.timeout))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for candidate in candidates:
        run_dir = output_root / _safe_path(candidate.forecast_symbol)
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "market_forecasting_engine.cli",
            "--ticker",
            candidate.forecast_symbol,
            "--provider",
            "yahoo",
            "--start",
            args.start,
            "--horizons",
            args.horizons,
            "--search-level",
            args.search_level,
            "--output-dir",
            str(run_dir),
        ]
        if args.end:
            command.extend(["--end", args.end])
        if args.calendar:
            command.extend(["--calendar", args.calendar])
        if args.llm_env_file:
            command.extend(["--llm-env-file", args.llm_env_file, "--long-term-source-env-file", args.llm_env_file])
        if args.disable_alt_news:
            command.append("--disable-alt-news")
        if args.disable_long_term_sources:
            command.append("--disable-long-term-sources")
        if args.disable_strategy_knowledge:
            command.append("--disable-strategy-knowledge")
        if args.strategy_knowledge_rebuild_index:
            command.append("--strategy-knowledge-rebuild-index")
        if args.strategy_knowledge_max_chunks:
            command.extend(["--strategy-knowledge-max-chunks", str(args.strategy_knowledge_max_chunks)])
        if args.dry_run:
            status = "dry_run"
            returncode = None
        else:
            completed = subprocess.run(command, cwd=args.project_dir, text=True, capture_output=True, check=False)
            status = "ok" if completed.returncode == 0 else "failed"
            returncode = completed.returncode
            (run_dir / "analyst_flow_stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (run_dir / "analyst_flow_stderr.txt").write_text(completed.stderr, encoding="utf-8")
        results.append(
            {
                "status": status,
                "returncode": returncode,
                "candidate": _candidate_dict(candidate),
                "run_dir": str(run_dir),
                "command": command,
            }
        )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "stockanalysis_visible_analyst_leaderboard",
        "results": results,
    }
    summary_path = output_root / "analyst_selected_forecast_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return summary


def _candidate_from_table_text(analyst: VisibleAnalyst, source_symbol: str, href: str, table_text: str) -> AnalystTickerCandidate:
    forecast_symbol = map_stockanalysis_symbol_to_forecast_symbol(source_symbol)
    # Public rows are pipe-delimited in visible order; this fallback stays auditable even when Pro rows are masked.
    row_text = ""
    marker = source_symbol
    start = table_text.find(marker)
    if start >= 0:
        next_start = table_text.find(" | ", start + len(marker))
        row_text = table_text[start : start + 400] if next_start >= 0 else table_text[start : start + 400]
    fields = [part.strip() for part in row_text.split("|") if part.strip()]
    company_name = fields[1] if len(fields) > 1 else ""
    rating_action = next((field for field in fields if field.endswith(":")), "")
    rating = next((field for field in fields if field in {"Buy", "Hold", "Sell", "Strong Buy", "Strong Sell"}), "")
    upside = next((field for field in fields if field.endswith("%") or field == "n/a"), "")
    updated = fields[-1] if fields and "," in fields[-1] else analyst.last_rating or ""
    price_target = next((field for field in fields if field.startswith("$")), "")
    current = ""
    dollar_fields = [field for field in fields if field.startswith("$")]
    if len(dollar_fields) >= 2:
        current = dollar_fields[-1]
    return AnalystTickerCandidate(
        analyst=analyst,
        source_symbol=source_symbol,
        forecast_symbol=forecast_symbol,
        company_name=company_name,
        rating_action=rating_action,
        rating=rating,
        price_target=price_target,
        current_price=current,
        upside=upside,
        updated=updated,
        source_url=urljoin(BASE_URL, href),
    )


def map_stockanalysis_symbol_to_forecast_symbol(symbol: str) -> str:
    clean = symbol.strip().upper()
    if clean.startswith("TSX: "):
        return clean.split(":", 1)[1].strip().replace(".", "-") + ".TO"
    if clean.startswith("TSXV: "):
        return clean.split(":", 1)[1].strip().replace(".", "-") + ".V"
    if ": " in clean:
        return clean.split(":", 1)[1].strip()
    return clean


def _looks_like_stock_symbol(value: str) -> bool:
    clean = value.strip()
    if not clean:
        return False
    if clean.startswith(("TSX: ", "TSXV: ")):
        return True
    return clean.upper() == clean and clean.replace(".", "").replace("-", "").isalnum() and 1 <= len(clean) <= 8


def _candidate_dict(candidate: AnalystTickerCandidate) -> dict[str, Any]:
    return {
        "analyst": candidate.analyst.__dict__,
        "source_symbol": candidate.source_symbol,
        "forecast_symbol": candidate.forecast_symbol,
        "company_name": candidate.company_name,
        "rating_action": candidate.rating_action,
        "rating": candidate.rating,
        "price_target": candidate.price_target,
        "current_price": candidate.current_price,
        "upside": candidate.upside,
        "updated": candidate.updated,
        "source_url": candidate.source_url,
    }


def _get_html(url: str, *, timeout: float) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; market-forecasting-engine/0.1; +local)"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _date_key(value: str | None) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce")


def _number(value: str) -> float | None:
    clean = value.replace(",", "").replace("$", "").strip()
    try:
        return float(clean)
    except ValueError:
        return None


def _percent(value: str) -> float | None:
    number = _number(value.replace("%", ""))
    return number / 100.0 if number is not None else None


def _safe_path(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_") or "ticker"


def main() -> int:
    parser = argparse.ArgumentParser(description="Select visible StockAnalysis analyst tickers and run the normal forecast suite.")
    parser.add_argument("--project-dir", default="/Users/ruddigarcia/Projects/invest")
    parser.add_argument("--output-root", default="automated_forecasting_engine/runs/stockanalysis_analyst_selected")
    parser.add_argument("--max-tickers", type=int, default=1)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--horizons", default="1,5,10")
    parser.add_argument("--calendar", default=None)
    parser.add_argument("--search-level", choices=("fast", "expanded"), default="fast")
    parser.add_argument("--llm-env-file", default=None)
    parser.add_argument("--disable-alt-news", action="store_true")
    parser.add_argument("--disable-long-term-sources", action="store_true")
    parser.add_argument("--disable-strategy-knowledge", action="store_true")
    parser.add_argument("--strategy-knowledge-rebuild-index", action="store_true")
    parser.add_argument("--strategy-knowledge-max-chunks", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = run_analyst_selected_forecasts(args)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
