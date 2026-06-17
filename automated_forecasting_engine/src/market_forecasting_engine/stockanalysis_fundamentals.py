from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd
from bs4 import BeautifulSoup

from market_forecasting_engine.stockanalysis_analyst_flow import find_visible_analyst_ratings_for_symbol


STOCKANALYSIS_BASE_URL = "https://stockanalysis.com"
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_TRANSCRIPT_DETAIL_PAGES = 3
MAX_VISIBLE_ANALYST_PAGES = 50


class StockAnalysisFundamentalsClient:
    def __init__(
        self,
        *,
        base_url: str = STOCKANALYSIS_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def fetch_html(self, path: str) -> str:
        url = f"{self.base_url}/{path.lstrip('/')}"
        return _get_html(url, timeout=self.timeout)

    def fetch_company_context(self, symbol: str) -> dict[str, Any]:
        slug = normalize_stockanalysis_symbol(symbol)
        endpoints = {
            "overview": f"stocks/{quote(slug)}/",
            "financials": f"stocks/{quote(slug)}/financials/",
            "forecast": f"stocks/{quote(slug)}/forecast/",
            "statistics": f"stocks/{quote(slug)}/statistics/",
            "dividend": f"stocks/{quote(slug)}/dividend/",
            "company": f"stocks/{quote(slug)}/company/",
            "transcripts": f"stocks/{quote(slug)}/transcripts/",
            "filings": f"stocks/{quote(slug)}/filings/",
        }
        raw: dict[str, Any] = {}
        endpoint_status: dict[str, Any] = {}
        for name, path in endpoints.items():
            try:
                html = self.fetch_html(path)
                raw[name] = html
                endpoint_status[name] = _html_status(html)
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                raw[name] = None
                endpoint_status[name] = {"status": "error", "error": _safe_error(exc)}
        transcript_detail_paths = _transcript_detail_paths(_soup(raw.get("transcripts")))
        transcript_details: dict[str, str | None] = {}
        for index, path in enumerate(transcript_detail_paths[:MAX_TRANSCRIPT_DETAIL_PAGES], start=1):
            endpoint_name = f"transcript_detail_{index}"
            try:
                html = self.fetch_html(path)
                transcript_details[path] = html
                endpoint_status[endpoint_name] = {**_html_status(html), "path": path}
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                transcript_details[path] = None
                endpoint_status[endpoint_name] = {"status": "error", "path": path, "error": _safe_error(exc)}
        if transcript_details:
            raw["transcript_details"] = transcript_details

        try:
            visible_analyst_ratings = find_visible_analyst_ratings_for_symbol(
                slug.upper(),
                timeout=self.timeout,
                max_analysts=MAX_VISIBLE_ANALYST_PAGES,
            )
            raw["visible_analyst_ratings"] = visible_analyst_ratings
            endpoint_status["visible_analyst_ratings"] = {
                "status": "ok",
                "items": len(visible_analyst_ratings),
                "source": f"{self.base_url}/analysts/",
                "max_analyst_pages": MAX_VISIBLE_ANALYST_PAGES,
            }
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            raw["visible_analyst_ratings"] = []
            endpoint_status["visible_analyst_ratings"] = {"status": "error", "error": _safe_error(exc)}

        normalized = normalize_stockanalysis_context(slug, raw)
        return {
            "symbol": symbol.upper().strip(),
            "stockanalysis_symbol": slug,
            "source": "stockanalysis",
            "source_base_url": self.base_url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "endpoint_status": endpoint_status,
            "normalized": normalized,
            "raw": raw,
            "raw_sha256": _json_sha256(raw),
        }


def normalize_stockanalysis_context(symbol: str, raw: dict[str, Any]) -> dict[str, Any]:
    overview = _soup(raw.get("overview"))
    financials = _soup(raw.get("financials"))
    forecast = _soup(raw.get("forecast"))
    statistics = _soup(raw.get("statistics"))
    dividend = _soup(raw.get("dividend"))
    company = _soup(raw.get("company"))
    transcripts = _soup(raw.get("transcripts"))
    filings = _soup(raw.get("filings"))

    overview_metrics = _label_value_pairs(overview)
    statistics_metrics = _label_value_pairs(statistics)
    dividend_metrics = _label_value_pairs(dividend)
    company_metrics = _label_value_pairs(company)
    forecast_summary = _forecast_summary(forecast)
    financial_rows = _financial_rows(raw.get("financials"))

    return {
        "identity": _compact_dict(
            {
                "symbol": symbol.upper(),
                "name": _page_company_name(overview) or _page_company_name(company),
                "exchange": _exchange_from_page(overview) or company_metrics.get("Exchange"),
                "currency": "USD" if "USD" in _text(overview)[:600] else company_metrics.get("Reporting Currency"),
                "sector": company_metrics.get("Sector") or _linked_metric(company, "Sector"),
                "industry": company_metrics.get("Industry") or _linked_metric(company, "Industry"),
                "country": company_metrics.get("Country"),
                "website": company_metrics.get("Website"),
                "description": _company_description(company) or _about_text(overview),
                "ceo": company_metrics.get("CEO"),
                "employees": _parse_number(company_metrics.get("Employees")),
                "ipo_date": company_metrics.get("IPO Date"),
                "cik": company_metrics.get("CIK Code"),
                "isin": company_metrics.get("ISIN Number"),
            }
        ),
        "quote": _compact_dict(
            {
                "price": _parse_number(_price_from_page(overview)),
                "date": _quote_timestamp(overview),
                "volume": _parse_number(overview_metrics.get("Volume")),
                "open": _parse_number(overview_metrics.get("Open")),
                "previous_close": _parse_number(overview_metrics.get("Previous Close")),
                "market_cap": _parse_number(overview_metrics.get("Market Cap")),
            }
        ),
        "valuation": _compact_dict(
            {
                "market_cap": _parse_number(statistics_metrics.get("Market Cap") or overview_metrics.get("Market Cap")),
                "enterprise_value": _parse_number(statistics_metrics.get("Enterprise Value")),
                "pe": _parse_number(statistics_metrics.get("PE Ratio") or overview_metrics.get("PE Ratio")),
                "forward_pe": _parse_number(statistics_metrics.get("Forward PE") or overview_metrics.get("Forward PE")),
                "pb": _parse_number(statistics_metrics.get("PB Ratio")),
                "ps": _parse_number(statistics_metrics.get("PS Ratio")),
                "pfcf": _parse_number(statistics_metrics.get("P/FCF Ratio")),
                "ev_to_ebitda": _parse_number(statistics_metrics.get("EV / EBITDA")),
                "ev_to_sales": _parse_number(statistics_metrics.get("EV / Sales")),
            }
        ),
        "income_statement": _compact_dict(
            {
                "date": _latest_financial_period(financial_rows),
                "revenue": _financial_value(financial_rows, "Revenue"),
                "gross_profit": _financial_value(financial_rows, "Gross Profit"),
                "operating_income": _financial_value(financial_rows, "Operating Income"),
                "net_income": _financial_value(financial_rows, "Net Income"),
                "eps": _financial_value(financial_rows, "EPS (Basic)"),
                "eps_diluted": _financial_value(financial_rows, "EPS (Diluted)"),
                "ebitda": _financial_value(financial_rows, "EBITDA"),
            }
        ),
        "cash_flow": _compact_dict(
            {
                "date": _latest_financial_period(financial_rows),
                "free_cash_flow": _financial_value(financial_rows, "Free Cash Flow"),
                "free_cash_flow_per_share": _financial_value(financial_rows, "Free Cash Flow Per Share"),
            }
        ),
        "profitability": _compact_dict(
            {
                "gross_margin": _parse_percent(_financial_value_raw(financial_rows, "Gross Margin")),
                "operating_margin": _parse_percent(_financial_value_raw(financial_rows, "Operating Margin")),
                "net_margin": _parse_percent(_financial_value_raw(financial_rows, "Profit Margin")),
                "fcf_margin": _parse_percent(_financial_value_raw(financial_rows, "FCF Margin")),
                "roic": _parse_percent(statistics_metrics.get("Return on Invested Capital (ROIC)")),
                "roce": _parse_percent(statistics_metrics.get("Return on Capital Employed (ROCE)")),
            }
        ),
        "financial_position": _compact_dict(
            {
                "current_ratio": _parse_number(statistics_metrics.get("Current Ratio")),
                "quick_ratio": _parse_number(statistics_metrics.get("Quick Ratio")),
                "debt_to_equity": _parse_number(statistics_metrics.get("Debt / Equity")),
                "debt_to_ebitda": _parse_number(statistics_metrics.get("Debt / EBITDA")),
                "interest_coverage": _parse_number(statistics_metrics.get("Interest Coverage")),
            }
        ),
        "analyst": _compact_dict(
            {
                "rating": forecast_summary.get("rating") or overview_metrics.get("Analysts"),
                "analyst_count": _parse_number(forecast_summary.get("analyst_count")),
                "target_consensus": _parse_number(forecast_summary.get("target_average") or overview_metrics.get("Price Target")),
                "target_low": _parse_number(forecast_summary.get("target_low")),
                "target_median": _parse_number(forecast_summary.get("target_median")),
                "target_high": _parse_number(forecast_summary.get("target_high")),
                "latest_ratings": _latest_forecast_rows(forecast),
                "visible_analyst_ratings": _visible_analyst_ratings(raw.get("visible_analyst_ratings")),
            }
        ),
        "dividends": _compact_dict(
            {
                "annual_dividend": _parse_number(dividend_metrics.get("Annual Dividend")),
                "dividend_yield": _parse_percent(dividend_metrics.get("Dividend Yield")),
                "ex_dividend_date": dividend_metrics.get("Ex-Dividend Date"),
                "payout_ratio": _parse_percent(dividend_metrics.get("Payout Ratio")),
                "buyback_yield": _parse_percent(dividend_metrics.get("Buyback Yield")),
                "shareholder_yield": _parse_percent(dividend_metrics.get("Shareholder Yield")),
                "history": _dividend_history(raw.get("dividend")),
            }
        ),
        "transcripts": {
            "items": _transcript_items(transcripts, raw.get("transcript_details") if isinstance(raw.get("transcript_details"), dict) else {}),
        },
        "filings": {
            "items": _filing_items(filings),
            "latest_sec_filings": _company_latest_sec_filings(company),
        },
        "recent_news": _overview_news(overview),
        "data_quality": _compact_dict(
            {
                "financials_last_updated": _after_label_text(financials, "Last updated:"),
                "financials_last_checked": _after_label_text(financials, "Last checked:"),
                "company_last_updated": _after_label_text(company, "Last updated:"),
                "dividend_last_updated": _after_label_text(dividend, "Last updated:"),
                "source_note": "HTML-derived public StockAnalysis pages; tables and visible sections only.",
            }
        ),
    }


def normalize_stockanalysis_symbol(symbol: str) -> str:
    clean = str(symbol or "").strip().lower()
    if not clean:
        raise ValueError("symbol is required.")
    return clean.split(".")[0].replace("_", "-")


def _financial_rows(html: str | None) -> dict[str, dict[str, Any]]:
    if not html:
        return {}
    tables = _html_tables(html)
    if not tables:
        return {}
    table = tables[0]
    if table.empty:
        return {}
    first_column = table.columns[0]
    result: dict[str, dict[str, Any]] = {}
    if re.search(r"Financials in millions", html, re.I):
        result["__units_multiplier__"] = {"value": 1_000_000}
    periods = [str(column) for column in table.columns[1:]]
    for _, row in table.iterrows():
        label = str(row.get(first_column, "")).strip()
        if not label:
            continue
        result[label] = {
            "periods": periods,
            "values": [None if pd.isna(row.get(column)) else row.get(column) for column in table.columns[1:]],
        }
    return result


def _latest_financial_period(rows: dict[str, dict[str, Any]]) -> str | None:
    fiscal = rows.get("Fiscal Year") or rows.get("Period Ending")
    if not fiscal:
        return None
    values = fiscal.get("values") or []
    return str(values[0]) if values else None


def _financial_value(rows: dict[str, dict[str, Any]], label: str) -> float | None:
    value = _parse_number(_financial_value_raw(rows, label))
    if value is None:
        return None
    if _financial_value_should_scale(label):
        return value * float(rows.get("__units_multiplier__", {}).get("value", 1))
    return value


def _financial_value_should_scale(label: str) -> bool:
    clean = label.lower()
    if any(token in clean for token in ("margin", "growth", "ratio", "per share", "eps", "shares")):
        return False
    return True


def _financial_value_raw(rows: dict[str, dict[str, Any]], label: str) -> Any:
    row = rows.get(label)
    values = row.get("values") if isinstance(row, dict) else None
    if not values:
        return None
    return values[0]


def _forecast_summary(soup: BeautifulSoup) -> dict[str, Any]:
    text = _text(soup)
    result: dict[str, Any] = {}
    match = re.search(r"According to\s+(\d+)\s+analysts?.*?consensus rating of \"([^\"]+)\".*?average price target of \$?([0-9.,]+)", text, re.I)
    if match:
        result["analyst_count"] = match.group(1)
        result["rating"] = match.group(2)
        result["target_average"] = match.group(3)
    target_match = re.search(r"Target Low Average Median High Price\$?([0-9.,]+)\$?([0-9.,]+)\$?([0-9.,]+)\$?([0-9.,]+)", text)
    if target_match:
        result["target_low"] = target_match.group(1)
        result["target_average"] = target_match.group(2)
        result["target_median"] = target_match.group(3)
        result["target_high"] = target_match.group(4)
    return result


def _latest_forecast_rows(soup: BeautifulSoup) -> list[dict[str, Any]]:
    text = _text(soup)
    rows = []
    pattern = re.compile(r"([A-Z][A-Za-z .'-]+)\s+([A-Za-z .&]+)\s+(Buy|Hold|Sell|Strong Buy|Strong Sell)\s+(Maintains|Initiates|Reiterates|Downgrades|Upgrades)\s+([$0-9., →]+)\s*([+-]?[0-9.]+%)\s*([A-Z][a-z]{2} \d{1,2}, \d{4})")
    for match in pattern.finditer(text):
        rows.append(
            {
                "analyst": match.group(1).strip(),
                "firm": match.group(2).strip(),
                "rating": match.group(3),
                "action": match.group(4),
                "price_target": match.group(5).strip(),
                "upside": match.group(6),
                "date": match.group(7),
            }
        )
        if len(rows) >= 10:
            break
    return rows


def _visible_analyst_ratings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows = []
    for item in value[:20]:
        if not isinstance(item, dict):
            continue
        analyst = item.get("analyst") if isinstance(item.get("analyst"), dict) else {}
        rows.append(
            _compact_dict(
                {
                    "analyst_name": analyst.get("name"),
                    "analyst_rank": analyst.get("rank"),
                    "analyst_company": analyst.get("company"),
                    "analyst_sector": analyst.get("sector"),
                    "analyst_success_rate": analyst.get("success_rate"),
                    "analyst_average_return": analyst.get("average_return"),
                    "source_symbol": item.get("source_symbol"),
                    "company_name": item.get("company_name"),
                    "rating_action": item.get("rating_action"),
                    "rating": item.get("rating"),
                    "price_target": item.get("price_target"),
                    "current_price": item.get("current_price"),
                    "upside": item.get("upside"),
                    "updated": item.get("updated"),
                    "source_url": item.get("source_url") or analyst.get("url"),
                    "analyst_url": analyst.get("url"),
                }
            )
        )
    return rows


def _dividend_history(html: str | None) -> list[dict[str, Any]]:
    if not html:
        return []
    tables = _html_tables(html)
    if not tables:
        return []
    records = []
    for row in tables[0].head(10).to_dict("records"):
        records.append({str(key): value for key, value in row.items() if not pd.isna(value)})
    return records


def _html_tables(html: str) -> list[pd.DataFrame]:
    soup = _soup(html)
    frames: list[pd.DataFrame] = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)
        if not rows:
            continue
        width = max(len(row) for row in rows)
        normalized_rows = [row + [""] * (width - len(row)) for row in rows]
        header = normalized_rows[0]
        body = normalized_rows[1:]
        if not body:
            continue
        frames.append(pd.DataFrame(body, columns=header))
    return frames


def _transcript_detail_paths(soup: BeautifulSoup) -> list[str]:
    paths = []
    seen: set[str] = set()
    for link in soup.find_all("a"):
        title = link.get_text(" ", strip=True)
        href = link.get("href")
        if "Earnings Call:" not in title or not href:
            continue
        path = str(href).strip()
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def _transcript_items(soup: BeautifulSoup, detail_html_by_path: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    items = []
    detail_html_by_path = detail_html_by_path or {}
    for link in soup.find_all("a"):
        title = link.get_text(" ", strip=True)
        href = link.get("href")
        if "Earnings Call:" not in title or not href:
            continue
        parent = link.find_parent()
        summary = parent.get_text(" ", strip=True) if parent else ""
        detail_html = detail_html_by_path.get(str(href).strip())
        detail = _transcript_detail_context(detail_html) if detail_html else {}
        items.append(
            _compact_dict({
                "title": title,
                "url": _absolute_url(href),
                "summary": _clean_space(summary.replace(title, ""))[:800],
                "detail_excerpt": detail.get("excerpt"),
                "prepared_remarks_excerpt": detail.get("prepared_remarks_excerpt"),
                "qa_excerpt": detail.get("qa_excerpt"),
                "detail_word_count": detail.get("word_count"),
            })
        )
    return items[:10]


def _transcript_detail_context(html: str | None) -> dict[str, Any]:
    if not html:
        return {}
    soup = _soup(html)
    article = soup.find("article") or soup.find("main") or soup.body or soup
    text = _clean_space(article.get_text(" ", strip=True))
    if not text:
        return {}
    prepared = _section_excerpt(text, ["Prepared Remarks", "Operator", "Presentation"], ["Question-and-Answer", "Questions and Answers", "Q&A"])
    qa = _section_excerpt(text, ["Question-and-Answer", "Questions and Answers", "Q&A"], [])
    return _compact_dict(
        {
            "excerpt": text[:2500],
            "prepared_remarks_excerpt": prepared,
            "qa_excerpt": qa,
            "word_count": len(text.split()),
        }
    )


def _section_excerpt(text: str, starts: list[str], stops: list[str]) -> str | None:
    lower = text.lower()
    start_idx = None
    for marker in starts:
        idx = lower.find(marker.lower())
        if idx >= 0 and (start_idx is None or idx < start_idx):
            start_idx = idx
    if start_idx is None:
        return None
    end_idx = len(text)
    for marker in stops:
        idx = lower.find(marker.lower(), start_idx + 1)
        if idx >= 0:
            end_idx = min(end_idx, idx)
    return text[start_idx:end_idx][:1800]


def _filing_items(soup: BeautifulSoup) -> list[dict[str, Any]]:
    text = _text(soup)
    matches = re.findall(r"((?:Q\d \d{4}|Registration Filing)\s+-\s+.+?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4})", text)
    return [{"title": _clean_space(match)} for match in matches[:20]]


def _company_latest_sec_filings(soup: BeautifulSoup) -> list[dict[str, Any]]:
    for table in _html_tables(str(soup)):
        columns = {str(column).lower(): column for column in table.columns}
        if {"date", "type", "title"}.issubset(columns):
            return [
                {
                    "date": str(row[columns["date"]]),
                    "type": str(row[columns["type"]]),
                    "title": str(row[columns["title"]]),
                }
                for _, row in table.head(20).iterrows()
            ]
    items = []
    text = _text(soup)
    pattern = re.compile(r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4})\s+([A-Z0-9/-]+)\s+(.+?)(?=(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}|Data Sources:|$)")
    for match in pattern.finditer(text):
        items.append({"date": match.group(1), "type": match.group(2), "title": _clean_space(match.group(3))})
        if len(items) >= 20:
            break
    return items


def _overview_news(soup: BeautifulSoup) -> list[dict[str, Any]]:
    items = []
    for link in soup.find_all("a"):
        href = link.get("href")
        title = link.get_text(" ", strip=True)
        if not href or not title or "/news/" not in href:
            continue
        items.append({"title": title, "url": _absolute_url(href)})
        if len(items) >= 10:
            break
    return items


def _label_value_pairs(soup: BeautifulSoup) -> dict[str, str]:
    text = _text(soup)
    labels = [
        "Market Cap",
        "Revenue (ttm)",
        "Net Income",
        "EPS",
        "Shares Out",
        "PE Ratio",
        "Forward PE",
        "Dividend",
        "Ex-Dividend Date",
        "Volume",
        "Open",
        "Previous Close",
        "Price Target",
        "Analysts",
        "Enterprise Value",
        "Earnings Date",
        "Shares Outstanding",
        "Shares Change (YoY)",
        "Owned by Insiders (%)",
        "Owned by Institutions (%)",
        "Float",
        "PS Ratio",
        "Forward PS",
        "PB Ratio",
        "P/TBV Ratio",
        "P/FCF Ratio",
        "P/OCF Ratio",
        "EV / Sales",
        "EV / EBITDA",
        "EV / EBIT",
        "EV / FCF",
        "Current Ratio",
        "Quick Ratio",
        "Debt / Equity",
        "Debt / EBITDA",
        "Debt / FCF",
        "Interest Coverage",
        "Return on Invested Capital (ROIC)",
        "Return on Capital Employed (ROCE)",
        "Employee Count",
        "Annual Dividend",
        "Dividend Yield",
        "Payout Ratio",
        "Buyback Yield",
        "Shareholder Yield",
        "Country",
        "Founded",
        "IPO Date",
        "Industry",
        "Sector",
        "Employees",
        "CEO",
        "Website",
        "Exchange",
        "Fiscal Year",
        "Reporting Currency",
        "IPO Price",
        "CIK Code",
        "ISIN Number",
    ]
    result = _pairs_from_two_column_tables(soup)
    for index, label in enumerate(labels):
        if label in result:
            continue
        next_labels = labels[:index] + labels[index + 1 :]
        value = _value_after_label(text, label, next_labels)
        if value:
            result[label] = value
    return result


def _pairs_from_two_column_tables(soup: BeautifulSoup) -> dict[str, str]:
    result: dict[str, str] = {}
    for table in _html_tables(str(soup)):
        if table.shape[1] != 2:
            continue
        columns = [str(column).strip() for column in table.columns]
        if columns[0] and columns[1]:
            result.setdefault(columns[0], columns[1])
        for _, row in table.iterrows():
            key = str(row.iloc[0]).strip()
            value = str(row.iloc[1]).strip()
            if key and value and key.lower() != "nan" and value.lower() != "nan":
                result.setdefault(key, value)
    return result


def _value_after_label(text: str, label: str, labels: list[str]) -> str | None:
    start = text.find(label)
    if start < 0:
        return None
    start += len(label)
    end = len(text)
    for other in labels:
        pos = text.find(other, start)
        if pos >= 0:
            end = min(end, pos)
    value = _clean_space(text[start:end])
    return value[:120] if value else None


def _page_company_name(soup: BeautifulSoup) -> str | None:
    h1 = soup.find("h1")
    if not h1:
        return None
    text = h1.get_text(" ", strip=True)
    return text.replace(" Corp. (", " Corp. (").strip() if text and "Income Statement" not in text else None


def _exchange_from_page(soup: BeautifulSoup) -> str | None:
    match = re.search(r"\b([A-Z]+):\s+[A-Z0-9.-]+\s+·", _text(soup))
    return match.group(1) if match else None


def _price_from_page(soup: BeautifulSoup) -> str | None:
    text = _text(soup)
    match = re.search(r"Real-Time Price\s+·\s+[A-Z]{3}\s+[^0-9]*([0-9]+(?:\.[0-9]+)?)", text)
    return match.group(1) if match else None


def _quote_timestamp(soup: BeautifulSoup) -> str | None:
    match = re.search(r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}\s+[AP]M\s+EDT)", _text(soup))
    return match.group(1) if match else None


def _company_description(soup: BeautifulSoup) -> str | None:
    h1 = soup.find("h1")
    if not h1:
        return None
    paragraphs = []
    for sibling in h1.find_all_next(["p", "h2"], limit=8):
        if sibling.name == "h2":
            break
        text = sibling.get_text(" ", strip=True)
        if text:
            paragraphs.append(text)
    return _clean_space(" ".join(paragraphs))[:1600] if paragraphs else None


def _about_text(soup: BeautifulSoup) -> str | None:
    h2 = next((tag for tag in soup.find_all("h2") if "About" in tag.get_text(" ", strip=True)), None)
    if not h2:
        return None
    paragraphs = []
    for sibling in h2.find_all_next(["p", "h2"], limit=5):
        if sibling.name == "h2":
            break
        text = sibling.get_text(" ", strip=True)
        if text:
            paragraphs.append(text)
    return _clean_space(" ".join(paragraphs))[:1200] if paragraphs else None


def _linked_metric(soup: BeautifulSoup, label: str) -> str | None:
    text = _text(soup)
    match = re.search(fr"{re.escape(label)}\s+([A-Za-z &,-]+)", text)
    return _clean_space(match.group(1)) if match else None


def _after_label_text(soup: BeautifulSoup, label: str) -> str | None:
    text = _text(soup)
    pos = text.find(label)
    if pos < 0:
        return None
    return _clean_space(text[pos + len(label) : pos + len(label) + 80])


def _soup(html: str | None) -> BeautifulSoup:
    return BeautifulSoup(html or "", "html.parser")


def _text(soup: BeautifulSoup) -> str:
    return soup.get_text(" ", strip=True)


def _clean_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _parse_percent(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value)
    if "%" not in text:
        return _parse_number(value)
    number = _parse_number(text.replace("%", ""))
    return number / 100.0 if number is not None else None


def _parse_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "nan", "-", "--"}:
        return None
    match = re.search(r"[-+]?\$?([0-9][0-9,]*\.?[0-9]*)\s*([KMBT]?)", text, re.I)
    if not match:
        return None
    number = float(match.group(1).replace(",", ""))
    suffix = match.group(2).upper()
    multiplier = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}.get(suffix, 1)
    if text.strip().startswith("-"):
        number *= -1
    return number * multiplier


def _compact_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None and value != "" and value != []}


def _absolute_url(href: str) -> str:
    if href.startswith("http"):
        return href
    return f"{STOCKANALYSIS_BASE_URL}/{href.lstrip('/')}"


def _html_status(html: str | None) -> dict[str, Any]:
    if not html:
        return {"status": "empty", "bytes": 0}
    soup = _soup(html)
    return {"status": "ok", "bytes": len(html.encode("utf-8")), "tables": len(soup.find_all("table"))}


def _safe_error(exc: BaseException) -> str:
    if isinstance(exc, HTTPError):
        return f"HTTP {exc.code}"
    return f"{exc.__class__.__name__}: {str(exc)[:200]}"


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _get_html(url: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; market-forecasting-engine/0.1; +local)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and normalize public StockAnalysis HTML data for one ticker.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--include-raw", action="store_true")
    args = parser.parse_args()

    result = StockAnalysisFundamentalsClient().fetch_company_context(args.symbol)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
    display = result if args.include_raw else {key: value for key, value in result.items() if key != "raw"}
    print(json.dumps(display, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
