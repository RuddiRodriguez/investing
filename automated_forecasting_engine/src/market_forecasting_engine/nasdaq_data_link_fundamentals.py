from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


NASDAQ_DATA_LINK_BASE_URL = "https://data.nasdaq.com/api/v3"
DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class NasdaqDataLinkTable:
    name: str
    code: str
    params: dict[str, Any]


class NasdaqDataLinkFundamentalsClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        env_file: str | None = None,
        base_url: str = NASDAQ_DATA_LINK_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        _load_env_file(env_file)
        self.api_key = api_key or os.getenv("NASDAQ_DATA_LINK_API_KEY") or os.getenv("NASDAQ_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("NASDAQ_DATA_LINK_API_KEY is required in the environment or .env file.")

    def fetch_datatable(self, table_code: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        query = {key: value for key, value in (params or {}).items() if value is not None}
        query["api_key"] = self.api_key
        url = f"{self.base_url}/datatables/{table_code}.json?{urlencode(query)}"
        return _get_json(url, timeout=self.timeout)

    def fetch_company_context(self, symbol: str, *, per_page: int = 10000) -> dict[str, Any]:
        symbol = symbol.upper().strip()
        if not symbol:
            raise ValueError("symbol is required.")
        tables = [
            NasdaqDataLinkTable("zacks_master", "ZACKS/MT", {"ticker": symbol, "qopts.per_page": per_page}),
            NasdaqDataLinkTable("zacks_fundamentals", "ZACKS/FC", {"ticker": symbol, "qopts.per_page": per_page}),
            NasdaqDataLinkTable("zacks_ratios", "ZACKS/FR", {"ticker": symbol, "qopts.per_page": per_page}),
            NasdaqDataLinkTable("zacks_market_value", "ZACKS/MKTV", {"ticker": symbol, "qopts.per_page": per_page}),
            NasdaqDataLinkTable("zacks_eps_estimates", "ZACKS/EE", {"ticker": symbol, "qopts.per_page": per_page}),
            NasdaqDataLinkTable("zacks_eps_surprises", "ZACKS/ES", {"ticker": symbol, "qopts.per_page": per_page}),
        ]

        raw: dict[str, Any] = {}
        endpoint_status: dict[str, Any] = {}
        for table in tables:
            try:
                payload = self.fetch_datatable(table.code, table.params)
                raw[table.name] = payload
                endpoint_status[table.name] = _datatable_status(payload)
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                raw[table.name] = None
                endpoint_status[table.name] = {"status": "error", "error": _safe_error(exc), "table": table.code}

        normalized = normalize_nasdaq_data_link_context(symbol, raw)
        return {
            "symbol": symbol,
            "source": "nasdaq_data_link",
            "source_base_url": self.base_url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "api_key_fingerprint": _key_fingerprint(self.api_key),
            "endpoint_status": endpoint_status,
            "normalized": normalized,
            "raw": raw,
            "raw_sha256": _json_sha256(raw),
        }


def normalize_nasdaq_data_link_context(symbol: str, raw: dict[str, Any]) -> dict[str, Any]:
    master_rows = _datatable_rows(raw.get("zacks_master"))
    fundamentals_rows = _datatable_rows(raw.get("zacks_fundamentals"))
    ratios_rows = _datatable_rows(raw.get("zacks_ratios"))
    market_value_rows = _datatable_rows(raw.get("zacks_market_value"))
    eps_estimate_rows = _datatable_rows(raw.get("zacks_eps_estimates"))
    eps_surprise_rows = _datatable_rows(raw.get("zacks_eps_surprises"))

    master = _first(master_rows)
    latest_annual = _latest_by_date([row for row in fundamentals_rows if row.get("per_type") == "A"], "per_end_date")
    latest_quarter = _latest_by_date([row for row in fundamentals_rows if row.get("per_type") == "Q"], "per_end_date")
    latest_ratios_annual = _latest_by_date([row for row in ratios_rows if row.get("per_type") == "A"], "per_end_date")
    latest_ratios_quarter = _latest_by_date([row for row in ratios_rows if row.get("per_type") == "Q"], "per_end_date")
    latest_market_value = _latest_by_date(market_value_rows, "per_end_date")
    latest_annual_estimate = _latest_by_date([row for row in eps_estimate_rows if row.get("per_type") == "A"], "per_end_date")
    latest_quarter_estimate = _latest_by_date([row for row in eps_estimate_rows if row.get("per_type") == "Q"], "per_end_date")
    latest_surprise = _latest_by_date(eps_surprise_rows, "act_rpt_date", fallback_date_key="per_end_date")

    return {
        "identity": _compact_dict(
            {
                "symbol": symbol,
                "name": master.get("comp_name_2") or master.get("comp_name"),
                "exchange": master.get("exchange"),
                "currency": master.get("currency_code"),
                "active_ticker_flag": master.get("active_ticker_flag"),
                "website": master.get("comp_url"),
                "sic_code": master.get("sic_4_code"),
                "sic_description": master.get("sic_4_desc"),
                "zacks_sector": master.get("zacks_x_sector_desc"),
                "zacks_industry": master.get("zacks_x_ind_desc") or master.get("zacks_m_ind_desc"),
                "cik": master.get("comp_cik"),
                "country": master.get("country_name"),
                "sp500_member": master.get("sp500_member_flag"),
                "optionable": master.get("optionable_flag"),
                "asset_type": master.get("asset_type"),
            }
        ),
        "latest_annual_fundamentals": _normalize_financial_row(latest_annual),
        "latest_quarter_fundamentals": _normalize_financial_row(latest_quarter),
        "latest_annual_ratios": _normalize_ratio_row(latest_ratios_annual),
        "latest_quarter_ratios": _normalize_ratio_row(latest_ratios_quarter),
        "market_value": _compact_dict(
            {
                "period_end": latest_market_value.get("per_end_date"),
                "period_type": latest_market_value.get("per_type"),
                "market_value": latest_market_value.get("mkt_val"),
                "enterprise_value": latest_market_value.get("ep_val"),
            }
        ),
        "eps_estimates": {
            "annual": _normalize_eps_estimate(latest_annual_estimate),
            "quarter": _normalize_eps_estimate(latest_quarter_estimate),
        },
        "latest_eps_surprise": _normalize_eps_surprise(latest_surprise),
        "coverage": {
            "zacks_master_rows": len(master_rows),
            "zacks_fundamental_rows": len(fundamentals_rows),
            "zacks_ratio_rows": len(ratios_rows),
            "zacks_market_value_rows": len(market_value_rows),
            "zacks_eps_estimate_rows": len(eps_estimate_rows),
            "zacks_eps_surprise_rows": len(eps_surprise_rows),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and normalize Nasdaq Data Link ZACKS data for one ticker.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--per-page", type=int, default=10000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--include-raw", action="store_true", help="Print raw endpoint payloads in stdout output.")
    args = parser.parse_args(argv)

    client = NasdaqDataLinkFundamentalsClient(env_file=args.env_file)
    result = client.fetch_company_context(args.symbol, per_page=args.per_page)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    display = result if args.include_raw else {key: value for key, value in result.items() if key != "raw"}
    print(json.dumps(display, indent=2, sort_keys=True))
    return 0


def _normalize_financial_row(row: dict[str, Any]) -> dict[str, Any]:
    return _compact_dict(
        {
            "period_end": row.get("per_end_date"),
            "period_type": row.get("per_type"),
            "fiscal_year": row.get("per_fisc_year"),
            "fiscal_quarter": row.get("per_fisc_qtr"),
            "filing_type": row.get("filing_type"),
            "filing_date": row.get("filing_date"),
            "revenue": row.get("tot_revnu"),
            "gross_profit": row.get("gross_profit"),
            "operating_income": row.get("oper_income"),
            "net_income": row.get("net_income_loss") or row.get("net_income_parent_comp") or row.get("net_income_loss_share_holder"),
            "eps_basic": row.get("basic_net_eps") or row.get("eps_basic_net"),
            "eps_diluted": row.get("diluted_net_eps") or row.get("eps_diluted_net"),
            "operating_cash_flow": row.get("cash_flow_oper_activity"),
            "investing_cash_flow": row.get("cash_flow_invst_activity"),
            "financing_cash_flow": row.get("cash_flow_fin_activity"),
            "cash_and_short_term_investments": row.get("cash_sterm_invst"),
            "total_assets": row.get("tot_asset"),
            "total_liabilities": row.get("tot_liab"),
            "total_shareholder_equity": row.get("tot_share_holder_equity") or row.get("tot_comm_equity"),
            "total_long_term_debt": row.get("tot_lterm_debt"),
            "current_debt": row.get("curr_portion_debt"),
        }
    )


def _normalize_ratio_row(row: dict[str, Any]) -> dict[str, Any]:
    return _compact_dict(
        {
            "period_end": row.get("per_end_date"),
            "period_type": row.get("per_type"),
            "current_ratio": row.get("curr_ratio"),
            "long_term_debt_to_capital": row.get("lterm_debt_cap"),
            "total_debt_to_equity": row.get("tot_debt_tot_equity"),
            "gross_margin": row.get("gross_margin"),
            "operating_profit_margin": row.get("oper_profit_margin"),
            "ebit_margin": row.get("ebit_margin"),
            "ebitda_margin": row.get("ebitda_margin"),
            "pretax_profit_margin": row.get("pretax_profit_margin"),
            "profit_margin": row.get("profit_margin"),
            "free_cash_flow": row.get("free_cash_flow"),
            "asset_turnover": row.get("asset_turn"),
            "inventory_turnover": row.get("invty_turn"),
            "receivables_turnover": row.get("rcv_turn"),
            "return_on_assets": row.get("ret_asset"),
            "return_on_equity": row.get("ret_equity"),
            "return_on_investment": row.get("ret_invst"),
        }
    )


def _normalize_eps_estimate(row: dict[str, Any]) -> dict[str, Any]:
    return _compact_dict(
        {
            "period_end": row.get("per_end_date"),
            "period_type": row.get("per_type"),
            "fiscal_year": row.get("per_fisc_year"),
            "fiscal_quarter": row.get("per_fisc_qtr"),
            "eps_mean_estimate": row.get("eps_mean_est"),
            "eps_high_estimate": row.get("eps_high_est"),
            "eps_low_estimate": row.get("eps_low_est"),
            "eps_estimate_count": row.get("eps_cnt_est"),
            "eps_pct_change_estimate": row.get("eps_pct_chg_est"),
            "eps_median_estimate": row.get("eps_median_est"),
            "eps_std_dev_estimate": row.get("eps_std_dev_est"),
        }
    )


def _normalize_eps_surprise(row: dict[str, Any]) -> dict[str, Any]:
    return _compact_dict(
        {
            "period_end": row.get("per_end_date"),
            "period_type": row.get("per_type"),
            "actual_report_date": row.get("act_rpt_date"),
            "eps_mean_estimate": row.get("eps_mean_est"),
            "eps_actual": row.get("eps_act"),
            "eps_surprise_amount": row.get("eps_amt_diff_surp"),
            "eps_surprise_pct": row.get("eps_pct_diff_surp"),
            "eps_estimate_count": row.get("eps_cnt_est"),
            "report_time": row.get("act_rpt_time"),
            "report_description": row.get("act_rpt_desc"),
        }
    )


def _get_json(url: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Any:
    request = Request(url, headers={"User-Agent": "market-forecasting-engine/0.1"})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Nasdaq Data Link returned non-JSON response: {payload[:200]}") from exc


def _load_env_file(env_file: str | None = None) -> None:
    paths = [Path(env_file).expanduser()] if env_file else []
    cwd = Path.cwd()
    paths.extend([cwd / ".env", *[parent / ".env" for parent in cwd.parents[:4]]])
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _datatable_rows(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    datatable = payload.get("datatable")
    if not isinstance(datatable, dict):
        return []
    columns = [str(column.get("name")) for column in datatable.get("columns", []) if isinstance(column, dict) and column.get("name")]
    rows = datatable.get("data")
    if not isinstance(rows, list) or not columns:
        return []
    return [dict(zip(columns, row)) for row in rows if isinstance(row, list)]


def _datatable_status(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("datatable"), dict):
        return {"status": "unexpected", "type": type(payload).__name__}
    datatable = payload["datatable"]
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    return {
        "status": "ok",
        "type": "datatable",
        "items": len(datatable.get("data") or []),
        "columns": len(datatable.get("columns") or []),
        "next_cursor_id": meta.get("next_cursor_id"),
    }


def _latest_by_date(rows: list[dict[str, Any]], date_key: str, *, fallback_date_key: str | None = None) -> dict[str, Any]:
    if not rows:
        return {}
    return max(rows, key=lambda row: str(row.get(date_key) or row.get(fallback_date_key or "") or ""))


def _first(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return rows[0] if rows else {}


def _compact_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None and value != ""}


def _safe_error(exc: BaseException) -> str:
    if isinstance(exc, HTTPError):
        return f"HTTP {exc.code}"
    return exc.__class__.__name__


def _key_fingerprint(api_key: str) -> str:
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return f"sha256:{digest[:12]}"


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
