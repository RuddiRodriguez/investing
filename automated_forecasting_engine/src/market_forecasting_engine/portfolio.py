from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HOLDING_PATTERN = re.compile(
    r"^(?P<quantity>\d+(?:\.\d+)?)\s+Pcs\.\s+"
    r"(?P<name>.+?)\s+"
    r"(?P<price>\d[\d,.]*)\s+"
    r"(?P<value>\d[\d,.]*)$"
)
ISIN_PATTERN = re.compile(r"ISIN:\s*(?P<isin>[A-Z]{2}[A-Z0-9]{9}\d)")

ISIN_SYMBOL_CANDIDATES = {
    "US8740391003": ["TSM"],
    "US30231G1022": ["XOM"],
    "US11135F1012": ["AVGO"],
    "NL0010273215": ["ASML.AS", "ASML"],
    "US67066G1040": ["NVDA"],
    "US22788C1053": ["CRWD"],
    "IE00BK5BQT80": ["VWCE.DE", "VWRP.L"],
    "US55306N1046": ["MKSI"],
    "IE00BYZK4552": ["2B76.DE", "RBOT.L"],
    "US5324571083": ["LLY"],
}

NAME_SYMBOL_CANDIDATES = {
    "ethereum": ["ETH-USD"],
    "solana": ["SOL-USD"],
}


@dataclass(frozen=True)
class PortfolioHolding:
    asset_type: str
    security_name: str
    quantity: float
    statement_price: float
    statement_value_eur: float
    isin: str | None = None
    symbol_candidates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["symbol_candidates"] = list(self.symbol_candidates)
        return data


def extract_portfolio_holdings(pdf_path: str | Path) -> list[PortfolioHolding]:
    """Extract brokerage and crypto holdings from a Trade Republic net-worth PDF."""

    text = _extract_pdf_text(pdf_path)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    holdings: list[dict[str, Any]] = []
    section: str | None = None
    current: dict[str, Any] | None = None

    for line in lines:
        if line == "BROKERAGE":
            section = "brokerage"
            continue
        if line == "CRYPTO WALLET":
            if current is not None:
                holdings.append(current)
                current = None
            section = "crypto"
            continue
        if line == "CASH":
            if current is not None:
                holdings.append(current)
                current = None
            section = "cash"
            continue
        if section not in {"brokerage", "crypto"}:
            continue

        match = HOLDING_PATTERN.match(line)
        if match:
            if current is not None:
                holdings.append(current)
            current = {
                "asset_type": section,
                "security_name": _clean_security_name(match.group("name")),
                "quantity": _parse_number(match.group("quantity")),
                "statement_price": _parse_number(match.group("price")),
                "statement_value_eur": _parse_number(match.group("value")),
                "isin": None,
            }
            continue

        isin_match = ISIN_PATTERN.search(line)
        if isin_match and current is not None:
            current["isin"] = isin_match.group("isin")

    if current is not None:
        holdings.append(current)

    return [
        PortfolioHolding(
            asset_type=str(item["asset_type"]),
            security_name=str(item["security_name"]),
            quantity=float(item["quantity"]),
            statement_price=float(item["statement_price"]),
            statement_value_eur=float(item["statement_value_eur"]),
            isin=item.get("isin"),
            symbol_candidates=tuple(_symbol_candidates(item)),
        )
        for item in holdings
    ]


def portfolio_projection_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    ordered_columns = [
        "asset_type",
        "security_name",
        "isin",
        "symbol",
        "quantity",
        "current_value_eur",
        "projection_horizon_days",
        "expected_return",
        "projected_value_eur",
        "projected_change_eur",
        "expected_direction",
        "directional_confidence",
        "suggested_action",
        "risk_level",
        "hold_reason",
        "selected_model",
        "forecast_price",
        "forecast_lower_price",
        "forecast_upper_price",
        "status",
        "error",
        "forecast_report",
    ]
    for column in ordered_columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[ordered_columns]


def portfolio_totals(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "current_value_eur": 0.0,
            "projected_value_eur": 0.0,
            "projected_change_eur": 0.0,
            "projected_return": 0.0,
        }
    current = float(pd.to_numeric(frame["current_value_eur"], errors="coerce").fillna(0).sum())
    projected = float(pd.to_numeric(frame["projected_value_eur"], errors="coerce").fillna(0).sum())
    return {
        "current_value_eur": current,
        "projected_value_eur": projected,
        "projected_change_eur": projected - current,
        "projected_return": projected / current - 1 if current else 0.0,
        "forecasted_positions": int((frame["status"] == "forecasted").sum()),
        "failed_positions": int((frame["status"] == "failed").sum()),
    }


def write_projection_artifacts(
    frame: pd.DataFrame,
    totals: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    csv_path = output / "portfolio_projection.csv"
    json_path = output / "portfolio_projection.json"
    html_path = output / "portfolio_projection.html"
    png_path = output / "portfolio_projection.png"
    plotly_path = output / "portfolio_projection_plotly.html"

    frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "totals": _json_ready(totals),
                "positions": _json_ready(frame.to_dict(orient="records")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    html_path.write_text(_projection_html(frame, totals), encoding="utf-8")
    _write_projection_png(frame, png_path)
    _write_projection_plotly(frame, plotly_path)

    return {
        "projection_csv": str(csv_path),
        "projection_json": str(json_path),
        "projection_html": str(html_path),
        "projection_png": str(png_path),
        "projection_plotly": str(plotly_path),
    }


def _extract_pdf_text(pdf_path: str | Path) -> str:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("pdfplumber is required to extract portfolio PDFs.") from exc

    with pdfplumber.open(str(pdf_path)) as pdf:
        return "\n".join(page.extract_text(x_tolerance=1, y_tolerance=3) or "" for page in pdf.pages)


def _symbol_candidates(item: dict[str, Any]) -> list[str]:
    isin = item.get("isin")
    if isin and isin in ISIN_SYMBOL_CANDIDATES:
        return ISIN_SYMBOL_CANDIDATES[isin]
    clean_name = str(item["security_name"]).lower()
    for key, candidates in NAME_SYMBOL_CANDIDATES.items():
        if key in clean_name:
            return candidates
    return []


def _clean_security_name(value: str) -> str:
    return " ".join(value.split())


def _parse_number(value: str) -> float:
    clean = value.replace(",", "")
    return float(clean)


def _projection_html(frame: pd.DataFrame, totals: dict[str, Any]) -> str:
    display = frame.copy()
    for column in ("current_value_eur", "projected_value_eur", "projected_change_eur"):
        display[column] = pd.to_numeric(display[column], errors="coerce").map(lambda value: f"{value:.2f}" if pd.notna(value) else "")
    display["expected_return"] = pd.to_numeric(display["expected_return"], errors="coerce").map(
        lambda value: f"{value:.2%}" if pd.notna(value) else ""
    )
    display["directional_confidence"] = pd.to_numeric(display["directional_confidence"], errors="coerce").map(
        lambda value: f"{value:.0%}" if pd.notna(value) else ""
    )
    table = display.to_html(index=False, classes="projection", border=0)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Portfolio Projection</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #17202a; }}
    h1 {{ font-size: 24px; margin-bottom: 4px; }}
    .summary {{ margin: 12px 0 20px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d6dbe1; padding: 7px 8px; text-align: left; }}
    th {{ background: #eef2f5; font-weight: 650; }}
    tr:nth-child(even) {{ background: #fafbfc; }}
  </style>
</head>
<body>
  <h1>Portfolio Projection</h1>
  <div class="summary">
    Current value: EUR {totals.get("current_value_eur", 0):.2f}<br>
    Projected value: EUR {totals.get("projected_value_eur", 0):.2f}<br>
    Projected change: EUR {totals.get("projected_change_eur", 0):.2f} ({totals.get("projected_return", 0):.2%})
  </div>
  {table}
</body>
</html>
"""


def _write_projection_png(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    plot_frame = frame[frame["asset_type"] != "cash"].copy()
    plot_frame["projected_change_eur"] = pd.to_numeric(plot_frame["projected_change_eur"], errors="coerce").fillna(0.0)
    labels = plot_frame["symbol"].fillna(plot_frame["security_name"]).astype(str).tolist()
    values = plot_frame["projected_change_eur"].tolist()
    colors = ["#1f7a4d" if value >= 0 else "#b42318" for value in values]

    fig_width = max(9, len(labels) * 0.75)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="#2f3a45", linewidth=0.8)
    ax.set_title("Projected EUR Change by Holding")
    ax.set_ylabel("Projected change (EUR)")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_projection_plotly(frame: pd.DataFrame, path: Path) -> None:
    import plotly.graph_objects as go

    plot_frame = frame[frame["asset_type"] != "cash"].copy()
    plot_frame["projected_change_eur"] = pd.to_numeric(plot_frame["projected_change_eur"], errors="coerce").fillna(0.0)
    labels = plot_frame["symbol"].fillna(plot_frame["security_name"]).astype(str)
    values = plot_frame["projected_change_eur"]
    colors = np.where(values >= 0, "#1f7a4d", "#b42318")
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                customdata=np.stack(
                    [
                        plot_frame["security_name"].astype(str),
                        pd.to_numeric(plot_frame["expected_return"], errors="coerce").fillna(0.0),
                        pd.to_numeric(plot_frame["directional_confidence"], errors="coerce").fillna(0.0),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Symbol: %{x}<br>"
                    "Projected change: EUR %{y:.2f}<br>"
                    "Expected return: %{customdata[1]:.2%}<br>"
                    "Confidence: %{customdata[2]:.0%}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title="Projected EUR Change by Holding",
        xaxis_title="Holding",
        yaxis_title="Projected change (EUR)",
        template="plotly_white",
    )
    fig.write_html(path, include_plotlyjs="cdn")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if pd.isna(value):
        return None
    return value
