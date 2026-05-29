"""Simple terminal dashboard for all active trader profiles. Auto-refreshes every N seconds.
Usage:
    python dashboard.py
    python dashboard.py --interval 30
    python dashboard.py --trader-name semiconductor_1
"""
import argparse
import os
import time
from datetime import datetime, timezone

from ingestion.db import get_connection, init_db
from mark_to_market.repositories import get_open_holdings
from trader.trader_agent import get_trader_status


def _all_trader_names() -> list[str]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT trader_name FROM trader_profiles ORDER BY trader_name ASC")
    rows = cursor.fetchall()
    conn.close()
    return [row["trader_name"] for row in rows]


def _fmt(value: object, decimals: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.{decimals}f}"
    return str(value)


def _table(title: str, rows: list[tuple[str, str]], indent: str = "") -> str:
    if not rows:
        return f"{indent}{title}\n{indent}  (empty)\n"
    key_w = max(len(k) for k, _ in rows)
    val_w = max(len(v) for _, v in rows)
    sep = indent + "+-" + "-" * key_w + "-+-" + "-" * val_w + "-+"
    lines = [f"\n{indent}{title}", sep]
    for k, v in rows:
        lines.append(indent + f"| {k.ljust(key_w)} | {v.ljust(val_w)} |")
    lines.append(sep)
    return "\n".join(lines)


def _positions_table(holdings: list, indent: str = "") -> str:
    if not holdings:
        return f"\n{indent}Open Positions\n{indent}  No open positions."

    cols = ["ticker", "company", "dir", "qty", "avg_entry", "price", "mkt_value", "pnl", "pnl%", "size"]
    rows = [
        [
            h.ticker,
            h.company_name,
            h.direction,
            f"{h.quantity:.4f}",
            f"{h.average_entry_price:.4f}",
            f"{h.current_price:.4f}",
            f"{h.market_value:.2f}",
            f"{h.unrealized_pnl:+.2f}",
            f"{h.unrealized_pnl_pct:+.2f}%",
            f"{h.position_size:.4f}",
        ]
        for h in holdings
    ]

    widths = [max(len(cols[i]), max(len(r[i]) for r in rows)) for i in range(len(cols))]
    sep = indent + "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header = indent + "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"

    lines = [f"\n{indent}Open Positions", sep, header, sep]
    for r in rows:
        pnl_val = float(r[7].replace("+", ""))
        marker = " ▲" if pnl_val > 0 else (" ▼" if pnl_val < 0 else "  ")
        lines.append(indent + "| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + f" |{marker}")
    lines.append(sep)
    return "\n".join(lines)


def _render_trader(trader_name: str) -> str:
    result = get_trader_status(trader_name)
    holdings = get_open_holdings(trader_name)

    if result.get("status") == "not_found":
        return f"\n  Trader '{trader_name}' not found."

    status = result.get("status", "-")
    status_icon = "● RUNNING" if status == "running" else "○ STOPPED"

    profile = result.get("profile") or {}
    state = result.get("portfolio_state") or {}

    initial = profile.get("initial_cash", 0) or 0
    total = state.get("total_portfolio_value", 0) or 0
    pnl = total - initial
    pnl_pct = (pnl / initial * 100) if initial else 0
    pnl_sign = "+" if pnl >= 0 else ""

    lines = [
        f"\n{'='*60}",
        f"  {status_icon}  |  {trader_name}  |  {profile.get('profile_type', '-').upper()}",
        f"{'='*60}",
    ]

    lines.append(_table("Portfolio", [
        ("initial_cash",    f"${_fmt(initial)}"),
        ("cash",            f"${_fmt(state.get('cash'))}"),
        ("invested",        f"${_fmt(state.get('invested_value'))}"),
        ("total_value",     f"${_fmt(total)}"),
        ("total_pnl",       f"{pnl_sign}${_fmt(abs(pnl))} ({pnl_sign}{_fmt(pnl_pct)}%)"),
        ("open_positions",  _fmt(state.get("open_positions_count"), 0)),
    ], indent="  "))

    lines.append(_positions_table(holdings, indent="  "))
    return "\n".join(lines)


def render_dashboard(trader_names: list[str], refreshed_at: str) -> str:
    sections = [f"\n  AI Trading Dashboard   [refreshed: {refreshed_at}]  (Ctrl+C to quit)"]
    for name in trader_names:
        sections.append(_render_trader(name))
    return "\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Live terminal dashboard for trader agents.")
    parser.add_argument("--trader-name", default=None, help="Show a single trader (default: all)")
    parser.add_argument("--interval", type=int, default=15, help="Auto-refresh interval in seconds (default: 15)")
    args = parser.parse_args()

    init_db()

    try:
        while True:
            names = [args.trader_name] if args.trader_name else _all_trader_names()
            if not names:
                os.system("clear")
                print("\n  No traders found in database.")
            else:
                refreshed_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                output = render_dashboard(names, refreshed_at)
                os.system("clear")
                print(output)

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
