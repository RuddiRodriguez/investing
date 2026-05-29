import argparse

from ingestion.db import get_connection, init_db
from mark_to_market.repositories import get_open_holdings
from trader.trader_agent import get_trader_status


def _print_table(title: str, rows: list[tuple[str, object]]) -> None:
    key_width = max(len(str(key)) for key, _ in rows)
    val_width = max(len(str(value)) for _, value in rows)
    line = "+-" + ("-" * key_width) + "-+-" + ("-" * val_width) + "-+"

    print(f"\n{title}")
    print(line)
    for key, value in rows:
        print(f"| {str(key).ljust(key_width)} | {str(value).ljust(val_width)} |")
    print(line)


def _print_all_traders_status() -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            tp.trader_name,
            tp.status,
            tp.profile_type,
            tp.current_cash,
            COALESCE(ps.invested_value, 0.0) AS invested_value,
            COALESCE(ps.total_portfolio_value, tp.current_cash) AS total_portfolio_value,
            COALESCE(ps.open_positions_count, 0) AS open_positions_count
        FROM trader_profiles tp
        LEFT JOIN portfolio_state ps
            ON tp.trader_name = ps.trader_name
        ORDER BY tp.trader_name
        """
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("\nAll Traders Status")
        print("No traders found.")
        return

    headers = [
        "trader_name",
        "status",
        "profile_type",
        "current_cash",
        "invested_value",
        "total_portfolio_value",
        "open_positions_count",
    ]
    values = [
        [
            str(row["trader_name"]),
            str(row["status"]),
            str(row["profile_type"]),
            f"{float(row['current_cash']):.2f}",
            f"{float(row['invested_value']):.2f}",
            f"{float(row['total_portfolio_value']):.2f}",
            str(int(row["open_positions_count"])),
        ]
        for row in rows
    ]

    col_widths = [
        max(len(header), max(len(v[i]) for v in values))
        for i, header in enumerate(headers)
    ]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"

    print("\nAll Traders Status")
    print(sep)
    print(header_row)
    print(sep)
    for row in values:
        print("| " + " | ".join(v.ljust(w) for v, w in zip(row, col_widths)) + " |")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show status for a trader agent.")
    parser.add_argument("--trader-name", default="semiconductor_aggressive_001", help="Trader name to inspect")
    parser.add_argument("--all", action="store_true", help="Show a summary for all traders")
    args = parser.parse_args()

    init_db()

    if args.all:
        _print_all_traders_status()
        return

    result = get_trader_status(
        trader_name=args.trader_name,
    )

    if result.get("status") == "not_found":
        print("\nTrader Status")
        print("Trader not found.")
        print(f"Trader Name: {result.get('trader_name')}")
        return

    _print_table(
        title="Trader Status",
        rows=[
            ("trader_name", result.get("trader_name")),
            ("status", result.get("status")),
        ],
    )

    profile = result.get("profile") or {}
    _print_table(
        title="Profile",
        rows=[
            ("profile_type", profile.get("profile_type")),
            ("trade_frequency", profile.get("trade_frequency")),
            ("risk_tolerance", profile.get("risk_tolerance")),
            ("max_position_size", profile.get("max_position_size")),
            ("max_portfolio_exposure", profile.get("max_portfolio_exposure")),
            ("min_cash_reserve", profile.get("min_cash_reserve")),
            ("initial_cash", profile.get("initial_cash")),
            ("current_cash", profile.get("current_cash")),
        ],
    )

    portfolio_state = result.get("portfolio_state") or {}
    _print_table(
        title="Portfolio State",
        rows=[
            ("cash", portfolio_state.get("cash")),
            ("invested_value", portfolio_state.get("invested_value")),
            ("total_portfolio_value", portfolio_state.get("total_portfolio_value")),
            ("open_positions_count", portfolio_state.get("open_positions_count")),
        ],
    )

    holdings = get_open_holdings(args.trader_name)
    if holdings:
        headers = ["ticker", "company_name", "direction", "quantity", "avg_entry", "current_price", "market_value", "unrealized_pnl", "unrealized_pnl_pct", "position_size"]
        col_widths = [max(len(h), max((len(str(getattr(h_row, col, "") or "")) for h_row in holdings), default=0)) for h, col in zip(headers, headers)]
        sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"

        print("\nOpen Positions")
        print(sep)
        print(header_row)
        print(sep)
        for h_row in holdings:
            values = [
                str(h_row.ticker),
                str(h_row.company_name),
                str(h_row.direction),
                f"{h_row.quantity:.4f}",
                f"{h_row.average_entry_price:.4f}",
                f"{h_row.current_price:.4f}",
                f"{h_row.market_value:.4f}",
                f"{h_row.unrealized_pnl:.4f}",
                f"{h_row.unrealized_pnl_pct:.2f}%",
                f"{h_row.position_size:.4f}",
            ]
            col_widths_actual = [max(len(h), len(v)) for h, v in zip(headers, values)]
            # reprint with proper widths per row
            print("| " + " | ".join(v.ljust(w) for v, w in zip(values, col_widths)) + " |")
        print(sep)
    else:
        print("\nOpen Positions\nNo open positions.")


if __name__ == "__main__":
    main()
