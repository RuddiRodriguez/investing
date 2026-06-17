from __future__ import annotations

import argparse

from market_forecasting_engine import stockanalysis_analyst_flow as flow


def test_map_stockanalysis_symbol_to_forecast_symbol() -> None:
    assert flow.map_stockanalysis_symbol_to_forecast_symbol("TSX: BDI") == "BDI.TO"
    assert flow.map_stockanalysis_symbol_to_forecast_symbol("TSXV: ABC") == "ABC.V"
    assert flow.map_stockanalysis_symbol_to_forecast_symbol("CIEN") == "CIEN"


def test_select_recent_visible_analyst_tickers_from_mocked_html(monkeypatch) -> None:
    leaderboard = """
    <table>
    <tr><th>#</th><th>Analyst Name</th><th>Company</th><th>Main Sector</th><th>Success Rate</th><th>Average Return</th><th>Ratings</th><th>Last Rating</th></tr>
    <tr><td>98</td><td><a href="/analysts/john-gibson-cfa/">John Gibson CFA</a></td><td>BMO Capital</td><td>Energy</td><td>66.70%</td><td>29.30%</td><td>536</td><td>Jun 11, 2026</td></tr>
    <tr><td>99</td><td><a href="/analysts/alexander-paris/">Alexander Paris</a></td><td>Barrington</td><td>Consumer Staples</td><td>67.80%</td><td>21.60%</td><td>1190</td><td>Jun 9, 2026</td></tr>
    </table>
    """
    analyst = """
    <a href="/stocks/compare/">Comparison Tool</a>
    <table>
    <tr><th>Stock</th><th>Action</th><th>Price Target</th><th>Current</th><th>Upside</th><th>Ratings</th><th>Updated</th></tr>
    <tr><td><a href="/stocks/bdi/">TSX: BDI</a> Black Diamond Group Limited</td><td>Upgraded:</td><td>Buy</td><td>$22.00 → $23.00</td><td>$19.12</td><td>+20.29%</td><td>Jun 11, 2026</td></tr>
    <tr><td><a href="/stocks/efx/">TSX: EFX</a> Enerflex Ltd.</td><td>Maintained:</td><td>Buy</td><td>$45.00</td><td>$33.90</td><td>+32.74%</td><td>May 27, 2026</td></tr>
    </table>
    <a href="/stocks/bdi/">TSX: BDI</a>
    """

    def fake_get_html(url: str, *, timeout: float) -> str:
        if url.endswith("/analysts/"):
            return leaderboard
        if url.endswith("/analysts/john-gibson-cfa/"):
            return analyst
        raise AssertionError(url)

    monkeypatch.setattr(flow, "_get_html", fake_get_html)

    candidates = flow.select_recent_visible_analyst_tickers(max_tickers=1)

    assert candidates[0].analyst.name == "John Gibson CFA"
    assert candidates[0].source_symbol == "TSX: BDI"
    assert candidates[0].forecast_symbol == "BDI.TO"


def test_analyst_selected_forecast_command_keeps_strategy_knowledge_default(tmp_path, monkeypatch) -> None:
    candidate = flow.AnalystTickerCandidate(
        analyst=flow.VisibleAnalyst(
            rank=1,
            name="Analyst",
            slug="analyst",
            company="Firm",
            sector="Tech",
            success_rate=0.7,
            average_return=0.2,
            ratings=10,
            last_rating="Jun 15, 2026",
            url="https://stockanalysis.com/analysts/analyst/",
        ),
        source_symbol="FPS",
        forecast_symbol="FPS",
        company_name="Fps Inc",
        rating_action="Maintained:",
        rating="Buy",
        price_target="$68",
        current_price="$59",
        upside="+15%",
        updated="Jun 15, 2026",
        source_url="https://stockanalysis.com/stocks/fps/",
    )
    monkeypatch.setattr(flow, "select_recent_visible_analyst_tickers", lambda max_tickers, *, timeout: [candidate])
    args = argparse.Namespace(
        max_tickers=1,
        timeout=1.0,
        output_root=str(tmp_path),
        start="2020-01-01",
        end=None,
        horizons="1",
        search_level="fast",
        calendar=None,
        llm_env_file=None,
        disable_alt_news=False,
        disable_long_term_sources=False,
        disable_strategy_knowledge=False,
        strategy_knowledge_rebuild_index=False,
        strategy_knowledge_max_chunks=8,
        dry_run=True,
        project_dir=str(tmp_path),
    )

    summary = flow.run_analyst_selected_forecasts(args)
    command = summary["results"][0]["command"]

    assert "--disable-strategy-knowledge" not in command
    assert "--strategy-knowledge-max-chunks" in command
    assert command[command.index("--strategy-knowledge-max-chunks") + 1] == "8"
