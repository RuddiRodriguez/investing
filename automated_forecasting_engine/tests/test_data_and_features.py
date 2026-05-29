from __future__ import annotations

import warnings
import sys
import types

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from market_forecasting_engine.data import data_version_hash, enrich_price_frame, load_indicator_csv, normalize_price_frame
from market_forecasting_engine.data_manifest import build_data_manifest
from market_forecasting_engine.data_providers import DataRequest, load_prices_with_provider
from market_forecasting_engine.data_quality import build_data_quality_report
from market_forecasting_engine.data_store import MarketDataStore
from market_forecasting_engine.alternative_data import (
    AlternativeNewsRequest,
    aggregate_news_sentiment_features,
    alternative_data_registry_entry,
    fetch_news_articles,
    fetch_openai_web_news_and_social,
    fetch_yfinance_news,
    score_news_articles,
)
from market_forecasting_engine.feature_registry import build_feature_registry
from market_forecasting_engine.features import add_forward_return_targets, build_feature_frame
from market_forecasting_engine.panel import (
    build_cross_sectional_panel_features,
    build_panel_frame,
    load_universe_csv,
    rank_universe_from_panel,
    select_ticker_panel_features,
    summarize_universe,
)
from market_forecasting_engine.security_master import load_security_master, resolve_security_metadata


def _prices(rows: int = 260) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=rows)
    close = 100 + np.linspace(0, 20, rows) + np.sin(np.arange(rows) / 5)
    volume = 1_000_000 + np.arange(rows) * 100
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": volume,
            "Sector Index": close * 1.01,
        }
    )


def test_normalize_price_frame_parses_date_and_columns() -> None:
    normalized = normalize_price_frame(_prices())

    assert isinstance(normalized.index, pd.DatetimeIndex)
    assert {"open", "high", "low", "close", "volume", "sector_index"}.issubset(normalized.columns)
    assert normalized.index.is_monotonic_increasing


def test_normalize_price_frame_handles_yahoo_single_ticker_multiindex() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    columns = pd.MultiIndex.from_tuples(
        [
            ("Open", "AAPL"),
            ("High", "AAPL"),
            ("Low", "AAPL"),
            ("Close", "AAPL"),
            ("Volume", "AAPL"),
        ]
    )
    raw = pd.DataFrame(
        [
            [100.0, 101.0, 99.0, 100.5, 1_000_000],
            [101.0, 102.0, 100.0, 101.5, 1_100_000],
            [102.0, 103.0, 101.0, 102.5, 1_200_000],
        ],
        index=dates,
        columns=columns,
    )

    normalized = normalize_price_frame(raw)

    assert list(normalized.columns) == ["open", "high", "low", "close", "volume"]
    assert normalized.loc[dates[-1], "close"] == 102.5


def test_data_version_hash_is_stable_for_same_data() -> None:
    normalized = normalize_price_frame(_prices())

    assert data_version_hash(normalized) == data_version_hash(normalized.copy())


def test_feature_frame_builds_expected_technical_and_external_columns() -> None:
    normalized = normalize_price_frame(_prices())
    features = build_feature_frame(normalized)

    expected = {
        "log_return_1d",
        "close_to_sma_20",
        "volatility_20d",
        "rsi_14",
        "macd_hist",
        "bollinger_z_20",
        "volume_z_20",
        "exo_sector_index",
        "relative_sector_index_return_21d",
        "relative_sector_index_beta_63d",
        "true_range_pct",
        "atr_14_pct",
        "parkinson_volatility_20d",
        "drawdown_63d",
        "dollar_volume_z_20",
        "amihud_illiquidity_20d",
        "on_balance_volume_z_63",
        "money_flow_index_14",
        "structure_close_to_support_63d",
        "structure_breakout_volume_confirmed_63d",
        "structure_trend_slope_50d",
        "structure_pivot_high_confirmed",
        "structure_close_to_pivot_resistance",
        "structure_failed_breakout_63d",
        "structure_gap_fill_pct",
        "structure_true_gap_size_pct",
        "structure_true_gap_atr_multiple",
        "structure_rectangle_consolidation_20d",
        "day_of_week",
    }
    assert expected.issubset(features.columns)


def test_feature_frame_handles_many_external_columns_without_fragmentation_warning() -> None:
    normalized = normalize_price_frame(_prices(rows=260))
    for idx in range(45):
        normalized[f"alt_context_{idx}"] = np.sin(np.arange(len(normalized)) / (idx + 2)) + idx

    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        features = build_feature_frame(normalized)

    assert "exo_alt_context_0" in features.columns
    assert "exo_alt_context_44_z_20" in features.columns


def test_feature_registry_describes_feature_families() -> None:
    normalized = normalize_price_frame(_prices())
    features = build_feature_frame(normalized)
    registry = build_feature_registry(features)

    assert registry["feature_count"] == len(features.columns)
    assert registry["family_counts"]["relative_strength"] >= 1
    assert registry["family_counts"]["volatility_risk"] >= 1
    assert registry["family_counts"]["liquidity_volume"] >= 1
    assert any(entry["name"] == "relative_sector_index_return_21d" for entry in registry["entries"])


def test_enrich_price_frame_joins_macro_and_event_csvs(tmp_path) -> None:
    prices = normalize_price_frame(_prices(rows=30))
    macro_csv = tmp_path / "macro.csv"
    macro_csv.write_text(
        "date,cpi,unemployment\n"
        "2024-01-02,3.1,4.0\n"
        "2024-01-10,3.2,4.1\n",
        encoding="utf-8",
    )
    events_csv = tmp_path / "events.csv"
    events_csv.write_text(
        "date,event_type\n"
        "2024-01-05,earnings\n"
        "2024-01-05,dividend\n",
        encoding="utf-8",
    )

    enriched = enrich_price_frame(
        prices,
        indicator_csvs={"macro": macro_csv},
        event_csvs={"event": events_csv},
    )

    assert {"macro_cpi", "macro_unemployment", "event_count", "event_earnings", "event_dividend"}.issubset(enriched.columns)
    assert enriched.loc[pd.Timestamp("2024-01-05"), "event_count"] == 2.0
    assert enriched["macro_cpi"].notna().any()


def test_alternative_news_sentiment_features_are_point_in_time() -> None:
    index = pd.bdate_range("2024-01-02", periods=8)
    articles = [
        {
            "date": "2024-01-02",
            "published_at": "2024-01-02T14:00:00+00:00",
            "ticker": "TEST",
            "title": "TEST beats profit expectations with strong growth",
            "summary": "",
            "source": "example.com",
            "source_category": "news",
        },
        {
            "date": "2024-01-04",
            "published_at": "2024-01-04T14:00:00+00:00",
            "ticker": "TEST",
            "title": "TEST faces downgrade after weak outlook",
            "summary": "",
            "source": "example.com",
            "source_category": "news",
        },
    ]

    scored, metadata = score_news_articles(articles, AlternativeNewsRequest(ticker="TEST"))
    features = aggregate_news_sentiment_features(pd.DataFrame(scored), index)
    registry = alternative_data_registry_entry(
        AlternativeNewsRequest(ticker="TEST"),
        pd.DataFrame(scored),
        features,
        {"status": "ok"},
        metadata,
    )

    assert features.loc[pd.Timestamp("2024-01-02"), "alt_news_volume_1d"] == 1.0
    assert features.loc[pd.Timestamp("2024-01-03"), "alt_news_volume_1d"] == 0.0
    assert features.loc[pd.Timestamp("2024-01-04"), "alt_news_negative_share_1d"] > 0.0
    assert registry["point_in_time_safe"] is True
    assert registry["collection_method"] == "download_or_scrape"


def test_openai_web_alternative_provider_skips_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    articles, metadata = fetch_news_articles(AlternativeNewsRequest(ticker="TEST", provider="openai_web"))

    assert articles == []
    assert metadata["status"] == "skipped"
    assert metadata["name"] == "openai_web"


def test_openai_web_provider_uses_structured_outputs_with_web_search(monkeypatch) -> None:
    captured_kwargs = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            return types.SimpleNamespace(
                output_text=(
                    '{"articles": [{"published_at": "2026-05-28T12:00:00Z", '
                    '"title": "ASML shares rise after strong EUV demand", '
                    '"summary": "Investors react to upbeat order commentary.", '
                    '"url": "https://example.com/asml-news", '
                    '"source": "Example News", "source_category": "news"}]}'
                )
            )

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            self.responses = FakeResponses()

    fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    articles, metadata = fetch_openai_web_news_and_social(
        AlternativeNewsRequest(ticker="ASML", provider="openai_web", max_items=40)
    )

    assert captured_kwargs["text"]["format"]["type"] == "json_schema"
    assert captured_kwargs["text"]["format"]["name"] == "alternative_news_articles"
    assert captured_kwargs["model"] == "gpt-5.4-mini-2026-03-17"
    assert captured_kwargs["reasoning"]["effort"] == "none"
    assert captured_kwargs["tools"][0]["type"] == "web_search"
    assert '"max_items": 12' in captured_kwargs["input"][1]["content"][0]["text"]
    assert metadata["status"] == "ok"
    assert metadata["article_count"] == 1
    assert metadata["requested_max_items"] == 40
    assert metadata["effective_max_items"] == 12
    assert articles[0]["retrieval_method"] == "openai_web_search"


def test_yfinance_news_provider_normalizes_nested_news_items(monkeypatch) -> None:
    class FakeTicker:
        def __init__(self, ticker: str) -> None:
            self.ticker = ticker

        @property
        def news(self) -> list[dict[str, object]]:
            return [
                {
                    "content": {
                        "title": "ASML beats expectations with strong backlog",
                        "summary": "Analysts cite AI-chip demand and strong EUV orders.",
                        "pubDate": "2026-05-28T16:59:00Z",
                        "provider": {"displayName": "Example News", "sourceId": "example"},
                        "canonicalUrl": {"url": "https://example.com/asml"},
                        "contentType": "STORY",
                    }
                }
            ]

    fake_yfinance = types.SimpleNamespace(Ticker=FakeTicker)
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance)

    articles, metadata = fetch_yfinance_news(AlternativeNewsRequest(ticker="ASML", provider="yahoo_news"))

    assert metadata["status"] == "ok"
    assert metadata["article_count"] == 1
    assert articles[0]["source"] == "Example News"
    assert articles[0]["retrieval_method"] == "yfinance_news"


def test_yahoo_rss_falls_back_to_yfinance_news(monkeypatch) -> None:
    class FakeTicker:
        def __init__(self, ticker: str) -> None:
            self.ticker = ticker

        @property
        def news(self) -> list[dict[str, object]]:
            return [
                {
                    "content": {
                        "title": "ASML upgrade after record demand",
                        "summary": "",
                        "pubDate": "2026-05-28T16:00:00Z",
                        "provider": {"displayName": "Example News"},
                        "canonicalUrl": {"url": "https://example.com/asml-upgrade"},
                    }
                }
            ]

    def raise_url_error(*args, **kwargs):
        raise OSError("rss unavailable")

    monkeypatch.setattr("urllib.request.urlopen", raise_url_error)
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))

    articles, metadata = fetch_news_articles(AlternativeNewsRequest(ticker="ASML", provider="yahoo_rss"))

    assert len(articles) == 1
    assert metadata["status"] == "fallback"
    assert metadata["fallback_provider"]["name"] == "yahoo_news"


def test_indicator_csv_uses_availability_dates_and_release_lags(tmp_path) -> None:
    macro_csv = tmp_path / "macro.csv"
    macro_csv.write_text(
        "date,available_at,cpi\n"
        "2024-01-02,2024-01-10,3.1\n"
        "2024-02-01,,3.2\n",
        encoding="utf-8",
    )

    indicators = load_indicator_csv(macro_csv, prefix="macro", release_lag_days=1)

    assert pd.Timestamp("2024-01-11") in indicators.index
    assert pd.Timestamp("2024-02-02") in indicators.index
    assert indicators.loc[pd.Timestamp("2024-01-11"), "macro_cpi"] == 3.1


def test_provider_store_cache_and_manifest_roundtrip(tmp_path) -> None:
    prices_csv = tmp_path / "prices.csv"
    _prices(rows=30).to_csv(prices_csv, index=False)
    store = MarketDataStore(tmp_path / "data")
    request = DataRequest(ticker="TEST", source_path=str(prices_csv), target_column="close")

    first = load_prices_with_provider("csv", request, store=store)
    second = load_prices_with_provider("csv", request, store=store)
    manifest = build_data_manifest(
        prices=second.frame,
        ticker="TEST",
        provider="csv",
        request=request.to_dict(),
        artifacts=second.metadata.get("artifacts", {}),
    )

    assert first.metadata["cache_hit"] is False
    assert second.metadata["cache_hit"] is True
    assert manifest["row_count"] == 30
    assert manifest["normalized_data_hash"] == data_version_hash(second.frame)


def test_data_quality_report_flags_bad_data_and_missing_sessions() -> None:
    prices = normalize_price_frame(_prices(rows=20))
    prices = prices.drop(prices.index[5])
    prices.loc[prices.index[3], "close"] = 0.0

    report = build_data_quality_report(prices)

    assert report["status"] in {"warn", "fail"}
    assert report["non_positive_prices"]["close"] == 1
    assert any(warning["code"] == "missing_trading_sessions" for warning in report["warnings"])


def test_security_master_and_panel_helpers(tmp_path) -> None:
    security_csv = tmp_path / "security_master.csv"
    security_csv.write_text(
        "ticker,exchange,currency,sector,active\n"
        "AAA,XNYS,USD,Technology,true\n"
        "BBB,XNAS,USD,Industrials,true\n",
        encoding="utf-8",
    )
    universe_csv = tmp_path / "universe.csv"
    universe_csv.write_text("symbol,sector\nAAA,Technology\nBBB,Industrials\n", encoding="utf-8")
    prices_a = normalize_price_frame(_prices(rows=20))
    prices_b = prices_a.assign(close=prices_a["close"] * 1.1)

    security_master = load_security_master(security_csv)
    metadata = resolve_security_metadata("AAA", prices_a, security_master=security_master)
    universe = load_universe_csv(universe_csv)
    panel = build_panel_frame({"AAA": prices_a, "BBB": prices_b})
    panel_features = build_cross_sectional_panel_features(panel)
    aaa_panel_features = select_ticker_panel_features(panel_features, "AAA")
    ranking = rank_universe_from_panel(panel, panel_features, top_n=2)
    summary = summarize_universe(universe["ticker"].tolist(), panel)

    assert metadata["exchange"] == "XNYS"
    assert metadata["sector"] == "Technology"
    assert panel.index.names == ["date", "ticker"]
    assert {"cs_return_1d_rank", "cs_momentum_21d_rank", "cs_dollar_volume_20d_rank"}.issubset(panel_features.columns)
    assert "panel_cs_return_1d_rank" in aaa_panel_features.columns
    assert len(ranking) == 2
    assert {"ticker", "score", "rank", "cross_sectional_momentum_rank"}.issubset(ranking[0])
    assert summary["ticker_count"] == 2
    assert summary["panel_rows"] == 40


def test_forward_targets_align_with_future_prices() -> None:
    normalized = normalize_price_frame(_prices(rows=40))
    features = build_feature_frame(normalized)
    supervised = add_forward_return_targets(features, normalized, horizons=(5,))

    first_date = supervised.index[0]
    expected = np.log(normalized.loc[supervised.index[5], "close"] / normalized.loc[first_date, "close"])

    assert np.isclose(supervised.loc[first_date, "target_log_return_5d"], expected)
    assert "target_upside_breakout_5d" in supervised.columns
    assert "target_downside_breakdown_5d" in supervised.columns
    assert "target_reward_to_risk_5d" in supervised.columns
    assert pd.isna(supervised.iloc[-1]["target_log_return_5d"])
    assert pd.isna(supervised.iloc[-1]["target_direction_5d"])
