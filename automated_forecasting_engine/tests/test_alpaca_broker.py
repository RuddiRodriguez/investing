from __future__ import annotations

from market_forecasting_engine.alpaca_broker import _chunks, _position_symbol_candidates


def test_position_symbol_candidates_include_crypto_compact_and_slash_forms() -> None:
    assert _position_symbol_candidates("ETH/USD") == ["ETH/USD", "ETHUSD"]
    assert _position_symbol_candidates("ETHUSD") == ["ETHUSD", "ETH/USD"]


def test_chunks_splits_option_snapshot_symbol_batches() -> None:
    values = [str(index) for index in range(205)]

    chunks = _chunks(values, 100)

    assert [len(chunk) for chunk in chunks] == [100, 100, 5]
