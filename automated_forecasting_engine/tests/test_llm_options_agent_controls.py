from market_forecasting_engine.llm_options_trader.agent import _entry_mandate, build_parser


def test_entry_bias_put_only_mandate_is_prompt_control_only() -> None:
    mandate = _entry_mandate("put_only")

    assert mandate["mode"] == "put_only"
    assert "open_put" in mandate["allowed_entry_intents"]
    assert "open_call" in mandate["disallowed_entry_intents"]
    assert "buy limit order" in mandate["order_policy"]


def test_entry_bias_parser_defaults_to_unrestricted() -> None:
    args = build_parser().parse_args([])

    assert args.entry_bias == "unrestricted"
