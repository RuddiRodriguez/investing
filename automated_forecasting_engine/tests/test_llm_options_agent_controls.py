from argparse import Namespace

from market_forecasting_engine.llm_options_trader.agent import _entry_mandate, _shadow_trading_budget, build_parser
from market_forecasting_engine.llm_options_trader.common import LLMOptionsRuntimeConfig
from market_forecasting_engine.llm_options_trader.profiles import strategy_mode_profile, trader_profile


def test_entry_bias_put_only_mandate_is_prompt_control_only() -> None:
    mandate = _entry_mandate("put_only")

    assert mandate["mode"] == "put_only"
    assert "open_put" in mandate["allowed_entry_intents"]
    assert "open_call" in mandate["disallowed_entry_intents"]
    assert "buy limit order" in mandate["order_policy"]


def test_entry_bias_parser_defaults_to_unrestricted() -> None:
    args = build_parser().parse_args([])

    assert args.entry_bias == "unrestricted"
    assert args.shadow_account_equity == 0.0
    assert args.shadow_max_entry_debit == 0.0
    assert args.shadow_max_session_debit == 0.0


def test_strategy_mode_profiles_are_config_driven() -> None:
    spot = strategy_mode_profile("crypto_spot_probe")
    options = strategy_mode_profile("crypto_options_directional")

    assert "open_spot_long" in spot["allowed_entry_intents"]
    assert "open_call" in options["allowed_entry_intents"]


def test_micro_scalper_profile_targets_small_repeatable_profits() -> None:
    profile = trader_profile("micro_scalper")

    assert profile["name"] == "Micro Scalper"
    assert "small repeatable option profits" in profile["style"]
    assert "5/10/15/30 minute" in profile["priorities"][0]
    assert "Protect small profits quickly" in profile["priorities"][4]


def test_shadow_trading_budget_uses_simulated_affordability() -> None:
    config = LLMOptionsRuntimeConfig(currency="ETH", instrument_currency="USDC", max_order_amount=2, max_order_price=100)
    args = Namespace(shadow_account_equity=500, shadow_max_entry_debit=75, shadow_max_session_debit=250)

    budget = _shadow_trading_budget(args=args, config=config)

    assert budget["mode"] == "simulation_only_budget"
    assert budget["currency"] == "USDC"
    assert budget["simulated_equity"] == 500
    assert budget["max_entry_debit"] == 75
    assert "not from the live wallet balance" in budget["instruction"]
