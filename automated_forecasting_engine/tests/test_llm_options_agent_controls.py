from argparse import Namespace

from market_forecasting_engine.llm_model_catalog import DEFAULT_FULL_LLM_OPTIONS_MODEL, DEFAULT_FULL_LLM_OPTIONS_PROVIDER
from market_forecasting_engine.llm_options_trader.agent import _branch_profile, _entry_mandate, _shadow_trading_budget, build_parser
from market_forecasting_engine.llm_options_trader.common import LLMOptionsRuntimeConfig
from market_forecasting_engine.llm_options_trader.profiles import strategy_mode_profile, trader_profile
from market_forecasting_engine.llm_options_trader.prompts import ENTRY_LLM_PROFILE, EXIT_LLM_PROFILE, PROFIT_POLICY_LLM_PROFILE
from market_forecasting_engine.openai_models import DEFAULT_OPENAI_MODEL


def test_entry_bias_put_only_mandate_is_prompt_control_only() -> None:
    mandate = _entry_mandate("put_only")

    assert mandate["mode"] == "put_only"
    assert "open_put" in mandate["allowed_entry_intents"]
    assert "open_call" in mandate["disallowed_entry_intents"]
    assert "buy limit order" in mandate["order_policy"]


def test_entry_bias_parser_defaults_to_unrestricted() -> None:
    args = build_parser().parse_args([])

    assert args.entry_bias == "unrestricted"
    assert args.llm_provider == DEFAULT_FULL_LLM_OPTIONS_PROVIDER
    assert args.shadow_account_equity == 0.0
    assert args.shadow_max_entry_debit == 0.0
    assert args.shadow_max_session_debit == 0.0


def test_full_llm_options_branch_profiles_default_to_local_model(monkeypatch) -> None:
    monkeypatch.delenv("ENTRY_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ENTRY_LLM_MODEL", raising=False)
    monkeypatch.delenv("EXIT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("EXIT_LLM_MODEL", raising=False)
    monkeypatch.delenv("PROFIT_POLICY_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("PROFIT_POLICY_LLM_MODEL", raising=False)
    args = build_parser().parse_args([])

    entry = _branch_profile(args, "entry", ENTRY_LLM_PROFILE)
    exit_profile = _branch_profile(args, "exit", EXIT_LLM_PROFILE)
    profit_policy = _branch_profile(args, "profit_policy", PROFIT_POLICY_LLM_PROFILE)

    assert entry.provider == DEFAULT_FULL_LLM_OPTIONS_PROVIDER
    assert entry.model == DEFAULT_FULL_LLM_OPTIONS_MODEL
    assert exit_profile.provider == DEFAULT_FULL_LLM_OPTIONS_PROVIDER
    assert exit_profile.model == DEFAULT_FULL_LLM_OPTIONS_MODEL
    assert profit_policy.provider == DEFAULT_FULL_LLM_OPTIONS_PROVIDER
    assert profit_policy.model == DEFAULT_FULL_LLM_OPTIONS_MODEL


def test_full_llm_options_branch_profiles_accept_step_specific_cli_models() -> None:
    args = build_parser().parse_args(
        [
            "--entry-llm-provider",
            "openai",
            "--entry-llm-model",
            "entry-model",
            "--exit-llm-provider",
            "bedrock",
            "--exit-llm-model",
            "exit-model",
            "--profit-policy-llm-provider",
            "huggingface",
            "--profit-policy-llm-model",
            "policy-model",
        ]
    )

    assert _branch_profile(args, "entry", ENTRY_LLM_PROFILE).provider == "openai"
    assert _branch_profile(args, "entry", ENTRY_LLM_PROFILE).model == "entry-model"
    assert _branch_profile(args, "exit", EXIT_LLM_PROFILE).provider == "bedrock"
    assert _branch_profile(args, "exit", EXIT_LLM_PROFILE).model == "exit-model"
    assert _branch_profile(args, "profit_policy", PROFIT_POLICY_LLM_PROFILE).provider == "huggingface"
    assert _branch_profile(args, "profit_policy", PROFIT_POLICY_LLM_PROFILE).model == "policy-model"


def test_full_llm_options_branch_profiles_accept_step_env_models(monkeypatch) -> None:
    monkeypatch.setenv("ENTRY_LLM_PROVIDER", "lm-studio")
    monkeypatch.setenv("ENTRY_LLM_MODEL", "nemotron-mini-4b-instruct")
    args = build_parser().parse_args(["--llm-provider", "openai"])

    entry = _branch_profile(args, "entry", ENTRY_LLM_PROFILE)

    assert entry.provider == "llm_studio"
    assert entry.model == "nemotron-mini-4b-instruct"


def test_full_llm_options_global_provider_without_model_uses_provider_default(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    args = build_parser().parse_args(["--llm-provider", "openai"])

    entry = _branch_profile(args, "entry", ENTRY_LLM_PROFILE)

    assert entry.provider == "openai"
    assert entry.model == DEFAULT_OPENAI_MODEL


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
