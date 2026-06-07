from market_forecasting_engine.llm_options_trader import prompts


def test_combined_prompt_frames_signals_as_discretionary_evidence() -> None:
    text = prompts.COMBINED_SYSTEM_MESSAGE

    assert "not a rule engine" in text
    assert "evidence and context, not automatic vetoes" in text
    assert "small exploratory position" in text
    assert "Opportunity cost is real" in text
    assert "minimum-size probe can be better than another theoretical hold" in text
    assert "Tiny account size should control position size" in text
    assert "prior holds missed" in text
    assert "forecast_error_feedback" in text
    assert "correction layer from prior forecast mistakes" in text
    assert "small repeatable profits" in text
    assert "do not wait until price is already sitting on support" in text
    assert "Expiry selection is part of the trade thesis" in text
    assert "whether the DTE matches the forecast/trade horizon" in text
