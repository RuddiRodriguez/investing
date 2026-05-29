from alpha.schemas import AlphaInput, CombinedAlphaSignal
from alpha.repositories import (
    get_alpha_inputs,
    get_combined_alpha_signal,
    save_combined_alpha_signal,
)


def clamp(value: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
    return max(minimum, min(value, maximum))


def safe_score(value: float | None) -> float:
    if value is None:
        return 0.0
    return clamp(value)


def safe_confidence(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(value, 1.0))


def chart_to_alpha_score(chart_score: float | None, chart_decision: str | None) -> float:
    if chart_score is None or chart_decision is None:
        return 0.0
    if chart_decision == "BUY":
        return chart_score
    if chart_decision == "WAIT_FOR_BREAKOUT":
        return min(chart_score, 0.20)
    if chart_decision == "HOLD":
        return 0.0
    if chart_decision in ["AVOID", "SELL"]:
        return -chart_score
    return 0.0


def get_alpha_label(alpha_score: float, sentiment_score: float, technical_score: float) -> str:
    opposite_signals = (
        sentiment_score > 0.25 and technical_score < -0.25
    ) or (
        sentiment_score < -0.25 and technical_score > 0.25
    )

    if opposite_signals and abs(alpha_score) < 0.35:
        return "mixed"
    if alpha_score >= 0.65:
        return "strong_bullish"
    if alpha_score >= 0.25:
        return "bullish"
    if alpha_score <= -0.65:
        return "strong_bearish"
    if alpha_score <= -0.25:
        return "bearish"
    return "neutral"


def get_main_driver(
    sentiment_score: float,
    sentiment_confidence: float,
    technical_score: float,
    technical_confidence: float,
) -> str:
    sentiment_power = abs(sentiment_score) * sentiment_confidence
    technical_power = abs(technical_score) * technical_confidence

    if sentiment_power == 0 and technical_power == 0:
        return "No strong sentiment or technical driver available."
    if sentiment_power > technical_power:
        if sentiment_score > 0:
            return "Positive news sentiment is the main driver."
        if sentiment_score < 0:
            return "Negative news sentiment is the main driver."
    if technical_power > sentiment_power:
        if technical_score > 0:
            return "Positive technical momentum is the main driver."
        if technical_score < 0:
            return "Negative technical momentum is the main driver."
    return "Sentiment and technical signals have similar influence."


def build_combined_alpha_signal(alpha_input: AlphaInput) -> CombinedAlphaSignal:
    sentiment_score = safe_score(alpha_input.sentiment_score)
    technical_score = safe_score(alpha_input.technical_score)
    sentiment_confidence = safe_confidence(alpha_input.sentiment_confidence)
    technical_confidence = safe_confidence(alpha_input.technical_confidence)
    chart_score = chart_to_alpha_score(
        chart_score=alpha_input.chart_score,
        chart_decision=alpha_input.chart_decision,
    )
    chart_confidence = safe_confidence(alpha_input.chart_confidence)

    sentiment_weight = 0.30
    technical_weight = 0.25
    growth_weight = 0.20
    chart_weight = 0.25

    available_weight = 0.0
    weighted_score = 0.0

    if alpha_input.sentiment_score is not None:
        weighted_score += sentiment_score * sentiment_weight * sentiment_confidence
        available_weight += sentiment_weight * sentiment_confidence
    if alpha_input.technical_score is not None:
        weighted_score += technical_score * technical_weight * technical_confidence
        available_weight += technical_weight * technical_confidence
    if alpha_input.chart_score is not None:
        weighted_score += chart_score * chart_weight * chart_confidence
        available_weight += chart_weight * chart_confidence
    if alpha_input.chart_score is None:
        available_weight += growth_weight * 0.0

    if available_weight == 0:
        alpha_score = 0.0
    else:
        alpha_score = weighted_score / available_weight
    alpha_score = clamp(alpha_score)

    if alpha_input.chart_decision == "WAIT_FOR_BREAKOUT" and alpha_score > 0.35:
        alpha_score = 0.35
    if alpha_input.chart_decision in ["AVOID", "SELL"] and alpha_score > 0:
        alpha_score = min(alpha_score, 0.10)

    if available_weight == 0:
        confidence = 0.0
    else:
        confidence = min(available_weight, 1.0)

    alpha_label = get_alpha_label(
        alpha_score=alpha_score,
        sentiment_score=sentiment_score,
        technical_score=technical_score,
    )
    main_driver = get_main_driver(
        sentiment_score=sentiment_score,
        sentiment_confidence=sentiment_confidence,
        technical_score=technical_score,
        technical_confidence=technical_confidence,
    )

    return CombinedAlphaSignal(
        sector=alpha_input.sector,
        ticker=alpha_input.ticker,
        company_name=alpha_input.company_name,
        signal_date=alpha_input.signal_date,
        sentiment_score=alpha_input.sentiment_score,
        sentiment_label=alpha_input.sentiment_label,
        sentiment_confidence=alpha_input.sentiment_confidence,
        technical_score=alpha_input.technical_score,
        technical_label=alpha_input.technical_label,
        technical_confidence=alpha_input.technical_confidence,
        chart_decision=alpha_input.chart_decision,
        chart_score=alpha_input.chart_score,
        chart_confidence=alpha_input.chart_confidence,
        trend_reading=alpha_input.trend_reading,
        breakout_status=alpha_input.breakout_status,
        volume_confirmation=alpha_input.volume_confirmation,
        entry_quality=alpha_input.entry_quality,
        support_level=alpha_input.support_level,
        resistance_level=alpha_input.resistance_level,
        buy_trigger=alpha_input.buy_trigger,
        invalid_buy_reason=alpha_input.invalid_buy_reason,
        reason_to_wait=alpha_input.reason_to_wait,
        current_price_stop_7_pct=alpha_input.current_price_stop_7_pct,
        current_price_stop_8_pct=alpha_input.current_price_stop_8_pct,
        breakout_entry_stop_7_pct=alpha_input.breakout_entry_stop_7_pct,
        breakout_entry_stop_8_pct=alpha_input.breakout_entry_stop_8_pct,
        stop_loss_7_pct=alpha_input.stop_loss_7_pct,
        stop_loss_8_pct=alpha_input.stop_loss_8_pct,
        danger_level=alpha_input.danger_level,
        alpha_score=alpha_score,
        alpha_label=alpha_label,
        confidence=confidence,
        main_driver=main_driver,
    )


def _equal_nullable_number(left: float | None, right: float | None, tolerance: float = 1e-9) -> bool:
    if left is None or right is None:
        return left is right
    return abs(left - right) <= tolerance


def _existing_signal_matches_input(existing: dict, alpha_input: AlphaInput) -> bool:
    return (
        existing["sector"] == alpha_input.sector
        and existing["ticker"] == alpha_input.ticker
        and existing["company_name"] == alpha_input.company_name
        and existing["signal_date"] == alpha_input.signal_date
        and _equal_nullable_number(existing["sentiment_score"], alpha_input.sentiment_score)
        and existing["sentiment_label"] == alpha_input.sentiment_label
        and _equal_nullable_number(existing["sentiment_confidence"], alpha_input.sentiment_confidence)
        and _equal_nullable_number(existing["technical_score"], alpha_input.technical_score)
        and existing["technical_label"] == alpha_input.technical_label
        and _equal_nullable_number(existing["technical_confidence"], alpha_input.technical_confidence)
        and existing.get("chart_decision") == alpha_input.chart_decision
        and _equal_nullable_number(existing.get("chart_score"), alpha_input.chart_score)
        and _equal_nullable_number(existing.get("chart_confidence"), alpha_input.chart_confidence)
        and existing.get("trend_reading") == alpha_input.trend_reading
        and existing.get("breakout_status") == alpha_input.breakout_status
        and existing.get("volume_confirmation") == alpha_input.volume_confirmation
        and existing.get("entry_quality") == alpha_input.entry_quality
        and _equal_nullable_number(existing.get("support_level"), alpha_input.support_level)
        and _equal_nullable_number(existing.get("resistance_level"), alpha_input.resistance_level)
        and existing.get("buy_trigger") == alpha_input.buy_trigger
        and existing.get("invalid_buy_reason") == alpha_input.invalid_buy_reason
        and existing.get("reason_to_wait") == alpha_input.reason_to_wait
        and _equal_nullable_number(existing.get("current_price_stop_7_pct"), alpha_input.current_price_stop_7_pct)
        and _equal_nullable_number(existing.get("current_price_stop_8_pct"), alpha_input.current_price_stop_8_pct)
        and _equal_nullable_number(existing.get("breakout_entry_stop_7_pct"), alpha_input.breakout_entry_stop_7_pct)
        and _equal_nullable_number(existing.get("breakout_entry_stop_8_pct"), alpha_input.breakout_entry_stop_8_pct)
        and _equal_nullable_number(existing.get("stop_loss_7_pct"), alpha_input.stop_loss_7_pct)
        and _equal_nullable_number(existing.get("stop_loss_8_pct"), alpha_input.stop_loss_8_pct)
        and _equal_nullable_number(existing.get("danger_level"), alpha_input.danger_level)
    )


def run_combined_alpha_signal_agent(sector: str, force_refresh: bool = False) -> dict:
    inputs = get_alpha_inputs(sector=sector)

    processed = 0
    saved = 0
    skipped_unchanged = 0
    strong_bullish = 0
    bullish = 0
    neutral = 0
    bearish = 0
    strong_bearish = 0
    mixed = 0
    unknown = 0

    for alpha_input in inputs:
        if not force_refresh:
            existing = get_combined_alpha_signal(
                ticker=alpha_input.ticker,
                signal_date=alpha_input.signal_date,
            )
            if existing is not None and _existing_signal_matches_input(existing, alpha_input):
                processed += 1
                skipped_unchanged += 1
                label = existing["alpha_label"]
                if label == "strong_bullish":
                    strong_bullish += 1
                elif label == "bullish":
                    bullish += 1
                elif label == "neutral":
                    neutral += 1
                elif label == "bearish":
                    bearish += 1
                elif label == "strong_bearish":
                    strong_bearish += 1
                elif label == "mixed":
                    mixed += 1
                else:
                    unknown += 1
                continue

        signal = build_combined_alpha_signal(alpha_input)
        save_combined_alpha_signal(signal)
        processed += 1
        saved += 1

        if signal.alpha_label == "strong_bullish":
            strong_bullish += 1
        elif signal.alpha_label == "bullish":
            bullish += 1
        elif signal.alpha_label == "neutral":
            neutral += 1
        elif signal.alpha_label == "bearish":
            bearish += 1
        elif signal.alpha_label == "strong_bearish":
            strong_bearish += 1
        elif signal.alpha_label == "mixed":
            mixed += 1
        elif signal.alpha_label == "unknown":
            unknown += 1

    return {
        "sector": sector,
        "processed": processed,
        "saved": saved,
        "skipped_unchanged": skipped_unchanged,
        "strong_bullish": strong_bullish,
        "bullish": bullish,
        "neutral": neutral,
        "bearish": bearish,
        "strong_bearish": strong_bearish,
        "mixed": mixed,
        "unknown": unknown,
    }
