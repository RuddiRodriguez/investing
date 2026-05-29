from chart_confirmation.chart_metrics import build_chart_metrics
from chart_confirmation.llm_chart_confirmation import (
    evaluate_chart_confirmation_with_llm,
)
from chart_confirmation.repositories import (
    get_price_bars_for_chart_confirmation,
    get_tickers_for_chart_confirmation,
    save_chart_confirmation_signal,
)
from chart_confirmation.schemas import ChartConfirmationSignal


def build_chart_confirmation_signal(
    metrics,
    decision,
) -> ChartConfirmationSignal:
    return ChartConfirmationSignal(
        sector=metrics.sector,
        ticker=metrics.ticker,
        company_name=metrics.company_name,
        signal_date=metrics.signal_date,
        current_price=metrics.current_price,
        open_price=metrics.open_price,
        high_price=metrics.high_price,
        low_price=metrics.low_price,
        close_price=metrics.close_price,
        latest_volume=metrics.latest_volume,
        trend_status=metrics.trend_status,
        trend_reading=metrics.trend_reading,
        base_status=metrics.base_status,
        support_level=metrics.support_level,
        resistance_level=metrics.resistance_level,
        breakout_status=metrics.breakout_status,
        breakout_price=metrics.breakout_price,
        volume_confirmation=metrics.volume_confirmation,
        volume_ratio=metrics.volume_ratio,
        entry_quality=metrics.entry_quality,
        extension_pct=metrics.extension_pct,
        buy_trigger=metrics.buy_trigger,
        invalid_buy_reason=metrics.invalid_buy_reason,
        reason_to_wait=metrics.reason_to_wait,
        current_price_stop_7_pct=metrics.current_price_stop_7_pct,
        current_price_stop_8_pct=metrics.current_price_stop_8_pct,
        breakout_entry_stop_7_pct=metrics.breakout_entry_stop_7_pct,
        breakout_entry_stop_8_pct=metrics.breakout_entry_stop_8_pct,
        danger_level=metrics.danger_level,
        sell_signal=metrics.sell_signal,
        chart_decision=decision.chart_decision,
        chart_score=decision.chart_score,
        chart_confidence=decision.chart_confidence,
        llm_chart_reason=decision.llm_chart_reason,
        chart_flags=decision.chart_flags,
    )


def run_chart_confirmation_agent(
    sector: str,
    price_history_limit: int = 120,
    timeframe: str = "1d",
) -> dict:
    tickers = get_tickers_for_chart_confirmation(sector=sector)
    processed = 0
    skipped = 0
    buy = 0
    wait = 0
    avoid = 0
    sell = 0
    hold = 0
    failed = []

    for item in tickers:
        ticker = item["ticker"]
        company_name = item["company_name"]
        try:
            bars = get_price_bars_for_chart_confirmation(
                ticker=ticker,
                limit=price_history_limit,
                timeframe=timeframe,
            )
            metrics = build_chart_metrics(
                sector=sector,
                ticker=ticker,
                company_name=company_name,
                bars=bars,
            )
            if metrics is None:
                skipped += 1
                continue
            decision = evaluate_chart_confirmation_with_llm(metrics)
            signal = build_chart_confirmation_signal(
                metrics=metrics,
                decision=decision,
            )
            save_chart_confirmation_signal(signal)
            processed += 1
            if signal.chart_decision == "BUY":
                buy += 1
            elif signal.chart_decision == "WAIT_FOR_BREAKOUT":
                wait += 1
            elif signal.chart_decision == "AVOID":
                avoid += 1
            elif signal.chart_decision == "SELL":
                sell += 1
            elif signal.chart_decision == "HOLD":
                hold += 1
        except Exception as error:
            failed.append(
                {
                    "ticker": ticker,
                    "error": str(error),
                }
            )

    return {
        "sector": sector,
        "tickers_found": len(tickers),
        "timeframe": timeframe,
        "processed": processed,
        "skipped": skipped,
        "buy": buy,
        "wait_for_breakout": wait,
        "avoid": avoid,
        "sell": sell,
        "hold": hold,
        "failed": failed,
    }
