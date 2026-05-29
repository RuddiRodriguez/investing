import math
from statistics import mean, stdev

from technical.repositories import (
    get_price_bars_for_ticker,
    get_tickers_with_price_data,
    save_technical_signal,
    get_latest_price_date_for_ticker,
    technical_signal_exists,
)
from technical.schemas import PriceBarInput, TechnicalSignal


def safe_return(
    current: float,
    previous: float | None,
) -> float | None:
    if previous is None:
        return None
    if previous == 0:
        return None
    return (current / previous) - 1


def safe_mean(values: list[float]) -> float | None:
    clean_values = [value for value in values if value is not None]
    if not clean_values:
        return None
    return mean(clean_values)


def calculate_daily_returns(prices: list[float]) -> list[float]:
    returns = []
    for index in range(1, len(prices)):
        previous = prices[index - 1]
        current = prices[index]
        if previous == 0:
            continue
        returns.append((current / previous) - 1)
    return returns


def calculate_volatility_20d(prices: list[float]) -> float | None:
    if len(prices) < 21:
        return None
    recent_prices = prices[-21:]
    returns = calculate_daily_returns(recent_prices)
    if len(returns) < 2:
        return None
    return stdev(returns) * math.sqrt(252)


def calculate_max_drawdown(prices: list[float]) -> float | None:
    if len(prices) < 2:
        return None
    peak = prices[0]
    max_drawdown = 0.0
    for price in prices:
        if price > peak:
            peak = price
        if peak == 0:
            continue
        drawdown = (price / peak) - 1
        if drawdown < max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


def normalize_momentum(value: float | None) -> float:
    if value is None:
        return 0.0
    if value >= 0.20:
        return 1.0
    if value <= -0.20:
        return -1.0
    return value / 0.20


def normalize_trend(price_vs_sma: float | None) -> float:
    if price_vs_sma is None:
        return 0.0
    if price_vs_sma >= 0.10:
        return 1.0
    if price_vs_sma <= -0.10:
        return -1.0
    return price_vs_sma / 0.10


def normalize_risk(
    volatility_20d: float | None,
    max_drawdown_60d: float | None,
) -> float:
    volatility_penalty = 0.0
    drawdown_penalty = 0.0
    if volatility_20d is not None:
        volatility_penalty = min(volatility_20d / 0.80, 1.0)
    if max_drawdown_60d is not None:
        drawdown_penalty = min(abs(max_drawdown_60d) / 0.30, 1.0)
    return max(volatility_penalty, drawdown_penalty)


def get_technical_label(score: float) -> str:
    if score >= 0.25:
        return "bullish"
    if score <= -0.25:
        return "bearish"
    return "neutral"


def calculate_confidence(
    bars_count: int,
    volatility_20d: float | None,
) -> float:
    data_confidence = min(bars_count / 60, 1.0)
    volatility_confidence = 1.0
    if volatility_20d is not None:
        if volatility_20d > 1.0:
            volatility_confidence = 0.60
        elif volatility_20d > 0.80:
            volatility_confidence = 0.75
        elif volatility_20d > 0.60:
            volatility_confidence = 0.85
    confidence = data_confidence * volatility_confidence
    return max(0.0, min(confidence, 1.0))


def build_technical_signal(
    bars: list[PriceBarInput],
) -> TechnicalSignal | None:
    if len(bars) < 20:
        return None

    latest = bars[-1]
    prices = [
        bar.adjusted_close if bar.adjusted_close is not None else bar.close
        for bar in bars
    ]

    current_price = prices[-1]
    price_5d_ago = prices[-6] if len(prices) >= 6 else None
    price_20d_ago = prices[-21] if len(prices) >= 21 else None
    price_60d_ago = prices[-61] if len(prices) >= 61 else None

    return_5d = safe_return(current_price, price_5d_ago)
    return_20d = safe_return(current_price, price_20d_ago)
    return_60d = safe_return(current_price, price_60d_ago)

    sma_20 = safe_mean(prices[-20:]) if len(prices) >= 20 else None
    sma_50 = safe_mean(prices[-50:]) if len(prices) >= 50 else None

    price_vs_sma_20 = safe_return(current_price, sma_20)
    price_vs_sma_50 = safe_return(current_price, sma_50)

    volatility_20d = calculate_volatility_20d(prices)
    max_drawdown_60d = calculate_max_drawdown(prices[-60:]) if len(prices) >= 60 else None

    momentum_20 = normalize_momentum(return_20d)
    momentum_60 = normalize_momentum(return_60d)
    momentum_score = 0.60 * momentum_20 + 0.40 * momentum_60

    trend_20 = normalize_trend(price_vs_sma_20)
    trend_50 = normalize_trend(price_vs_sma_50)
    trend_score = 0.50 * trend_20 + 0.50 * trend_50

    risk_penalty = normalize_risk(
        volatility_20d=volatility_20d,
        max_drawdown_60d=max_drawdown_60d,
    )

    technical_score = (
        0.45 * momentum_score
        + 0.40 * trend_score
        - 0.15 * risk_penalty
    )
    technical_score = max(-1.0, min(technical_score, 1.0))

    technical_label = get_technical_label(technical_score)
    confidence = calculate_confidence(
        bars_count=len(bars),
        volatility_20d=volatility_20d,
    )

    return TechnicalSignal(
        sector=latest.sector,
        ticker=latest.ticker,
        signal_date=latest.timestamp,
        close=current_price,
        return_5d=return_5d,
        return_20d=return_20d,
        return_60d=return_60d,
        volatility_20d=volatility_20d,
        sma_20=sma_20,
        sma_50=sma_50,
        price_vs_sma_20=price_vs_sma_20,
        price_vs_sma_50=price_vs_sma_50,
        max_drawdown_60d=max_drawdown_60d,
        technical_score=technical_score,
        technical_label=technical_label,
        confidence=confidence,
    )


def run_technical_signal_agent(
    sector: str,
    price_history_limit: int = 120,
    timeframe: str = "1d",
) -> dict:
    tickers = get_tickers_with_price_data(sector=sector, timeframe=timeframe)

    processed = 0
    skipped = 0
    failed = []
    bullish = 0
    bearish = 0
    neutral = 0

    for ticker in tickers:
        try:
            latest_price_date = get_latest_price_date_for_ticker(ticker, timeframe=timeframe)
            if latest_price_date is None:
                skipped += 1
                continue

            if technical_signal_exists(ticker=ticker, signal_date=latest_price_date):
                skipped += 1
                continue

            bars = get_price_bars_for_ticker(
                ticker=ticker,
                limit=price_history_limit,
                timeframe=timeframe,
            )
            signal = build_technical_signal(bars)
            if signal is None:
                skipped += 1
                continue

            save_technical_signal(signal)
            processed += 1

            if signal.technical_label == "bullish":
                bullish += 1
            elif signal.technical_label == "bearish":
                bearish += 1
            else:
                neutral += 1
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
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
        "failed": failed,
    }
