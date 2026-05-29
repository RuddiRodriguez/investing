from statistics import mean

from chart_confirmation.schemas import ChartMetrics, PriceBarForChart


def get_price(bar: PriceBarForChart) -> float:
    return bar.adjusted_close if bar.adjusted_close is not None else bar.close


def safe_mean(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return mean(clean)


def safe_return(
    current: float,
    previous: float | None,
) -> float | None:
    if previous is None or previous == 0:
        return None
    return (current / previous) - 1


def calculate_trend_status(
    prices: list[float],
) -> str:
    if len(prices) < 60:
        return "unknown"
    current_price = prices[-1]
    sma_20 = safe_mean(prices[-20:])
    sma_50 = safe_mean(prices[-50:])
    price_20d_ago = prices[-21]
    price_60d_ago = prices[-61]
    return_20d = safe_return(current_price, price_20d_ago)
    return_60d = safe_return(current_price, price_60d_ago)
    if (
        sma_20 is not None
        and sma_50 is not None
        and return_20d is not None
        and return_60d is not None
    ):
        if (
            current_price > sma_20
            and sma_20 > sma_50
            and return_20d > 0
            and return_60d > 0
        ):
            return "uptrend"
        if (
            current_price < sma_20
            and sma_20 < sma_50
            and return_20d < 0
            and return_60d < 0
        ):
            return "downtrend"
    return "sideways"


def calculate_trend_reading(
    prices: list[float],
) -> str:
    if len(prices) < 60:
        return "unknown"
    current_price = prices[-1]
    price_20d_ago = prices[-21]
    price_60d_ago = prices[-61]
    return_20d = safe_return(current_price, price_20d_ago)
    return_60d = safe_return(current_price, price_60d_ago)
    sma_20 = safe_mean(prices[-20:])
    sma_50 = safe_mean(prices[-50:])
    if (
        return_20d is None
        or return_60d is None
        or sma_20 is None
        or sma_50 is None
    ):
        return "unknown"
    if (
        current_price > sma_20
        and sma_20 > sma_50
        and return_20d > 0.05
        and return_60d > 0.10
    ):
        return "strong_upward"
    if current_price > sma_20 and return_20d > 0:
        return "weak_upward"
    if (
        current_price < sma_20
        and sma_20 < sma_50
        and return_20d < -0.05
        and return_60d < -0.10
    ):
        return "strong_downward"
    if current_price < sma_20 and return_20d < 0:
        return "weak_downward"
    return "sideways"


def calculate_resistance_level(
    prices: list[float],
) -> float | None:
    if len(prices) < 60:
        return None
    previous_range = prices[-55:-5]
    if not previous_range:
        return None
    return max(previous_range)


def calculate_support_level(
    prices: list[float],
) -> float | None:
    if len(prices) < 60:
        return None
    previous_range = prices[-55:-5]
    if not previous_range:
        return None
    return min(previous_range)


def calculate_volume_ratio(
    bars: list[PriceBarForChart],
) -> float | None:
    if len(bars) < 20:
        return None
    latest_volume = bars[-1].volume
    if latest_volume is None:
        return None
    recent_volumes = [
        float(bar.volume)
        for bar in bars[-20:]
        if bar.volume is not None
    ]
    if not recent_volumes:
        return None
    avg_volume = mean(recent_volumes)
    if avg_volume == 0:
        return None
    return float(latest_volume) / avg_volume


def calculate_breakout_status(
    prices: list[float],
    resistance_level: float | None,
) -> str:
    if resistance_level is None or len(prices) < 10:
        return "unknown"
    current_price = prices[-1]
    previous_price = prices[-2]
    if current_price > resistance_level * 1.01:
        return "confirmed_breakout"
    if resistance_level * 0.98 <= current_price <= resistance_level:
        return "near_breakout"
    was_above_recently = any(
        price > resistance_level * 1.01
        for price in prices[-5:-1]
    )
    if was_above_recently and current_price < resistance_level:
        return "failed_breakout"
    if previous_price > resistance_level and current_price < resistance_level:
        return "failed_breakout"
    return "no_breakout"


def calculate_volume_confirmation(
    bars: list[PriceBarForChart],
    volume_ratio: float | None,
) -> str:
    if len(bars) < 2 or volume_ratio is None:
        return "unknown"
    previous_price = get_price(bars[-2])
    current_price = get_price(bars[-1])
    if current_price < previous_price and volume_ratio >= 1.4:
        return "distribution_volume"
    if current_price > previous_price and volume_ratio >= 1.4:
        return "strong_volume"
    if volume_ratio < 1.1:
        return "weak_volume"
    return "unknown"


def calculate_base_status(
    prices: list[float],
) -> str:
    if len(prices) < 60:
        return "unknown"
    recent_20 = prices[-20:]
    recent_60 = prices[-60:]
    current_price = prices[-1]
    high_60 = max(recent_60)
    low_20 = min(recent_20)
    high_20 = max(recent_20)
    if high_60 == 0 or low_20 == 0:
        return "unknown"
    recent_range_pct = (high_20 / low_20) - 1
    distance_from_60_high = (current_price / high_60) - 1
    if recent_range_pct <= 0.15 and distance_from_60_high >= -0.15:
        return "building_base"
    if current_price > high_60 * 1.10:
        return "extended"
    return "no_base"


def calculate_entry_quality(
    current_price: float,
    resistance_level: float | None,
    breakout_status: str,
) -> tuple[str, float | None]:
    if resistance_level is None:
        return "avoid", None
    extension_pct = (current_price / resistance_level) - 1
    if breakout_status == "confirmed_breakout":
        if extension_pct <= 0.05:
            return "proper_entry", extension_pct
        if extension_pct <= 0.08:
            return "too_late", extension_pct
        return "overextended", extension_pct
    if breakout_status == "near_breakout":
        return "too_early", extension_pct
    if breakout_status == "failed_breakout":
        return "avoid", extension_pct
    return "avoid", extension_pct


def calculate_sell_signal(
    breakout_status: str,
    volume_confirmation: str,
    entry_quality: str,
) -> str:
    if breakout_status == "failed_breakout":
        return "failed_breakout"
    if volume_confirmation == "distribution_volume":
        return "heavy_distribution"
    if entry_quality == "overextended":
        return "overextended_reversal"
    return "none"


def calculate_danger_level(
    support_level: float | None,
    current_price_stop_8_pct: float | None,
) -> float | None:
    if support_level is not None:
        return support_level
    return current_price_stop_8_pct


def build_buy_trigger(
    resistance_level: float | None,
) -> str:
    if resistance_level is None:
        return "No valid buy trigger because resistance could not be identified."
    return (
        f"Break above {resistance_level:.2f}, hold above that level, "
        "and confirm the move with strong volume."
    )


def build_invalid_buy_reason(
    trend_reading: str,
    breakout_status: str,
    volume_confirmation: str,
    entry_quality: str,
    resistance_level: float | None,
) -> str:
    reasons = []
    if trend_reading in ["weak_downward", "strong_downward"]:
        reasons.append("trend is weak")
    if resistance_level is None:
        reasons.append("resistance is not clearly identified")
    if breakout_status != "confirmed_breakout":
        reasons.append("breakout is not confirmed")
    if volume_confirmation != "strong_volume":
        reasons.append("volume confirmation is weak or missing")
    if entry_quality != "proper_entry":
        reasons.append(f"entry quality is {entry_quality}")
    if not reasons:
        return "No invalid buy reason detected."
    return "Invalid buy setup because " + ", ".join(reasons) + "."


def build_reason_to_wait(
    breakout_status: str,
    resistance_level: float | None,
    volume_confirmation: str,
) -> str:
    if breakout_status == "near_breakout" and resistance_level is not None:
        return (
            f"Wait for price to break above {resistance_level:.2f} "
            "with strong volume before considering a buy."
        )
    if breakout_status == "no_breakout" and resistance_level is not None:
        return (
            f"Price is still below resistance at {resistance_level:.2f}; "
            "wait for confirmed breakout with stronger volume."
        )
    if volume_confirmation != "strong_volume":
        return "Wait for stronger volume confirmation."
    return "No wait condition detected."


def build_chart_metrics(
    sector: str,
    ticker: str,
    company_name: str,
    bars: list[PriceBarForChart],
) -> ChartMetrics | None:
    if len(bars) < 60:
        return None
    prices = [get_price(bar) for bar in bars]
    latest_bar = bars[-1]
    current_price = prices[-1]
    signal_date = latest_bar.timestamp
    trend_status = calculate_trend_status(prices)
    trend_reading = calculate_trend_reading(prices)
    base_status = calculate_base_status(prices)
    support_level = calculate_support_level(prices)
    resistance_level = calculate_resistance_level(prices)
    breakout_status = calculate_breakout_status(
        prices=prices,
        resistance_level=resistance_level,
    )
    volume_ratio = calculate_volume_ratio(bars)
    volume_confirmation = calculate_volume_confirmation(
        bars=bars,
        volume_ratio=volume_ratio,
    )
    entry_quality, extension_pct = calculate_entry_quality(
        current_price=current_price,
        resistance_level=resistance_level,
        breakout_status=breakout_status,
    )
    current_price_stop_7_pct = current_price * 0.93
    current_price_stop_8_pct = current_price * 0.92
    if resistance_level is not None:
        breakout_entry_stop_7_pct = resistance_level * 0.93
        breakout_entry_stop_8_pct = resistance_level * 0.92
    else:
        breakout_entry_stop_7_pct = None
        breakout_entry_stop_8_pct = None
    danger_level = calculate_danger_level(
        support_level=support_level,
        current_price_stop_8_pct=current_price_stop_8_pct,
    )
    sell_signal = calculate_sell_signal(
        breakout_status=breakout_status,
        volume_confirmation=volume_confirmation,
        entry_quality=entry_quality,
    )
    buy_trigger = build_buy_trigger(
        resistance_level=resistance_level,
    )
    invalid_buy_reason = build_invalid_buy_reason(
        trend_reading=trend_reading,
        breakout_status=breakout_status,
        volume_confirmation=volume_confirmation,
        entry_quality=entry_quality,
        resistance_level=resistance_level,
    )
    reason_to_wait = build_reason_to_wait(
        breakout_status=breakout_status,
        resistance_level=resistance_level,
        volume_confirmation=volume_confirmation,
    )
    return ChartMetrics(
        sector=sector,
        ticker=ticker,
        company_name=company_name,
        signal_date=signal_date,
        current_price=current_price,
        open_price=latest_bar.open,
        high_price=latest_bar.high,
        low_price=latest_bar.low,
        close_price=latest_bar.close,
        latest_volume=float(latest_bar.volume) if latest_bar.volume is not None else None,
        trend_status=trend_status,
        trend_reading=trend_reading,
        base_status=base_status,
        support_level=support_level,
        resistance_level=resistance_level,
        breakout_status=breakout_status,
        breakout_price=resistance_level,
        volume_confirmation=volume_confirmation,
        volume_ratio=volume_ratio,
        entry_quality=entry_quality,
        extension_pct=extension_pct,
        buy_trigger=buy_trigger,
        invalid_buy_reason=invalid_buy_reason,
        reason_to_wait=reason_to_wait,
        current_price_stop_7_pct=current_price_stop_7_pct,
        current_price_stop_8_pct=current_price_stop_8_pct,
        breakout_entry_stop_7_pct=breakout_entry_stop_7_pct,
        breakout_entry_stop_8_pct=breakout_entry_stop_8_pct,
        danger_level=danger_level,
        sell_signal=sell_signal,
    )
