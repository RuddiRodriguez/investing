from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CalendarSummary:
    calendar: str
    start_date: str | None
    end_date: str | None
    expected_sessions: int
    observed_sessions: int
    missing_sessions: list[str]
    extra_sessions: list[str]
    calendar_source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "calendar": self.calendar,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "expected_sessions": self.expected_sessions,
            "observed_sessions": self.observed_sessions,
            "missing_sessions": self.missing_sessions,
            "extra_sessions": self.extra_sessions,
            "calendar_source": self.calendar_source,
        }


def expected_trading_sessions(
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    calendar: str = "XNYS",
) -> tuple[pd.DatetimeIndex, str]:
    """Return expected trading sessions using exchange_calendars when available.

    The project does not require exchange_calendars as a hard dependency. When it
    is unavailable, business days are used as a deterministic fallback.
    """

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if end_ts < start_ts:
        return pd.DatetimeIndex([], name="date"), "empty_range"

    try:
        import exchange_calendars as xcals  # type: ignore

        exchange = xcals.get_calendar(calendar)
        sessions = exchange.sessions_in_range(start_ts, end_ts)
        sessions = pd.DatetimeIndex(sessions).tz_localize(None).normalize()
        return sessions, "exchange_calendars"
    except Exception:
        sessions = pd.bdate_range(start_ts, end_ts, name="date")
        return pd.DatetimeIndex(sessions).normalize(), "pandas_business_day_fallback"


def summarize_calendar_alignment(prices: pd.DataFrame, calendar: str = "XNYS") -> dict[str, object]:
    if prices.empty:
        return CalendarSummary(
            calendar=calendar,
            start_date=None,
            end_date=None,
            expected_sessions=0,
            observed_sessions=0,
            missing_sessions=[],
            extra_sessions=[],
            calendar_source="empty",
        ).to_dict()

    observed = pd.DatetimeIndex(prices.index).tz_localize(None).normalize().unique().sort_values()
    expected, source = expected_trading_sessions(observed.min(), observed.max(), calendar=calendar)
    expected_set = set(expected)
    observed_set = set(observed)
    missing = sorted(expected_set - observed_set)
    extra = sorted(observed_set - expected_set)

    return CalendarSummary(
        calendar=calendar,
        start_date=str(observed.min().date()),
        end_date=str(observed.max().date()),
        expected_sessions=int(len(expected)),
        observed_sessions=int(len(observed)),
        missing_sessions=[str(value.date()) for value in missing[:250]],
        extra_sessions=[str(value.date()) for value in extra[:250]],
        calendar_source=source,
    ).to_dict()


def normalize_session_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a datetime index to naive midnight session dates."""

    output = frame.copy()
    index = pd.DatetimeIndex(output.index)
    if index.tz is not None:
        index = index.tz_convert(None)
    output.index = index.normalize()
    output = output.sort_index()
    return output[~output.index.duplicated(keep="last")]
