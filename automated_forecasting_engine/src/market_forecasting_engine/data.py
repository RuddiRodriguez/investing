from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


_COLUMN_ALIASES = {
    "adj close": "close",
    "adjusted close": "close",
    "close": "close",
    "open": "open",
    "high": "high",
    "low": "low",
    "volume": "volume",
    "dividends": "dividends",
    "stock splits": "stock_splits",
}


def normalize_price_frame(prices: pd.DataFrame, target_column: str = "close") -> pd.DataFrame:
    """Return a sorted OHLCV-like frame with lowercase column names and a datetime index."""

    if prices.empty:
        raise ValueError("Price data is empty.")

    frame = prices.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = ["_".join(str(part) for part in col if part) for col in frame.columns]

    rename_map: dict[str, str] = {}
    for column in frame.columns:
        rename_map[column] = _canonical_column_name(column)
    frame = frame.rename(columns=rename_map)

    if not isinstance(frame.index, pd.DatetimeIndex):
        first_column = frame.columns[0]
        parsed = pd.to_datetime(frame[first_column], errors="coerce", utc=True)
        if parsed.notna().mean() > 0.8:
            frame = frame.drop(columns=[first_column])
            frame.index = parsed.dt.tz_convert(None)
        else:
            raise ValueError("Price data must have a DatetimeIndex or a parseable first date column.")
    elif frame.index.tz is not None:
        frame.index = frame.index.tz_convert(None)

    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    target = target_column.lower()
    if target not in frame.columns:
        raise ValueError(f"Price data must include a `{target}` column.")

    numeric_columns = []
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().any():
            frame[column] = converted
            numeric_columns.append(column)

    return frame[numeric_columns].dropna(subset=[target])


def _canonical_column_name(column: object) -> str:
    normalized = str(column).strip().lower().replace("_", " ")
    if normalized in _COLUMN_ALIASES:
        return _COLUMN_ALIASES[normalized]

    # yfinance may return single-ticker MultiIndex columns flattened as
    # Close_AAPL, Volume_AAPL, etc. Keep those as canonical OHLCV names.
    for raw_name, canonical_name in sorted(_COLUMN_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if normalized.startswith(f"{raw_name} "):
            return canonical_name

    return normalized.replace(" ", "_")


def load_price_csv(path: str | Path, target_column: str = "close") -> pd.DataFrame:
    """Load historical prices from a CSV file."""

    raw = pd.read_csv(path)
    return normalize_price_frame(raw, target_column=target_column)


_AVAILABILITY_COLUMNS = ("available_at", "availability_date", "release_date", "published_at")
_REVISION_COLUMNS = ("revision_as_of", "vintage", "as_of")


def load_indicator_csv(
    path: str | Path,
    prefix: str,
    release_lag_days: int = 0,
    date_column: str | None = None,
    availability_column: str | None = None,
) -> pd.DataFrame:
    """Load macro/rate indicator CSV data with point-in-time availability.

    The first date-like column is treated as the economic observation date. If an
    availability/release column exists, that date becomes the index used for
    modeling. Otherwise, release_lag_days is applied to the observation date.
    """

    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"Indicator CSV `{path}` is empty.")

    observation_column = date_column or str(raw.columns[0])
    dates = pd.to_datetime(raw[observation_column], errors="coerce", utc=True)
    if dates.notna().mean() <= 0.8:
        raise ValueError(f"Indicator CSV `{path}` must have a parseable date column first.")

    available_dates = _availability_dates(
        raw=raw,
        observation_dates=dates,
        availability_column=availability_column,
        release_lag_days=release_lag_days,
    )
    metadata_columns = {observation_column}
    metadata_columns.update(_matching_columns(raw.columns, _AVAILABILITY_COLUMNS))
    metadata_columns.update(_matching_columns(raw.columns, _REVISION_COLUMNS))

    frame = raw.drop(columns=[column for column in metadata_columns if column in raw.columns]).copy()
    frame.index = available_dates.dt.tz_convert(None)
    frame = frame.sort_index()
    revision_column = _first_matching_column(raw.columns, _REVISION_COLUMNS)
    if revision_column is not None:
        revisions = pd.to_datetime(raw[revision_column], errors="coerce", utc=True).dt.tz_convert(None)
        frame["__revision_as_of"] = revisions.to_numpy()
        frame = frame.sort_values("__revision_as_of").drop(columns=["__revision_as_of"])
    frame = frame[~frame.index.duplicated(keep="last")]

    output = pd.DataFrame(index=frame.index)
    clean_prefix = _safe_label(prefix)
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().any():
            output[f"{clean_prefix}_{_safe_label(str(column))}"] = converted
    if output.empty:
        raise ValueError(f"Indicator CSV `{path}` has no numeric columns.")
    return output


def load_event_indicators(
    path: str | Path,
    target_index: pd.DatetimeIndex,
    prefix: str = "event",
    release_lag_days: int = 0,
    date_column: str | None = None,
    availability_column: str | None = None,
) -> pd.DataFrame:
    """Load dated corporate/event rows and align available event indicators."""

    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"Event CSV `{path}` is empty.")

    event_date_column = date_column or str(raw.columns[0])
    parsed_dates = pd.to_datetime(raw[event_date_column], errors="coerce", utc=True)
    available_dates = _availability_dates(
        raw=raw,
        observation_dates=parsed_dates,
        availability_column=availability_column,
        release_lag_days=release_lag_days,
    ).dt.tz_convert(None)
    dates = parsed_dates.dt.tz_convert(None)
    events = raw.loc[dates.notna()].copy()
    available_dates = available_dates.loc[dates.notna()]
    event_dates = pd.DatetimeIndex(available_dates.dt.normalize())

    output = pd.DataFrame(index=pd.DatetimeIndex(target_index).normalize().unique())
    clean_prefix = _safe_label(prefix)
    counts = pd.Series(1.0, index=event_dates).groupby(level=0).sum()
    output[f"{clean_prefix}_count"] = counts.reindex(output.index).fillna(0.0)

    type_column = _find_event_type_column(events)
    if type_column is not None:
        event_types = events[type_column].fillna("unknown").astype(str).map(_safe_label)
        typed = pd.DataFrame({"date": event_dates, "event_type": event_types})
        for event_type, group in typed.groupby("event_type"):
            counts_by_type = pd.Series(1.0, index=pd.DatetimeIndex(group["date"])).groupby(level=0).sum()
            output[f"{clean_prefix}_{event_type}"] = counts_by_type.reindex(output.index).fillna(0.0)

    aligned = output.reindex(pd.DatetimeIndex(target_index).normalize())
    aligned.index = target_index
    return aligned


def fetch_yahoo_prices(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    target_column: str = "close",
) -> pd.DataFrame:
    """Fetch daily prices from Yahoo Finance."""

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required for Yahoo Finance downloads.") from exc

    prices = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    return normalize_price_frame(prices, target_column=target_column)


def enrich_price_frame(
    prices: pd.DataFrame,
    yahoo_context: dict[str, str] | None = None,
    start: str | None = None,
    end: str | None = None,
    indicator_csvs: dict[str, str | Path] | None = None,
    event_csvs: dict[str, str | Path] | None = None,
    target_column: str = "close",
) -> pd.DataFrame:
    """Join optional benchmark, macro, rate, and event data into the price frame."""

    enriched = normalize_price_frame(prices, target_column=target_column)
    start_date = start or str(enriched.index.min().date())
    end_date = end

    for label, ticker in (yahoo_context or {}).items():
        if not ticker:
            continue
        context = fetch_yahoo_prices(ticker, start=start_date, end=end_date, target_column="close")
        close = context["close"].rename(f"{_safe_label(label)}_{_safe_label(ticker)}")
        enriched = enriched.join(close, how="left")

    for prefix, path in (indicator_csvs or {}).items():
        indicators = load_indicator_csv(path, prefix=prefix)
        enriched = enriched.join(indicators, how="left")

    for prefix, path in (event_csvs or {}).items():
        events = load_event_indicators(path, enriched.index, prefix=prefix)
        enriched = enriched.join(events, how="left")

    base_columns = {"open", "high", "low", "close", "volume", "dividends", "stock_splits"}
    external_columns = [column for column in enriched.columns if column not in base_columns]
    if external_columns:
        enriched[external_columns] = enriched[external_columns].ffill()
    return enriched.dropna(subset=[target_column])


def data_version_hash(prices: pd.DataFrame) -> str:
    """Create a stable hash for the data used by a forecast run."""

    normalized = prices.copy()
    normalized.index = pd.to_datetime(normalized.index)
    normalized = normalized.sort_index()
    payload = normalized.to_csv(index=True, float_format="%.12g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_label(label: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in label).strip("_")


def _find_event_type_column(events: pd.DataFrame) -> str | None:
    for candidate in ("event_type", "type", "event", "category"):
        for column in events.columns:
            if _safe_label(str(column)) == candidate:
                return str(column)
    return None


def _availability_dates(
    raw: pd.DataFrame,
    observation_dates: pd.Series,
    availability_column: str | None,
    release_lag_days: int,
) -> pd.Series:
    available_column = availability_column or _first_matching_column(raw.columns, _AVAILABILITY_COLUMNS)
    if available_column is not None:
        available = pd.to_datetime(raw[available_column], errors="coerce", utc=True)
        available = available.where(available.notna(), observation_dates)
    else:
        available = observation_dates
    if release_lag_days:
        available = available + pd.offsets.BDay(int(release_lag_days))
    return available


def _matching_columns(columns: pd.Index, candidates: tuple[str, ...]) -> list[str]:
    normalized = {_safe_label(str(column)): str(column) for column in columns}
    return [normalized[candidate] for candidate in candidates if candidate in normalized]


def _first_matching_column(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    matches = _matching_columns(columns, candidates)
    return matches[0] if matches else None
