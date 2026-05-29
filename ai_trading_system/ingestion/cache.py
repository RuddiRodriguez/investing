from datetime import datetime, timedelta, timezone

from ingestion.db import get_connection, utc_now


def _now() -> datetime:
    return datetime.now(timezone.utc)


def build_cache_key(*parts: object) -> str:
    return ":".join(str(part).lower().strip() for part in parts)


def cache_exists(cache_key: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT expires_at
        FROM ingestion_cache
        WHERE cache_key = ?
        """,
        (cache_key,),
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return False

    expires_at = row["expires_at"]
    if expires_at is None:
        return True

    expires_dt = datetime.fromisoformat(expires_at)
    return expires_dt > _now()


def set_cache(
    cache_key: str,
    cache_type: str,
    value: str | None = None,
    ttl_hours: int | None = None,
) -> None:
    expires_at = None
    if ttl_hours is not None:
        expires_at = (_now() + timedelta(hours=ttl_hours)).isoformat()

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO ingestion_cache (
            cache_key,
            cache_type,
            value,
            created_at,
            expires_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            cache_key,
            cache_type,
            value,
            utc_now(),
            expires_at,
        ),
    )
    conn.commit()
    conn.close()


def get_cache_value(cache_key: str) -> str | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT value, expires_at
        FROM ingestion_cache
        WHERE cache_key = ?
        """,
        (cache_key,),
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    expires_at = row["expires_at"]
    if expires_at is None:
        return row["value"]

    expires_dt = datetime.fromisoformat(expires_at)
    if expires_dt <= _now():
        return None

    return row["value"]
