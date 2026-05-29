from pydantic import BaseModel


class PriceBar(BaseModel):
    sector: str
    ticker: str
    timestamp: str
    timeframe: str
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    adjusted_close: float | None
    volume: int | None
    source: str
