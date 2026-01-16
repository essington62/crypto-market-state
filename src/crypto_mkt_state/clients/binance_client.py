"""
Binance API client for fetching OHLCV spot daily data.

This module provides a pure client for downloading historical
spot market OHLCV (kline) data from Binance.

Design principles:
- Pure client (no Kedro, no IO, no side effects)
- UTC timestamps only
- No schema opinionation (raw Binance schema preserved)
- Secrets resolved via environment variables if not provided
"""

from datetime import datetime, timezone
from typing import Optional, Tuple
import os

import pandas as pd
from binance.client import Client


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------
def _resolve_binance_credentials(
    api_key: Optional[str],
    api_secret: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve Binance credentials.

    Priority:
    1. Explicit function arguments
    2. Environment variables

    Environment variables:
    - BINANCE_API_KEY
    - BINANCE_API_SECRET
    """
    return (
        api_key or os.getenv("BINANCE_API_KEY"),
        api_secret or os.getenv("BINANCE_API_SECRET"),
    )


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def fetch_spot_daily_klines(
    symbol: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV (kline) data for a spot trading pair from Binance.

    This function downloads historical daily candlestick data
    and returns it in raw Binance schema form.

    Args:
        symbol:
            Trading pair symbol (e.g. 'BTCUSDT', 'ETHUSDT').
            Must be uppercase.
        start_date:
            Start date (inclusive). Naive datetimes are assumed UTC.
        end_date:
            End date (inclusive). Naive datetimes are assumed UTC.
        api_key:
            Optional Binance API key (public data does not require it).
        api_secret:
            Optional Binance API secret.

    Returns:
        pandas.DataFrame with columns:
            - open_time (datetime64[ns, UTC])
            - open (float)
            - high (float)
            - low (float)
            - close (float)
            - volume (float)
            - close_time (datetime64[ns, UTC])
            - quote_volume (float)
            - trades (int)
            - taker_buy_base_volume (float)
            - taker_buy_quote_volume (float)

        Rows are sorted by open_time ascending.
        No index is set.
    """
    # ----------------------------------------------------------------
    # Resolve credentials
    # ----------------------------------------------------------------
    api_key, api_secret = _resolve_binance_credentials(api_key, api_secret)
    client = Client(api_key=api_key, api_secret=api_secret)

    symbol = symbol.upper()

    # ----------------------------------------------------------------
    # Date handling (UTC → milliseconds)
    # ----------------------------------------------------------------
    start_str = None
    end_str = None

    if start_date is not None:
        start_date = _ensure_utc(start_date)
        start_str = int(start_date.timestamp() * 1000)

    if end_date is not None:
        end_date = _ensure_utc(end_date)
        end_str = int(end_date.timestamp() * 1000)

    # ----------------------------------------------------------------
    # Fetch data from Binance
    # ----------------------------------------------------------------
    klines = client.get_historical_klines(
        symbol=symbol,
        interval=Client.KLINE_INTERVAL_1DAY,
        start_str=start_str,
        end_str=end_str,
    )

    # ----------------------------------------------------------------
    # Build DataFrame (raw schema preserved)
    # ----------------------------------------------------------------
    df = pd.DataFrame(
        klines,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )

    # Timestamps → UTC
    df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(
        df["close_time"].astype(int), unit="ms", utc=True
    )

    # Numeric casting
    float_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    df[float_cols] = df[float_cols].astype(float)
    df["trades"] = df["trades"].astype(int)

    # Drop Binance filler column
    df = df.drop(columns=["ignore"])

    # Sort deterministically
    df = df.sort_values("open_time").reset_index(drop=True)

    return df
