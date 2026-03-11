"""
L1 ingestion: incremental update for Binance spot OHLCV (24/7 daily).

Layer L1 constraints:
- Raw mirror of Binance API
- No feature engineering
- No resampling, rolling windows or forward-fill
- Only timestamp parsing, dtype normalization, sorting, deduplication
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, List
from collections.abc import Callable

import pandas as pd
import requests

BINANCE_SPOT_BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"
API_LIMIT = 1000


def _fetch_binance_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int | None = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV klines from Binance REST API using forward pagination on startTime.

    Returns a raw DataFrame with Binance columns (ms timestamps, strings for numerics).
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "limit": API_LIMIT,
    }
    if end_ms is not None:
        params["endTime"] = end_ms

    url = f"{BINANCE_SPOT_BASE_URL}{KLINES_ENDPOINT}"
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return pd.DataFrame()

    # Binance kline schema:
    # [
    #   0  open time,
    #   1  open,
    #   2  high,
    #   3  low,
    #   4  close,
    #   5  volume,
    #   6  close time,
    #   7  quote asset volume,
    #   8  number of trades,
    #   9  taker buy base asset volume,
    #   10 taker buy quote asset volume,
    #   11 ignore
    # ]
    columns = [
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
    ]

    df = pd.DataFrame(data, columns=columns)
    return df


def _normalize_klines_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw Binance kline DataFrame to L1 contract:
    - open_time, close_time: datetime64[ns, UTC]
    - OHLCV and volumes: float64
    - trades: int64
    """
    if raw_df.empty:
        return raw_df

    df = raw_df.copy()

    # Timestamps in ms -> datetime64[ns, UTC]
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Numeric columns: use to_numeric + astype
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    if "trades" in df.columns:
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("int64")

    # Drop Binance's ignore column if present
    if "ignore" in df.columns:
        df = df.drop(columns=["ignore"])

    return df


def update_binance_spot_incremental(
    existing_data: Dict[str, Callable[[], pd.DataFrame]],
    spot_symbols: List[str],
    interval: str,
    global_start_date: str,
) -> Dict[str, pd.DataFrame]:
    """
    Incrementally update Binance spot OHLCV L1 dataset (PartitionedDataset).

    Parameters
    ----------
    existing_data :
        Mapping partition_key -> loader callable that returns existing L1 DataFrame.
        Typically partition_key == symbol (e.g., "BTCUSDT").
    spot_symbols :
        List of symbols to update (e.g., ["BTCUSDT", "ETHUSDT", ...]).
    interval :
        Binance kline interval string (e.g., "1d").
    global_start_date :
        ISO date string (UTC) used as initial start_time when there is no existing data
        for a given symbol (e.g., params:global.start_date).

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping partition_key -> updated DataFrame with new candles appended
        and deduplicated on open_time, sorted in ascending order.
    """
    if not spot_symbols:
        raise ValueError(
            "ingestion.binance.spot_incremental: params:binance.spot_symbols is empty."
        )

    start_date_utc = pd.to_datetime(global_start_date, utc=True)

    now_utc = datetime.now(timezone.utc)
    now_ms = int(now_utc.timestamp() * 1000)

    updated: Dict[str, pd.DataFrame] = {}

    for symbol in spot_symbols:
        # Load existing data if present
        if symbol in existing_data:
            df_existing = existing_data[symbol]()
        else:
            df_existing = pd.DataFrame()

        df_existing = df_existing.copy()

        # Determine start_ms for incremental fetch
        if not df_existing.empty and "open_time" in df_existing.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_existing["open_time"]):
                df_existing["open_time"] = pd.to_datetime(
                    df_existing["open_time"], utc=True
                )
            last_open_time = df_existing["open_time"].max()
            if last_open_time.tzinfo is None:
                last_open_time = last_open_time.tz_localize("UTC")
            start_ts = last_open_time + pd.Timedelta(milliseconds=1)
        else:
            start_ts = start_date_utc

        start_ms = int(start_ts.timestamp() * 1000)

        # Already up-to-date
        if start_ms >= now_ms:
            if not df_existing.empty:
                df_existing = df_existing.sort_values("open_time")
                df_existing = df_existing.drop_duplicates("open_time", keep="last")
            updated[symbol] = df_existing
            continue

        # Forward pagination from start_ms to now_ms
        pages: List[pd.DataFrame] = []
        current_start_ms = start_ms

        while current_start_ms < now_ms:
            raw_page = _fetch_binance_klines(
                symbol=symbol,
                interval=interval,
                start_ms=current_start_ms,
                end_ms=now_ms,
            )

            if raw_page.empty:
                break

            page = _normalize_klines_df(raw_page)
            if page.empty:
                break

            page = page.sort_values("open_time")
            pages.append(page)

            last_open = page["open_time"].max()
            last_open_ms = int(last_open.timestamp() * 1000)

            if last_open_ms >= now_ms or len(page) < API_LIMIT:
                break

            current_start_ms = last_open_ms + 1
            time.sleep(0.4)  # L1: respect Binance rate limits

        if pages:
            df_new = pd.concat(pages, axis=0, ignore_index=True)
        else:
            df_new = pd.DataFrame()

        # Merge existing + new
        if not df_existing.empty and not df_new.empty:
            df_merged = pd.concat([df_existing, df_new], axis=0, ignore_index=True)
        elif df_existing.empty and not df_new.empty:
            df_merged = df_new
        else:
            df_merged = df_existing

        if df_merged.empty:
            updated[symbol] = df_merged
            continue

        # Final L1 contract: UTC timestamps, sorted, deduped
        if not pd.api.types.is_datetime64_any_dtype(df_merged["open_time"]):
            df_merged["open_time"] = pd.to_datetime(df_merged["open_time"], utc=True)
        if "close_time" in df_merged.columns and not pd.api.types.is_datetime64_any_dtype(
            df_merged["close_time"]
        ):
            df_merged["close_time"] = pd.to_datetime(
                df_merged["close_time"], utc=True
            )

        df_merged = df_merged.sort_values("open_time")
        df_merged = df_merged.drop_duplicates("open_time", keep="last")

        updated[symbol] = df_merged

    return updated