"""
L1 Ingestion — CoinGlass Order Book Depth (4h) incremental.

Endpoint: /api/futures/orderbook/ask-bids-history
  - No pagination — single request, max 1000 records (most recent candles).
  - No start_time / end_time params — they cause 400 errors; never use them.

Incremental strategy:
  - Load existing partition (if present) from PartitionedDataset
  - Fetch latest ≤1000 records from API
  - Merge: concat + drop_duplicates on timestamp + sort ascending
  - Idempotent: safe to run multiple times per day

Partition scheme (one file per asset × range):
  {asset}_r05   (range 0.5%)
  {asset}_r1    (range 1%)
  {asset}_r2    (range 2%)

L1 Contract:
  - Mirror API response — no feature engineering, aggregations, or windows
  - All timestamps: datetime64[ns, UTC]
  - All numerics: float64 via pd.to_numeric(..., errors="coerce")
  - drop_duplicates on timestamp is mandatory
  - NaN permitted where API returns missing data
  - API key via function argument only — never hardcoded
"""

from __future__ import annotations

import logging
import time
from typing import Dict

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://open-api-v4.coinglass.com"
_ORDERBOOK_PATH = "/api/futures/orderbook/ask-bids-history"
_VALUE_COLS = ["bids_usd", "bids_quantity", "asks_usd", "asks_quantity"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _range_to_partition_key(asset: str, range_str: str) -> str:
    """Map (asset, range string) → partition key.

    Examples
    --------
    "BTCUSDT", "0.5"  →  "BTCUSDT_r05"
    "BTCUSDT", "1"    →  "BTCUSDT_r1"
    "BTCUSDT", "2"    →  "BTCUSDT_r2"
    """
    suffix = range_str.replace(".", "")
    return f"{asset}_r{suffix}"


def _fetch_orderbook(
    exchange: str,
    symbol: str,
    interval: str,
    limit: int,
    range_pct: str,
    api_key: str,
) -> list[dict] | None:
    """Single GET to the CoinGlass order book history endpoint.

    Returns the list of records on success.
    Logs a warning and returns None on any failure — never raises.
    """
    headers = {"accept": "application/json", "CG-API-KEY": api_key}
    params = {
        "exchange": exchange,
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "range": range_pct,
    }
    try:
        resp = requests.get(
            _BASE_URL + _ORDERBOOK_PATH,
            headers=headers,
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") != "0":
            logger.warning(
                "orderbook/%s range=%s: API error code=%s msg=%s",
                symbol,
                range_pct,
                body.get("code"),
                body.get("msg"),
            )
            return None
        return body.get("data", [])
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "orderbook/%s range=%s: request failed: %s", symbol, range_pct, exc
        )
        return None


def _records_to_df(records: list[dict], range_pct: str) -> pd.DataFrame:
    """Convert raw API records to an L1-compliant DataFrame.

    Output columns
    --------------
    timestamp       datetime64[ns, UTC]  — from ``time`` field (Unix ms)
    bids_usd        float64
    bids_quantity   float64
    asks_usd        float64
    asks_quantity   float64
    range_pct       float64              — which depth bucket this partition covers
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Parse Unix milliseconds → UTC datetime
    df["timestamp"] = pd.to_datetime(
        pd.to_numeric(df["time"], errors="coerce").astype("int64"),
        unit="ms",
        utc=True,
    )

    # Ensure all value columns exist (NaN if API omits a field)
    for col in _VALUE_COLS:
        df[col] = pd.to_numeric(df.get(col), errors="coerce").astype("float64")

    df["range_pct"] = float(range_pct)

    return (
        df[["timestamp"] + _VALUE_COLS + ["range_pct"]]
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _load_existing(existing_data: dict, partition_key: str) -> pd.DataFrame:
    """Resolve a lazy-callable or eager DataFrame from a PartitionedDataset dict."""
    obj = existing_data.get(partition_key)
    if obj is None:
        return pd.DataFrame()
    df = obj() if callable(obj) else obj.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _merge_incremental(df_existing: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """Concat existing and new, deduplicate on timestamp, sort ascending."""
    if df_existing.empty:
        return df_new
    if df_new.empty:
        return df_existing
    merged = pd.concat([df_existing, df_new], ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
    return (
        merged
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------


def update_orderbook_4h_incremental(
    existing_data: Dict[str, pd.DataFrame],
    params: dict,
    api_key: str,
) -> Dict[str, pd.DataFrame]:
    """Incrementally update CoinGlass order book depth partitions (4h, L1).

    For each asset × range combination:
      1. Load existing partition (if present) from the PartitionedDataset
      2. Fetch up to ``limit`` most-recent records from the API
      3. Merge: concat + deduplicate on timestamp + sort ascending
      4. Return the updated partition dict

    Parameters
    ----------
    existing_data:
        Dict of callables or DataFrames keyed by partition name, as provided
        by Kedro's PartitionedDataset loader.  Empty dict on first run.
    params:
        ``ingestion.coinglass.orderbook_4h`` block from parameters.yml.
    api_key:
        CoinGlass API key (``api_keys.coinglass`` from credentials).

    Returns
    -------
    Dict[str, pd.DataFrame]
        One entry per (asset × range) partition, ready for PartitionedDataset
        to write as individual Parquet files.
    """
    if existing_data is None:
        existing_data = {}

    exchange = params["exchange"]
    interval = params["interval"]
    limit = int(params["limit"])
    assets = params["assets"]
    ranges = params["ranges"]
    sleep_rng = float(params.get("sleep_between_ranges", 0.3))
    sleep_asset = float(params.get("sleep_between_assets", 0.5))

    updated: Dict[str, pd.DataFrame] = {}

    for asset_idx, asset in enumerate(assets):

        for range_idx, range_str in enumerate(ranges):
            partition_key = _range_to_partition_key(asset, range_str)

            df_existing = _load_existing(existing_data, partition_key)

            records = _fetch_orderbook(
                exchange=exchange,
                symbol=asset,
                interval=interval,
                limit=limit,
                range_pct=range_str,
                api_key=api_key,
            )

            if records is None:
                # API failure — preserve existing partition unchanged
                if not df_existing.empty:
                    updated[partition_key] = df_existing
                logger.warning(
                    "orderbook/%s range=%s: skipped due to API error; "
                    "existing partition preserved (%d rows)",
                    asset,
                    range_str,
                    len(df_existing),
                )
                if range_idx < len(ranges) - 1:
                    time.sleep(sleep_rng)
                continue

            df_new = _records_to_df(records, range_str)
            df_merged = _merge_incremental(df_existing, df_new)
            updated[partition_key] = df_merged

            if not df_merged.empty:
                logger.info(
                    "orderbook/%s range=%s: fetched %d rows, total %d (%s → %s)",
                    asset,
                    range_str,
                    len(df_new),
                    len(df_merged),
                    df_merged["timestamp"].min().date(),
                    df_merged["timestamp"].max().date(),
                )
            else:
                logger.warning("orderbook/%s range=%s: no data after merge", asset, range_str)

            if range_idx < len(ranges) - 1:
                time.sleep(sleep_rng)

        if asset_idx < len(assets) - 1:
            time.sleep(sleep_asset)

    return updated
