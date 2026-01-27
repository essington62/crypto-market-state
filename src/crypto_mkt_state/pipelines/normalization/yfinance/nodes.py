"""
L2 normalization nodes for Yahoo Finance macro data.

This module contains nodes that perform schema/metadata normalization only
for Yahoo Finance macro time series:
- Adds metadata columns (symbol, asset, category, source, interval, ingestion_ts)
- Ensures clean date index (sorted, duplicate dates removed)

No resampling, merging or value transformations are performed here.
Each asset (partition) is processed independently.
"""

from typing import Any, Callable, Dict

import pandas as pd


def normalize_yfinance_macro(
    data: Dict[str, Callable[[], pd.DataFrame]],
    assets_meta: list[dict],
) -> Dict[str, pd.DataFrame]:
    """
    Normalize Yahoo Finance macro series at L2 (intermediate) level.

    This node:
    - Processes each partition (ticker) independently
    - Adds columns:
        - symbol: ticker identifier (partition key)
        - asset: asset name from metadata (params:yfinance.assets)
        - category: category from metadata
        - source: constant string "yfinance"
        - interval: constant string "1d"
        - ingestion_ts: UTC timestamp of normalization
    - Sorts by date
    - Removes duplicate dates (keeping the last occurrence)

    The function is pure and does not access the filesystem or Kedro datasets.

    Args:
        data:
            Dictionary mapping ticker (partition keys) to callables that
            return DataFrames from L1 (`yfinance_macro_raw`). Each DataFrame must
            contain:
            - date (datetime64[ns, UTC])
            - open, high, low, close, volume (float)
        assets_meta:
            List of dicts from `params:yfinance.assets`. Each dict is expected to
            contain at least:
            - ticker: ticker identifier, matching the partition key
            - name: human-readable asset name
            - category: category label for the asset

    Returns:
        Dictionary mapping ticker to normalized DataFrames that are
        compatible with a PartitionedDataset:
        - Columns:
            - date
            - open
            - high
            - low
            - close
            - volume
            - symbol
            - asset
            - category
            - source
            - interval
            - ingestion_ts
        - Sorted by date ascending
        - Duplicate dates removed (keep last)
    """
    # Build metadata lookup keyed by ticker
    meta_by_ticker: Dict[str, dict] = {
        m["ticker"]: m for m in assets_meta if "ticker" in m
    }

    normalized: Dict[str, pd.DataFrame] = {}

    for ticker, loader in data.items():
        # Support both callables (PartitionedDataset standard) and direct DataFrames
        if callable(loader):
            df = loader()
        else:
            df = loader  # type: ignore[assignment]

        # Defensive copy to avoid mutating upstream objects
        df_norm = df.copy()

        # Keep only original yfinance OHLCV schema
        df_norm = df_norm[["date", "open", "high", "low", "close", "volume"]]

        # Attach metadata
        meta: dict[str, Any] = meta_by_ticker.get(ticker, {})
        category = meta.get("category")
        asset = meta.get("name")

        df_norm["symbol"] = ticker
        df_norm["asset"] = asset
        df_norm["category"] = category
        df_norm["source"] = "yfinance"
        df_norm["interval"] = "1d"
        df_norm["ingestion_ts"] = pd.Timestamp.utcnow()

        # Ensure clean time axis: sorted and without duplicate dates
        df_norm = df_norm.sort_values("date")
        df_norm = df_norm.drop_duplicates(subset=["date"], keep="last")

        normalized[ticker] = df_norm

    return normalized
