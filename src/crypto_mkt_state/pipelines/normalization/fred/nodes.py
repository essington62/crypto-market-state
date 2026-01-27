"""
L2 normalization nodes for FRED macro data.

This module contains nodes that perform schema/metadata normalization only
for FRED macro time series:
- Adds metadata columns (series_id, category, source)
- Ensures clean date index (sorted, duplicate dates removed)

No resampling, merging or value transformations are performed here.
Each series (partition) is processed independently.
"""

from typing import Any, Callable, Dict, List

import pandas as pd


def normalize_fred_macro(
    data: Dict[str, Callable[[], pd.DataFrame]],
    series_meta: list[dict],
) -> Dict[str, pd.DataFrame]:
    """
    Normalize FRED macro series at L2 (intermediate) level.

    This node:
    - Processes each partition (series) independently
    - Adds columns:
        - series_id: identifier of the series (partition key)
        - category: category from series metadata (params:fred.series)
        - source: constant string "fred"
    - Sorts by date
    - Removes duplicate dates (keeping the last occurrence)

    The function is pure and does not access the filesystem or Kedro datasets.

    Args:
        data:
            Dictionary mapping series_id (partition keys) to callables that
            return DataFrames from L1 (`fred_macro_raw`). Each DataFrame must
            contain:
            - date (datetime64[ns, UTC])
            - value (float)
        series_meta:
            List of dicts from `params:fred.series`. Each dict is expected to
            contain at least:
            - id: series identifier, matching the partition key
            - name: human-readable series name
            - category: category label for the series

    Returns:
        Dictionary mapping series_id to normalized DataFrames that are
        compatible with a PartitionedDataset:
        - Columns:
            - date
            - value
            - series_id
            - asset
            - category
            - source
            - interval
            - ingestion_ts
        - Sorted by date ascending
        - Duplicate dates removed (keep last)
    """
    # Build metadata lookup keyed by series id
    meta_by_id: Dict[str, dict] = {m["id"]: m for m in series_meta if "id" in m}

    normalized: Dict[str, pd.DataFrame] = {}

    for series_id, loader in data.items():
        # Support both callables (PartitionedDataset standard) and direct DataFrames
        if callable(loader):
            df = loader()
        else:
            df = loader  # type: ignore[assignment]

        # Defensive copy to avoid mutating upstream objects
        df_norm = df.copy()

        # Keep only original FRED schema (no synthetic financial fields)
        df_norm = df_norm[["date", "value"]]

        # Attach metadata
        meta: dict[str, Any] = meta_by_id.get(series_id, {})
        category = meta.get("category")
        asset = meta.get("name")

        df_norm["series_id"] = series_id
        df_norm["asset"] = asset
        df_norm["category"] = category
        df_norm["source"] = "fred"
        df_norm["interval"] = "1d"
        df_norm["ingestion_ts"] = pd.Timestamp.utcnow()

        # Ensure clean time axis: sorted and without duplicate dates
        df_norm = df_norm.sort_values("date")
        df_norm = df_norm.drop_duplicates(subset=["date"], keep="last")

        normalized[series_id] = df_norm

    return normalized

