"""
L2 Normalization — CoinGlass Order Book Depth (4h).

Responsibilities (L2 contract):
  - Set timestamp as DatetimeIndex (UTC), sort ascending
  - Enforce float64 on all numeric columns via pd.to_numeric(..., errors="coerce")
  - Validate schema: raise ValueError on missing required columns
  - Validate timestamps: raise ValueError on duplicates
  - Validate values: raise ValueError on negative bids_usd / asks_usd
  - Log per partition: key, shape, date range, NaN counts

Forbidden (L2 contract):
  rolling windows, pct_change, z-score, feature engineering,
  dropna, fillna, resampling, cross-asset operations, derived columns.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "timestamp",
    "bids_usd",
    "bids_quantity",
    "asks_usd",
    "asks_quantity",
    "range_pct",
]
NUMERIC_COLUMNS = [
    "bids_usd",
    "bids_quantity",
    "asks_usd",
    "asks_quantity",
    "range_pct",
]


def _normalize_partition(df: pd.DataFrame, partition_key: str) -> pd.DataFrame:
    """Normalize a single L1 order book partition to the L2 contract.

    Parameters
    ----------
    df:
        Raw L1 partition loaded from PartitionedDataset.
    partition_key:
        Partition identifier used in log and error messages.

    Returns
    -------
    pd.DataFrame
        L2-compliant DataFrame with UTC DatetimeIndex named ``timestamp``.

    Raises
    ------
    ValueError
        If any required column is missing, duplicate timestamps are found,
        or bids_usd / asks_usd contain negative values.
    """
    # ── 1. Schema validation ────────────────────────────────────────────────
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[L2 orderbook] {partition_key}: missing required columns: {missing}"
        )

    out = df.copy()

    # ── 2. Timestamp enforcement ────────────────────────────────────────────
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)

    # ── 3. Duplicate timestamp check ────────────────────────────────────────
    dupes = out["timestamp"].duplicated()
    if dupes.any():
        raise ValueError(
            f"[L2 orderbook] {partition_key}: {dupes.sum()} duplicate timestamp(s) found."
        )

    # ── 4. Set DatetimeIndex, sort ascending ────────────────────────────────
    out = out.set_index("timestamp").sort_index(ascending=True)

    # ── 5. Float64 enforcement via pd.to_numeric ────────────────────────────
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    # ── 6. Negative-value validation ────────────────────────────────────────
    for col in ("bids_usd", "asks_usd"):
        neg_mask = out[col] < 0
        if neg_mask.any():
            raise ValueError(
                f"[L2 orderbook] {partition_key}: {neg_mask.sum()} negative value(s) "
                f"in column '{col}'."
            )

    # ── 7. Logging ───────────────────────────────────────────────────────────
    nan_counts = out[NUMERIC_COLUMNS].isna().sum()
    nan_summary = {c: int(n) for c, n in nan_counts.items() if n > 0}
    logger.info(
        "[L2 orderbook] %s — shape: %s | range: %s → %s | NaN: %s",
        partition_key,
        out.shape,
        out.index.min().date(),
        out.index.max().date(),
        nan_summary if nan_summary else "none",
    )

    return out


def normalize_orderbook_4h(
    partitions: Dict[str, Callable[[], pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    """Normalize all L1 order book partitions to L2 contract.

    Parameters
    ----------
    partitions:
        Kedro PartitionedDataset mapping ``{partition_key: load_callable}``.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Normalized partitions ready for PartitionedDataset to write.

    Raises
    ------
    ValueError
        If no partitions are provided, or if any partition fails validation.
    """
    if not partitions:
        raise ValueError("[L2 orderbook] No partitions provided.")

    result: Dict[str, pd.DataFrame] = {}

    for partition_key, load_func in partitions.items():
        df = load_func() if callable(load_func) else load_func
        result[partition_key] = _normalize_partition(df, partition_key)

    return result
