"""
L2 Normalization — CoinGlass daily indices.

Responsibilities:
  - Ensure DatetimeIndex UTC
  - Enforce float64 on all value columns
  - Validate no duplicate timestamps
  - Sort ascending

Forbidden: rolling windows, feature engineering, fill, resample,
           forward fill, interpolation, cross-asset operations.
"""
from __future__ import annotations

from typing import Callable, Dict

import pandas as pd


def normalize_coinglass_indices(
    raw: Dict[str, Callable[[], pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    """
    L2 normalization for CoinGlass daily index partitions.

    Applies schema validation to each partition:
      - DatetimeIndex UTC (index.name = 'date')
      - float64 enforcement on all columns
      - Deduplication on timestamp (keep last)
      - Sort ascending

    Parameters
    ----------
    raw : dict
        PartitionedDataset input {partition_name: loader_fn}.

    Returns
    -------
    dict
        {partition_name: cleaned_df} for all non-empty partitions.

    Raises
    ------
    ValueError
        If a partition index is not DatetimeIndex.
    """
    cleaned: Dict[str, pd.DataFrame] = {}

    for name, loader in raw.items():
        df = loader()

        if df is None or df.empty:
            print(f"[L2 coinglass_indices] {name}: empty — skipping")
            continue

        # ── 1. Validate DatetimeIndex ────────────────────────────────────────
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"[L2 coinglass_indices] {name}: index is not DatetimeIndex "
                f"(got {type(df.index).__name__}). L1 contract violated."
            )

        # ── 2. Enforce UTC ───────────────────────────────────────────────────
        df = df.copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        elif str(df.index.tz) != "UTC":
            df.index = df.index.tz_convert("UTC")

        df.index.name = "date"

        # ── 3. Enforce float64 on all columns ────────────────────────────────
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        # ── 4. Deduplicate on timestamp (keep last) ──────────────────────────
        n_dup = df.index.duplicated().sum()
        if n_dup > 0:
            print(f"[L2 coinglass_indices] {name}: {n_dup} duplicate timestamps — keeping last")
            df = df[~df.index.duplicated(keep="last")]

        # ── 5. Sort ascending ────────────────────────────────────────────────
        df = df.sort_index(ascending=True)

        print(
            f"[L2 coinglass_indices] {name}: "
            f"{len(df)} rows, {df.shape[1]} columns, "
            f"{df.index.min().date()} → {df.index.max().date()}"
        )
        cleaned[name] = df

    return cleaned
