"""
Temporal governance helpers for L1 ingestion.

This module defines the L1 temporal contract that ALL ingestion nodes
must respect:

- Global start_date and interval come from `params:global.*`
- Raw data is filtered at L1 (no "fetch extra and cut later")
- Frequency is validated (assert) for the configured interval

IMPORTANT:
- This helper must ONLY be used in L1 (ingestion) nodes.
- L2+ layers must NOT call this function or apply temporal logic.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd


Interval = Literal["1d"]


def enforce_l1_temporal_contract(
    df: pd.DataFrame,
    start_date: str,
    interval: Interval,
) -> pd.DataFrame:
    """
    Enforce the global L1 temporal contract on a single DataFrame.

    This function MUST be called by all L1 loaders (Binance, FRED, YFinance):
    - Normalizes the `date` column to UTC
    - Applies the global start_date cut
    - Validates the expected interval (currently only '1d')

    Args:
        df:
            Input DataFrame. It is not mutated; a filtered copy is returned.
            Must contain a `date` column.
        start_date:
            Global start date (ISO string, e.g. '2020-10-01', UTC).
        interval:
            Global interval string. Currently only '1d' is supported.

    Returns:
        Filtered DataFrame with:
        - `date` as timezone-aware UTC
        - Only rows with date >= start_date

    Raises:
        AssertionError:
            If the data does not respect the configured frequency.
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    # Normalize and ensure UTC timezone
    out["date"] = pd.to_datetime(out["date"], utc=True)

    # Apply global start_date cut
    start_ts = pd.to_datetime(start_date, utc=True)
    out = out[out["date"] >= start_ts]

    if out.empty:
        return out

    # Validate frequency
    if interval == "1d":
        deltas = out["date"].diff().dropna()
        if not deltas.empty:
            mode_delta = deltas.mode().iloc[0]
            assert mode_delta == pd.Timedelta(days=1), (
                f"L1 contract violated: non-daily data detected "
                f"(mode delta = {mode_delta})"
            )

    return out

