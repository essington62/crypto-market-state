"""
L2 Macro Daily: validate and standardize schema for macro daily partitions.

Structural normalization only. No feature engineering, interpolation, fill, or calendar validation.
Macro can be business day, weekly, monthly, or irregular; L2 does not validate delta between timestamps.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import pandas as pd


def _temporal_column_name(df: pd.DataFrame) -> str:
    if "timestamp" in df.columns:
        return "timestamp"
    if "date" in df.columns:
        return "date"
    raise ValueError(
        "L2 Macro Daily: missing temporal column ('date' or 'timestamp')."
    )


def validate_macro_daily_contract(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and validate a single partition to L2 Macro Daily contract.

    Renames date/timestamp to canonical 'timestamp', enforces DatetimeIndex UTC,
    sorts, enforces monotonicity and no duplicates. All non-timestamp columns
    converted to numeric via pd.to_numeric(errors='coerce'). Does not alter
    economic values. Does not validate calendar or gaps.
    """
    if df is None or df.empty:
        raise ValueError("L2 Macro Daily: partition is None or empty.")

    out = df.copy()
    temporal = _temporal_column_name(out)
    if temporal != "timestamp":
        out = out.rename(columns={temporal: "timestamp"})

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    value_columns: List[str] = [c for c in out.columns if c != "timestamp"]
    if not value_columns:
        raise ValueError("L2 Macro Daily: no value columns (only timestamp).")

    for col in value_columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    out = out.dropna(how="all", subset=value_columns)
    if out.empty:
        raise ValueError("L2 Macro Daily: no rows after dropping fully empty rows.")

    out = out.set_index("timestamp")
    out.index.name = "timestamp"

    if out.index.duplicated().any():
        raise ValueError("L2 Macro Daily: duplicate timestamps in index.")
    out = out.sort_index(ascending=True)
    if not out.index.is_monotonic_increasing:
        raise ValueError("L2 Macro Daily: index must be monotonic increasing.")
    if out.index.tz is None or str(out.index.tz) != "UTC":
        raise ValueError("L2 Macro Daily: index must be UTC.")

    if out.select_dtypes(include=["object"]).shape[1] > 0:
        raise ValueError("L2 Macro Daily: no object columns allowed.")

    return out


def normalize_macro_daily_partitions(
    partitions: Dict[str, Callable[[], pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    """Normalize each L1 macro daily partition to L2 contract. Returns partition_id -> DataFrame."""
    if not partitions:
        raise ValueError("L2 Macro Daily: no partitions provided.")
    result: Dict[str, pd.DataFrame] = {}
    for partition_id, load_func in partitions.items():
        df = load_func()
        result[partition_id] = validate_macro_daily_contract(df)
    return result
