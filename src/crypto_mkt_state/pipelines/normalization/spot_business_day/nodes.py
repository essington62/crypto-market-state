"""
L2 Spot Business Day: validate and standardize schema for native business-day assets.

For assets already in L1 as business day (e.g. gold, nasdaq, sp500).

Contract:
- Structural validation only.
- No artificial calendar enforcement.
- No gap validation (market holidays allowed).
- No 24/7 conversion.
- No aggregation or interpolation.
"""

from __future__ import annotations

from typing import Callable, Dict
import pandas as pd


REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}
OPTIONAL_COLUMNS = {"trades"}


def validate_business_day_contract(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and validate a single partition to L2 Spot Business Day contract.

    Structural validation only.
    Does NOT enforce fixed calendar gaps.
    Does NOT alter economic values.
    """

    if df is None or df.empty:
        raise ValueError("L2 Spot Business Day: partition is None or empty.")

    out = df.copy()

    # --- Rename temporal column to canonical 'timestamp'
    if "timestamp" not in out.columns:
        if "date" in out.columns:
            out = out.rename(columns={"date": "timestamp"})
        else:
            raise ValueError(
                "L2 Spot Business Day: missing temporal column ('date' or 'timestamp')."
            )

    # --- Required columns check
    missing = REQUIRED_COLUMNS - set(out.columns)
    if missing:
        raise ValueError(
            f"L2 Spot Business Day: missing required columns: {sorted(missing)}."
        )

    # --- Enforce dtypes
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)

    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")

    if "trades" in out.columns:
        out["trades"] = pd.to_numeric(out["trades"], errors="coerce")

    # --- Set index
    out = out.set_index("timestamp")
    out.index.name = "timestamp"

    if out.index.duplicated().any():
        raise ValueError("L2 Spot Business Day: duplicate timestamps in index.")

    out = out.sort_index()

    if not out.index.is_monotonic_increasing:
        raise ValueError("L2 Spot Business Day: index must be monotonic increasing.")

    if out.index.tz is None or str(out.index.tz) != "UTC":
        raise ValueError("L2 Spot Business Day: timestamp must be UTC.")

    # --- Structural integrity
    if out[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError("L2 Spot Business Day: NaN in required columns.")

    if out.select_dtypes(include=["object"]).shape[1] > 0:
        raise ValueError("L2 Spot Business Day: no object columns allowed.")

    # --- OHLC integrity
    o, h, l, c = out["open"], out["high"], out["low"], out["close"]

    if ((h < o) | (h < c)).any():
        raise ValueError("L2 Spot Business Day: high must be >= max(open, close).")

    if ((l > o) | (l > c)).any():
        raise ValueError("L2 Spot Business Day: low must be <= min(open, close).")

    if (h < l).any():
        raise ValueError("L2 Spot Business Day: high must be >= low.")

    if (o <= 0).any() or (h <= 0).any() or (l <= 0).any() or (c <= 0).any():
        raise ValueError(
            "L2 Spot Business Day: open, high, low, close must be > 0."
        )

    # --- Volume integrity
    if (out["volume"] < 0).any():
        raise ValueError("L2 Spot Business Day: volume must be >= 0.")

    if "trades" in out.columns and (out["trades"] < 0).any():
        raise ValueError("L2 Spot Business Day: trades must be >= 0.")

    return out


def normalize_spot_business_day_partitions(
    partitions: Dict[str, Callable[[], pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    """
    Normalize each L1 business-day partition to L2 canonical contract.
    """

    if not partitions:
        raise ValueError("L2 Spot Business Day: no partitions provided.")

    result: Dict[str, pd.DataFrame] = {}

    for partition_id, load_func in partitions.items():
        df = load_func()
        result[partition_id] = validate_business_day_contract(df)

    return result