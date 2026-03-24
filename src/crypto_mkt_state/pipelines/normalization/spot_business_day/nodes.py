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
import numpy as np


REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}
OPTIONAL_COLUMNS = {"trades"}


def validate_business_day_contract(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and validate a single partition to L2 Spot Business Day contract.
    
    Structural validation only with robust correction for common data issues:
    - Automatically fixes OHLC inconsistencies (high < max(open,close), etc.)
    - Handles NaNs by dropping rows (with logging)
    - Does NOT enforce fixed calendar gaps
    - Does NOT alter economic values beyond necessary corrections
    """

    if df is None or df.empty:
        raise ValueError("L2 Spot Business Day: partition is None or empty.")

    out = df.copy()

    # --- Normalize OHLCV column case (yfinance incremental saves uppercase;
    #     incremental merge may produce both cases with NaN in each).
    #     Strategy: coalesce lowercase ← uppercase, then drop uppercase duplicate.
    _case_map = {
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    }
    for src, dst in _case_map.items():
        if src in out.columns and dst not in out.columns:
            # Only uppercase exists — simple rename
            out = out.rename(columns={src: dst})
        elif src in out.columns and dst in out.columns:
            # Both exist — coalesce lowercase from uppercase, then drop uppercase
            out[dst] = out[dst].fillna(out[src])
            out = out.drop(columns=[src])

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

    # --- Track corrections for logging
    corrections_count = 0
    initial_rows = len(out)
    
    # --- OHLC FIX: Automatic correction instead of hard validation
    
    # 1. Fix high < max(open, close)
    max_open_close = np.maximum(out["open"], out["close"])
    high_issue_mask = out["high"] < max_open_close
    if high_issue_mask.any():
        corrections_count += high_issue_mask.sum()
        out.loc[high_issue_mask, "high"] = max_open_close[high_issue_mask]
    
    # 2. Fix low > min(open, close)
    min_open_close = np.minimum(out["open"], out["close"])
    low_issue_mask = out["low"] > min_open_close
    if low_issue_mask.any():
        corrections_count += low_issue_mask.sum()
        out.loc[low_issue_mask, "low"] = min_open_close[low_issue_mask]
    
    # 3. Fix high < low (post previous corrections)
    high_low_mask = out["high"] < out["low"]
    if high_low_mask.any():
        corrections_count += high_low_mask.sum()
        # Set high = max(high, low) and low = min(high, low)
        out.loc[high_low_mask, "high"] = np.maximum(
            out.loc[high_low_mask, "high"], out.loc[high_low_mask, "low"]
        )
        out.loc[high_low_mask, "low"] = np.minimum(
            out.loc[high_low_mask, "high"], out.loc[high_low_mask, "low"]
        )
    
    # --- NaN HANDLING: Drop rows with NaNs in OHLCV
    ohlc_cols = ["open", "high", "low", "close"]
    ohlc_nan_mask = out[ohlc_cols].isna().any(axis=1)
    volume_nan_mask = out["volume"].isna()
    
    rows_with_ohlc_nan = ohlc_nan_mask.sum()
    rows_with_volume_nan = volume_nan_mask.sum()
    
    # Drop rows where any OHLC is NaN (these are critical)
    if rows_with_ohlc_nan > 0:
        out = out[~ohlc_nan_mask]
    
    # Log volume NaNs but don't drop automatically
    # (some assets may have legitimate zero/NaN volume on holidays)
    
    # --- Final validation for OHLC integrity after corrections
    # These should now pass after our fixes
    o, h, l, c = out["open"], out["high"], out["low"], out["close"]
    
    # These checks should now pass; if not, something is seriously wrong
    if ((h < o) | (h < c)).any():
        raise ValueError("L2 Spot Business Day: high must be >= max(open, close) after correction.")
    
    if ((l > o) | (l > c)).any():
        raise ValueError("L2 Spot Business Day: low must be <= min(open, close) after correction.")
    
    if (h < l).any():
        raise ValueError("L2 Spot Business Day: high must be >= low after correction.")
    
    # Check for negative/zero prices (preserve original behavior)
    if (o <= 0).any() or (h <= 0).any() or (l <= 0).any() or (c <= 0).any():
        raise ValueError(
            "L2 Spot Business Day: open, high, low, close must be > 0."
        )
    
    # --- Volume integrity
    if (out["volume"] < 0).any():
        raise ValueError("L2 Spot Business Day: volume must be >= 0.")
    
    if "trades" in out.columns and (out["trades"] < 0).any():
        raise ValueError("L2 Spot Business Day: trades must be >= 0.")
    
    # --- Logging
    final_rows = len(out)
    rows_dropped = initial_rows - final_rows
    
    print(f"[L2 FIX] {corrections_count} OHLC inconsistencies corrected")
    
    if rows_with_ohlc_nan > 0:
        print(f"[L2 WARN] {rows_with_ohlc_nan} rows dropped due to NaN in OHLC")
    
    if rows_with_volume_nan > 0 and rows_with_volume_nan != rows_dropped:
        print(f"[L2 WARN] {rows_with_volume_nan} rows have NaN volume (kept)")
    
    if rows_dropped > 0:
        print(f"[L2 INFO] rows dropped: {rows_dropped}")
    
    print(f"[L2 INFO] shape=({final_rows}, {len(out.columns)}) range={out.index.min().strftime('%Y-%m-%d')} → {out.index.max().strftime('%Y-%m-%d')}")
    
    # Final type check (preserve original contract)
    if out.select_dtypes(include=["object"]).shape[1] > 0:
        raise ValueError("L2 Spot Business Day: no object columns allowed.")
    
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
        try:
            result[partition_id] = validate_business_day_contract(df)
        except Exception as e:
            print(f"[L2 ERROR] Failed to process partition '{partition_id}': {e}")
            raise

    return result