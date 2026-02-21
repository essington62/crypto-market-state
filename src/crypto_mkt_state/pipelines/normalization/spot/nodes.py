"""
L2 Spot Daily: validate and standardize schema. API-agnostic, asset-type oriented.

Responsibilities: standardize schema, structural integrity, 24/7 frequency.
Does not alter values, no harmonization, no feature engineering, no fill/resample/interpolation.
"""

from __future__ import annotations

from typing import Callable, Dict

import pandas as pd


# Required columns after rename (L1 may have open_time -> timestamp).
REQUIRED_COLUMNS = {
    "timestamp",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trades",
}
OPTIONAL_VOLUME_COLUMNS = {"quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume"}
ONE_DAY = pd.Timedelta(days=1)


def validate_and_standardize_spot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and standardize a single partition to L2 Spot Daily contract.

    - Renames open_time -> timestamp.
    - Enforces dtypes (timestamp/close_time datetime64[ns, UTC]; OHLCV float64; trades int64).
    - Sets index to timestamp (DatetimeIndex UTC), sorted ascending, no duplicates.
    - Validates OHLC integrity, volume, and 24/7 daily frequency (exactly 1 day between rows).
    - Does not change any values; raises ValueError if contract is violated.

    Parameters
    ----------
    df : pd.DataFrame
        Raw L1 partition (must contain open_time, close_time, open, high, low, close, volume, trades).

    Returns
    -------
    pd.DataFrame
        Standardized L2 dataframe with index=timestamp, same values.
    """
    if df is None or df.empty:
        raise ValueError("L2 Spot: partition is None or empty.")

    # ----- Rename -----
    if "open_time" not in df.columns:
        raise ValueError("L2 Spot: missing required column 'open_time'.")
    out = df.rename(columns={"open_time": "timestamp"}).copy()

    # ----- Required columns -----
    missing = REQUIRED_COLUMNS - set(out.columns)
    if missing:
        raise ValueError(f"L2 Spot: missing required columns: {sorted(missing)}.")

    # ----- Types (no value change: ensure dtypes only) -----
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["close_time"] = pd.to_datetime(out["close_time"], utc=True)

    for col in ("open", "high", "low", "close", "volume"):
        out[col] = out[col].astype("float64")
    out["trades"] = out["trades"].astype("int64")

    for col in OPTIONAL_VOLUME_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("float64")

    # ----- Index: timestamp, sorted, no duplicates -----
    out = out.set_index("timestamp")
    if out.index.duplicated().any():
        raise ValueError("L2 Spot: duplicate timestamps in partition.")
    out = out.sort_index(ascending=True)

    # ----- No NaN in required (structural integrity) -----
    required = list(REQUIRED_COLUMNS - {"timestamp"})  # timestamp is index
    if out[required].isna().any().any():
        raise ValueError("L2 Spot: NaN in required columns.")

    # ----- Temporal: close_time >= timestamp per candle -----
    if (out["close_time"] < out.index).any():
        raise ValueError("L2 Spot: close_time must be >= timestamp for each row.")

    # ----- OHLC integrity -----
    o, h, l, c = out["open"], out["high"], out["low"], out["close"]
    if ((h < o) | (h < c)).any():
        raise ValueError("L2 Spot: high must be >= max(open, close).")
    if ((l > o) | (l > c)).any():
        raise ValueError("L2 Spot: low must be <= min(open, close).")
    if (h < l).any():
        raise ValueError("L2 Spot: high must be >= low.")
    if (o <= 0).any() or (h <= 0).any() or (l <= 0).any() or (c <= 0).any():
        raise ValueError("L2 Spot: open, high, low, close must be > 0.")

    # ----- Volume integrity -----
    if (out["volume"] < 0).any():
        raise ValueError("L2 Spot: volume must be >= 0.")
    if "taker_buy_base_volume" in out.columns:
        if (out["taker_buy_base_volume"] < 0).any():
            raise ValueError("L2 Spot: taker_buy_base_volume must be >= 0.")
        if (out["taker_buy_base_volume"] > out["volume"]).any():
            raise ValueError("L2 Spot: taker_buy_base_volume must be <= volume.")
    if (out["trades"] < 0).any():
        raise ValueError("L2 Spot: trades must be >= 0.")
    if "quote_volume" in out.columns and (out["quote_volume"] < 0).any():
        raise ValueError("L2 Spot: quote_volume must be >= 0.")
    if "taker_buy_quote_volume" in out.columns:
        if (out["taker_buy_quote_volume"] < 0).any():
            raise ValueError("L2 Spot: taker_buy_quote_volume must be >= 0.")
        if "quote_volume" in out.columns and (
            out["taker_buy_quote_volume"] > out["quote_volume"]
        ).any():
            raise ValueError("L2 Spot: taker_buy_quote_volume must be <= quote_volume.")

    # ----- 24/7 continuous daily frequency -----
    if len(out) > 1:
        diffs = out.index.to_series().diff().dropna()
        if (diffs != ONE_DAY).any():
            raise ValueError(
                "L2 Spot: timestamps must be exactly 1 day apart (24/7 daily). "
                "Gaps or non-daily intervals are not allowed."
            )

    return out


def normalize_spot_daily_partitions(
    partitions: Dict[str, Callable[[], pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:

    if not partitions:
        raise ValueError("L2 Spot: no partitions provided.")

    result: Dict[str, pd.DataFrame] = {}

    for partition_id, load_func in partitions.items():
        print(f"Processing partition: {partition_id}")
        df = load_func()
        print("Columns:", df.columns.tolist())
        result[partition_id] = validate_and_standardize_spot(df)

    return result
