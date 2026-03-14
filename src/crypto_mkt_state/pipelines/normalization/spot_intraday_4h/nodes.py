"""
L2 Spot Intraday 4h: schema standardization only.

Responsibilities:
  - Rename open_time → timestamp
  - Enforce dtypes (datetime64[ns, UTC], float64, int64)
  - Set DatetimeIndex, sort ascending, drop duplicates
  - Warn (never raise) on 4h frequency gaps
  - Retain only the last max_candles rows (≈ 6 months at 4h)
  - Raise ValueError on schema violations

Forbidden: rolling windows, feature engineering, fill, resample,
           forward fill, interpolation, cross-asset operations.
"""
from __future__ import annotations

from typing import Callable, Dict

import pandas as pd


REQUIRED_COLUMNS = {
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trades",
}
OPTIONAL_FLOAT_COLUMNS = {
    "quote_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
}
FOUR_HOURS = pd.Timedelta(hours=4)


def validate_4h_frequency(df: pd.DataFrame, symbol: str) -> None:
    """
    Log a warning if any gap in the DatetimeIndex is not exactly 4 hours.
    Never raises — warning only, as per L2 contract.
    """
    if len(df) < 2:
        return
    diffs   = df.index.to_series().diff().dropna()
    bad     = diffs[diffs != FOUR_HOURS]
    if bad.empty:
        return
    print(
        f"[L2 4h] WARNING: {len(bad)} missing 4h candle(s) detected in {symbol} "
        f"(first gap at {bad.index[0].isoformat()})"
    )


def normalize_binance_spot_4h(
    df: pd.DataFrame,
    max_candles: int = 1080,
    symbol: str = "UNKNOWN",
) -> pd.DataFrame:
    """
    Normalize a single L1 4h partition to L2 contract.

    Parameters
    ----------
    df : pd.DataFrame
        Raw L1 partition (must contain open_time).
    max_candles : int
        Maximum number of candles to retain (tail). Default 1080 ≈ 6 months.
    symbol : str
        Symbol name used for log messages.

    Returns
    -------
    pd.DataFrame
        Standardized L2 DataFrame with DatetimeIndex UTC.
    """
    if df is None or df.empty:
        raise ValueError(f"[L2 4h] {symbol}: partition is None or empty.")

    n_raw = len(df)
    print(f"[L2 4h] Normalizing {symbol} — {n_raw} raw candles")

    # ── 1. Rename open_time → timestamp ────────────────────────────────────
    if "open_time" not in df.columns:
        raise ValueError(f"[L2 4h] {symbol}: missing required column 'open_time'.")
    out = df.rename(columns={"open_time": "timestamp"}).copy()

    # ── 2. Required columns check ───────────────────────────────────────────
    missing = REQUIRED_COLUMNS - set(out.columns)
    if missing:
        raise ValueError(
            f"[L2 4h] {symbol}: missing required columns: {sorted(missing)}."
        )

    # ── 3. Dtype enforcement ────────────────────────────────────────────────
    out["timestamp"]  = pd.to_datetime(out["timestamp"],  utc=True)
    out["close_time"] = pd.to_datetime(out["close_time"], utc=True)

    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    out["trades"] = pd.to_numeric(out["trades"], errors="coerce").astype("int64")

    for col in OPTIONAL_FLOAT_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    # ── 4. DatetimeIndex — sort ascending, drop duplicates ──────────────────
    out = out.set_index("timestamp")
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index(ascending=True)

    # ── 5. Frequency validation (warning only) ──────────────────────────────
    validate_4h_frequency(out, symbol)

    # ── 6. Anchored window — align to Coinglass start, then cap at max_candles
    WINDOW_START = pd.Timestamp("2025-09-13 04:00:00", tz="UTC")
    out = out[out.index >= WINDOW_START]
    if len(out) > max_candles:
        out = out.tail(max_candles)

    # ── 7. NaN check — warn, do not fill ────────────────────────────────────
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    nan_counts = out[ohlcv_cols].isna().sum()
    if nan_counts.any():
        print(
            f"[L2 4h] WARNING: NaN in OHLCV for {symbol}: "
            + ", ".join(f"{c}={nan_counts[c]}" for c in ohlcv_cols if nan_counts[c] > 0)
        )

    # ── 8. Final schema validation ───────────────────────────────────────────
    missing_final = REQUIRED_COLUMNS - set(out.columns)
    if missing_final:
        raise ValueError(
            f"[L2 4h] {symbol}: final schema missing columns: {sorted(missing_final)}."
        )

    date_start = out.index[0].isoformat()
    date_end   = out.index[-1].isoformat()
    print(f"[L2 4h] {symbol} normalized — shape: {out.shape}")
    print(f"[L2 4h] Window anchored to 2025-09-13 04:00 UTC  ({date_start} → {date_end})")

    return out


def normalize_spot_4h_partitions(
    partitions: Dict[str, Callable[[], pd.DataFrame]],
    max_candles: int,
) -> Dict[str, pd.DataFrame]:
    """
    Apply normalize_binance_spot_4h to every partition in the PartitionedDataset.

    Parameters
    ----------
    partitions : dict
        Kedro PartitionedDataset mapping {partition_id: load_callable}.
    max_candles : int
        Passed through to normalize_binance_spot_4h.

    Returns
    -------
    dict
        {partition_id: normalized_DataFrame}
    """
    if not partitions:
        raise ValueError("[L2 4h] No partitions provided.")

    result: Dict[str, pd.DataFrame] = {}

    for partition_id, load_func in partitions.items():
        df     = load_func()
        result[partition_id] = normalize_binance_spot_4h(
            df,
            max_candles=max_candles,
            symbol=partition_id,
        )

    return result
