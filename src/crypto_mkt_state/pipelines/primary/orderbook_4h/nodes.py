"""
L3 Feature Engineering — CoinGlass Order Book Depth (4h, BTC).

Input
-----
PartitionedDataset with 3 partitions (DatetimeIndex UTC):
  BTCUSDT_r05  — 0.5% depth range
  BTCUSDT_r1   — 1.0% depth range
  BTCUSDT_r2   — 2.0% depth range

Columns per partition: bids_usd, bids_quantity, asks_usd, asks_quantity, range_pct

Features computed
-----------------
Per range (suffix _r05, _r1, _r2):
  book_imbalance_{r}   = (bids_usd − asks_usd) / (bids_usd + asks_usd), clip [−1,1]
  bid_ask_ratio_{r}    = bids_usd / asks_usd
  book_depth_{r}       = bids_usd + asks_usd  (total USD liquidity)

Cross-range gradient (requires all 3 ranges):
  depth_gradient_near  = book_imbalance_r05 − book_imbalance_r1
  depth_gradient_far   = book_imbalance_r1  − book_imbalance_r2
  total_depth          = book_depth_r1 + book_depth_r2

Rolling (window = rolling_window candles, strictly backward-looking):
  imbalance_ma_6       = book_imbalance_r1.rolling(window).mean()
  imbalance_std_6      = book_imbalance_r1.rolling(window).std()
  depth_zscore_24h     = (book_depth_r1 − mean) / std  (NaN if std == 0)

L2 columns preserved from r1 partition (renamed with _r1 suffix):
  bids_usd_r1, asks_usd_r1, bids_quantity_r1, asks_quantity_r1

L3 contract:
  - All windows strictly backward-looking (no lookahead)
  - NaN permitted at burn-in start — never dropna
  - L2 columns preserved unchanged
  - No cross-asset operations
  - No label creation
"""
from __future__ import annotations

import logging
from typing import Callable, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Required L2 columns present in every partition
_REQUIRED_L2 = ["bids_usd", "bids_quantity", "asks_usd", "asks_quantity", "range_pct"]

# Expected partition suffixes derived from partition keys: BTCUSDT_{suffix}
_EXPECTED_SUFFIXES = {"r05", "r1", "r2"}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _suffix_from_key(partition_key: str) -> str:
    """Extract range suffix from partition key.

    Examples
    --------
    "BTCUSDT_r05"  →  "r05"
    "BTCUSDT_r1"   →  "r1"
    "BTCUSDT_r2"   →  "r2"
    """
    return partition_key.split("_")[-1]


def _compute_range_features(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Compute per-range features for a single depth partition.

    Parameters
    ----------
    df:
        L2 partition with DatetimeIndex UTC and L2 columns.
    suffix:
        Range identifier (e.g. "r05", "r1", "r2").

    Returns
    -------
    pd.DataFrame
        Index: same DatetimeIndex as input.
        Columns: book_imbalance, bid_ask_ratio, book_depth (all suffixed),
                 plus L2 raw columns (also suffixed) for downstream filtering.
    """
    total = df["bids_usd"] + df["asks_usd"]
    denom_safe = total.where(total != 0.0, other=np.nan)
    asks_safe = df["asks_usd"].where(df["asks_usd"] != 0.0, other=np.nan)

    out = pd.DataFrame(index=df.index)
    out[f"book_imbalance_{suffix}"] = (
        (df["bids_usd"] - df["asks_usd"]) / denom_safe
    ).clip(lower=-1.0, upper=1.0)
    out[f"bid_ask_ratio_{suffix}"] = df["bids_usd"] / asks_safe
    out[f"book_depth_{suffix}"]    = total

    # L2 columns carried forward for final column set (only r1 survives the join)
    out[f"bids_usd_{suffix}"]      = df["bids_usd"]
    out[f"asks_usd_{suffix}"]      = df["asks_usd"]
    out[f"bids_quantity_{suffix}"] = df["bids_quantity"]
    out[f"asks_quantity_{suffix}"] = df["asks_quantity"]

    return out


def _validate_partitions(
    feature_frames: dict[str, pd.DataFrame],
) -> None:
    """Raise ValueError if any expected suffix is absent."""
    present = set(feature_frames.keys())
    missing = _EXPECTED_SUFFIXES - present
    if missing:
        raise ValueError(
            f"[L3 orderbook_4h] Missing partitions for suffixes: {sorted(missing)}. "
            f"Cross-range gradient features cannot be computed."
        )


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------


def compute_orderbook_features_4h(
    partitions: Dict[str, Callable[[], pd.DataFrame]],
    rolling_window: int,
) -> pd.DataFrame:
    """Compute L3 order book features from the 3 depth-range partitions.

    Parameters
    ----------
    partitions:
        Kedro PartitionedDataset mapping ``{partition_key: load_callable}``.
        Expected keys: BTCUSDT_r05, BTCUSDT_r1, BTCUSDT_r2.
    rolling_window:
        Lookback candles for rolling features (e.g. 6 → 24h at 4h cadence).

    Returns
    -------
    pd.DataFrame
        Single merged DataFrame with UTC DatetimeIndex, 19 columns.
        NaN at burn-in start is expected and permitted.

    Raises
    ------
    ValueError
        If any required partition or L2 column is missing, or DataFrame is empty.
    """
    if not partitions:
        raise ValueError("[L3 orderbook_4h] No partitions provided.")

    # ── 1. Load and validate each partition; compute per-range features ──────
    feature_frames: dict[str, pd.DataFrame] = {}

    for partition_key, load_func in partitions.items():
        df = load_func() if callable(load_func) else load_func

        missing_cols = [c for c in _REQUIRED_L2 if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"[L3 orderbook_4h] {partition_key}: missing L2 columns: {missing_cols}"
            )
        if df.empty:
            raise ValueError(f"[L3 orderbook_4h] {partition_key}: partition is empty.")

        suffix = _suffix_from_key(partition_key)
        feature_frames[suffix] = _compute_range_features(df, suffix)

    _validate_partitions(feature_frames)

    # ── 2. Outer join all 3 ranges on timestamp ──────────────────────────────
    merged = pd.concat(
        [feature_frames["r05"], feature_frames["r1"], feature_frames["r2"]],
        axis=1,
        join="outer",
    )

    # ── 3. Cross-range gradient features ────────────────────────────────────
    merged["depth_gradient_near"] = (
        merged["book_imbalance_r05"] - merged["book_imbalance_r1"]
    )
    merged["depth_gradient_far"] = (
        merged["book_imbalance_r1"] - merged["book_imbalance_r2"]
    )
    merged["total_depth"] = merged["book_depth_r1"] + merged["book_depth_r2"]

    # ── 4. Rolling features (strictly backward-looking) ─────────────────────
    imb_r1 = merged["book_imbalance_r1"]
    dep_r1 = merged["book_depth_r1"]

    merged[f"imbalance_ma_{rolling_window}"]  = imb_r1.rolling(rolling_window).mean()
    merged[f"imbalance_std_{rolling_window}"] = imb_r1.rolling(rolling_window).std()

    dep_mean = dep_r1.rolling(rolling_window).mean()
    dep_std  = dep_r1.rolling(rolling_window).std()
    merged["depth_zscore_24h"] = (dep_r1 - dep_mean) / dep_std.where(dep_std != 0.0, other=np.nan)

    # ── 5. Select final column set (drop r05/r2 L2 raw, keep r1 L2 raw) ─────
    final_cols = [
        # per-range imbalance
        "book_imbalance_r05", "book_imbalance_r1", "book_imbalance_r2",
        # per-range ratio
        "bid_ask_ratio_r05",  "bid_ask_ratio_r1",  "bid_ask_ratio_r2",
        # per-range depth
        "book_depth_r05",     "book_depth_r1",     "book_depth_r2",
        # cross-range gradient
        "depth_gradient_near", "depth_gradient_far", "total_depth",
        # rolling
        f"imbalance_ma_{rolling_window}",
        f"imbalance_std_{rolling_window}",
        "depth_zscore_24h",
        # L2 preserved from r1
        "bids_usd_r1", "asks_usd_r1", "bids_quantity_r1", "asks_quantity_r1",
    ]
    out = merged[final_cols].sort_index(ascending=True)

    logger.info(
        "[L3 orderbook_4h] Features computed — shape: %s | range: %s → %s",
        out.shape,
        out.index.min().date(),
        out.index.max().date(),
    )

    return out
