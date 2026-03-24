"""
L3 Feature Engineering — CoinGlass daily indices.

Features computed per index (strictly backward-looking):
  {name}_zscore_{zscore_window}d  = (value - rolling_mean) / rolling_std
  {name}_change_{change_window}d  = value.pct_change(change_window)
  {name}_ma_ratio                 = value / value.rolling(ma_window).mean()

Forbidden: lookahead, cross-asset aggregations, labels, dropna.
NaN at start of each series (window burn-in) is expected — do not fill.
L2 raw columns are NOT preserved in output (daily index → forward-filled in L5).
"""
from __future__ import annotations

from typing import Callable, Dict, List

import pandas as pd


def compute_coinglass_indices_features(
    clean: Dict[str, Callable[[], pd.DataFrame]],
    coinglass_indices: List[dict],
    zscore_window: int,
    change_window: int,
    ma_window: int,
) -> pd.DataFrame:
    """
    Compute per-index features from L2 CoinGlass daily index partitions.

    For each index in ``coinglass_indices``:
      1. Loads the corresponding L2 partition.
      2. Selects the primary ``value_col``.
      3. Computes zscore, pct_change, and MA-ratio features.

    All Series are outer-merged into a single daily DataFrame indexed
    by UTC date.  Different index histories start at different dates;
    NaN at the start of any series is expected burn-in — do not dropna.

    Parameters
    ----------
    clean : dict
        L2 PartitionedDataset input {partition_name: loader_fn}.
    coinglass_indices : list of dict
        From parameters.yml — each entry: {name: str, value_col: str}.
    zscore_window : int
        Rolling window for z-score computation.
    change_window : int
        Lookback period for pct_change.
    ma_window : int
        Rolling window for MA-ratio.

    Returns
    -------
    pd.DataFrame
        Daily DatetimeIndex UTC.  Feature columns:
        {name}_zscore_{zscore_window}d, {name}_change_{change_window}d,
        {name}_ma_ratio — one triple per configured index.
        NaN permitted at start (burn-in).

    Raises
    ------
    ValueError
        If no features could be computed (all indices missing or empty).
    """
    feature_series: Dict[str, pd.Series] = {}

    for idx_cfg in coinglass_indices:
        name = idx_cfg["name"]
        value_col = idx_cfg["value_col"]

        if name not in clean:
            print(f"[L3 coinglass_indices] {name}: partition not found in L2 — skipping")
            continue

        df = clean[name]()

        if df is None or df.empty:
            print(f"[L3 coinglass_indices] {name}: empty partition — skipping")
            continue

        if value_col not in df.columns:
            print(
                f"[L3 coinglass_indices] {name}: value_col '{value_col}' "
                f"not found (columns: {df.columns.tolist()}) — skipping"
            )
            continue

        s = df[value_col].copy()

        # zscore_{zscore_window}d  (backward-looking)
        roll_mean = s.rolling(zscore_window, min_periods=1).mean()
        roll_std  = s.rolling(zscore_window, min_periods=2).std()
        feature_series[f"{name}_zscore_{zscore_window}d"] = (s - roll_mean) / roll_std

        # change_{change_window}d  (backward-looking pct_change)
        feature_series[f"{name}_change_{change_window}d"] = s.pct_change(change_window)

        # ma_ratio  (value / rolling mean — normalized momentum)
        ma = s.rolling(ma_window, min_periods=1).mean()
        feature_series[f"{name}_ma_ratio"] = s / ma

        print(
            f"[L3 coinglass_indices] {name}: "
            f"{len(s)} rows, 3 features, "
            f"{s.index.min().date()} → {s.index.max().date()}"
        )

    if not feature_series:
        raise ValueError(
            "[L3 coinglass_indices] No features computed — all indices were skipped. "
            "Check L2 partitions and parameters.yml coinglass_indices.indices."
        )

    # Outer merge: preserves each index's full date range
    result = pd.concat(feature_series, axis=1).sort_index(ascending=True)

    print(
        f"[L3 coinglass_indices] Consolidated: "
        f"{result.shape[1]} feature columns across {len(coinglass_indices)} indices, "
        f"{len(result)} daily rows, "
        f"{result.index.min().date()} → {result.index.max().date()}"
    )
    return result
