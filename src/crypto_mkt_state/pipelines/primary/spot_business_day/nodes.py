"""
L3 Primary features for spot business day. Baseline v0 for HMM.

Operates only on L2 spot_business_day_clean. Adds deterministic statistical features.
Does not alter L2 values, calendar, or mix assets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Dict, Any


def _build_features_one_partition(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Build L3 features for a single partition. Keeps all L2 columns."""
    out = df.copy()
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()

    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"]

    ret_short_window = int(params["ret_short_window"])
    ret_long_window = int(params["ret_long_window"])
    vol_short_window = int(params["vol_short_window"])
    vol_long_window = int(params["vol_long_window"])
    drawdown_window = int(params["drawdown_window"])
    ma_window = int(params["ma_window"])
    volume_window = int(params["volume_window"])
    use_log_returns = params["use_log_returns"]
    use_vol_ratio = params["use_vol_ratio"]
    use_drawdown = params["use_drawdown"]
    use_volume_zscore = params["use_volume_zscore"]
    use_range_relative = params["use_range_relative"]

    # 1) log_return
    log_return = np.log(close / close.shift(1))
    out["log_return"] = log_return

    # 2) ret_short, 3) ret_long
    if use_log_returns:
        out["ret_short"] = log_return.rolling(ret_short_window).sum()
        out["ret_long"] = log_return.rolling(ret_long_window).sum()
    else:
        out["ret_short"] = close / close.shift(ret_short_window) - 1.0
        out["ret_long"] = close / close.shift(ret_long_window) - 1.0

    # 4) vol_short, 5) vol_long
    out["vol_short"] = log_return.rolling(vol_short_window).std()
    out["vol_long"] = log_return.rolling(vol_long_window).std()

    # 6) vol_ratio
    if use_vol_ratio:
        vol_long = out["vol_long"]
        out["vol_ratio"] = np.where(
            vol_long > 0,
            out["vol_short"] / vol_long,
            np.nan,
        )

    # 7) drawdown
    if use_drawdown:
        rolling_max = close.rolling(drawdown_window, min_periods=1).max()
        out["drawdown"] = close / rolling_max - 1.0

    # 8) dist_to_ma
    ma = close.rolling(ma_window).mean()
    out["dist_to_ma"] = np.where(ma > 0, (close - ma) / ma, np.nan)

    # 9) range_rel
    if use_range_relative:
        out["range_rel"] = np.where(close > 0, (high - low) / close, np.nan)

    # 10) volume_z
    if use_volume_zscore:
        vol_mean = volume.rolling(volume_window).mean()
        vol_std = volume.rolling(volume_window).std()
        out["volume_z"] = np.where(vol_std > 0, (volume - vol_mean) / vol_std, np.nan)

    feature_cols = [c for c in out.columns if c not in df.columns]
    if feature_cols:
        out = out.dropna(how="any", subset=feature_cols)
    return out


def build_spot_business_day_features_partitions(
    partitions: Dict[str, Callable[[], pd.DataFrame]],
    params: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Build L3 primary features per partition from L2 spot_business_day_clean.

    Input: partition_id -> callable that returns DataFrame (index=timestamp, OHLCV).
    Params: l3.crypto.spot_daily (ret_short_window, vol_short_window, toggles, etc.).
    Output: partition_id -> DataFrame with all L2 columns plus feature columns.
    """
    if not partitions:
        raise ValueError("L3 spot_business_day: no partitions provided.")

    required_keys = {
        "ret_short_window", "ret_long_window",
        "vol_short_window", "vol_long_window",
        "drawdown_window", "ma_window", "volume_window",
        "use_log_returns", "use_vol_ratio", "use_drawdown",
        "use_volume_zscore", "use_range_relative",
    }
    missing = required_keys - set(params)
    if missing:
        raise ValueError(f"L3 spot_business_day: missing params: {sorted(missing)}.")

    result: Dict[str, pd.DataFrame] = {}
    for partition_id, load_func in partitions.items():
        df = load_func()
        if df is None or df.empty:
            raise ValueError(f"L3 spot_business_day: partition {partition_id} is empty.")
        for col in ("close", "high", "low", "volume"):
            if col not in df.columns:
                raise ValueError(
                    f"L3 spot_business_day: partition {partition_id} missing column '{col}'."
                )
        result[partition_id] = _build_features_one_partition(df, params)
    return result
