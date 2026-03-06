"""
L3 primary features for spot daily (24/7 crypto).

Consumes L2 `spot_daily_clean` (PartitionedDataset)
and produces L3 `spot_daily_features` and `btc_spot_daily_model_input`.

Pure deterministic feature engineering.
No model logic. UTC enforced.
All window parameters read from params:l3.crypto.spot_daily.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
import pandas as pd

from crypto_mkt_state.utils.utils_l3_semantic import (
    compute_bb_width_20d,
    compute_dist_to_ma_200d,
    compute_er,
    compute_high_52w_dist,
    compute_hurst_rolling,
    compute_kurt_rolling,
    compute_ma_50_200_ratio,
    compute_skew_rolling,
    compute_slope_21d,
)

# Core model-input features (shorter burn-in, ~30d). Used for dropna gate
# and for btc_spot_daily_model_input extraction.
_FEATURE_COLS = [
    "log_return",
    "ret_short",
    "ret_long",
    "vol_short",
    "vol_long",
    "vol_ratio",
    "drawdown",
    "dist_to_ma",
    "range_rel",
    "volume_z",
    "skew_30d",
    "kurt_30d",
    "hurst_30d",
    "er_10d",
]

# Chartist / structural features (200d burn-in). Included in the full L3
# dataset and in the model input, but NaN rows are allowed in early history.
_CHARTIST_COLS = [
    "dist_to_ma_200d",
    "ma_50_200_ratio",
    "high_52w_dist",
    "slope_21d",
    "bb_width_20d",
]


def create_btc_l3_features(
    partitions: Dict[str, Callable[[], pd.DataFrame]],
    params: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Generate deterministic rolling statistical features from L2 OHLCV data.

    Expected L2 columns: open, high, low, close, volume
    Params read from params:l3.crypto.spot_daily.

    Generated features (added to all L2 columns):
        log_return, ret_short, ret_long
        vol_short, vol_long, vol_ratio
        drawdown, dist_to_ma, range_rel, volume_z
        skew_30d, kurt_30d, hurst_30d, er_10d
        dist_to_ma_200d, ma_50_200_ratio, high_52w_dist, slope_21d, bb_width_20d
    """
    if not partitions:
        raise ValueError("primary.spot_crypto: received empty partitions.")

    ret_short_window: int = int(params["ret_short_window"])
    ret_long_window: int = int(params["ret_long_window"])
    vol_short_window: int = int(params["vol_short_window"])
    vol_long_window: int = int(params["vol_long_window"])
    drawdown_window: int = int(params["drawdown_window"])
    ma_window: int = int(params["ma_window"])
    volume_window: int = int(params["volume_window"])
    skew_window: int = int(params["skew_window"])
    kurt_window: int = int(params["kurt_window"])
    hurst_window: int = int(params["hurst_window"])
    er_window: int = int(params["er_window"])
    ma_200_window: int = int(params["ma_200_window"])
    ma_50_window: int = int(params["ma_50_window"])
    ma_52w_window: int = int(params["ma_52w_window"])
    slope_window: int = int(params["slope_window"])
    bb_window: int = int(params["bb_window"])

    results: Dict[str, pd.DataFrame] = {}

    for asset, load_fn in partitions.items():

        df = load_fn()

        if df is None or df.empty:
            raise ValueError(f"{asset}: empty DataFrame.")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{asset}: index must be DatetimeIndex.")

        if df.index.tz is None or str(df.index.tz) != "UTC":
            raise ValueError(f"{asset}: index must be UTC.")

        df = df.sort_index()

        required_cols = {"close", "high", "low", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"{asset}: missing columns {sorted(missing)}. "
                f"Available: {sorted(df.columns.tolist())}"
            )

        out = df.copy()

        close = out["close"].astype("float64")
        high = out["high"].astype("float64")
        low = out["low"].astype("float64")
        volume = out["volume"].astype("float64")

        # ── Feature Engineering ───────────────────────────────────────────

        # Log return — must be first; other features depend on it
        log_return = np.log(close / close.shift(1))
        out["log_return"] = log_return

        # Rolling returns
        out["ret_short"] = close / close.shift(ret_short_window) - 1.0
        out["ret_long"] = close / close.shift(ret_long_window) - 1.0

        # Rolling volatility
        out["vol_short"] = log_return.rolling(vol_short_window).std()
        out["vol_long"] = log_return.rolling(vol_long_window).std()

        # Volatility ratio
        out["vol_ratio"] = np.where(
            out["vol_long"] > 0,
            out["vol_short"] / out["vol_long"],
            np.nan,
        )

        # Drawdown (rolling window)
        rolling_max = close.rolling(drawdown_window, min_periods=1).max()
        out["drawdown"] = close / rolling_max - 1.0

        # Distance to moving average
        ma = close.rolling(ma_window).mean()
        out["dist_to_ma"] = np.where(ma > 0, close / ma - 1.0, np.nan)

        # Relative range
        out["range_rel"] = np.where(close > 0, (high - low) / close, np.nan)

        # Volume z-score
        vol_mean = volume.rolling(volume_window).mean()
        vol_std = volume.rolling(volume_window).std()
        out["volume_z"] = np.where(vol_std > 0, (volume - vol_mean) / vol_std, np.nan)

        # Rolling skewness
        out["skew_30d"] = compute_skew_rolling(log_return, skew_window)

        # Rolling kurtosis
        out["kurt_30d"] = compute_kurt_rolling(log_return, kurt_window)

        # Rolling Hurst exponent via R/S analysis
        out["hurst_30d"] = compute_hurst_rolling(log_return, hurst_window)

        # Efficiency Ratio
        out["er_10d"] = compute_er(close, er_window)

        # ── Chartist / structural features ────────────────────────────────

        # Distance to long-term MA (200d)
        out["dist_to_ma_200d"] = compute_dist_to_ma_200d(close, ma_200_window)

        # Golden/Death cross: MA50 / MA200 ratio
        out["ma_50_200_ratio"] = compute_ma_50_200_ratio(close, ma_50_window, ma_200_window)

        # Distance from 52-week high
        out["high_52w_dist"] = compute_high_52w_dist(close, ma_52w_window)

        # Normalized linear regression slope (21d)
        out["slope_21d"] = compute_slope_21d(close, slope_window)

        # Bollinger Band width (20d)
        out["bb_width_20d"] = compute_bb_width_20d(close, bb_window)

        # Drop burn-in rows where any feature is NaN
        out = out.dropna(subset=_FEATURE_COLS)

        results[asset] = out

    return results


def extract_btc_model_input(
    partitions: Dict[str, Callable[[], pd.DataFrame]],
    top_positions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract BTCUSDT from spot_daily_features and return feature-only DataFrame.

    Also joins top_position_ratio from L1 Coinglass long/short top positions.
    Produces btc_spot_daily_model_input (no OHLCV columns).
    Raises ValueError if BTCUSDT partition is missing.
    """
    btc_key = "BTCUSDT"
    if btc_key not in partitions:
        raise ValueError(
            f"extract_btc_model_input: partition '{btc_key}' not found. "
            f"Available: {sorted(partitions.keys())}"
        )

    df = partitions[btc_key]()

    # Join top_position_ratio from L1 Coinglass top positions
    ratio_col = [c for c in top_positions.columns if "ratio" in c.lower()][0]
    tp = top_positions[[ratio_col]].rename(columns={ratio_col: "top_position_ratio"})
    df = df.join(tp, how="left")

    all_cols = _FEATURE_COLS + _CHARTIST_COLS + ["top_position_ratio"]
    available = [c for c in all_cols if c in df.columns]
    out = df[available].copy()
    out.index.name = "date"
    return out
