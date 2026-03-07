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

# Technical features approved in XGBoost R5 ablation (Fase 1B).
# Shorter burn-in (~30d). NaN permitted in early history.
_TECH_COLS = [
    # MOMENTUM
    "rsi_14_z",
    "rsi_30",
    "macd_hist_norm",
    "roc_5",
    "roc_10",
    "roc_21",
    # STRUCTURE
    "price_vs_high_30d",
    "price_vs_low_30d",
    "range_position_30d",
    "bb_position",
    "atr_14_norm",
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
    rsi_short_window: int = int(params["rsi_short_window"])
    rsi_long_window: int = int(params["rsi_long_window"])
    macd_fast: int = int(params["macd_fast"])
    macd_slow: int = int(params["macd_slow"])
    macd_signal_period: int = int(params["macd_signal_period"])
    atr_window: int = int(params["atr_window"])
    range_window: int = int(params["range_window"])

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

        # ── Technical features — R5 approved (Fase 1B) ────────────────────

        # RSI — short and long
        _delta = close.diff()
        _gain_s = _delta.clip(lower=0).rolling(rsi_short_window).mean()
        _loss_s = (-_delta.clip(upper=0)).rolling(rsi_short_window).mean()
        _rs_s = _gain_s / _loss_s.replace(0, np.nan)
        _rsi_14 = 100 - (100 / (1 + _rs_s))
        out["rsi_14_z"] = (_rsi_14 - 50) / 20

        _gain_l = _delta.clip(lower=0).rolling(rsi_long_window).mean()
        _loss_l = (-_delta.clip(upper=0)).rolling(rsi_long_window).mean()
        _rs_l = _gain_l / _loss_l.replace(0, np.nan)
        out["rsi_30"] = 100 - (100 / (1 + _rs_l))

        # MACD histogram normalised by close
        _ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
        _ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
        _macd_line = _ema_fast - _ema_slow
        _macd_sig = _macd_line.ewm(span=macd_signal_period, adjust=False).mean()
        out["macd_hist_norm"] = (_macd_line - _macd_sig) / close

        # Rate of change
        out["roc_5"]  = close.pct_change(5)
        out["roc_10"] = close.pct_change(10)
        out["roc_21"] = close.pct_change(21)

        # Range-based structure
        _high_range = close.rolling(range_window).max()
        _low_range  = close.rolling(range_window).min()
        _range_span = _high_range - _low_range
        out["price_vs_high_30d"]  = (close - _high_range) / _high_range
        out["price_vs_low_30d"]   = (close - _low_range) / _low_range
        out["range_position_30d"] = np.where(
            _range_span > 0, (close - _low_range) / _range_span, np.nan
        )

        # Bollinger band position (reuses bb_window param)
        _bb_mid  = close.rolling(bb_window).mean()
        _bb_std  = close.rolling(bb_window).std()
        _bb_upper = _bb_mid + 2 * _bb_std
        _bb_lower = _bb_mid - 2 * _bb_std
        _bb_range = _bb_upper - _bb_lower
        out["bb_position"] = np.where(
            _bb_range > 0, (close - _bb_lower) / _bb_range, np.nan
        )

        # ATR normalised (requires high and low from L2)
        _tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        out["atr_14_norm"] = _tr.rolling(atr_window).mean() / close

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

    all_cols = _FEATURE_COLS + _CHARTIST_COLS + _TECH_COLS + ["top_position_ratio"]
    available = [c for c in all_cols if c in df.columns]
    out = df[available].copy()
    out.index.name = "date"
    return out
