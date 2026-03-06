"""
L3 Primary features for spot — multiple calendars.

Three pipelines in this module:
  create_pipeline():        legacy per-asset features (gold, nasdaq, sp500)
  create_bday_pipeline():   BTC BDay + cross-crypto (deprecated, kept for reference)
  create_daily_pipeline():  BTC 24/7 daily + cross-crypto (Fase 2 canonical)

Fase 2 canonical reads spot_daily_clean (24/7 L2), normalises to midnight UTC,
and produces a consolidated model-ready DataFrame.
Index: DatetimeIndex freq="D" UTC — includes weekends (crypto 24/7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Callable, Dict

from crypto_mkt_state.utils.utils_l3_semantic import (
    _slope_normalized,
    compute_bb_width_20d,
    compute_dist_to_ma_200d,
    compute_high_52w_dist,
    compute_ma_50_200_ratio,
    compute_slope_21d,
)


# ── Shared internal helpers ────────────────────────────────────────────────────

def _prepare_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a 24/7 partition to midnight UTC DatetimeIndex.

    Steps:
      1. Localise to UTC if tz-naive
      2. Sort index
      3. Normalise timestamps to midnight (removes time component)
      4. Deduplicate — keep last record per day
    """
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    df = df.copy().sort_index()
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ── Legacy: per-asset features (gold, nasdaq, sp500) ──────────────────────────

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

    log_return = np.log(close / close.shift(1))
    out["log_return"] = log_return

    if use_log_returns:
        out["ret_short"] = log_return.rolling(ret_short_window).sum()
        out["ret_long"] = log_return.rolling(ret_long_window).sum()
    else:
        out["ret_short"] = close / close.shift(ret_short_window) - 1.0
        out["ret_long"] = close / close.shift(ret_long_window) - 1.0

    out["vol_short"] = log_return.rolling(vol_short_window).std()
    out["vol_long"] = log_return.rolling(vol_long_window).std()

    if use_vol_ratio:
        vol_long = out["vol_long"]
        out["vol_ratio"] = np.where(
            vol_long > 0,
            out["vol_short"] / vol_long,
            np.nan,
        )

    if use_drawdown:
        rolling_max = close.rolling(drawdown_window, min_periods=1).max()
        out["drawdown"] = close / rolling_max - 1.0

    ma = close.rolling(ma_window).mean()
    out["dist_to_ma"] = np.where(ma > 0, (close - ma) / ma, np.nan)

    if use_range_relative:
        out["range_rel"] = np.where(close > 0, (high - low) / close, np.nan)

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


# ── Fase 2 canonical: BTC 24/7 daily + cross-crypto ───────────────────────────

# Core columns required to be non-NaN for model input validity.
# Includes 4 cross-crypto features — they burn in within the first ~30d.
_CORE_COLS = [
    "log_return", "vol_short", "vol_ratio",
    "drawdown", "volume_z", "slope_21d",
    "eth_btc_rs_5d", "alt_dispersion_5d",
    "mean_alt_beta_30d", "mean_alt_dd_rel",
]


def compute_btc_daily_features(
    partitions: Dict[str, Callable[[], pd.DataFrame]],
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Calculate BTC features on 24/7 daily series.

    No resample, no calendar conversion.
    Index: DatetimeIndex freq="D" UTC, normalised to midnight.
    Weekends are preserved (crypto trades 24/7).

    Core features (short burn-in, ~7-30d):
      log_return, vol_short, vol_long, vol_ratio, drawdown, volume_z

    Chartist features (longer burn-in, up to 252d):
      slope_21d, dist_to_ma_200d, ma_50_200_ratio, high_52w_dist, bb_width_20d
    """
    btc_key = "BTCUSDT"
    if btc_key not in partitions:
        raise ValueError(
            f"compute_btc_daily_features: partition '{btc_key}' not found. "
            f"Available: {sorted(partitions.keys())}"
        )

    df = _prepare_daily(partitions[btc_key]())

    if df.empty:
        raise ValueError("compute_btc_daily_features: BTCUSDT partition is empty.")

    for col in ("close", "high", "low", "volume"):
        if col not in df.columns:
            raise ValueError(f"compute_btc_daily_features: missing column '{col}'.")

    vol_short_w: int = int(params["vol_short_window"])
    vol_long_w: int = int(params["vol_long_window"])
    drawdown_w: int = int(params["drawdown_window"])
    volume_w: int = int(params["volume_window"])
    slope_w: int = int(params["slope_window"])
    bb_w: int = int(params["bb_window"])
    ma_50_w: int = int(params["ma_50_window"])
    ma_200_w: int = int(params["ma_200_window"])
    ma_52w_w: int = int(params["ma_52w_window"])

    close = df["close"].astype("float64")
    volume = df["volume"].astype("float64")

    # ── Core features ─────────────────────────────────────────────────────
    log_ret = np.log(close / close.shift(1))
    df["log_return"] = log_ret

    df["vol_short"] = log_ret.rolling(vol_short_w).std()
    df["vol_long"] = log_ret.rolling(vol_long_w).std()
    df["vol_ratio"] = np.where(
        df["vol_long"] > 0,
        df["vol_short"] / df["vol_long"],
        np.nan,
    )

    rolling_max = close.rolling(drawdown_w, min_periods=1).max()
    df["drawdown"] = (close - rolling_max) / rolling_max

    vol_mean = volume.rolling(volume_w).mean()
    vol_std = volume.rolling(volume_w).std()
    df["volume_z"] = np.where(vol_std > 0, (volume - vol_mean) / vol_std, np.nan)

    # ── Chartist features ──────────────────────────────────────────────────
    df["slope_21d"] = close.rolling(slope_w).apply(_slope_normalized, raw=True)
    df["dist_to_ma_200d"] = compute_dist_to_ma_200d(close, ma_200_w)
    df["ma_50_200_ratio"] = compute_ma_50_200_ratio(close, ma_50_w, ma_200_w)
    df["high_52w_dist"] = compute_high_52w_dist(close, ma_52w_w)
    df["bb_width_20d"] = compute_bb_width_20d(close, bb_w)

    return df


def compute_cross_crypto_features(
    partitions: Dict[str, Callable[[], pd.DataFrame]],
    btc_features: pd.DataFrame,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Calculate 4 cross-crypto aggregate features on 24/7 daily series.

    All assets are crypto with identical 24/7 calendars.
    Inner join on BTC index — no ffill, no resample.
    If an alt lacks data on a BTC date, it is excluded from that day's mean
    (line is not dropped).

    Features:
      eth_btc_rs_5d      — ETH 5d rolling return minus BTC 5d rolling return
      alt_dispersion_5d  — cross-sectional std of alt 5d log returns
      mean_alt_beta_30d  — mean 30d rolling OLS beta of alts vs BTC
      mean_alt_dd_rel    — mean alt 90d drawdown minus BTC 90d drawdown
    """
    rs_w: int = int(params.get("rs_window", 5))
    beta_w: int = int(params.get("beta_window", 30))
    dd_w: int = int(params.get("dd_rel_window", 90))

    btc_idx = btc_features.index
    ret_btc = btc_features["log_return"]
    dd_btc = btc_features["drawdown"]

    # Load and prepare all non-BTC partitions
    alt_close: Dict[str, pd.Series] = {}
    for name, load_fn in partitions.items():
        if name == "BTCUSDT":
            continue
        df = _prepare_daily(load_fn())
        common = btc_idx.intersection(df.index)
        if len(common) == 0:
            continue
        alt_close[name] = df.loc[common, "close"].astype("float64")

    # ── 1. eth_btc_rs_5d ──────────────────────────────────────────────────
    if "ETHUSDT" in alt_close:
        eth_c = alt_close["ETHUSDT"]
        ret_eth = np.log(eth_c / eth_c.shift(1))
        ret_btc_eth = ret_btc.loc[eth_c.index]
        eth_btc_rs_5d = (
            ret_eth.rolling(rs_w).sum()
            - ret_btc_eth.rolling(rs_w).sum()
        )
    else:
        eth_btc_rs_5d = pd.Series(np.nan, index=btc_idx, name="eth_btc_rs_5d")

    # ── 2. alt_dispersion_5d ──────────────────────────────────────────────
    ret_5d: Dict[str, pd.Series] = {
        name: np.log(c / c.shift(rs_w))
        for name, c in alt_close.items()
    }
    if ret_5d:
        alt_dispersion_5d = pd.DataFrame(ret_5d).std(axis=1, skipna=True)
    else:
        alt_dispersion_5d = pd.Series(np.nan, index=btc_idx, name="alt_dispersion_5d")

    # ── 3. mean_alt_beta_30d ──────────────────────────────────────────────
    beta_series: Dict[str, pd.Series] = {}
    for name, c in alt_close.items():
        r_alt = np.log(c / c.shift(1))
        r_btc_aligned = ret_btc.loc[c.index]
        cov = r_alt.rolling(beta_w).cov(r_btc_aligned)
        var = r_btc_aligned.rolling(beta_w).var()
        beta_series[name] = pd.Series(
            np.where(var > 0, cov / var, np.nan),
            index=c.index,
        )
    if beta_series:
        mean_alt_beta_30d = pd.DataFrame(beta_series).mean(axis=1, skipna=True)
    else:
        mean_alt_beta_30d = pd.Series(np.nan, index=btc_idx, name="mean_alt_beta_30d")

    # ── 4. mean_alt_dd_rel ────────────────────────────────────────────────
    dd_series: Dict[str, pd.Series] = {}
    for name, c in alt_close.items():
        roll_max = c.rolling(dd_w, min_periods=1).max()
        dd_series[name] = (c - roll_max) / roll_max
    if dd_series:
        dd_df = pd.DataFrame(dd_series)
        btc_dd_aligned = dd_btc.reindex(dd_df.index)
        mean_alt_dd_rel = dd_df.mean(axis=1, skipna=True) - btc_dd_aligned
    else:
        mean_alt_dd_rel = pd.Series(np.nan, index=btc_idx, name="mean_alt_dd_rel")

    return pd.DataFrame(
        {
            "eth_btc_rs_5d": eth_btc_rs_5d,
            "alt_dispersion_5d": alt_dispersion_5d,
            "mean_alt_beta_30d": mean_alt_beta_30d,
            "mean_alt_dd_rel": mean_alt_dd_rel,
        },
    ).reindex(btc_idx)


def consolidate_btc_model_input_daily(
    btc_features: pd.DataFrame,
    cross_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner join BTC features + cross-crypto features on index.

    Both inputs are 24/7 crypto on the same DatetimeIndex — inner join
    produces no data loss. Applies dropna only on core columns (10 total):
      log_return, vol_short, vol_ratio, drawdown, volume_z, slope_21d,
      eth_btc_rs_5d, alt_dispersion_5d, mean_alt_beta_30d, mean_alt_dd_rel

    Chartist features (dist_to_ma_200d, etc.) may retain NaN in early rows.
    """
    combined = btc_features.join(cross_features, how="inner")
    core_cols = [c for c in _CORE_COLS if c in combined.columns]
    return combined.dropna(subset=core_cols)
