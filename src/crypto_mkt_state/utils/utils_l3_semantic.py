"""
L3 semantic feature resolution helpers.

This module resolves which features are allowed for each asset based on
explicit rules defined in parameters.yml under `l3_semantic.assets` and
`l3_semantic.indices`.

Also provides pure feature computation helpers for rolling statistics
used in L3 primary feature engineering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List


# ── Semantic resolution ───────────────────────────────────────────────────────

def normalize_asset_name(asset_name: str) -> str:
    """
    Normalize asset name for lookup in semantic config.

    Rules:
    - Convert to lowercase
    - Remove 'usdt' suffix if present (e.g. 'BTCUSDT' -> 'btc')
    - Strip whitespace

    Args:
        asset_name:
            Raw asset name (e.g. 'BTCUSDT', 'SP500', 'VIX').

    Returns:
        Normalized asset name (e.g. 'btc', 'sp500', 'vix').
    """
    normalized = asset_name.lower().strip()

    # Remove 'usdt' suffix for crypto pairs
    if normalized.endswith("usdt"):
        normalized = normalized[:-4]

    return normalized


def get_semantic_features(
    asset_name: str,
    semantic_config: Dict,
) -> List[str]:
    """
    Get the list of allowed features for an asset from semantic config.

    Search order:
    1. l3_semantic.assets[normalized_name]
    2. l3_semantic.indices[normalized_name]

    Args:
        asset_name:
            Asset name (e.g. 'BTCUSDT', 'sp500', 'VIX', 'cpi').
            Will be normalized before lookup.
        semantic_config:
            The `l3_semantic` section from parameters.yml.

    Returns:
        List of feature names allowed for this asset.

    Raises:
        ValueError:
            If the asset is not found in either assets or indices sections.
    """
    normalized = normalize_asset_name(asset_name)

    # Search in assets first
    assets_config = semantic_config.get("assets", {})
    if normalized in assets_config:
        asset_config = assets_config[normalized]
        features = asset_config.get("features", [])
        if isinstance(features, list):
            return features
        return []

    # Search in indices
    indices_config = semantic_config.get("indices", {})
    if normalized in indices_config:
        index_config = indices_config[normalized]
        features = index_config.get("features", [])
        if isinstance(features, list):
            return features
        return []

    # Asset not found
    raise ValueError(
        f"Asset '{asset_name}' (normalized: '{normalized}') not found in "
        "l3_semantic.assets or l3_semantic.indices. "
        "Please add it to parameters.yml under l3_semantic."
    )


# ── Rolling feature computation helpers ───────────────────────────────────────

def compute_skew_rolling(log_return: pd.Series, window: int) -> pd.Series:
    """
    Rolling skewness of log_return.

    skew_Nd = log_return.rolling(window).skew()

    NaN for the first (window - 1) rows (burn-in).
    """
    return log_return.rolling(window).skew()


def compute_kurt_rolling(log_return: pd.Series, window: int) -> pd.Series:
    """
    Rolling excess kurtosis of log_return.

    kurt_Nd = log_return.rolling(window).kurt()

    NaN for the first (window - 1) rows (burn-in).
    """
    return log_return.rolling(window).kurt()


def _hurst_rs(x: np.ndarray) -> float:
    """
    Hurst exponent via R/S analysis on a 1-D array.

    Returns 0.5 if std == 0 or n < 10 (safe fallback: random walk).
    """
    n = len(x)
    if n < 10:
        return 0.5
    mean = x.mean()
    devs = x - mean
    cumdev = np.cumsum(devs)
    R = cumdev.max() - cumdev.min()
    S = x.std()
    if S == 0 or R == 0:
        return 0.5
    return float(np.log(R / S) / np.log(n))


def compute_hurst_rolling(log_return: pd.Series, window: int) -> pd.Series:
    """
    Rolling Hurst exponent via R/S analysis.

    > 0.5 = persistent trend (structural bull/bear)
    < 0.5 = mean-reverting (temporary correction)
    = 0.5 = random walk

    NaN for the first (window - 1) rows (burn-in).
    """
    return log_return.rolling(window).apply(_hurst_rs, raw=True)


def compute_er(close: pd.Series, window: int) -> pd.Series:
    """
    Efficiency Ratio over a rolling window.

    er = |close - close.shift(window)| / sum(|diff(close)|, window)

    > 0.7 = strong directional move
    < 0.3 = chaotic / sideways / transitional
    Returns 0.0 when path == 0.

    NaN for the first (window) rows (burn-in).
    """
    direction = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window).sum()
    er = direction / path
    er = er.where(path > 0, other=0.0)
    return er


# ── Chartist / structural feature helpers ─────────────────────────────────────

def compute_dist_to_ma_200d(close: pd.Series, window: int) -> pd.Series:
    """
    Distance of close to its long-term moving average.

    dist_to_ma_200d = (close - MA_window) / MA_window

    > 0 = above MA (structural bull)
    < 0 = below MA (structural bear)
    NaN for the first (window - 1) rows (burn-in).
    """
    ma = close.rolling(window).mean()
    return (close - ma) / ma


def compute_ma_50_200_ratio(close: pd.Series,
                             fast_window: int,
                             slow_window: int) -> pd.Series:
    """
    Ratio of fast MA to slow MA minus 1.

    ma_50_200_ratio = MA_fast / MA_slow - 1.0

    > 0 = golden cross (fast above slow, bullish)
    < 0 = death cross (fast below slow, bearish)
    NaN for the first (slow_window - 1) rows (burn-in).
    """
    ma_fast = close.rolling(fast_window).mean()
    ma_slow = close.rolling(slow_window).mean()
    return ma_fast / ma_slow - 1.0


def compute_high_52w_dist(close: pd.Series, window: int) -> pd.Series:
    """
    Distance of close from its rolling 52-week (252-day) high.

    high_52w_dist = (close - rolling_max) / rolling_max

    Always <= 0. Closer to 0 = near cycle top.
    NaN for the first (window - 1) rows (burn-in).
    """
    rolling_high = close.rolling(window).max()
    return (close - rolling_high) / rolling_high


def _slope_normalized(x: np.ndarray) -> float:
    """
    Linear regression slope of the window, normalised by last close.

    slope_normalized = linregress(range(n), x).slope / x[-1]

    Returns 0.0 if last value is zero or window is too short.
    """
    n = len(x)
    if n < 2 or x[-1] == 0:
        return 0.0
    t = np.arange(n, dtype=float)
    # Closed-form OLS slope
    t_mean = t.mean()
    x_mean = x.mean()
    slope = ((t - t_mean) * (x - x_mean)).sum() / ((t - t_mean) ** 2).sum()
    return float(slope / x[-1])


def compute_slope_21d(close: pd.Series, window: int) -> pd.Series:
    """
    Rolling normalised linear regression slope of close over `window` days.

    slope_21d = linregress(range(window), close_window).slope / close_window[-1]

    > 0 = accelerating uptrend
    < 0 = downtrend / deceleration
    NaN for the first (window - 1) rows (burn-in).
    """
    return close.rolling(window).apply(_slope_normalized, raw=True)


def compute_bb_width_20d(close: pd.Series, window: int) -> pd.Series:
    """
    Bollinger Band width as a fraction of the moving average.

    bb_width = (4 * std_window) / ma_window

    Low  = volatility compression (potential breakout incoming)
    High = volatility expansion (trending / stress)
    NaN for the first (window - 1) rows (burn-in).
    """
    ma  = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (4.0 * std) / ma
