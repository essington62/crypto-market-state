"""
L3 primary feature engineering nodes for Yahoo Finance macro data.

This module contains nodes that compute primary statistical features from L2
intermediate data:
- Price-based variations and returns
- Rolling statistics (mean, std, z-scores, realized volatility)
- Relative state (deviation from mean, percentile rank)
- Volume-based normalization

No economic structural logic is applied here. Features are purely statistical.
Each asset (symbol) is processed independently.
"""

from math import sqrt
from typing import Callable, Dict

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _compute_rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling z-score: (value - rolling_mean) / rolling_std.
    """
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / rolling_std


def _compute_realized_volatility(rolling_std: pd.Series) -> pd.Series:
    """
    Compute annualized realized volatility using sqrt(252).
    """
    return rolling_std * sqrt(252.0)


def _compute_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling percentile rank (0–1) using a rolling window.
    """
    return series.rolling(window).apply(
        lambda x: (x <= x.iloc[-1]).mean() if len(x) == window else np.nan,
        raw=False,
    )


# -------------------------------------------------------------------
# Main node
# -------------------------------------------------------------------
def build_yfinance_primary_features(
    data: Dict[str, Callable[[], pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    """
    Build primary statistical features from Yahoo Finance L2 intermediate data.

    Features:
    - Returns: 1d, 5d, 21d
    - Log return (1d)
    - Rolling mean, std, z-score (21, 63, 252)
    - Annualized realized volatility (21, 63, 252)
    - Relative price state vs 252d mean
    - Rolling percentile rank (252)
    - Volume normalization (mean and z-score, 21)

    The function is pure:
    - No filesystem access
    - No Kedro datasets
    - No resampling or forward-fill
    - No cross-asset logic
    """
    primary_features: Dict[str, pd.DataFrame] = {}

    for symbol, loader in data.items():
        # Support both callable loaders and direct DataFrames
        df = loader() if callable(loader) else loader  # type: ignore[assignment]

        # Defensive copy
        df_features = df.copy()

        # Ensure clean time axis
        df_features = df_features.sort_values("date")
        df_features = df_features.drop_duplicates(subset=["date"], keep="last")

        close = df_features["close"]
        volume = df_features["volume"]

        # ===============================================================
        # 1. RETURNS
        # ===============================================================
        df_features["return_1d"] = close.pct_change(1, fill_method=None)
        df_features["return_5d"] = close.pct_change(5, fill_method=None)
        df_features["return_21d"] = close.pct_change(21, fill_method=None)

        ratio = close / close.shift(1)
        df_features["log_return_1d"] = np.where(
            ratio > 0,
            np.log(ratio),
            np.nan,
        )

        # ===============================================================
        # 2. ROLLING STATISTICS (close)
        # ===============================================================
        for window in (21, 63, 252):
            df_features[f"rolling_mean_{window}"] = close.rolling(window).mean()
            df_features[f"rolling_std_{window}"] = close.rolling(window).std()
            df_features[f"zscore_{window}"] = _compute_rolling_zscore(
                close, window=window
            )

        # ===============================================================
        # 3. REALIZED VOLATILITY
        # ===============================================================
        for window in (21, 63, 252):
            df_features[f"realized_vol_{window}"] = _compute_realized_volatility(
                df_features[f"rolling_std_{window}"]
            )

        # ===============================================================
        # 4. RELATIVE STATE
        # ===============================================================
        df_features["price_minus_mean_252"] = (
            close - df_features["rolling_mean_252"]
        )

        df_features["percentile_rank_252"] = _compute_percentile_rank(
            close, window=252
        )

        # ===============================================================
        # 5. VOLUME FEATURES
        # ===============================================================
        df_features["volume_mean_21"] = volume.rolling(21).mean()
        volume_std_21 = volume.rolling(21).std()
        df_features["volume_zscore_21"] = (
            volume - df_features["volume_mean_21"]
        ) / volume_std_21

        primary_features[symbol] = df_features

    return primary_features
