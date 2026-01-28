"""
L3 primary feature engineering nodes for FRED macro data.

This module contains nodes that compute primary statistical features from L2
intermediate data:
- Variations (deltas and percentage changes)
- Rolling statistics (mean, std, z-scores)
- Relative state (deviation from mean, percentile rank)

No economic structural logic is applied here. Features are purely statistical.
"""

from typing import Callable, Dict

import pandas as pd


def _compute_rolling_zscore(
    series: pd.Series,
    window: int,
) -> pd.Series:
    """
    Compute rolling z-score: (value - rolling_mean) / rolling_std.

    Helper function for standardized rolling statistics.

    Args:
        series: Input time series
        window: Rolling window size

    Returns:
        Series with rolling z-scores (NaN where std is 0 or insufficient data)
    """
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / rolling_std


def build_fred_primary_features(
    data: Dict[str, Callable[[], pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    """
    Build primary statistical features from FRED L2 intermediate data.

    This node computes statistical features for each FRED series independently:
    - Variations: deltas and percentage changes (1d, 21d)
    - Rolling statistics: mean, std, z-scores (windows: 21, 63, 252)
    - Relative state: deviation from 252d mean, percentile rank (252d)

    The function is pure and does not access the filesystem or Kedro datasets.
    No economic structural logic is applied - features are purely statistical.

    Args:
        data:
            Dictionary mapping series_id (partition keys) to callables that
            return DataFrames from L2 (`fred_macro_intermediate`). Each DataFrame
            must contain:
            - date (datetime64[ns, UTC])
            - value (float)
            - series_id, asset, category, source, interval, ingestion_ts (metadata)

    Returns:
        Dictionary mapping series_id to DataFrames with:
        - All original L2 columns (metadata + value)
        - Statistical features:
            - delta_1d, pct_change_1d
            - delta_21d, pct_change_21d
            - rolling_mean_21, rolling_std_21, zscore_21
            - rolling_mean_63, rolling_std_63, zscore_63
            - rolling_mean_252, rolling_std_252, zscore_252
            - value_minus_mean_252
            - percentile_rank_252
        - Sorted by date ascending
        - Duplicate dates removed (keep last)
        - NaNs are preserved (burn-in period at series start)
    """
    primary_features: Dict[str, pd.DataFrame] = {}

    for series_id, loader in data.items():
        # Support both callables (PartitionedDataset standard) and direct DataFrames
        if callable(loader):
            df = loader()
        else:
            df = loader  # type: ignore[assignment]

        # Defensive copy to avoid mutating upstream objects
        df_features = df.copy()

        # Ensure clean time axis: sorted and without duplicate dates
        df_features = df_features.sort_values("date")
        df_features = df_features.drop_duplicates(subset=["date"], keep="last")

        # Extract value series for feature computation
        value = df_features["value"]

        # ====================================================================
        # 1. VARIATIONS (Deltas and Percentage Changes)
        # ====================================================================
        # Capture short-term and medium-term changes in the series

        # Daily variations
        df_features["delta_1d"] = value.diff(1)
        df_features["pct_change_1d"] = value.pct_change(1)

        # 21-day variations (approximately 1 month)
        df_features["delta_21d"] = value.diff(21)
        df_features["pct_change_21d"] = value.pct_change(21)

        # ====================================================================
        # 2. ROLLING STATISTICS (Windows: 21, 63, 252)
        # ====================================================================
        # Rolling statistics capture local trends and volatility
        # Windows: 21d (~1 month), 63d (~3 months), 252d (~1 year)

        # Window 21 (approximately 1 month)
        df_features["rolling_mean_21"] = value.rolling(21).mean()
        df_features["rolling_std_21"] = value.rolling(21).std()
        df_features["zscore_21"] = _compute_rolling_zscore(value, window=21)

        # Window 63 (approximately 3 months)
        df_features["rolling_mean_63"] = value.rolling(63).mean()
        df_features["rolling_std_63"] = value.rolling(63).std()
        df_features["zscore_63"] = _compute_rolling_zscore(value, window=63)

        # Window 252 (approximately 1 year)
        df_features["rolling_mean_252"] = value.rolling(252).mean()
        df_features["rolling_std_252"] = value.rolling(252).std()
        df_features["zscore_252"] = _compute_rolling_zscore(value, window=252)

        # ====================================================================
        # 3. RELATIVE STATE
        # ====================================================================
        # Measures how current value relates to historical distribution

        # Deviation from 252-day mean
        df_features["value_minus_mean_252"] = value - df_features["rolling_mean_252"]

        # Percentile rank over 252-day window (normalized between 0 and 1)
        # 0 = at minimum, 1 = at maximum
        def compute_percentile_rank(series: pd.Series, window: int) -> pd.Series:
            """Compute rolling percentile rank (0-1 scale)."""
            ranks = pd.Series(index=series.index, dtype=float)
            for i in range(window - 1, len(series)):
                window_values = series.iloc[i - window + 1 : i + 1]
                if len(window_values) == window:
                    current_value = series.iloc[i]
                    # Count values <= current, normalize by (window - 1)
                    rank = (window_values <= current_value).sum() / (window - 1)
                    ranks.iloc[i] = rank
            return ranks

        df_features["percentile_rank_252"] = compute_percentile_rank(value, window=252)

        primary_features[series_id] = df_features

    return primary_features
