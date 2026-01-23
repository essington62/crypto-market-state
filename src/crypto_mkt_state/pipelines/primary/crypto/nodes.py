"""
L3 primary feature engineering nodes for crypto OHLCV data.

This module contains nodes that compute primary per-asset features:
- Returns (log returns in multiple windows)
- Volatilidade (multiple measures and windows)
- Liquidez (volumes, trades, buy pressure)
- Tendência (moving averages, slopes, relative position)
- Compressão/expansão (candle ratios, shadows)
- Predictability (autocorrelation, Hurst)
"""

from typing import Dict

import numpy as np
import pandas as pd


def compute_primary_features(
    intermediate_data: Dict[str, callable],
) -> Dict[str, pd.DataFrame]:
    """
    Compute primary features from L2 intermediate data.

    This node computes 41 primary features per asset, preserving all
    original columns from L2. Features are organized in categories:
    - Returns (4 features)
    - Volatilidade (7 features)
    - Liquidez (12 features)
    - Tendência (7 features)
    - Compressão/expansão (9 features)
    - Predictability (2 features)

    Args:
        intermediate_data:
            Dictionary mapping partition keys (asset symbols) to callables
            that return DataFrames from L2. Each DataFrame has:
            - Index: timestamp (DatetimeIndex, UTC)
            - Columns: open, high, low, close, volume, quote_volume,
              close_time, trades, taker_buy_base_volume, taker_buy_quote_volume

    Returns:
        Dictionary mapping partition keys to DataFrames with:
        - Index: timestamp (DatetimeIndex, UTC) - preserved from L2
        - Columns: All L2 columns + 41 primary features
        - Sorted by timestamp ascending
    """
    primary_features = {}

    for partition_key, load_df in intermediate_data.items():
        # Load the actual DataFrame
        df = load_df()

        # Create a copy to avoid modifying input
        df_features = df.copy()

        # ====================================================================
        # 1. RETURNS (Log Returns)
        # ====================================================================
        # Log returns are fundamental for characterizing regime, stress, and
        # predictability. Multiple windows capture different time horizons.

        # Daily log return
        df_features["log_return_1d"] = np.log(df_features["close"] / df_features["close"].shift(1))

        # Cumulative log returns for different windows
        df_features["log_return_7d"] = np.log(df_features["close"] / df_features["close"].shift(7))
        df_features["log_return_21d"] = np.log(df_features["close"] / df_features["close"].shift(21))
        df_features["log_return_63d"] = np.log(df_features["close"] / df_features["close"].shift(63))

        # ====================================================================
        # 2. VOLATILIDADE
        # ====================================================================
        # Volatilidade measures capture market uncertainty and stress.
        # Multiple measures provide robustness.

        # Standard deviation of log returns (rolling windows)
        log_return_1d = df_features["log_return_1d"]
        df_features["volatility_7d"] = log_return_1d.rolling(7).std() * np.sqrt(7)
        df_features["volatility_21d"] = log_return_1d.rolling(21).std() * np.sqrt(21)
        df_features["volatility_63d"] = log_return_1d.rolling(63).std() * np.sqrt(63)

        # Price range normalized by mean price
        rolling_high_max = df_features["high"].rolling(7).max()
        rolling_low_min = df_features["low"].rolling(7).min()
        rolling_close_mean = df_features["close"].rolling(7).mean()
        df_features["price_range_7d"] = (rolling_high_max - rolling_low_min) / rolling_close_mean

        rolling_high_max = df_features["high"].rolling(21).max()
        rolling_low_min = df_features["low"].rolling(21).min()
        rolling_close_mean = df_features["close"].rolling(21).mean()
        df_features["price_range_21d"] = (rolling_high_max - rolling_low_min) / rolling_close_mean

        # Realized volatility (sum of absolute log returns)
        df_features["realized_volatility_7d"] = log_return_1d.abs().rolling(7).sum()
        df_features["realized_volatility_21d"] = log_return_1d.abs().rolling(21).sum()

        # ====================================================================
        # 3. LIQUIDEZ / VOLUME
        # ====================================================================
        # Liquidez features help identify regime stability and stress events.
        # Volume anomalies often precede significant moves.

        # Volume moving averages
        df_features["volume_ma_7d"] = df_features["volume"].rolling(7).mean()
        df_features["volume_ma_21d"] = df_features["volume"].rolling(21).mean()

        # Volume z-scores (standardized deviations from mean)
        volume_ma_7d = df_features["volume"].rolling(7).mean()
        volume_std_7d = df_features["volume"].rolling(7).std()
        df_features["volume_zscore_7d"] = (df_features["volume"] - volume_ma_7d) / volume_std_7d

        volume_ma_21d = df_features["volume"].rolling(21).mean()
        volume_std_21d = df_features["volume"].rolling(21).std()
        df_features["volume_zscore_21d"] = (df_features["volume"] - volume_ma_21d) / volume_std_21d

        # Volume change (percentage change over 7 days)
        df_features["volume_change_7d"] = (
            (df_features["volume"] - df_features["volume"].shift(7)) / df_features["volume"].shift(7)
        )

        # Quote volume moving averages
        df_features["quote_volume_ma_7d"] = df_features["quote_volume"].rolling(7).mean()
        df_features["quote_volume_ma_21d"] = df_features["quote_volume"].rolling(21).mean()

        # Trades moving averages
        df_features["trades_ma_7d"] = df_features["trades"].rolling(7).mean()
        df_features["trades_ma_21d"] = df_features["trades"].rolling(21).mean()

        # Trades z-score
        trades_ma_7d = df_features["trades"].rolling(7).mean()
        trades_std_7d = df_features["trades"].rolling(7).std()
        df_features["trades_zscore_7d"] = (df_features["trades"] - trades_ma_7d) / trades_std_7d

        # Buy pressure (ratio of taker buy volume to total quote volume)
        buy_pressure = df_features["taker_buy_quote_volume"] / df_features["quote_volume"]
        df_features["buy_pressure_7d"] = buy_pressure.rolling(7).mean()
        df_features["buy_pressure_21d"] = buy_pressure.rolling(21).mean()

        # ====================================================================
        # 4. TENDÊNCIA
        # ====================================================================
        # Tendência features characterize market regime (uptrend, downtrend, sideways).
        # Slopes and relative positions provide directional information.

        # Price moving averages
        df_features["price_ma_7d"] = df_features["close"].rolling(7).mean()
        df_features["price_ma_21d"] = df_features["close"].rolling(21).mean()
        df_features["price_ma_63d"] = df_features["close"].rolling(63).mean()

        # Price slopes (linear regression coefficient over rolling windows)
        # Using simple linear regression: y = a + b*x where x is days
        def compute_slope(series, window):
            """Compute slope of linear regression over rolling window."""
            slopes = pd.Series(index=series.index, dtype=float)
            for i in range(window - 1, len(series)):
                y = series.iloc[i - window + 1 : i + 1].values
                x = np.arange(len(y))
                if len(y) == window and not np.isnan(y).any():
                    coeffs = np.polyfit(x, y, 1)
                    slopes.iloc[i] = coeffs[0]
            return slopes

        df_features["price_slope_21d"] = compute_slope(df_features["close"], 21)
        df_features["price_slope_63d"] = compute_slope(df_features["close"], 63)

        # Price position within rolling range (0 = at low, 1 = at high)
        rolling_high_max_7d = df_features["high"].rolling(7).max()
        rolling_low_min_7d = df_features["low"].rolling(7).min()
        range_7d = rolling_high_max_7d - rolling_low_min_7d
        df_features["price_position_7d"] = (df_features["close"] - rolling_low_min_7d) / range_7d

        rolling_high_max_21d = df_features["high"].rolling(21).max()
        rolling_low_min_21d = df_features["low"].rolling(21).min()
        range_21d = rolling_high_max_21d - rolling_low_min_21d
        df_features["price_position_21d"] = (df_features["close"] - rolling_low_min_21d) / range_21d

        # ====================================================================
        # 5. COMPRESSÃO / EXPANSÃO DE PREÇO
        # ====================================================================
        # Compressão/expansão features capture price action structure.
        # Compression often precedes significant moves (predictability).

        # Candle body ratio (body size relative to total range)
        candle_range = df_features["high"] - df_features["low"]
        candle_body = (df_features["close"] - df_features["open"]).abs()
        df_features["candle_body_ratio"] = candle_body / candle_range.replace(0, np.nan)
        df_features["candle_body_ratio_7d"] = df_features["candle_body_ratio"].rolling(7).mean()
        df_features["candle_body_ratio_21d"] = df_features["candle_body_ratio"].rolling(21).mean()

        # Upper and lower shadow ratios
        upper_shadow = df_features["high"] - df_features[["open", "close"]].max(axis=1)
        lower_shadow = df_features[["open", "close"]].min(axis=1) - df_features["low"]
        df_features["upper_shadow_ratio"] = upper_shadow / candle_range.replace(0, np.nan)
        df_features["lower_shadow_ratio"] = lower_shadow / candle_range.replace(0, np.nan)

        # Combined shadow ratio (average of upper + lower)
        shadow_sum = df_features["upper_shadow_ratio"] + df_features["lower_shadow_ratio"]
        df_features["shadow_ratio_7d"] = shadow_sum.rolling(7).mean()
        df_features["shadow_ratio_21d"] = shadow_sum.rolling(21).mean()

        # High/low ratio (expansion measure)
        df_features["high_low_ratio_7d"] = (
            df_features["high"].rolling(7).max() / df_features["low"].rolling(7).min()
        )
        df_features["high_low_ratio_21d"] = (
            df_features["high"].rolling(21).max() / df_features["low"].rolling(21).min()
        )

        # ====================================================================
        # 6. PREDICTABILITY
        # ====================================================================
        # Predictability features measure structure and persistence in returns.
        # Autocorrelation and Hurst exponent indicate regime characteristics.

        # Autocorrelation of log returns (lag 1, rolling 21 days)
        def compute_autocorr(series, lag, window):
            """Compute autocorrelation with given lag over rolling window."""
            autocorrs = pd.Series(index=series.index, dtype=float)
            for i in range(window - 1, len(series)):
                y = series.iloc[i - window + 1 : i + 1].values
                if len(y) == window and not np.isnan(y).any():
                    if len(y) > lag:
                        autocorrs.iloc[i] = np.corrcoef(y[:-lag], y[lag:])[0, 1]
            return autocorrs

        df_features["autocorr_return_1d"] = compute_autocorr(log_return_1d, lag=1, window=21)

        # Hurst exponent (approximated via R/S analysis)
        def compute_hurst(series, window):
            """Compute Hurst exponent via R/S analysis over rolling window."""
            hurst_values = pd.Series(index=series.index, dtype=float)
            for i in range(window - 1, len(series)):
                y = series.iloc[i - window + 1 : i + 1].values
                if len(y) == window and not np.isnan(y).any():
                    # R/S analysis
                    mean_y = np.mean(y)
                    deviations = y - mean_y
                    cumulative_deviations = np.cumsum(deviations)
                    R = np.max(cumulative_deviations) - np.min(cumulative_deviations)  # Range
                    S = np.std(y)  # Standard deviation
                    if S > 0:
                        RS = R / S
                        # Approximate Hurst: H ≈ log(RS) / log(window)
                        if RS > 0:
                            hurst_values.iloc[i] = np.log(RS) / np.log(window)
            return hurst_values

        df_features["hurst_exponent_63d"] = compute_hurst(log_return_1d, window=63)

        # Ensure ascending order (safety check)
        df_features = df_features.sort_index()

        primary_features[partition_key] = df_features

    return primary_features
