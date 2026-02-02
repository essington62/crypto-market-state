"""
L3 primary feature engineering nodes for crypto OHLCV data.

This module computes primary per-asset features conditionally based on
explicit semantic rules defined in parameters.yml.

Features are computed ONLY if they are explicitly allowed for the asset
in l3_semantic.assets or l3_semantic.indices.
"""

from typing import Dict

import pandas as pd

from crypto_mkt_state.utils.utils_l3_semantic import (
    get_semantic_features,
    normalize_asset_name,
)


def compute_primary_features(
    intermediate_data: Dict[str, callable],
    semantic_config: Dict,
) -> Dict[str, pd.DataFrame]:
    """
    Compute primary features from L2 intermediate data.

    Features are computed conditionally based on explicit semantic rules:
    - Each asset must be defined in l3_semantic.assets (e.g. 'btc')
    - Only features listed in the config are computed
    - Crypto assets get returns, volatility, normalization, momentum
    - Optional: SMA/EMA and price-vs-average relationships (see below)

    SMA/EMA and price-relationship features (e.g. btc_sma_20, btc_close_vs_sma_200):
    These represent heuristics widely used by human traders (personas de mercado).
    They are for modeling collective behavior and future L5 policy use, not for
    prediction. They must NOT be used as target and do NOT enter the HMM in this
    baseline. Rolling windows produce NaN at the start of the series; no drop,
    forward-fill or backfill is applied in L3.

    Args:
        intermediate_data:
            Dictionary mapping partition keys (asset symbols) to callables
            that return DataFrames from L2. Each DataFrame has:
            - Index: timestamp (DatetimeIndex, UTC)
            - Columns: open, high, low, close, volume, quote_volume,
              close_time, trades, taker_buy_base_volume, taker_buy_quote_volume
        semantic_config:
            The `l3_semantic` section from parameters.yml.

    Returns:
        Dictionary mapping partition keys to DataFrames with:
        - Index: timestamp (DatetimeIndex, UTC) - preserved from L2
        - Columns: All L2 columns + conditionally computed features
        - Sorted by timestamp ascending
    """
    primary_features = {}

    for partition_key, load_df in intermediate_data.items():
        df = load_df()

        if df is None or df.empty:
            raise ValueError(
                f"Crypto L3: partition {partition_key} returned empty DataFrame."
            )

        df_features = df.copy()

        asset_name = normalize_asset_name(partition_key)
        try:
            features_allowed = get_semantic_features(
                asset_name=asset_name,
                semantic_config=semantic_config,
            )
        except ValueError as e:
            raise ValueError(
                f"Crypto asset {partition_key} (normalized: {asset_name}): {e}"
            )

        if "close" not in df_features.columns:
            raise ValueError(
                f"Crypto L3: partition {partition_key} missing required column 'close'. "
                f"Available: {list(df_features.columns)}"
            )

        close = df_features["close"]

        # ====================================================================
        # RETURNS FEATURES (only if allowed)
        # ====================================================================
        if "return_1d" in features_allowed:
            df_features["return_1d"] = close.pct_change(1, fill_method=None)

        if "return_5d" in features_allowed:
            df_features["return_5d"] = close.pct_change(5, fill_method=None)

        if "return_21d" in features_allowed:
            df_features["return_21d"] = close.pct_change(21, fill_method=None)

        # ====================================================================
        # VOLATILITY FEATURES (only if allowed)
        # ====================================================================
        if "rolling_std_21" in features_allowed:
            returns_1d = df_features.get("return_1d", close.pct_change(1, fill_method=None))
            df_features["rolling_std_21"] = returns_1d.rolling(21).std()

        if "rolling_std_63" in features_allowed:
            returns_1d = df_features.get("return_1d", close.pct_change(1, fill_method=None))
            df_features["rolling_std_63"] = returns_1d.rolling(63).std()

        # ====================================================================
        # NORMALIZATION FEATURES (z-scores, only if allowed)
        # ====================================================================
        if "zscore_21" in features_allowed:
            rolling_mean_21 = close.rolling(21).mean()
            rolling_std_21 = close.rolling(21).std()
            df_features["zscore_21"] = (close - rolling_mean_21) / rolling_std_21

        if "zscore_63" in features_allowed:
            rolling_mean_63 = close.rolling(63).mean()
            rolling_std_63 = close.rolling(63).std()
            df_features["zscore_63"] = (close - rolling_mean_63) / rolling_std_63

        # ====================================================================
        # MOMENTUM FEATURES (only if allowed)
        # ====================================================================
        if "momentum_21" in features_allowed:
            df_features["momentum_21"] = close / close.shift(21) - 1

        if "momentum_63" in features_allowed:
            df_features["momentum_63"] = close / close.shift(63) - 1

        # ====================================================================
        # SMA / EMA e relações de preço (apenas se permitido no semantic)
        # Heurísticas de traders; não são target e não entram no HMM.
        # NaN no início da série são permitidos (sem drop/ffill/bfill).
        # ====================================================================
        if "btc_sma_20" in features_allowed:
            df_features["btc_sma_20"] = close.rolling(20).mean()
        if "btc_sma_50" in features_allowed:
            df_features["btc_sma_50"] = close.rolling(50).mean()
        if "btc_sma_100" in features_allowed:
            df_features["btc_sma_100"] = close.rolling(100).mean()
        if "btc_sma_200" in features_allowed:
            df_features["btc_sma_200"] = close.rolling(200).mean()

        if "btc_ema_20" in features_allowed:
            df_features["btc_ema_20"] = close.ewm(span=20, adjust=False).mean()
        if "btc_ema_50" in features_allowed:
            df_features["btc_ema_50"] = close.ewm(span=50, adjust=False).mean()
        if "btc_ema_200" in features_allowed:
            df_features["btc_ema_200"] = close.ewm(span=200, adjust=False).mean()

        if "btc_close_vs_sma_200" in features_allowed and "btc_sma_200" in df_features.columns:
            df_features["btc_close_vs_sma_200"] = (close / df_features["btc_sma_200"]) - 1
        if "btc_close_above_sma_200" in features_allowed and "btc_sma_200" in df_features.columns:
            df_features["btc_close_above_sma_200"] = close > df_features["btc_sma_200"]

        if "btc_sma_50_vs_200" in features_allowed:
            if "btc_sma_50" in df_features.columns and "btc_sma_200" in df_features.columns:
                df_features["btc_sma_50_vs_200"] = (
                    df_features["btc_sma_50"] / df_features["btc_sma_200"]
                ) - 1
        if "btc_golden_cross" in features_allowed:
            if "btc_sma_50" in df_features.columns and "btc_sma_200" in df_features.columns:
                df_features["btc_golden_cross"] = (
                    df_features["btc_sma_50"] > df_features["btc_sma_200"]
                )

        df_features = df_features.sort_index()
        primary_features[partition_key] = df_features

    return primary_features
