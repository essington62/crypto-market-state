"""
L3 primary feature engineering nodes for FRED macro data.

This module computes primary statistical features from L2 intermediate data,
conditionally based on explicit semantic rules defined in parameters.yml.

Features are computed ONLY if they are explicitly allowed for the asset
in l3_semantic.assets or l3_semantic.indices.
"""

from __future__ import annotations

from typing import Callable, Dict

import pandas as pd

from crypto_mkt_state.utils.utils_l3_semantic import get_semantic_features


def build_fred_primary_features(
    data: Dict[str, Callable[[], pd.DataFrame]],
    semantic_config: Dict,
) -> Dict[str, pd.DataFrame]:
    """
    Build L3 primary features for FRED macro series.

    Features are computed conditionally based on explicit semantic rules:
    - Each series must be defined in l3_semantic.assets or l3_semantic.indices
    - Only features listed in the config are computed
    - No returns or volatility for macro indices (e.g. CPI, rates)

    Contract:
    - Recebe dados já cortados e validados temporalmente pela L1
    - NÃO aplica start_date
    - NÃO altera frequência
    - NÃO faz resample
    """

    output: Dict[str, pd.DataFrame] = {}

    for series_id, loader in data.items():
        df = loader() if callable(loader) else loader

        if df is None or df.empty:
            output[series_id] = df
            continue

        df = df.copy()

        # Sanidade mínima (permitida)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date")
        df = df.drop_duplicates(subset=["date"], keep="last")

        # Get asset name from metadata (L2 adds 'asset' column)
        if "asset" not in df.columns:
            raise ValueError(
                f"FRED series {series_id} missing 'asset' column. "
                "L2 normalization should have added this."
            )

        asset_name = str(df["asset"].iloc[0])

        # Resolve allowed features from semantic config
        try:
            features_allowed = get_semantic_features(
                asset_name=asset_name,
                semantic_config=semantic_config,
            )
        except ValueError as e:
            raise ValueError(
                f"FRED series {series_id} (asset={asset_name}): {e}"
            )

        # Compute features conditionally
        if "value" not in df.columns:
            output[series_id] = df
            continue

        value = df["value"]

        # Level feature
        if "value" in features_allowed:
            # Value is already in the DataFrame, no computation needed
            pass

        # Changes features
        if "delta_1d" in features_allowed:
            df["delta_1d"] = value.diff(1)

        if "delta_21d" in features_allowed:
            df["delta_21d"] = value.diff(21)

        # Normalization features
        if "zscore_63" in features_allowed:
            rolling_mean_63 = value.rolling(63).mean()
            rolling_std_63 = value.rolling(63).std()
            df["zscore_63"] = (value - rolling_mean_63) / rolling_std_63

        # Trends features
        if "rolling_mean_63" in features_allowed:
            df["rolling_mean_63"] = value.rolling(63).mean()

        output[series_id] = df

    return output
