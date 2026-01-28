"""
L3 primary features pipeline for Yahoo Finance macro data.

This pipeline computes primary per-asset statistical features from L2
intermediate data:
- Price-based returns and rolling statistics
- Realized volatility
- Relative price state and volume normalization
"""

from kedro.pipeline import Pipeline, node

from crypto_mkt_state.pipelines.primary.yfinance.nodes import (
    build_yfinance_primary_features,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create L3 primary features pipeline for Yahoo Finance macro data.

    Returns:
        Pipeline that computes primary statistical features from
        `yfinance_macro_intermediate` to `yfinance_macro_primary` format.
    """
    return Pipeline(
        [
            node(
                func=build_yfinance_primary_features,
                inputs="yfinance_macro_intermediate",
                outputs="yfinance_macro_primary",
                name="build_yfinance_primary_features",
            ),
        ]
    )

