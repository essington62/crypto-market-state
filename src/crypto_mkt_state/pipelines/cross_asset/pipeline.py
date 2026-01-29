"""
L4 cross-asset features pipeline.

This pipeline builds a single, non-partitioned dataset of global market
state features by combining:
- `fred_macro_primary`        (FRED macro indices, L3)
- `yfinance_indices_primary`  (Yahoo Finance macro indices, L3)

The output `cross_asset_features` contains one row per date with:
- Risk / stress regimes (e.g. volatility, dollar strength)
- Rates and inflation regimes
- Macro state categorical signals

Contract:
- Consumes ONLY L3 primary data
- Does NOT compute returns, rolling stats, or z-scores
- All logic driven by params:l4
"""

from kedro.pipeline import Pipeline, node

from crypto_mkt_state.pipelines.cross_asset.nodes import (
    build_cross_asset_features,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create L4 cross-asset features pipeline.

    Returns:
        Pipeline that builds `cross_asset_features` from:
        - `fred_macro_primary`
        - `yfinance_indices_primary`
    """
    return Pipeline(
        [
            node(
                func=build_cross_asset_features,
                inputs={
                    "fred": "fred_macro_primary",
                    "yfinance": "yfinance_indices_primary",
                    "l4_config": "params:l4",
                },
                outputs="cross_asset_features",
                name="build_cross_asset_features",
            ),
        ]
    )


