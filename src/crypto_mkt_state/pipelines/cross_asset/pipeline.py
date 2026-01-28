"""
L4 cross-asset features pipeline.

This pipeline builds a single, non-partitioned dataset of global market
state features by combining:
- `fred_macro_primary`  (FRED macro series, L3)
- `yfinance_macro_primary` (Yahoo Finance macro assets, L3)

The output `cross_asset_features` contains one row per date with:
- Risk / stress indicators
- Liquidity / dollar measures
- Macro regime differentials
- Cross-asset relationships
- Volatility regime flags
"""

from kedro.pipeline import Pipeline, node

from crypto_mkt_state.pipelines.cross_asset.nodes import build_cross_asset_features


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create L4 cross-asset features pipeline.

    Returns:
        Pipeline that builds `cross_asset_features` from:
        - `fred_macro_primary`
        - `yfinance_macro_primary`
    """
    return Pipeline(
        [
            node(
                func=build_cross_asset_features,
                inputs={
                    "fred": "fred_macro_primary",
                    "yfinance": "yfinance_macro_primary",
                },
                outputs="cross_asset_features",
                name="build_cross_asset_features",
            ),
        ]
    )

