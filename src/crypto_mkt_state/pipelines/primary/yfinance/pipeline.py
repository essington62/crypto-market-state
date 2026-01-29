from kedro.pipeline import Pipeline, node

from crypto_mkt_state.pipelines.primary.yfinance.nodes import (
    build_yfinance_assets_primary_features,
    build_yfinance_indices_primary_features,
    merge_yfinance_primary_partitions,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    L3 primary pipeline for YFinance: two semantic nodes + merge.

    - build_yfinance_assets_primary_features: equity_index, commodity only (returns, vol, momentum).
    - build_yfinance_indices_primary_features: volatility, rates, fx only (value, deltas, zscore, rolling_mean).
    - merge_yfinance_primary_partitions: single yfinance_macro_primary for L4.
    """
    return Pipeline(
        [
            node(
                func=build_yfinance_assets_primary_features,
                inputs={
                    "data": "yfinance_macro_intermediate",
                    "semantic_config": "params:l3_semantic",
                },
                outputs="yfinance_assets_primary",
                name="build_yfinance_assets_primary_features",
            ),
            node(
                func=build_yfinance_indices_primary_features,
                inputs={
                    "data": "yfinance_macro_intermediate",
                    "semantic_config": "params:l3_semantic",
                },
                outputs="yfinance_indices_primary",
                name="build_yfinance_indices_primary_features",
            ),
            node(
                func=merge_yfinance_primary_partitions,
                inputs={
                    "assets_data": "yfinance_assets_primary",
                    "indices_data": "yfinance_indices_primary",
                },
                outputs="yfinance_macro_primary",
                name="merge_yfinance_primary_partitions",
            ),
        ]
    )
