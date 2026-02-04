from kedro.pipeline import Pipeline, node

from crypto_mkt_state.pipelines.primary.yfinance.nodes import (
    build_yfinance_assets_primary_features,
    build_yfinance_indices_primary_features,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    L3 primary pipeline for YFinance: strict semantic separation.

    - build_yfinance_assets_primary_features: reads yfinance_assets_intermediate,
      categories {equity_index, commodity}, features: return_*, rolling_std_*, zscore_*, momentum_*.
    - build_yfinance_indices_primary_features: reads yfinance_indices_intermediate,
      categories {volatility, rates, fx}, features: value, delta_*, zscore_63, rolling_mean_63.

    Outputs: yfinance_assets_primary, yfinance_indices_primary (no unification).
    Date column is never converted; already UTC from L2.
    """
    return Pipeline(
        [
            node(
                func=build_yfinance_assets_primary_features,
                inputs={
                    "data": "yfinance_assets_intermediate",
                    "semantic_config": "params:l3_semantic",
                },
                outputs="yfinance_assets_primary",
                name="build_yfinance_assets_primary_features",
            ),
            node(
                func=build_yfinance_indices_primary_features,
                inputs={
                    "data": "yfinance_indices_intermediate",
                    "semantic_config": "params:l3_semantic",
                },
                outputs="yfinance_indices_primary",
                name="build_yfinance_indices_primary_features",
            ),
        ]
    )
