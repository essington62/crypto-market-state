"""
L3 primary features pipeline for crypto OHLCV daily data.

This pipeline computes primary per-asset features from L2 intermediate data:
- Returns, Volatilidade, Liquidez, Tendência, Compressão/Expansão, Predictability
"""

from kedro.pipeline import Pipeline, node
from crypto_mkt_state.pipelines.primary.crypto.nodes import (
    compute_primary_features,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create L3 primary features pipeline for crypto OHLCV daily data.

    Returns:
        Pipeline that computes primary features from crypto_ohlcv_daily_intermediate
        to crypto_ohlcv_daily_primary format.
    """
    return Pipeline(
        [
            node(
                func=compute_primary_features,
                inputs={
                    "intermediate_data": "crypto_ohlcv_daily_intermediate",
                    "semantic_config": "params:l3_semantic",
                },
                outputs="crypto_ohlcv_daily_primary",
                name="compute_primary_features",
            ),
        ]
    )
