"""
L3 primary pipeline for BTC spot daily → model input.

Node 1: create_btc_l3_features
  spot_daily_clean (L2 PartitionedDataset) + params:l3.crypto.spot_daily
  → spot_daily_features (L3 PartitionedDataset, data/03_primary/spot/daily/)

Node 2: extract_btc_model_input
  spot_daily_features → btc_spot_daily_model_input (data/04_model_input/spot/daily/)
"""

from kedro.pipeline import Pipeline, node

from .nodes import create_btc_l3_features, extract_btc_model_input


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=create_btc_l3_features,
                inputs={
                    "partitions": "spot_daily_clean",
                    "params": "params:l3.crypto.spot_daily",
                },
                outputs="spot_daily_features",
                name="create_btc_l3_features",
            ),
            node(
                func=extract_btc_model_input,
                inputs="spot_daily_features",
                outputs="btc_spot_daily_model_input",
                name="extract_btc_model_input",
            ),
        ]
    )
