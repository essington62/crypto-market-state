"""
Kedro pipeline — L3 order book depth 4h feature engineering.

Reads:  btc_orderbook_4h_clean      (L2 PartitionedDataset — 3 depth ranges)
Writes: btc_orderbook_features_4h   (L3 single-file ParquetDataset)

Run:
    kedro run --pipeline primary.orderbook_4h
"""
from kedro.pipeline import Pipeline, node

from .nodes import compute_orderbook_features_4h


def create_pipeline(**kwargs) -> Pipeline:
    """Create L3 CoinGlass order book 4h feature engineering pipeline."""
    return Pipeline(
        [
            node(
                func=compute_orderbook_features_4h,
                inputs={
                    "partitions":     "btc_orderbook_4h_clean",
                    "rolling_window": "params:primary.orderbook_4h.rolling_window",
                },
                outputs="btc_orderbook_features_4h",
                name="compute_orderbook_features_4h",
            ),
        ]
    )
