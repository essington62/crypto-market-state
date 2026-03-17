"""
Kedro pipeline — L2 order book depth 4h normalization.

Reads:  btc_orderbook_raw_4h    (L1 PartitionedDataset)
Writes: btc_orderbook_4h_clean  (L2 PartitionedDataset)

Run:
    kedro run --pipeline normalization.orderbook_4h
"""
from kedro.pipeline import Pipeline, node

from .nodes import normalize_orderbook_4h


def create_pipeline(**kwargs) -> Pipeline:
    """Create L2 CoinGlass order book 4h normalization pipeline."""
    return Pipeline(
        [
            node(
                func=normalize_orderbook_4h,
                inputs={"partitions": "btc_orderbook_raw_4h"},
                outputs="btc_orderbook_4h_clean",
                name="normalize_orderbook_4h",
            ),
        ]
    )
