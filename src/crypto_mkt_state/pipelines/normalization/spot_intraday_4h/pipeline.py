"""
Kedro pipeline — L2 Spot Intraday 4h normalization.

Reads:  spot_crypto_4h_updated  (L1 PartitionedDataset)
Writes: spot_crypto_4h_normalized (L2 PartitionedDataset)

Run:
    kedro run --pipeline normalization.spot_intraday_4h
"""
from kedro.pipeline import Pipeline, node

from .nodes import normalize_spot_4h_partitions


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=normalize_spot_4h_partitions,
                inputs={
                    "partitions": "spot_crypto_4h_updated",
                    "max_candles": "params:normalization.spot_intraday_4h.max_candles",
                },
                outputs="spot_crypto_4h_normalized",
                name="normalize_spot_intraday_4h",
            )
        ]
    )
