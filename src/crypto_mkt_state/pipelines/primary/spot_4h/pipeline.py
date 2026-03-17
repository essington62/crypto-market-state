"""
Kedro pipeline — L3 BTC spot 4h feature engineering.

Reads:  btc_spot_4h_clean      (L2 single-file ParquetDataset)
Writes: btc_spot_features_4h   (L3 single-file ParquetDataset)

Run:
    kedro run --pipeline primary.spot_4h
"""
from kedro.pipeline import Pipeline, node

from .nodes import compute_spot_features_4h


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=compute_spot_features_4h,
                inputs="btc_spot_4h_clean",
                outputs="btc_spot_features_4h",
                name="compute_spot_features_4h",
            )
        ]
    )
