"""
L3 CoinGlass daily indices feature pipeline.

Reads:   coinglass_indices_clean          (L2 PartitionedDataset)
         params:coinglass_indices.*        (windows + index config)
Writes:  btc_coinglass_indices_features   (data/03_primary/derivatives/coinglass/indices/BTCUSDT_daily.parquet)

Run:
    kedro run --pipeline primary.coinglass_indices
"""
from kedro.pipeline import Pipeline, node

from .nodes import compute_coinglass_indices_features


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=compute_coinglass_indices_features,
                inputs={
                    "clean":             "coinglass_indices_clean",
                    "coinglass_indices": "params:coinglass_indices.indices",
                    "zscore_window":     "params:coinglass_indices.zscore_window",
                    "change_window":     "params:coinglass_indices.change_window",
                    "ma_window":         "params:coinglass_indices.ma_window",
                },
                outputs="btc_coinglass_indices_features",
                name="compute_coinglass_indices_features",
            )
        ]
    )
