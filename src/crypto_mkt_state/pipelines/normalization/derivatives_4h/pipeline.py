"""
Kedro pipeline — L2 CoinGlass derivatives 4h normalization.

Reads:  btc_coinglass_raw_4h       (L1 single-file ParquetDataset)
Writes: btc_coinglass_4h_clean     (L2 single-file ParquetDataset)

Run:
    kedro run --pipeline normalization.derivatives_4h
"""
from kedro.pipeline import Pipeline, node

from .nodes import normalize_coinglass_derivatives_4h


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=normalize_coinglass_derivatives_4h,
                inputs="btc_coinglass_raw_4h",
                outputs="btc_coinglass_4h_clean",
                name="normalize_coinglass_derivatives_4h",
            )
        ]
    )
