"""
L2 CoinGlass daily indices normalization pipeline.

Reads:   coinglass_indices_raw   (data/01_raw/derivatives/coinglass/indices/)
Writes:  coinglass_indices_clean (data/02_intermediate/derivatives/coinglass/indices/)

Run:
    kedro run --pipeline normalization.coinglass_indices
"""
from kedro.pipeline import Pipeline, node

from .nodes import normalize_coinglass_indices


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=normalize_coinglass_indices,
                inputs="coinglass_indices_raw",
                outputs="coinglass_indices_clean",
                name="normalize_coinglass_indices",
            )
        ]
    )
