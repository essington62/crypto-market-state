"""
Kedro pipeline for L2 Spot Daily normalization (asset-type oriented, API-agnostic).

Reads spot_daily_raw (L1), applies validate_and_standardize_spot per partition,
writes spot_daily_clean (L2). Run: kedro run --pipeline normalization.spot
"""

from kedro.pipeline import Pipeline, node

from .nodes import normalize_spot_daily_partitions


def create_pipeline() -> Pipeline:
    """Create L2 Spot Daily normalization pipeline."""
    return Pipeline(
        [
            node(
                func=normalize_spot_daily_partitions,
                inputs="spot_daily_raw",
                outputs="spot_daily_clean",
                name="normalize_spot_daily",
            )
        ]
    )
