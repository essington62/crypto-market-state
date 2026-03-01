"""
L3 Primary features for crypto spot business day (baseline v0 HMM).

Reads spot_business_day_clean (L2), adds deterministic features, writes spot_business_day_features (L3).
Run: kedro run --pipeline primary.spot_business_day
"""

from kedro.pipeline import Pipeline, node

from .nodes import build_spot_business_day_features_partitions


def create_pipeline() -> Pipeline:
    """Create L3 spot business day primary features pipeline."""
    return Pipeline(
        [
            node(
                func=build_spot_business_day_features_partitions,
                inputs={
                    "partitions": "spot_business_day_clean",
                    "params": "params:l3.crypto.spot_daily",
                },
                outputs="spot_business_day_features",
                name="build_spot_business_day_features_l3",
            )
        ]
    )
