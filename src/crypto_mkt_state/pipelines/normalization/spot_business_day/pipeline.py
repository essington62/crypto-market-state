"""
L2 Spot Business Day: validate and normalize native business-day assets (gold, nasdaq, sp500).

Reads spot_business_day_raw (L1), applies validate_business_day_contract per partition,
writes spot_business_day_clean (L2). Run: kedro run --pipeline normalization.spot_business_day
"""

from kedro.pipeline import Pipeline, node

from .nodes import normalize_spot_business_day_partitions


def create_pipeline() -> Pipeline:
    """Create L2 Spot Business Day normalization pipeline (structural validation only)."""
    return Pipeline(
        [
            node(
                func=normalize_spot_business_day_partitions,
                inputs="spot_business_day_raw",
                outputs="spot_business_day_clean",
                name="normalize_spot_business_day",
            )
        ]
    )
