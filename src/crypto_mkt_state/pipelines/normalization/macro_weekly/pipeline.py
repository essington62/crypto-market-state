"""
L2 Macro Weekly: normalize macro weekly partitions (FRED) to canonical L2 schema.

Reads macro_weekly_raw (L1), applies validate_macro_weekly_contract per partition,
writes macro_weekly_clean (L2). Run: kedro run --pipeline normalization.macro_weekly
"""

from kedro.pipeline import Pipeline, node

from .nodes import normalize_macro_weekly_partitions


def create_pipeline() -> Pipeline:
    """Create L2 Macro Weekly normalization pipeline (structural only)."""
    return Pipeline(
        [
            node(
                func=normalize_macro_weekly_partitions,
                inputs="macro_weekly_raw",
                outputs="macro_weekly_clean",
                name="normalize_macro_weekly",
            )
        ]
    )
