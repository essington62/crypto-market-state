"""
L2 Macro Daily: normalize macro daily partitions (FRED, Yahoo) to canonical L2 schema.

Reads macro_daily_raw (L1), applies validate_macro_daily_contract per partition,
writes macro_daily_clean (L2). Run: kedro run --pipeline normalization.macro_daily
"""

from kedro.pipeline import Pipeline, node

from .nodes import normalize_macro_daily_partitions


def create_pipeline() -> Pipeline:
    """Create L2 Macro Daily normalization pipeline (structural only)."""
    return Pipeline(
        [
            node(
                func=normalize_macro_daily_partitions,
                inputs="macro_daily_raw",
                outputs="macro_daily_clean",
                name="normalize_macro_daily",
            )
        ]
    )
