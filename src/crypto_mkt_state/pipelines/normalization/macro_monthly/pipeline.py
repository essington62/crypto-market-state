"""
L2 Macro Monthly: normalize macro monthly partitions (FRED) to canonical L2 schema.

Reads macro_monthly_raw (L1), applies validate_macro_monthly_contract per partition,
writes macro_monthly_clean (L2). Run: kedro run --pipeline normalization.macro_monthly
"""

from kedro.pipeline import Pipeline, node

from .nodes import normalize_macro_monthly_partitions


def create_pipeline() -> Pipeline:
    """Create L2 Macro Monthly normalization pipeline (structural only)."""
    return Pipeline(
        [
            node(
                func=normalize_macro_monthly_partitions,
                inputs="macro_monthly_raw",
                outputs="macro_monthly_clean",
                name="normalize_macro_monthly",
            )
        ]
    )
