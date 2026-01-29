"""
Kedro pipeline for FRED L1 ingestion.
"""

from kedro.pipeline import Pipeline, node

from .nodes import load_fred_l1


def create_pipeline() -> Pipeline:
    """Create FRED ingestion pipeline (L1)."""
    return Pipeline(
        [
            node(
                func=load_fred_l1,
                inputs={
                    "series": "params:fred.series",
                    "start_date": "params:global.start_date",
                    "interval": "params:global.interval",
                },
                outputs="fred_macro_raw",
                name="load_fred_l1",
            )
        ]
    )
