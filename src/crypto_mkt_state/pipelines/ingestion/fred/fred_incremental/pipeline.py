"""
Kedro pipeline for incremental FRED macro daily ingestion (L1).

Reads existing L1 PartitionedDataset (macro_daily_raw), fetches missing
observations from FRED, and writes updated partitions to
macro_daily_incremental.

Run example:
    kedro run --pipeline ingestion.fred.incremental
"""

from kedro.pipeline import Pipeline, node

from .nodes import update_fred_incremental


def create_pipeline() -> Pipeline:
    """Create L1 incremental FRED ingestion pipeline."""
    return Pipeline(
        [
            node(
                func=update_fred_incremental,
                inputs={
                    "macro_daily_raw": "macro_daily_raw",
                    "fred_series": "params:fred.series",
                    "global_start_date": "params:global.start_date",
                },
                outputs="macro_daily_incremental",
                name="update_fred_incremental",
            ),
        ]
    )

