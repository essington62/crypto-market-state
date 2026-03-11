"""
Kedro pipeline for incremental Coinglass intraday derivatives (1h) ingestion (L1).

Reads existing PartitionedDataset (coinglass_derivatives_1h_raw), fetches missing
intraday data for configured assets, and writes updated data to
coinglass_derivatives_1h_updated.

Run:
    kedro run --pipeline ingestion.coinglass.derivatives_1h
"""

from kedro.pipeline import Pipeline, node

from .nodes import update_coinglass_1h_incremental


def create_pipeline() -> Pipeline:
    """Create L1 incremental Coinglass 1h derivatives pipeline."""
    return Pipeline(
        [
            node(
                func=update_coinglass_1h_incremental,
                inputs={
                    "existing_data": "params:coinglass_intraday.dummy",
                    "assets": "params:coinglass_intraday.assets",
                    "start_date": "params:coinglass_intraday.start_date",
                    "interval": "params:coinglass_intraday.interval",
                },
                outputs="coinglass_derivatives_1h_updated",
                name="update_coinglass_1h_incremental",
            ),
        ]
    )

