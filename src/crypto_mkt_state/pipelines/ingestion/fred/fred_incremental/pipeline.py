"""
L1 FRED Incremental Ingestion pipeline.

Incrementally updates FRED series data from FRED API.
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
                    "params_fred": "params:fred",  # CORRETO: passa o dicionário completo
                    "global_start_date": "params:global.start_date",
                },
                outputs="macro_daily_incremental",
                name="update_fred_incremental",
            )
        ]
    )