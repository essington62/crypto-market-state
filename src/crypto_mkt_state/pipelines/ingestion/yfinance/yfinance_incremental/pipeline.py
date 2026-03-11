"""
Kedro pipeline for incremental Yahoo Finance ingestion (L1).

Updates spot_business_day_raw (indices + assets) incrementally and writes
the result to spot_business_day_incremental.

Run example:
    kedro run --pipeline ingestion.yfinance.incremental
"""

from kedro.pipeline import Pipeline, node

from .nodes import update_yfinance_incremental


def create_pipeline() -> Pipeline:
    """Create L1 incremental YFinance ingestion pipeline."""
    return Pipeline(
        [
            node(
                func=update_yfinance_incremental,
                inputs={
                    "spot_business_day_raw": "spot_business_day_raw",
                    "yfinance_indices": "params:yfinance.indices",
                    "yfinance_assets": "params:yfinance.assets",
                    "global_start_date": "params:global.start_date",
                },
                outputs="spot_business_day_incremental",
                name="update_yfinance_incremental",
            ),
        ]
    )

