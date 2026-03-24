"""
L1 Yahoo Finance incremental ingestion pipeline.

Two nodes:
  - update_yfinance_indices → macro_daily_incremental  (VIX, DXY)
  - update_yfinance_assets  → spot_business_day_incremental (SP500, Oil, etc.)
"""

from kedro.pipeline import Pipeline, node

from .nodes import update_yfinance_assets, update_yfinance_indices


def create_pipeline() -> Pipeline:
    """Create L1 incremental Yahoo Finance ingestion pipeline."""
    return Pipeline(
        [
            node(
                func=update_yfinance_indices,
                inputs={
                    "macro_daily_raw": "macro_daily_raw",
                    "yfinance_indices": "params:yfinance.indices",
                    "global_start_date": "params:global.start_date",
                },
                outputs="macro_daily_incremental",
                name="update_yfinance_indices",
            ),
            node(
                func=update_yfinance_assets,
                inputs={
                    "spot_business_day_raw": "spot_business_day_raw",
                    "yfinance_assets": "params:yfinance.assets",
                    "global_start_date": "params:global.start_date",
                },
                outputs="spot_business_day_incremental",
                name="update_yfinance_assets",
            ),
        ]
    )
