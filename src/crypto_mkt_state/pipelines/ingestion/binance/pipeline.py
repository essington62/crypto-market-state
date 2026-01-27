"""
Kedro pipeline for Binance L1 ingestion.
"""

from kedro.pipeline import Pipeline, node

from .nodes import load_binance_ohlcv_daily


def create_pipeline() -> Pipeline:
    """Create Binance ingestion pipeline (L1)."""
    return Pipeline(
        [
            node(
                func=load_binance_ohlcv_daily,
                inputs={
                    "assets": "params:binance.assets",
                    "start_date": "params:binance.start_date",
                },
                outputs="binance_spot_ohlcv_daily_raw",
                name="load_binance_ohlcv_daily",
            )
        ]
    )
