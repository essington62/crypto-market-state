"""
Kedro pipeline for Binance Spot L1 ingestion (24x7 market).
"""

from kedro.pipeline import Pipeline, node

from .nodes import load_binance_ohlcv_daily


def create_pipeline() -> Pipeline:
    """Create Binance Spot ingestion pipeline (L1)."""

    return Pipeline(
        [
            node(
                func=load_binance_ohlcv_daily,
                inputs={
                    "assets": "params:binance.assets",
                    "start_date": "params:global.start_date",
                    "interval": "params:binance.interval",
                },
                outputs="spot_crypto_daily_24x7_raw",
                name="load_binance_spot_ohlcv_daily",
            )
        ]
    )