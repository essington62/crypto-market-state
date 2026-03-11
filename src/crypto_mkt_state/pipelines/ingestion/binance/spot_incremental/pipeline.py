"""
Kedro pipeline for incremental Binance Spot OHLCV ingestion (L1).
"""

from kedro.pipeline import Pipeline, node
from .nodes import update_binance_spot_incremental


def create_pipeline() -> Pipeline:
    """Create L1 incremental ingestion pipeline for Binance spot OHLCV."""

    return Pipeline(
        [
            node(
                func=update_binance_spot_incremental,
                inputs=dict(
                    existing_data="spot_crypto_daily_24x7_raw",
                    spot_symbols="params:binance.spot_symbols",
                    interval="params:binance.interval",
                    global_start_date="params:global.start_date",
                ),
                outputs="spot_crypto_daily_24x7_updated",
                name="update_binance_spot_incremental",
            )
        ]
    )