"""
L2 normalization pipeline for crypto OHLCV daily data.

This pipeline normalizes schema from L1 (raw) to L2 (intermediate) format:
- Renames columns (open_time → timestamp)
- Sets canonical index (timestamp as DatetimeIndex)
- Preserves all values exactly as received
"""

from kedro.pipeline import Pipeline, node
from crypto_mkt_state.pipelines.normalization.crypto.nodes import (
    normalize_ohlcv_daily_schema,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create L2 normalization pipeline for crypto OHLCV daily data.

    Returns:
        Pipeline that normalizes binance_spot_ohlcv_daily_raw
        to crypto_ohlcv_daily_intermediate format.
    """
    return Pipeline(
        [
            node(
                func=normalize_ohlcv_daily_schema,
                inputs="binance_spot_ohlcv_daily_raw",
                outputs="crypto_ohlcv_daily_intermediate",
                name="normalize_ohlcv_daily_schema",
            ),
        ]
    )
