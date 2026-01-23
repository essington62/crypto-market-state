"""
L2 normalization nodes for crypto OHLCV data.

This module contains nodes that perform schema normalization only:
- Column renaming (open_time → timestamp)
- Index setting (timestamp as DatetimeIndex)
- Type preservation (no value transformations)
"""

from typing import Dict

import pandas as pd

"""
    Normalize OHLCV daily schema from L1 to L2 canonical format.

    This node performs ONLY schema normalization:
    - Renames open_time → timestamp
    - Sets timestamp as DatetimeIndex (UTC)
    - Ensures ascending order
    - Preserves all values exactly as received from L1

    Processes each partition (asset) independently, maintaining
    the partitioned structure.

    Args:
        raw_data:
            Dictionary mapping partition keys (asset symbols) to DataFrames
            from L1. Each DataFrame has columns:
            - open_time (datetime64[ns, UTC])
            - open, high, low, close (float64)
            - volume, quote_volume (float64)
            - close_time (datetime64[ns, UTC])
            - trades (int64)
            - taker_buy_base_volume, taker_buy_quote_volume (float64)

    Returns:
        Dictionary mapping partition keys to normalized DataFrames with:
        - Index: timestamp (DatetimeIndex, UTC)
        - Columns: same as L1 except open_time renamed to timestamp
        - All values preserved exactly from L1
        - Sorted by timestamp ascending
    """
    
def normalize_ohlcv_daily_schema(
    raw_data: Dict[str, callable],
) -> Dict[str, pd.DataFrame]:
    normalized = {}

    for partition_key, load_df in raw_data.items():
        # 🔑 Load the actual DataFrame
        df = load_df()

        # Defensive copy
        df_normalized = df.copy()

        # Rename open_time → timestamp
        df_normalized = df_normalized.rename(columns={"open_time": "timestamp"})

        # Set index
        df_normalized = df_normalized.set_index("timestamp")

        # Ensure order
        df_normalized = df_normalized.sort_index()

        normalized[partition_key] = df_normalized

    return normalized
