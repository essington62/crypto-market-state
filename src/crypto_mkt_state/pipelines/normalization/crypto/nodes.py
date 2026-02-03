"""
L2 normalization node for crypto OHLCV data (Binance).

Purpose:
- Canonicalize column naming ONLY.
- No temporal logic.
- No timezone conversion.
- No sorting assumptions.
- No value transformations.

L2 contract:
- `date` must exist as a column
- Index is non-semantic
- All other columns preserved exactly
"""

from typing import Dict, Callable
import pandas as pd


def normalize_ohlcv_daily_schema(
    raw_data: Dict[str, Callable[[], pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    """
    Normalize OHLCV daily schema from L1 to L2 canonical format.

    Transformation (and ONLY this):
    - open_time -> date
    """

    normalized: Dict[str, pd.DataFrame] = {}

    for partition_key, load_df in raw_data.items():
        df = load_df()

        # Pass-through empty partitions
        if df is None or df.empty:
            normalized[partition_key] = df
            continue

        df = df.copy()

        if "open_time" not in df.columns:
            raise ValueError(
                f"L2 normalize_ohlcv_daily_schema: missing 'open_time' "
                f"in partition '{partition_key}'. Columns: {list(df.columns)}"
            )

        # 🔑 ONLY rename
        df = df.rename(columns={"open_time": "date"})

        normalized[partition_key] = df

    return normalized
