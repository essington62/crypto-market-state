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

from typing import Dict, Union, Callable
import pandas as pd


def normalize_ohlcv_daily_schema(
    raw_data: Dict[str, Union[pd.DataFrame, Callable[[], pd.DataFrame]]],
) -> Dict[str, pd.DataFrame]:
    """
    Normalize OHLCV daily schema from L1 to L2 canonical format.

    Transformation (and ONLY this):
    - open_time -> date
    """

    if not raw_data:
        raise ValueError("L2 normalize_ohlcv_daily_schema: raw_data is empty")

    normalized: Dict[str, pd.DataFrame] = {}

    for partition_key, obj in raw_data.items():
        # ✅ Support lazy AND eager partitions
        df = obj() if callable(obj) else obj

        if df is None or df.empty:
            continue

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"L2 normalize_ohlcv_daily_schema: partition '{partition_key}' "
                f"is not a DataFrame (got {type(df)})"
            )

        if "open_time" not in df.columns:
            raise ValueError(
                f"L2 normalize_ohlcv_daily_schema: missing 'open_time' "
                f"in partition '{partition_key}'. Columns: {list(df.columns)}"
            )

        out = df.rename(columns={"open_time": "date"})

        normalized[partition_key] = out

    if not normalized:
        raise ValueError(
            "L2 normalize_ohlcv_daily_schema: no valid partitions after normalization"
        )

    return normalized
