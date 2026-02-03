"""
L2 normalization nodes for FRED macro data.

L2 contract:
- Preserve date EXACTLY as received from L1
- Do NOT touch timezone
- Add structural metadata only
"""

from typing import Any, Callable, Dict
import pandas as pd


def normalize_fred_macro(
    data: Dict[str, Callable[[], pd.DataFrame]],
    series_meta: list[dict],
) -> Dict[str, pd.DataFrame]:
    """
    L2 normalization for FRED macro series.

    Rules:
    - date column is passed through untouched
    - no timezone conversion
    - no resampling
    - no value transformations
    """

    meta_by_id = {m["id"]: m for m in series_meta if "id" in m}
    normalized: Dict[str, pd.DataFrame] = {}

    for series_id, loader in data.items():
        df = loader() if callable(loader) else loader

        if df is None or df.empty:
            normalized[series_id] = df
            continue

        df_norm = df.copy()

        # Keep original schema
        df_norm = df_norm[["date", "value"]]

        # Attach metadata (ONLY additions)
        meta = meta_by_id.get(series_id, {})
        df_norm["series_id"] = series_id
        df_norm["asset"] = meta.get("name")
        df_norm["category"] = meta.get("category")
        df_norm["source"] = "fred"
        df_norm["interval"] = "1d"
        df_norm["ingestion_ts"] = pd.Timestamp.utcnow()

        # Optional safety (does NOT touch timezone)
        df_norm = (
            df_norm
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
        )

        normalized[series_id] = df_norm

    return normalized
