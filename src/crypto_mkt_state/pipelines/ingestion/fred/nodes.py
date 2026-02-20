from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd

from crypto_mkt_state.clients.fred_client import fetch_fred_batch
from crypto_mkt_state.utils.utils_temporal import enforce_l1_temporal_contract


def load_fred_l1(
    series: List[dict],
    start_date: str,
    interval: str,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    L1 ingestion for FRED macro data.

    Contract (STRICT):
    - Exact mirror of FRED origin
    - Original frequency preserved (daily / weekly / monthly)
    - UTC timestamps
    - No resampling
    - No forward-fill
    - No feature engineering
    - Only temporal cut enforcement
    """

    raw = fetch_fred_batch(
        series_ids=[s["id"] for s in series],
        start_date=start_date,
        end_date=end_date,
    )

    output: Dict[str, pd.DataFrame] = {}

    for cfg in series:
        series_id = cfg["id"]
        df = raw.get(series_id)

        if df is None or df.empty:
            output[series_id] = df if df is not None else pd.DataFrame()
            continue

        # Defensive copy
        df = df.copy()

        # Ensure UTC
        df["date"] = pd.to_datetime(df["date"], utc=True)

        # Sort & deduplicate
        df = (
            df.sort_values("date")
              .drop_duplicates(subset=["date"], keep="last")
              .reset_index(drop=True)
        )

        # Enforce temporal cut only (no frequency assertion)
        df = enforce_l1_temporal_contract(
            df=df,
            start_date=start_date,
            interval=interval,
            assert_daily=False,   # FRED is not necessarily daily
        )

        output[series_id] = df

    return output
