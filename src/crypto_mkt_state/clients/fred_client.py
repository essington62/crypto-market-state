"""
FRED API client for fetching macroeconomic time series data.

Design principles:
- Pure client (no Kedro, no IO side effects)
- Explicit control of API interaction
- UTC timestamps only
- Raw FRED schema preserved
- Secrets resolved via environment variables
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd
import requests


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------
def _resolve_fred_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY not found in environment or arguments")
    return key


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def fetch_fred_series(
    series_id: str,
    start_date: str,
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch a single FRED time series.

    L1 Contract:
    - No resampling
    - No renaming
    - No enrichment
    - UTC timestamps
    - Raw frequency preserved
    """

    api_key = _resolve_fred_api_key(api_key)

    params: Dict[str, str] = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }

    # Same logic as yfinance/binance:
    # If end_date is None → fetch until today
    if end_date is not None:
        params["observation_end"] = end_date

    response = requests.get(FRED_BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    observations = response.json().get("observations", [])

    df = pd.DataFrame(observations)

    if df.empty:
        return df

    # Keep only raw schema
    df = df[["date", "value"]].copy()

    # Enforce UTC
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Sort + deduplicate (L1 hygiene, no transformation)
    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    return df


def fetch_fred_batch(
    series_ids: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch multiple FRED series sequentially.

    Returns:
        Dict[series_id, DataFrame]

    Designed for Kedro PartitionedDataset (L1).
    """

    output: Dict[str, pd.DataFrame] = {}

    for series_id in series_ids:
        output[series_id] = fetch_fred_series(
            series_id=series_id,
            start_date=start_date,
            end_date=end_date,  # None = fetch until today
            api_key=api_key,
        )

    return output