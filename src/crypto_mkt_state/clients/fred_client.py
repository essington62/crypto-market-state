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
    """
    Resolve FRED API key.

    Priority:
    1. Explicit argument
    2. Environment variable FRED_API_KEY
    """
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

    Returns raw FRED schema:
        - date (datetime64[ns, UTC])
        - value (float, NaN if missing)

    No resampling, no renaming, no enrichment.
    """
    api_key = _resolve_fred_api_key(api_key)

    params: Dict[str, str] = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }

    if end_date is not None:
        params["observation_end"] = end_date

    response = requests.get(FRED_BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    observations = response.json().get("observations", [])

    df = pd.DataFrame(observations, columns=["date", "value"])

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

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

    Designed for use by Kedro PartitionedDataset.
    """
    output: Dict[str, pd.DataFrame] = {}

    for series_id in series_ids:
        df = fetch_fred_series(
            series_id=series_id,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
        )
        output[series_id] = df

    return output
