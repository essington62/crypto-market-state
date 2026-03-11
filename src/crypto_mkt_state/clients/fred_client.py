"""
FRED API client for fetching macroeconomic time series data.

Design principles
-----------------
- Pure client (no Kedro dependencies)
- Explicit API interaction
- UTC timestamps only
- Raw FRED schema preserved
- Secrets resolved via environment variables or arguments
- Safe retry logic for API stability
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import pandas as pd
import requests


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_SLEEP = 1.5


# -------------------------------------------------------------------
# API Key Resolver
# -------------------------------------------------------------------


def _resolve_fred_api_key(api_key: Optional[str]) -> str:
    """
    Resolve FRED API key.

    Priority:
    1. Explicit function argument
    2. Environment variable FRED_API_KEY
    """

    if api_key:
        return api_key

    env_key = os.getenv("FRED_API_KEY")

    if env_key:
        return env_key

    raise RuntimeError(
        "FRED_API_KEY not found.\n"
        "Provide via:\n"
        "- function argument api_key\n"
        "- environment variable FRED_API_KEY"
    )


# -------------------------------------------------------------------
# HTTP Fetch
# -------------------------------------------------------------------


def _fetch_observations(params: Dict[str, str]) -> Dict:
    """
    Execute request with retry logic.
    """

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                FRED_BASE_URL,
                params=params,
                timeout=DEFAULT_TIMEOUT,
            )

            if response.status_code == 200:
                return response.json()

            last_error = RuntimeError(
                f"FRED API returned status {response.status_code}"
            )

        except Exception as e:
            last_error = e

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_SLEEP)

    raise RuntimeError(f"FRED request failed after {MAX_RETRIES} attempts: {last_error}")


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

    L1 Contract
    ----------
    - No resampling
    - No renaming
    - No enrichment
    - UTC timestamps
    - Raw frequency preserved

    Returns
    -------
    DataFrame with columns:

        date
        value
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

    payload = _fetch_observations(params)

    observations = payload.get("observations", [])

    df = pd.DataFrame(observations)

    if df.empty:
        return pd.DataFrame(columns=["date", "value"])

    # Preserve raw schema
    df = df[["date", "value"]].copy()

    # UTC timestamp
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # Numeric conversion
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # L1 hygiene
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

    Designed for Kedro PartitionedDataset ingestion.

    Parameters
    ----------
    series_ids
        List of FRED series IDs

    start_date
        observation_start

    end_date
        observation_end (None → until today)

    Returns
    -------
    Dict[str, DataFrame]

    Example
    -------
    {
        "CPIAUCSL": DataFrame,
        "UNRATE": DataFrame,
        ...
    }
    """

    api_key = _resolve_fred_api_key(api_key)

    output: Dict[str, pd.DataFrame] = {}

    for series_id in series_ids:
        output[series_id] = fetch_fred_series(
            series_id=series_id,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
        )

    return output