"""
Yahoo Finance client for fetching historical market data.

Design principles:
- Pure client (no Kedro, no IO side effects)
- Explicit control of execution (no implicit parallelism)
- Sequential downloads by default
- UTC timestamps
- Minimal schema opinionation
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def fetch_yfinance_series(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical data for a single Yahoo Finance ticker.

    Returns a DataFrame with:
        - date (datetime64[ns, UTC])
        - open
        - high
        - low
        - close
        - volume

    No IO, no resampling, no enrichment.
    """
    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,   # 🔴 critical: disable internal parallelism
    )

    if df.empty:
        return df

    # Flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Index → column
    df = df.reset_index()

    # Normalize column names (light opinion, consistent)
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]

    # Ensure UTC
    df["date"] = pd.to_datetime(df["date"], utc=True)

    return df


def fetch_yfinance_batch(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    auto_adjust: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch multiple Yahoo Finance tickers sequentially.

    Returns:
        Dict[ticker, DataFrame]

    Designed for use with Kedro PartitionedDataset.
    """
    output: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        df = fetch_yfinance_series(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            auto_adjust=auto_adjust,
        )
        output[ticker] = df

    return output
