"""
Kedro node for Yahoo Finance L1 ingestion.

This node is a thin wrapper around the yfinance client.
No business logic is implemented here.
"""

from typing import Dict, List

import pandas as pd

from crypto_mkt_state.clients.yfinance_client import fetch_yfinance_batch


def load_yfinance_l1(
    assets: List[dict],
    start_date: str,
) -> Dict[str, pd.DataFrame]:
    """
    Load Yahoo Finance data for multiple assets (L1 raw).

    Args:
        assets:
            List of dicts from parameters.yml (expects key 'ticker').
        start_date:
            Observation start date (YYYY-MM-DD).

    Returns:
        Dict[ticker, DataFrame] compatible with PartitionedDataset.
    """
    tickers = [asset["ticker"] for asset in assets]

    return fetch_yfinance_batch(
        tickers=tickers,
        start_date=start_date,
    )
